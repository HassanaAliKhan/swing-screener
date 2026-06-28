#!/usr/bin/env python3
"""
Friday Covered-Call Screener — Mark-Priced Version

Finds one lowest-strike call per ticker for the nearest listed Friday expiry
when all of the following are true:
  * strike is at least X% below the underlying spot price;
  * the selected call-credit basis produces Y% to Z% assignment profit;
  * option liquidity/spread filters pass.

Default credit basis: MARK = (BID + ASK) / 2

Core formulas:
  Mark                         = (Bid + Ask) / 2
  AssignmentBreakEven           = Strike + PremiumUsed
  AssignmentProfit_pct          = (AssignmentBreakEven - Spot) / Spot * 100
  MaxFallBeforeLoss_pct         = PremiumUsed / Spot * 100
  CoveredCallDownsideBreakeven  = Spot - PremiumUsed

The default strategy range is 1% to 5% assignment profit, with the call
strike at least 20% below the underlying spot price.

This is a research/review tool. It does not place orders. Yahoo/yfinance
option quotes may be delayed, stale, incomplete, or unavailable. Mark is a
midpoint estimate, NOT a guaranteed fill. Confirm the live bid, ask, spread,
liquidity, and actual limit-order fill with your broker before trading.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=UserWarning, module=r"^yfinance(\..*)?$")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing packages. Install with:\n"
        "python -m pip install -U yfinance pandas numpy"
    ) from exc


DEFAULT_WATCHLIST = Path(__file__).with_name("watchlist.txt")


@dataclass(frozen=True)
class ScanConfig:
    min_strike_discount_pct: float = 20.0
    min_assignment_profit_pct: float = 1.0
    max_assignment_profit_pct: float = 5.0
    premium_basis: str = "mark"  # mark, bid, last
    min_open_interest: int = 10
    min_option_volume: int = 0
    max_bid_ask_spread_pct: float = 100.0
    include_extended_spot: bool = False
    max_workers: int = 3
    retries: int = 2
    retry_delay_seconds: float = 0.8
    today: date | None = None


@dataclass
class ScanOutput:
    candidates: list[dict]
    diagnostics: list[dict]
    errors: list[dict]


def safe_float(value: object) -> float:
    """Convert a value to a finite float; otherwise return NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def safe_int(value: object) -> int:
    number = safe_float(value)
    return int(number) if math.isfinite(number) else 0


def round_or_nan(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if math.isfinite(value) else math.nan


def parse_tickers(text: str) -> list[str]:
    """Accept one ticker per line, comma-separated lists, and # comments."""
    tickers: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        clean_line = raw_line.split("#", 1)[0].replace(",", " ").strip()
        for token in clean_line.split():
            ticker = token.upper().strip()
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def load_tickers(path: Path | None = None) -> list[str]:
    selected = path or DEFAULT_WATCHLIST
    if not selected.exists():
        raise FileNotFoundError(
            f"Ticker file not found: {selected}. Put watchlist.txt next to this script."
        )
    tickers = parse_tickers(selected.read_text(encoding="utf-8"))
    if not tickers:
        raise ValueError(f"No ticker symbols found in {selected}")
    return tickers


def next_friday_on_or_after(day: date) -> date:
    """Return the Friday falling on or after `day`."""
    return day.fromordinal(day.toordinal() + (4 - day.weekday()) % 7)


def choose_nearest_friday_expiry(
    expiration_strings: Iterable[str],
    today: date | None = None,
) -> tuple[str, str]:
    """
    Choose the coming listed Friday expiry.

    When a holiday moves weekly expiration to Thursday, use a listed expiry within
    one day of Friday. If no Friday is listed, fall back to the nearest available
    future expiry and expose the reason in output.
    """
    today = today or date.today()
    parsed: list[tuple[date, str]] = []
    for raw in expiration_strings:
        try:
            parsed.append((datetime.strptime(str(raw), "%Y-%m-%d").date(), str(raw)))
        except ValueError:
            continue

    future = sorted((item for item in parsed if item[0] >= today), key=lambda item: item[0])
    if not future:
        raise ValueError("Yahoo returned no future option expirations")

    target = next_friday_on_or_after(today)
    exact = [item for item in future if item[0] == target]
    if exact:
        return exact[0][1], "Exact Friday expiration"

    holiday_shift = [item for item in future if abs((item[0] - target).days) <= 1]
    if holiday_shift:
        chosen = min(holiday_shift, key=lambda item: (abs((item[0] - target).days), item[0]))
        return chosen[1], "Nearest listed weekly expiry (holiday-adjusted)"

    friday_after = [item for item in future if item[0].weekday() == 4 and item[0] > target]
    if friday_after:
        return friday_after[0][1], "Next listed Friday expiration"

    return future[0][1], "No Friday expiration listed; nearest available expiration used"


def _last_non_null_close(frame: pd.DataFrame) -> tuple[float, str | None]:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return math.nan, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return math.nan, None
    return safe_float(close.iloc[-1]), str(close.index[-1])


def fetch_spot(ticker_obj: yf.Ticker, include_extended: bool) -> tuple[float, str, str | None]:
    """Get underlying reference price using Yahoo fallbacks."""
    try:
        fast = ticker_obj.fast_info
        for key in ("lastPrice", "last_price"):
            value = safe_float(fast.get(key))
            if value > 0:
                return value, "fast_info.lastPrice", None
    except Exception:
        pass

    for interval in ("1m", "5m"):
        try:
            history = ticker_obj.history(
                period="5d",
                interval=interval,
                auto_adjust=False,
                prepost=include_extended,
                raise_errors=False,
            )
            value, timestamp = _last_non_null_close(history)
            if value > 0:
                return value, f"history {interval} close", timestamp
        except Exception:
            pass

    raise ValueError("Unable to obtain a valid underlying price from Yahoo")


def quote_mark(bid: float, ask: float) -> float:
    """Return midpoint mark = (bid + ask) / 2 only when both quotes are valid."""
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return math.nan


def premium_from_quote(row: pd.Series, basis: str) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))
    mark = quote_mark(bid, ask)

    if basis == "mark":
        return mark
    if basis == "bid":
        return bid if bid > 0 else math.nan
    if basis == "last":
        return last if last > 0 else math.nan
    raise ValueError(f"Unsupported premium basis: {basis}")


def bid_ask_spread_pct(bid: float, ask: float) -> float:
    mark = quote_mark(bid, ask)
    if not math.isfinite(mark) or mark <= 0:
        return math.nan
    return (ask - bid) / mark * 100.0


def _regular_contract_mask(frame: pd.DataFrame) -> pd.Series:
    if "contractSize" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["contractSize"].fillna("REGULAR").astype(str).eq("REGULAR")


def select_lowest_strike_call(
    ticker: str,
    spot: float,
    calls: pd.DataFrame,
    expiry: str,
    expiry_note: str,
    config: ScanConfig,
    spot_source: str,
    spot_timestamp: str | None,
) -> tuple[dict | None, str]:
    """Calculate all quote metrics and return the lowest-strike qualifying call."""
    if calls is None or calls.empty:
        return None, "Yahoo returned an empty call chain"

    frame = calls.copy()
    required = [
        "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility",
    ]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["Mark"] = frame.apply(
        lambda row: quote_mark(safe_float(row["bid"]), safe_float(row["ask"])),
        axis=1,
    )
    frame["PremiumUsed"] = frame.apply(
        lambda row: premium_from_quote(row, config.premium_basis), axis=1
    )
    frame["PremiumBasis"] = config.premium_basis

    frame["StrikeDiscount_pct"] = (spot - frame["strike"]) / spot * 100.0
    frame["RequiredCredit_1pct"] = spot * (1.0 + config.min_assignment_profit_pct / 100.0) - frame["strike"]
    frame["RequiredCredit_5pct"] = spot * (1.0 + config.max_assignment_profit_pct / 100.0) - frame["strike"]

    # The field historically requested as "break even" by the strategy: capped
    # call-away value if stock is assigned. Actual downside breakeven is separate.
    frame["AssignmentBreakEven"] = frame["strike"] + frame["PremiumUsed"]
    frame["AssignmentProfit_pct"] = (frame["AssignmentBreakEven"] - spot) / spot * 100.0
    frame["MarkAssignmentBreakEven"] = frame["strike"] + frame["Mark"]
    frame["MarkAssignmentProfit_pct"] = (frame["MarkAssignmentBreakEven"] - spot) / spot * 100.0
    frame["BidAssignmentBreakEven"] = frame["strike"] + frame["bid"]
    frame["BidAssignmentProfit_pct"] = (frame["BidAssignmentBreakEven"] - spot) / spot * 100.0
    frame["AskAssignmentBreakEven"] = frame["strike"] + frame["ask"]
    frame["AskAssignmentProfit_pct"] = (frame["AskAssignmentBreakEven"] - spot) / spot * 100.0

    frame["MaxFallBeforeCoveredCallLoss_pct"] = frame["PremiumUsed"] / spot * 100.0
    frame["CoveredCallDownsideBreakeven"] = spot - frame["PremiumUsed"]
    frame["BidAskSpread_pct"] = frame.apply(
        lambda row: bid_ask_spread_pct(safe_float(row["bid"]), safe_float(row["ask"])),
        axis=1,
    )

    spread_ok = frame["BidAskSpread_pct"].isna() | (
        frame["BidAskSpread_pct"] <= config.max_bid_ask_spread_pct
    )
    regular_contract_ok = _regular_contract_mask(frame)

    qualifying = frame.loc[
        (frame["strike"] > 0)
        & (frame["PremiumUsed"] > 0)
        & (frame["StrikeDiscount_pct"] >= config.min_strike_discount_pct)
        & (frame["AssignmentProfit_pct"] >= config.min_assignment_profit_pct)
        & (frame["AssignmentProfit_pct"] <= config.max_assignment_profit_pct)
        & (frame["openInterest"].fillna(0) >= config.min_open_interest)
        & (frame["volume"].fillna(0) >= config.min_option_volume)
        & spread_ok
        & regular_contract_ok
    ].copy()

    if qualifying.empty:
        return None, "No call met active strike, mark/credit, liquidity, and spread filters"

    # User requested the least / lowest eligible strike. Break ties by profit,
    # premium buffer, then tighter bid-ask spread.
    qualifying["_spread_sort"] = qualifying["BidAskSpread_pct"].fillna(float("inf"))
    qualifying = qualifying.sort_values(
        by=["strike", "AssignmentProfit_pct", "PremiumUsed", "_spread_sort"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    chosen = qualifying.iloc[0]

    last_trade = chosen.get("lastTradeDate", "")
    last_trade_text = str(last_trade) if pd.notna(last_trade) else ""

    row = {
        "Ticker": ticker,
        "Spot": round_or_nan(spot),
        "SpotSource": spot_source,
        "SpotTimestamp": spot_timestamp or "",
        "Expiry": expiry,
        "ExpirySelection": expiry_note,
        "ContractSymbol": str(chosen.get("contractSymbol", "")),
        "Strike": round_or_nan(safe_float(chosen["strike"])),
        "StrikeDiscount_pct": round_or_nan(safe_float(chosen["StrikeDiscount_pct"])),
        "Bid": round_or_nan(safe_float(chosen["bid"])),
        "Ask": round_or_nan(safe_float(chosen["ask"])),
        "Mark": round_or_nan(safe_float(chosen["Mark"])),
        "Last": round_or_nan(safe_float(chosen["lastPrice"])),
        "PremiumBasis": config.premium_basis,
        "PremiumUsed": round_or_nan(safe_float(chosen["PremiumUsed"])),
        "RequiredCredit_MinProfit": round_or_nan(safe_float(chosen["RequiredCredit_1pct"])),
        "RequiredCredit_MaxProfit": round_or_nan(safe_float(chosen["RequiredCredit_5pct"])),
        "AssignmentBreakEven": round_or_nan(safe_float(chosen["AssignmentBreakEven"])),
        "AssignmentProfit_pct": round_or_nan(safe_float(chosen["AssignmentProfit_pct"])),
        "MarkAssignmentBreakEven": round_or_nan(safe_float(chosen["MarkAssignmentBreakEven"])),
        "MarkAssignmentProfit_pct": round_or_nan(safe_float(chosen["MarkAssignmentProfit_pct"])),
        "BidAssignmentProfit_pct": round_or_nan(safe_float(chosen["BidAssignmentProfit_pct"])),
        "AskAssignmentProfit_pct": round_or_nan(safe_float(chosen["AskAssignmentProfit_pct"])),
        "MaxFallBeforeCoveredCallLoss_pct": round_or_nan(
            safe_float(chosen["MaxFallBeforeCoveredCallLoss_pct"])
        ),
        "CoveredCallDownsideBreakeven": round_or_nan(
            safe_float(chosen["CoveredCallDownsideBreakeven"])
        ),
        "BidAskSpread_pct": round_or_nan(safe_float(chosen["BidAskSpread_pct"])),
        "OptionVolume": safe_int(chosen["volume"]),
        "OpenInterest": safe_int(chosen["openInterest"]),
        "ImpliedVolatility_pct": round_or_nan(safe_float(chosen["impliedVolatility"]) * 100.0),
        "InTheMoney": bool(chosen.get("inTheMoney", False)),
        "LastTradeDate": last_trade_text,
        "Strategy": "COVERED_CALL_REVIEW",
    }
    return row, "Selected lowest qualifying strike"


def scan_one_ticker(ticker: str, config: ScanConfig) -> tuple[dict | None, dict, dict | None]:
    for attempt in range(1, max(1, config.retries) + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            spot, spot_source, spot_timestamp = fetch_spot(
                ticker_obj, config.include_extended_spot
            )
            expiry, expiry_note = choose_nearest_friday_expiry(
                ticker_obj.options,
                today=config.today,
            )
            chain = ticker_obj.option_chain(expiry)
            candidate, reason = select_lowest_strike_call(
                ticker=ticker,
                spot=spot,
                calls=chain.calls,
                expiry=expiry,
                expiry_note=expiry_note,
                config=config,
                spot_source=spot_source,
                spot_timestamp=spot_timestamp,
            )
            diagnostic = {
                "Ticker": ticker,
                "Spot": round_or_nan(spot),
                "Expiry": expiry,
                "ExpirySelection": expiry_note,
                "PremiumBasis": config.premium_basis,
                "Result": "QUALIFIED" if candidate is not None else "NO_MATCH",
                "Reason": reason,
            }
            return candidate, diagnostic, None
        except Exception as exc:
            if attempt >= max(1, config.retries):
                error = {"Ticker": ticker, "Error": f"{type(exc).__name__}: {exc}"}
                diagnostic = {
                    "Ticker": ticker,
                    "Spot": math.nan,
                    "Expiry": "",
                    "ExpirySelection": "",
                    "PremiumBasis": config.premium_basis,
                    "Result": "ERROR",
                    "Reason": error["Error"],
                }
                return None, diagnostic, error
            time.sleep(max(0.0, config.retry_delay_seconds))

    raise RuntimeError("Unreachable retry state")


def scan_tickers(
    tickers: list[str],
    config: ScanConfig,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> ScanOutput:
    candidates: list[dict] = []
    diagnostics: list[dict] = []
    errors: list[dict] = []

    if not tickers:
        return ScanOutput(candidates, diagnostics, errors)

    workers = max(1, min(int(config.max_workers), 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_one_ticker, ticker, config): ticker for ticker in tickers}
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                candidate, diagnostic, error = future.result()
            except Exception as exc:
                candidate = None
                diagnostic = {
                    "Ticker": ticker,
                    "Spot": math.nan,
                    "Expiry": "",
                    "ExpirySelection": "",
                    "PremiumBasis": config.premium_basis,
                    "Result": "ERROR",
                    "Reason": f"Worker error: {type(exc).__name__}: {exc}",
                }
                error = {"Ticker": ticker, "Error": diagnostic["Reason"]}

            diagnostics.append(diagnostic)
            if candidate is not None:
                candidates.append(candidate)
            if error is not None:
                errors.append(error)
            if progress_callback is not None:
                progress_callback(completed, len(tickers), ticker, diagnostic["Result"])

    # Requested order: highest premium downside buffer / maximum fall protection first.
    candidates.sort(
        key=lambda row: (
            safe_float(row["MaxFallBeforeCoveredCallLoss_pct"]),
            safe_float(row["StrikeDiscount_pct"]),
            safe_float(row["AssignmentProfit_pct"]),
        ),
        reverse=True,
    )
    diagnostics.sort(key=lambda row: row["Ticker"])
    errors.sort(key=lambda row: row["Ticker"])
    return ScanOutput(candidates, diagnostics, errors)


def write_csv(rows: list[dict], path: Path, columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen coming-Friday calls for deep-ITM covered-call review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers-file", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--min-strike-discount-pct", type=float, default=20.0)
    parser.add_argument("--min-assignment-profit-pct", type=float, default=1.0)
    parser.add_argument("--max-assignment-profit-pct", type=float, default=5.0)
    parser.add_argument("--premium-basis", choices=["mark", "bid", "last"], default="mark")
    parser.add_argument("--min-open-interest", type=int, default=10)
    parser.add_argument("--min-option-volume", type=int, default=0)
    parser.add_argument("--max-bid-ask-spread-pct", type=float, default=100.0)
    parser.add_argument("--include-extended-spot", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = load_tickers(args.tickers_file)
    if args.min_assignment_profit_pct > args.max_assignment_profit_pct:
        raise SystemExit("Minimum assignment profit cannot exceed maximum assignment profit.")

    config = ScanConfig(
        min_strike_discount_pct=args.min_strike_discount_pct,
        min_assignment_profit_pct=args.min_assignment_profit_pct,
        max_assignment_profit_pct=args.max_assignment_profit_pct,
        premium_basis=args.premium_basis,
        min_open_interest=args.min_open_interest,
        min_option_volume=args.min_option_volume,
        max_bid_ask_spread_pct=args.max_bid_ask_spread_pct,
        include_extended_spot=args.include_extended_spot,
        max_workers=args.max_workers,
        retries=args.retries,
    )

    print(
        f"Scanning {len(tickers)} tickers | strike discount >= {config.min_strike_discount_pct:.2f}% | "
        f"assignment profit {config.min_assignment_profit_pct:.2f}% to "
        f"{config.max_assignment_profit_pct:.2f}% | premium={config.premium_basis}"
    )

    def progress(done: int, total: int, ticker: str, status: str) -> None:
        print(f"[{done:>3}/{total}] {ticker:<7} {status}")

    output = scan_tickers(tickers, config, progress_callback=progress)
    args.outdir.mkdir(parents=True, exist_ok=True)

    result_columns = [
        "Ticker", "Spot", "Expiry", "ExpirySelection", "ContractSymbol", "Strike",
        "StrikeDiscount_pct", "Bid", "Ask", "Mark", "Last", "PremiumBasis", "PremiumUsed",
        "RequiredCredit_MinProfit", "RequiredCredit_MaxProfit", "AssignmentBreakEven",
        "AssignmentProfit_pct", "MarkAssignmentBreakEven", "MarkAssignmentProfit_pct",
        "BidAssignmentProfit_pct", "AskAssignmentProfit_pct", "MaxFallBeforeCoveredCallLoss_pct",
        "CoveredCallDownsideBreakeven", "BidAskSpread_pct", "OptionVolume", "OpenInterest",
        "ImpliedVolatility_pct", "InTheMoney", "LastTradeDate", "SpotSource", "SpotTimestamp",
        "Strategy",
    ]
    candidate_path = args.outdir / "friday_covered_call_candidates.csv"
    write_csv(output.candidates, candidate_path, result_columns)
    if args.debug:
        write_csv(output.diagnostics, args.outdir / "friday_covered_call_diagnostics.csv")
    if output.errors:
        write_csv(output.errors, args.outdir / "friday_covered_call_errors.csv")

    print("\n=== LOWEST-STRIKE QUALIFYING COVERED-CALL CANDIDATES ===")
    if output.candidates:
        display_columns = [
            "Ticker", "Spot", "Expiry", "Strike", "Bid", "Ask", "Mark", "PremiumUsed",
            "AssignmentBreakEven", "AssignmentProfit_pct", "MaxFallBeforeCoveredCallLoss_pct",
            "StrikeDiscount_pct", "BidAskSpread_pct", "OpenInterest", "OptionVolume",
        ]
        print(pd.DataFrame(output.candidates)[display_columns].to_string(index=False))
    else:
        print("No calls passed the active filters.")
    print(f"\nSaved: {candidate_path}")
    if args.debug:
        print(f"Diagnostics: {args.outdir / 'friday_covered_call_diagnostics.csv'}")
    if output.errors:
        print(f"Errors: {args.outdir / 'friday_covered_call_errors.csv'}")


if __name__ == "__main__":
    main()
