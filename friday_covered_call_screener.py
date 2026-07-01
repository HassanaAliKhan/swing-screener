#!/usr/bin/env python3
"""
Option-Income Screener — covered calls and cash-secured puts.

Strategies:
  * COVERED_CALL: own 100 shares and sell one call.
  * CASH_SECURED_PUT: reserve cash for 100 shares and sell one put.

For cash-secured puts, the screener selects the farthest-out-of-the-money
(lowest-strike) put that meets the selected premium-yield, liquidity, spread,
and estimated-delta filters. It is a research tool only; assignment remains
possible and must be financially acceptable.
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
from typing import Callable, Iterable, Literal

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
Strategy = Literal["covered_call", "cash_secured_put"]


@dataclass(frozen=True)
class ScanConfig:
    strategy: Strategy = "cash_secured_put"
    min_strike_discount_pct: float = 20.0
    min_return_pct: float = 1.0
    max_return_pct: float = 8.0
    premium_basis: str = "bid"  # mark, bid, last
    max_abs_put_delta: float = 0.15
    min_open_interest: int = 25
    min_option_volume: int = 1
    max_bid_ask_spread_pct: float = 15.0
    include_extended_spot: bool = False
    max_workers: int = 3
    retries: int = 2
    retry_delay_seconds: float = 0.8
    max_expiry_days: int = 11
    today: date | None = None


@dataclass
class ScanOutput:
    candidates: list[dict]
    diagnostics: list[dict]
    errors: list[dict]


def safe_float(value: object) -> float:
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


class NoEligibleExpiry(ValueError):
    """Raised when Yahoo has no listed option expiry inside the configured window."""


def choose_furthest_expiry_within_window(
    expiration_strings: Iterable[str],
    today: date | None = None,
    max_expiry_days: int = 11,
) -> tuple[str, str]:
    """Choose the latest listed option expiry no more than ``max_expiry_days`` away.

    This deliberately does not force a Friday-only chain. For example, with an
    11-day cap, it chooses the furthest listed expiration on or before
    ``today + 11 calendar days``. This maximizes time value while keeping the
    contract within the requested short-duration window.
    """
    today = today or date.today()
    max_expiry_days = int(max_expiry_days)
    if max_expiry_days < 1:
        raise ValueError("max_expiry_days must be at least 1")

    parsed: list[tuple[date, str]] = []
    for raw in expiration_strings:
        try:
            expiry_date = datetime.strptime(str(raw), "%Y-%m-%d").date()
        except ValueError:
            continue
        if expiry_date >= today:
            parsed.append((expiry_date, str(raw)))

    future = sorted(parsed, key=lambda item: item[0])
    if not future:
        raise NoEligibleExpiry("Yahoo returned no future option expirations")

    cutoff = today.fromordinal(today.toordinal() + max_expiry_days)
    in_window = [item for item in future if item[0] <= cutoff]
    if not in_window:
        raise NoEligibleExpiry(
            f"No listed option expiration within {max_expiry_days} calendar days "
            f"(nearest Yahoo expiry: {future[0][0].isoformat()})"
        )

    chosen_date, chosen_expiry = max(in_window, key=lambda item: item[0])
    dte = (chosen_date - today).days
    return (
        chosen_expiry,
        f"Furthest listed expiry within {max_expiry_days} calendar days ({dte} DTE)",
    )


def _last_non_null_close(frame: pd.DataFrame) -> tuple[float, str | None]:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return math.nan, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return math.nan, None
    return safe_float(close.iloc[-1]), str(close.index[-1])


def fetch_spot(ticker_obj: yf.Ticker, include_extended: bool) -> tuple[float, str, str | None]:
    """Get an underlying reference price using Yahoo fallbacks."""
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


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def estimated_abs_put_delta(
    spot: float,
    strike: float,
    implied_volatility: float,
    days_to_expiry: int,
) -> float:
    """Approximate absolute put delta from Black-Scholes using Yahoo IV.

    This is a quote-derived risk proxy, not an assignment probability or guarantee.
    """
    if (
        not math.isfinite(spot)
        or not math.isfinite(strike)
        or not math.isfinite(implied_volatility)
        or spot <= 0
        or strike <= 0
        or implied_volatility <= 0
    ):
        return math.nan

    years = max(float(days_to_expiry), 1.0) / 365.0
    sigma_sqrt_t = implied_volatility * math.sqrt(years)
    if sigma_sqrt_t <= 0:
        return math.nan

    d1 = (math.log(spot / strike) + 0.5 * implied_volatility**2 * years) / sigma_sqrt_t
    put_delta = normal_cdf(d1) - 1.0
    return abs(put_delta)


def _regular_contract_mask(frame: pd.DataFrame) -> pd.Series:
    if "contractSize" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["contractSize"].fillna("REGULAR").astype(str).eq("REGULAR")


def _normalise_chain(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    required = [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]
    for column in required:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["Mark"] = result.apply(
        lambda row: quote_mark(safe_float(row["bid"]), safe_float(row["ask"])),
        axis=1,
    )
    result["BidAskSpread_pct"] = result.apply(
        lambda row: bid_ask_spread_pct(safe_float(row["bid"]), safe_float(row["ask"])),
        axis=1,
    )
    result["MarkBidGap"] = result["Mark"] - result["bid"]
    result["MarkBidGap_pct"] = np.where(
        result["Mark"] > 0,
        result["MarkBidGap"] / result["Mark"] * 100.0,
        np.nan,
    )
    return result


def select_covered_call(
    ticker: str,
    spot: float,
    calls: pd.DataFrame,
    expiry: str,
    expiry_note: str,
    config: ScanConfig,
    spot_source: str,
    spot_timestamp: str | None,
) -> tuple[dict | None, str]:
    """Select the lowest-strike qualifying covered call under the legacy logic."""
    frame = _normalise_chain(calls)
    if frame.empty:
        return None, "Yahoo returned an empty call chain"

    frame["PremiumUsed"] = frame.apply(
        lambda row: premium_from_quote(row, config.premium_basis), axis=1
    )
    frame["StrikeDiscount_pct"] = (spot - frame["strike"]) / spot * 100.0
    frame["AssignmentBreakEven"] = frame["strike"] + frame["PremiumUsed"]
    frame["AssignmentProfit_pct"] = (frame["AssignmentBreakEven"] - spot) / spot * 100.0
    frame["MaxFallBeforeCoveredCallLoss_pct"] = frame["PremiumUsed"] / spot * 100.0
    frame["CoveredCallDownsideBreakeven"] = spot - frame["PremiumUsed"]

    spread_ok = frame["BidAskSpread_pct"].isna() | (
        frame["BidAskSpread_pct"] <= config.max_bid_ask_spread_pct
    )
    qualifying = frame.loc[
        (frame["strike"] > 0)
        & (frame["PremiumUsed"] > 0)
        & (frame["StrikeDiscount_pct"] >= config.min_strike_discount_pct)
        & (frame["AssignmentProfit_pct"] >= config.min_return_pct)
        & (frame["AssignmentProfit_pct"] <= config.max_return_pct)
        & (frame["openInterest"].fillna(0) >= config.min_open_interest)
        & (frame["volume"].fillna(0) >= config.min_option_volume)
        & spread_ok
        & _regular_contract_mask(frame)
    ].copy()

    if qualifying.empty:
        return None, "No call met active strike, return, liquidity, and spread filters"

    qualifying["_spread_sort"] = qualifying["BidAskSpread_pct"].fillna(float("inf"))
    qualifying = qualifying.sort_values(
        by=["strike", "AssignmentProfit_pct", "PremiumUsed", "_spread_sort"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    chosen = qualifying.iloc[0]
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - (config.today or date.today())).days

    return {
        "Strategy": "COVERED_CALL_REVIEW",
        "Ticker": ticker,
        "Spot": round_or_nan(spot),
        "SpotSource": spot_source,
        "SpotTimestamp": spot_timestamp or "",
        "Expiry": expiry,
        "ExpirySelection": expiry_note,
        "DaysToExpiry": dte,
        "ContractSymbol": str(chosen.get("contractSymbol", "")),
        "Strike": round_or_nan(safe_float(chosen["strike"])),
        "StrikeDiscount_pct": round_or_nan(safe_float(chosen["StrikeDiscount_pct"])),
        "Bid": round_or_nan(safe_float(chosen["bid"])),
        "Ask": round_or_nan(safe_float(chosen["ask"])),
        "Mark": round_or_nan(safe_float(chosen["Mark"])),
        "MarkBidGap": round_or_nan(safe_float(chosen["MarkBidGap"])),
        "MarkBidGap_pct": round_or_nan(safe_float(chosen["MarkBidGap_pct"])),
        "Last": round_or_nan(safe_float(chosen["lastPrice"])),
        "PremiumBasis": config.premium_basis,
        "PremiumUsed": round_or_nan(safe_float(chosen["PremiumUsed"])),
        "AssignmentBreakEven": round_or_nan(safe_float(chosen["AssignmentBreakEven"])),
        "AssignmentProfit_pct": round_or_nan(safe_float(chosen["AssignmentProfit_pct"])),
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
        "LastTradeDate": str(chosen.get("lastTradeDate", "")),
    }, "Selected lowest qualifying covered-call strike"


def select_cash_secured_put(
    ticker: str,
    spot: float,
    puts: pd.DataFrame,
    expiry: str,
    expiry_note: str,
    config: ScanConfig,
    spot_source: str,
    spot_timestamp: str | None,
) -> tuple[dict | None, str]:
    """Select the lowest-strike cash-secured put that meets the user's downside buffer.

    The return filter is premium divided by strike collateral, not premium divided
    by spot. The key downside buffer is to the effective purchase breakeven:
    strike minus premium.
    """
    frame = _normalise_chain(puts)
    if frame.empty:
        return None, "Yahoo returned an empty put chain"

    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - (config.today or date.today())).days
    frame["PremiumUsed"] = frame.apply(
        lambda row: premium_from_quote(row, config.premium_basis), axis=1
    )
    frame["StrikeDiscount_pct"] = (spot - frame["strike"]) / spot * 100.0
    frame["CashCollateral_perContract"] = frame["strike"] * 100.0
    frame["PremiumCredit_perContract"] = frame["PremiumUsed"] * 100.0
    frame["PremiumYieldOnCollateral_pct"] = (
        frame["PremiumUsed"] / frame["strike"] * 100.0
    )
    frame["PremiumYieldOnSpot_pct"] = frame["PremiumUsed"] / spot * 100.0
    frame["PutBreakeven"] = frame["strike"] - frame["PremiumUsed"]
    frame["MaxFallBeforePutLoss_pct"] = (spot - frame["PutBreakeven"]) / spot * 100.0
    frame["EstimatedAbsPutDelta"] = frame.apply(
        lambda row: estimated_abs_put_delta(
            spot=spot,
            strike=safe_float(row["strike"]),
            implied_volatility=safe_float(row["impliedVolatility"]),
            days_to_expiry=dte,
        ),
        axis=1,
    )

    spread_ok = frame["BidAskSpread_pct"].isna() | (
        frame["BidAskSpread_pct"] <= config.max_bid_ask_spread_pct
    )
    delta_ok = frame["EstimatedAbsPutDelta"].notna() & (
        frame["EstimatedAbsPutDelta"] <= config.max_abs_put_delta
    )

    qualifying = frame.loc[
        (frame["strike"] > 0)
        & (frame["PremiumUsed"] > 0)
        & (frame["StrikeDiscount_pct"] >= config.min_strike_discount_pct)
        & (frame["PremiumYieldOnCollateral_pct"] >= config.min_return_pct)
        & (frame["PremiumYieldOnCollateral_pct"] <= config.max_return_pct)
        & delta_ok
        & (frame["openInterest"].fillna(0) >= config.min_open_interest)
        & (frame["volume"].fillna(0) >= config.min_option_volume)
        & spread_ok
        & _regular_contract_mask(frame)
    ].copy()

    if qualifying.empty:
        return None, (
            "No put met active downside-buffer, premium-yield, estimated-delta, "
            "liquidity, and spread filters"
        )

    qualifying["_gap_sort"] = qualifying["MarkBidGap_pct"].fillna(float("inf"))
    qualifying["_spread_sort"] = qualifying["BidAskSpread_pct"].fillna(float("inf"))
    qualifying = qualifying.sort_values(
        by=[
            "MaxFallBeforePutLoss_pct",
            "EstimatedAbsPutDelta",
            "_gap_sort",
            "_spread_sort",
            "openInterest",
            "strike",
        ],
        ascending=[False, True, True, True, False, True],
        kind="stable",
    )
    chosen = qualifying.iloc[0]

    return {
        "Strategy": "CASH_SECURED_PUT_REVIEW",
        "Ticker": ticker,
        "Spot": round_or_nan(spot),
        "SpotSource": spot_source,
        "SpotTimestamp": spot_timestamp or "",
        "Expiry": expiry,
        "ExpirySelection": expiry_note,
        "DaysToExpiry": dte,
        "ContractSymbol": str(chosen.get("contractSymbol", "")),
        "Strike": round_or_nan(safe_float(chosen["strike"])),
        "StrikeDiscount_pct": round_or_nan(safe_float(chosen["StrikeDiscount_pct"])),
        "Bid": round_or_nan(safe_float(chosen["bid"])),
        "Ask": round_or_nan(safe_float(chosen["ask"])),
        "Mark": round_or_nan(safe_float(chosen["Mark"])),
        "MarkBidGap": round_or_nan(safe_float(chosen["MarkBidGap"])),
        "MarkBidGap_pct": round_or_nan(safe_float(chosen["MarkBidGap_pct"])),
        "Last": round_or_nan(safe_float(chosen["lastPrice"])),
        "PremiumBasis": config.premium_basis,
        "PremiumUsed": round_or_nan(safe_float(chosen["PremiumUsed"])),
        "CashCollateral_perContract": round_or_nan(
            safe_float(chosen["CashCollateral_perContract"])
        ),
        "PremiumCredit_perContract": round_or_nan(
            safe_float(chosen["PremiumCredit_perContract"])
        ),
        "PremiumYieldOnCollateral_pct": round_or_nan(
            safe_float(chosen["PremiumYieldOnCollateral_pct"])
        ),
        "PremiumYieldOnSpot_pct": round_or_nan(
            safe_float(chosen["PremiumYieldOnSpot_pct"])
        ),
        "PutBreakeven": round_or_nan(safe_float(chosen["PutBreakeven"])),
        "MaxFallBeforePutLoss_pct": round_or_nan(
            safe_float(chosen["MaxFallBeforePutLoss_pct"])
        ),
        "EstimatedAbsPutDelta": round_or_nan(
            safe_float(chosen["EstimatedAbsPutDelta"]), 3
        ),
        "BidAskSpread_pct": round_or_nan(safe_float(chosen["BidAskSpread_pct"])),
        "OptionVolume": safe_int(chosen["volume"]),
        "OpenInterest": safe_int(chosen["openInterest"]),
        "ImpliedVolatility_pct": round_or_nan(safe_float(chosen["impliedVolatility"]) * 100.0),
        "InTheMoney": bool(chosen.get("inTheMoney", False)),
        "LastTradeDate": str(chosen.get("lastTradeDate", "")),
    }, "Selected maximum-buffer qualifying cash-secured put"


def scan_one_ticker(
    ticker: str,
    config: ScanConfig,
) -> tuple[dict | None, dict, dict | None]:
    for attempt in range(1, max(1, config.retries) + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            spot, spot_source, spot_timestamp = fetch_spot(
                ticker_obj,
                config.include_extended_spot,
            )
            expiry, expiry_note = choose_furthest_expiry_within_window(
                ticker_obj.options,
                today=config.today,
                max_expiry_days=config.max_expiry_days,
            )
            chain = ticker_obj.option_chain(expiry)

            if config.strategy == "cash_secured_put":
                candidate, reason = select_cash_secured_put(
                    ticker=ticker,
                    spot=spot,
                    puts=chain.puts,
                    expiry=expiry,
                    expiry_note=expiry_note,
                    config=config,
                    spot_source=spot_source,
                    spot_timestamp=spot_timestamp,
                )
            else:
                candidate, reason = select_covered_call(
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
                "Strategy": config.strategy,
                "Spot": round_or_nan(spot),
                "Expiry": expiry,
                "ExpirySelection": expiry_note,
                "PremiumBasis": config.premium_basis,
                "MaxExpiryDays": config.max_expiry_days,
                "Result": "QUALIFIED" if candidate is not None else "NO_MATCH",
                "Reason": reason,
            }
            return candidate, diagnostic, None

        except NoEligibleExpiry as exc:
            diagnostic = {
                "Ticker": ticker,
                "Strategy": config.strategy,
                "Spot": math.nan,
                "Expiry": "",
                "ExpirySelection": "",
                "PremiumBasis": config.premium_basis,
                "MaxExpiryDays": config.max_expiry_days,
                "Result": "NO_EXPIRY_WINDOW",
                "Reason": str(exc),
            }
            return None, diagnostic, None

        except Exception as exc:
            if attempt >= max(1, config.retries):
                error = {"Ticker": ticker, "Error": f"{type(exc).__name__}: {exc}"}
                diagnostic = {
                    "Ticker": ticker,
                    "Strategy": config.strategy,
                    "Spot": math.nan,
                    "Expiry": "",
                    "ExpirySelection": "",
                    "PremiumBasis": config.premium_basis,
                    "MaxExpiryDays": config.max_expiry_days,
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
        futures = {
            executor.submit(scan_one_ticker, ticker, config): ticker
            for ticker in tickers
        }

        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                candidate, diagnostic, error = future.result()
            except Exception as exc:
                candidate = None
                diagnostic = {
                    "Ticker": ticker,
                    "Strategy": config.strategy,
                    "Spot": math.nan,
                    "Expiry": "",
                    "ExpirySelection": "",
                    "PremiumBasis": config.premium_basis,
                    "MaxExpiryDays": config.max_expiry_days,
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

    if config.strategy == "cash_secured_put":
        candidates.sort(
            key=lambda row: (
                -safe_float(row.get("MaxFallBeforePutLoss_pct")),
                safe_float(row.get("EstimatedAbsPutDelta")),
                safe_float(row.get("MarkBidGap_pct")),
            )
        )
    else:
        candidates.sort(
            key=lambda row: (
                safe_float(row.get("MarkBidGap_pct")),
                safe_float(row.get("MarkBidGap")),
                -safe_float(row.get("MaxFallBeforeCoveredCallLoss_pct")),
            )
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
        description="Screen covered calls or cash-secured puts using the furthest listed expiry inside the DTE cap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers-file", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument(
        "--strategy",
        choices=["covered_call", "cash_secured_put"],
        default="cash_secured_put",
    )
    parser.add_argument("--min-strike-discount-pct", type=float, default=20.0)
    parser.add_argument("--min-return-pct", type=float, default=1.0)
    parser.add_argument("--max-return-pct", type=float, default=8.0)
    parser.add_argument("--premium-basis", choices=["mark", "bid", "last"], default="bid")
    parser.add_argument("--max-abs-put-delta", type=float, default=0.15)
    parser.add_argument("--min-open-interest", type=int, default=25)
    parser.add_argument("--min-option-volume", type=int, default=1)
    parser.add_argument("--max-bid-ask-spread-pct", type=float, default=15.0)
    parser.add_argument("--max-expiry-days", type=int, default=11)
    parser.add_argument("--include-extended-spot", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_return_pct > args.max_return_pct:
        raise SystemExit("Minimum return cannot exceed maximum return.")
    if not 0 < args.max_abs_put_delta <= 1:
        raise SystemExit("max_abs_put_delta must be greater than 0 and at most 1.")

    tickers = load_tickers(args.tickers_file)
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = ScanConfig(
        strategy=args.strategy,
        min_strike_discount_pct=args.min_strike_discount_pct,
        min_return_pct=args.min_return_pct,
        max_return_pct=args.max_return_pct,
        premium_basis=args.premium_basis,
        max_abs_put_delta=args.max_abs_put_delta,
        min_open_interest=args.min_open_interest,
        min_option_volume=args.min_option_volume,
        max_bid_ask_spread_pct=args.max_bid_ask_spread_pct,
        max_expiry_days=args.max_expiry_days,
        include_extended_spot=args.include_extended_spot,
        max_workers=args.max_workers,
        retries=args.retries,
    )

    print(
        f"Scanning {len(tickers)} symbols | strategy={config.strategy} | "
        f"return={config.min_return_pct:.2f}%–{config.max_return_pct:.2f}%"
    )
    output = scan_tickers(tickers, config)

    candidates_path = args.outdir / f"{config.strategy}_candidates.csv"
    diagnostics_path = args.outdir / f"{config.strategy}_diagnostics.csv"
    errors_path = args.outdir / f"{config.strategy}_errors.csv"
    write_csv(output.candidates, candidates_path)
    if args.debug:
        write_csv(output.diagnostics, diagnostics_path)
    if output.errors:
        write_csv(output.errors, errors_path)

    print(f"Qualified candidates: {len(output.candidates)}")
    print(f"Saved: {candidates_path}")
    if args.debug:
        print(f"Diagnostics: {diagnostics_path}")
    if output.errors:
        print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()
