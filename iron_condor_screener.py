#!/usr/bin/env python3
from __future__ import annotations

import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Iterable

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^yfinance(\\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^yfinance(\\..*)?$")
warnings.filterwarnings("ignore", category=UserWarning, module=r"^yfinance(\\..*)?$")

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class IronCondorConfig:
    min_open_interest: int = 5
    min_option_volume: int = 1
    max_bid_ask_spread_pct: float = 500.0
    max_expiry_days: int = 11
    include_extended_spot: bool = True
    max_workers: int = 3
    retries: int = 2
    retry_delay_seconds: float = 0.8
    min_credit_per_share: float = 0.05
    min_profit_loss_ratio: float = 1.0
    min_breakeven_balance_ratio: float = 0.50
    max_wing_width_pct_of_spot: float = 15.0
    max_short_distance_pct_of_spot: float = 35.0
    top_n: int = 25
    today: date | None = None


@dataclass
class IronCondorOutput:
    candidates: list[dict]
    diagnostics: list[dict]
    errors: list[dict]


class NoEligibleExpiry(ValueError):
    pass


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


SCORE_WEIGHTS = {
    "breakeven_range": 0.45,
    "profit_loss": 0.35,
    "max_profit": 0.15,
    "low_max_loss": 0.05,
}


def _normalise_component(values: list[float], higher_is_better: bool = True) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]

    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        return [100.0 if math.isfinite(value) else 0.0 for value in values]

    scores: list[float] = []
    for value in values:
        if not math.isfinite(value):
            scores.append(0.0)
            continue
        scaled = (value - low) / (high - low)
        if not higher_is_better:
            scaled = 1.0 - scaled
        scores.append(scaled * 100.0)
    return scores


def score_candidates(rows: list[dict]) -> list[dict]:
    """Return copies of candidate rows with 0-100 component and overall scores."""
    if not rows:
        return []

    scored = [dict(row) for row in rows]
    breakeven_scores = _normalise_component(
        [safe_float(row.get("BreakevenRange_pct", row.get("ProfitZoneWidth_pct"))) for row in scored]
    )
    profit_loss_scores = _normalise_component(
        [safe_float(row.get("ProfitLossRatio")) for row in scored]
    )
    max_profit_scores = _normalise_component(
        [safe_float(row.get("MaxProfit")) for row in scored]
    )
    low_max_loss_scores = _normalise_component(
        [safe_float(row.get("MaxLoss")) for row in scored],
        higher_is_better=False,
    )

    for row, be_score, pl_score, profit_score, loss_score in zip(
        scored,
        breakeven_scores,
        profit_loss_scores,
        max_profit_scores,
        low_max_loss_scores,
    ):
        overall = (
            SCORE_WEIGHTS["breakeven_range"] * be_score
            + SCORE_WEIGHTS["profit_loss"] * pl_score
            + SCORE_WEIGHTS["max_profit"] * profit_score
            + SCORE_WEIGHTS["low_max_loss"] * loss_score
        )
        row["BreakevenRangeScore"] = round_or_nan(be_score, 1)
        row["ProfitLossScore"] = round_or_nan(pl_score, 1)
        row["MaxProfitScore"] = round_or_nan(profit_score, 1)
        row["LowMaxLossScore"] = round_or_nan(loss_score, 1)
        row["OverallScore"] = round_or_nan(overall, 1)

    return scored


def parse_tickers(text: str) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        clean = raw_line.split("#", 1)[0].replace(",", " ").strip()
        for token in clean.split():
            ticker = token.upper().strip()
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def choose_furthest_expiry_within_window(
    expiration_strings: Iterable[str],
    today: date | None = None,
    max_expiry_days: int = 11,
) -> tuple[str, str]:
    today = today or date.today()
    parsed: list[tuple[date, str]] = []

    for raw in expiration_strings:
        try:
            expiry_date = datetime.strptime(str(raw), "%Y-%m-%d").date()
        except ValueError:
            continue
        if expiry_date >= today:
            parsed.append((expiry_date, str(raw)))

    if not parsed:
        raise NoEligibleExpiry("Yahoo returned no future option expirations")

    cutoff = today.fromordinal(today.toordinal() + int(max_expiry_days))
    in_window = [item for item in parsed if item[0] <= cutoff]

    if not in_window:
        nearest = min(parsed, key=lambda x: x[0])[0]
        raise NoEligibleExpiry(
            f"No listed option expiration within {max_expiry_days} calendar days "
            f"(nearest Yahoo expiry: {nearest.isoformat()})"
        )

    chosen_date, chosen_expiry = max(in_window, key=lambda item: item[0])
    dte = (chosen_date - today).days
    return chosen_expiry, (
        f"Furthest listed expiry within {max_expiry_days} calendar days ({dte} DTE)"
    )


def _last_non_null_close(frame: pd.DataFrame) -> tuple[float, str | None]:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return math.nan, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return math.nan, None
    return safe_float(close.iloc[-1]), str(close.index[-1])


def fetch_spot(ticker_obj: yf.Ticker, include_extended: bool) -> tuple[float, str, str | None]:
    if include_extended:
        for interval in ("1m", "5m"):
            try:
                history = ticker_obj.history(
                    period="5d",
                    interval=interval,
                    auto_adjust=False,
                    prepost=True,
                    raise_errors=False,
                )
                value, timestamp = _last_non_null_close(history)
                if value > 0:
                    return value, f"history {interval} prepost close", timestamp
            except Exception:
                pass

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
                prepost=False,
                raise_errors=False,
            )
            value, timestamp = _last_non_null_close(history)
            if value > 0:
                return value, f"history {interval} regular close", timestamp
        except Exception:
            pass

    raise ValueError("Unable to obtain a valid underlying price from Yahoo")


def fetch_previous_close(ticker_obj: yf.Ticker) -> float:
    try:
        fast = ticker_obj.fast_info
        for key in ("previousClose", "previous_close", "regularMarketPreviousClose"):
            value = safe_float(fast.get(key))
            if value > 0:
                return value
    except Exception:
        pass

    try:
        history = ticker_obj.history(
            period="10d",
            interval="1d",
            auto_adjust=False,
            prepost=False,
            raise_errors=False,
        )
        close = pd.to_numeric(history.get("Close"), errors="coerce").dropna()
        if len(close) >= 2:
            return safe_float(close.iloc[-2])
        if len(close) == 1:
            return safe_float(close.iloc[-1])
    except Exception:
        pass

    return math.nan


def day_change_pct(spot: float, previous_close: float) -> float:
    if not math.isfinite(spot) or not math.isfinite(previous_close) or previous_close <= 0:
        return math.nan
    return (spot - previous_close) / previous_close * 100.0


def quote_mark(bid: float, ask: float) -> float:
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return math.nan


def bid_ask_spread_pct(bid: float, ask: float) -> float:
    mark = quote_mark(bid, ask)
    if not math.isfinite(mark) or mark <= 0:
        return math.nan
    return (ask - bid) / mark * 100.0


def _regular_contract_mask(frame: pd.DataFrame) -> pd.Series:
    if "contractSize" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["contractSize"].fillna("REGULAR").astype(str).eq("REGULAR")


def _normalise_chain(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    for column in [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["BidAskSpread_pct"] = result.apply(
        lambda row: bid_ask_spread_pct(
            safe_float(row["bid"]),
            safe_float(row["ask"]),
        ),
        axis=1,
    )
    return result


def select_iron_condor(
    ticker: str,
    spot: float,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expiry: str,
    expiry_note: str,
    config: IronCondorConfig,
    spot_source: str,
    spot_timestamp: str | None,
    underlying_day_change_pct: float,
) -> tuple[dict | None, str]:
    put_frame = _normalise_chain(puts)
    call_frame = _normalise_chain(calls)

    if put_frame.empty or call_frame.empty:
        return None, "Yahoo returned an empty put or call chain"

    dte = (
        datetime.strptime(expiry, "%Y-%m-%d").date()
        - (config.today or date.today())
    ).days

    def valid_long_leg(frame: pd.DataFrame) -> pd.DataFrame:
        # Protective long wings only need a valid executable ask and a regular contract.
        # Do not require same-day volume or minimum OI: farther-OTM wings often print
        # zero volume even when they are available to trade.
        return frame.loc[
            (frame["strike"] > 0)
            & (frame["ask"] > 0)
            & _regular_contract_mask(frame)
        ].copy()

    put_frame = valid_long_leg(put_frame)
    call_frame = valid_long_leg(call_frame)

    if put_frame.empty or call_frame.empty:
        return None, "No valid put/call contracts with executable asks"

    # Liquidity and spread thresholds apply to the two short premium-selling legs.
    put_short_spread_ok = put_frame["BidAskSpread_pct"].notna() & (
        put_frame["BidAskSpread_pct"] <= config.max_bid_ask_spread_pct
    )
    call_short_spread_ok = call_frame["BidAskSpread_pct"].notna() & (
        call_frame["BidAskSpread_pct"] <= config.max_bid_ask_spread_pct
    )

    max_short_dist = spot * config.max_short_distance_pct_of_spot / 100.0
    max_wing_width = max(0.5, spot * config.max_wing_width_pct_of_spot / 100.0)

    short_puts = put_frame.loc[
        (put_frame["strike"] < spot)
        & ((spot - put_frame["strike"]) <= max_short_dist)
        & (put_frame["bid"] > 0)
        & (put_frame["openInterest"].fillna(0) >= config.min_open_interest)
        & (put_frame["volume"].fillna(0) >= config.min_option_volume)
        & put_short_spread_ok
    ].sort_values("strike", ascending=False).head(16)

    short_calls = call_frame.loc[
        (call_frame["strike"] > spot)
        & ((call_frame["strike"] - spot) <= max_short_dist)
        & (call_frame["bid"] > 0)
        & (call_frame["openInterest"].fillna(0) >= config.min_open_interest)
        & (call_frame["volume"].fillna(0) >= config.min_option_volume)
        & call_short_spread_ok
    ].sort_values("strike", ascending=True).head(16)

    if short_puts.empty or short_calls.empty:
        return None, "No OTM short put/call legs passed short-leg bid, OI, volume, and spread filters"

    valid_candidates: list[dict] = []

    for _, sp in short_puts.iterrows():
        short_put_strike = safe_float(sp["strike"])
        long_puts = put_frame.loc[
            (put_frame["strike"] < short_put_strike)
            & ((short_put_strike - put_frame["strike"]) <= max_wing_width)
        ].sort_values("strike", ascending=False).head(10)

        if long_puts.empty:
            continue

        for _, sc in short_calls.iterrows():
            short_call_strike = safe_float(sc["strike"])
            long_calls = call_frame.loc[
                (call_frame["strike"] > short_call_strike)
                & ((call_frame["strike"] - short_call_strike) <= max_wing_width)
            ].sort_values("strike", ascending=True).head(10)

            if long_calls.empty:
                continue

            short_credit = safe_float(sp["bid"]) + safe_float(sc["bid"])
            if not math.isfinite(short_credit):
                continue

            for _, lp in long_puts.iterrows():
                long_put_strike = safe_float(lp["strike"])
                put_width = short_put_strike - long_put_strike
                if put_width <= 0:
                    continue

                for _, lc in long_calls.iterrows():
                    long_call_strike = safe_float(lc["strike"])
                    call_width = long_call_strike - short_call_strike
                    if call_width <= 0:
                        continue

                    long_debit = safe_float(lp["ask"]) + safe_float(lc["ask"])
                    if not math.isfinite(long_debit):
                        continue

                    net_credit = short_credit - long_debit
                    if net_credit < config.min_credit_per_share:
                        continue

                    max_loss_per_share = max(put_width, call_width) - net_credit
                    if max_loss_per_share <= 0:
                        continue

                    max_profit = net_credit * 100.0
                    max_loss = max_loss_per_share * 100.0
                    profit_loss_ratio = max_profit / max_loss

                    if profit_loss_ratio < config.min_profit_loss_ratio:
                        continue

                    lower_be = short_put_strike - net_credit
                    upper_be = short_call_strike + net_credit

                    # Keep the current stock price safely inside the breakeven band,
                    # and reject highly lopsided ranges where spot sits near one edge.
                    downside_buffer = spot - lower_be
                    upside_buffer = upper_be - spot
                    if downside_buffer <= 0 or upside_buffer <= 0:
                        continue

                    larger_buffer = max(downside_buffer, upside_buffer)
                    breakeven_balance_ratio = (
                        min(downside_buffer, upside_buffer) / larger_buffer
                        if larger_buffer > 0
                        else 0.0
                    )
                    if breakeven_balance_ratio < config.min_breakeven_balance_ratio:
                        continue

                    profit_zone = upper_be - lower_be
                    profit_zone_pct = profit_zone / spot * 100.0
                    downside_buffer_pct = downside_buffer / spot * 100.0
                    upside_buffer_pct = upside_buffer / spot * 100.0

                    nearest_short_distance_pct = min(
                        (spot - short_put_strike) / spot * 100.0,
                        (short_call_strike - spot) / spot * 100.0,
                    )

                    total_oi = sum(
                        safe_int(x["openInterest"]) for x in (lp, sp, sc, lc)
                    )
                    total_volume = sum(
                        safe_int(x["volume"]) for x in (lp, sp, sc, lc)
                    )

                    candidate = {
                        "Ticker": ticker,
                        "Spot": round_or_nan(spot),
                        "SpotSource": spot_source,
                        "SpotTimestamp": spot_timestamp or "",
                        "UnderlyingDayChange_pct": round_or_nan(
                            underlying_day_change_pct
                        ),
                        "Expiry": expiry,
                        "ExpirySelection": expiry_note,
                        "DaysToExpiry": dte,
                        "LongPutStrike": round_or_nan(long_put_strike),
                        "ShortPutStrike": round_or_nan(short_put_strike),
                        "ShortCallStrike": round_or_nan(short_call_strike),
                        "LongCallStrike": round_or_nan(long_call_strike),
                        "LongPutAsk": round_or_nan(safe_float(lp["ask"])),
                        "ShortPutBid": round_or_nan(safe_float(sp["bid"])),
                        "ShortCallBid": round_or_nan(safe_float(sc["bid"])),
                        "LongCallAsk": round_or_nan(safe_float(lc["ask"])),
                        "PutWingWidth": round_or_nan(put_width),
                        "CallWingWidth": round_or_nan(call_width),
                        "NetCredit_perShare": round_or_nan(net_credit),
                        "NetCredit_perCondor": round_or_nan(net_credit * 100.0),
                        "MaxProfit": round_or_nan(max_profit),
                        "MaxLoss": round_or_nan(max_loss),
                        "ProfitLossRatio": round_or_nan(profit_loss_ratio, 3),
                        "LowerBreakeven": round_or_nan(lower_be),
                        "UpperBreakeven": round_or_nan(upper_be),
                        "DownsideBreakevenBuffer": round_or_nan(downside_buffer),
                        "UpsideBreakevenBuffer": round_or_nan(upside_buffer),
                        "DownsideBreakevenBuffer_pct": round_or_nan(downside_buffer_pct),
                        "UpsideBreakevenBuffer_pct": round_or_nan(upside_buffer_pct),
                        "BreakevenBalanceRatio": round_or_nan(breakeven_balance_ratio, 3),
                        "BreakevenRange": round_or_nan(profit_zone),
                        "BreakevenRange_pct": round_or_nan(profit_zone_pct),
                        # Backward-compatible aliases used by older exports/UI.
                        "ProfitZoneWidth": round_or_nan(profit_zone),
                        "ProfitZoneWidth_pct": round_or_nan(profit_zone_pct),
                        "NearestShortDistance_pct": round_or_nan(
                            nearest_short_distance_pct
                        ),
                        "TotalOpenInterest": total_oi,
                        "TotalOptionVolume": total_volume,
                    }
                    valid_candidates.append(candidate)

    if not valid_candidates:
        return None, "No iron condor met credit, profit/loss, wing-width, and centered-breakeven filters after short-leg liquidity screening"

    # Pick the best structure for this ticker using the same score philosophy
    # used for global ranking. The scores are recomputed globally after all
    # tickers finish so the displayed 0-100 values are comparable across stocks.
    ticker_scored = score_candidates(valid_candidates)
    best = max(
        ticker_scored,
        key=lambda row: (
            safe_float(row.get("OverallScore")),
            safe_float(row.get("BreakevenRange_pct")),
            safe_float(row.get("BreakevenBalanceRatio")),
            safe_float(row.get("ProfitLossRatio")),
            safe_float(row.get("MaxProfit")),
            -safe_float(row.get("MaxLoss")),
        ),
    )
    return best, (
        "Selected by weighted score: 45% breakeven range, 35% profit/loss, "
        "15% max profit, 5% lower max loss"
    )


def scan_one_ticker(
    ticker: str,
    config: IronCondorConfig,
) -> tuple[dict | None, dict, dict | None]:
    for attempt in range(1, max(1, config.retries) + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            spot, source, timestamp = fetch_spot(
                ticker_obj,
                config.include_extended_spot,
            )
            previous_close = fetch_previous_close(ticker_obj)
            underlying_day_change = day_change_pct(spot, previous_close)

            expiry, expiry_note = choose_furthest_expiry_within_window(
                ticker_obj.options,
                today=config.today,
                max_expiry_days=config.max_expiry_days,
            )
            chain = ticker_obj.option_chain(expiry)

            candidate, reason = select_iron_condor(
                ticker=ticker,
                spot=spot,
                calls=chain.calls,
                puts=chain.puts,
                expiry=expiry,
                expiry_note=expiry_note,
                config=config,
                spot_source=source,
                spot_timestamp=timestamp,
                underlying_day_change_pct=underlying_day_change,
            )

            diagnostic = {
                "Ticker": ticker,
                "Spot": round_or_nan(spot),
                "Expiry": expiry,
                "ExpirySelection": expiry_note,
                "UnderlyingDayChange_pct": round_or_nan(underlying_day_change),
                "Result": "QUALIFIED" if candidate is not None else "NO_MATCH",
                "Reason": reason,
            }
            return candidate, diagnostic, None

        except NoEligibleExpiry as exc:
            return None, {
                "Ticker": ticker,
                "Spot": math.nan,
                "Result": "NO_EXPIRY_WINDOW",
                "Reason": str(exc),
            }, None
        except Exception as exc:
            if attempt >= max(1, config.retries):
                error = {
                    "Ticker": ticker,
                    "Error": f"{type(exc).__name__}: {exc}",
                }
                return None, {
                    "Ticker": ticker,
                    "Spot": math.nan,
                    "Result": "ERROR",
                    "Reason": error["Error"],
                }, error
            time.sleep(max(0.0, config.retry_delay_seconds))

    raise RuntimeError("Unreachable retry state")


def scan_iron_condors(
    tickers: list[str],
    config: IronCondorConfig,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> IronCondorOutput:
    candidates: list[dict] = []
    diagnostics: list[dict] = []
    errors: list[dict] = []

    if not tickers:
        return IronCondorOutput(candidates, diagnostics, errors)

    workers = max(1, min(int(config.max_workers), 8))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_ticker, ticker, config): ticker
            for ticker in tickers
        }

        for done, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                candidate, diagnostic, error = future.result()
            except Exception as exc:
                candidate = None
                diagnostic = {
                    "Ticker": ticker,
                    "Spot": math.nan,
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
                progress_callback(
                    done,
                    len(tickers),
                    ticker,
                    diagnostic["Result"],
                )

    candidates = score_candidates(candidates)
    candidates.sort(
        key=lambda row: (
            -safe_float(row.get("OverallScore")),
            -safe_float(row.get("BreakevenRange_pct")),
            -safe_float(row.get("BreakevenBalanceRatio")),
            -safe_float(row.get("ProfitLossRatio")),
            -safe_float(row.get("MaxProfit")),
            safe_float(row.get("MaxLoss")),
            -safe_float(row.get("TotalOpenInterest")),
        )
    )
    candidates = candidates[: max(1, int(config.top_n))]

    diagnostics.sort(key=lambda row: row["Ticker"])
    errors.sort(key=lambda row: row["Ticker"])

    return IronCondorOutput(candidates, diagnostics, errors)
