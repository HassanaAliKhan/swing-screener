#!/usr/bin/env python3
"""
Hourly Swing Screener — confirmed and fresh-momentum long setups.

The scanner keeps the existing confirmed patterns and adds two earlier signals:
  - FRESH_BREAKOUT: the latest completed hourly candle has just broken a base.
  - EARLY_RECOVERY: a recent weak move is reclaiming the 20 EMA / short structure.

Use the Fresh momentum profile in the Streamlit app for these early signals.
It intentionally requires tighter entry extension and stronger volume than the
relaxed profiles because first breakouts can fail more often.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"The 'generic' unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=UserWarning, module=r"^yfinance(\..*)?$")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing packages. Run setup.bat or:\n"
        "python -m pip install -U yfinance pandas numpy"
    ) from exc


DEFAULT_WATCHLIST = Path(__file__).with_name("watchlist.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen hourly long setups with confirmed and fresh-momentum patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core I/O / data.
    parser.add_argument("--tickers-file", type=Path, default=None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="60m")
    parser.add_argument("--include-prepost", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--min-completed-bars", type=int, default=60)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)

    # Global filters.
    parser.add_argument("--target-pct", type=float, default=5.0)
    parser.add_argument("--min-score", type=int, default=72)
    parser.add_argument("--max-risk-pct", type=float, default=3.25)
    parser.add_argument("--min-reward-risk", type=float, default=1.60)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--min-hourly-dollar-volume", type=float, default=500_000)
    parser.add_argument("--min-rel-volume", type=float, default=0.85)

    # Confirmation / resistance gates.
    parser.add_argument("--allow-resistance-before-target", action="store_true")
    parser.add_argument("--resistance-target-buffer-pct", type=float, default=0.50)
    parser.add_argument("--allow-neutral-candle", action="store_true")

    # Existing confirmed-pattern controls.
    parser.add_argument("--pullback-touch-pct", type=float, default=1.20)
    parser.add_argument("--pullback-max-ema20-distance-pct", type=float, default=3.00)
    parser.add_argument("--pullback-volume-multiplier", type=float, default=1.10)
    parser.add_argument("--allow-uptrend-continuation", action="store_true")
    parser.add_argument("--breakout-event-volume-multiplier", type=float, default=1.25)
    parser.add_argument("--breakout-retest-tolerance-pct", type=float, default=1.50)
    parser.add_argument("--breakout-max-extension-pct", type=float, default=2.50)
    parser.add_argument("--range-support-distance-pct", type=float, default=2.00)
    parser.add_argument("--reversal-higher-low-pct", type=float, default=0.60)
    parser.add_argument("--reversal-structure-break-pct", type=float, default=0.10)
    parser.add_argument("--reversal-min-rel-volume", type=float, default=1.00)
    parser.add_argument("--allow-reversal-below-ema50", action="store_true")

    # Earlier, current-hour momentum patterns. Disabled by default so CLI behavior
    # remains confirmation-oriented unless a caller deliberately enables them.
    parser.add_argument("--allow-fresh-breakout", action="store_true")
    parser.add_argument("--fresh-breakout-lookback-bars", type=int, default=20)
    parser.add_argument("--fresh-breakout-min-rel-volume", type=float, default=1.00)
    parser.add_argument("--fresh-breakout-max-extension-pct", type=float, default=1.50)
    parser.add_argument("--allow-early-recovery", action="store_true")
    parser.add_argument("--early-recovery-lookback-bars", type=int, default=6)
    parser.add_argument("--early-recovery-min-rel-volume", type=float, default=0.90)
    parser.add_argument("--early-recovery-max-ema20-distance-pct", type=float, default=1.75)
    parser.add_argument("--early-recovery-structure-break-pct", type=float, default=0.10)

    return parser.parse_args()


def load_tickers(path: Path | None) -> list[str]:
    selected = path or DEFAULT_WATCHLIST
    if not selected.exists():
        raise SystemExit(
            f"Ticker file not found: {selected}\n"
            "Keep watchlist.txt beside this script or pass --tickers-file FILE."
        )

    result: list[str] = []
    seen: set[str] = set()
    for raw_line in selected.read_text(encoding="utf-8").splitlines():
        clean_line = raw_line.split("#", 1)[0].replace(",", " ").strip()
        for token in clean_line.split():
            ticker = token.upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                result.append(ticker)

    if not result:
        raise SystemExit(f"No ticker symbols found in {selected}")
    return result


def _field_positions(data: pd.DataFrame, field: str) -> list[int]:
    """Find physical columns matching an OHLCV field in flat or MultiIndex data."""
    wanted = field.upper()
    positions: list[int] = []

    if isinstance(data.columns, pd.MultiIndex):
        for index, label in enumerate(data.columns):
            if wanted in {str(part).upper() for part in label}:
                positions.append(index)
    else:
        for index, label in enumerate(data.columns):
            if str(label).upper() == wanted:
                positions.append(index)

    return positions


def _select_field_series(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Return exactly one numeric series even when Yahoo yields duplicate labels."""
    positions = _field_positions(data, field)
    if not positions:
        raise ValueError(f"Missing {field} column in Yahoo response")

    ticker_upper = ticker.upper()

    def quality(position: int) -> tuple[int, int]:
        label = data.columns[position]
        label_parts = (
            {str(part).upper() for part in label}
            if isinstance(label, tuple)
            else {str(label).upper()}
        )
        ticker_match = int(ticker_upper in label_parts)
        series = pd.to_numeric(data.iloc[:, position], errors="coerce")
        return ticker_match, int(series.notna().sum())

    chosen = max(positions, key=quality)
    series = pd.to_numeric(data.iloc[:, chosen], errors="coerce")
    series.name = field
    return series


def normalize_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize Yahoo output into a one-dimensional OHLCV table."""
    if data is None or data.empty:
        raise ValueError("Empty Yahoo response")

    required = ["Open", "High", "Low", "Close", "Volume"]
    normalized = pd.DataFrame(index=data.index)
    for field in required:
        normalized[field] = _select_field_series(data, field, ticker)

    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    normalized = normalized.dropna(subset=required)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    if normalized.empty:
        raise ValueError("Yahoo response had no complete OHLCV rows after normalization")
    return normalized


def _download_once(
    ticker: str,
    period: str,
    interval: str,
    include_prepost: bool,
) -> pd.DataFrame:
    common = dict(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=include_prepost,
        progress=False,
        threads=False,
        timeout=20,
        group_by="column",
    )
    try:
        return yf.download(**common, multi_level_index=False)
    except TypeError:
        return yf.download(**common)


def _history_fallback(
    ticker: str,
    period: str,
    interval: str,
    include_prepost: bool,
) -> pd.DataFrame:
    fallback_interval = "1h" if interval == "60m" else interval
    return yf.Ticker(ticker).history(
        period=period,
        interval=fallback_interval,
        auto_adjust=False,
        prepost=include_prepost,
        raise_errors=False,
    )


def _interval_duration(interval: str) -> pd.Timedelta:
    if interval == "60m":
        return pd.Timedelta(minutes=60)
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    return pd.Timedelta(minutes=60)


def completed_bars_only(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Drop zero-volume placeholders and remove the latest bar only when it is still open.

    The prior implementation removed the final nonzero bar unconditionally. That made
    after-close scans stale by an additional hour. Yahoo labels intraday bars by their
    start timestamp, so a bar is considered complete after its full interval plus a
    two-minute delivery grace period.
    """
    cleaned = data.loc[data["Volume"].fillna(0).gt(0)].copy()
    if cleaned.empty:
        return cleaned

    last_timestamp = pd.Timestamp(cleaned.index[-1])
    duration = _interval_duration(interval)

    if last_timestamp.tzinfo is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        now = pd.Timestamp.now(tz=last_timestamp.tz)

    if now < last_timestamp + duration + pd.Timedelta(minutes=2):
        cleaned = cleaned.iloc[:-1].copy()

    return cleaned


def fetch_hourly(
    ticker: str,
    period: str,
    interval: str,
    include_prepost: bool,
    min_completed_bars: int,
    download_retries: int,
    retry_delay_seconds: float,
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Download hourly data with retries, fallback, and completed-candle handling."""
    attempts: list[str] = []
    retries = max(1, int(download_retries))

    for attempt in range(1, retries + 1):
        for source, fetcher in (
            ("download", _download_once),
            ("history fallback", _history_fallback),
        ):
            try:
                raw = fetcher(ticker, period, interval, include_prepost)
                if raw is None or raw.empty:
                    attempts.append(f"{source} {attempt}: empty response")
                    continue

                data = completed_bars_only(normalize_columns(raw, ticker), interval)
                if len(data) >= min_completed_bars:
                    return ticker, data, None

                attempts.append(
                    f"{source} {attempt}: only {len(data)} completed bars "
                    f"(need {min_completed_bars})"
                )
            except Exception as exc:
                attempts.append(f"{source} {attempt}: {type(exc).__name__}: {exc}")

        if attempt < retries:
            time.sleep(max(0.0, retry_delay_seconds))

    return ticker, None, " | ".join(attempts[-4:])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()

    previous_close = out["Close"].shift(1)
    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - previous_close).abs(),
            (out["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = true_range.rolling(14).mean()
    out["AvgVol20"] = out["Volume"].rolling(20).mean()
    out["RelVol20"] = out["Volume"] / out["AvgVol20"].replace(0, np.nan)
    out["AvgDollarVol20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["EMA20Slope5"] = out["EMA20"] - out["EMA20"].shift(5)
    out["EMA50Slope8"] = out["EMA50"] - out["EMA50"].shift(8)
    return out.dropna().copy()


def pct(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return math.nan
    return (float(value) / float(reference) - 1.0) * 100.0


def num(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if np.isfinite(value) else math.nan


def bullish_confirmation(bar: pd.Series, allow_neutral: bool) -> bool:
    if allow_neutral:
        return bool(bar["Close"] >= bar["Open"] * 0.998)
    candle_range = max(float(bar["High"] - bar["Low"]), 1e-9)
    close_location = float((bar["Close"] - bar["Low"]) / candle_range)
    return bool(bar["Close"] > bar["Open"] and close_location >= 0.60)


def bearish_confirmation(bar: pd.Series) -> bool:
    candle_range = max(float(bar["High"] - bar["Low"]), 1e-9)
    close_location = float((bar["Close"] - bar["Low"]) / candle_range)
    return bool(bar["Close"] < bar["Open"] and close_location <= 0.40)


# ---------------------------------------------------------------------------
# Chart-state helpers.
# ---------------------------------------------------------------------------

def detect_uptrend(df: pd.DataFrame) -> bool:
    bar = df.iloc[-1]
    return bool(
        bar["Close"] > bar["EMA20"] > bar["EMA50"]
        and bar["EMA20Slope5"] > 0
        and bar["EMA50Slope8"] >= 0
    )


def detect_downtrend(df: pd.DataFrame) -> bool:
    bar = df.iloc[-1]
    return bool(
        bar["Close"] < bar["EMA20"] < bar["EMA50"]
        and bar["EMA20Slope5"] < 0
        and bar["EMA50Slope8"] <= 0
    )


def range_levels(df: pd.DataFrame) -> tuple[bool, float, float]:
    window = df.iloc[-31:]
    bar = df.iloc[-1]
    support = float(window["Low"].min())
    resistance = float(window["High"].max())
    width_pct = pct(resistance, support)
    ema_gap_pct = abs(pct(bar["EMA20"], bar["EMA50"]))
    flat_emas = abs(float(bar["EMA20Slope5"])) <= float(bar["ATR14"]) * 0.45
    is_range = bool(4.0 <= width_pct <= 18.0 and ema_gap_pct <= 1.50 and flat_emas)
    return is_range, support, resistance


def breakout_level(df: pd.DataFrame, args: argparse.Namespace) -> tuple[bool, float]:
    base = df.iloc[-49:-9]
    recent = df.iloc[-9:-2]
    if len(base) < 20 or len(recent) < 4:
        return False, math.nan

    resistance = float(base["High"].max())
    breakout_bars = recent[
        (recent["Close"] > resistance * 1.002)
        & (recent["Volume"] >= recent["AvgVol20"] * args.breakout_event_volume_multiplier)
    ]
    return not breakout_bars.empty, resistance


def breakdown_level(df: pd.DataFrame) -> tuple[bool, float]:
    base = df.iloc[-49:-9]
    recent = df.iloc[-9:]
    if len(base) < 20:
        return False, math.nan

    support = float(base["Low"].min())
    latest = df.iloc[-1]
    broke_support = float(recent["Low"].min()) < support * 0.992
    still_below = latest["Close"] < support * 0.998
    bearish_now = bearish_confirmation(latest) or latest["Close"] < latest["EMA20"]
    return bool(broke_support and still_below and bearish_now), support


def reversal_structure(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[bool, float, float]:
    bar = df.iloc[-1]
    before = df.iloc[-23:-8]
    recent = df.iloc[-8:-1]
    if len(before) < 10 or len(recent) < 5:
        return False, math.nan, math.nan

    prior_low = float(before["Low"].min())
    higher_low = float(recent["Low"].min())
    structure_high = float(recent["High"].max())
    was_weak = bool(
        before.iloc[-1]["Close"] < before.iloc[-1]["EMA50"]
        or before.iloc[-1]["EMA20"] < before.iloc[-1]["EMA50"]
    )
    ema_ok = bar["Close"] > bar["EMA20"] and (
        bar["Close"] > bar["EMA50"] or args.allow_reversal_below_ema50
    )
    confirmed = bool(
        was_weak
        and higher_low >= prior_low * (1.0 + args.reversal_higher_low_pct / 100.0)
        and bar["Close"] >= structure_high * (1.0 + args.reversal_structure_break_pct / 100.0)
        and ema_ok
        and bar["EMA20Slope5"] >= 0
        and bar["RelVol20"] >= args.reversal_min_rel_volume
        and bullish_confirmation(bar, args.allow_neutral_candle)
    )
    return confirmed, higher_low, structure_high


def nearest_overhead_resistance(df: pd.DataFrame, entry: float, lookback: int = 40) -> float:
    """Return the nearest recent high genuinely above entry, if one exists."""
    window = df.iloc[-(lookback + 1):-1]
    if window.empty:
        return math.nan

    overhead = window.loc[window["High"] > entry * 1.001, "High"]
    return float(overhead.min()) if not overhead.empty else math.nan


def target_clear_of_resistance(
    entry: float,
    target_pct: float,
    resistance: float,
    args: argparse.Namespace,
) -> bool:
    if args.allow_resistance_before_target or not np.isfinite(resistance):
        return True
    target = entry * (1.0 + target_pct / 100.0)
    allowed_resistance = resistance * (1.0 + args.resistance_target_buffer_pct / 100.0)
    return bool(target <= allowed_resistance)


def make_candidate(
    ticker: str,
    pattern: str,
    score: int,
    bar: pd.Series,
    entry: float,
    stop: float,
    target_pct: float,
    support: float,
    resistance: float,
    reason: str,
    signal_phase: str = "Confirmed",
    signal_age_bars: int | None = None,
) -> dict[str, Any] | None:
    if not np.isfinite(stop) or stop >= entry:
        return None

    target = entry * (1.0 + target_pct / 100.0)
    risk_pct = (entry - stop) / entry * 100.0
    reward_risk = target_pct / risk_pct if risk_pct > 0 else math.nan
    resistance_distance = pct(resistance, entry) if np.isfinite(resistance) else math.nan

    return {
        "Ticker": ticker,
        "Pattern": pattern,
        "SignalPhase": signal_phase,
        "SignalAgeBars": signal_age_bars if signal_age_bars is not None else math.nan,
        "Action": "LONG_ENTRY_REVIEW",
        "Score": int(score),
        "LastCompletedHourlyBar": str(bar.name),
        "Entry": num(entry),
        "Stop": num(stop),
        "Target_5pct": num(target),
        "Risk_pct": num(risk_pct),
        "RewardRisk_to_Target": num(reward_risk),
        "Close": num(bar["Close"]),
        "EMA20": num(bar["EMA20"]),
        "EMA50": num(bar["EMA50"]),
        "Distance_to_EMA20_pct": num(pct(bar["Close"], bar["EMA20"])),
        "Distance_to_EMA50_pct": num(pct(bar["Close"], bar["EMA50"])),
        "RelVol20": num(bar["RelVol20"]),
        "AvgHourlyDollarVol20": num(bar["AvgDollarVol20"], 0),
        "Support": num(support),
        "Resistance": num(resistance),
        "ResistanceDistance_pct": num(resistance_distance),
        "ResistanceBefore5pctTarget": bool(
            np.isfinite(resistance_distance) and resistance_distance < target_pct
        ),
        "Reason": reason,
    }


# ---------------------------------------------------------------------------
# Earlier, current-hour signals.
# ---------------------------------------------------------------------------

def try_fresh_breakout(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Surface the first completed hourly candle through a defined base high."""
    if not args.allow_fresh_breakout:
        return None

    lookback = max(10, int(args.fresh_breakout_lookback_bars))
    base = df.iloc[-(lookback + 1):-1]
    if len(base) < lookback:
        return None

    bar = df.iloc[-1]
    entry = float(bar["Close"])
    base_high = float(base["High"].max())
    extension_pct = pct(entry, base_high)
    recent_lows = df.iloc[-3:]["Low"]
    support = float(recent_lows.min())
    resistance = nearest_overhead_resistance(df, entry, lookback=max(40, lookback))

    candle_range = max(float(bar["High"] - bar["Low"]), 1e-9)
    close_location = float((bar["Close"] - bar["Low"]) / candle_range)
    ema_turning = float(bar["EMA20"]) >= float(df.iloc[-3]["EMA20"])
    price_above_ema20 = bool(bar["Close"] > bar["EMA20"])
    target_room_ok = target_clear_of_resistance(entry, args.target_pct, resistance, args)

    is_fresh = bool(
        entry > base_high * 1.001
        and 0.10 <= extension_pct <= args.fresh_breakout_max_extension_pct
        and close_location >= 0.65
        and bar["RelVol20"] >= args.fresh_breakout_min_rel_volume
        and price_above_ema20
        and ema_turning
        and target_room_ok
    )
    if not is_fresh:
        return None

    stop = min(
        support - 0.10 * bar["ATR14"],
        base_high - 0.30 * bar["ATR14"],
    )
    score = 58
    score += 12 if bar["RelVol20"] >= 1.35 else 0
    score += 8 if bar["Close"] > bar["EMA50"] else 0
    score += 7 if extension_pct <= 0.75 else 0
    score += 5 if bar["Close"] > df.iloc[-2]["High"] else 0

    return make_candidate(
        ticker=ticker,
        pattern="FRESH_BREAKOUT",
        score=score,
        bar=bar,
        entry=entry,
        stop=float(stop),
        target_pct=args.target_pct,
        support=support,
        resistance=resistance,
        reason=(
            "Latest completed hourly candle broke the recent base high with a close "
            "near its high, volume confirmation, and limited extension."
        ),
        signal_phase="Fresh current-hour breakout",
        signal_age_bars=0,
    )


def try_early_recovery(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Find an early reclaim rather than waiting for the full EMA50 trend confirmation."""
    if not args.allow_early_recovery:
        return None

    lookback = max(5, int(args.early_recovery_lookback_bars))
    recent = df.iloc[-(lookback + 1):-1]
    if len(recent) < lookback:
        return None

    bar = df.iloc[-1]
    previous = df.iloc[-2]
    entry = float(bar["Close"])
    support = float(df.iloc[-4:]["Low"].min())
    short_structure_high = float(recent["High"].max())
    resistance = nearest_overhead_resistance(df, entry, lookback=40)

    distance_to_ema20 = pct(entry, float(bar["EMA20"]))
    recent_weakness = bool((recent["Close"] <= recent["EMA20"] * 1.005).any())
    reclaimed_ema20 = bool(
        previous["Close"] <= previous["EMA20"] * 1.005
        and bar["Close"] > bar["EMA20"] * 1.001
    )
    short_structure_break = bool(
        bar["Close"] >= short_structure_high
        * (1.0 + args.early_recovery_structure_break_pct / 100.0)
    )
    ema_not_falling_hard = bool(
        bar["EMA20"] >= df.iloc[-3]["EMA20"] - 0.10 * bar["ATR14"]
    )
    target_room_ok = target_clear_of_resistance(entry, args.target_pct, resistance, args)

    is_early_recovery = bool(
        recent_weakness
        and bullish_confirmation(bar, False)
        and 0.0 <= distance_to_ema20 <= args.early_recovery_max_ema20_distance_pct
        and (reclaimed_ema20 or short_structure_break)
        and ema_not_falling_hard
        and bar["RelVol20"] >= args.early_recovery_min_rel_volume
        and target_room_ok
    )
    if not is_early_recovery:
        return None

    stop = min(
        support - 0.10 * bar["ATR14"],
        bar["EMA20"] - 0.45 * bar["ATR14"],
    )
    score = 55
    score += 12 if bar["RelVol20"] >= 1.25 else 0
    score += 8 if short_structure_break else 0
    score += 7 if bar["Close"] > bar["EMA50"] else 0
    score += 5 if reclaimed_ema20 else 0
    score += 5 if distance_to_ema20 <= 1.00 else 0

    return make_candidate(
        ticker=ticker,
        pattern="EARLY_RECOVERY",
        score=score,
        bar=bar,
        entry=entry,
        stop=float(stop),
        target_pct=args.target_pct,
        support=support,
        resistance=resistance,
        reason=(
            "Recent weakness is reclaiming the 20 EMA or breaking short structure "
            "on the latest completed hourly candle before a full mature trend forms."
        ),
        signal_phase="Fresh current-hour recovery",
        signal_age_bars=0,
    )


# ---------------------------------------------------------------------------
# Existing confirmed long patterns.
# ---------------------------------------------------------------------------

def try_uptrend_pullback(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not detect_uptrend(df):
        return None

    bar = df.iloc[-1]
    recent5 = df.iloc[-5:]
    previous3 = df.iloc[-4:-1]
    prior20 = df.iloc[-21:-1]
    prior40 = df.iloc[-41:-1]

    entry = float(bar["Close"])
    resistance = float(prior40["High"].max())
    support = float(recent5["Low"].min())
    recent_low = support

    touched_ema20 = recent_low <= bar["EMA20"] * (1.0 + args.pullback_touch_pct / 100.0)
    close_near_ema20 = pct(entry, bar["EMA20"]) <= args.pullback_max_ema20_distance_pct
    confirmation = bullish_confirmation(bar, args.allow_neutral_candle)
    pullback_volume_ok = float(previous3["Volume"].mean()) <= float(prior20["Volume"].mean()) * args.pullback_volume_multiplier
    target_room_ok = target_clear_of_resistance(entry, args.target_pct, resistance, args)

    if not (touched_ema20 and close_near_ema20 and confirmation and pullback_volume_ok and target_room_ok):
        return None

    stop = min(
        recent_low - 0.10 * bar["ATR14"],
        bar["EMA50"] - 0.35 * bar["ATR14"],
    )
    score = 55
    score += 10 if bar["RelVol20"] >= 1.10 else 0
    score += 8 if pullback_volume_ok else 0
    score += 7 if target_clear_of_resistance(entry, args.target_pct, resistance, args) else 0
    score += 5 if pct(bar["EMA20"], bar["EMA50"]) >= 1.00 else 0

    return make_candidate(
        ticker, "UPTREND_PULLBACK", score, bar, entry, float(stop), args.target_pct,
        support, resistance,
        "Hourly uptrend; recent pullback reached the 20 EMA and the latest completed candle confirmed upward.",
    )


def try_uptrend_continuation(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.allow_uptrend_continuation or not detect_uptrend(df):
        return None

    bar = df.iloc[-1]
    recent5 = df.iloc[-5:]
    prior40 = df.iloc[-41:-1]
    entry = float(bar["Close"])
    support = float(recent5["Low"].min())
    resistance = float(prior40["High"].max())
    distance = pct(entry, bar["EMA20"])

    if not (
        0.0 <= distance <= args.pullback_max_ema20_distance_pct
        and bullish_confirmation(bar, args.allow_neutral_candle)
        and target_clear_of_resistance(entry, args.target_pct, resistance, args)
    ):
        return None

    stop = min(
        support - 0.15 * bar["ATR14"],
        bar["EMA50"] - 0.30 * bar["ATR14"],
    )
    score = 53
    score += 8 if bar["RelVol20"] >= 0.85 else 0
    score += 7 if pct(bar["EMA20"], bar["EMA50"]) >= 0.75 else 0
    score += 5 if bar["Close"] > df.iloc[-2]["High"] else 0

    return make_candidate(
        ticker, "UPTREND_CONTINUATION", score, bar, entry, float(stop), args.target_pct,
        support, resistance,
        "Optional continuation setup: price remains above rising hourly EMAs but did not meet the stricter pullback definition.",
    )


def try_breakout_retest(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    is_breakout, breakout_level_value = breakout_level(df, args)
    if not is_breakout:
        return None

    bar = df.iloc[-1]
    retest_window = df.iloc[-3:]
    prior40 = df.iloc[-41:-1]
    entry = float(bar["Close"])
    support = float(retest_window["Low"].min())
    resistance = float(prior40["High"].max())

    held_level = support >= breakout_level_value * (1.0 - args.breakout_retest_tolerance_pct / 100.0)
    confirmation = bullish_confirmation(bar, args.allow_neutral_candle)
    trend_ok = bar["Close"] > bar["EMA50"] and bar["EMA20Slope5"] >= 0
    not_extended = pct(entry, breakout_level_value) <= args.breakout_max_extension_pct
    target_room_ok = target_clear_of_resistance(entry, args.target_pct, resistance, args)

    if not (held_level and confirmation and trend_ok and not_extended and target_room_ok):
        return None

    stop = min(
        support - 0.10 * bar["ATR14"],
        breakout_level_value - 0.45 * bar["ATR14"],
    )
    score = 62
    score += 10 if bar["RelVol20"] >= 1.00 else 0
    score += 8 if bar["EMA20"] > bar["EMA50"] else 0
    score += 5 if entry <= breakout_level_value * 1.02 else 0
    score += 5 if pct(resistance, entry) >= 2.0 else 0

    return make_candidate(
        ticker, "BREAKOUT_RETEST", score, bar, entry, float(stop), args.target_pct,
        support, max(resistance, breakout_level_value),
        "Recent breakout was followed by a retest that held prior resistance as support.",
    )


def try_range_support_bounce(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    is_range, support, resistance = range_levels(df)
    if not is_range:
        return None

    bar = df.iloc[-1]
    entry = float(bar["Close"])
    near_support = entry <= support * (1.0 + args.range_support_distance_pct / 100.0)
    confirmation = bullish_confirmation(bar, args.allow_neutral_candle)
    target_room_ok = target_clear_of_resistance(entry, args.target_pct, resistance, args)

    if not (near_support and confirmation and target_room_ok):
        return None

    stop = support - 0.25 * bar["ATR14"]
    score = 56
    score += 8 if bar["RelVol20"] >= 1.00 else 0
    score += 5 if entry > df.iloc[-2]["High"] else 0
    score += 5 if pct(resistance, entry) >= args.target_pct else 0

    return make_candidate(
        ticker, "RANGE_SUPPORT_BOUNCE", score, bar, entry, float(stop), args.target_pct,
        support, resistance,
        "Sideways range; price is near established support with latest completed candle confirming upward.",
    )


def try_reversal_confirm(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    confirmed, higher_low, structure_high = reversal_structure(df, args)
    if not confirmed:
        return None

    bar = df.iloc[-1]
    prior40 = df.iloc[-61:-1]
    entry = float(bar["Close"])
    resistance = float(prior40["High"].max())
    if not target_clear_of_resistance(entry, args.target_pct, resistance, args):
        return None

    stop = min(
        higher_low - 0.20 * bar["ATR14"],
        bar["EMA50"] - 0.30 * bar["ATR14"],
    )
    score = 60
    score += 10 if bar["RelVol20"] >= 1.25 else 0
    score += 7 if bar["EMA20"] > bar["EMA50"] else 0
    score += 4 if entry > structure_high * 1.01 else 0

    return make_candidate(
        ticker, "REVERSAL_CONFIRM", score, bar, entry, float(stop), args.target_pct,
        higher_low, resistance,
        "A higher low, local-structure break, hourly EMA reclaim, and candle/volume confirmation formed after weakness.",
    )


def classify_ticker(
    ticker: str,
    raw_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    df = add_indicators(raw_df)
    bar = df.iloc[-1]

    uptrend = detect_uptrend(df)
    downtrend = detect_downtrend(df)
    is_range, range_support, range_resistance = range_levels(df)
    is_breakout, breakout_resistance = breakout_level(df, args)
    is_breakdown, breakdown_support = breakdown_level(df)
    is_reversal, higher_low, local_structure = reversal_structure(df, args)

    if is_breakdown:
        primary_state = "BREAKDOWN"
    elif is_reversal:
        primary_state = "REVERSAL_CONFIRM"
    elif is_breakout:
        primary_state = "BREAKOUT_RETEST"
    elif uptrend:
        primary_state = "UPTREND"
    elif downtrend:
        primary_state = "DOWNTREND"
    elif is_range:
        primary_state = "RANGE_SIDEWAYS"
    else:
        primary_state = "NO_CLEAR_SETUP"

    debug = {
        "Ticker": ticker,
        "PrimaryState": primary_state,
        "LastCompletedHourlyBar": str(bar.name),
        "Close": num(bar["Close"]),
        "EMA20": num(bar["EMA20"]),
        "EMA50": num(bar["EMA50"]),
        "RelVol20": num(bar["RelVol20"]),
        "AvgHourlyDollarVol20": num(bar["AvgDollarVol20"], 0),
        "Uptrend": uptrend,
        "Downtrend": downtrend,
        "Range": is_range,
        "Breakout": is_breakout,
        "Breakdown": is_breakdown,
        "Reversal": is_reversal,
        "RangeSupport": num(range_support),
        "RangeResistance": num(range_resistance),
        "BreakoutResistance": num(breakout_resistance),
        "BreakdownSupport": num(breakdown_support),
        "Reason": "",
    }

    # Safeguards apply to both fresh and confirmed profiles.
    if bar["Close"] < args.min_price:
        debug["Reason"] = f"Below minimum price (${args.min_price:.2f})"
        return None, debug
    if bar["AvgDollarVol20"] < args.min_hourly_dollar_volume:
        debug["Reason"] = "Insufficient average hourly dollar volume"
        return None, debug
    if bar["RelVol20"] < args.min_rel_volume:
        debug["Reason"] = "Current relative volume below threshold"
        return None, debug
    if is_breakdown:
        debug["Reason"] = "Breakdown state: no long entry"
        return None, debug
    if downtrend and not is_reversal:
        debug["Reason"] = "Downtrend state: no long entry"
        return None, debug

    choices = [
        try_fresh_breakout(df, ticker, args),
        try_early_recovery(df, ticker, args),
        try_breakout_retest(df, ticker, args),
        try_uptrend_pullback(df, ticker, args),
        try_uptrend_continuation(df, ticker, args),
        try_range_support_bounce(df, ticker, args),
        try_reversal_confirm(df, ticker, args),
    ]

    candidates: list[dict[str, Any]] = []
    for row in choices:
        if row is None:
            continue
        if row["Score"] < args.min_score:
            continue
        if row["Risk_pct"] > args.max_risk_pct:
            continue
        if row["RewardRisk_to_Target"] < args.min_reward_risk:
            continue
        candidates.append(row)

    if not candidates:
        debug["Reason"] = "No complete long-entry setup passed active filters"
        return None, debug

    # Fresh signals come first in the tie-break only when their score is equal;
    # this preserves a higher-quality confirmed setup when it is objectively stronger.
    selected = max(
        candidates,
        key=lambda row: (
            row["Score"],
            row["RewardRisk_to_Target"],
            row["RelVol20"],
            -int(row.get("SignalAgeBars", 99) if np.isfinite(row.get("SignalAgeBars", math.nan)) else 99),
        ),
    )
    debug["Reason"] = f"Selected {selected['Pattern']}"
    return selected, debug


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.interval != "60m":
        print("Warning: this strategy is designed around 60m hourly candles.")

    tickers = load_tickers(args.tickers_file)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Scanning {len(tickers)} symbols using completed {args.interval} candles | "
        f"target={args.target_pct:.2f}%"
    )
    print(
        f"Filters: score>={args.min_score}; risk<={args.max_risk_pct:.2f}%; "
        f"R/R>={args.min_reward_risk:.2f}; rel-vol>={args.min_rel_volume:.2f}; "
        f"avg hourly $vol>=${args.min_hourly_dollar_volume:,.0f}"
    )
    print(
        "Signals: "
        f"fresh-breakout={'ON' if args.allow_fresh_breakout else 'OFF'}; "
        f"early-recovery={'ON' if args.allow_early_recovery else 'OFF'}; "
        f"continuation={'ON' if args.allow_uptrend_continuation else 'OFF'}"
    )

    downloaded: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_hourly,
                ticker,
                args.period,
                args.interval,
                args.include_prepost,
                args.min_completed_bars,
                args.download_retries,
                args.retry_delay_seconds,
            ): ticker
            for ticker in tickers
        }
        for index, future in enumerate(as_completed(future_to_ticker), start=1):
            ticker, data, error = future.result()
            if data is None:
                errors.append({"Ticker": ticker, "Error": error or "Unknown download failure"})
            else:
                downloaded[ticker] = data
            print(f"[{index:>3}/{len(tickers)}] {ticker:<7} {'OK' if data is not None else 'ERROR'}")

    good_entries: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for ticker in tickers:
        raw = downloaded.get(ticker)
        if raw is None:
            continue
        try:
            row, debug = classify_ticker(ticker, raw, args)
            classifications.append(debug)
            if row is not None:
                good_entries.append(row)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": f"Classification error: {type(exc).__name__}: {exc}"})

    good_entries.sort(
        key=lambda row: (row["Score"], row["RewardRisk_to_Target"], row["RelVol20"]),
        reverse=True,
    )

    result_columns = [
        "Ticker", "Pattern", "SignalPhase", "SignalAgeBars", "Action", "Score", "LastCompletedHourlyBar",
        "Entry", "Stop", "Target_5pct", "Risk_pct", "RewardRisk_to_Target", "Close", "EMA20", "EMA50",
        "Distance_to_EMA20_pct", "Distance_to_EMA50_pct", "RelVol20", "AvgHourlyDollarVol20", "Support",
        "Resistance", "ResistanceDistance_pct", "ResistanceBefore5pctTarget", "Reason",
    ]
    output_file = args.outdir / "good_entries.csv"
    write_csv(good_entries, output_file, result_columns)
    if errors:
        write_csv(errors, args.outdir / "errors.csv", ["Ticker", "Error"])
    if args.debug:
        classifications.sort(key=lambda row: row["Ticker"])
        write_csv(classifications, args.outdir / "all_classifications.csv")

    print("\n=== GOOD LONG-ENTRY CANDIDATES ONLY ===")
    if not good_entries:
        print("No tickers passed the active filters.")
    else:
        display = pd.DataFrame(good_entries)[
            [
                "Ticker", "Pattern", "SignalPhase", "Score", "Entry", "Stop", "Target_5pct",
                "Risk_pct", "RewardRisk_to_Target", "RelVol20", "ResistanceDistance_pct",
                "ResistanceBefore5pctTarget",
            ]
        ]
        print(display.to_string(index=False))

    print(f"\nSaved: {output_file}")
    if args.debug:
        print(f"Debug: {args.outdir / 'all_classifications.csv'}")
    if errors:
        print(f"Errors: {args.outdir / 'errors.csv'}")


if __name__ == "__main__":
    main()
