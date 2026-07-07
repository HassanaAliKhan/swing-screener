#!/usr/bin/env python3
"""
Hourly Swing Screener — confirmed and fresh-momentum long setups.

The scanner keeps confirmed momentum setups and adds early-entry signals:
  - FRESH_BREAKOUT: the latest completed hourly candle has just broken a base.
  - EARLY_RECOVERY: a recent weak move is reclaiming the 20 EMA / short structure.
  - CRASH_RECOVERY: a heavily sold-off name has made its first volume-backed
    short-structure break; this intentionally does not require an EMA20 reclaim.

The crash-recovery profile looks for an early bounce after a meaningful recent
hourly drawdown. It is a review signal, not a prediction or an automatic order.
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

    # Deep-crash recovery. This is intentionally independent of EMA20/EMA50
    # placement so it can surface the first structure break after a major selloff.
    parser.add_argument("--allow-crash-recovery", action="store_true")
    parser.add_argument(
        "--crash-recovery-only",
        action="store_true",
        help="When set, only emit CRASH_RECOVERY candidates; do not let legacy momentum patterns compete.",
    )
    parser.add_argument("--crash-recovery-high-lookback-bars", type=int, default=300)
    parser.add_argument("--crash-recovery-min-drawdown-pct", type=float, default=20.0)
    parser.add_argument(
        "--crash-recovery-low-lookback-bars",
        type=int,
        default=80,
        help="Search this many completed hourly bars for the crash low used as the recovery reference.",
    )
    parser.add_argument(
        "--crash-recovery-signal-lookback-bars",
        type=int,
        default=8,
        help="Allow the initial recovery trigger to have occurred within this many completed hourly bars.",
    )
    parser.add_argument("--crash-recovery-min-recovery-from-low-pct", type=float, default=2.0)
    parser.add_argument("--crash-recovery-structure-bars", type=int, default=3)
    parser.add_argument("--crash-recovery-min-positive-bars", type=int, default=3)
    parser.add_argument("--crash-recovery-min-rel-volume", type=float, default=0.70)
    parser.add_argument("--crash-recovery-max-recovery-from-low-pct", type=float, default=35.0)
    parser.add_argument("--crash-recovery-min-practical-target-pct", type=float, default=0.50)
    parser.add_argument("--crash-recovery-target-buffer-pct", type=float, default=0.20)

    # High-conviction crash recovery: entry-only subset of the broad recovery watchlist.
    parser.add_argument("--crash-recovery-high-conviction", action="store_true")
    parser.add_argument("--crash-recovery-max-results", type=int, default=12)
    parser.add_argument("--crash-recovery-max-signal-age-bars", type=int, default=2)
    parser.add_argument("--crash-recovery-min-current-rel-volume", type=float, default=0.80)
    parser.add_argument("--crash-recovery-min-recent-rel-volume", type=float, default=1.10)
    parser.add_argument("--crash-recovery-max-risk-pct", type=float, default=4.0)
    parser.add_argument("--crash-recovery-min-target-pct", type=float, default=3.0)
    parser.add_argument(
        "--crash-recovery-min-upside-to-midpoint-pct",
        type=float,
        default=12.0,
        help="Minimum remaining upside to the midpoint between crash low and pre-crash high.",
    )
    parser.add_argument("--crash-recovery-min-reward-risk", type=float, default=1.25)
    parser.add_argument("--crash-recovery-max-extension-pct", type=float, default=2.5)

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
    # EMA8 is used only to recognize that a post-crash rebound is still active.
    # It is intentionally not a substitute for the slower EMA20/EMA50 trend filters.
    out["EMA8"] = out["Close"].ewm(span=8, adjust=False).mean()
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
    out["EMA8Slope3"] = out["EMA8"] - out["EMA8"].shift(3)
    out["EMA20Slope5"] = out["EMA20"] - out["EMA20"].shift(5)
    out["EMA50Slope8"] = out["EMA50"] - out["EMA50"].shift(8)
    return out.dropna().copy()


def pct(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return math.nan
    return (float(value) / float(reference) - 1.0) * 100.0


def num(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if np.isfinite(value) else math.nan


def safe_float_for_sort(value: Any, default: float = math.inf) -> float:
    """Return a deterministic numeric sort value for candidate diagnostics."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


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


def nearest_overhead_resistance(
    df: pd.DataFrame,
    entry: float,
    lookback: int = 40,
    min_distance_pct: float = 0.10,
) -> float:
    """Return the nearest meaningful recent high above entry, if one exists."""
    window = df.iloc[-(lookback + 1):-1]
    if window.empty:
        return math.nan

    threshold = entry * (1.0 + max(0.0, min_distance_pct) / 100.0)
    overhead = window.loc[window["High"] > threshold, "High"]
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
    practical_target: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one candidate row.

    ``Target_5pct`` remains the theoretical target implied by the UI. ``Target``
    can be lower when the pattern deliberately caps the trade at the first
    meaningful overhead resistance. Reward/risk uses the actual Target.
    """
    if not np.isfinite(stop) or stop >= entry:
        return None

    theoretical_target = entry * (1.0 + target_pct / 100.0)
    target = theoretical_target
    if practical_target is not None and np.isfinite(practical_target) and practical_target > entry:
        target = min(theoretical_target, float(practical_target))

    effective_target_pct = pct(target, entry)
    risk_pct = (entry - stop) / entry * 100.0
    reward_risk = effective_target_pct / risk_pct if risk_pct > 0 else math.nan
    resistance_distance = pct(resistance, entry) if np.isfinite(resistance) else math.nan

    result: dict[str, Any] = {
        "Ticker": ticker,
        "Pattern": pattern,
        "SignalPhase": signal_phase,
        "SignalAgeBars": signal_age_bars if signal_age_bars is not None else math.nan,
        "Action": "LONG_ENTRY_REVIEW",
        "Score": int(score),
        "LastCompletedHourlyBar": str(bar.name),
        "Entry": num(entry),
        "Stop": num(stop),
        "Target": num(target),
        "Target_pct": num(effective_target_pct),
        "Target_5pct": num(theoretical_target),
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
    if extra_fields:
        result.update(extra_fields)
    return result


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


def _recent_structure_triggers(
    df: pd.DataFrame,
    structure_bars: int,
    signal_lookback: int,
) -> list[int]:
    """Return hourly-bar positions that broke the immediately preceding local high.

    The old crash profile required the breakout to happen on the *single latest*
    completed hourly bar. That is too fragile for a watchlist: a stock can make
    the first break, consolidate for a few hours, and still be in the desired
    early recovery phase. This helper preserves the price-structure idea but
    allows that trigger to be recent rather than exactly one bar old.
    """
    triggers: list[int] = []
    start = max(structure_bars, len(df) - max(2, signal_lookback))
    for pos in range(start, len(df)):
        bar = df.iloc[pos]
        prior = df.iloc[pos - structure_bars:pos]
        if prior.empty:
            continue
        prior_high = float(prior["High"].max())
        candle_range = max(float(bar["High"] - bar["Low"]), 1e-9)
        close_location = float((bar["Close"] - bar["Low"]) / candle_range)
        bullish = bool(bar["Close"] > bar["Open"] and close_location >= 0.55)
        if bullish and float(bar["Close"]) > prior_high * 1.001:
            triggers.append(pos)
    return triggers


def try_crash_recovery(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Build a ranked deep-recovery watchlist rather than requiring a perfect entry candle.

    Earlier versions required the crash low to be extremely recent *and* the
    latest completed hour to break structure on elevated volume. That is valid
    for a narrow entry trigger, but it frequently produces a zero-row scan.

    This version keeps the central idea—still materially below the pre-crash
    high and demonstrably above the recent trough—but separates *recovery state*
    from *entry readiness*:
      - WATCH: early rebound; wait for structure confirmation.
      - BUILDING: constructive rebound or a recent prior trigger.
      - TRIGGERED: the latest completed hour breaks short structure.

    Every row remains a review candidate, not an automatic buy signal.
    """
    if not getattr(args, "allow_crash_recovery", False):
        return None

    high_lookback = max(80, int(getattr(args, "crash_recovery_high_lookback_bars", 300)))
    low_lookback = max(30, int(getattr(args, "crash_recovery_low_lookback_bars", 120)))
    signal_lookback = max(4, int(getattr(args, "crash_recovery_signal_lookback_bars", 12)))
    structure_bars = max(2, int(getattr(args, "crash_recovery_structure_bars", 3)))
    min_positive_bars = max(1, int(getattr(args, "crash_recovery_min_positive_bars", 2)))

    high_window = df.iloc[-(high_lookback + 1):-1]
    low_window = df.iloc[-(low_lookback + 1):-1]
    if len(high_window) < high_lookback or len(low_window) < low_lookback:
        return None

    bar = df.iloc[-1]
    entry = float(bar["Close"])

    # The high reference is strictly before the selected trough, so it cannot
    # use a rebound peak after the crash low.
    crash_low_bar = low_window["Low"].idxmin()
    crash_low = float(low_window.loc[crash_low_bar, "Low"])
    pre_crash_window = high_window.loc[high_window.index <= crash_low_bar]
    if pre_crash_window.empty:
        return None
    highest_before_crash_bar = pre_crash_window["High"].idxmax()
    crash_high = float(pre_crash_window.loc[highest_before_crash_bar, "High"])

    drawdown_from_high = pct(entry, crash_high)
    recovery_from_low = pct(entry, crash_low)
    min_drawdown = abs(float(getattr(args, "crash_recovery_min_drawdown_pct", 20.0)))
    min_recovery = max(0.0, float(getattr(args, "crash_recovery_min_recovery_from_low_pct", 1.0)))
    max_recovery = float(getattr(args, "crash_recovery_max_recovery_from_low_pct", 45.0))

    # Recovery-state diagnostics—not all are hard filters.
    trigger_positions = _recent_structure_triggers(
        df=df,
        structure_bars=structure_bars,
        signal_lookback=signal_lookback,
    )
    latest_trigger_pos = trigger_positions[-1] if trigger_positions else None
    trigger_freshness = (
        len(df) - 1 - latest_trigger_pos
        if latest_trigger_pos is not None
        else math.nan
    )
    current_trigger = bool(latest_trigger_pos == len(df) - 1)

    signal_window = df.iloc[-signal_lookback:]
    positive_bars = int((signal_window["Close"] > signal_window["Open"]).sum())
    recent_return = pct(entry, float(signal_window.iloc[0]["Close"]))
    recent_max_relvol = float(signal_window["RelVol20"].max())
    ema8_rising = bool(float(bar["EMA8"]) >= float(df.iloc[-4]["EMA8"]))
    above_ema8 = bool(entry >= float(bar["EMA8"]) * 0.995)
    building = bool(
        positive_bars >= min_positive_bars
        and (above_ema8 or ema8_rising or recent_return >= 0.50)
    )

    # The only hard signal requirement is that the stock is actually off its
    # selected trough. WATCH rows may be early, but do not represent a falling
    # name that has not bounced at all.
    early_rebound = bool(recovery_from_low >= min_recovery)
    if not (
        drawdown_from_high <= -min_drawdown
        and early_rebound
        and recovery_from_low <= max_recovery
    ):
        return None

    if current_trigger:
        recovery_status = "TRIGGERED"
        action = "ENTRY_REVIEW"
        entry_readiness = "Latest completed hour broke the prior short-term high."
    elif latest_trigger_pos is not None or building:
        recovery_status = "BUILDING"
        action = "WAIT_FOR_TRIGGER"
        entry_readiness = "Recovery is constructive; wait for a fresh short-term high break."
    else:
        recovery_status = "WATCH"
        action = "WATCH_ONLY"
        entry_readiness = "Early rebound only; no usable short-term trigger yet."

    # Use nearby recovery support for a manageable reference stop, not the
    # original crash low. This is displayed even for WATCH rows and is not a
    # claim that the position should be opened now.
    short_support_window = df.iloc[-(signal_lookback + 1):-1]
    short_support = float(short_support_window["Low"].min())
    stop = short_support - 0.20 * float(bar["ATR14"])

    resistance = nearest_overhead_resistance(
        df,
        entry,
        lookback=min(high_lookback, 120),
        min_distance_pct=0.25,
    )
    theoretical_target = entry * (1.0 + args.target_pct / 100.0)
    practical_target = theoretical_target
    if np.isfinite(resistance):
        practical_target = min(
            theoretical_target,
            resistance
            * (
                1.0
                - float(getattr(args, "crash_recovery_target_buffer_pct", 0.20))
                / 100.0
            ),
        )

    score = 40
    score += 15 if recovery_status == "TRIGGERED" else 8 if recovery_status == "BUILDING" else 0
    score += 10 if drawdown_from_high <= -30.0 else 0
    score += 7 if drawdown_from_high <= -45.0 else 0
    score += 6 if recovery_from_low <= 15.0 else 0
    score += 5 if recent_max_relvol >= 1.00 else 0
    score += 3 if entry > float(bar["EMA20"]) else 0

    result = make_candidate(
        ticker=ticker,
        pattern="CRASH_RECOVERY",
        score=score,
        bar=bar,
        entry=entry,
        stop=float(stop),
        target_pct=args.target_pct,
        support=short_support,
        resistance=resistance,
        reason=(
            "Stock remains materially below its pre-crash high and has bounced "
            "from a recent trough. RecoveryStatus distinguishes an early watch "
            "from a building recovery and a current-hour entry trigger."
        ),
        signal_phase=f"Crash recovery — {recovery_status.title()}",
        signal_age_bars=(
            int(trigger_freshness) if np.isfinite(trigger_freshness) else math.nan
        ),
        practical_target=float(practical_target),
        extra_fields={
            "Action": action,
            "RecoveryStatus": recovery_status,
            "EntryReadiness": entry_readiness,
            "HighestBeforeCrash": num(crash_high),
            "HighestBeforeCrashBar": str(highest_before_crash_bar),
            "HighReferenceLookbackBars": int(high_lookback),
            "CrashLow": num(crash_low),
            "CrashLowBar": str(crash_low_bar),
            "CrashLowLookbackBars": int(low_lookback),
            "DrawdownFromCrashHigh_pct": num(drawdown_from_high),
            "RecoveryFromCrashLow_pct": num(recovery_from_low),
            "RecoveryStage": recovery_status,
            "RecoveryTriggerFreshnessBars": (
                int(trigger_freshness) if np.isfinite(trigger_freshness) else math.nan
            ),
            "RecentPositiveBars": positive_bars,
            "RecentMaxRelVol20": num(recent_max_relvol),
            "VolumeConfirmed": bool(recent_max_relvol >= 1.00),
            "ShortTermSupport": num(short_support),
            "ShortStructureHigh": num(
                float(df.iloc[-(structure_bars + 1):-1]["High"].max())
            ),
            "EMA20Required": False,
            "CrashRecoveryOnly": bool(
                getattr(args, "crash_recovery_only", False)
            ),
        },
    )
    return result


def meaningful_overhead_pivot_resistance(
    df: pd.DataFrame,
    entry: float,
    lookback: int = 120,
    min_distance_pct: float = 3.0,
    pivot_radius: int = 2,
) -> float:
    """Return the nearest *structural* pivot high above ``entry``.

    The previous crash-recovery view used the nearest raw hourly high, which
    frequently found a trivial wick 0.2% above price and reduced the displayed
    target to almost zero. This helper ignores those micro-highs and looks only
    for local pivot highs that are at least ``min_distance_pct`` overhead.
    """
    window = df.iloc[-(max(12, lookback) + 1):-1]
    if len(window) < (pivot_radius * 2 + 1):
        return math.nan

    threshold = entry * (1.0 + max(0.0, min_distance_pct) / 100.0)
    highs = window["High"].to_numpy(dtype=float)
    pivots: list[float] = []

    for pos in range(pivot_radius, len(highs) - pivot_radius):
        high = float(highs[pos])
        if not np.isfinite(high) or high < threshold:
            continue
        left = highs[pos - pivot_radius:pos]
        right = highs[pos + 1:pos + 1 + pivot_radius]
        if high >= float(np.nanmax(left)) and high >= float(np.nanmax(right)):
            pivots.append(high)

    return float(min(pivots)) if pivots else math.nan


def try_high_conviction_crash_recovery(
    df: pd.DataFrame,
    ticker: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Return a compact, ranked recovery shortlist instead of a brittle exact trigger.

    This profile is designed to find deeply sold-off names that are recovering
    constructively *before* they become mature momentum names.  It intentionally
    uses a few structural safety gates, then ranks the survivors.  Earlier
    versions required every quality feature to be perfect at the same time
    (fresh 2-hour breakout, rising EMA20, high current volume, 3% target after
    resistance, 1.40 reward/risk), which made valid recovery candidates vanish.

    A returned row is a research candidate, not a guarantee or an automatic
    order.  The Action column tells whether the recovery already has a recent
    breakout or still needs an hourly confirmation.
    """
    if not getattr(args, "allow_crash_recovery", False):
        return None

    high_lookback = max(80, int(getattr(args, "crash_recovery_high_lookback_bars", 300)))
    low_lookback = max(30, int(getattr(args, "crash_recovery_low_lookback_bars", 120)))
    signal_lookback = max(8, int(getattr(args, "crash_recovery_signal_lookback_bars", 12)))
    structure_bars = max(3, int(getattr(args, "crash_recovery_structure_bars", 3)))
    max_signal_age = max(3, int(getattr(args, "crash_recovery_max_signal_age_bars", 10)))

    high_window = df.iloc[-(high_lookback + 1):-1]
    low_window = df.iloc[-(low_lookback + 1):-1]
    if len(high_window) < high_lookback or len(low_window) < low_lookback:
        return None

    bar = df.iloc[-1]
    entry = float(bar["Close"])
    crash_low_bar = low_window["Low"].idxmin()
    crash_low = float(low_window.loc[crash_low_bar, "Low"])
    pre_crash_window = high_window.loc[high_window.index <= crash_low_bar]
    if pre_crash_window.empty:
        return None

    highest_before_crash_bar = pre_crash_window["High"].idxmax()
    crash_high = float(pre_crash_window.loc[highest_before_crash_bar, "High"])
    drawdown_from_high = pct(entry, crash_high)
    recovery_from_low = pct(entry, crash_low)

    min_drawdown = abs(float(getattr(args, "crash_recovery_min_drawdown_pct", 20.0)))
    min_recovery = max(2.0, float(getattr(args, "crash_recovery_min_recovery_from_low_pct", 3.0)))
    max_recovery = float(getattr(args, "crash_recovery_max_recovery_from_low_pct", 35.0))
    if not (
        drawdown_from_high <= -min_drawdown
        and min_recovery <= recovery_from_low <= max_recovery
    ):
        return None

    # Keep the shortlist tradeable. This is intentionally higher than the broad
    # discovery profile's minimum but lower than the old $5M hard gate when the
    # user's watchlist contains smaller, actively traded recovery names.
    min_dollar_volume = float(getattr(args, "min_hourly_dollar_volume", 3_000_000))
    if float(bar["AvgDollarVol20"]) < min_dollar_volume:
        return None

    signal_window = df.iloc[-signal_lookback:]
    positive_bars = int((signal_window["Close"] > signal_window["Open"]).sum())
    recent_max_relvol = float(signal_window["RelVol20"].max())
    current_relvol = float(bar["RelVol20"])
    ema20_slope = float(bar["EMA20Slope5"])
    ema8_slope = float(bar["EMA8Slope3"])
    distance_to_ema20 = pct(entry, float(bar["EMA20"]))

    # The price may be a little below EMA20 in a genuine early recovery. The
    # old requirement to be above a *rising* EMA20 rejected most of the good
    # early candidates before the indicator had time to turn.
    if distance_to_ema20 < -3.0 or distance_to_ema20 > 5.0:
        return None
    if positive_bars < max(3, int(getattr(args, "crash_recovery_min_positive_bars", 3))):
        return None
    if recent_max_relvol < float(getattr(args, "crash_recovery_min_recent_rel_volume", 0.80)):
        return None
    if ema8_slope < 0.0 and ema20_slope < -0.35 * float(bar["ATR14"]):
        return None

    trigger_positions = _recent_structure_triggers(
        df=df,
        structure_bars=structure_bars,
        signal_lookback=max_signal_age + 1,
    )
    latest_trigger_pos = trigger_positions[-1] if trigger_positions else None
    signal_age = (
        len(df) - 1 - latest_trigger_pos
        if latest_trigger_pos is not None
        else math.nan
    )
    trigger_bar = df.iloc[latest_trigger_pos] if latest_trigger_pos is not None else None
    trigger_relvol = float(trigger_bar["RelVol20"]) if trigger_bar is not None else math.nan

    # A recent breakout is a quality advantage, not a mandatory one. For names
    # without one, require that the recovery is at least holding near EMA20;
    # these are surfaced as confirmation candidates rather than immediate buys.
    has_recent_trigger = bool(np.isfinite(signal_age) and signal_age <= max_signal_age)
    if not has_recent_trigger and distance_to_ema20 < -1.5:
        return None

    if trigger_bar is not None:
        extension_from_trigger_high = pct(entry, float(trigger_bar["High"]))
        max_extension = float(getattr(args, "crash_recovery_max_extension_pct", 4.0))
        if entry < float(trigger_bar["Close"]) * 0.97 or extension_from_trigger_high > max_extension:
            return None
    else:
        extension_from_trigger_high = math.nan

    # A support-based stop makes the downside test visible. It is deliberately
    # not placed at the original crash low, which would make nearly every setup
    # look untradeable.
    short_support_window = df.iloc[-max(6, structure_bars + 3):-1]
    if short_support_window.empty:
        return None
    short_support = float(short_support_window["Low"].min())
    stop = short_support - 0.10 * float(bar["ATR14"])
    risk_pct = (entry - stop) / entry * 100.0
    max_risk = float(getattr(args, "crash_recovery_max_risk_pct", 6.0))
    if not np.isfinite(risk_pct) or risk_pct <= 0.0 or risk_pct > max_risk:
        return None

    crash_midpoint = crash_low + (crash_high - crash_low) * 0.50
    upside_to_midpoint = pct(crash_midpoint, entry)
    min_upside_to_midpoint = float(
        getattr(args, "crash_recovery_min_upside_to_midpoint_pct", 8.0)
    )
    if not np.isfinite(upside_to_midpoint) or upside_to_midpoint < min_upside_to_midpoint:
        return None

    # Use structural pivot resistance, not the nearest hourly wick. The old
    # code treated a 0.2% intrabar wick as resistance and reduced almost every
    # target to near zero. A pivot at least 2% above entry is more meaningful.
    desired_target_pct = min(max(3.0, float(args.target_pct)), 5.0)
    resistance = meaningful_overhead_pivot_resistance(
        df,
        entry,
        lookback=min(high_lookback, 120),
        min_distance_pct=2.0,
        pivot_radius=2,
    )
    theoretical_target = entry * (1.0 + desired_target_pct / 100.0)
    practical_target = theoretical_target
    if np.isfinite(resistance):
        capped_resistance = resistance * (
            1.0 - float(getattr(args, "crash_recovery_target_buffer_pct", 0.10)) / 100.0
        )
        if capped_resistance > entry:
            practical_target = min(theoretical_target, capped_resistance)

    actual_target_pct = pct(practical_target, entry)
    reward_risk = actual_target_pct / risk_pct if risk_pct > 0 else math.nan
    min_target_pct = float(getattr(args, "crash_recovery_min_target_pct", 2.0))
    min_reward_risk = float(getattr(args, "crash_recovery_min_reward_risk", 0.80))
    if (
        not np.isfinite(actual_target_pct)
        or actual_target_pct < min_target_pct
        or not np.isfinite(reward_risk)
        or reward_risk < min_reward_risk
    ):
        return None

    # Rank rather than over-filter. The score favors room to run, volume and
    # fresh structure; it penalizes wider stops and overly extended rebounds.
    score = 45
    score += 9 if has_recent_trigger and signal_age <= 1 else 6 if has_recent_trigger and signal_age <= 3 else 3 if has_recent_trigger else 0
    score += 8 if trigger_bar is not None and trigger_relvol >= 1.15 else 4 if trigger_bar is not None and trigger_relvol >= 0.70 else 0
    score += 8 if recent_max_relvol >= 1.50 else 5 if recent_max_relvol >= 1.00 else 2
    score += 7 if distance_to_ema20 >= 0.0 else 3
    score += 6 if ema20_slope >= 0.0 else 3 if ema8_slope >= 0.0 else 0
    score += 6 if positive_bars >= 5 else 3
    score += 11 if risk_pct <= 2.5 else 7 if risk_pct <= 4.0 else 3
    score += 7 if reward_risk >= 1.50 else 4 if reward_risk >= 1.00 else 1
    score += 6 if actual_target_pct >= 4.0 else 3
    score += 5 if drawdown_from_high <= -35.0 else 2
    score += 4 if recovery_from_low <= 20.0 else 1
    score += 5 if upside_to_midpoint >= 20.0 else 2
    score += 3 if np.isfinite(resistance) and pct(resistance, entry) >= 3.0 else 0
    score += 2 if current_relvol >= 0.60 else 0

    min_quality_score = int(getattr(args, "min_score", 62))
    if score < min_quality_score:
        return None

    if has_recent_trigger and signal_age <= 3:
        action = "ENTRY_REVIEW"
        readiness = "Recent local breakout is holding; verify current price and news before entry."
        phase = "Ranked recovery — breakout held"
    else:
        action = "CONFIRM_ON_BREAK"
        readiness = "Recovery is constructive but needs a fresh hourly break above short-term structure."
        phase = "Ranked recovery — confirmation pending"

    quality_tier = "A" if score >= 86 and action == "ENTRY_REVIEW" else "B" if score >= 76 else "C"

    return make_candidate(
        ticker=ticker,
        pattern="RANKED_CRASH_RECOVERY",
        score=score,
        bar=bar,
        entry=entry,
        stop=float(stop),
        target_pct=desired_target_pct,
        support=short_support,
        resistance=resistance,
        reason=(
            "Deeply sold-off recovery ranked for liquidity, constructive price action, "
            "volume participation, controlled support-based risk, and remaining room to the crash midpoint."
        ),
        signal_phase=phase,
        signal_age_bars=(int(signal_age) if np.isfinite(signal_age) else math.nan),
        practical_target=float(practical_target),
        extra_fields={
            "Action": action,
            "RecoveryStatus": "RANKED_SHORTLIST",
            "RecoveryQualityTier": quality_tier,
            "EntryReadiness": readiness,
            "HighestBeforeCrash": num(crash_high),
            "HighestBeforeCrashBar": str(highest_before_crash_bar),
            "HighReferenceLookbackBars": int(high_lookback),
            "CrashLow": num(crash_low),
            "CrashLowBar": str(crash_low_bar),
            "CrashLowLookbackBars": int(low_lookback),
            "DrawdownFromCrashHigh_pct": num(drawdown_from_high),
            "RecoveryFromCrashLow_pct": num(recovery_from_low),
            "RecoveryStage": "RANKED_SHORTLIST",
            "RecoveryTriggerFreshnessBars": int(signal_age) if np.isfinite(signal_age) else math.nan,
            "RecentPositiveBars": positive_bars,
            "RecentMaxRelVol20": num(recent_max_relvol),
            "TriggerRelVol20": num(trigger_relvol),
            "CurrentRelVol20": num(current_relvol),
            "VolumeConfirmed": bool(recent_max_relvol >= 0.80),
            "ShortTermSupport": num(short_support),
            "ShortStructureHigh": num(float(df.iloc[-(structure_bars + 1):-1]["High"].max())),
            "MeaningfulResistance": num(resistance),
            "MeaningfulResistanceDistance_pct": num(pct(resistance, entry)),
            "UpsideToCrashMidpoint_pct": num(upside_to_midpoint),
            "EMA20Slope5": num(ema20_slope),
            "EMA8Slope3": num(ema8_slope),
            "DistanceToEMA20_pct": num(distance_to_ema20),
            "ExtensionFromTriggerHigh_pct": num(extension_from_trigger_high),
            "QualityScore": int(score),
            "EMA20Required": False,
            "CrashRecoveryOnly": bool(getattr(args, "crash_recovery_only", False)),
            "HighConvictionRecovery": True,
        },
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

    # Evaluate the crash-recovery candidate before downtrend/breakdown exclusion.
    # A genuine early recovery is often still below EMA20/EMA50 and would otherwise
    # be discarded by the confirmation-first state classifier.
    crash_recovery_candidate = (
        try_high_conviction_crash_recovery(df, ticker, args)
        if getattr(args, "crash_recovery_high_conviction", False)
        else try_crash_recovery(df, ticker, args)
    )
    crash_recovery = crash_recovery_candidate is not None

    if crash_recovery:
        primary_state = "CRASH_RECOVERY"
    elif is_breakdown:
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
        "CrashRecovery": crash_recovery,
        "CrashRecoveryOnly": bool(getattr(args, "crash_recovery_only", False)),
        "RangeSupport": num(range_support),
        "RangeResistance": num(range_resistance),
        "BreakoutResistance": num(breakout_resistance),
        "BreakdownSupport": num(breakdown_support),
        "Reason": "",
    }

    # Safeguards apply to every profile.
    if bar["Close"] < args.min_price:
        debug["Reason"] = f"Below minimum price (${args.min_price:.2f})"
        return None, debug
    if bar["AvgDollarVol20"] < args.min_hourly_dollar_volume:
        debug["Reason"] = "Insufficient average hourly dollar volume"
        return None, debug
    # Crash recovery evaluates the strongest volume confirmation in the recent
    # recovery window. Requiring the exact current bar to clear the global
    # relative-volume gate was another source of false zero-result scans.
    if bar["RelVol20"] < args.min_rel_volume and not getattr(args, "crash_recovery_only", False):
        debug["Reason"] = "Current relative volume below threshold"
        return None, debug
    if is_breakdown and not crash_recovery:
        debug["Reason"] = "Breakdown state: no long entry"
        return None, debug
    if downtrend and not is_reversal and not crash_recovery:
        debug["Reason"] = "Downtrend state: no long entry"
        return None, debug

    if getattr(args, "crash_recovery_only", False):
        # This is critical: the crash-recovery profile must never return an ATH/
        # momentum continuation just because it received a higher legacy score.
        choices = [crash_recovery_candidate]
    else:
        choices = [
            crash_recovery_candidate,
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
        if getattr(args, "crash_recovery_only", False):
            # This is a ranked recovery watchlist. Risk is displayed for review,
            # but it is not a rejection gate because WATCH/BUILDING rows are not
            # instructions to enter immediately.
            candidates.append(row)
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

    selected = max(
        candidates,
        key=lambda row: (
            row["Score"],
            row["RewardRisk_to_Target"],
            row["RelVol20"],
            -int(
                row.get("SignalAgeBars", 99)
                if np.isfinite(row.get("SignalAgeBars", math.nan))
                else 99
            ),
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
        f"crash-recovery={'ON' if args.allow_crash_recovery else 'OFF'}; "
        f"crash-only={'ON' if getattr(args, 'crash_recovery_only', False) else 'OFF'}; "
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

    if getattr(args, "crash_recovery_high_conviction", False):
        good_entries.sort(
            key=lambda row: (
                safe_float_for_sort(row.get("QualityScore", row.get("Score", 0))),
                safe_float_for_sort(row.get("RewardRisk_to_Target")),
                safe_float_for_sort(row.get("Target_pct")),
                -safe_float_for_sort(row.get("Risk_pct")),
                safe_float_for_sort(row.get("RecentMaxRelVol20", row.get("RelVol20", 0))),
                safe_float_for_sort(row.get("UpsideToCrashMidpoint_pct")),
            ),
            reverse=True,
        )
        good_entries = good_entries[:max(1, int(getattr(args, "crash_recovery_max_results", 5)))]
    elif getattr(args, "crash_recovery_only", False):
        _status_rank = {"TRIGGERED": 0, "BUILDING": 1, "WATCH": 2}
        good_entries.sort(
            key=lambda row: (
                _status_rank.get(str(row.get("RecoveryStatus", "WATCH")), 3),
                safe_float_for_sort(row.get("DrawdownFromCrashHigh_pct")),
                safe_float_for_sort(row.get("RecoveryFromCrashLow_pct")),
                -safe_float_for_sort(row.get("RecentMaxRelVol20")),
            )
        )
    else:
        good_entries.sort(
            key=lambda row: (row["Score"], row["RewardRisk_to_Target"], row["RelVol20"]),
            reverse=True,
        )

    result_columns = [
        "Ticker", "Pattern", "SignalPhase", "SignalAgeBars", "Action", "Score", "LastCompletedHourlyBar",
        "Entry", "Stop", "Target", "Target_pct", "Target_5pct", "Risk_pct", "RewardRisk_to_Target",
        "Close", "EMA20", "EMA50", "Distance_to_EMA20_pct", "Distance_to_EMA50_pct", "RelVol20",
        "AvgHourlyDollarVol20", "Support", "Resistance", "ResistanceDistance_pct",
        "ResistanceBefore5pctTarget", "CrashHigh", "DrawdownFromCrashHigh_pct", "RecentLow",
        "RecoveryFromRecentLow_pct", "ShortStructureHigh", "EMA20Required", "Reason",
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
                "Ticker", "Pattern", "SignalPhase", "Score", "Entry", "Stop", "Target", "Target_pct",
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
