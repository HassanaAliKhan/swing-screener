from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from typing import Any

import pandas as pd
import streamlit as st

import swing_screener_6_patterns as scanner


st.set_page_config(
    page_title="Hourly Swing Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CRASH_RECOVERY_DEFAULTS: dict[str, Any] = {
    "allow_crash_recovery": False,
    "crash_recovery_only": False,
    "crash_recovery_high_lookback_bars": 300,
    "crash_recovery_low_lookback_bars": 120,
    "crash_recovery_signal_lookback_bars": 12,
    "crash_recovery_min_drawdown_pct": 20.0,
    "crash_recovery_min_recovery_from_low_pct": 1.0,
    "crash_recovery_structure_bars": 3,
    "crash_recovery_min_positive_bars": 2,
    "crash_recovery_min_rel_volume": 0.00,
    "crash_recovery_max_recovery_from_low_pct": 45.0,
    "crash_recovery_min_practical_target_pct": 0.00,
    "crash_recovery_target_buffer_pct": 0.20,
    "crash_recovery_high_conviction": False,
    "crash_recovery_max_results": 8,
    "crash_recovery_max_signal_age_bars": 10,
    "crash_recovery_min_current_rel_volume": 0.55,
    "crash_recovery_min_recent_rel_volume": 0.80,
    "crash_recovery_max_risk_pct": 6.00,
    "crash_recovery_min_target_pct": 2.00,
    "crash_recovery_min_upside_to_midpoint_pct": 8.00,
    "crash_recovery_min_reward_risk": 0.80,
    "crash_recovery_max_extension_pct": 4.00,
}


PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "High-conviction recovery": {
        # Ranked shortlist: a limited list of deep recoveries. Recent breakout,
        # volume, trend and risk are ranked rather than all being hard gates.
        "min_score": 62,
        "max_risk_pct": 6.0,
        "min_reward_risk": 0.80,
        "min_rel_volume": 0.00,
        "min_hourly_dollar_volume": 3_000_000,
        "min_price": 3.00,
        "pullback_touch_pct": 2.0,
        "pullback_max_ema20_distance_pct": 2.0,
        "pullback_volume_multiplier": 1.20,
        "breakout_event_volume_multiplier": 1.10,
        "breakout_retest_tolerance_pct": 2.0,
        "breakout_max_extension_pct": 2.0,
        "range_support_distance_pct": 2.0,
        "reversal_higher_low_pct": 0.25,
        "reversal_structure_break_pct": 0.10,
        "reversal_min_rel_volume": 0.90,
        "allow_resistance_before_target": True,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": True,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": False,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": False,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
        "allow_crash_recovery": True,
        "crash_recovery_only": True,
        "crash_recovery_high_conviction": True,
        "crash_recovery_high_lookback_bars": 300,
        "crash_recovery_low_lookback_bars": 120,
        "crash_recovery_signal_lookback_bars": 12,
        "crash_recovery_min_drawdown_pct": 20.0,
        "crash_recovery_min_recovery_from_low_pct": 3.0,
        "crash_recovery_structure_bars": 3,
        "crash_recovery_min_positive_bars": 3,
        "crash_recovery_min_rel_volume": 0.00,
        "crash_recovery_max_recovery_from_low_pct": 35.0,
        "crash_recovery_min_practical_target_pct": 2.0,
        "crash_recovery_target_buffer_pct": 0.20,
        "crash_recovery_max_results": 8,
        "crash_recovery_max_signal_age_bars": 10,
        "crash_recovery_min_current_rel_volume": 0.55,
        "crash_recovery_min_recent_rel_volume": 0.80,
        "crash_recovery_max_risk_pct": 6.0,
        "crash_recovery_min_target_pct": 2.0,
        "crash_recovery_min_upside_to_midpoint_pct": 8.0,
        "crash_recovery_min_reward_risk": 0.80,
        "crash_recovery_max_extension_pct": 4.00,
    },
    "Crash recovery": {
        # Exact-only mode: legacy ATH/momentum patterns are deliberately disabled.
        # This is a ranked recovery watchlist. It can return a fresh structure
        # trigger or a still-building recovery whose trigger occurred recently.
        "min_score": 40,
        "max_risk_pct": 25.0,
        "min_reward_risk": 0.00,
        "min_rel_volume": 0.30,
        "min_hourly_dollar_volume": 150_000,
        "min_price": 3.00,
        "pullback_touch_pct": 2.0,
        "pullback_max_ema20_distance_pct": 2.0,
        "pullback_volume_multiplier": 1.20,
        "breakout_event_volume_multiplier": 1.10,
        "breakout_retest_tolerance_pct": 2.0,
        "breakout_max_extension_pct": 2.0,
        "range_support_distance_pct": 2.0,
        "reversal_higher_low_pct": 0.25,
        "reversal_structure_break_pct": 0.10,
        "reversal_min_rel_volume": 0.90,
        "allow_resistance_before_target": True,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": True,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": False,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": False,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
        "allow_crash_recovery": True,
        "crash_recovery_only": True,
        "crash_recovery_high_lookback_bars": 300,
        "crash_recovery_low_lookback_bars": 120,
        "crash_recovery_signal_lookback_bars": 12,
        "crash_recovery_min_drawdown_pct": 20.0,
        "crash_recovery_min_recovery_from_low_pct": 1.0,
        "crash_recovery_structure_bars": 3,
        "crash_recovery_min_positive_bars": 2,
        "crash_recovery_min_rel_volume": 0.00,
        "crash_recovery_max_recovery_from_low_pct": 45.0,
        "crash_recovery_min_practical_target_pct": 0.00,
        "crash_recovery_target_buffer_pct": 0.20,
    },
    "Fresh momentum": {
        "min_score": 60,
        "max_risk_pct": 3.75,
        "min_reward_risk": 1.25,
        "min_rel_volume": 0.80,
        "min_hourly_dollar_volume": 250_000,
        "min_price": 3.00,
        "pullback_touch_pct": 2.0,
        "pullback_max_ema20_distance_pct": 2.0,
        "pullback_volume_multiplier": 1.20,
        "breakout_event_volume_multiplier": 1.10,
        "breakout_retest_tolerance_pct": 2.0,
        "breakout_max_extension_pct": 2.0,
        "range_support_distance_pct": 2.0,
        "reversal_higher_low_pct": 0.25,
        "reversal_structure_break_pct": 0.10,
        "reversal_min_rel_volume": 0.90,
        "allow_resistance_before_target": False,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": False,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": True,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": True,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
    },
    "Fresh momentum — broader": {
        "min_score": 55,
        "max_risk_pct": 5.0,
        "min_reward_risk": 1.00,
        "min_rel_volume": 0.60,
        "min_hourly_dollar_volume": 150_000,
        "min_price": 3.00,
        "pullback_touch_pct": 2.5,
        "pullback_max_ema20_distance_pct": 3.0,
        "pullback_volume_multiplier": 1.30,
        "breakout_event_volume_multiplier": 0.90,
        "breakout_retest_tolerance_pct": 2.5,
        "breakout_max_extension_pct": 3.0,
        "range_support_distance_pct": 3.0,
        "reversal_higher_low_pct": 0.10,
        "reversal_structure_break_pct": 0.0,
        "reversal_min_rel_volume": 0.75,
        "allow_resistance_before_target": True,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": True,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": True,
        "fresh_breakout_lookback_bars": 12,
        "fresh_breakout_min_rel_volume": 0.80,
        "fresh_breakout_max_extension_pct": 2.00,
        "allow_early_recovery": True,
        "early_recovery_lookback_bars": 5,
        "early_recovery_min_rel_volume": 0.75,
        "early_recovery_max_ema20_distance_pct": 2.50,
        "early_recovery_structure_break_pct": 0.0,
    },
    "Balanced": {
        "min_score": 58,
        "max_risk_pct": 4.5,
        "min_reward_risk": 1.10,
        "min_rel_volume": 0.45,
        "min_hourly_dollar_volume": 100_000,
        "min_price": 1.00,
        "pullback_touch_pct": 3.0,
        "pullback_max_ema20_distance_pct": 5.0,
        "pullback_volume_multiplier": 1.30,
        "breakout_event_volume_multiplier": 0.90,
        "breakout_retest_tolerance_pct": 3.0,
        "breakout_max_extension_pct": 4.0,
        "range_support_distance_pct": 4.0,
        "reversal_higher_low_pct": 0.25,
        "reversal_structure_break_pct": 0.0,
        "reversal_min_rel_volume": 0.60,
        "allow_resistance_before_target": True,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": False,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": False,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": False,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
    },
    "Relaxed review": {
        "min_score": 52,
        "max_risk_pct": 5.0,
        "min_reward_risk": 1.00,
        "min_rel_volume": 0.30,
        "min_hourly_dollar_volume": 75_000,
        "min_price": 1.00,
        "pullback_touch_pct": 4.0,
        "pullback_max_ema20_distance_pct": 7.0,
        "pullback_volume_multiplier": 1.50,
        "breakout_event_volume_multiplier": 0.70,
        "breakout_retest_tolerance_pct": 4.0,
        "breakout_max_extension_pct": 6.0,
        "range_support_distance_pct": 5.0,
        "reversal_higher_low_pct": 0.10,
        "reversal_structure_break_pct": 0.0,
        "reversal_min_rel_volume": 0.45,
        "allow_resistance_before_target": True,
        "allow_neutral_candle": True,
        "allow_reversal_below_ema50": True,
        "allow_uptrend_continuation": True,
        "allow_fresh_breakout": False,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": False,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
    },
    "Strict": {
        "min_score": 72,
        "max_risk_pct": 3.25,
        "min_reward_risk": 1.60,
        "min_rel_volume": 0.85,
        "min_hourly_dollar_volume": 500_000,
        "min_price": 3.00,
        "pullback_touch_pct": 1.20,
        "pullback_max_ema20_distance_pct": 3.0,
        "pullback_volume_multiplier": 1.10,
        "breakout_event_volume_multiplier": 1.25,
        "breakout_retest_tolerance_pct": 1.50,
        "breakout_max_extension_pct": 2.50,
        "range_support_distance_pct": 2.0,
        "reversal_higher_low_pct": 0.60,
        "reversal_structure_break_pct": 0.10,
        "reversal_min_rel_volume": 1.00,
        "allow_resistance_before_target": False,
        "allow_neutral_candle": False,
        "allow_reversal_below_ema50": False,
        "allow_uptrend_continuation": False,
        "allow_fresh_breakout": False,
        "fresh_breakout_lookback_bars": 20,
        "fresh_breakout_min_rel_volume": 1.00,
        "fresh_breakout_max_extension_pct": 1.50,
        "allow_early_recovery": False,
        "early_recovery_lookback_bars": 6,
        "early_recovery_min_rel_volume": 0.90,
        "early_recovery_max_ema20_distance_pct": 1.75,
        "early_recovery_structure_break_pct": 0.10,
    },
}


def default_watchlist() -> str:
    path = scanner.DEFAULT_WATCHLIST
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_watchlist(raw_text: str) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    for line in raw_text.splitlines():
        cleaned = line.split("#", 1)[0].replace(",", " ").strip()
        for token in cleaned.split():
            ticker = token.upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)

    return tickers


def build_args(
    profile: str,
    target_pct: float,
    show_prepost: bool,
    workers: int,
    custom_rel_volume: float,
    custom_min_score: int,
) -> SimpleNamespace:
    settings = dict(CRASH_RECOVERY_DEFAULTS)
    settings.update(PROFILE_SETTINGS[profile])
    settings["target_pct"] = float(target_pct)
    settings["min_rel_volume"] = float(custom_rel_volume)
    settings["min_score"] = int(custom_min_score)
    settings.update(
        {
            "period": "60d",
            "interval": "60m",
            "include_prepost": bool(show_prepost),
            "min_completed_bars": 60,
            "download_retries": 3,
            "retry_delay_seconds": 1.0,
            "max_workers": int(workers),
            "resistance_target_buffer_pct": 0.50,
        }
    )
    return SimpleNamespace(**settings)


def scan_watchlist(
    tickers: list[str],
    args: SimpleNamespace,
    progress_callback,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_by_ticker: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []
    classifications: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    completed = 0
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.max_workers)
    ) as executor:
        jobs = {
            executor.submit(
                scanner.fetch_hourly,
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

        for future in concurrent.futures.as_completed(jobs):
            ticker, frame, error = future.result()
            completed += 1

            if frame is None:
                errors.append({"Ticker": ticker, "Error": error or "Unknown Yahoo data error"})
            else:
                data_by_ticker[ticker] = frame

            progress_callback(completed, total, ticker)

    for ticker in tickers:
        frame = data_by_ticker.get(ticker)
        if frame is None:
            continue

        try:
            candidate, debug = scanner.classify_ticker(ticker, frame, args)
            classifications.append(debug)
            if candidate is not None:
                candidates.append(candidate)
        except Exception as exc:
            errors.append(
                {
                    "Ticker": ticker,
                    "Error": f"Classification error: {type(exc).__name__}: {exc}",
                }
            )

    if getattr(args, "crash_recovery_high_conviction", False):
        # Entry-only list: rank the candidates that already passed every hard
        # filter. Prioritize setup quality, reward/risk, target room, low risk,
        # volume confirmation, and remaining recovery runway.
        candidates.sort(
            key=lambda row: (
                float(row.get("QualityScore", row.get("Score", 0))),
                float(row.get("RewardRisk_to_Target", 0.0)),
                float(row.get("Target_pct", 0.0)),
                -float(row.get("Risk_pct", float("inf"))),
                float(row.get("RecentMaxRelVol20", row.get("RelVol20", 0.0))),
                float(row.get("UpsideToCrashMidpoint_pct", 0.0)),
            ),
            reverse=True,
        )
        # This is a cap, not a fill target. Fewer than the requested number is
        # the correct outcome when only a few names meet the strict rules.
        max_results = max(1, int(getattr(args, "crash_recovery_max_results", 5)))
        candidates = candidates[:max_results]
    elif getattr(args, "crash_recovery_only", False):
        # Broad discovery list: show triggers, then building recoveries, then
        # watch names. It intentionally remains separate from entry quality.
        status_rank = {"TRIGGERED": 0, "BUILDING": 1, "WATCH": 2}
        candidates.sort(
            key=lambda row: (
                status_rank.get(str(row.get("RecoveryStatus", "WATCH")), 3),
                float(row.get("DrawdownFromCrashHigh_pct", 0.0)),
                float(row.get("RecoveryFromCrashLow_pct", float("inf"))),
                -float(row.get("RecentMaxRelVol20", 0.0)),
            )
        )
    else:
        candidates.sort(
            key=lambda row: (
                row.get("Score", 0),
                row.get("RewardRisk_to_Target", 0),
                row.get("RelVol20", 0),
            ),
            reverse=True,
        )
    classifications.sort(key=lambda row: row.get("Ticker", ""))
    errors.sort(key=lambda row: row.get("Ticker", ""))

    return pd.DataFrame(candidates), pd.DataFrame(classifications), pd.DataFrame(errors)


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("Hourly Swing Screener")
st.caption(
    "On-demand 60-minute scan. Crash recovery ranks stocks still far below their "
    "pre-crash high whose recovery is active or recently triggered; other profiles focus on momentum confirmation."
)

if "watchlist_text" not in st.session_state:
    st.session_state.watchlist_text = default_watchlist()

if "profile_select" not in st.session_state:
    st.session_state.profile_select = "High-conviction recovery"
if "min_score_control" not in st.session_state:
    st.session_state.min_score_control = int(PROFILE_SETTINGS["High-conviction recovery"]["min_score"])
if "min_rel_volume_control" not in st.session_state:
    st.session_state.min_rel_volume_control = float(
        PROFILE_SETTINGS["High-conviction recovery"]["min_rel_volume"]
    )

def reset_profile_controls() -> None:
    """Keep the visible score and volume controls aligned with the selected profile."""
    selected = st.session_state.profile_select
    st.session_state.min_score_control = int(PROFILE_SETTINGS[selected]["min_score"])
    st.session_state.min_rel_volume_control = float(
        PROFILE_SETTINGS[selected]["min_rel_volume"]
    )

with st.expander("Watchlist and scan settings", expanded=False):
    if st.button("Reload saved watchlist from GitHub"):
        st.session_state.watchlist_text = default_watchlist()
        st.rerun()

    st.text_area(
        "Watchlist",
        key="watchlist_text",
        height=250,
        help="One symbol per line. Blank lines and # comments are ignored. Changes apply only to this browser session.",
    )
    st.caption(
        "To permanently change the default list, edit `watchlist.txt` in the GitHub repository and commit it."
    )

    controls_1, controls_2, controls_3 = st.columns(3)
    with controls_1:
        profile = st.selectbox(
            "Scan profile",
            options=[
                "High-conviction recovery",
                "Crash recovery",
                "Fresh momentum",
                "Fresh momentum — broader",
                "Balanced",
                "Relaxed review",
                "Strict",
            ],
            key="profile_select",
            on_change=reset_profile_controls,
            help=(
                "High-conviction recovery is the compact ranked shortlist. It keeps deeply sold-off names with constructive recovery behavior, "
                "liquidity, controlled support-based risk, and room to the crash midpoint. A recent breakout improves the rank but is not a brittle all-or-nothing gate. "
                "Crash recovery remains the wider discovery watchlist."
            ),
        )
    with controls_2:
        target_pct = st.number_input(
            "Profit target (%)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
        )
    with controls_3:
        include_prepost = st.checkbox("Include premarket / after-hours", value=True)

    controls_4, controls_5, controls_6 = st.columns(3)
    if profile in {"Crash recovery", "High-conviction recovery"}:
        # Recovery profiles have intentionally fixed, profile-specific rules.
        # Keeping the ordinary sliders hidden prevents stale Streamlit session
        # values from turning a curated profile into an unbounded watchlist.
        min_score = int(PROFILE_SETTINGS[profile]["min_score"])
        min_rel_volume = float(PROFILE_SETTINGS[profile]["min_rel_volume"])
        with controls_4:
            st.caption("Setup-score filter")
            if profile == "High-conviction recovery":
                st.info(f"Fixed at {min_score} for shortlist ranking")
            else:
                st.info("Not used in Crash recovery")
        with controls_5:
            st.caption("Current-bar relative-volume filter")
            if profile == "High-conviction recovery":
                st.info("Fixed by the recent-breakout volume rules")
            else:
                st.info("Not used in Crash recovery")
    else:
        # Session state can survive a Streamlit Cloud deployment. Clamp stale
        # crash-recovery values before constructing the ordinary profile sliders.
        current_score = int(st.session_state.get("min_score_control", 45))
        if current_score < 45 or current_score > 90:
            st.session_state.min_score_control = int(PROFILE_SETTINGS[profile]["min_score"])

        current_rel_volume = float(st.session_state.get("min_rel_volume_control", 0.20))
        if current_rel_volume < 0.20 or current_rel_volume > 1.50:
            st.session_state.min_rel_volume_control = float(
                PROFILE_SETTINGS[profile]["min_rel_volume"]
            )

        with controls_4:
            min_score = st.slider(
                "Minimum setup score",
                min_value=45,
                max_value=90,
                step=1,
                key="min_score_control",
            )
        with controls_5:
            min_rel_volume = st.slider(
                "Minimum relative volume",
                min_value=0.20,
                max_value=1.50,
                step=0.05,
                key="min_rel_volume_control",
            )
    with controls_6:
        workers = st.select_slider(
            "Yahoo request concurrency",
            options=[1, 2, 3, 4],
            value=3,
            help="Use 1–2 if Yahoo temporarily returns errors. Higher is faster but can be less reliable.",
        )

    if profile == "Crash recovery":
        crash_1, crash_2, crash_3 = st.columns(3)
        with crash_1:
            crash_min_drawdown = st.slider(
                "Minimum remaining drop from pre-crash high (%)",
                min_value=15.0,
                max_value=70.0,
                value=float(PROFILE_SETTINGS["Crash recovery"]["crash_recovery_min_drawdown_pct"]),
                step=1.0,
                help="The stock must still be this far below its highest hourly high before the crash low.",
            )
        with crash_2:
            crash_high_lookback = st.select_slider(
                "Pre-crash high lookback (completed hourly bars)",
                options=[120, 180, 240, 300, 360, 480],
                value=int(PROFILE_SETTINGS["Crash recovery"]["crash_recovery_high_lookback_bars"]),
                help="The table shows the highest hourly high before the selected crash low within this lookback. It is not literal all-time-high data.",
            )
        with crash_3:
            crash_max_recovery = st.slider(
                "Maximum bounce already made from crash low (%)",
                min_value=10.0,
                max_value=60.0,
                value=float(PROFILE_SETTINGS["Crash recovery"]["crash_recovery_max_recovery_from_low_pct"]),
                step=1.0,
                help="Excludes mature rebounds while allowing damaged names whose recovery started more than a few hours ago.",
            )
        high_conviction_max_candidates = None
        st.caption(
            "Crash recovery is the wide discovery watchlist. It emits WATCH, BUILDING, or TRIGGERED rows and is not the profile for a short entry list."
        )
    elif profile == "High-conviction recovery":
        crash_min_drawdown = None
        crash_high_lookback = None
        crash_max_recovery = None
        high_conviction_max_candidates = st.select_slider(
            "Maximum candidates to show",
            options=[5, 8, 10],
            value=8,
            help="The scanner returns only the best-ranked qualified entries. It does not fill the list with weaker names when fewer qualify.",
        )
        st.caption(
            "Ranked-shortlist rules: still at least 20% below the pre-crash high; 3%–35% off the crash low; at least 8% runway to the crash midpoint; average hourly dollar volume at least $3M; price no more than 3% below EMA20; constructive recent candles and volume; support-based risk no more than 6%. A recent breakout improves rank but is not an all-or-nothing requirement."
        )
    else:
        crash_min_drawdown = None
        crash_high_lookback = None
        crash_max_recovery = None
        high_conviction_max_candidates = None

    show_debug = st.checkbox("Show classifications and data errors after scan", value=True)

    if profile == "Crash recovery":
        st.caption(
            "The practical target is capped at the first meaningful overhead resistance; it can be below the 5% UI target. "
            "The table includes the highest hourly high before the selected recovery low within the selected lookback; it is not a literal all-time high."
        )
    elif profile == "High-conviction recovery":
        st.caption(
            "This profile returns a short ranked list rather than requiring a perfect current-hour breakout. Rows marked ENTRY_REVIEW already have a recent breakout; CONFIRM_ON_BREAK rows are constructive recoveries that still need an hourly confirmation."
        )
    elif profile.startswith("Fresh momentum"):
        st.caption(
            "Fresh momentum profiles only accept signals from the latest completed hourly bar. "
            "The broader mode gives more early ideas but has more false starts; it is not for automatic orders."
        )

tickers = parse_watchlist(st.session_state.watchlist_text)
status_left, status_right = st.columns([3, 1])
with status_left:
    st.write(f"**{len(tickers)} unique tickers loaded**")
with status_right:
    run_clicked = st.button("Run scan", type="primary", use_container_width=True)

if run_clicked:
    if not tickers:
        st.error("No valid ticker symbols in the watchlist.")
        st.stop()

    args = build_args(
        profile=profile,
        target_pct=target_pct,
        show_prepost=include_prepost,
        workers=workers,
        custom_rel_volume=min_rel_volume,
        custom_min_score=min_score,
    )
    if profile == "Crash recovery":
        args.crash_recovery_min_drawdown_pct = float(crash_min_drawdown)
        args.crash_recovery_high_lookback_bars = int(crash_high_lookback)
        args.crash_recovery_max_recovery_from_low_pct = float(crash_max_recovery)
    elif profile == "High-conviction recovery":
        args.crash_recovery_max_results = int(high_conviction_max_candidates)

    progress = st.progress(0, text="Starting scan…")
    last_ticker = st.empty()

    def update_progress(done: int, total: int, ticker: str) -> None:
        progress.progress(
            int(done / total * 100),
            text=f"Downloading hourly data: {done}/{total}",
        )
        last_ticker.caption(f"Latest completed data request: {ticker}")

    with st.spinner("Scanning completed hourly candles…"):
        good_entries, classifications, errors = scan_watchlist(
            tickers,
            args,
            update_progress,
        )

    progress.empty()
    last_ticker.empty()

    st.session_state.last_good_entries = good_entries
    st.session_state.last_classifications = classifications
    st.session_state.last_errors = errors
    st.session_state.last_profile = profile
    st.session_state.last_target_pct = target_pct
    st.session_state.last_show_debug = show_debug

if "last_good_entries" not in st.session_state:
    st.info("Choose settings, then tap **Run scan**.")
    st.stop()

good_entries: pd.DataFrame = st.session_state.last_good_entries
classifications: pd.DataFrame = st.session_state.last_classifications
errors: pd.DataFrame = st.session_state.last_errors
target_pct: float = st.session_state.last_target_pct
profile: str = st.session_state.last_profile

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Potential long entries", len(good_entries))
metric_2.metric("Classified tickers", len(classifications))
metric_3.metric("Data errors / unavailable", len(errors))

st.subheader(
    "High-conviction recovery candidates" if profile == "High-conviction recovery"
    else "Crash-recovery candidates" if profile == "Crash recovery"
    else "Good long-entry candidates"
)

if good_entries.empty:
    st.warning(
        "No names currently met the selected recovery conditions. This does not mean the scan failed. "
        "Use the classifications table to see the state of every ticker, and do not force a trade."
    )
else:
    display_columns = [
        "Ticker",
        "Pattern",
        "SignalPhase",
        "Action",
        "RecoveryStatus",
        "RecoveryQualityTier",
        "EntryReadiness",
        "QualityScore",
        "Score",
        "LastCompletedHourlyBar",
        "Entry",
        "Stop",
        "Target",
        "Target_pct",
        "Risk_pct",
        "RewardRisk_to_Target",
        "RelVol20",
        "HighestBeforeCrash",
        "HighestBeforeCrashBar",
        "CrashLow",
        "CrashLowBar",
        "DrawdownFromCrashHigh_pct",
        "RecoveryFromCrashLow_pct",
        "RecoveryStage",
        "RecoveryTriggerFreshnessBars",
        "RecentPositiveBars",
        "RecentMaxRelVol20",
        "TriggerRelVol20",
        "UpsideToCrashMidpoint_pct",
        "MeaningfulResistance",
        "MeaningfulResistanceDistance_pct",
        "DistanceToEMA20_pct",
        "VolumeConfirmed",
        "ShortTermSupport",
        "ResistanceDistance_pct",
        "ResistanceBefore5pctTarget",
    ]
    available_columns = [column for column in display_columns if column in good_entries.columns]

    st.dataframe(
        good_entries[available_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "QualityScore": st.column_config.NumberColumn(format="%d"),
            "Score": st.column_config.NumberColumn(format="%d"),
            "Entry": st.column_config.NumberColumn(format="$%.2f"),
            "Stop": st.column_config.NumberColumn(format="$%.2f"),
            "Target": st.column_config.NumberColumn(format="$%.2f"),
            "Target_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "Risk_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "RewardRisk_to_Target": st.column_config.NumberColumn(format="%.2f"),
            "RelVol20": st.column_config.NumberColumn(format="%.2f×"),
            "HighestBeforeCrash": st.column_config.NumberColumn(format="$%.2f"),
            "CrashLow": st.column_config.NumberColumn(format="$%.2f"),
            "ShortTermSupport": st.column_config.NumberColumn(format="$%.2f"),
            "DrawdownFromCrashHigh_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "RecoveryFromCrashLow_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "RecentMaxRelVol20": st.column_config.NumberColumn(format="%.2f×"),
            "TriggerRelVol20": st.column_config.NumberColumn(format="%.2f×"),
            "UpsideToCrashMidpoint_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "MeaningfulResistance": st.column_config.NumberColumn(format="$%.2f"),
            "MeaningfulResistanceDistance_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "DistanceToEMA20_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "ResistanceDistance_pct": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    st.download_button(
        "Download good entries CSV",
        data=dataframe_csv(good_entries),
        file_name="good_entries.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("How to read these results", expanded=False):
        st.markdown(
            f"""
- **RANKED_CRASH_RECOVERY** is the compact recovery shortlist. It returns at most the selected number of highest-ranked names. It does not promise a trade or a profit: **ENTRY_REVIEW** means a recent local breakout is holding; **CONFIRM_ON_BREAK** means the recovery is constructive but still needs a fresh hourly confirmation.
- **CRASH_RECOVERY** remains the wide recovery watchlist. Its **TRIGGERED**, **BUILDING**, and **WATCH** labels are for discovery, not automatic entry.
- **HighestBeforeCrash** is the highest hourly high *before* the selected crash low inside the selected lookback. It is a pre-crash reference, **not** literal all-time-high data.
- **CrashLow** is the lowest hourly low found in the recovery-reference window. **RecoveryFromCrashLow_pct** shows how much of the initial rebound has already occurred.
- **DrawdownFromCrashHigh_pct** is the remaining percentage below the pre-crash high. More negative means the stock is still further below that reference.
- **ShortTermSupport** and **Stop** are trade-management references. The stop is based on recent recovery support, not the original crash low.
- **Target** uses a 3%–5% trade-management objective, capped only by a meaningful structural pivot rather than a tiny hourly wick. **MeaningfulResistance** remains a review field, not a claim that price will reach Target.
- Check current price, news, earnings, and liquidity before acting. Deep recoveries can fail sharply or resume the downtrend.
"""
        )



if st.session_state.last_show_debug:
    with st.expander("All classifications", expanded=False):
        if classifications.empty:
            st.info("No ticker classifications available.")
        else:
            st.dataframe(classifications, use_container_width=True, hide_index=True)
            st.download_button(
                "Download classifications CSV",
                data=dataframe_csv(classifications),
                file_name="all_classifications.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with st.expander("Yahoo / data errors", expanded=False):
        if errors.empty:
            st.success("No provider/data errors.")
        else:
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button(
                "Download errors CSV",
                data=dataframe_csv(errors),
                file_name="errors.csv",
                mime="text/csv",
                use_container_width=True,
            )
