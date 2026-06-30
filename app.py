```python
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


PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
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
    },
}


def default_watchlist() -> str:
    """Read the latest default watchlist from the deployed repository."""
    path = scanner.DEFAULT_WATCHLIST
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_watchlist(raw_text: str) -> list[str]:
    """Parse watchlist text while ignoring comments, blanks, commas, and duplicates."""
    out: list[str] = []
    seen: set[str] = set()

    for line in raw_text.splitlines():
        without_comment = line.split("#", 1)[0].replace(",", " ").strip()
        for token in without_comment.split():
            ticker = token.strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                out.append(ticker)

    return out


def build_args(
    profile: str,
    target_pct: float,
    show_prepost: bool,
    workers: int,
    custom_rel_volume: float,
    custom_min_score: int,
) -> SimpleNamespace:
    settings = dict(PROFILE_SETTINGS[profile])

    # Front-end controls override profile defaults.
    settings["target_pct"] = float(target_pct)
    settings["min_rel_volume"] = float(custom_rel_volume)
    settings["min_score"] = int(custom_min_score)

    # Shared scanner/runtime settings.
    settings.update(
        {
            "period": "60d",
            "interval": "60m",
            "include_prepost": bool(show_prepost),
            "min_completed_bars": 60,
            "download_retries": 3,
            "retry_delay_seconds": 1.0,
            "max_workers": int(workers),

            # Required by target_clear_of_resistance() in the backend scanner.
            # Matches the command-line script default.
            "resistance_target_buffer_pct": 0.50,
        }
    )

    return SimpleNamespace(**settings)


def scan_watchlist(
    tickers: list[str],
    args: SimpleNamespace,
    progress_callback,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the same backend engine as the desktop script.

    Returns:
      - potential long entries
      - all classifications
      - Yahoo/provider/data errors
    """
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
                errors.append(
                    {
                        "Ticker": ticker,
                        "Error": error or "Unknown Yahoo data error",
                    }
                )
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
                    "Error": (
                        f"Classification error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

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

    return (
        pd.DataFrame(candidates),
        pd.DataFrame(classifications),
        pd.DataFrame(errors),
    )


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def title_block() -> None:
    st.title("Hourly Swing Screener")
    st.caption(
        "On-demand 60-minute chart scan. Output contains only potential "
        "long-entry candidates using the selected swing framework."
    )


title_block()

if "watchlist_text" not in st.session_state:
    st.session_state.watchlist_text = default_watchlist()

with st.expander("Watchlist and scan settings", expanded=False):
    if st.button("Reload saved watchlist from GitHub"):
        st.session_state.watchlist_text = default_watchlist()
        st.rerun()

    st.text_area(
        "Watchlist",
        key="watchlist_text",
        height=250,
        help=(
            "One symbol per line. Blank lines and # comments are ignored. "
            "Changes apply only to this browser session."
        ),
    )

    st.caption(
        "To permanently change the default list, edit `watchlist.txt` "
        "in the GitHub repository and commit it."
    )

    controls_1, controls_2, controls_3 = st.columns(3)

    with controls_1:
        profile = st.selectbox(
            "Scan profile",
            options=["Balanced", "Relaxed review", "Strict"],
            index=0,
            help=(
                "Balanced is the recommended default. Relaxed review produces "
                "more charts for manual inspection. Strict produces fewer, "
                "tighter setups."
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
        include_prepost = st.checkbox(
            "Include premarket / after-hours",
            value=True,
        )

    baseline = PROFILE_SETTINGS[profile]
    controls_4, controls_5, controls_6 = st.columns(3)

    with controls_4:
        min_score = st.slider(
            "Minimum setup score",
            min_value=45,
            max_value=90,
            value=int(baseline["min_score"]),
            step=1,
        )

    with controls_5:
        min_rel_volume = st.slider(
            "Minimum relative volume",
            min_value=0.20,
            max_value=1.50,
            value=float(baseline["min_rel_volume"]),
            step=0.05,
        )

    with controls_6:
        workers = st.select_slider(
            "Yahoo request concurrency",
            options=[1, 2, 3, 4],
            value=3,
            help=(
                "Use 1–2 if Yahoo temporarily returns errors. "
                "Higher is faster but can be less reliable."
            ),
        )

    show_debug = st.checkbox(
        "Show classifications and data errors after scan",
        value=True,
    )


tickers = parse_watchlist(st.session_state.watchlist_text)

status_left, status_right = st.columns([3, 1])

with status_left:
    st.write(f"**{len(tickers)} unique tickers loaded**")

with status_right:
    run_clicked = st.button(
        "Run scan",
        type="primary",
        use_container_width=True,
    )


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

total_classified = len(classifications)
entry_count = len(good_entries)
data_error_count = len(errors)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Potential long entries", entry_count)
metric_2.metric("Classified tickers", total_classified)
metric_3.metric("Data errors / unavailable", data_error_count)

st.subheader("Good long-entry candidates")

if good_entries.empty:
    st.warning(
        "No names passed the selected rules. This is a normal result; "
        "do not force trades. Use **Relaxed review** only to create a "
        "broader manual-review list."
    )
else:
    display_columns = [
        "Ticker",
        "Pattern",
        "Score",
        "LastCompletedHourlyBar",
        "Entry",
        "Stop",
        "Target_5pct",
        "Risk_pct",
        "RewardRisk_to_Target",
        "RelVol20",
        "ResistanceDistance_pct",
        "ResistanceBefore5pctTarget",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in good_entries.columns
    ]

    st.dataframe(
        good_entries[available_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%d"),
            "Entry": st.column_config.NumberColumn(format="$%.2f"),
            "Stop": st.column_config.NumberColumn(format="$%.2f"),
            "Target_5pct": st.column_config.NumberColumn(format="$%.2f"),
            "Risk_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "RewardRisk_to_Target": st.column_config.NumberColumn(format="%.2f"),
            "RelVol20": st.column_config.NumberColumn(format="%.2f×"),
            "ResistanceDistance_pct": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
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
- **LastCompletedHourlyBar** shows how fresh the scan data is.
- **Entry** is the last fully completed hourly candle close, not a command to market-buy.
- **Stop** is the structural invalidation point for that pattern.
- **Target** is Entry × **{target_pct:.1f}%**.
- **Reward/Risk** is the potential target reward divided by the entry-to-stop risk. Higher is better.
- **RelVol20** above `1.0×` means the final completed hour traded more volume than the 20-hour average.
- **ResistanceBefore5pctTarget = True** means a marked resistance level appears before the full target. Treat it as a decision point: a clean break/hold can justify continuing; rejection can justify reducing or exiting.
- Every result still needs a live-price check, daily-chart resistance check, and earnings/news check before an order.
"""
        )


if st.session_state.last_show_debug:
    with st.expander("All classifications", expanded=False):
        if classifications.empty:
            st.info("No ticker classifications available.")
        else:
            st.dataframe(
                classifications,
                use_container_width=True,
                hide_index=True,
            )

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
            st.dataframe(
                errors,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download errors CSV",
                data=dataframe_csv(errors),
                file_name="errors.csv",
                mime="text/csv",
                use_container_width=True,
            )
```
