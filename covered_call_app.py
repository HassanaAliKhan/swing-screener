from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import friday_covered_call_screener as screener


st.set_page_config(
    page_title="Friday Covered-Call Screener",
    page_icon="📞",
    layout="wide",
)


DEFAULT_WATCHLIST = Path(__file__).with_name("watchlist.txt")


def default_watchlist() -> str:
    if DEFAULT_WATCHLIST.exists():
        return DEFAULT_WATCHLIST.read_text(encoding="utf-8")
    return "# Add one ticker per line\n"


def currency(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


st.title("Friday Covered-Call Screener")
st.caption(
    "Finds the lowest-strike qualifying call per ticker for the nearest listed Friday expiry. "
    "This is an options-chain review tool; it does not place orders."
)

st.warning(
    "Assumption: covered call — you own or buy 100 shares and sell one call. "
    "Yahoo/yfinance option quotes can be delayed, stale, incomplete, or unavailable. "
    "Confirm the live bid, ask, contract multiplier, expiry, and liquidity with your broker before trading."
)

with st.expander("How the calculations work", expanded=False):
    st.markdown(
        """
- **Assignment break-even / capped-sale value** = `strike + premium received`.
- **Assignment profit %** = `(strike + premium − spot) / spot × 100`.
- **Max fall before covered-call loss %** = `premium / spot × 100`. This is the premium buffer for the combined stock-plus-call position at expiry, before commissions, taxes, slippage, early assignment, and any changes to the option price before expiry.
- **Covered-call downside break-even** = `spot − premium`.
- The screener selects **one contract per ticker: the lowest strike** satisfying every active rule.
        """
    )

if "cc_watchlist_text" not in st.session_state:
    st.session_state.cc_watchlist_text = default_watchlist()

with st.expander("Watchlist and scan settings", expanded=True):
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("Reload saved watchlist from GitHub"):
            st.session_state.cc_watchlist_text = default_watchlist()
            st.rerun()
    with top_right:
        st.caption("For permanent defaults, edit `watchlist.txt` in GitHub and commit the change.")

    st.text_area(
        "Watchlist",
        key="cc_watchlist_text",
        height=230,
        help="One ticker per line. Commas are accepted. Lines beginning with # are ignored.",
    )

    row1 = st.columns(4)
    with row1[0]:
        min_strike_discount = st.number_input(
            "Minimum strike discount below spot (%)",
            min_value=0.0,
            max_value=95.0,
            value=20.0,
            step=1.0,
        )
    with row1[1]:
        min_profit = st.number_input(
            "Minimum assignment profit (%)",
            min_value=-50.0,
            max_value=100.0,
            value=1.0,
            step=0.25,
        )
    with row1[2]:
        max_profit = st.number_input(
            "Maximum assignment profit (%)",
            min_value=-50.0,
            max_value=100.0,
            value=5.0,
            step=0.25,
        )
    with row1[3]:
        premium_basis_label = st.selectbox(
            "Premium used in calculation",
            options=["Bid — conservative", "Midpoint", "Last trade"],
            index=0,
            help="Bid is the conservative default because it is the price currently offered by buyers.",
        )

    row2 = st.columns(4)
    with row2[0]:
        min_open_interest = st.number_input(
            "Minimum open interest",
            min_value=0,
            max_value=1_000_000,
            value=10,
            step=5,
        )
    with row2[1]:
        min_option_volume = st.number_input(
            "Minimum option volume",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1,
        )
    with row2[2]:
        max_spread = st.number_input(
            "Maximum bid-ask spread (%)",
            min_value=1.0,
            max_value=300.0,
            value=30.0,
            step=5.0,
        )
    with row2[3]:
        workers = st.slider("Yahoo request concurrency", min_value=1, max_value=5, value=3)

    col_a, col_b = st.columns(2)
    with col_a:
        include_extended_spot = st.checkbox(
            "Use premarket / after-hours stock price when Yahoo supplies it",
            value=False,
        )
    with col_b:
        show_debug = st.checkbox("Show all ticker diagnostics and data errors", value=True)

run_scan = st.button("Run Friday covered-call scan", type="primary", use_container_width=True)

if run_scan:
    tickers = screener.parse_tickers(st.session_state.cc_watchlist_text)
    if not tickers:
        st.error("Add at least one ticker to the watchlist.")
        st.stop()
    if min_profit > max_profit:
        st.error("Minimum assignment profit cannot exceed maximum assignment profit.")
        st.stop()

    basis_map = {
        "Bid — conservative": "bid",
        "Midpoint": "midpoint",
        "Last trade": "last",
    }
    config = screener.ScanConfig(
        min_strike_discount_pct=float(min_strike_discount),
        min_assignment_profit_pct=float(min_profit),
        max_assignment_profit_pct=float(max_profit),
        premium_basis=basis_map[premium_basis_label],
        min_open_interest=int(min_open_interest),
        min_option_volume=int(min_option_volume),
        max_bid_ask_spread_pct=float(max_spread),
        include_extended_spot=include_extended_spot,
        max_workers=int(workers),
    )

    progress_bar = st.progress(0, text="Starting Yahoo option-chain requests…")
    progress_text = st.empty()

    def update_progress(done: int, total: int, ticker: str, status: str) -> None:
        progress_bar.progress(done / total, text=f"{done}/{total}: {ticker} — {status}")
        progress_text.caption("Yahoo option chains can be incomplete or temporarily rate-limited.")

    started_at = datetime.now(timezone.utc)
    with st.spinner("Scanning option chains…"):
        output = screener.scan_tickers(tickers, config, progress_callback=update_progress)
    progress_bar.empty()
    progress_text.empty()

    finished_at = datetime.now(timezone.utc)
    st.session_state.cc_output = output
    st.session_state.cc_scanned_at = finished_at
    st.session_state.cc_elapsed_seconds = (finished_at - started_at).total_seconds()

if "cc_output" in st.session_state:
    output: screener.ScanOutput = st.session_state.cc_output
    candidates = pd.DataFrame(output.candidates)
    diagnostics = pd.DataFrame(output.diagnostics)
    errors = pd.DataFrame(output.errors)

    st.caption(
        f"Last scan completed {st.session_state.cc_scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"in {st.session_state.cc_elapsed_seconds:.1f}s."
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Potential covered-call candidates", len(candidates))
    metric_2.metric("Tickers checked", len(diagnostics))
    metric_3.metric("Data errors / unavailable", len(errors))

    st.subheader("Lowest-strike qualifying covered-call candidates")
    if candidates.empty:
        st.info("No call contracts passed every active filter in this scan.")
    else:
        visible_columns = [
            "Ticker", "Spot", "Expiry", "Strike", "StrikeDiscount_pct", "Bid", "Ask",
            "PremiumUsed", "AssignmentBreakEven", "AssignmentProfit_pct",
            "MaxFallBeforeCoveredCallLoss_pct", "CoveredCallDownsideBreakeven",
            "BidAskSpread_pct", "OpenInterest", "OptionVolume", "ImpliedVolatility_pct",
            "ContractSymbol",
        ]
        visible = candidates.reindex(columns=visible_columns).copy()
        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spot": st.column_config.NumberColumn(format="$%.2f"),
                "Strike": st.column_config.NumberColumn(format="$%.2f"),
                "Bid": st.column_config.NumberColumn(format="$%.2f"),
                "Ask": st.column_config.NumberColumn(format="$%.2f"),
                "PremiumUsed": st.column_config.NumberColumn(format="$%.2f"),
                "AssignmentBreakEven": st.column_config.NumberColumn(format="$%.2f"),
                "CoveredCallDownsideBreakeven": st.column_config.NumberColumn(format="$%.2f"),
                "StrikeDiscount_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "AssignmentProfit_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "MaxFallBeforeCoveredCallLoss_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "BidAskSpread_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "ImpliedVolatility_pct": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "Sorted from maximum to minimum premium downside buffer: `PremiumUsed / Spot`. "
            "The next sort key is strike distance below spot."
        )
        st.download_button(
            "Download covered-call candidates CSV",
            data=candidates.to_csv(index=False).encode("utf-8"),
            file_name="friday_covered_call_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if show_debug:
        st.subheader("All ticker diagnostics")
        if diagnostics.empty:
            st.info("No diagnostics were produced.")
        else:
            st.dataframe(diagnostics, use_container_width=True, hide_index=True)
            st.download_button(
                "Download diagnostics CSV",
                data=diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="friday_covered_call_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if not errors.empty:
            st.subheader("Data errors / unavailable chains")
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button(
                "Download errors CSV",
                data=errors.to_csv(index=False).encode("utf-8"),
                file_name="friday_covered_call_errors.csv",
                mime="text/csv",
                use_container_width=True,
            )
