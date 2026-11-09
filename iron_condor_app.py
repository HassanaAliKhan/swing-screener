from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

import iron_condor_screener as screener


st.set_page_config(
    page_title="Iron Condor Screener",
    page_icon="🦅",
    layout="wide",
)

WATCHLIST_FILES = {
    "Stocktwits": Path(__file__).with_name("watchlist_stocktwits.txt"),
    "DC infrastructure": Path(__file__).with_name("watchlist_dc_infra.txt"),
}


def load_watchlist(name: str) -> str:
    path = WATCHLIST_FILES[name]
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "# Add one ticker per line\n"


def robinhood_option_chain_url(ticker: object) -> str:
    symbol = quote(str(ticker).strip().upper(), safe="")
    return f"https://robinhood.com/options/chains/{symbol}"


def add_robinhood_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty and "Ticker" in result.columns:
        result["RobinhoodChain"] = result["Ticker"].map(robinhood_option_chain_url)
    return result


st.title("Iron Condor Screener")
st.caption(
    "Ranks candidates by widest expiration profit zone first, then highest "
    "max-profit/max-loss ratio. Short legs use Bid and protective long legs use Ask."
)

if "ic_watchlist_choice" not in st.session_state:
    st.session_state.ic_watchlist_choice = "Stocktwits"
if "ic_loaded_watchlist" not in st.session_state:
    st.session_state.ic_loaded_watchlist = "Stocktwits"
if "ic_watchlist_text" not in st.session_state:
    st.session_state.ic_watchlist_text = load_watchlist("Stocktwits")

with st.expander("Watchlist and scan settings", expanded=True):
    c1, c2, c3 = st.columns([2, 1, 4])

    with c1:
        selected_watchlist = st.selectbox(
            "Watchlist",
            options=list(WATCHLIST_FILES),
            key="ic_watchlist_choice",
        )

    if selected_watchlist != st.session_state.ic_loaded_watchlist:
        st.session_state.ic_watchlist_text = load_watchlist(selected_watchlist)
        st.session_state.ic_loaded_watchlist = selected_watchlist
        st.rerun()

    with c2:
        st.write("")
        st.write("")
        if st.button("Reload selected"):
            st.session_state.ic_watchlist_text = load_watchlist(selected_watchlist)
            st.session_state.ic_loaded_watchlist = selected_watchlist
            st.rerun()

    with c3:
        st.caption(
            "Temporary edits can be made directly in the ticker box below. "
            "The scan uses the furthest listed expiration inside the selected DTE window."
        )

    st.text_area("Tickers", key="ic_watchlist_text", height=200)

    r1 = st.columns(4)
    with r1[0]:
        max_expiry_days = st.number_input(
            "Maximum days to expiry", min_value=1, max_value=60, value=11, step=1
        )
    with r1[1]:
        top_n = st.number_input(
            "Maximum candidates", min_value=1, max_value=100, value=25, step=1
        )
    with r1[2]:
        min_credit = st.number_input(
            "Minimum net credit / share",
            min_value=0.01,
            max_value=20.0,
            value=0.05,
            step=0.05,
        )
    with r1[3]:
        min_pl = st.number_input(
            "Minimum max-profit / max-loss ratio",
            min_value=0.0,
            max_value=10.0,
            value=0.05,
            step=0.05,
        )

    r2 = st.columns(4)
    with r2[0]:
        min_oi = st.number_input(
            "Minimum OI per short leg", min_value=0, max_value=1_000_000, value=5, step=5
        )
    with r2[1]:
        min_vol = st.number_input(
            "Minimum volume per short leg", min_value=0, max_value=1_000_000, value=1, step=1
        )
    with r2[2]:
        max_spread = st.number_input(
            "Maximum bid-ask spread per short leg (%)",
            min_value=1.0,
            max_value=1000.0,
            value=500.0,
            step=1.0,
        )
    with r2[3]:
        extended = st.checkbox("Use premarket / after-hours spot", value=True)

    r3 = st.columns(3)
    with r3[0]:
        max_wing = st.number_input(
            "Maximum wing width (% of spot)",
            min_value=0.5,
            max_value=50.0,
            value=15.0,
            step=0.5,
        )
    with r3[1]:
        max_short_distance = st.number_input(
            "Maximum short-strike distance (% of spot)",
            min_value=1.0,
            max_value=80.0,
            value=35.0,
            step=1.0,
        )
    with r3[2]:
        workers = st.slider("Yahoo request concurrency", 1, 5, 3)

    show_debug = st.checkbox("Show diagnostics", value=True)

run = st.button("Run iron-condor scan", type="primary", use_container_width=True)

if run:
    tickers = screener.parse_tickers(st.session_state.ic_watchlist_text)
    if not tickers:
        st.error("Add at least one ticker.")
        st.stop()

    config = screener.IronCondorConfig(
        min_open_interest=int(min_oi),
        min_option_volume=int(min_vol),
        max_bid_ask_spread_pct=float(max_spread),
        max_expiry_days=int(max_expiry_days),
        include_extended_spot=bool(extended),
        max_workers=int(workers),
        min_credit_per_share=float(min_credit),
        min_profit_loss_ratio=float(min_pl),
        max_wing_width_pct_of_spot=float(max_wing),
        max_short_distance_pct_of_spot=float(max_short_distance),
        top_n=int(top_n),
    )

    bar = st.progress(0, text="Starting iron-condor scan…")
    msg = st.empty()

    def update_progress(done: int, total: int, ticker: str, status: str) -> None:
        bar.progress(done / total, text=f"{done}/{total}: {ticker} — {status}")
        msg.caption("Verify all four Robinhood live quotes before placing a trade.")

    started = datetime.now(timezone.utc)
    with st.spinner("Scanning four-leg combinations…"):
        out = screener.scan_iron_condors(
            tickers,
            config,
            progress_callback=update_progress,
        )
    finished = datetime.now(timezone.utc)

    bar.empty()
    msg.empty()

    st.session_state.ic_output = out
    st.session_state.ic_scanned_at = finished
    st.session_state.ic_elapsed = (finished - started).total_seconds()
    st.session_state.ic_show_debug = show_debug

if "ic_output" in st.session_state:
    out = st.session_state.ic_output
    candidates = pd.DataFrame(out.candidates)
    diagnostics = pd.DataFrame(out.diagnostics)
    errors = pd.DataFrame(out.errors)

    st.caption(
        f"Last scan: {st.session_state.ic_scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"({st.session_state.ic_elapsed:.1f}s)"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates", len(candidates))
    m2.metric("Tickers checked", len(diagnostics))
    m3.metric("Errors", len(errors))

    st.subheader("Ranked iron-condor candidates")

    if candidates.empty:
        st.info("No iron-condor combinations passed every active filter.")
    else:
        display = add_robinhood_column(candidates)
        visible = display.reindex(
            columns=[
                "Ticker",
                "Spot",
                "RobinhoodChain",
                "UnderlyingDayChange_pct",
                "Expiry",
                "DaysToExpiry",
                "LongPutStrike",
                "ShortPutStrike",
                "ShortCallStrike",
                "LongCallStrike",
                "NetCredit_perCondor",
                "MaxProfit",
                "MaxLoss",
                "ProfitLossRatio",
                "LowerBreakeven",
                "UpperBreakeven",
                "ProfitZoneWidth",
                "ProfitZoneWidth_pct",
                "NearestShortDistance_pct",
                "PutWingWidth",
                "CallWingWidth",
                "TotalOpenInterest",
                "TotalOptionVolume",
            ]
        )

        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", pinned=True, width="small"),
                "Spot": st.column_config.NumberColumn("Spot", format="$%.2f", pinned=True, width="small"),
                "RobinhoodChain": st.column_config.LinkColumn("Robinhood", display_text="Open chain"),
                "UnderlyingDayChange_pct": st.column_config.NumberColumn("Day move", format="%.2f%%"),
                "LongPutStrike": st.column_config.NumberColumn("Long put", format="$%.2f"),
                "ShortPutStrike": st.column_config.NumberColumn("Short put", format="$%.2f"),
                "ShortCallStrike": st.column_config.NumberColumn("Short call", format="$%.2f"),
                "LongCallStrike": st.column_config.NumberColumn("Long call", format="$%.2f"),
                "NetCredit_perCondor": st.column_config.NumberColumn("Net credit", format="$%.0f"),
                "MaxProfit": st.column_config.NumberColumn("Max profit", format="$%.0f"),
                "MaxLoss": st.column_config.NumberColumn("Max loss", format="$%.0f"),
                "ProfitLossRatio": st.column_config.NumberColumn("Profit/loss", format="%.3f"),
                "LowerBreakeven": st.column_config.NumberColumn("Lower BE", format="$%.2f"),
                "UpperBreakeven": st.column_config.NumberColumn("Upper BE", format="$%.2f"),
                "ProfitZoneWidth": st.column_config.NumberColumn("Profit-zone width", format="$%.2f"),
                "ProfitZoneWidth_pct": st.column_config.NumberColumn("Profit-zone width %", format="%.2f%%"),
                "NearestShortDistance_pct": st.column_config.NumberColumn("Nearest short distance", format="%.2f%%"),
                "PutWingWidth": st.column_config.NumberColumn("Put wing", format="$%.2f"),
                "CallWingWidth": st.column_config.NumberColumn("Call wing", format="$%.2f"),
            },
        )

        st.caption(
            "Ranking: widest expiration profit zone first, then highest max-profit/max-loss ratio, "
            "then higher executable net credit. Max profit equals net credit; max loss is the wider "
            "wing minus the net credit."
        )

        st.download_button(
            "Download candidates CSV",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name="iron_condor_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.get("ic_show_debug", True):
        st.subheader("Diagnostics")
        if diagnostics.empty:
            st.info("No diagnostics were produced.")
        else:
            st.dataframe(diagnostics, use_container_width=True, hide_index=True)

        if not errors.empty:
            st.subheader("Errors")
            st.dataframe(errors, use_container_width=True, hide_index=True)
