from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def payoff_chart(row: dict) -> go.Figure:
    long_put = float(row["LongPutStrike"])
    short_put = float(row["ShortPutStrike"])
    short_call = float(row["ShortCallStrike"])
    long_call = float(row["LongCallStrike"])
    credit = float(row["NetCredit_perShare"])
    spot = float(row["Spot"])
    lower_be = float(row["LowerBreakeven"])
    upper_be = float(row["UpperBreakeven"])

    structure_width = max(long_call - long_put, 1.0)
    margin = max(structure_width * 0.18, spot * 0.03, 1.0)
    prices = np.linspace(max(0.01, long_put - margin), long_call + margin, 181)

    payoff_per_share = (
        credit
        + np.maximum(long_put - prices, 0.0)
        - np.maximum(short_put - prices, 0.0)
        - np.maximum(prices - short_call, 0.0)
        + np.maximum(prices - long_call, 0.0)
    )
    payoff = payoff_per_share * 100.0

    positive = np.where(payoff >= 0.0, payoff, np.nan)
    negative = np.where(payoff < 0.0, payoff, np.nan)

    fig = go.Figure()
    fig.add_vrect(
        x0=lower_be,
        x1=upper_be,
        fillcolor="rgba(0, 180, 80, 0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=prices,
            y=positive,
            mode="lines",
            line=dict(width=3, color="#16a34a"),
            fill="tozeroy",
            fillcolor="rgba(22, 163, 74, 0.16)",
            hovertemplate="Stock $%{x:.2f}<br>P/L $%{y:.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices,
            y=negative,
            mode="lines",
            line=dict(width=3, color="#f97316"),
            fill="tozeroy",
            fillcolor="rgba(249, 115, 22, 0.14)",
            hovertemplate="Stock $%{x:.2f}<br>P/L $%{y:.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(140,140,140,0.8)")
    fig.add_vline(
        x=spot,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(80,120,255,0.9)",
    )
    fig.add_vline(
        x=lower_be,
        line_width=1,
        line_dash="dot",
        line_color="rgba(140,140,140,0.7)",
    )
    fig.add_vline(
        x=upper_be,
        line_width=1,
        line_dash="dot",
        line_color="rgba(140,140,140,0.7)",
    )
    fig.update_layout(
        height=245,
        margin=dict(l=8, r=8, t=6, b=8),
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10), fixedrange=False)
    fig.update_yaxes(
        zeroline=False,
        gridcolor="rgba(128,128,128,0.12)",
        tickprefix="$",
        tickfont=dict(size=10),
        fixedrange=False,
    )
    return fig


st.title("Iron Condor Screener")
st.caption(
    "Ranks valid iron condors with a 0-100 score: 45% wider breakeven range, "
    "35% higher max-profit/max-loss ratio, 15% higher max profit, and "
    "5% lower max loss. By default, max profit must be at least max loss and "
    "the current stock price must be reasonably centered between the two breakevens. "
    "Sell legs use Bid; protective buy legs use Ask."
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
            value=1.0,
            step=0.05,
            help="1.00 means max profit must be at least as large as max loss.",
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

    r3 = st.columns(4)
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
        min_be_balance = st.number_input(
            "Minimum breakeven balance",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help=(
                "Smaller distance from current price to a breakeven divided by the larger distance. "
                "0.50 means the smaller safety buffer must be at least half the larger one; "
                "1.00 means perfectly centered."
            ),
        )
    with r3[3]:
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
        min_breakeven_balance_ratio=float(min_be_balance),
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
        st.subheader("Top payoff profiles")
        st.caption(
            "Interactive expiration P/L charts. Green is profit, orange is loss, "
            "the dashed blue line is the current stock price, and the shaded band "
            "is the breakeven range. Every candidate is shown, with four cards per row on wide screens."
        )

        chart_rows = candidates.to_dict("records")
        for start in range(0, len(chart_rows), 4):
            cols = st.columns(4)
            for offset, row in enumerate(chart_rows[start : start + 4]):
                rank = start + offset + 1
                with cols[offset]:
                    st.markdown(
                        f"**#{rank} {row['Ticker']} · Score {float(row['OverallScore']):.1f}/100**"
                    )
                    st.plotly_chart(
                        payoff_chart(row),
                        use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False},
                        key=f"ic_payoff_{rank}_{row['Ticker']}",
                    )
                    st.caption(
                        f"Buy Put {row['LongPutStrike']:.2f} · Sell Put {row['ShortPutStrike']:.2f} · "
                        f"Sell Call {row['ShortCallStrike']:.2f} · Buy Call {row['LongCallStrike']:.2f}\n\n"
                        f"BE {row['LowerBreakeven']:.2f}–{row['UpperBreakeven']:.2f}\n\n"
                        f"Current stock price: $" f"{row['Spot']:.2f}\n\n"
                        f"Down {row['DownsideBreakevenBuffer_pct']:.1f}% · "
                        f"Up {row['UpsideBreakevenBuffer_pct']:.1f}% · "
                        f"Max +$" f"{row['MaxProfit']:.0f} / -$" f"{row['MaxLoss']:.0f}"
                    )

        display = add_robinhood_column(candidates)
        visible = display.reindex(
            columns=[
                "OverallScore",
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
                "DownsideBreakevenBuffer_pct",
                "UpsideBreakevenBuffer_pct",
                "BreakevenBalanceRatio",
                "BreakevenRange",
                "BreakevenRange_pct",
                "BreakevenRangeScore",
                "ProfitLossScore",
                "MaxProfitScore",
                "LowMaxLossScore",
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
                "OverallScore": st.column_config.NumberColumn("Score", format="%.1f", pinned=True, width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", pinned=True, width="small"),
                "Spot": st.column_config.NumberColumn("Spot", format="$%.2f", pinned=True, width="small"),
                "RobinhoodChain": st.column_config.LinkColumn("Robinhood", display_text="Open chain"),
                "UnderlyingDayChange_pct": st.column_config.NumberColumn("Day move", format="%.2f%%"),
                "LongPutStrike": st.column_config.NumberColumn("Buy put", format="$%.2f"),
                "ShortPutStrike": st.column_config.NumberColumn("Sell put", format="$%.2f"),
                "ShortCallStrike": st.column_config.NumberColumn("Sell call", format="$%.2f"),
                "LongCallStrike": st.column_config.NumberColumn("Buy call", format="$%.2f"),
                "NetCredit_perCondor": st.column_config.NumberColumn("Net credit", format="$%.0f"),
                "MaxProfit": st.column_config.NumberColumn("Max profit", format="$%.0f"),
                "MaxLoss": st.column_config.NumberColumn("Max loss", format="$%.0f"),
                "ProfitLossRatio": st.column_config.NumberColumn("Profit/loss", format="%.3f"),
                "LowerBreakeven": st.column_config.NumberColumn("Lower BE", format="$%.2f"),
                "UpperBreakeven": st.column_config.NumberColumn("Upper BE", format="$%.2f"),
                "DownsideBreakevenBuffer_pct": st.column_config.NumberColumn("Downside to BE", format="%.2f%%"),
                "UpsideBreakevenBuffer_pct": st.column_config.NumberColumn("Upside to BE", format="%.2f%%"),
                "BreakevenBalanceRatio": st.column_config.NumberColumn("BE balance", format="%.3f"),
                "BreakevenRange": st.column_config.NumberColumn("BE range", format="$%.2f"),
                "BreakevenRange_pct": st.column_config.NumberColumn("BE range %", format="%.2f%%"),
                "BreakevenRangeScore": st.column_config.NumberColumn("BE score", format="%.1f"),
                "ProfitLossScore": st.column_config.NumberColumn("P/L score", format="%.1f"),
                "MaxProfitScore": st.column_config.NumberColumn("Profit score", format="%.1f"),
                "LowMaxLossScore": st.column_config.NumberColumn("Loss score", format="%.1f"),
                "NearestShortDistance_pct": st.column_config.NumberColumn("Nearest short distance", format="%.2f%%"),
                "PutWingWidth": st.column_config.NumberColumn("Put wing", format="$%.2f"),
                "CallWingWidth": st.column_config.NumberColumn("Call wing", format="$%.2f"),
            },
        )

        st.caption(
            "Overall score = 45% breakeven-range score + 35% profit/loss score + "
            "15% max-profit score + 5% low-max-loss score. Components are normalized "
            "0-100 against the valid candidates in the scan. Liquidity, executable "
            "Bid/Ask, positive credit, OTM short strikes, expiry, wing width, and the "
            "minimum profit/loss threshold and minimum breakeven-balance threshold remain hard filters."
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
