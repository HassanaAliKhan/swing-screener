from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

import friday_covered_call_screener as screener


st.set_page_config(
    page_title="Option-Income Screener",
    page_icon="💵",
    layout="wide",
)

DEFAULT_WATCHLIST = Path(__file__).with_name("watchlist.txt")

STRATEGY_LABELS = {
    "Covered calls — maximum ATM premium yield": "premium_yield_call",
    "Cash-secured puts — prioritize downside buffer": "cash_secured_put",
    "Covered calls — deep ITM assignment return": "covered_call",
}


def default_watchlist() -> str:
    if DEFAULT_WATCHLIST.exists():
        return DEFAULT_WATCHLIST.read_text(encoding="utf-8")
    return "# Add one ticker per line\n"


def basis_value(label: str) -> str:
    return {
        "Bid — conservative / executable reference": "bid",
        "Mark — (Bid + Ask) / 2": "mark",
        "Last trade": "last",
    }[label]


def robinhood_option_chain_url(ticker: object) -> str:
    symbol = quote(str(ticker).strip().upper(), safe="")
    return f"https://robinhood.com/options/chains/{symbol}"


def add_robinhood_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty and "Ticker" in result.columns:
        result["RobinhoodChain"] = result["Ticker"].map(robinhood_option_chain_url)
    return result


def strategy_title(strategy: str) -> str:
    if strategy == "premium_yield_call":
        return "Top ATM premium-yield covered calls"
    if strategy == "cash_secured_put":
        return "Cash-secured put candidates"
    return "Deep-ITM covered-call candidates"


def strategy_button_text(strategy: str) -> str:
    if strategy == "premium_yield_call":
        return "Run ATM premium-yield scan"
    if strategy == "cash_secured_put":
        return "Run cash-secured-put scan"
    return "Run deep-ITM covered-call scan"


def candidate_columns(strategy: str) -> list[str]:
    common = [
        "Ticker",
        "RobinhoodChain",
        "Spot",
        "Expiry",
        "DaysToExpiry",
        "Strike",
        "Bid",
        "Ask",
        "Mark",
        "PremiumBasis",
        "PremiumUsed",
    ]

    if strategy == "premium_yield_call":
        return common + [
            "StrikeDistance_pct",
            "StockInvestment",
            "PremiumCredit_perContract",
            "PremiumYieldOnInvestment_pct",
            "StrikePlusPremium",
            "CoveredCallBreakeven",
            "DownsideCushion_pct",
            "MaxProfitIfCalled_pct",
            "UnderlyingDayChange_pct",
            "BidAskSpread_pct",
            "OpenInterest",
            "OptionVolume",
            "ImpliedVolatility_pct",
            "InTheMoney",
            "ContractSymbol",
        ]

    if strategy == "cash_secured_put":
        return common + [
            "StrikeDiscount_pct",
            "CashCollateral_perContract",
            "PremiumCredit_perContract",
            "PremiumYieldOnCollateral_pct",
            "PremiumYieldOnSpot_pct",
            "PutBreakeven",
            "MaxFallBeforePutLoss_pct",
            "EstimatedAbsPutDelta",
            "BidAskSpread_pct",
            "OpenInterest",
            "OptionVolume",
            "ImpliedVolatility_pct",
            "InTheMoney",
            "ContractSymbol",
        ]

    return common + [
        "StrikeDiscount_pct",
        "AssignmentBreakEven",
        "AssignmentProfit_pct",
        "SafetyAdjustedAssignmentProfit_pct",
        "UnderlyingDayChange_pct",
        "MaxFallBeforeCoveredCallLoss_pct",
        "CoveredCallDownsideBreakeven",
        "BidAskSpread_pct",
        "OpenInterest",
        "OptionVolume",
        "ImpliedVolatility_pct",
        "InTheMoney",
        "ContractSymbol",
    ]


def candidate_column_config(strategy: str) -> dict:
    config = {
        "RobinhoodChain": st.column_config.LinkColumn(
            "Robinhood chain",
            display_text="Open chain",
        ),
        "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
        "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
        "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
        "Mark": st.column_config.NumberColumn("Mark", format="$%.2f"),
        "PremiumUsed": st.column_config.NumberColumn("Premium used", format="$%.2f"),
        "BidAskSpread_pct": st.column_config.NumberColumn("Spread", format="%.2f%%"),
        "OpenInterest": st.column_config.NumberColumn("OI"),
        "OptionVolume": st.column_config.NumberColumn("Volume"),
        "ImpliedVolatility_pct": st.column_config.NumberColumn("IV", format="%.2f%%"),
    }

    if strategy == "premium_yield_call":
        config.update(
            {
                "StrikeDistance_pct": st.column_config.NumberColumn(
                    "Strike vs spot", format="%.2f%%"
                ),
                "StockInvestment": st.column_config.NumberColumn(
                    "100-share investment", format="$%.0f"
                ),
                "PremiumCredit_perContract": st.column_config.NumberColumn(
                    "Premium credit", format="$%.0f"
                ),
                "PremiumYieldOnInvestment_pct": st.column_config.NumberColumn(
                    "Premium yield", format="%.2f%%"
                ),
                "StrikePlusPremium": st.column_config.NumberColumn(
                    "Strike + premium", format="$%.2f"
                ),
                "CoveredCallBreakeven": st.column_config.NumberColumn(
                    "Stock breakeven", format="$%.2f"
                ),
                "DownsideCushion_pct": st.column_config.NumberColumn(
                    "Premium cushion", format="%.2f%%"
                ),
                "MaxProfitIfCalled_pct": st.column_config.NumberColumn(
                    "Max profit if called", format="%.2f%%"
                ),
                "UnderlyingDayChange_pct": st.column_config.NumberColumn(
                    "Stock day move", format="%.2f%%"
                ),
            }
        )
    elif strategy == "cash_secured_put":
        config.update(
            {
                "StrikeDiscount_pct": st.column_config.NumberColumn(
                    "Strike discount", format="%.2f%%"
                ),
                "CashCollateral_perContract": st.column_config.NumberColumn(
                    "Collateral", format="$%.0f"
                ),
                "PremiumCredit_perContract": st.column_config.NumberColumn(
                    "Premium credit", format="$%.0f"
                ),
                "PremiumYieldOnCollateral_pct": st.column_config.NumberColumn(
                    "Yield on collateral", format="%.2f%%"
                ),
                "PremiumYieldOnSpot_pct": st.column_config.NumberColumn(
                    "Yield on spot", format="%.2f%%"
                ),
                "PutBreakeven": st.column_config.NumberColumn(
                    "Effective buy price", format="$%.2f"
                ),
                "MaxFallBeforePutLoss_pct": st.column_config.NumberColumn(
                    "Fall before loss", format="%.2f%%"
                ),
                "EstimatedAbsPutDelta": st.column_config.NumberColumn(
                    "Est. |put delta|", format="%.3f"
                ),
            }
        )
    else:
        config.update(
            {
                "StrikeDiscount_pct": st.column_config.NumberColumn(
                    "Strike discount", format="%.2f%%"
                ),
                "AssignmentBreakEven": st.column_config.NumberColumn(
                    "Strike + premium", format="$%.2f"
                ),
                "AssignmentProfit_pct": st.column_config.NumberColumn(
                    "Assignment profit", format="%.2f%%"
                ),
                "SafetyAdjustedAssignmentProfit_pct": st.column_config.NumberColumn(
                    "Safety-adjusted profit", format="%.2f%%"
                ),
                "UnderlyingDayChange_pct": st.column_config.NumberColumn(
                    "Stock day move", format="%.2f%%"
                ),
                "MaxFallBeforeCoveredCallLoss_pct": st.column_config.NumberColumn(
                    "Premium cushion", format="%.2f%%"
                ),
                "CoveredCallDownsideBreakeven": st.column_config.NumberColumn(
                    "Stock breakeven", format="$%.2f"
                ),
            }
        )

    return config


st.title("Option-Income Screener")
st.caption(
    "The default mode buys 100 shares conceptually, chooses the nearest ATM call "
    "inside your strike-distance limit, and returns the top stocks by call premium "
    "as a percentage of the 100-share investment."
)

if "option_income_watchlist_text" not in st.session_state:
    st.session_state.option_income_watchlist_text = default_watchlist()

with st.expander("Watchlist and scan settings", expanded=True):
    left, right = st.columns([1, 4])
    with left:
        if st.button("Reload saved watchlist from GitHub"):
            st.session_state.option_income_watchlist_text = default_watchlist()
            st.rerun()
    with right:
        st.caption("For permanent defaults, edit `watchlist.txt` in GitHub.")

    st.text_area(
        "Watchlist",
        key="option_income_watchlist_text",
        height=220,
    )

    strategy_label = st.selectbox(
        "Strategy",
        options=list(STRATEGY_LABELS),
        index=0,
    )
    strategy = STRATEGY_LABELS[strategy_label]

    row1 = st.columns(4)

    if strategy == "premium_yield_call":
        with row1[0]:
            max_atm_distance = st.number_input(
                "Maximum ATM strike distance (%)",
                min_value=0.1,
                max_value=20.0,
                value=3.0,
                step=0.5,
                help="The selected call strike must be within this percentage of spot.",
            )
        with row1[1]:
            min_return = st.number_input(
                "Minimum premium yield (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.5,
                step=0.25,
            )
        with row1[2]:
            max_return = st.number_input(
                "Maximum premium yield (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=0.5,
            )
        with row1[3]:
            top_n = st.number_input(
                "Maximum candidates",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
            )
        min_strike_discount = 0.0
    else:
        with row1[0]:
            min_strike_discount = st.number_input(
                "Minimum strike discount below spot (%)",
                min_value=0.0,
                max_value=95.0,
                value=20.0,
                step=1.0,
            )
        with row1[1]:
            min_return = st.number_input(
                "Minimum return (%)",
                min_value=-50.0 if strategy == "covered_call" else 0.0,
                max_value=100.0,
                value=1.0,
                step=0.25,
            )
        with row1[2]:
            max_return = st.number_input(
                "Maximum return (%)",
                min_value=-50.0 if strategy == "covered_call" else 0.0,
                max_value=100.0,
                value=8.0,
                step=0.25,
            )
        with row1[3]:
            top_n = 10
            max_atm_distance = 3.0

    row2 = st.columns(4)
    with row2[0]:
        premium_basis_label = st.selectbox(
            "Premium used",
            options=[
                "Bid — conservative / executable reference",
                "Mark — (Bid + Ask) / 2",
                "Last trade",
            ],
            index=0,
        )
    with row2[1]:
        min_open_interest = st.number_input(
            "Minimum open interest",
            min_value=0,
            max_value=1_000_000,
            value=25,
            step=5,
        )
    with row2[2]:
        min_option_volume = st.number_input(
            "Minimum option volume",
            min_value=0,
            max_value=1_000_000,
            value=1,
            step=1,
        )
    with row2[3]:
        max_spread = st.number_input(
            "Maximum bid-ask spread (%)",
            min_value=1.0,
            max_value=300.0,
            value=15.0,
            step=1.0,
        )

    row3 = st.columns(4)
    with row3[0]:
        max_expiry_days = st.number_input(
            "Maximum days to expiry",
            min_value=1,
            max_value=21,
            value=11,
            step=1,
        )
    with row3[1]:
        if strategy == "premium_yield_call":
            premium_yield_price_cap = st.number_input(
                "Maximum stock price ($, 0 = no cap)",
                min_value=0.0,
                max_value=10_000.0,
                value=0.0,
                step=10.0,
            )
        else:
            premium_yield_price_cap = 0.0
    with row3[2]:
        workers = st.slider(
            "Yahoo request concurrency",
            min_value=1,
            max_value=5,
            value=3,
        )
    with row3[3]:
        include_extended_spot = st.checkbox(
            "Use premarket / after-hours spot",
            value=False,
        )

    if strategy == "cash_secured_put":
        row4 = st.columns(3)
        with row4[0]:
            max_abs_put_delta = st.slider(
                "Maximum estimated |put delta|",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
            )
        with row4[1]:
            max_csp_underlying_price = st.number_input(
                "Maximum CSP underlying price ($)",
                min_value=1.0,
                max_value=1_000.0,
                value=451.0,
                step=1.0,
            )
        cc_live_quote_safety = 0.0
        max_cc_day_move = 0.0
    elif strategy == "covered_call":
        row4 = st.columns(3)
        with row4[0]:
            cc_live_quote_safety = st.number_input(
                "CC live-quote safety buffer (%)",
                min_value=0.0,
                max_value=5.0,
                value=1.5,
                step=0.25,
            )
        with row4[1]:
            max_cc_day_move = st.number_input(
                "Max stock day move for CC (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
            )
        max_abs_put_delta = 1.0
        max_csp_underlying_price = None
    else:
        max_abs_put_delta = 1.0
        max_csp_underlying_price = None
        cc_live_quote_safety = 0.0
        max_cc_day_move = 0.0

    show_debug = st.checkbox(
        "Show all ticker diagnostics and data errors",
        value=True,
    )

run_scan = st.button(
    strategy_button_text(strategy),
    type="primary",
    use_container_width=True,
)

if run_scan:
    tickers = screener.parse_tickers(
        st.session_state.option_income_watchlist_text
    )
    if not tickers:
        st.error("Add at least one ticker to the watchlist.")
        st.stop()
    if min_return > max_return:
        st.error("Minimum return cannot exceed maximum return.")
        st.stop()

    config = screener.ScanConfig(
        strategy=strategy,
        min_strike_discount_pct=float(min_strike_discount),
        min_return_pct=float(min_return),
        max_return_pct=float(max_return),
        premium_basis=basis_value(premium_basis_label),
        max_abs_put_delta=float(max_abs_put_delta),
        min_open_interest=int(min_open_interest),
        min_option_volume=int(min_option_volume),
        max_bid_ask_spread_pct=float(max_spread),
        covered_call_live_quote_safety_pct=float(cc_live_quote_safety),
        max_cc_underlying_day_change_abs_pct=float(max_cc_day_move),
        max_csp_underlying_price=(
            float(max_csp_underlying_price)
            if strategy == "cash_secured_put"
            else None
        ),
        max_atm_strike_distance_pct=float(max_atm_distance),
        max_premium_yield_stock_price=(
            float(premium_yield_price_cap)
            if strategy == "premium_yield_call"
            and float(premium_yield_price_cap) > 0
            else None
        ),
        top_n=int(top_n),
        max_expiry_days=int(max_expiry_days),
        include_extended_spot=include_extended_spot,
        max_workers=int(workers),
    )

    progress_bar = st.progress(
        0,
        text="Starting Yahoo option-chain requests…",
    )
    progress_text = st.empty()

    def update_progress(
        done: int,
        total: int,
        ticker: str,
        status: str,
    ) -> None:
        progress_bar.progress(
            done / total,
            text=f"{done}/{total}: {ticker} — {status}",
        )
        progress_text.caption(
            "Yahoo option chains can be delayed or differ from Robinhood. "
            "Confirm every candidate against Robinhood's live bid."
        )

    started_at = datetime.now(timezone.utc)
    with st.spinner("Scanning option chains…"):
        output = screener.scan_tickers(
            tickers,
            config,
            progress_callback=update_progress,
        )
    finished_at = datetime.now(timezone.utc)

    progress_bar.empty()
    progress_text.empty()

    st.session_state.option_income_output = output
    st.session_state.option_income_scanned_at = finished_at
    st.session_state.option_income_elapsed_seconds = (
        finished_at - started_at
    ).total_seconds()
    st.session_state.option_income_strategy = strategy
    st.session_state.option_income_show_debug = show_debug


if "option_income_output" in st.session_state:
    output: screener.ScanOutput = st.session_state.option_income_output
    displayed_strategy: str = st.session_state.option_income_strategy
    candidates = pd.DataFrame(output.candidates)
    diagnostics = pd.DataFrame(output.diagnostics)
    errors = pd.DataFrame(output.errors)

    st.caption(
        f"Last scan completed "
        f"{st.session_state.option_income_scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"in {st.session_state.option_income_elapsed_seconds:.1f}s."
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric(strategy_title(displayed_strategy), len(candidates))
    metric_2.metric("Tickers checked", len(diagnostics))
    metric_3.metric("Data errors / unavailable", len(errors))

    st.subheader(strategy_title(displayed_strategy))

    if candidates.empty:
        st.info("No option contracts passed every active filter in this scan.")
    else:
        display = add_robinhood_review_columns(candidates)
        visible = display.reindex(
            columns=candidate_columns(displayed_strategy)
        ).copy()

        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            column_config=candidate_column_config(displayed_strategy),
        )

        if displayed_strategy == "premium_yield_call":
            st.caption(
                "Sorted by premium yield on the current 100-share investment. "
                "The selected contract is the nearest strike to spot within your "
                "ATM-distance limit. Confirm the live Robinhood bid before buying shares."
            )
            filename = "top_atm_premium_yield_calls.csv"
        elif displayed_strategy == "cash_secured_put":
            filename = "cash_secured_put_candidates.csv"
        else:
            filename = "deep_itm_covered_call_candidates.csv"

        st.download_button(
            "Download candidates CSV",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.option_income_show_debug:
        st.subheader("All ticker diagnostics")
        if diagnostics.empty:
            st.info("No diagnostics were produced.")
        else:
            st.dataframe(
                diagnostics,
                use_container_width=True,
                hide_index=True,
            )

        if errors.empty:
            st.success("No provider/data errors.")
        else:
            st.subheader("Data errors")
            st.dataframe(
                errors,
                use_container_width=True,
                hide_index=True,
            )
