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
    "Cash-secured puts — prioritize downside buffer": "cash_secured_put",
    "Covered calls — stock already owned": "covered_call",
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


def strategy_title(strategy: str) -> str:
    return "Cash-secured put candidates" if strategy == "cash_secured_put" else "Covered-call candidates"


def strategy_button_text(strategy: str) -> str:
    return "Run cash-secured-put scan" if strategy == "cash_secured_put" else "Run covered-call scan"


def robinhood_option_chain_url(ticker: object) -> str:
    """Open Robinhood's option-chain page for the ticker.

    This is intentionally a review link, not a prefilled order. Robinhood may
    still require login and manual selection of expiration, strike, side, and
    Sell/Buy action.
    """
    symbol = quote(str(ticker).strip().upper(), safe="")
    return f"https://robinhood.com/options/chains/{symbol}"


def option_action_note(row: pd.Series, strategy: str) -> str:
    side = "PUT" if strategy == "cash_secured_put" else "CALL"
    action = "SELL PUT / CSP" if strategy == "cash_secured_put" else "SELL CALL / CC"
    expiry = str(row.get("Expiry", "")).strip()
    strike = pd.to_numeric(pd.Series([row.get("Strike")]), errors="coerce").iloc[0]
    bid = pd.to_numeric(pd.Series([row.get("Bid")]), errors="coerce").iloc[0]

    strike_text = f"${strike:.2f}" if pd.notna(strike) else "strike?"
    bid_text = f"bid ${bid:.2f}" if pd.notna(bid) else "bid?"
    return f"{action}: {expiry} {strike_text} {side} | {bid_text}"


def add_robinhood_review_columns(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    result = frame.copy()
    if result.empty or "Ticker" not in result.columns:
        return result

    result["RobinhoodChain"] = result["Ticker"].map(robinhood_option_chain_url)
    result["OrderToFind"] = result.apply(lambda row: option_action_note(row, strategy), axis=1)

    strike = pd.to_numeric(result.get("Strike"), errors="coerce")
    premium = pd.to_numeric(result.get("PremiumUsed"), errors="coerce")
    spot = pd.to_numeric(result.get("Spot"), errors="coerce")

    if strategy == "covered_call":
        # This is the number the user asked for: strike + premium received.
        # If the stock is at/above this by assignment, the covered-call package
        # is at or above breakeven/profit before fees and taxes.
        result["StrikePlusPremium"] = strike + premium
        result["StockMoveNeeded_pct"] = (result["StrikePlusPremium"] - spot) / spot * 100.0
    else:
        # CSP equivalent: effective stock basis after received premium.
        result["EffectiveBuyPrice"] = strike - premium

    return result


def candidate_columns(strategy: str) -> list[str]:
    """Keep the detailed table columns while adding Robinhood review links.

    The prior compact table hid too much detail. This restores the full quote,
    spread, intrinsic/extrinsic, breakeven, and contract-symbol columns.
    """
    common = [
        "Ticker",
        "RobinhoodChain",
        "OrderToFind",
        "Spot",
        "Expiry",
        "DaysToExpiry",
        "Strike",
        "StrikeDiscount_pct",
        "Bid",
        "Ask",
        "Mark",
        "MarkBidGap",
        "MarkBidGap_pct",
        "PremiumBasis",
        "PremiumUsed",
    ]

    if strategy == "cash_secured_put":
        return common + [
            "CashCollateral_perContract",
            "PremiumCredit_perContract",
            "PremiumYieldOnCollateral_pct",
            "PremiumYieldOnSpot_pct",
            "EffectiveBuyPrice",
            "PutBreakeven",
            "MaxFallBeforePutLoss_pct",
            "EstimatedAbsPutDelta",
            "BidAskSpread_pct",
            "OpenInterest",
            "OptionVolume",
            "ImpliedVolatility_pct",
            "InTheMoney",
            "LastTradeDate",
            "ContractSymbol",
        ]

    return common + [
        "StrikePlusPremium",
        "StockMoveNeeded_pct",
        "CallIntrinsic",
        "BidExtrinsic",
        "MarkExtrinsic",
        "AssignmentBreakEven",
        "AssignmentProfit_pct",
        "MaxFallBeforeCoveredCallLoss_pct",
        "CoveredCallDownsideBreakeven",
        "BidAskSpread_pct",
        "OpenInterest",
        "OptionVolume",
        "ImpliedVolatility_pct",
        "InTheMoney",
        "LastTradeDate",
        "ContractSymbol",
    ]


def candidate_column_config(strategy: str) -> dict:
    config = {
        "RobinhoodChain": st.column_config.LinkColumn(
            "Robinhood chain",
            display_text="Open chain",
            help="Opens Robinhood's option-chain page for this ticker. Manually select the exact expiration, strike, call/put side, Sell action, and limit price.",
        ),
        "OrderToFind": st.column_config.TextColumn(
            "Option to find",
            help="Manual checklist for the option chain after opening Robinhood.",
        ),
        "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
        "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
        "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
        "Mark": st.column_config.NumberColumn("Mark", format="$%.2f"),
        "MarkBidGap": st.column_config.NumberColumn(
            "Mark − Bid",
            format="$%.2f",
            help="Midpoint mark minus bid. Smaller is closer to the bid used as a conservative execution reference.",
        ),
        "MarkBidGap_pct": st.column_config.NumberColumn("Mark − Bid %", format="%.2f%%"),
        "PremiumUsed": st.column_config.NumberColumn("PremiumUsed", format="$%.2f"),
        "StrikeDiscount_pct": st.column_config.NumberColumn("StrikeDiscount_pct", format="%.2f%%"),
        "BidAskSpread_pct": st.column_config.NumberColumn("BidAskSpread_pct", format="%.2f%%"),
        "ImpliedVolatility_pct": st.column_config.NumberColumn("ImpliedVolatility_pct", format="%.2f%%"),
        "OpenInterest": st.column_config.NumberColumn("OpenInterest"),
        "OptionVolume": st.column_config.NumberColumn("OptionVolume"),
    }

    if strategy == "cash_secured_put":
        config.update(
            {
                "CashCollateral_perContract": st.column_config.NumberColumn(
                    "CashCollateral_perContract",
                    format="$%.0f",
                ),
                "PremiumCredit_perContract": st.column_config.NumberColumn(
                    "PremiumCredit_perContract",
                    format="$%.0f",
                ),
                "PremiumYieldOnCollateral_pct": st.column_config.NumberColumn(
                    "PremiumYieldOnCollateral_pct",
                    format="%.2f%%",
                    help="Premium used divided by strike collateral, before fees and taxes.",
                ),
                "PremiumYieldOnSpot_pct": st.column_config.NumberColumn(
                    "PremiumYieldOnSpot_pct",
                    format="%.2f%%",
                ),
                "EffectiveBuyPrice": st.column_config.NumberColumn(
                    "Strike − premium",
                    format="$%.2f",
                    help="Same economics as PutBreakeven: strike minus premium used.",
                ),
                "PutBreakeven": st.column_config.NumberColumn(
                    "PutBreakeven",
                    format="$%.2f",
                    help="Effective stock basis if assigned, before fees and taxes.",
                ),
                "MaxFallBeforePutLoss_pct": st.column_config.NumberColumn(
                    "MaxFallBeforePutLoss_pct",
                    format="%.2f%%",
                ),
                "EstimatedAbsPutDelta": st.column_config.NumberColumn(
                    "EstimatedAbsPutDelta",
                    format="%.3f",
                ),
            }
        )
    else:
        config.update(
            {
                "StrikePlusPremium": st.column_config.NumberColumn(
                    "Strike + premium",
                    format="$%.2f",
                    help="Strike plus premium used. Same economics as AssignmentBreakEven before fees and taxes.",
                ),
                "StockMoveNeeded_pct": st.column_config.NumberColumn(
                    "Needed move",
                    format="%.2f%%",
                    help="(Strike + premium − current stock price) / current stock price.",
                ),
                "CallIntrinsic": st.column_config.NumberColumn(
                    "Intrinsic value",
                    format="$%.2f",
                    help="max(spot - strike, 0). Used to detect stale deep-ITM call quotes.",
                ),
                "BidExtrinsic": st.column_config.NumberColumn(
                    "Bid extrinsic",
                    format="$%.2f",
                    help="Bid minus intrinsic value. Very high values on deep-ITM calls are usually stale/out-of-sync quotes.",
                ),
                "MarkExtrinsic": st.column_config.NumberColumn("Mark extrinsic", format="$%.2f"),
                "AssignmentBreakEven": st.column_config.NumberColumn(
                    "AssignmentBreakEven",
                    format="$%.2f",
                    help="Strike plus premium used; this is the final stock price needed for the covered-call package to be profitable before fees/taxes.",
                ),
                "AssignmentProfit_pct": st.column_config.NumberColumn("AssignmentProfit_pct", format="%.2f%%"),
                "MaxFallBeforeCoveredCallLoss_pct": st.column_config.NumberColumn("MaxFallBeforeCoveredCallLoss_pct", format="%.2f%%"),
                "CoveredCallDownsideBreakeven": st.column_config.NumberColumn("CoveredCallDownsideBreakeven", format="$%.2f"),
            }
        )

    return config

st.title("Option-Income Screener")
st.caption(
    "On-demand scan for covered calls or cash-secured puts. For each ticker, the scanner uses the furthest listed option expiry within your selected DTE cap. "
    "The put mode ranks qualifying contracts by the largest downside buffer first."
)

if "option_income_watchlist_text" not in st.session_state:
    st.session_state.option_income_watchlist_text = default_watchlist()

with st.expander("How the strategies work", expanded=False):
    st.markdown(
        """
### Cash-secured put
- You sell one put and reserve `strike × 100` cash in case you are assigned 100 shares.
- **Premium yield on collateral** = `premium used / strike × 100`.
- **Effective purchase breakeven** = `strike − premium used`.
- **Max fall before put loss** = `(spot − (strike − premium used)) / spot × 100`.
- A larger fall buffer and a lower estimated absolute put delta reduce risk, but neither prevents assignment or protects against a sharp decline.

### Covered call
- You own 100 shares and sell one call.
- **Assignment profit** = `(strike + premium used − spot) / spot × 100`.
- The premium offers only a limited downside cushion while upside is capped at the strike.

The screener is for research and does not place orders. The Robinhood link opens the ticker option-chain page only; manually confirm the exact expiration, strike, call/put side, Sell action, and limit price before submitting any order. Confirm the live option chain, limit-order fill, earnings date, liquidity, and whether assignment is acceptable before trading.
        """
    )

with st.expander("Watchlist and scan settings", expanded=True):
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("Reload saved watchlist from GitHub"):
            st.session_state.option_income_watchlist_text = default_watchlist()
            st.rerun()
    with top_right:
        st.caption("For permanent defaults, edit `watchlist.txt` in GitHub and commit the change.")

    st.text_area(
        "Watchlist",
        key="option_income_watchlist_text",
        height=220,
        help="One ticker per line. Commas are accepted. Lines beginning with # are ignored.",
    )

    strategy_label = st.selectbox(
        "Strategy",
        options=list(STRATEGY_LABELS),
        index=0,
        help="Cash-secured puts are the default because this mode prioritizes the largest fall buffer while targeting premium income.",
    )
    strategy = STRATEGY_LABELS[strategy_label]

    row1 = st.columns(4)
    with row1[0]:
        min_strike_discount = st.number_input(
            "Minimum strike discount below spot (%)",
            min_value=0.0,
            max_value=95.0,
            value=20.0,
            step=1.0,
            help="For a put, the strike must be at least this far below spot before premium. A 20% strike discount plus premium creates a slightly larger breakeven buffer.",
        )
    with row1[1]:
        min_return = st.number_input(
            "Minimum premium yield on collateral (%)" if strategy == "cash_secured_put" else "Minimum assignment profit (%)",
            min_value=-50.0 if strategy == "covered_call" else 0.0,
            max_value=100.0,
            value=1.0,
            step=0.25,
        )
    with row1[2]:
        max_return = st.number_input(
            "Maximum premium yield on collateral (%)" if strategy == "cash_secured_put" else "Maximum assignment profit (%)",
            min_value=-50.0 if strategy == "covered_call" else 0.0,
            max_value=100.0,
            value=8.0,
            step=0.25,
        )
    with row1[3]:
        premium_basis_label = st.selectbox(
            "Premium used in calculation",
            options=[
                "Bid — conservative / executable reference",
                "Mark — (Bid + Ask) / 2",
                "Last trade",
            ],
            index=0,
            help="Bid is the conservative reference. A midpoint Mark is an estimate and may not be fillable.",
        )

    row2 = st.columns(4)
    with row2[0]:
        min_open_interest = st.number_input(
            "Minimum open interest",
            min_value=0,
            max_value=1_000_000,
            value=25,
            step=5,
        )
    with row2[1]:
        min_option_volume = st.number_input(
            "Minimum option volume",
            min_value=0,
            max_value=1_000_000,
            value=1,
            step=1,
        )
    with row2[2]:
        max_spread = st.number_input(
            "Maximum bid-ask spread (%)",
            min_value=1.0,
            max_value=300.0,
            value=15.0,
            step=1.0,
            help="A tighter cap avoids very illiquid contracts. The Bid setting still controls the premium calculation.",
        )
    with row2[3]:
        max_expiry_days = st.number_input(
            "Maximum days to expiry",
            min_value=1,
            max_value=21,
            value=11,
            step=1,
            help="For each ticker, selects the furthest listed option expiration that is no more than this many calendar days away.",
        )

    row3 = st.columns(4)
    with row3[0]:
        if strategy == "cash_secured_put":
            max_abs_put_delta = st.slider(
                "Maximum estimated |put delta|",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
                help="Lower is more conservative. This is an IV-based delta estimate, not the probability of assignment.",
            )
        else:
            max_abs_put_delta = 1.0
            st.caption("Covered-call mode rejects likely stale deep-ITM quotes when bid extrinsic is above 2% of spot.")
    with row3[1]:
        if strategy == "cash_secured_put":
            max_csp_underlying_price = st.number_input(
                "Maximum CSP underlying price ($)",
                min_value=1.0,
                max_value=1_000.0,
                value=451.0,
                step=1.0,
                help=(
                    "Only scan cash-secured puts on stocks at or below this price. "
                    "This is a stock-price cap, not a broker collateral guarantee."
                ),
            )
            st.caption(
                f"At a {min_strike_discount:.0f}% strike discount, a ${max_csp_underlying_price:.0f} stock implies a strike near or below "
                f"${max_csp_underlying_price * (1 - min_strike_discount / 100):.2f}, or about "
                f"${max_csp_underlying_price * (1 - min_strike_discount / 100) * 100:,.0f} collateral per contract before broker checks."
            )
        else:
            max_csp_underlying_price = None
            st.caption("Maximum underlying price is used only in cash-secured-put mode.")
    with row3[2]:
        workers = st.slider("Yahoo request concurrency", min_value=1, max_value=5, value=3)
    with row3[3]:
        include_extended_spot = st.checkbox(
            "Use premarket / after-hours stock price when Yahoo supplies it",
            value=False,
        )

    show_debug = st.checkbox("Show all ticker diagnostics and data errors", value=True)

run_scan = st.button(strategy_button_text(strategy), type="primary", use_container_width=True)

if run_scan:
    tickers = screener.parse_tickers(st.session_state.option_income_watchlist_text)
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
        max_csp_underlying_price=(
            float(max_csp_underlying_price)
            if strategy == "cash_secured_put"
            else None
        ),
        max_expiry_days=int(max_expiry_days),
        include_extended_spot=include_extended_spot,
        max_workers=int(workers),
    )

    progress_bar = st.progress(0, text="Starting Yahoo option-chain requests…")
    progress_text = st.empty()

    def update_progress(done: int, total: int, ticker: str, status: str) -> None:
        progress_bar.progress(done / total, text=f"{done}/{total}: {ticker} — {status}")
        progress_text.caption("Yahoo option chains can be delayed, incomplete, or temporarily rate-limited.")

    started_at = datetime.now(timezone.utc)
    with st.spinner("Scanning option chains…"):
        output = screener.scan_tickers(tickers, config, progress_callback=update_progress)
    finished_at = datetime.now(timezone.utc)

    progress_bar.empty()
    progress_text.empty()

    st.session_state.option_income_output = output
    st.session_state.option_income_scanned_at = finished_at
    st.session_state.option_income_elapsed_seconds = (finished_at - started_at).total_seconds()
    st.session_state.option_income_strategy = strategy
    st.session_state.option_income_show_debug = show_debug


if "option_income_output" in st.session_state:
    output: screener.ScanOutput = st.session_state.option_income_output
    displayed_strategy: str = st.session_state.option_income_strategy
    candidates = pd.DataFrame(output.candidates)
    diagnostics = pd.DataFrame(output.diagnostics)
    errors = pd.DataFrame(output.errors)

    st.caption(
        f"Last scan completed {st.session_state.option_income_scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"in {st.session_state.option_income_elapsed_seconds:.1f}s."
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric(f"Potential {strategy_title(displayed_strategy).lower()}", len(candidates))
    metric_2.metric("Tickers checked", len(diagnostics))
    metric_3.metric("Data errors / unavailable", len(errors))

    st.subheader(
        "Maximum-buffer qualifying cash-secured-put candidates"
        if displayed_strategy == "cash_secured_put"
        else "Lowest-strike qualifying covered-call candidates"
    )

    if candidates.empty:
        st.info("No option contracts passed every active filter in this scan.")
    else:
        candidates_for_display = add_robinhood_review_columns(candidates, displayed_strategy)
        visible = candidates_for_display.reindex(columns=candidate_columns(displayed_strategy)).copy()
        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            column_config=candidate_column_config(displayed_strategy),
        )

        st.caption(
            "Robinhood links open the ticker option-chain page, not a prefilled order. Manually match the exact expiration, strike, call/put side, and Sell action shown in `Option to find`."
        )

        if displayed_strategy == "cash_secured_put":
            st.caption(
                "Sorted by the largest fall before an assigned put position reaches its premium-adjusted breakeven, "
                "then lower estimated absolute put delta and tighter Mark-to-Bid gap. "
                "Assignment can still occur, and a sharp decline can create a large loss after assignment."
            )
            filename = "friday_cash_secured_put_candidates.csv"
        else:
            st.caption(
                "Sorted by smallest Mark-to-Bid gap percentage, then smallest dollar gap. "
                "A smaller gap means the midpoint Mark is closer to the executable Bid."
            )
            filename = "friday_covered_call_candidates.csv"

        st.download_button(
            "Download candidates CSV",
            data=candidates_for_display.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.option_income_show_debug:
        st.subheader("All ticker diagnostics")
        if diagnostics.empty:
            st.info("No diagnostics were produced.")
        else:
            st.dataframe(diagnostics, use_container_width=True, hide_index=True)
            st.download_button(
                "Download diagnostics CSV",
                data=diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="option_income_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if errors.empty:
            st.success("No provider/data errors.")
        else:
            st.subheader("Data errors")
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button(
                "Download errors CSV",
                data=errors.to_csv(index=False).encode("utf-8"),
                file_name="option_income_errors.csv",
                mime="text/csv",
                use_container_width=True,
            )
