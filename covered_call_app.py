from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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


def candidate_columns(strategy: str) -> list[str]:
    common = [
        "Ticker",
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
            "PutBreakeven",
            "MaxFallBeforePutLoss_pct",
            "EstimatedAbsPutDelta",
            "BidAskSpread_pct",
            "OpenInterest",
            "OptionVolume",
            "ImpliedVolatility_pct",
            "ContractSymbol",
        ]

    return common + [
        "AssignmentBreakEven",
        "AssignmentProfit_pct",
        "MaxFallBeforeCoveredCallLoss_pct",
        "CoveredCallDownsideBreakeven",
        "BidAskSpread_pct",
        "OpenInterest",
        "OptionVolume",
        "ImpliedVolatility_pct",
        "ContractSymbol",
    ]


def candidate_column_config(strategy: str) -> dict:
    config = {
        "Spot": st.column_config.NumberColumn(format="$%.2f"),
        "Strike": st.column_config.NumberColumn(format="$%.2f"),
        "Bid": st.column_config.NumberColumn(format="$%.2f"),
        "Ask": st.column_config.NumberColumn(format="$%.2f"),
        "Mark": st.column_config.NumberColumn(format="$%.2f"),
        "MarkBidGap": st.column_config.NumberColumn(
            "Mark − Bid",
            format="$%.2f",
            help="Midpoint mark minus bid. Smaller is closer to the bid used as a conservative execution reference.",
        ),
        "MarkBidGap_pct": st.column_config.NumberColumn(
            "Mark − Bid %",
            format="%.2f%%",
        ),
        "PremiumUsed": st.column_config.NumberColumn(format="$%.2f"),
        "StrikeDiscount_pct": st.column_config.NumberColumn(format="%.2f%%"),
        "BidAskSpread_pct": st.column_config.NumberColumn(format="%.2f%%"),
        "ImpliedVolatility_pct": st.column_config.NumberColumn(format="%.2f%%"),
    }

    if strategy == "cash_secured_put":
        config.update(
            {
                "CashCollateral_perContract": st.column_config.NumberColumn(
                    "Cash collateral / contract",
                    format="$%.0f",
                ),
                "PremiumCredit_perContract": st.column_config.NumberColumn(
                    "Premium credit / contract",
                    format="$%.0f",
                ),
                "PremiumYieldOnCollateral_pct": st.column_config.NumberColumn(
                    "Premium yield on collateral",
                    format="%.2f%%",
                    help="Premium used divided by strike cash collateral, before fees and taxes.",
                ),
                "PutBreakeven": st.column_config.NumberColumn(
                    "Effective purchase breakeven",
                    format="$%.2f",
                    help="Strike minus premium used. Assignment means buying 100 shares at the strike; the premium reduces the economic basis.",
                ),
                "MaxFallBeforePutLoss_pct": st.column_config.NumberColumn(
                    "Max fall before put loss",
                    format="%.2f%%",
                    help="Distance from spot to strike minus premium. Higher gives more downside room before an assigned position has an unrealized loss versus current spot.",
                ),
                "EstimatedAbsPutDelta": st.column_config.NumberColumn(
                    "Estimated |put delta|",
                    format="%.3f",
                    help="Black-Scholes estimate using Yahoo implied volatility and DTE. It is a risk proxy, not an assignment probability or guarantee.",
                ),
            }
        )
    else:
        config.update(
            {
                "AssignmentBreakEven": st.column_config.NumberColumn(format="$%.2f"),
                "AssignmentProfit_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "MaxFallBeforeCoveredCallLoss_pct": st.column_config.NumberColumn(format="%.2f%%"),
                "CoveredCallDownsideBreakeven": st.column_config.NumberColumn(format="$%.2f"),
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

The screener is for research and does not place orders. Confirm the live option chain, limit-order fill, earnings date, liquidity, and whether assignment is acceptable before trading.
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

    row3 = st.columns(3)
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
            st.caption("Estimated put delta is used only in cash-secured-put mode.")
    with row3[1]:
        workers = st.slider("Yahoo request concurrency", min_value=1, max_value=5, value=3)
    with row3[2]:
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
        visible = candidates.reindex(columns=candidate_columns(displayed_strategy)).copy()
        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
            column_config=candidate_column_config(displayed_strategy),
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
            data=candidates.to_csv(index=False).encode("utf-8"),
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
