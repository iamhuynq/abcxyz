"""E. Backtest and paper trading: backtest runs and curves, forward paper portfolio vs random controls."""

import json

import pandas as pd
import streamlit as st

from stock_ui import context, exposure_panel, research_data
from stock_ui.exposure_data import paper_weights
from stock_ui.charts import equity_figure, paper_figure

st.title("Backtest and paper trading")
reader = context.reader()
PCT = st.column_config.NumberColumn(format="percent")

tab_paper, tab_bt = st.tabs(["Paper portfolio (forward)", "Backtests"])

with tab_paper:
    paper = research_data.paper(reader)
    if paper.value is None and not paper.stale and paper.note is None:
        st.info("No paper portfolio yet: it is created by `quant daily`.")
    elif context.freshness(paper, "Paper portfolio"):
        p = paper.value
        if p["run_date"] is None or p["daily"].empty:
            st.info("Not started: the frozen strategy needs at least two forward sessions (from 2026-10-03).")
        else:
            d = p["daily"]
            last, base = d.iloc[-1], d["equity"].iloc[0]
            st.caption(f"Frozen strategy 72c851c7 (hold 10, 20 positions, renew, 1bn VND), re-simulated on "
                       f"{p['run_date']} over {len(d)} forward sessions.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy", f"{last['equity'] / base - 1:+.2%}")
            c2.metric("Random median", f"{last['random_median'] / base - 1:+.2%}",
                      f"band {last['random_p05'] / base - 1:+.2%} .. {last['random_p95'] / base - 1:+.2%}",
                      delta_color="off")
            c3.metric("Equal weight", f"{last['equal_weight'] / base - 1:+.2%}")
            c4.metric("VNINDEX", f"{last['vnindex'] / base - 1:+.2%}")
            wide = last.get("wide_exceedance_p")
            if wide is not None and pd.notna(wide):
                st.caption(f"Diagnostic, not the pre-registered criterion: against 200 random controls the strategy's "
                           f"exceedance p is {wide:.3f} (median {last['wide_random_median'] / base - 1:+.2%}, "
                           f"p95 {last['wide_random_p95'] / base - 1:+.2%}). The pre-registered test uses the 20 "
                           "controls above.")
            st.plotly_chart(paper_figure(d), width="stretch")
            st.markdown(f"**Open positions ({len(p['positions'])})**")
            st.dataframe(p["positions"], hide_index=True, width="stretch", column_config={"pnl": PCT})
            weights = paper_weights(reader).value
            if weights:
                st.markdown("**Exposure of the open positions** (weighted by market value)")
                exposure_panel.render(reader, weights, key="paper")

with tab_bt:
    bt = research_data.backtest_runs(reader)
    if bt.value is None and not bt.stale and bt.note is None:
        st.info("No backtest has been run yet (`uv run quant backtest`).")
    elif context.freshness(bt, "Backtests") and not bt.value.empty:
        df = bt.value
        periods = st.multiselect("Period", sorted(df["period"].unique()), default=sorted(df["period"].unique()))
        view = df[df["period"].isin(periods)]
        st.caption("beat_random_share = share of 20 random-selection controls the strategy beat; excess_p = p-value "
                   "of the excess over equal weight (Newey-West).")
        st.dataframe(view, hide_index=True, width="stretch", height=320,
                     column_config={c: PCT for c in ("cagr", "max_drawdown", "exposure", "random_median_cagr",
                                                     "equal_weight_cagr", "beat_random_share")})
        pick = st.selectbox("Backtest detail", view["run_id"].tolist(), index=0 if len(view) else None)
        if pick:
            detail = research_data.backtest_detail(reader, pick)
            if context.freshness(detail, "Backtest detail"):
                d = detail.value
                if not d["daily"].empty:
                    st.plotly_chart(equity_figure(d["daily"], pick), width="stretch")
                m = d["metrics"]
                main = m[~m["series"].str.contains("=")].pivot(index="metric", columns="series", values="value")
                yearly = m[m["series"].str.startswith("year=")]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Metrics**")
                    st.dataframe(main, width="stretch", height=520)
                with c2:
                    if not yearly.empty:
                        st.markdown("**By year**")
                        y = yearly.assign(year=yearly["series"].str[5:]).pivot(index="year", columns="metric",
                                                                                values="value")
                        st.dataframe(y, width="stretch", column_config={c: PCT for c in y.columns})
                    st.markdown("**Exits**")
                    st.dataframe(d["exits"], hide_index=True, width="stretch", column_config={"avg_return": PCT})
                with st.expander(f"Trades (latest {len(d['trades'])})"):
                    st.dataframe(d["trades"], hide_index=True, width="stretch")
                with st.expander("Parameters"):
                    st.json(json.loads(d["params"]["params"].iloc[0]))
