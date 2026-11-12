"""F. Risk: stored Risk Engine evaluations (limits, stress scenarios, positions). Read-only: every number comes from
the risk_* tables written by `quant risk evaluate`; nothing is recomputed, and nothing here changes a portfolio."""

import streamlit as st

from stock_ui import context, risk_data

st.title("Risk")
reader = context.reader()
PCT = st.column_config.NumberColumn(format="percent")
STATUS = {"PASS": st.success, "WARNING": st.warning, "BREACH": st.error, "INCOMPLETE": st.warning}

listing = risk_data.runs(reader)
if listing.value is None and not listing.stale and listing.note is None:
    st.info("No risk evaluation yet: run `uv run quant risk evaluate --paper` (or the Risk task on Run tasks).")
elif context.freshness(listing, "Risk runs"):
    df = listing.value
    sources = sorted({str(t).split(":")[0] for t in df["target"].dropna()})
    pick_src = st.multiselect("Source", sources, default=sources)
    view = df[df["target"].isna() | df["target"].fillna("").str.split(":").str[0].isin(pick_src)]
    failed = view[view["risk_status"].isna()]
    if not failed.empty:
        st.caption("Runs without a stored evaluation (failed, running or invalid): "
                   + "; ".join(f"{r.run_id} ({r.run_status})" for r in failed.itertuples()))
    done = view[view["risk_status"].notna()]
    if done.empty:
        st.info("No completed risk evaluation for this selection.")
    else:
        labels = {f"{r.valuation_date} {r.target} [{r.risk_status}] {r.run_id}": r.run_id for r in done.itertuples()}
        run_id = labels[st.selectbox("Risk run", list(labels))]
        res = risk_data.run(reader, run_id)
        if context.freshness(res, "Risk run") and res.value is not None:
            r = res.value
            ev = r["evaluation"]
            STATUS.get(ev["status"], st.info)(f"{ev['target']} at {ev['valuation_date']}: **{ev['status']}**")
            st.caption(f"Policy `{ev['policy_id']}/{ev['policy_version']}` (provisional), risk model "
                       f"`{ev['model_version']}`, scenario set `{r['scenarios'] or 'none'}`. NAV {ev['nav']:,.0f} VND. "
                       "A PASS means the declared policy is met with the available inputs, not that the portfolio "
                       "is safe; INCOMPLETE means inputs were missing.")
            for note in str(ev.get("notes") or "").splitlines():
                st.caption(note)
            tabs = st.tabs(["Limits", "Stress scenarios", "Positions", "Metrics", "Re-simulation"])
            with tabs[0]:
                st.dataframe(r["limits"], hide_index=True, width="stretch")
            with tabs[1]:
                if r["stress"].empty:
                    st.info("No stress scenarios in this run (risk model v1, or --no-stress).")
                else:
                    st.caption("Illustrations, not forecasts or policy limits: each scenario is MEASURED or "
                               "INCOMPLETE and none is summed with another. Valuation loss and liquidation cost are "
                               "separate; idiosyncratic return is 0.")
                    st.dataframe(r["stress"].drop(columns=["seq"]), hide_index=True, width="stretch",
                                 column_config={"valuation_loss_pct": PCT, "coverage": PCT})
            with tabs[2]:
                st.dataframe(r["positions"], hide_index=True, width="stretch",
                             column_config={"weight": PCT, "risk_share": PCT})
            with tabs[3]:
                m = r["metrics"]
                for scope in ("portfolio", "concentration", "curve:", "valuation:", "liquidity", "tilt"):
                    part = m[m["scope"].str.startswith(scope)]
                    if not part.empty:
                        st.markdown(f"**{scope.rstrip(':')}**")
                        st.dataframe(part, hide_index=True, width="stretch")
            with tabs[4]:
                rec = r["reconciliation"]
                if rec.empty:
                    st.info("Not a re-simulated portfolio.")
                else:
                    ok = bool(rec["within"].all())
                    (st.success if ok else st.warning)(
                        f"Re-simulation of a construction configuration with today's code and data; "
                        f"{'it reconciles with' if ok else 'it does NOT reconcile with'} the stored run "
                        f"{rec['source_run'].iloc[0]} (relative tolerance 1e-9). It is not the historical run.")
                    st.dataframe(rec.drop(columns=["source_run"]), hide_index=True, width="stretch")
