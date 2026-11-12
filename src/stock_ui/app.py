"""Local management UI: `scripts/ui.sh` (binds 127.0.0.1 only)."""

from pathlib import Path

import streamlit as st

from stock_ui import context
from stock_ui.tasks import busy_reason

HERE = Path(__file__).parent

st.set_page_config(page_title="Stock system", layout="wide")
nav = st.navigation([
    st.Page(HERE / "pages" / "watch.py", title="Market watch", default=True),
    st.Page(HERE / "pages" / "overview.py", title="Overview"),
    st.Page(HERE / "pages" / "tasks.py", title="Run tasks"),
    st.Page(HERE / "pages" / "symbol.py", title="Symbol lookup"),
    st.Page(HERE / "pages" / "research.py", title="Research"),
    st.Page(HERE / "pages" / "cross.py", title="Cross-stock"),
    st.Page(HERE / "pages" / "backtest.py", title="Backtest and paper"),
    st.Page(HERE / "pages" / "risk.py", title="Risk"),
    st.Page(HERE / "pages" / "daily.py", title="Daily report"),
])
with st.sidebar:
    data_dir = context.data_dir()
    busy = busy_reason(data_dir)
    if busy:
        st.warning(f"Updating: {busy}. Pages show the last data read before it started.")
    st.caption(f"Data: {data_dir}")
    if st.button("Refresh data"):
        context.reader().clear()
        st.rerun()
nav.run()
