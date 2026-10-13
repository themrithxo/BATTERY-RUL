from pathlib import Path

import streamlit as st

from utils import db
from utils.data import (USER_PATH, find_default_csv, init_state, read_table, set_dataset)
from utils.theme import insight, page_header

init_state()
page_header("Data Hub", "Load the default dataset or upload a new one. Every other page follows what you load here.")

src = st.radio("Dataset source", ["Default dataset", "Upload a new dataset"], horizontal=True)

if src == "Default dataset":
    found = find_default_csv()
    path = st.text_input("File path", value=str(found) if found else USER_PATH,
                         help="Tip: copy Battery_RUL.csv into the app's data folder and it loads automatically.")
    if st.button("Load dataset", type="primary"):
        try:
            p = Path(path)
            if p.is_dir():
                p = sorted(p.glob("*.csv"))[0]
            set_dataset(read_table(p), p.name)
            st.success(f"Loaded {p.name}")
        except Exception as e:
            st.error(f"Could not load that file: {e}")
else:
    up = st.file_uploader("CSV or Excel file with numeric columns", type=["csv", "xlsx", "xls"])
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.upload_sig != sig:
            try:
                set_dataset(read_table(up, up.name), up.name)
                st.session_state.upload_sig = sig
                st.success(f"Loaded {up.name}. Pipeline, models and predictions now use this dataset.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")

df = st.session_state.raw_df
if df is None:
    insight("No dataset loaded yet. Press **Load dataset** or upload a file.", "warn")
    st.stop()

st.subheader(f"Schema: {st.session_state.source}")
num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.error("This file has no numeric columns, so there is nothing to model.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    t_idx = num_cols.index(st.session_state.target) if st.session_state.target in num_cols else len(num_cols) - 1
    new_target = st.selectbox("Target column (what to predict)", num_cols, index=t_idx)
with c2:
    opts = ["None"] + [c for c in num_cols if c != new_target]
    cur = st.session_state.cycle_col if st.session_state.cycle_col in opts else "None"
    new_cycle = st.selectbox("Cycle column (optional)", opts, index=opts.index(cur))
new_cycle = None if new_cycle == "None" else new_cycle
if new_target != st.session_state.target or new_cycle != st.session_state.cycle_col:
    st.session_state.target, st.session_state.cycle_col = new_target, new_cycle
    st.session_state.results = None
    st.session_state.full_models = None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", df.shape[1])
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2 = st.tabs(["Preview", "Column types"])
with tab1:
    st.dataframe(df.head(200))
with tab2:
    st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())

st.subheader("Backend connectivity")
checks = {
    "Dataset loaded in session": True,
    "Target column is numeric": st.session_state.target in num_cols,
    "SQLite backend reachable": db.ping(),
    "At least 2 numeric features": len(num_cols) >= 2,
}
for label, ok in checks.items():
    cls, mark = ("status-ok", "✔") if ok else ("status-bad", "✘")
    st.markdown(f"<span class='{cls}'>{mark}</span> {label}", unsafe_allow_html=True)

st.subheader("Upload history")
hist = db.read_table("uploads")
if hist.empty:
    st.caption("No uploads logged yet.")
else:
    st.dataframe(hist, hide_index=True)
