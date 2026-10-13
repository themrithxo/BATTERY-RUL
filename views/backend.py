import streamlit as st

from utils import db
from utils.data import init_state
from utils.theme import insight, page_header

init_state()
page_header("Backend Monitor", "Everything the app records in its SQLite database, plus how the pieces connect.")

up, runs, preds = db.read_table("uploads"), db.read_table("runs"), db.read_table("predictions")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Backend status", "Connected" if db.ping() else "Unavailable")
k2.metric("Datasets loaded", len(up))
k3.metric("Training runs logged", len(runs))
k4.metric("Predictions logged", len(preds))

st.subheader("Data flow")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#111A30", color="#22D3EE", fontcolor="#E6EDF7", fontname="Helvetica"];
  edge [color="#8FA1C2"];
  src [label="Battery_RUL.csv\\nor uploaded file"];
  hub [label="Data Hub\\nschema detection"];
  clean [label="Cleaning\\nand EDA"];
  split [label="Blocked split"];
  models [label="Linear / Tree /\\nRandom Forest"];
  res [label="Results and\\nbusiness insights"];
  pred [label="Predict\\nsingle and batch"];
  db [label="SQLite log", shape=cylinder, color="#8B5CF6"];
  src -> hub -> clean -> split -> models -> res;
  models -> pred;
  hub -> db; models -> db; pred -> db;
}
""")

t1, t2, t3 = st.tabs(["Training runs", "Predictions", "Datasets"])
with t1:
    st.dataframe(runs, hide_index=True) if not runs.empty else st.caption("No runs logged yet.")
    if not runs.empty:
        st.download_button("Export run history", runs.to_csv(index=False).encode(), "run_history.csv", "text/csv")
with t2:
    st.dataframe(preds, hide_index=True) if not preds.empty else st.caption("No predictions logged yet.")
with t3:
    st.dataframe(up, hide_index=True) if not up.empty else st.caption("No datasets logged yet.")

st.divider()
if st.checkbox("I want to clear all logs"):
    if st.button("Clear logs"):
        db.clear_all()
        st.rerun()
