import streamlit as st

from utils.theme import inject_css

st.set_page_config(page_title="Battery RUL Lab", page_icon="🔋", layout="wide")
inject_css()

pages = [
    st.Page("views/home.py", title="Problem and Literature", icon=":material/menu_book:", default=True),
    st.Page("views/data_hub.py", title="Data Hub", icon=":material/database:"),
    st.Page("views/eda.py", title="Preprocessing and EDA", icon=":material/monitoring:"),
    st.Page("views/modeling.py", title="Model Building", icon=":material/model_training:"),
    st.Page("views/results.py", title="Results and Insights", icon=":material/insights:"),
    st.Page("views/predict.py", title="Predict and Demo", icon=":material/bolt:"),
    st.Page("views/backend.py", title="Backend Monitor", icon=":material/dns:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### Battery RUL Lab")
    st.caption("22MDC56 · Machine Learning Laboratory")
    if st.session_state.get("raw_df") is not None:
        st.caption(f"Dataset: {st.session_state.source} · {len(st.session_state.raw_df):,} rows")

nav.run()
