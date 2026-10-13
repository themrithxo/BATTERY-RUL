import numpy as np
import plotly.express as px
import streamlit as st

from utils import db
from utils.data import ensure_data, get_clean
from utils.ml import DEFAULT_PARAMS, SPLITS, metrics_frame, run_training
from utils.theme import C, MODEL_COLORS, insight, page_header, show

ensure_data()
s = st.session_state
page_header("Model Building", "Train Linear Regression, Decision Tree and Random Forest on the cleaned data.")

clean, _ = get_clean()
target = s.target
all_feats = [c for c in clean.columns if c != target]

with st.form("train_form"):
    st.markdown("**Features and validation**")
    c1, c2 = st.columns([3, 2])
    with c1:
        features = st.multiselect("Features", all_feats, default=all_feats)
    with c2:
        split_mode = st.selectbox("Split strategy", SPLITS,
                                  help="Blocked holds out whole stretches of consecutive cycles, so the test set "
                                       "is not a near-copy of the training set. Random rows leak between neighbours.")
    c3, c4, c5 = st.columns(3)
    test_size = c3.slider("Test share", 0.1, 0.4, 0.2, 0.05)
    n_blocks = c4.slider("Number of blocks (blocked split)", 5, 50, 20)
    seed = c5.number_input("Random seed", 0, 9999, DEFAULT_PARAMS["seed"])
    st.markdown("**Hyperparameters** (depth 0 means unlimited)")
    h1, h2, h3, h4, h5 = st.columns(5)
    dt_depth = h1.slider("Tree depth", 0, 30, DEFAULT_PARAMS["dt_depth"])
    dt_leaf = h2.slider("Tree min leaf", 1, 50, DEFAULT_PARAMS["dt_leaf"])
    rf_trees = h3.slider("Forest trees", 20, 500, DEFAULT_PARAMS["rf_trees"], 20)
    rf_depth = h4.slider("Forest depth", 0, 30, DEFAULT_PARAMS["rf_depth"])
    rf_leaf = h5.slider("Forest min leaf", 1, 50, DEFAULT_PARAMS["rf_leaf"])
    go_train = st.form_submit_button("Train models")

if go_train:
    if len(features) < 1:
        st.error("Pick at least one feature.")
    else:
        params = dict(dt_depth=dt_depth, dt_leaf=dt_leaf, rf_trees=rf_trees, rf_depth=rf_depth,
                      rf_leaf=rf_leaf, seed=int(seed))
        with st.spinner("Training three models..."):
            res = run_training(clean, features, target, split_mode, test_size, n_blocks, params)
        res["prep"] = dict(s.prep)
        res["dataset"] = s.source
        s.results = res
        s.full_models = None
        for name, v in res["models"].items():
            m = v["metrics"]
            db.log_run(s.source, split_mode, name, m["MAE"], m["RMSE"], m["R2"], m["Fit s"], params)

res = s.results
if res is None:
    insight("Choose your settings and press **Train models**. Results appear here and on the Results and Insights page.")
    st.stop()

st.subheader("Train/test split")
n = len(clean)
tag = np.where(np.isin(np.arange(n), res["te"]), "test", "train")
view = clean[[target]].copy()
view["row"] = range(n)
view["set"] = tag
fig = px.scatter(view, x="row", y=target, color="set", render_mode="webgl",
                 color_discrete_map={"train": C["cyan"], "test": C["amber"]})
fig.update_layout(title=f"{res['split_mode']}: {len(res['tr']):,} train rows, {len(res['te']):,} test rows",
                  xaxis_title="Row order in file")
show(fig, "split")

st.subheader("Test-set performance")
mf = metrics_frame(res)
best = mf["R2"].idxmax()
b1, b2, b3, b4 = st.columns(4)
b1.metric("Best model", best)
b2.metric("R²", f"{mf.loc[best, 'R2']:.4f}")
b3.metric("MAE", f"{mf.loc[best, 'MAE']:.2f}")
b4.metric("RMSE", f"{mf.loc[best, 'RMSE']:.2f}")
st.dataframe(mf.round(4))

l, r = st.columns(2)
with l:
    fig = px.bar(mf.reset_index(names="Model").melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
                 x="Model", y="value", color="variable", barmode="group", title="Error (same units as the target)")
    show(fig, "err")
with r:
    fig = px.bar(mf.reset_index(names="Model"), x="Model", y="R2", color="Model",
                 color_discrete_map=MODEL_COLORS, title="R² on unseen rows")
    fig.update_layout(showlegend=False)
    show(fig, "r2")
