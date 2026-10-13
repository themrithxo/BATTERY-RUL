import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import db
from utils.data import ensure_data, get_clean, read_table
from utils.ml import refit_full, run_training
from utils.theme import C, insight, page_header, show

ensure_data()
s = st.session_state
page_header("Predict and Demo", "Estimate remaining useful life for one battery state, or score a whole file.")

clean, _ = get_clean()
if s.results is None:
    insight("No trained models yet.", "warn")
    if st.button("Quick-train with default settings", type="primary"):
        feats0 = [c for c in clean.columns if c != s.target]
        with st.spinner("Training..."):
            r = run_training(clean, feats0, s.target)
        r["prep"], r["dataset"] = dict(s.prep), s.source
        s.results = r
        st.rerun()
    st.stop()

res = s.results
feats, stats = res["features"], res["stats"]
names = list(res["models"])
best = max(names, key=lambda n: res["models"][n]["metrics"]["R2"])

with st.sidebar:
    st.markdown("**Prediction model**")
    model_name = st.selectbox("Model", names, index=names.index(best))
    use_full = st.checkbox("Use models refit on 100% of the data", value=False)
if use_full:
    if s.full_models is None:
        with st.spinner("Refitting on all rows..."):
            s.full_models = refit_full(res, clean)
    models = s.full_models
else:
    models = {n: v["model"] for n, v in res["models"].items()}

tab1, tab2 = st.tabs(["Single battery", "Batch file"])

with tab1:
    st.caption("Sliders start at the dataset median. Move them to describe the battery's current state.")
    cols = st.columns(2)
    vals = {}
    for i, f in enumerate(feats):
        st_ = stats[f]
        box = cols[i % 2]
        if st_["max"] > st_["min"]:
            if st_["is_int"]:
                vals[f] = box.slider(f, int(st_["min"]), int(st_["max"]), int(st_["median"]), key=f"in_{f}")
            else:
                step = (st_["max"] - st_["min"]) / 200
                vals[f] = box.slider(f, st_["min"], st_["max"], st_["median"], step=step, format="%.4g", key=f"in_{f}")
        else:
            vals[f] = box.number_input(f, value=st_["min"], key=f"in_{f}")
    X_in = pd.DataFrame([vals])[feats]
    all_pred = {n: float(m.predict(X_in)[0]) for n, m in models.items()}
    pred = all_pred[model_name]
    frac = max(0.0, min(1.0, pred / res["y_max"])) if res["y_max"] else 0
    status, tone = ("Healthy", "good") if frac > 0.5 else (("Watch closely", "warn") if frac > 0.2 else ("Plan replacement", "bad"))

    l, r = st.columns([2, 3])
    with l:
        st.markdown("**Predicted remaining useful life**")
        st.markdown(f"<div class='big-number'>{pred:,.0f}</div>", unsafe_allow_html=True)
        insight(f"{status}: about {frac * 100:.0f}% of the longest life seen in the data.", tone)
        st.dataframe(pd.DataFrame({"Model": list(all_pred), "Predicted RUL": [round(v, 1) for v in all_pred.values()]}),
                     hide_index=True)
        if st.button("Save this prediction to the backend log"):
            db.log_prediction(model_name, "single", 1, json.dumps(vals), pred)
            st.success("Logged. See Backend Monitor.")
    with r:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred,
            gauge=dict(axis=dict(range=[0, res["y_max"]]), bar=dict(color=C["cyan"]),
                       bgcolor="#0F1830", borderwidth=0,
                       steps=[dict(range=[0, res["y_max"] * 0.2], color="rgba(248,113,113,.35)"),
                              dict(range=[res["y_max"] * 0.2, res["y_max"] * 0.5], color="rgba(251,191,36,.30)"),
                              dict(range=[res["y_max"] * 0.5, res["y_max"]], color="rgba(52,211,153,.25)")])))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=10))
        show(fig, "gauge")

with tab2:
    st.markdown(f"Upload a CSV or Excel file containing these columns: `{', '.join(feats)}`.")
    st.download_button("Download a template file", clean[feats].head(5).to_csv(index=False).encode(),
                       "rul_template.csv", "text/csv")
    up = st.file_uploader("File to score", type=["csv", "xlsx", "xls"], key="batch_up")
    if up is not None:
        try:
            dfb = read_table(up, up.name)
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            st.stop()
        missing = [f for f in feats if f not in dfb.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        else:
            Xb = dfb[feats].apply(pd.to_numeric, errors="coerce")
            for f in feats:
                Xb[f] = Xb[f].fillna(stats[f]["median"])
            out = dfb.copy()
            out["Predicted RUL"] = models[model_name].predict(Xb)
            a, b, c = st.columns(3)
            a.metric("Rows scored", f"{len(out):,}")
            b.metric("Mean predicted RUL", f"{out['Predicted RUL'].mean():,.1f}")
            c.metric("Lowest predicted RUL", f"{out['Predicted RUL'].min():,.1f}")
            st.dataframe(out.head(500))
            fig = go.Figure(go.Histogram(x=out["Predicted RUL"], nbinsx=50, marker_color=C["cyan"]))
            fig.update_layout(title="Distribution of predicted RUL", xaxis_title="Predicted RUL")
            show(fig, "bhist")
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "rul_predictions.csv", "text/csv")
            sig = (up.name, up.size, model_name)
            if s.get("batch_sig") != sig:
                db.log_prediction(model_name, "batch", len(out), f"file:{up.name}", out["Predicted RUL"].mean())
                s.batch_sig = sig
