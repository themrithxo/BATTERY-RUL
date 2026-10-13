import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data import ensure_data, get_clean
from utils.ml import SPLITS, importances, metrics_frame, permutation, run_training
from utils.theme import C, MODEL_COLORS, insight, page_header, show

ensure_data()
s = st.session_state
page_header("Results and Insights", "Compare the models, see where they fail, and turn predictions into maintenance decisions.")

res = s.results
if res is None:
    insight("No trained models yet. Train them on the **Model Building** page first.", "warn")
    st.stop()
if res.get("prep") != s.prep:
    insight("The cleaning settings changed after these models were trained. Retrain to apply them.", "warn")

mf = metrics_frame(res)
best = mf["R2"].idxmax()
y = res["y_test"]
preds = {n: v["pred_test"] for n, v in res["models"].items()}
feats = res["features"]
unit = "cycles"

tabs = st.tabs(["Scorecard", "Predictions", "Error analysis", "What drives RUL", "Leakage check", "Business insights"])

with tabs[0]:
    st.dataframe(mf.round(4))
    d = mf.reset_index(names="Model").melt(id_vars="Model", value_vars=["Train R2", "R2"])
    d["variable"] = d["variable"].replace({"R2": "Test R2"})
    fig = px.bar(d, x="Model", y="value", color="variable", barmode="group",
                 color_discrete_sequence=[C["violet"], C["cyan"]], title="Train versus test R² (a wide gap means overfitting)")
    show(fig, "gap")
    insight(f"**{best}** scores highest on unseen cycle blocks (R² {mf.loc[best, 'R2']:.4f}, MAE {mf.loc[best, 'MAE']:.2f}).", "good")

with tabs[1]:
    m = st.selectbox("Model", list(preds), index=list(preds).index(best))
    pr = preds[m]
    lo, hi = float(min(y.min(), pr.min())), float(max(y.max(), pr.max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=y, y=pr, mode="markers", name="Predictions",
                               marker=dict(color=MODEL_COLORS[m], size=5, opacity=0.5)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=C["muted"], dash="dash")))
    fig.update_layout(title=f"{m}: actual versus predicted", xaxis_title="Actual", yaxis_title="Predicted")
    show(fig, "avp")
    a, b = st.columns(2)
    with a:
        fig = px.histogram(x=y - pr, nbins=50, title="Residuals (actual minus predicted)")
        fig.update_traces(marker_color=MODEL_COLORS[m])
        fig.update_layout(xaxis_title="Residual")
        show(fig, "resid")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(y=y, mode="lines", name="Actual", line=dict(color=C["text"], width=1)))
        fig.add_trace(go.Scattergl(y=pr, mode="lines", name="Predicted", line=dict(color=MODEL_COLORS[m], width=1)))
        fig.update_layout(title="Test rows in order", xaxis_title="Test row")
        show(fig, "line")

with tabs[2]:
    bins = pd.qcut(pd.Series(y), 5, duplicates="drop")
    labels = [f"{iv.left:,.0f} to {iv.right:,.0f}" for iv in bins.cat.categories]
    rows = []
    for n_, pr in preds.items():
        err = pd.Series(np.abs(y - pr)).groupby(bins.cat.codes.to_numpy()).mean()
        rows += [dict(Model=n_, **{"Actual RUL band": labels[int(k)], "MAE": v}) for k, v in err.items()]
    fig = px.bar(pd.DataFrame(rows), x="Actual RUL band", y="MAE", color="Model", barmode="group",
                 color_discrete_map=MODEL_COLORS, title="Error by how much life the battery really has left",
                 category_orders={"Actual RUL band": labels})
    show(fig, "band")
    low_mae = float(np.abs(y - preds[best])[bins.cat.codes.to_numpy() == 0].mean())
    tone = "warn" if low_mae > mf.loc[best, "MAE"] else "good"
    insight(f"For the batteries closest to end of life, {best} has MAE {low_mae:,.1f} versus {mf.loc[best, 'MAE']:,.1f} overall. "
            "This band matters most for safety.", tone)

with tabs[3]:
    imp = pd.DataFrame({n: importances(v["model"], feats) for n, v in res["models"].items()})
    fig = px.bar(imp.reset_index(names="Feature").melt(id_vars="Feature"), x="value", y="Feature", color="variable",
                 orientation="h", barmode="group", color_discrete_map=MODEL_COLORS,
                 title="Relative importance (each model sums to 1)")
    fig.update_layout(xaxis_title="Importance", legend_title=None)
    show(fig, "imp")
    st.caption("Linear Regression uses standardised coefficient size; the tree models use impurity reduction.")
    if st.button("Run permutation importance for the best model"):
        with st.spinner("Shuffling each feature..."):
            pi = permutation(res["models"][best]["model"], res["X_test"], pd.Series(y), res["params"]["seed"])
        fig = px.bar(pi.sort_values(), orientation="h", title=f"Permutation importance ({best}, rise in MAE when shuffled)")
        fig.update_layout(showlegend=False, xaxis_title=f"Increase in MAE ({unit})", yaxis_title=None)
        show(fig, "perm")

with tabs[4]:
    st.markdown("Does a random row split flatter the model? This re-trains Random Forest under all three split strategies.")
    if st.button("Run leakage check"):
        clean, _ = get_clean()
        out = []
        with st.spinner("Training Random Forest three times..."):
            for mode in SPLITS:
                r = run_training(clean, feats, res["target"], mode, res["test_size"], res["n_blocks"],
                                 res["params"], only=["Random Forest"])
                out.append(dict(Split=mode, **{k: r["models"]["Random Forest"]["metrics"][k] for k in ("R2", "MAE", "RMSE")}))
        s.leak = pd.DataFrame(out)
    if "leak" in s:
        lk = s.leak
        st.dataframe(lk.round(4), hide_index=True)
        fig = px.bar(lk, x="Split", y="MAE", color="Split", title="Random Forest MAE by split strategy",
                     color_discrete_sequence=[C["cyan"], C["violet"], C["amber"]])
        fig.update_layout(showlegend=False)
        show(fig, "leak")
        rnd = lk[lk["Split"].str.startswith("Random")].iloc[0]
        blk = lk[lk["Split"].str.startswith("Blocked")].iloc[0]
        if rnd["MAE"] < blk["MAE"]:
            insight(f"Random rows report MAE {rnd['MAE']:,.2f} against {blk['MAE']:,.2f} for held-out blocks. "
                    "The random split looks better because neighbouring cycles are near-copies, so it overstates real-world accuracy.", "warn")
        else:
            insight("Random and blocked splits agree closely on this data, so leakage is not inflating the scores here.", "good")

with tabs[5]:
    r2_gap = mf.loc["Random Forest", "R2"] - mf.loc["Linear Regression", "R2"]
    top_feat = importances(res["models"]["Random Forest"]["model"], feats).idxmax()
    top_share = importances(res["models"]["Random Forest"]["model"], feats).max()
    st.markdown("**Automatic findings**")
    insight(f"**{best}** predicts RUL within about ±{mf.loc[best, 'MAE']:,.0f} {unit} on average, "
            f"{mf.loc[best, 'MAE'] / res['y_mean'] * 100:.1f}% of the average RUL ({res['y_mean']:,.0f}).")
    insight(f"**{top_feat}** is the biggest driver in Random Forest ({top_share * 100:.0f}% of importance). "
            "Monitoring this signal gives the earliest warning.")
    if abs(r2_gap) < 0.02:
        insight("Linear Regression is within 0.02 R² of Random Forest, so a simple, explainable model is good enough here.", "good")
    else:
        insight(f"Random Forest beats Linear Regression by {r2_gap:.3f} R², so the degradation is clearly non-linear "
                "and the extra model complexity pays off.")
    gap = mf.loc["Random Forest", "Overfit gap"]
    if gap > 0.05:
        insight(f"Random Forest fits training data better than unseen data (R² gap {gap:.3f}). "
                "Try a larger minimum leaf size or limited depth.", "warn")

    st.markdown("**Maintenance simulator**")
    st.caption("Flag a battery for replacement when predicted RUL falls at or below the threshold plus a safety margin. "
               "Costs are illustrative; edit them to match your case.")
    model_sel = st.selectbox("Model used for flagging", list(preds), index=list(preds).index(best), key="sim_model")
    pr = preds[model_sel]
    c1, c2, c3, c4 = st.columns(4)
    thr = c1.slider(f"Failure threshold ({unit})", float(np.floor(y.min())), float(np.ceil(np.percentile(y, 60))),
                    float(np.round(np.percentile(y, 15))))
    margin = c2.slider("Safety margin", 0.0, float(max(1.0, y.std())), 0.0)
    cost_miss = c3.number_input("Cost of an unplanned failure", 0, 10_000_000, 1000)
    cost_early = c4.number_input("Cost of an unnecessary early replacement", 0, 10_000_000, 150)
    actual_fail = y <= thr
    flagged = pr <= thr + margin
    tp, fp = int((actual_fail & flagged).sum()), int((~actual_fail & flagged).sum())
    fn, tn = int((actual_fail & ~flagged).sum()), int((~actual_fail & ~flagged).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    cost_model = fn * cost_miss + fp * cost_early
    cost_none = int(actual_fail.sum()) * cost_miss
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failures caught (recall)", "n/a" if np.isnan(recall) else f"{recall * 100:.1f}%")
    m2.metric("Flags that were right (precision)", "n/a" if np.isnan(precision) else f"{precision * 100:.1f}%")
    m3.metric("Cost with the model", f"{cost_model:,.0f}")
    m4.metric("Cost with no model", f"{cost_none:,.0f}")
    saved = cost_none - cost_model
    insight(f"Using {model_sel} to flag batteries {'saves' if saved >= 0 else 'costs an extra'} "
            f"{abs(saved):,.0f} on this test set ({tp} caught, {fn} missed, {fp} false alarms, {tn} correctly left alone).",
            "good" if saved >= 0 else "bad")
