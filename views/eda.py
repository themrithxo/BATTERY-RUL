import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.data import ensure_data, get_clean
from utils.theme import C, insight, page_header, show

ensure_data()
s = st.session_state
page_header("Preprocessing and EDA", "Check data quality, choose the cleaning rules, then explore how the signals relate to RUL.")

with st.sidebar:
    st.markdown("**Cleaning settings**")
    p = s.prep
    p["drop_dupes"] = st.checkbox("Drop duplicate rows", p["drop_dupes"])
    opts_m = ["median", "mean", "drop rows"]
    p["missing"] = st.selectbox("Missing feature values", opts_m, index=opts_m.index(p["missing"]))
    opts_o = ["keep", "clip (IQR)", "drop rows (IQR)"]
    p["outliers"] = st.selectbox("Outliers in features", opts_o, index=opts_o.index(p["outliers"]))
    p["iqr_k"] = st.slider("IQR multiplier", 1.0, 5.0, float(p["iqr_k"]), 0.5)

raw, target, cycle = s.raw_df, s.target, s.cycle_col
clean, log = get_clean()
feats = [c for c in clean.columns if c != target]
num_raw = raw.select_dtypes(include="number")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (raw)", f"{len(raw):,}")
k2.metric("Rows (clean)", f"{len(clean):,}", delta=f"{len(clean) - len(raw):,}" if len(clean) != len(raw) else None)
k3.metric("Features", len(feats))
k4.metric("Target", target)

tab_q, tab_d, tab_r, tab_c = st.tabs(["Data quality", "Distributions", "Relationships", "Degradation curve"])

with tab_q:
    st.markdown("**What the cleaning rules did**")
    for line in log:
        st.markdown(f"- {line}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Summary statistics (clean data)**")
        st.dataframe(clean.describe().T.round(3))
    with right:
        miss = num_raw.isna().sum()
        if miss.sum() == 0:
            insight("No missing values in the numeric columns.", "good")
        else:
            fig = px.bar(miss[miss > 0], title="Missing values per column")
            fig.update_layout(showlegend=False)
            show(fig, "miss")
        skew = clean[feats].skew().sort_values(key=abs, ascending=False).round(2)
        st.markdown("**Skewness (largest first)**")
        st.dataframe(skew.rename("skew").to_frame())

with tab_d:
    col = st.selectbox("Column", clean.columns.tolist(), index=len(clean.columns) - 1 if target == clean.columns[-1] else 0)
    fig = px.histogram(clean, x=col, nbins=60, marginal="box", title=f"Distribution of {col}")
    show(fig, "hist")
    d = clean[col]
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    n_out = int(((d < q1 - 3 * (q3 - q1)) | (d > q3 + 3 * (q3 - q1))).sum())
    insight(f"Mean {d.mean():,.3f}, median {d.median():,.3f}, skew {d.skew():.2f}. "
            f"{n_out} value(s) lie beyond 3 x IQR.")

with tab_r:
    corr = clean.corr()
    fig = px.imshow(corr, zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, C["violet"]], [0.5, "#0F1830"], [1, C["cyan"]]],
                    title="Correlation matrix")
    show(fig, "corr")
    tc = corr[target].drop(target).sort_values(key=abs, ascending=True)
    fig = px.bar(tc, orientation="h", title=f"Correlation with {target}")
    fig.update_layout(showlegend=False, xaxis_title="Pearson r", yaxis_title=None)
    show(fig, "tcorr")
    strongest = tc.abs().idxmax()
    insight(f"**{strongest}** has the strongest linear relationship with {target} (r = {tc[strongest]:.2f}).")
    x = st.selectbox("Scatter: feature against target", feats)
    sample = clean.sample(min(len(clean), 6000), random_state=1)
    color = cycle if cycle in clean.columns else None
    fig = px.scatter(sample, x=x, y=target, color=color, render_mode="webgl",
                     color_continuous_scale=[C["violet"], C["cyan"]], opacity=0.6)
    show(fig, "scatter")

with tab_c:
    st.caption("Rows are plotted in file order, which keeps each battery's cycles together.")
    x_axis = clean[cycle] if cycle in clean.columns else pd.Series(np.arange(len(clean)), name="row")
    feat = st.selectbox("Overlay a feature", feats)
    win = st.slider("Smoothing window (rows)", 1, 200, 25)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(x=x_axis, y=clean[target], mode="lines", name=target,
                               line=dict(color=C["cyan"], width=1.5)), secondary_y=False)
    fig.add_trace(go.Scattergl(x=x_axis, y=clean[feat].rolling(win, min_periods=1).mean(), mode="lines",
                               name=f"{feat} (smoothed)", line=dict(color=C["amber"], width=1.5)), secondary_y=True)
    fig.update_yaxes(title_text=target, secondary_y=False)
    fig.update_yaxes(title_text=feat, secondary_y=True, showgrid=False)
    fig.update_layout(title="Degradation over the battery's life", xaxis_title=cycle or "row")
    show(fig, "curve")
