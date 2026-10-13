"""Visual identity: deep-navy lab console, cyan + violet signal colours."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

C = dict(bg="#0A1020", panel="#111A30", panel2="#16213D", line="#22304F", text="#E6EDF7",
         muted="#8FA1C2", cyan="#22D3EE", violet="#8B5CF6", amber="#FBBF24",
         pink="#F472B6", green="#34D399", red="#F87171")
COLORWAY = [C["cyan"], C["violet"], C["amber"], C["pink"], C["green"], "#FB923C"]
MODEL_COLORS = {"Linear Regression": C["amber"], "Decision Tree": C["violet"], "Random Forest": C["cyan"]}

_axis = dict(gridcolor=C["line"], zerolinecolor=C["line"], linecolor=C["line"])
_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    font=dict(family="Space Grotesk, sans-serif", color=C["text"], size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,26,48,0.65)",
    colorway=COLORWAY, xaxis=_axis, yaxis=_axis,
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=50, b=10),
    title=dict(font=dict(size=16)),
)
pio.templates["battery"] = _tpl
pio.templates.default = "battery"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background-color: #0A1020;
    background-image:
        linear-gradient(rgba(34,211,238,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,211,238,.045) 1px, transparent 1px);
    background-size: 44px 44px;
}
[data-testid="stSidebar"] { background: #0D1428; border-right: 1px solid #22304F; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
.pg-head { margin: 0 0 1.2rem 0; }
.pg-head h1 { font-size: 2.1rem; font-weight: 700; margin: 0; padding: 0 0 0 14px;
              border-left: 4px solid #22D3EE; line-height: 1.15; }
.pg-head p { color: #8FA1C2; margin: .45rem 0 0 18px; font-size: 1rem; }
[data-testid="stMetric"] {
    background: #111A30; border: 1px solid #22304F; border-left: 3px solid #22D3EE;
    border-radius: 6px; padding: 12px 16px;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }
[data-testid="stMetricLabel"] { color: #8FA1C2; }
.card { background: #111A30; border: 1px solid #22304F; border-radius: 8px; padding: 16px 18px; height: 100%; }
.card h4 { margin: 0 0 .4rem 0; color: #22D3EE; font-size: 1.05rem; }
.card p, .card li { color: #C9D6EC; font-size: .95rem; margin: 0; }
.card.violet h4 { color: #8B5CF6; } .card.amber h4 { color: #FBBF24; }
.insight { border-left: 3px solid #22D3EE; background: rgba(34,211,238,.07);
           padding: 10px 14px; border-radius: 4px; margin: .5rem 0; color: #DCE6F7; }
.insight.warn { border-color: #FBBF24; background: rgba(251,191,36,.08); }
.insight.good { border-color: #34D399; background: rgba(52,211,153,.08); }
.insight.bad  { border-color: #F87171; background: rgba(248,113,113,.08); }
.big-number { font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 600; color: #22D3EE; line-height: 1; }
.status-ok { color: #34D399; } .status-bad { color: #F87171; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #22304F; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8FA1C2; padding: 8px 14px; }
.stTabs [aria-selected="true"] { color: #22D3EE !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22D3EE !important; }
button[kind="primary"], button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button {
    background: #22D3EE; color: #06101F; border: 0; font-weight: 600;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #67E8F9; color: #06101F; }
[data-testid="stExpander"] { border: 1px solid #22304F; border-radius: 6px; background: #0F1830; }
hr { border-color: #22304F; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="pg-head"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, body, tone=""):
    st.markdown(f'<div class="card {tone}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def insight(text, tone=""):
    st.markdown(f'<div class="insight {tone}">{text}</div>', unsafe_allow_html=True)


def show(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key)
