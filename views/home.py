import streamlit as st

from utils.data import init_state
from utils.theme import card, insight, page_header

init_state()
page_header("Battery Lifecycle Prediction",
            "Predicting remaining useful life (RUL) from usage, charging and temperature-cycle signals")

st.subheader("Problem identification")
c1, c2, c3 = st.columns(3)
with c1:
    card("Context", "Batteries degrade over charge/discharge cycles. Knowing remaining useful life matters "
                    "for EVs, energy storage and consumer devices, where unexpected failure means safety "
                    "risk and cost.")
with c2:
    card("Problem", "Manual and physics-based degradation monitoring doesn't scale and can't adapt to varied "
                    "usage conditions such as temperature, charge protocol and chemistry.", "violet")
with c3:
    card("Why machine learning", "Regression models learn degradation patterns directly from usage, charging, "
                                 "temperature and cycle data, with no hand-built physical model per battery type.",
         "amber")

st.subheader("Literature survey")
PAPERS = {
    "Model comparisons": [
        (1, "Umayal and Darapaneni, 2023", "Random Forest against other ML models for RUL in an EV battery management system; best accuracy with a shorter prediction cycle."),
        (3, "Preprint, 2025", "Decision Tree, Random Forest, SVR, Gaussian Process and LSTM compared, framed as predictive maintenance."),
        (4, "Preprint, 2024", "Random Forest, Gradient Boosting, Ridge, KNN, DNN and CNN on engineered voltage-time features; Random Forest led."),
        (5, "SN Computer Science, 2026", "Ensemble models explained with LIME and SHAP; Random Forest strongest and interpretable."),
        (6, "Algorithms 18(6), 2025", "Web-based RUL tool; Extra Trees trained in about 16 s versus about 954 s for LSTM."),
    ],
    "Feature engineering": [
        (2, "Applied Sciences 13(13), 2023", "Linear, bagging and random forest regressors on a physical test rig scored with MSE and RMSE."),
        (7, "arXiv:2603.00875", "Data-driven lifetime modelling; PCA reduces predictors before a Random Forest."),
        (8, "Preprint, 2021", "Early-cycle features such as discharge capacity and internal resistance transformed into a more predictive space."),
    ],
    "Datasets and benchmarks": [
        (9, "Tan et al., KDD 2025", "BatteryLife: 16 sub-datasets, 8 battery formats, 59 chemical systems; tests whether popular time-series models transfer."),
        (10, "arXiv:2603.27186", "Hybrid deep learning with temporal augmentation on standard NASA PCoE and CALCE benchmarks."),
    ],
}
KEY = {
    "Model comparisons": "Random Forest repeatedly beats Linear Regression, SVR and even LSTM/CNN on engineered tabular battery features, and trains far faster.",
    "Feature engineering": "Raw voltage and current signals alone are weak. Reduction, early-cycle transforms and indicators like cycle count matter as much as model choice.",
    "Datasets and benchmarks": "Most work uses narrow, small lab datasets (NASA PCoE, CALCE), which limits confidence in generalisation.",
}
tabs = st.tabs(list(PAPERS))
for tab, (theme, rows) in zip(tabs, PAPERS.items()):
    with tab:
        insight(KEY[theme])
        for n, src, what in rows:
            st.markdown(f"**[{n}]** {what}  \n<span style='color:#8FA1C2'>{src}</span>", unsafe_allow_html=True)

st.subheader("Gap analysis")
g1, g2, g3 = st.columns(3)
with g1:
    card("Side-by-side simplicity", "No paper compares Linear, Decision Tree and Random Forest regressors on the "
                                    "same combined feature set to show the accuracy/simplicity trade-off.")
with g2:
    card("Leakage-aware validation", "Most studies use random row-wise splits, which leak information between "
                                     "neighbouring cycles of the same battery. This app holds out whole cycle blocks.",
         "violet")
with g3:
    card("Narrow datasets", "Most data is lab-sourced and narrow. This app accepts any new dataset so the same "
                            "pipeline can be re-run on other batteries.", "amber")

st.subheader("Where each component lives")
st.dataframe(
    {
        "Component": ["Problem identification and literature survey", "User interface and backend connectivity",
                      "Preprocessing and EDA", "Model building", "Result analysis and business insights",
                      "Deployment and demo"],
        "Page": ["Problem and Literature", "Data Hub, Backend Monitor", "Preprocessing and EDA",
                 "Model Building", "Results and Insights", "Predict and Demo"],
    },
    hide_index=True,
)

with st.expander("References"):
    st.markdown("""
1. S. Umayal and N. Darapaneni, "Machine Learning based Remaining Useful Life Prediction of Lithium-ion Batteries in Electric Vehicle Battery Management System," 2023.
2. "Experimental Study on Remaining Useful Life Prediction of Lithium-Ion Batteries Based on Three Regression Models for Electric Vehicle Application," *Applied Sciences*, 13(13), 7660, 2023.
3. "Machine Learning-Based framework for battery Remaining Useful Life Prediction," 2025.
4. "Battery Remaining Useful Life Prediction Using Machine Learning Models: A Comparative Study," 2024.
5. "Interpretable Ensemble Learning to Predict Remaining Useful Life of Li-Ion Battery Using LIME and SHAP," *SN Computer Science*, 2026.
6. "Forecasting the Remaining Useful Life of Lithium-Ion Batteries Using Machine Learning Models—A Web-Based Application," *Algorithms*, 18(6), 303, 2025.
7. "Battery Lifetime Prediction using Data-driven Modeling Approaches," arXiv:2603.00875.
8. "Early prediction of battery lifetime via a machine learning based framework," 2021.
9. R. Tan et al., "BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction," arXiv:2502.18807, KDD 2025.
10. "Hybrid Deep Learning with Temporal Data Augmentation for Accurate Remaining Useful Life Prediction of Lithium-Ion Batteries," arXiv:2603.27186.
""")
