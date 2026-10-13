<<<<<<< HEAD
# Battery RUL Lab

Streamlit app for 22MDC56 Machine Learning Laboratory: predicts battery remaining useful life (RUL)
with Linear Regression, Decision Tree and Random Forest regressors.

## Run it (Windows)

```
cd "C:\Users\MRITH\OneDrive\Desktop\ML PROJECT\battery_rul_app"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Copy `Battery_RUL.csv` into the `data` folder. The app also tries your original path automatically,
and the Data Hub page lets you upload any other numeric CSV/Excel file.

## Pages (one dashboard per marked component)

| Page | Component |
|---|---|
| Problem and Literature | Problem identification, literature survey, gap analysis |
| Data Hub | Load default data or upload a new dataset, schema detection |
| Preprocessing and EDA | Cleaning rules, quality checks, distributions, correlations, degradation curve |
| Model Building | Three regressors, blocked/chronological/random splits, hyperparameters |
| Results and Insights | Scorecard, error analysis, drivers, leakage check, maintenance cost simulator |
| Predict and Demo | Single-battery prediction with gauge, batch file scoring and download |
| Backend Monitor | SQLite log of uploads, runs and predictions, data-flow diagram |

## Deploy (Streamlit Community Cloud)

1. Put the project (with `data/Battery_RUL.csv`) in a GitHub repository.
2. On share.streamlit.io choose the repo and set the main file to `app.py`.
3. The SQLite log resets when the cloud app restarts; that is expected.

## Project layout

```
app.py            navigation and theme
views/            one file per page
utils/data.py     loading, schema detection, cleaning
utils/ml.py       splits, models, metrics, importances
utils/db.py       SQLite backend
utils/theme.py    colours, CSS, Plotly template
```
=======
# BATTERY-RUL
Battery RUL Prediction using Machine Learning estimates battery lifespan from historical data. Built with Python and Streamlit, it features data preprocessing, EDA, Linear Regression, Decision Tree, Random Forest, model evaluation, batch prediction, interactive dashboards, maintenance cost analysis, and SQLite-based activity monitoring.
>>>>>>> 08bcc1b89262d8493eeed974977aa7dd5926a319
