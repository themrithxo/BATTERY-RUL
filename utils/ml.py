"""Model building: splits, the three regressors, metrics, importances."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SPLITS = ["Blocked by cycle (recommended)", "Chronological holdout", "Random rows (leaky baseline)"]
DEFAULT_PARAMS = dict(dt_depth=0, dt_leaf=5, rf_trees=200, rf_depth=0, rf_leaf=2, seed=42)


def _depth(v):
    return None if not v else int(v)


def make_split(n, mode, test_size=0.2, n_blocks=20, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * test_size)))
    if mode.startswith("Random"):
        test = np.sort(rng.permutation(n)[:k])
    elif mode.startswith("Chrono"):
        test = idx[n - k:]
    else:
        edges = np.linspace(0, n, n_blocks + 1).astype(int)
        n_test = min(n_blocks - 1, max(1, int(round(n_blocks * test_size))))
        chosen = sorted(rng.choice(n_blocks, size=n_test, replace=False))
        test = np.concatenate([idx[edges[b]:edges[b + 1]] for b in chosen])
    return np.setdiff1d(idx, test), test


def make_models(p):
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=_depth(p["dt_depth"]),
                                               min_samples_leaf=int(p["dt_leaf"]), random_state=p["seed"]),
        "Random Forest": RandomForestRegressor(n_estimators=int(p["rf_trees"]), max_depth=_depth(p["rf_depth"]),
                                               min_samples_leaf=int(p["rf_leaf"]), n_jobs=-1,
                                               random_state=p["seed"]),
    }


def _metrics(y, pred):
    return dict(MAE=mean_absolute_error(y, pred), RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                R2=r2_score(y, pred))


def run_training(clean, features, target, split_mode=SPLITS[0], test_size=0.2, n_blocks=20,
                 params=None, only=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    X, y = clean[features], clean[target]
    tr, te = make_split(len(clean), split_mode, test_size, n_blocks, p["seed"])
    fitted = {}
    for name, model in make_models(p).items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0
        pred_te, pred_tr = model.predict(X.iloc[te]), model.predict(X.iloc[tr])
        m = _metrics(y.iloc[te], pred_te)
        m["Train R2"] = r2_score(y.iloc[tr], pred_tr)
        m["Fit s"] = fit_s
        fitted[name] = dict(model=model, pred_test=pred_te, metrics=m)
    stats = {f: dict(min=float(clean[f].min()), max=float(clean[f].max()), median=float(clean[f].median()),
                     is_int=bool(pd.api.types.is_integer_dtype(clean[f]))) for f in features}
    return dict(features=list(features), target=target, split_mode=split_mode, test_size=test_size,
                n_blocks=n_blocks, params=p, tr=tr, te=te, models=fitted,
                y_test=y.iloc[te].to_numpy(), X_test=X.iloc[te], stats=stats,
                y_max=float(y.max()), y_mean=float(y.mean()))


def metrics_frame(res):
    df = pd.DataFrame({n: v["metrics"] for n, v in res["models"].items()}).T
    df["Overfit gap"] = df["Train R2"] - df["R2"]
    return df[["MAE", "RMSE", "R2", "Train R2", "Overfit gap", "Fit s"]]


def importances(model, features):
    est = model[-1] if hasattr(model, "steps") else model
    raw = np.abs(est.coef_) if hasattr(est, "coef_") else est.feature_importances_
    s = pd.Series(raw, index=features, dtype=float)
    return s / s.sum() if s.sum() else s


def permutation(model, X, y, seed=42):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=-1,
                               scoring="neg_mean_absolute_error")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def refit_full(res, clean):
    X, y = clean[res["features"]], clean[res["target"]]
    return {n: clone(v["model"]).fit(X, y) for n, v in res["models"].items()}
