"""Dataset loading, schema detection, cleaning and shared session state."""
import copy
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import streamlit as st

from utils import db

APP_DIR = Path(__file__).resolve().parent.parent
USER_PATH = r"C:\Users\MRITH\OneDrive\Desktop\ML PROJECT\Battery_RUL.csv\Battery_RUL.csv"
CANDIDATES = [
    APP_DIR / "data" / "Battery_RUL.csv",
    APP_DIR / "Battery_RUL.csv",
    Path(USER_PATH),
    Path(str(PureWindowsPath(USER_PATH).parent)),  # the folder is itself named Battery_RUL.csv
]

PREP_DEFAULTS = {"drop_dupes": True, "missing": "median", "outliers": "keep", "iqr_k": 3.0}
STATE_DEFAULTS = {"raw_df": None, "source": "", "target": None, "cycle_col": None,
                  "prep": PREP_DEFAULTS, "results": None, "upload_sig": None, "full_models": None}
TARGET_HINTS = ("rul", "remaining", "life", "soh", "capacity", "degrad")


def init_state():
    for k, v in STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = copy.deepcopy(v)


def find_default_csv():
    for p in CANDIDATES:
        try:
            if p.is_file():
                return p
            if p.is_dir():
                hits = sorted(p.glob("*.csv"))
                if hits:
                    return hits[0]
        except OSError:
            continue
    return None


def read_table(src, name=None):
    name = str(name or src).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(src)
    else:
        try:
            df = pd.read_csv(src)
        except UnicodeDecodeError:
            if hasattr(src, "seek"):
                src.seek(0)
            df = pd.read_csv(src, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_schema(df):
    num = df.select_dtypes(include="number").columns.tolist()
    target = None
    for hint in TARGET_HINTS:
        target = next((c for c in num if hint in c.lower()), None)
        if target:
            break
    if target is None and num:
        target = num[-1]
    cycle = next((c for c in num if "cycle" in c.lower() and c != target), None)
    return target, cycle


def set_dataset(df, source):
    init_state()
    st.session_state.raw_df = df
    st.session_state.source = source
    st.session_state.target, st.session_state.cycle_col = detect_schema(df)
    st.session_state.results = None
    st.session_state.full_models = None
    db.log_upload(source, len(df), df.shape[1], st.session_state.target)


def ensure_data():
    """Call at the top of every analysis page; auto-loads the default file."""
    init_state()
    if st.session_state.raw_df is None:
        p = find_default_csv()
        if p is not None:
            try:
                set_dataset(read_table(p), p.name)
            except Exception as e:
                st.error(f"Could not read {p}: {e}")
    if st.session_state.raw_df is None or st.session_state.target is None:
        st.warning("No dataset loaded yet. Open **Data Hub** to load Battery_RUL.csv or upload a file.")
        st.stop()


def clean_data(df, target, cycle_col, cfg):
    """Returns (clean numeric frame, list of log lines)."""
    log = []
    out = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    skipped = [c for c in df.columns if c not in out.columns]
    if skipped:
        log.append(f"Ignored {len(skipped)} non-numeric column(s): {', '.join(map(str, skipped))}")
    n_missing_target = int(out[target].isna().sum())
    if n_missing_target:
        out = out.dropna(subset=[target])
        log.append(f"Dropped {n_missing_target} row(s) with a missing target")
    if cfg["drop_dupes"]:
        n_dup = int(out.duplicated().sum())
        out = out.drop_duplicates()
        log.append(f"Removed {n_dup} duplicate row(s)")
    feats = [c for c in out.columns if c != target]
    n_missing = int(out[feats].isna().sum().sum())
    if cfg["missing"] == "drop rows":
        out = out.dropna(subset=feats)
        log.append(f"Dropped rows containing the {n_missing} missing feature value(s)")
    else:
        fill = out[feats].median() if cfg["missing"] == "median" else out[feats].mean()
        out[feats] = out[feats].fillna(fill)
        log.append(f"Filled {n_missing} missing feature value(s) with the column {cfg['missing']}")
    cols = [c for c in feats if c != cycle_col]
    if cfg["outliers"] != "keep" and cols:
        q1, q3 = out[cols].quantile(0.25), out[cols].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - cfg["iqr_k"] * iqr, q3 + cfg["iqr_k"] * iqr
        flagged = (out[cols] < lo) | (out[cols] > hi)
        if cfg["outliers"].startswith("clip"):
            log.append(f"Clipped {int(flagged.sum().sum())} outlying value(s) (IQR x {cfg['iqr_k']})")
            out[cols] = out[cols].clip(lo, hi, axis=1)
        else:
            n_rows = int(flagged.any(axis=1).sum())
            out = out[~flagged.any(axis=1)]
            log.append(f"Dropped {n_rows} row(s) containing outliers (IQR x {cfg['iqr_k']})")
    const = [c for c in out.columns if c != target and out[c].nunique() <= 1]
    if const:
        out = out.drop(columns=const)
        log.append(f"Dropped constant column(s): {', '.join(map(str, const))}")
    return out.reset_index(drop=True), log


def get_clean():
    s = st.session_state
    return clean_data(s.raw_df, s.target, s.cycle_col, s.prep)
