"""SQLite backend: logs dataset uploads, training runs and predictions."""
import datetime as dt
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "battery_backend.db"
TABLES = ("uploads", "runs", "predictions")

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, rows INTEGER, cols INTEGER, target TEXT);
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, dataset TEXT, split_mode TEXT, model TEXT,
    mae REAL, rmse REAL, r2 REAL, fit_seconds REAL, params TEXT);
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model TEXT, kind TEXT, n_rows INTEGER,
    inputs TEXT, predicted_rul REAL);
"""


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _execute(sql, params=()):
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as con:
            con.executescript(SCHEMA)
            con.execute(sql, params)
            con.commit()
        return True
    except Exception:
        return False


def ping():
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as con:
            con.executescript(SCHEMA)
            con.commit()
        return True
    except Exception:
        return False


def log_upload(name, rows, cols, target):
    _execute("INSERT INTO uploads(ts,name,rows,cols,target) VALUES (?,?,?,?,?)",
             (_now(), name, int(rows), int(cols), str(target)))


def log_run(dataset, split_mode, model, mae, rmse, r2, fit_seconds, params):
    _execute("INSERT INTO runs(ts,dataset,split_mode,model,mae,rmse,r2,fit_seconds,params) VALUES (?,?,?,?,?,?,?,?,?)",
             (_now(), dataset, split_mode, model, float(mae), float(rmse), float(r2), float(fit_seconds), str(params)))


def log_prediction(model, kind, n_rows, inputs, predicted):
    _execute("INSERT INTO predictions(ts,model,kind,n_rows,inputs,predicted_rul) VALUES (?,?,?,?,?,?)",
             (_now(), model, kind, int(n_rows), str(inputs), float(predicted)))


def read_table(name):
    if name not in TABLES:
        raise ValueError("unknown table")
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as con:
            con.executescript(SCHEMA)
            return pd.read_sql_query(f"SELECT * FROM {name} ORDER BY id DESC", con)
    except Exception:
        return pd.DataFrame()


def clear_all():
    for t in TABLES:
        _execute(f"DELETE FROM {t}")
