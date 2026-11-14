import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import CONDITIONS, Condition

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _finish(df: pd.DataFrame, cond: Condition) -> pd.DataFrame:
    # models see numbers as float, binaries as 0/1 float, categories as plain strings (or NaN)
    out = pd.DataFrame(index=df.index)
    for f in cond.features:
        col = df[f.name]
        if f.kind == "number":
            out[f.name] = pd.to_numeric(col, errors="coerce").astype(float)
        elif f.kind == "binary":
            out[f.name] = col.astype(float)
        else:
            out[f.name] = col.astype(object).where(col.notna(), np.nan)
    return out


def load_diabetes(data_dir: Path):
    path = data_dir / "diabetes_prediction_dataset.csv"
    df = pd.read_csv(path)
    n0 = len(df)
    # the file has a few thousand exact duplicate rows. left in, the same patient ends up in both train and test
    df = df.drop_duplicates().reset_index(drop=True)
    # only 18 rows say Other, too few to learn anything and the app only offers two options
    df = df[df["gender"].isin(["Female", "Male"])].reset_index(drop=True)
    notes = [f"dropped {n0 - len(df)} rows (exact duplicates and gender 'Other')"]
    y = df["diabetes"].astype(int)
    return _finish(df, CONDITIONS["diabetes"]), y, notes, path


def load_heart(data_dir: Path):
    path = data_dir / "processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(path, names=cols, na_values="?")
    y = (df["target"] > 0).astype(int)
    df["sex"] = df["sex"].map({0.0: "Female", 1.0: "Male"})
    for c in ["cp", "restecg", "slope", "ca", "thal"]:
        df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    notes = [f"kept all {len(df)} rows, {int(df[['ca', 'thal']].isna().any(axis=1).sum())} have a missing ca/thal and are imputed inside the pipeline"]
    return _finish(df, CONDITIONS["heart"]), y, notes, path


def load_ckd(data_dir: Path):
    path = data_dir / "chronic_kidney_disease_full.arff"
    cond = CONDITIONS["ckd"]
    cols, rows, started, bad = [], [], False, 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            cols.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            started = True
        elif started:
            vals = [v.strip() for v in line.rstrip(",").split(",")]
            if len(vals) != len(cols):
                bad += 1
                continue
            rows.append(vals)
    df = pd.DataFrame(rows, columns=cols).replace({"?": np.nan, "": np.nan})
    df = df[df["class"].isin(["ckd", "notckd"])].reset_index(drop=True)
    y = (df["class"] == "ckd").astype(int)
    for f in cond.features:
        if f.kind == "binary":
            df[f.name] = df[f.name].map({"yes": 1.0, "no": 0.0})
    for c in ["al", "su"]:
        df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(int(float(v))))
    # rows with gaps are kept and filled by the model's imputer, dropping them would lose 242 of 400 patients
    notes = [f"kept all {len(df)} rows; {bad} malformed line(s) skipped; rows with gaps are kept and imputed"]
    return _finish(df, cond), y, notes, path


LOADERS = {"diabetes": load_diabetes, "heart": load_heart, "ckd": load_ckd}


def load(key: str, data_dir: Path):
    X, y, notes, path = LOADERS[key](data_dir)
    return X, y, notes, file_sha256(path)
