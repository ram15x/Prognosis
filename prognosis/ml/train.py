import argparse
import datetime as dt
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from ..config import get_settings
from . import candidates, data, evaluate, registry
from .spec import CONDITIONS

log = logging.getLogger("train")

SEED = 42
# small datasets get many repeats because one 5-fold split is mostly noise at 300 rows
REPEATS = {"diabetes": 1, "heart": 5, "ckd": 5}


def oof_predictions(model, X, y, repeats, seed=SEED):
    """Out-of-fold probabilities, one vector per repeat, plus the fold index sets."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    P = np.zeros((repeats, len(y)))
    folds = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        P[i // 5, te] = m.predict_proba(X.iloc[te])[:, 1]
        folds.append(te)
    return P, folds


def cv_table(names, cond, X, y, repeats):
    rows, store = [], {}
    for name in names:
        t0 = time.time()
        model = candidates.build(name, cond, SEED)
        P, folds = oof_predictions(model, X, y, repeats)
        method, cal0 = evaluate.calibrate_oof(y, P[0])
        # calibrated brier for every repeat, per fold, so we can get a standard error
        fold_brier, fold_auc, raw_b, raw_e, cal_e = [], [], [], [], []
        for r in range(repeats):
            cal = cal0 if r == 0 else evaluate.calibrate_oof(y, P[r])[1]
            raw_b.append(brier_score_loss(y, P[r]))
            raw_e.append(evaluate.ece(y, P[r]))
            cal_e.append(evaluate.ece(y, cal))
            for te in folds[r * 5:(r + 1) * 5]:
                fold_brier.append(brier_score_loss(y.iloc[te], cal[te]))
                fold_auc.append(roc_auc_score(y.iloc[te], P[r][te]))
        row = {
            "model": name,
            "roc_auc": float(np.mean(fold_auc)),
            "roc_auc_sd": float(np.std(fold_auc)),
            "brier_raw": float(np.mean(raw_b)),
            "brier_cal": float(np.mean(fold_brier)),
            "brier_cal_se": float(np.std(fold_brier) / np.sqrt(len(fold_brier))),
            "ece_raw": float(np.mean(raw_e)),
            "ece_cal": float(np.mean(cal_e)),
            "calibration": method,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        store[name] = (method, cal0)
        log.info("%s %s auc=%.4f brier_cal=%.4f (%s) %.0fs", cond.key, name, row["roc_auc"], row["brier_cal"], method, row["seconds"])
    return rows, store


def pick(rows):
    # one-standard-error rule on calibrated brier: anything within one SE of the best counts as tied,
    # and among the tied models we take the simplest one (candidates.ORDER)
    best = min(rows, key=lambda r: r["brier_cal"])
    limit = best["brier_cal"] + best["brier_cal_se"]
    tied = [r for r in rows if r["brier_cal"] <= limit]
    return min(tied, key=lambda r: candidates.ORDER.index(r["model"])), best


def training_ranges(X, cond):
    out = {}
    for f in cond.features:
        col = X[f.name].dropna()
        if f.kind == "number":
            out[f.name] = {"p1": float(col.quantile(0.01)), "p99": float(col.quantile(0.99)),
                           "min": float(col.min()), "max": float(col.max()), "median": float(col.median())}
        else:
            out[f.name] = {"seen": sorted(map(str, col.unique()))}
        out[f.name]["missing_rate"] = float(X[f.name].isna().mean())
    return out


def run(key, models_dir, data_dir, quick=False, sample=None):
    cond = CONDITIONS[key]
    X, y, notes, sha = data.load(key, data_dir)
    if sample and sample < len(y):
        # only for tests and smoke runs, never for the shipped models
        X, _, y, _ = train_test_split(X, y, train_size=sample, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    log.info("%s: %d rows, prevalence %.3f, train %d test %d", key, len(y), y.mean(), len(ytr), len(yte))

    names = ["logreg", "hist_gb"] if quick else candidates.ORDER
    repeats = 1 if quick else REPEATS[key]
    rows, store = cv_table(names, cond, Xtr, ytr, repeats)
    chosen, best = pick(rows)
    log.info("%s: best brier is %s, chosen (one-SE rule) is %s", key, best["model"], chosen["model"])

    # thresholds from out-of-fold scores of the chosen model, never from the test set
    method, cal_oof = store[chosen["model"]]
    low, high = evaluate.pick_thresholds(ytr.values, cal_oof)

    base = candidates.build(chosen["model"], cond, SEED)
    final = base if method == "none" else CalibratedClassifierCV(base, method=method, cv=5)
    final.fit(Xtr, ytr)

    p = final.predict_proba(Xte)[:, 1]
    test = evaluate.scores(yte, p)
    ci = {
        "roc_auc": evaluate.bootstrap_ci(yte, p, roc_auc_score),
        "brier": evaluate.bootstrap_ci(yte, p, brier_score_loss),
    }

    # honest comparison point: what a plain logistic regression does on the same untouched test set
    ref = candidates.build("logreg", cond, SEED).fit(Xtr, ytr)
    ref_auc = float(roc_auc_score(yte, ref.predict_proba(Xte)[:, 1]))

    imp = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
    importance = sorted(
        ({"feature": c, "auc_drop": float(m), "sd": float(s)} for c, m, s in zip(Xte.columns, imp.importances_mean, imp.importances_std)),
        key=lambda d: -d["auc_drop"])

    extra = {}
    if key == "diabetes" and not quick:
        # how much of the score is just the two lab values? most real screening happens before labs exist
        cols = [c for c in X.columns if c not in ("HbA1c_level", "blood_glucose_level")]
        from dataclasses import replace
        sub = replace(cond, features=tuple(f for f in cond.features if f.name in cols))
        m = candidates.build("hist_gb", sub, SEED).fit(Xtr[cols], ytr)
        extra["without_lab_values"] = {
            "model": "hist_gb", "test_roc_auc": float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1])),
            "note": "same split, HbA1c and blood glucose removed"}

    card = {
        "condition": key,
        "title": cond.title,
        "description": cond.description,
        "label": cond.label_note,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data": {"rows": len(y), "prevalence": float(y.mean()), "sha256_16": sha, "notes": notes,
                 "train_rows": len(ytr), "test_rows": len(yte)},
        "cv": {"folds": "5-fold stratified", "repeats": repeats, "table": rows},
        "selected": chosen["model"],
        "best_by_brier": best["model"],
        "calibration": method,
        "thresholds": {"low_moderate": low, "moderate_high": high,
                       "rule": "low cut keeps 95% sensitivity, high cut keeps 95% specificity (out-of-fold, train split)"},
        "test": {"metrics": test, "ci95": ci, "logreg_roc_auc": ref_auc,
                 "bands": evaluate.band_table(yte, p, low, high),
                 "at_low_cut": evaluate.operating_point(yte, p, low),
                 "at_high_cut": evaluate.operating_point(yte, p, high)},
        "importance": importance,
        "ranges": training_ranges(Xtr, cond),
        "extra": extra,
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "seed": SEED},
    }
    bg = Xtr.sample(min(100, len(Xtr)), random_state=SEED)
    registry.save(models_dir, key, final, card, bg)
    return card


def rethreshold(key, models_dir, data_dir):
    """Recompute risk-band cut points for an already trained model, without retraining or re-running the comparison.

    Same split, same seed, so the out-of-fold scores are the ones the original run saw.
    """
    cond = CONDITIONS[key]
    model, card, _ = registry.load(models_dir, key)
    X, y, _, _ = data.load(key, data_dir)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    base = candidates.build(card["selected"], cond, SEED)
    P, _ = oof_predictions(base, Xtr, ytr, card["cv"]["repeats"])
    method, cal = evaluate.calibrate_oof(ytr, P[0])
    low, high = evaluate.pick_thresholds(ytr.values, cal)
    p = model.predict_proba(Xte)[:, 1]
    card["thresholds"].update({"low_moderate": low, "moderate_high": high})
    card["test"]["bands"] = evaluate.band_table(yte, p, low, high)
    card["test"]["at_low_cut"] = evaluate.operating_point(yte, p, low)
    card["test"]["at_high_cut"] = evaluate.operating_point(yte, p, high)
    card["thresholds"]["rule"] = ("low cut keeps 95% sensitivity but never above 0.30, high cut keeps 95% specificity and sits at "
                                  "least 0.05 above the low cut (out-of-fold, train split)")
    registry.save(models_dir, key, model, card, pd.read_csv(Path(models_dir) / key / "background.csv"))
    return card


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="all", choices=["all", *CONDITIONS])
    ap.add_argument("--rethreshold", action="store_true", help="only redo the risk-band cut points of saved models")
    ap.add_argument("--quick", action="store_true", help="two cheap models, one repeat, for a smoke run")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    s = get_settings()
    keys = list(CONDITIONS) if a.condition == "all" else [a.condition]
    if a.rethreshold:
        for k in keys:
            c = rethreshold(k, a.models_dir or s.models_dir, a.data_dir or s.data_dir)
            log.info("%s thresholds %s bands %s", k, {x: round(v, 3) for x, v in c["thresholds"].items() if x != "rule"},
                     {b: (v["n"], v["observed_rate"]) for b, v in c["test"]["bands"].items()})
        return
    for k in keys:
        card = run(k, a.models_dir or s.models_dir, a.data_dir or s.data_dir, a.quick)
        t = card["test"]["metrics"]
        log.info("%s done: %s auc %.3f brier %.3f", k, card["selected"], t["roc_auc"], t["brier"])


if __name__ == "__main__":
    main()
