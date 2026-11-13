import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry
from .explain import shapley
from .spec import CONDITIONS, Condition

YES = {"yes", "y", "true", "1", "1.0"}
NO = {"no", "n", "false", "0", "0.0"}


class InputError(ValueError):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and math.isnan(v))


def clean_inputs(cond: Condition, raw: dict, ranges: dict | None = None):
    """Turn what the caller sent into one model-ready row.

    Returns (row, imputed, warnings). Raises InputError listing every problem at once, so the form
    can show them together instead of one per submit.
    """
    problems, warnings, imputed, row = [], [], [], {}
    unknown = sorted(set(raw) - set(cond.names))
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    for f in cond.features:
        v = raw.get(f.name)
        if _missing(v):
            if f.required:
                problems.append(f"{f.label} is required")
            row[f.name] = np.nan
            imputed.append(f.name)
            continue
        if f.kind == "number":
            try:
                x = float(v)
            except (TypeError, ValueError):
                problems.append(f"{f.label} must be a number")
                continue
            if not math.isfinite(x) or (f.min is not None and x < f.min) or (f.max is not None and x > f.max):
                problems.append(f"{f.label} {v} is outside the allowed range {f.min} to {f.max}")
                continue
            r = (ranges or {}).get(f.name)
            if r and (x < r["min"] or x > r["max"]):
                warnings.append(f"{f.label} {x:g} is outside the range the model was trained on ({r['min']:g} to {r['max']:g}), so treat this estimate with extra caution")
            row[f.name] = x
        elif f.kind == "binary":
            s = str(v).strip().lower()
            if s in YES:
                row[f.name] = 1.0
            elif s in NO:
                row[f.name] = 0.0
            else:
                problems.append(f"{f.label} must be yes or no")
        else:
            s = str(int(v)) if isinstance(v, float) and v == int(v) else str(v).strip()
            match = next((o for o in f.options if o.lower() == s.lower()), None)
            if match is None:
                problems.append(f"{f.label} must be one of {', '.join(f.options)}")
            else:
                row[f.name] = match
    if problems:
        raise InputError(problems)
    optional = [f for f in cond.features if not f.required]
    if optional and sum(n in imputed for n in [f.name for f in optional]) > len(optional) / 2:
        warnings.append("more than half of the optional inputs are empty, missing values are filled with typical training values and the estimate is less reliable")
    return row, imputed, warnings


def band_for(p: float, thresholds: dict) -> str:
    if p < thresholds["low_moderate"]:
        return "low"
    if p < thresholds["moderate_high"]:
        return "moderate"
    return "high"


class Predictor:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._cache = {}

    def _get(self, key):
        if key not in CONDITIONS:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = registry.load(self.models_dir, key)
        return self._cache[key]

    def available(self):
        return registry.available(self.models_dir)

    def card(self, key):
        return self._get(key)[1]

    def predict(self, key: str, raw: dict, explain: bool = True, n_perm: int = 40) -> dict:
        model, card, background = self._get(key)
        cond = CONDITIONS[key]
        row, imputed, warnings = clean_inputs(cond, raw, card.get("ranges"))
        frame = pd.DataFrame([row])[cond.names]
        for f in cond.features:
            if f.kind != "category":
                frame[f.name] = frame[f.name].astype(float)
            else:
                frame[f.name] = frame[f.name].astype(object)
        prob = float(model.predict_proba(frame)[0, 1])
        out = {
            "condition": key,
            "title": cond.title,
            "probability": prob,
            "band": band_for(prob, card["thresholds"]),
            "thresholds": card["thresholds"],
            "imputed": imputed,
            "warnings": warnings,
            "model": {
                "name": card["selected"],
                "calibration": card["calibration"],
                "trained_at": card["trained_at"],
                "test_roc_auc": card["test"]["metrics"]["roc_auc"],
            },
        }
        if explain:
            bg = background[cond.names].copy()
            for f in cond.features:
                bg[f.name] = bg[f.name].astype(float) if f.kind != "category" else bg[f.name].astype(object)
            phi, base, fx = shapley(lambda d: model.predict_proba(d)[:, 1], frame.iloc[0], bg, n_perm=n_perm)
            out["baseline"] = base
            out["contributions"] = sorted(
                ({
                    "feature": f.name, "label": f.label,
                    "value": None if f.name in imputed else (row[f.name] if f.kind != "binary" else ("yes" if row[f.name] else "no")),
                    "unit": f.unit if f.kind == "number" else "",
                    "effect": float(e), "imputed": f.name in imputed,
                } for f, e in zip(cond.features, phi)),
                key=lambda d: -abs(d["effect"]))
        return out
