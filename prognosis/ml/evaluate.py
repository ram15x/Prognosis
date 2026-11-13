import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def ece(y, p, bins=10):
    # expected calibration error with equal-width bins
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece(y, p),
    }


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate_oof(y, p, seed=0):
    """Try sigmoid and isotonic on out-of-fold scores, judged by cross-validation.

    Returns the method with the lower Brier score ("none" if neither beats the raw scores)
    and the calibrated out-of-fold probabilities for that method.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = ("none", p, brier_score_loss(y, p))
    sig = cross_val_predict(LogisticRegression(C=1e6, max_iter=1000), _logit(p), y,
                            cv=skf, method="predict_proba")[:, 1]
    iso = np.zeros_like(p, dtype=float)
    for tr, te in skf.split(p, y):
        iso[te] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr]).predict(p[te])
    for name, cal in (("sigmoid", sig), ("isotonic", iso)):
        b = brier_score_loss(y, np.clip(cal, 1e-6, 1 - 1e-6))
        if b < best[2] - 1e-5:
            best = (name, cal, b)
    return best[0], np.clip(best[1], 1e-6, 1 - 1e-6)


def bootstrap_ci(y, p, fn, n=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def pick_thresholds(y, p, sens=0.95, spec=0.95, max_low=0.30, min_gap=0.05):
    """Risk band cut points from out-of-fold probabilities.

    low/moderate: the highest cut that still catches `sens` of the true cases.
    moderate/high: the lowest cut where `spec` of the non-cases are below it.
    Two guard rails, because the rules above misbehave when the classes barely overlap:
    a probability above `max_low` is never called "low", and the high cut always sits at
    least `min_gap` above the low cut.
    """
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    low = min(float(np.quantile(pos, 1 - sens)), max_low)
    high = max(float(np.quantile(neg, spec)), low + min_gap)
    return low, min(high, 0.99)


def band_table(y, p, low, high):
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for name, m in (("low", p < low), ("moderate", (p >= low) & (p < high)), ("high", p >= high)):
        out[name] = {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "observed_rate": float(y[m].mean()) if m.any() else None,
        }
    return out


def operating_point(y, p, cut):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= cut
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    return {
        "cut": float(cut),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
