# draws the figures in docs/ from the saved model cards and the saved models (nothing here retrains)
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prognosis.ml import data, registry  # noqa: E402

KEYS = ["diabetes", "heart", "ckd"]
docs = ROOT / "docs"
docs.mkdir(exist_ok=True)
cards = {k: json.loads((ROOT / "models" / k / "card.json").read_text()) for k in KEYS}

# 1. cross-validated comparison
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for j, k in enumerate(KEYS):
    t = cards[k]["cv"]["table"]
    names = [r["model"] for r in t]
    x = np.arange(len(t))
    sel = cards[k]["selected"]
    colors = ["#c62828" if n == sel else "#7a8da8" for n in names]
    lo = min(r["roc_auc"] for r in t) - 0.02
    axes[0, j].bar(x, [r["roc_auc"] for r in t], yerr=[r["roc_auc_sd"] for r in t], color=colors, capsize=3)
    axes[0, j].set_ylim(max(lo, 0.5), min(1.0, max(r["roc_auc"] for r in t) + 0.015))
    axes[0, j].set_title(f"{cards[k]['title']}: CV ROC AUC")
    axes[1, j].bar(x, [r["brier_cal"] for r in t], yerr=[r["brier_cal_se"] for r in t], color=colors, capsize=3)
    axes[1, j].set_title("calibrated Brier (lower is better)")
    for ax in axes[:, j]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
fig.suptitle("Cross-validated model comparison on the training split (red = shipped model)")
fig.tight_layout()
fig.savefig(docs / "model_comparison.png", dpi=130)
plt.close(fig)

# 2. reliability on the untouched test split
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, k in zip(axes, KEYS):
    X, y, _, _ = data.load(k, ROOT / "data" / "raw")
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model, card, _ = registry.load(ROOT / "models", k)
    p = model.predict_proba(Xte)[:, 1]
    # quantile bins so the sparse high-probability end of the rare-disease model is not empty
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 9)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mp = [p[idx == b].mean() for b in range(len(edges) - 1)]
    ob = [yte.values[idx == b].mean() for b in range(len(edges) - 1)]
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.plot(mp, ob, "o-", color="#c62828")
    ax.set_title(f"{card['title']} ({card['selected']}), n={len(yte)}")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed rate")
fig.suptitle("Calibration on the held-out test split (8 equal-count bins)")
fig.tight_layout()
fig.savefig(docs / "calibration.png", dpi=130)
plt.close(fig)
print("wrote docs/model_comparison.png and docs/calibration.png")
