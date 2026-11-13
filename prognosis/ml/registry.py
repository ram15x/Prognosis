import json
from pathlib import Path

import joblib
import pandas as pd


# a model is a folder: model.joblib, card.json (everything we know about it) and background.csv (for explanations)
def save(models_dir: Path, key: str, model, card: dict, background: pd.DataFrame):
    d = Path(models_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib", compress=3)
    (d / "card.json").write_text(json.dumps(card, indent=2))
    background.to_csv(d / "background.csv", index=False)


def load(models_dir: Path, key: str):
    d = Path(models_dir) / key
    if not (d / "model.joblib").exists():
        raise FileNotFoundError(f"no trained model for '{key}' in {d}. run: python -m prognosis.ml.train")
    model = joblib.load(d / "model.joblib")
    card = json.loads((d / "card.json").read_text())
    background = pd.read_csv(d / "background.csv")
    return model, card, background


def available(models_dir: Path):
    d = Path(models_dir)
    return sorted(p.name for p in d.iterdir() if (p / "model.joblib").exists()) if d.exists() else []
