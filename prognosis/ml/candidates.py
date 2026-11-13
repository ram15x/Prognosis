import warnings

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .spec import Condition

# order matters, it is also the "simplest first" order used by the selection rule
ORDER = ["logreg", "hist_gb", "random_forest", "xgboost", "lightgbm", "stack"]


def make_preprocessor(cond: Condition, scale: bool) -> ColumnTransformer:
    num = [f.name for f in cond.features if f.kind == "number"]
    binary = [f.name for f in cond.features if f.kind == "binary"]
    cat = [f.name for f in cond.features if f.kind == "category"]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = []
    if num:
        parts.append(("num", Pipeline(num_steps), num))
    if binary:
        parts.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(parts, remainder="drop")


def _base_estimators(seed: int):
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=3000),
        "hist_gb": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, eval_metric="logloss", n_jobs=2, random_state=seed, verbosity=0),
        "lightgbm": LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, n_jobs=2,
            random_state=seed, verbose=-1),
    }


def build(name: str, cond: Condition, seed: int = 0) -> Pipeline:
    warnings.filterwarnings("ignore", category=UserWarning)
    est = _base_estimators(seed)
    if name == "stack":
        # the hybrid: a linear model plus two different tree learners, blended by a small logistic regression.
        # stacking uses out-of-fold predictions for the blender so it does not just trust the overfit base models
        stack = StackingClassifier(
            estimators=[("lr", est["logreg"]), ("hgb", est["hist_gb"]), ("rf", est["random_forest"])],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            stack_method="predict_proba", cv=5, n_jobs=1, passthrough=False)
        return Pipeline([("prep", make_preprocessor(cond, scale=True)), ("clf", stack)])
    scale = name == "logreg"
    return Pipeline([("prep", make_preprocessor(cond, scale=scale)), ("clf", est[name])])
