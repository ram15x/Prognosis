import numpy as np
import pandas as pd
import pytest

from prognosis.config import ROOT
from prognosis.ml import candidates, data, evaluate
from prognosis.ml.explain import shapley
from prognosis.ml.predict import InputError, clean_inputs
from prognosis.ml.spec import CONDITIONS

RAW = ROOT / "data" / "raw"

HEART_OK = dict(age=60, sex="Male", cp=4, trestbps=140, chol=250, fbs="no", restecg=0,
                thalach=130, exang="yes", oldpeak=2.0, slope=2)


def test_loaders_shapes_and_cleaning():
    X, y, notes, _ = data.load("diabetes", RAW)
    assert X.duplicated().sum() < len(X)  # sanity, and the loader dropped exact duplicates
    assert set(X["gender"].unique()) == {"Female", "Male"}
    Xh, yh, _, _ = data.load("heart", RAW)
    assert len(Xh) == 303 and set(yh.unique()) == {0, 1}
    Xc, yc, _, _ = data.load("ckd", RAW)
    # the old notebook kept 158 rows, imputation keeps everything that parses
    assert len(Xc) >= 395 and yc.mean() > 0.5
    assert list(Xc.columns) == CONDITIONS["ckd"].names


def test_diabetes_has_no_duplicate_rows_after_loading():
    X, y, _, _ = data.load("diabetes", RAW)
    both = X.assign(_y=y.values)
    assert both.duplicated().sum() == 0


@pytest.mark.parametrize("name", candidates.ORDER)
def test_every_candidate_fits_and_predicts_with_gaps(name):
    X, y, _, _ = data.load("heart", RAW)
    m = candidates.build(name, CONDITIONS["heart"], 0).fit(X, y)
    Xg = X.head(20).copy()
    Xg.loc[:, "chol"] = np.nan  # a missing numeric
    Xg.loc[:, "ca"] = np.nan  # and a missing category
    p = m.predict_proba(Xg)[:, 1]
    assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all()


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    s = evaluate.scores(y, p)
    assert s["roc_auc"] == pytest.approx(0.75)
    assert s["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert evaluate.ece(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0]) * 0.999) < 0.01


def test_thresholds_keep_requested_sensitivity_and_specificity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = np.clip(rng.normal(0.3 + 0.35 * y, 0.15), 0.001, 0.999)
    low, high = evaluate.pick_thresholds(y, p)
    assert low < high
    assert (p[y == 1] >= low).mean() >= 0.94
    assert (p[y == 0] < high).mean() >= 0.94
    bands = evaluate.band_table(y, p, low, high)
    assert bands["low"]["observed_rate"] < bands["moderate"]["observed_rate"] < bands["high"]["observed_rate"]


def test_shapley_is_exactly_additive_and_finds_the_driving_feature():
    rng = np.random.default_rng(0)
    bg = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60), "c": rng.normal(size=60)})

    def f(d):  # only "a" matters, plus a small interaction
        z = 2.0 * d["a"].to_numpy() + 0.3 * d["b"].to_numpy() * d["c"].to_numpy()
        return 1 / (1 + np.exp(-z))

    x = pd.Series({"a": 1.5, "b": 0.2, "c": -0.4})
    phi, base, fx = shapley(f, x, bg, n_perm=60, seed=3)
    assert phi.sum() == pytest.approx(fx - base, abs=1e-9)
    assert np.argmax(np.abs(phi)) == 0 and phi[0] > 0.1


def test_shapley_handles_string_and_missing_columns():
    bg = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0], "c": np.array(["u", "v", "u", "v"], dtype=object)})
    f = lambda d: np.where(d["c"] == "v", 0.8, 0.2) + 0.0 * d["n"].fillna(0)
    phi, base, fx = shapley(f, pd.Series({"n": np.nan, "c": "v"}), bg, n_perm=10)
    assert phi[1] > phi[0] and phi.sum() == pytest.approx(fx - base)


class TestValidation:
    cond = CONDITIONS["heart"]

    def test_ok_and_optional_gaps_are_reported_as_imputed(self):
        row, imputed, warns = clean_inputs(self.cond, HEART_OK)
        assert row["sex"] == "Male" and row["fbs"] == 0.0 and row["cp"] == "4"
        assert set(imputed) == {"ca", "thal"}

    def test_every_problem_is_listed_together(self):
        bad = dict(HEART_OK, age=500, sex="x", chol="abc")
        del bad["thalach"]
        with pytest.raises(InputError) as e:
            clean_inputs(self.cond, bad)
        assert len(e.value.problems) == 4

    def test_unknown_field_rejected(self):
        with pytest.raises(InputError):
            clean_inputs(self.cond, dict(HEART_OK, favourite_colour="blue"))

    def test_out_of_training_range_warns_but_does_not_fail(self):
        ranges = {"chol": {"min": 126, "max": 564}}
        _, _, warns = clean_inputs(self.cond, dict(HEART_OK, chol=600), ranges)
        assert any("outside the range" in w for w in warns)

    def test_nan_and_inf_rejected(self):
        for v in (float("nan"), float("inf")):
            with pytest.raises(InputError):
                clean_inputs(self.cond, dict(HEART_OK, chol=v))

    def test_binary_accepts_bool_and_strings(self):
        row, *_ = clean_inputs(self.cond, dict(HEART_OK, fbs=True, exang="No"))
        assert row["fbs"] == 1.0 and row["exang"] == 0.0


def test_predictor_end_to_end_and_monotone_sanity(predictor):
    low = predictor.predict("heart", dict(HEART_OK, age=30, sex="Female", cp=1, trestbps=105, chol=170, exang="no",
                                          oldpeak=0.0, thalach=185, slope=1, ca=0, thal=3), n_perm=15)
    high = predictor.predict("heart", dict(HEART_OK, age=67, cp=4, trestbps=160, chol=300, exang="yes", oldpeak=3.5,
                                           thalach=100, slope=2, ca=3, thal=7), n_perm=15)
    assert 0 <= low["probability"] < high["probability"] <= 1
    total = sum(c["effect"] for c in high["contributions"])
    assert total == pytest.approx(high["probability"] - high["baseline"], abs=0.1)  # background sample differs a little
    assert high["band"] in ("moderate", "high")


def test_missing_ckd_labs_still_predict(predictor):
    r = predictor.predict("ckd", {"age": 55, "sc": 5.0, "hemo": 8.0}, explain=False)
    assert r["probability"] > 0.5 and len(r["imputed"]) == len(CONDITIONS["ckd"].features) - 3
