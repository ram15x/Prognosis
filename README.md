# Prognosis

A small clinic tool that estimates the risk of three conditions (diabetes, heart disease, chronic
kidney disease), explains each estimate, keeps a history per patient and produces a PDF report.

This began as a college assignment: three logistic regressions behind a Streamlit page. It was
rebuilt properly. The old version is not included here, but what was wrong with it is listed below,
because it shaped most of the choices.

This is decision support built on public research datasets. It is not a medical device and has not
been validated on any real clinic's patients.

## What was wrong with the first version

- The app padded missing inputs with zeros, and its forms did not match the trained models. The
  diabetes form had 6 fields but the notebook's model takes 8 inputs plus encoders, and the CKD form
  asked for BMI, which is not in the kidney data at all. So the numbers the app showed were not the
  numbers the notebooks measured.
- The kidney notebook dropped every row with a gap and trained on 158 of 400 patients.
- The diabetes file has 3,854 exact duplicate rows, which leaked between train and test.
- One 80/20 split, one model per disease, no calibration. On 60 test rows that says very little.
- Passwords stored in plain text with a built-in `doctor / doctor123`.
- `patient_id` was the primary key across all clinics, so one clinic could collide with another.
- Predictions were saved with `str(dict)`, which cannot be read back safely.
- An API key sat in a plain text file in the project folder. The new version reads it from an environment variable only.

## How it is put together

```
Streamlit UI  --http-->  FastAPI  -->  SQLAlchemy (SQLite, any SQL database works)
                           |
                           +-->  Predictor  -->  models/<condition>/{model.joblib, card.json}
                           +-->  advice (rules text, optional LLM rewrite)  +  PDF report
```

- `prognosis/ml/spec.py` is the single description of every input (label, unit, allowed range,
  options, required or optional). The trainer, the API validation and the UI forms all read it, so
  they cannot disagree the way the old ones did.
- A model is a folder: the fitted pipeline, a `card.json` with everything measured about it, and a
  small background sample used for explanations.
- Inputs are validated before they reach a model: unknown fields, out of range values, NaN and wrong
  options are rejected with all problems listed at once. Empty optional fields are filled by the
  pipeline's own imputer (fitted on training data), not with zeros. Values outside the range seen in
  training produce a warning on the result.
- Passwords are hashed with argon2, logins are throttled, tokens are short-lived JWTs, every query is
  scoped to the caller's clinic (another clinic gets a 404, not a 403), and security-relevant actions
  go to an audit log. There is no default account: the first admin comes from environment variables
  or `python -m prognosis.cli create-admin`.

## The models

For each condition the same procedure runs (`python -m prognosis.ml.train`):

1. Stratified 80/20 split, fixed seed. The 20% is touched once, at the end.
2. Six candidates on the training part with repeated stratified 5-fold CV: logistic regression,
   histogram gradient boosting, random forest, XGBoost, LightGBM, and a stacked hybrid (logistic
   regression + gradient boosting + random forest, blended by a logistic regression on out-of-fold
   predictions).
3. Each candidate's probabilities are checked for calibration; sigmoid or isotonic calibration is
   kept only if it lowers the Brier score.
4. Selection: lowest calibrated Brier score, but anything within one standard error of the best counts
   as tied and the simplest tied model wins. This stops a 300-row dataset from picking a complicated
   model because of noise.
5. Risk bands (low / moderate / high) come from out-of-fold scores: the low cut keeps 95% sensitivity
   (never above 0.30), the high cut keeps 95% specificity. The test set is not used for this.
6. Test metrics with bootstrap confidence intervals, permutation importance, and the observed event
   rate inside each band.

Explanations are Shapley values computed by sampling feature orderings against a 100-row background
(`prognosis/ml/explain.py`), reported in percentage points of probability for the original clinical
inputs, not for one-hot columns. They add up exactly to the prediction minus the background average
(there is a test for that). They describe what the model does, not what causes the disease.

### Results

Cross-validated on the training split (mean over folds, calibrated Brier in the last column):

| Condition | Model | CV ROC AUC | Calibrated Brier |
|---|---|---|---|
| Diabetes (76,902 train rows, 5-fold x1) | logistic regression | 0.962 | 0.0325 |
| | hist gradient boosting | 0.979 | 0.0234 |
| | random forest | 0.971 | 0.0244 |
| | **XGBoost (shipped)** | 0.980 | 0.0232 |
| | LightGBM | 0.980 | 0.0233 |
| | stack (hybrid) | 0.978 | 0.0235 |
| Heart (242 train rows, 5-fold x5) | **logistic regression (shipped)** | 0.893 +/- 0.044 | 0.1281 |
| | hist gradient boosting | 0.872 | 0.1464 |
| | random forest | 0.897 | 0.1302 |
| | XGBoost | 0.869 | 0.1443 |
| | LightGBM | 0.867 | 0.1460 |
| | stack (hybrid) | 0.896 | 0.1305 |
| Kidney (319 train rows, 5-fold x5) | **logistic regression (shipped)** | 0.9997 | 0.0091 |
| | hist gradient boosting | 0.9993 | 0.0103 |
| | random forest | 0.9989 | 0.0105 |
| | XGBoost | 0.9986 | 0.0135 |
| | LightGBM | 0.9994 | 0.0094 |
| | stack (hybrid) | 0.9995 | 0.0083 |

Shipped models on the untouched test split:

| Condition | Test rows | ROC AUC (95% CI) | Brier | Plain logistic regression AUC |
|---|---|---|---|---|
| Diabetes | 19,226 | 0.978 (0.975 to 0.980) | 0.0233 | 0.960 |
| Heart | 61 | 0.965 (0.922 to 0.995) | 0.080 | 0.965 (same model) |
| Kidney | 80 | 1.000 (1.000 to 1.000) | 0.011 | 1.000 (same model) |

Observed event rate inside the risk bands on the test split (a band system should climb):

| Condition | Low | Moderate | High |
|---|---|---|---|
| Diabetes | 0.6% of 15,388 | 10.8% of 1,432 | 60.0% of 2,406 |
| Heart | 0% of 20 | 37% of 19 | 95% of 22 |
| Kidney | 0% of 29 | 0% of 1 | 100% of 50 |

Charts: `docs/model_comparison.png`, `docs/calibration.png`.

### What these numbers do and do not say

- Diabetes: gradient boosting really is better than logistic regression here (0.98 against 0.96, with
  tiny fold-to-fold spread). The stacked hybrid did not beat its best member, so it is not shipped.
  Most of the signal is the two lab values: permutation importance puts HbA1c first and blood glucose
  second, and with both removed the test AUC drops to 0.826. This dataset looks synthetic (HbA1c and
  glucose almost define the label), so 0.98 will not carry over to a real clinic.
- Heart: nothing beat logistic regression, and the differences between models are inside the noise
  (fold spread is +/- 0.04). The 0.965 test AUC comes from 61 patients and is probably lucky; the
  cross-validated 0.89 is the better guide. Do not read much into the band table above, the groups
  have about 20 patients each.
- Kidney: the data is close to perfectly separable (AUC about 1.0 for every model), so the comparison
  cannot tell the models apart, and the "1.000" does not mean the model would be perfect on new
  patients. It is a single-hospital set from 2015 with about 400 patients. The moderate band is almost
  empty because there is almost nothing between the two groups.
- Predicted probabilities reflect the prevalence in each training set (8.8%, 46%, 63%), not the
  prevalence in whatever population is being screened.
- The old notebooks reported ROC AUC 0.963 (diabetes, with duplicates in the data) and 0.950 (heart,
  60 test rows). The diabetes number was close but for the wrong reasons, and neither is comparable
  to the repeated cross-validation here.

## Running it

```
pip install -r requirements.txt
python -m prognosis.ml.train                    # trains all three, about 20 minutes on 2 cores (the stack is the slow part)
python -c "import secrets; print(secrets.token_urlsafe(48))"   # use the output as the JWT secret

export PROGNOSIS_JWT_SECRET=...                 # see .env.example for everything else
python -m prognosis.cli create-admin --username admin
uvicorn prognosis.api.app:app_factory --factory --port 8000
streamlit run prognosis/ui/app.py
```

`python -m prognosis.demo` adds eight clearly fictional patients with three diabetes visits each, so
the dashboard and trend pages have something to show. `python -m prognosis.ml.train --quick` is a
two-model smoke run. `python -m prognosis.ml.train --rethreshold` recomputes only the band cut points
of saved models. `python scripts/make_docs.py` redraws the charts from the saved models.

The datasets are in `data/raw/`: the Kaggle diabetes prediction dataset, the UCI Cleveland heart
disease data and the UCI chronic kidney disease data. Check each source's own terms before reusing them.

Tests: `pip install -r requirements-dev.txt && pytest` (36 tests, about 30 seconds). They cover
loading and cleaning, every candidate model with missing values, metrics, band thresholds, the
Shapley estimator (additivity and finding the real driver), input validation, and the API: auth,
lockout, expired tokens, clinic separation, audit log, PDF output, trends, and the dashboard.

`Dockerfile` and `docker-compose.yml` are included but I could not run Docker where this was built, so
they are untested.

## Limits

- Public datasets, not clinical validation. Not for diagnosis or treatment decisions.
- The optional language-model rewrite is off by default, sends only clinical values (never name or
  MRN) and falls back to the rules text on any failure. It is tested against a mock server, not a
  live one.
- The reference cut-offs in the advice text (HbA1c 6.5%, creatinine, blood pressure and so on) are
  common textbook values, written as pointers for a clinician.
- Login throttling is in memory and per process. SQLite is fine for one machine; use Postgres
  (`PROGNOSIS_DB_URL`) for anything shared. There is no HTTPS in this repo, put it behind a proxy.
- Single seed for the split and the models. The heart and kidney comparisons would shuffle a little
  with another seed.

## Layout

```
prognosis/ml/        spec, data loading, candidates, evaluation, training, explanations, predictor
prognosis/api/       FastAPI app, schemas, dependencies
prognosis/db/        SQLAlchemy models and sessions
prognosis/services/  security, advice text, PDF report, trend summary
prognosis/ui/        Streamlit app and API client
models/              trained models and their cards
data/raw/            the three datasets
tests/  scripts/  docs/  assets/fonts/
```

## License

MIT for the code. The DejaVu fonts in `assets/fonts/` keep their own license file.
