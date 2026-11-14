# Notes on the models

- Prognosis is decision support built on public research datasets. It is not a medical device, has not been validated on any real clinic's patients, and isn't for diagnosis or treatment decisions.
- **Diabetes.** Gradient boosting really beats logistic regression here (0.98 against 0.96 AUC, with tiny fold-to-fold spread). The stacked hybrid did not beat its best member, so it isn't shipped. Most of the signal is two lab values: with HbA1c and blood glucose removed, test AUC drops to 0.826. The dataset looks synthetic, since those two values almost define the label, so 0.98 will not carry over to a real clinic.
- **Heart.** Nothing beat logistic regression, and the differences between models sit inside the noise (fold spread about +/- 0.04). The 0.965 test AUC comes from 61 patients and is probably lucky, so the cross-validated 0.89 is the better guide. The risk bands hold about 20 patients each.
- **Kidney.** The data is nearly perfectly separable (AUC about 1.0 for every model), so the comparison can't tell models apart and 1.000 doesn't mean the model would be perfect on new patients. It is a single-hospital set from 2015 with about 400 patients. The moderate band is almost empty because almost nothing sits between the two groups.
- Predicted probabilities reflect the prevalence in each training set (8.8%, 46% and 63%), not the prevalence in the population being screened.
- Explanations describe what the model does, not what causes the disease.
- The language-model rewrite of the advice text is tested against a mock server only. The reference cut-offs in the advice text (HbA1c 6.5%, creatinine, blood pressure and so on) are common textbook values, written as pointers for a clinician.
- Login throttling is in memory and per process. SQLite is fine for one machine, so use Postgres (`PROGNOSIS_DB_URL`) for anything shared. The repo has no HTTPS, so put it behind a proxy.
- One seed for the split and the models. The heart and kidney comparisons could shuffle a little with another seed.
- The Dockerfile and docker-compose.yml have not been run.
