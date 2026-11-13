import logging

import httpx

log = logging.getLogger(__name__)

# plain, widely used reference cut-offs. these are pointers for the clinician, not diagnoses.
# each rule only looks at values that were actually entered


def findings(condition: str, inputs: dict) -> list[str]:
    g = inputs.get
    out = []
    if condition == "diabetes":
        a1c, glu, bmi = g("HbA1c_level"), g("blood_glucose_level"), g("bmi")
        if a1c is not None:
            if a1c >= 6.5:
                out.append(f"HbA1c {a1c:g}% is in the diabetes range (6.5% or higher).")
            elif a1c >= 5.7:
                out.append(f"HbA1c {a1c:g}% is in the prediabetes range (5.7 to 6.4%).")
        if glu is not None and glu >= 200:
            out.append(f"Blood glucose {glu:g} mg/dL is at or above 200, which is diagnostic if it is a random reading with symptoms.")
        if bmi is not None and bmi >= 30:
            out.append(f"BMI {bmi:g} is in the obese range.")
        if str(g("hypertension")).lower() in ("1", "1.0", "yes", "true"):
            out.append("Hypertension is recorded.")
    elif condition == "heart":
        bp, chol, hr, op = g("trestbps"), g("chol"), g("thalach"), g("oldpeak")
        if bp is not None and bp >= 140:
            out.append(f"Resting blood pressure {bp:g} mmHg is high.")
        if chol is not None and chol >= 240:
            out.append(f"Cholesterol {chol:g} mg/dL is high (240 or above).")
        if op is not None and op >= 2:
            out.append(f"ST depression of {op:g} mm on exercise is marked.")
        if str(g("exang")).lower() in ("1", "1.0", "yes", "true"):
            out.append("Exercise induced angina is recorded.")
        if str(g("cp")) == "4":
            out.append("Chest pain type is asymptomatic, which in this dataset is the group with the most confirmed disease.")
    elif condition == "ckd":
        sc, hemo, al, bp, bu = g("sc"), g("hemo"), g("al"), g("bp"), g("bu")
        if sc is not None and sc > 1.3:
            out.append(f"Serum creatinine {sc:g} mg/dL is above the usual adult upper limit (about 1.2 to 1.3).")
        if bu is not None and bu > 50:
            out.append(f"Blood urea {bu:g} mg/dL is raised.")
        if hemo is not None and hemo < 12:
            out.append(f"Haemoglobin {hemo:g} g/dL is low, anaemia is common in CKD.")
        if al is not None and str(al) not in ("0", "0.0"):
            out.append(f"Urine albumin dipstick is {al}+, persistent albuminuria is a CKD marker.")
        if bp is not None and bp >= 90:
            out.append(f"Diastolic pressure {bp:g} mmHg is high.")
    return out


FOLLOW_UP = {
    "diabetes": {
        "low": "Routine screening interval. Lifestyle advice as usual.",
        "moderate": "Consider a confirmatory fasting glucose or HbA1c and review weight, diet and activity.",
        "high": "Confirmatory laboratory testing is warranted. If confirmed, start the usual diabetes care pathway.",
    },
    "heart": {
        "low": "No further cardiac work-up is suggested by the numbers alone. Manage risk factors as usual.",
        "moderate": "Consider further assessment such as a stress test, guided by symptoms.",
        "high": "Cardiology review and non-invasive or invasive imaging should be considered.",
    },
    "ckd": {
        "low": "No kidney concern from these results. Repeat labs at the usual interval.",
        "moderate": "Repeat creatinine with eGFR and a urine albumin to creatinine ratio to confirm.",
        "high": "Confirm with eGFR and urine ACR over three months and consider nephrology referral.",
    },
}


def pct(p: float) -> str:
    # 0.9996 should not read as a flat 100%
    if p >= 0.995:
        return "over 99%"
    if p < 0.005:
        return "under 1%"
    return f"{p:.0%}"


def rules_summary(condition: str, band: str, probability: float, inputs: dict) -> str:
    notes = findings(condition, inputs)
    head = f"Model estimate: {pct(probability)} ({band} risk band)."
    body = " ".join(notes) if notes else "None of the entered values crossed the common reference cut-offs on their own."
    return f"{head} {body} Suggested next step: {FOLLOW_UP[condition][band]}"


def llm_summary(api_key: str, model: str, condition: str, band: str, probability: float,
                inputs: dict, base_url: str = "https://openrouter.ai/api/v1", transport=None) -> str | None:
    """Optional rewrite of the rules summary in softer wording. Only clinical values go out, never the
    patient's name or id. Returns None on any failure so the caller falls back to the rules text."""
    if not api_key:
        return None
    facts = "; ".join(f"{k}={v}" for k, v in inputs.items() if v is not None)
    prompt = (
        "You help a clinician read a risk-model output. Using ONLY the facts below, write at most 4 sentences "
        "in plain language. Do not add diagnoses, drug names, doses or numbers that are not given. "
        "Say it is decision support and not a diagnosis.\n"
        f"Condition: {condition}\nModel estimate: {pct(probability)} ({band} band)\nValues: {facts}\n"
        f"Reference notes: {' '.join(findings(condition, inputs)) or 'none'}"
    )
    try:
        with httpx.Client(timeout=15, transport=transport) as c:
            r = c.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                       json={"model": model, "temperature": 0.2, "max_tokens": 220,
                             "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("llm summary failed: %s", type(e).__name__)
        return None
