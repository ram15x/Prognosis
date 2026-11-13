from dataclasses import dataclass

# one place that says what each condition model takes as input.
# the trainer, the API validation and the UI forms all read from here, so they cannot drift apart.


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str  # number | binary | category
    unit: str = ""
    # hard limits, anything outside is a typo and gets rejected
    min: float | None = None
    max: float | None = None
    options: tuple = ()
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class Condition:
    key: str
    title: str
    target: str
    features: tuple
    description: str = ""
    # what the label means, used verbatim in the model card
    label_note: str = ""

    @property
    def names(self):
        return [f.name for f in self.features]

    def by_name(self, name):
        return next(f for f in self.features if f.name == name)


YESNO = ("no", "yes")

DIABETES = Condition(
    key="diabetes",
    title="Diabetes",
    target="diabetes",
    description="Screening risk from routine measurements and two blood tests.",
    label_note="diabetes flag in the Kaggle diabetes prediction dataset (100k rows, synthetic-looking, single source).",
    features=(
        Feature("age", "Age", "number", "years", 0, 120, required=True),
        Feature("gender", "Sex", "category", options=("Female", "Male"), required=True),
        Feature("hypertension", "Hypertension", "binary", options=YESNO, required=True),
        Feature("heart_disease", "Heart disease", "binary", options=YESNO, required=True),
        Feature("smoking_history", "Smoking history", "category",
                options=("never", "former", "current", "not current", "ever", "No Info"), required=True),
        Feature("bmi", "BMI", "number", "kg/m2", 10, 100, required=True),
        Feature("HbA1c_level", "HbA1c", "number", "%", 3, 15, help="Leave out if there is no recent HbA1c."),
        Feature("blood_glucose_level", "Blood glucose", "number", "mg/dL", 20, 800),
    ),
)

HEART = Condition(
    key="heart",
    title="Heart disease",
    target="target",
    description="Presence of angiographic heart disease from the Cleveland clinical features.",
    label_note="Cleveland 'num' column collapsed to 0 = none, 1 = any narrowing (original values 1 to 4).",
    features=(
        Feature("age", "Age", "number", "years", 20, 110, required=True),
        Feature("sex", "Sex", "category", options=("Female", "Male"), required=True),
        Feature("cp", "Chest pain type", "category", options=("1", "2", "3", "4"), required=True,
                help="1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic"),
        Feature("trestbps", "Resting blood pressure", "number", "mmHg", 60, 260, required=True),
        Feature("chol", "Cholesterol", "number", "mg/dL", 80, 700, required=True),
        Feature("fbs", "Fasting sugar > 120", "binary", options=YESNO, required=True),
        Feature("restecg", "Resting ECG", "category", options=("0", "1", "2"), required=True,
                help="0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy"),
        Feature("thalach", "Max heart rate", "number", "bpm", 50, 250, required=True),
        Feature("exang", "Exercise induced angina", "binary", options=YESNO, required=True),
        Feature("oldpeak", "ST depression", "number", "mm", -3, 10, required=True),
        Feature("slope", "ST slope", "category", options=("1", "2", "3"), required=True,
                help="1 upsloping, 2 flat, 3 downsloping"),
        Feature("ca", "Major vessels coloured", "category", options=("0", "1", "2", "3"),
                help="Fluoroscopy result, leave empty if not done."),
        Feature("thal", "Thalassemia", "category", options=("3", "6", "7"),
                help="3 normal, 6 fixed defect, 7 reversible defect. Leave empty if not done."),
    ),
)

CKD = Condition(
    key="ckd",
    title="Chronic kidney disease",
    target="class",
    description="Probability of CKD from urine and blood work, most labs optional.",
    label_note="ckd versus notckd in the UCI chronic kidney disease set (about 400 patients, one hospital, 2015).",
    features=(
        Feature("age", "Age", "number", "years", 0, 120, required=True),
        Feature("bp", "Blood pressure (diastolic)", "number", "mmHg", 30, 200),
        Feature("sg", "Urine specific gravity", "number", "", 1.0, 1.04),
        Feature("al", "Albumin (urine dipstick)", "category", options=("0", "1", "2", "3", "4", "5")),
        Feature("su", "Sugar (urine dipstick)", "category", options=("0", "1", "2", "3", "4", "5")),
        Feature("rbc", "Red blood cells (urine)", "category", options=("normal", "abnormal")),
        Feature("pc", "Pus cells", "category", options=("normal", "abnormal")),
        Feature("pcc", "Pus cell clumps", "category", options=("notpresent", "present")),
        Feature("ba", "Bacteria", "category", options=("notpresent", "present")),
        Feature("bgr", "Random blood glucose", "number", "mg/dL", 20, 800),
        Feature("bu", "Blood urea", "number", "mg/dL", 1, 400),
        Feature("sc", "Serum creatinine", "number", "mg/dL", 0.1, 30),
        Feature("sod", "Sodium", "number", "mEq/L", 90, 180),
        Feature("pot", "Potassium", "number", "mEq/L", 1, 12),
        Feature("hemo", "Haemoglobin", "number", "g/dL", 3, 22),
        Feature("pcv", "Packed cell volume", "number", "%", 8, 70),
        Feature("wbcc", "White cell count", "number", "cells/uL", 1000, 60000),
        Feature("rbcc", "Red cell count", "number", "millions/uL", 1.5, 9),
        Feature("htn", "Hypertension", "binary", options=YESNO),
        Feature("dm", "Diabetes mellitus", "binary", options=YESNO),
        Feature("cad", "Coronary artery disease", "binary", options=YESNO),
        Feature("appet", "Appetite", "category", options=("good", "poor")),
        Feature("pe", "Pedal oedema", "binary", options=YESNO),
        Feature("ane", "Anaemia", "binary", options=YESNO),
    ),
)

CONDITIONS = {c.key: c for c in (DIABETES, HEART, CKD)}
