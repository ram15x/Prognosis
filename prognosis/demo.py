"""Fill a database with made-up patients so the UI has something to show. Every name here is fictional."""
import datetime as dt
import random

from sqlalchemy import select

from .config import get_settings
from .db.models import Patient, Prediction, User
from .db.session import make_engine, make_session_factory
from .ml.predict import Predictor
from .services import advice

NAMES = ["Asha Rao", "Vikram Nair", "Meera Iyer", "Karthik Raja", "Divya Menon", "Arun Kumar", "Lakshmi Devi", "Rahul Shah"]


def main():
    s = get_settings()
    factory = make_session_factory(make_engine(s.db_url))
    pred = Predictor(s.models_dir)
    rnd = random.Random(7)
    with factory() as db:
        user = db.scalar(select(User))
        if not user:
            raise SystemExit("create an admin first: python -m prognosis.cli create-admin --username admin")
        for i, name in enumerate(NAMES):
            p = Patient(clinic_id=user.clinic_id, mrn=f"DEMO{i + 1:03d}", name=name, sex="Female" if i % 2 == 0 else "Male", birth_year=1950 + i * 4)
            db.add(p)
            db.flush()
            sick = rnd.random()
            for visit in range(3):
                when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90 * (2 - visit))
                drift = sick + 0.15 * visit * (1 if i % 2 else -1)
                inputs = {"age": 40 + i * 4, "gender": p.sex, "hypertension": "yes" if drift > 0.6 else "no", "heart_disease": "no",
                          "smoking_history": rnd.choice(["never", "former", "current"]), "bmi": round(24 + 10 * drift, 1),
                          "HbA1c_level": round(5.0 + 3.5 * drift, 1), "blood_glucose_level": int(95 + 150 * drift)}
                res = pred.predict("diabetes", inputs, n_perm=15)
                clean = {c["feature"]: c["value"] for c in res["contributions"] if not c["imputed"]}
                res["summary"] = advice.rules_summary("diabetes", res["band"], res["probability"], clean)
                db.add(Prediction(patient_id=p.id, clinic_id=user.clinic_id, condition="diabetes", probability=res["probability"],
                                  band=res["band"], inputs=clean, result=res, model_trained_at=res["model"]["trained_at"],
                                  created_by=user.id, created_at=when))
        db.commit()
    print("demo patients added (fictional)")


if __name__ == "__main__":
    main()
