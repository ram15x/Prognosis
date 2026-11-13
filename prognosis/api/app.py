import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db.models import AuditLog, Clinic, Patient, Prediction, User
from ..db.session import make_engine, make_session_factory
from ..ml.predict import InputError, Predictor
from ..ml.spec import CONDITIONS
from ..services import advice, report, security, trends
from . import schemas
from .deps import admin_only, audit, clinic_name, current_user, get_db

log = logging.getLogger("prognosis.api")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or get_settings()
    if not settings.jwt_secret or len(settings.jwt_secret) < 16:
        raise RuntimeError("set PROGNOSIS_JWT_SECRET to a random string of at least 16 characters")

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = make_engine(settings.db_url)
        app.state.session_factory = make_session_factory(app.state.engine)
        _bootstrap(app.state.session_factory, settings)
        yield
        app.state.engine.dispose()

    app = FastAPI(title="Prognosis", version="2.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.predictor = predictor or Predictor(settings.models_dir)
    app.state.throttle = security.LoginThrottle()

    @app.get("/health")
    def health():
        return {"status": "ok", "models": app.state.predictor.available()}

    # ---- auth
    @app.post("/auth/login", response_model=schemas.Token)
    def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
        name = form.username.strip().lower()
        if app.state.throttle.blocked(name):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many failed attempts, try again later")
        user = db.scalar(select(User).where(func.lower(User.username) == name))
        # verify even when the user does not exist so response time does not reveal which usernames are real
        ok = security.verify_password(form.password, user.password_hash if user else security.DUMMY_HASH)
        if not (user and user.active and ok):
            app.state.throttle.fail(name)
            audit(db, None, "login_failed", "user", name)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong username or password")
        app.state.throttle.clear(name)
        audit(db, user, "login", "user", user.id)
        return schemas.Token(access_token=security.make_token(settings.jwt_secret, user.id, settings.jwt_minutes))

    @app.get("/me", response_model=schemas.UserOut)
    def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
        return schemas.UserOut(id=user.id, username=user.username, role=user.role,
                               clinic_id=user.clinic_id, clinic_name=clinic_name(db, user.clinic_id))

    @app.post("/users", response_model=schemas.UserOut, status_code=201)
    def add_user(body: schemas.UserIn, admin: User = Depends(admin_only), db: Session = Depends(get_db)):
        u = User(username=body.username, password_hash=security.hash_password(body.password),
                 role=body.role, clinic_id=admin.clinic_id)
        db.add(u)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, "username already taken")
        audit(db, admin, "user_created", "user", u.id)
        return schemas.UserOut(id=u.id, username=u.username, role=u.role, clinic_id=u.clinic_id,
                               clinic_name=clinic_name(db, u.clinic_id))

    # ---- models
    @app.get("/models")
    def list_models(user: User = Depends(current_user)):
        out = []
        for key in app.state.predictor.available():
            c = app.state.predictor.card(key)
            out.append({"key": key, "title": c["title"], "description": c["description"], "selected": c["selected"],
                        "roc_auc": c["test"]["metrics"]["roc_auc"], "trained_at": c["trained_at"]})
        return out

    @app.get("/models/{key}")
    def model_detail(key: str, user: User = Depends(current_user)):
        if key not in CONDITIONS or key not in app.state.predictor.available():
            raise HTTPException(404, "unknown model")
        cond = CONDITIONS[key]
        return {
            "card": app.state.predictor.card(key),
            "features": [{"name": f.name, "label": f.label, "kind": f.kind, "unit": f.unit, "min": f.min,
                          "max": f.max, "options": list(f.options), "required": f.required, "help": f.help}
                         for f in cond.features],
        }

    # ---- patients
    def _patient(db, user, pid) -> Patient:
        # every patient lookup is scoped to the caller's clinic, other clinics get a plain 404
        p = db.scalar(select(Patient).where(Patient.id == pid, Patient.clinic_id == user.clinic_id))
        if not p:
            raise HTTPException(404, "patient not found")
        return p

    def _p_out(p):
        return schemas.PatientOut(id=p.id, mrn=p.mrn, name=p.name, sex=p.sex, birth_year=p.birth_year, created_at=p.created_at)

    @app.post("/patients", response_model=schemas.PatientOut, status_code=201)
    def add_patient(body: schemas.PatientIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
        p = Patient(clinic_id=user.clinic_id, mrn=body.mrn.strip(), name=body.name.strip(), sex=body.sex, birth_year=body.birth_year)
        db.add(p)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, "a patient with this MRN already exists in your clinic")
        audit(db, user, "patient_created", "patient", p.id)
        return _p_out(p)

    @app.get("/patients", response_model=list[schemas.PatientOut])
    def list_patients(q: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
        stmt = select(Patient).where(Patient.clinic_id == user.clinic_id).order_by(Patient.name)
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where(Patient.name.ilike(like) | Patient.mrn.ilike(like))
        return [_p_out(p) for p in db.scalars(stmt.limit(500))]

    @app.get("/patients/{pid}", response_model=schemas.PatientOut)
    def get_patient(pid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
        p = _patient(db, user, pid)
        audit(db, user, "patient_viewed", "patient", pid)
        return _p_out(p)

    # ---- predictions
    def _pred_out(r: Prediction):
        return schemas.PredictionOut(id=r.id, patient_id=r.patient_id, condition=r.condition, probability=r.probability,
                                     band=r.band, created_at=r.created_at, result=r.result)

    @app.post("/patients/{pid}/predictions", response_model=schemas.PredictionOut, status_code=201)
    def predict(pid: int, body: schemas.PredictIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
        patient = _patient(db, user, pid)
        if body.condition not in app.state.predictor.available():
            raise HTTPException(404, "no trained model for that condition")
        try:
            res = app.state.predictor.predict(body.condition, body.inputs)
        except InputError as e:
            raise HTTPException(422, {"problems": e.problems})
        clean = {c["feature"]: c["value"] for c in res["contributions"] if not c["imputed"]}
        text = None
        if body.use_llm:
            text = advice.llm_summary(settings.llm_api_key, settings.llm_model, body.condition, res["band"], res["probability"], clean)
            res["summary_source"] = "llm" if text else "rules (llm unavailable)"
        res["summary"] = text or advice.rules_summary(body.condition, res["band"], res["probability"], clean)
        res.setdefault("summary_source", "rules")
        row = Prediction(patient_id=patient.id, clinic_id=user.clinic_id, condition=body.condition,
                         probability=res["probability"], band=res["band"], inputs=clean, result=res,
                         model_trained_at=res["model"]["trained_at"], created_by=user.id)
        db.add(row)
        db.commit()
        audit(db, user, "prediction_created", "prediction", row.id, f"{body.condition} {res['band']}")
        return _pred_out(row)

    @app.get("/patients/{pid}/predictions", response_model=list[schemas.PredictionOut])
    def patient_predictions(pid: int, condition: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
        _patient(db, user, pid)
        stmt = select(Prediction).where(Prediction.patient_id == pid, Prediction.clinic_id == user.clinic_id).order_by(Prediction.created_at)
        if condition:
            stmt = stmt.where(Prediction.condition == condition)
        return [_pred_out(r) for r in db.scalars(stmt)]

    @app.get("/patients/{pid}/trend/{condition}")
    def trend(pid: int, condition: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
        _patient(db, user, pid)
        rows = list(db.scalars(select(Prediction).where(Prediction.patient_id == pid, Prediction.condition == condition,
                                                        Prediction.clinic_id == user.clinic_id).order_by(Prediction.created_at)))
        pts = [{"at": r.created_at, "probability": r.probability, "band": r.band} for r in rows]
        return {"points": [{"at": p["at"].isoformat(), "probability": p["probability"], "band": p["band"]} for p in pts],
                "summary": trends.summarise(pts)}

    @app.get("/predictions/{prid}/report.pdf")
    def pdf(prid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
        r = db.scalar(select(Prediction).where(Prediction.id == prid, Prediction.clinic_id == user.clinic_id))
        if not r:
            raise HTTPException(404, "prediction not found")
        p = db.get(Patient, r.patient_id)
        pred = dict(r.result, created_at=r.created_at.strftime("%Y-%m-%d %H:%M UTC"), band=r.band, probability=r.probability)
        card = app.state.predictor.card(r.condition)
        data = report.build_pdf({"name": p.name, "mrn": p.mrn, "sex": p.sex}, pred, r.result.get("summary", ""), card)
        audit(db, user, "report_downloaded", "prediction", prid)
        return Response(data, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{r.condition}_report_{prid}.pdf"'})

    # ---- clinic dashboard
    @app.get("/dashboard")
    def dashboard(user: User = Depends(current_user), db: Session = Depends(get_db)):
        # newest prediction per patient and condition. ids only go up so max(id) is the latest
        latest = (select(func.max(Prediction.id)).where(Prediction.clinic_id == user.clinic_id)
                  .group_by(Prediction.patient_id, Prediction.condition))
        rows = db.execute(select(Prediction, Patient).join(Patient, Patient.id == Prediction.patient_id)
                          .where(Prediction.id.in_(latest)).order_by(Prediction.probability.desc())).all()
        items = [{"prediction_id": r.id, "patient_id": p.id, "patient": p.name, "mrn": p.mrn, "condition": r.condition,
                  "probability": r.probability, "band": r.band, "at": r.created_at.isoformat()} for r, p in rows]
        counts = {}
        for it in items:
            counts.setdefault(it["condition"], {"low": 0, "moderate": 0, "high": 0})[it["band"]] += 1
        n_pat = db.scalar(select(func.count()).select_from(Patient).where(Patient.clinic_id == user.clinic_id))
        return {"patients": n_pat, "assessments": len(items), "counts": counts, "items": items}

    @app.get("/audit")
    def audit_log(limit: int = 100, admin: User = Depends(admin_only), db: Session = Depends(get_db)):
        rows = db.scalars(select(AuditLog).where(AuditLog.clinic_id == admin.clinic_id).order_by(AuditLog.id.desc()).limit(min(limit, 500)))
        return [{"at": r.at.isoformat(), "user_id": r.user_id, "action": r.action, "entity": r.entity,
                 "entity_id": r.entity_id, "detail": r.detail} for r in rows]

    return app


def _bootstrap(factory, settings: Settings):
    # no built-in doctor/doctor123 any more. the first admin comes from env vars or the cli
    with factory() as db:
        if db.scalar(select(func.count()).select_from(User)) == 0:
            if settings.bootstrap_user and settings.bootstrap_password:
                clinic = db.scalar(select(Clinic)) or Clinic(name="Main Clinic")
                db.add(clinic)
                db.flush()
                db.add(User(username=settings.bootstrap_user, password_hash=security.hash_password(settings.bootstrap_password),
                            role="admin", clinic_id=clinic.id))
                db.commit()
                log.info("created first admin '%s'", settings.bootstrap_user)
            else:
                log.warning("no users yet. run: python -m prognosis.cli create-admin")


def app_factory() -> FastAPI:
    # `uvicorn prognosis.api.app:app_factory --factory`
    return create_app()
