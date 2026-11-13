import pytest

from prognosis.api.app import create_app
from prognosis.services import security
from tests.conftest import login

HEART = dict(age=60, sex="Male", cp=4, trestbps=150, chol=260, fbs="no", restecg=0,
             thalach=120, exang="yes", oldpeak=2.5, slope=2, ca=2, thal=7)


def make_patient(client, auth, mrn="P1", name="Asha Rao"):
    r = client.post("/patients", json={"mrn": mrn, "name": name, "sex": "Female", "birth_year": 1970}, headers=auth)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_refuses_to_start_without_a_secret(settings):
    settings.jwt_secret = ""
    with pytest.raises(RuntimeError):
        create_app(settings)


def test_passwords_are_hashed_not_stored(client, settings):
    from sqlalchemy import create_engine, text
    e = create_engine(settings.db_url)
    with e.connect() as c:
        h = c.execute(text("select password_hash from users")).scalar()
    assert h.startswith("$argon2") and "correct horse" not in h


def test_login_wrong_password_and_lockout(client):
    for _ in range(5):
        assert client.post("/auth/login", data={"username": "admin1", "password": "nope"}).status_code == 401
    # sixth attempt is blocked even with the right password
    assert client.post("/auth/login", data={"username": "admin1", "password": "correct horse battery"}).status_code == 429


def test_endpoints_need_a_token(client):
    for path in ("/me", "/patients", "/dashboard", "/models"):
        assert client.get(path).status_code == 401
    assert client.get("/me", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_expired_token_rejected(client, settings):
    t = security.make_token(settings.jwt_secret, 1, minutes=-1)
    assert client.get("/me", headers={"Authorization": f"Bearer {t}"}).status_code == 401


def test_patient_flow_and_duplicate_mrn(client, auth):
    pid = make_patient(client, auth)
    assert client.post("/patients", json={"mrn": "P1", "name": "Other", "sex": "Male"}, headers=auth).status_code == 409
    assert [p["id"] for p in client.get("/patients?q=asha", headers=auth).json()] == [pid]
    assert client.post("/patients", json={"mrn": "P9", "name": "X", "sex": "Robot"}, headers=auth).status_code == 422


def test_prediction_saved_report_and_trend(client, auth):
    pid = make_patient(client, auth)
    r = client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": HEART}, headers=auth)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["band"] in ("low", "moderate", "high") and body["result"]["summary"]
    assert body["result"]["contributions"]
    # second visit with better numbers
    better = dict(HEART, trestbps=120, chol=190, exang="no", oldpeak=0.2, thalach=165, ca=0, thal=3)
    r2 = client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": better}, headers=auth).json()
    assert r2["probability"] < body["probability"]
    t = client.get(f"/patients/{pid}/trend/heart", headers=auth).json()
    assert len(t["points"]) == 2 and t["summary"]["direction"] == "falling"
    pdf = client.get(f"/predictions/{body['id']}/report.pdf", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    d = client.get("/dashboard", headers=auth).json()
    assert d["assessments"] == 1 and d["items"][0]["prediction_id"] == r2["id"]  # latest per patient+condition


def test_bad_inputs_return_every_problem(client, auth):
    pid = make_patient(client, auth)
    r = client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": {"age": 999}}, headers=auth)
    assert r.status_code == 422 and len(r.json()["detail"]["problems"]) > 3
    assert client.post(f"/patients/{pid}/predictions", json={"condition": "nope", "inputs": {}}, headers=auth).status_code == 404


def test_clinics_cannot_see_each_other(client, auth, settings):
    pid = make_patient(client, auth)
    pr = client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": HEART}, headers=auth).json()
    # a second clinic with its own admin, made straight in the db
    from prognosis.db.models import Clinic, User
    from prognosis.db.session import make_engine, make_session_factory
    f = make_session_factory(make_engine(settings.db_url))
    with f() as db:
        c = Clinic(name="Other Clinic"); db.add(c); db.flush()
        db.add(User(username="other", password_hash=security.hash_password("another long pass"), role="admin", clinic_id=c.id))
        db.commit()
    other = login(client, "other", "another long pass")
    assert client.get(f"/patients/{pid}", headers=other).status_code == 404
    assert client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": HEART}, headers=other).status_code == 404
    assert client.get(f"/predictions/{pr['id']}/report.pdf", headers=other).status_code == 404
    assert client.get(f"/patients/{pid}/trend/heart", headers=other).status_code == 404
    assert client.get("/patients", headers=other).json() == []
    assert client.get("/dashboard", headers=other).json()["assessments"] == 0
    # the same MRN is fine in a different clinic
    assert client.post("/patients", json={"mrn": "P1", "name": "Someone", "sex": "Male"}, headers=other).status_code == 201


def test_only_admin_can_add_users_and_audit_is_written(client, auth):
    r = client.post("/users", json={"username": "dr.rao", "password": "a long password", "role": "doctor"}, headers=auth)
    assert r.status_code == 201
    doc = login(client, "dr.rao", "a long password")
    assert client.post("/users", json={"username": "x" * 5, "password": "a long password"}, headers=doc).status_code == 403
    assert client.get("/audit", headers=doc).status_code == 403
    actions = {a["action"] for a in client.get("/audit", headers=auth).json()}
    assert {"login", "user_created"} <= actions
    assert client.post("/users", json={"username": "weak", "password": "short"}, headers=auth).status_code == 422


def test_llm_off_by_default_and_falls_back(client, auth):
    pid = make_patient(client, auth)
    r = client.post(f"/patients/{pid}/predictions", json={"condition": "heart", "inputs": HEART, "use_llm": True}, headers=auth).json()
    # no key configured in tests, so it must fall back and say so
    assert r["result"]["summary_source"].startswith("rules")
