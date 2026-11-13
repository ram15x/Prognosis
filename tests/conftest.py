import pytest
from fastapi.testclient import TestClient

from prognosis.api.app import create_app
from prognosis.config import ROOT, Settings
from prognosis.ml import train
from prognosis.ml.predict import Predictor


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    # tiny quick models so the suite stays fast. real metrics live in the shipped model cards
    d = tmp_path_factory.mktemp("models")
    for key, sample in (("diabetes", 6000), ("heart", None), ("ckd", None)):
        train.run(key, d, ROOT / "data" / "raw", quick=True, sample=sample)
    return d


@pytest.fixture()
def settings(tmp_path, models_dir):
    return Settings(db_url=f"sqlite:///{tmp_path / 't.db'}", models_dir=models_dir,
                    jwt_secret="test-secret-0123456789abcdef", bootstrap_user="admin1",
                    bootstrap_password="correct horse battery")


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def login(client, user="admin1", pw="correct horse battery"):
    r = client.post("/auth/login", data={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def auth(client):
    return login(client)


@pytest.fixture(scope="session")
def predictor(models_dir):
    return Predictor(models_dir)
