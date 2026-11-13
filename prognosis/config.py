import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# everything the app needs from the environment lives here, nothing else reads os.environ
@dataclass
class Settings:
    db_url: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_DB_URL", f"sqlite:///{ROOT / 'prognosis.db'}"))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("PROGNOSIS_MODELS_DIR", ROOT / "models")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PROGNOSIS_DATA_DIR", ROOT / "data" / "raw")))
    # no default on purpose, if this is missing in production the app should refuse to start
    jwt_secret: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_JWT_SECRET", ""))
    jwt_minutes: int = field(default_factory=lambda: int(os.environ.get("PROGNOSIS_JWT_MINUTES", "60")))
    api_url: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_API_URL", "http://127.0.0.1:8000"))
    # optional, only used for the plain-language summary. rules-based text is used when it is empty
    llm_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_LLM_MODEL", "openai/gpt-4o-mini"))
    # first-run admin, created only when the users table is empty
    bootstrap_user: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_BOOTSTRAP_USER", ""))
    bootstrap_password: str = field(default_factory=lambda: os.environ.get("PROGNOSIS_BOOTSTRAP_PASSWORD", ""))


def get_settings() -> Settings:
    return Settings()
