import datetime as dt
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    clinic_id: int
    clinic_name: str


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=60, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=128)
    role: str = "doctor"

    @field_validator("role")
    @classmethod
    def _role(cls, v):
        if v not in ("doctor", "admin"):
            raise ValueError("role must be doctor or admin")
        return v


class PatientIn(BaseModel):
    mrn: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    sex: str
    birth_year: int | None = Field(default=None, ge=1900, le=2100)

    @field_validator("sex")
    @classmethod
    def _sex(cls, v):
        if v not in ("Female", "Male"):
            raise ValueError("sex must be Female or Male")
        return v


class PatientOut(PatientIn):
    id: int
    created_at: dt.datetime


class PredictIn(BaseModel):
    condition: str
    inputs: dict[str, Any]
    # off by default so a doctor's wording is never silently sent to a third party
    use_llm: bool = False


class PredictionOut(BaseModel):
    id: int
    patient_id: int
    condition: str
    probability: float
    band: str
    created_at: dt.datetime
    result: dict[str, Any]
