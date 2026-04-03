from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Patient(BaseModel):
    age_years: int
    weight_kg: float
    sex: str  # "male" / "female"
    is_pediatric: bool


class Prescription(BaseModel):
    drug_name: str
    dose_mg: float
    frequency: str       # e.g. "q8h", "q12h", "once_daily"
    route: str           # e.g. "oral", "IV"
    duration_days: int


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    patient: Patient
    prescriptions: List[Prescription]
    instruction: str
    step_count: int


class Action(BaseModel):
    drug_name: str
    is_valid: bool
    drug_class: Optional[str] = None
    verdict: Optional[str] = None          # "safe" | "overdose" | "underdose"
    recommended_dose_mg: Optional[float] = None
    flag_dangerous: bool = False
    interactions_found: Optional[List[str]] = None
    reason: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
