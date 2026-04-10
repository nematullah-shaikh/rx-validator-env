from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any


class Patient(BaseModel):
    age_years: int
    weight_kg: float
    sex: str
    is_pediatric: bool


class Prescription(BaseModel):
    drug_name: str
    dose_mg: float
    frequency: str
    route: str
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
    verdict: Optional[str] = None
    recommended_dose_mg: Optional[float] = None
    flag_dangerous: bool = False
    interactions_found: Optional[List[str]] = None
    reason: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

    # 🔥 Clamp reward strictly between (0,1)
    @validator("reward")
    def clamp_reward(cls, v):
        return max(0.0001, min(0.9999, float(v)))

    # 🔒 Clamp grader_score inside info
    @validator("info")
    def clamp_info(cls, v):
        if isinstance(v, dict) and "grader_score" in v:
            try:
                score = float(v["grader_score"])
                v["grader_score"] = max(0.0001, min(0.9999, score))
            except:
                v["grader_score"] = 0.5
        return v
