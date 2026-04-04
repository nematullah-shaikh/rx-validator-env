from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models import Action, Observation, StepResponse
from environment import RxValidatorEnv

app = FastAPI(
    title="Prescription Validator OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to validate "
        "medical prescriptions: drug identification, adult dosage verification, "
        "and pediatric drug interaction checking."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = RxValidatorEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"


@app.get("/")
def health():
    return {"status": "ok", "env": "rx-validator-env", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id", "task1")
    except:
        task_id = "task1"
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    return env.step(action)


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "drug_identification",
                "difficulty": "easy",
                "description": (
                    "Identify whether a drug name is valid, determine its pharmacological class, "
                    "and flag clearly fraudulent or misspelled drug names."
                ),
            },
            {
                "id": "task2",
                "name": "adult_dosage_verification",
                "difficulty": "medium",
                "description": (
                    "Verify whether a prescribed dose is safe, an overdose, or an underdose "
                    "for a given adult patient. Recommend the correct dose range."
                ),
            },
            {
                "id": "task3",
                "name": "pediatric_interaction_check",
                "difficulty": "hard",
                "description": (
                    "Perform full safety review of a pediatric multi-drug prescription: "
                    "catch drug-drug interactions, identify contraindications, and verify "
                    "weight-based pediatric dosing. Flag dangerous prescriptions."
                ),
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
