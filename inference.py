"""
Baseline inference script — Prescription Validator OpenEnv
Filename: inference.py (root directory)

Log format (required by judges):
  [START] task=<task_id> env=rx-validator-env model=<MODEL_NAME>
  [STEP] step=<n> action=<action_summary> reward=<0.00> done=<true/false> error=<null|msg>
  [END] success=<true/false> steps=<n> rewards=<0.00,...>

Required env vars:
  HF_TOKEN      — Hugging Face token (used as API key)
  API_BASE_URL  — LLM endpoint (default: HF router)
  MODEL_NAME    — Model identifier
  ENV_URL       — Deployed HF Space URL
"""

import os
import json
import requests
from typing import Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "dummy-key")
client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

SYSTEM_PROMPT = (
    "You are a highly skilled clinical pharmacist AI. "
    "You validate prescriptions with precision and always respond in valid JSON only."
)

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.5

# ── Logging (required format) ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(obs: dict) -> dict:
    patient = obs.get("patient", {})
    prescriptions = obs.get("prescriptions", [])
    instruction = obs.get("instruction", "")
    task_id = obs.get("task_id", "task1")

    patient_text = (
        f"Patient: {patient.get('age_years')} years old, "
        f"{patient.get('weight_kg')} kg, {patient.get('sex')}, "
        f"Pediatric: {patient.get('is_pediatric')}"
    )

    pres_lines = []
    for p in prescriptions:
        pres_lines.append(
            f"  Drug: {p['drug_name']}, Dose: {p['dose_mg']}mg, "
            f"Frequency: {p['frequency']}, Route: {p['route']}, "
            f"Duration: {p['duration_days']} days"
        )
    prescriptions_text = "Prescriptions:\n" + "\n".join(pres_lines) if pres_lines else "No prescriptions."

    if task_id == "task1":
        schema_note = (
            "Return JSON with: drug_name (string), is_valid (bool), "
            "drug_class (string or null), flag_dangerous (bool), reason (string)."
        )
    elif task_id == "task2":
        schema_note = (
            "Return JSON with: drug_name (string), is_valid (bool), "
            "drug_class (string), verdict (safe/overdose/underdose), "
            "recommended_dose_mg (number), flag_dangerous (bool), reason (string)."
        )
    else:
        schema_note = (
            "Return JSON with: drug_name (string, use first drug), is_valid (bool), "
            "verdict (safe_to_dispense or do_not_dispense), "
            "flag_dangerous (bool), "
            "interactions_found (list of strings describing each interaction), "
            "reason (string with full clinical justification)."
        )

    prompt = f"""{instruction}

{patient_text}

{prescriptions_text}

{schema_note}

Rules:
- Aspirin in children under 12 -> contraindicated (Reye's syndrome) -> flag_dangerous=true
- Ciprofloxacin under 18 -> contraindicated -> flag_dangerous=true
- Warfarin + Aspirin or Ibuprofen -> HIGH bleeding risk -> flag_dangerous=true
- Dose above therapeutic range -> overdose -> flag_dangerous=true if severe

Respond with JSON only, no markdown:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()

    if "```" in content:
        for part in content.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                content = stripped[4:].strip()
                break
            elif stripped.startswith("{"):
                content = stripped
                break

    def call_llm(obs: dict) -> dict:
    try:
        patient = obs.get("patient", {})
        prescriptions = obs.get("prescriptions", [])
        instruction = obs.get("instruction", "")
        task_id = obs.get("task_id", "task1")

        patient_text = (
            f"Patient: {patient.get('age_years')} years old, "
            f"{patient.get('weight_kg')} kg, {patient.get('sex')}, "
            f"Pediatric: {patient.get('is_pediatric')}"
        )

        pres_lines = []
        for p in prescriptions:
            pres_lines.append(
                f"  Drug: {p['drug_name']}, Dose: {p['dose_mg']}mg, "
                f"Frequency: {p['frequency']}, Route: {p['route']}, "
                f"Duration: {p['duration_days']} days"
            )

        prescriptions_text = "Prescriptions:\n" + "\n".join(pres_lines) if pres_lines else "No prescriptions."

        prompt = f"""{instruction}

{patient_text}

{prescriptions_text}

Respond with JSON only."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=600,
        )

        content = response.choices[0].message.content.strip()

        if "```" in content:
            for part in content.split("```"):
                stripped = part.strip()
                if stripped.startswith("json"):
                    content = stripped[4:].strip()
                    break
                elif stripped.startswith("{"):
                    content = stripped
                    break

        return json.loads(content)

    except Exception as e:
        print("LLM ERROR:", e)
        return {
            "drug_name": "Paracetamol",
            "is_valid": True,
            "verdict": "safe",
            "flag_dangerous": False,
            "reason": "Fallback response due to error"
      }
if
