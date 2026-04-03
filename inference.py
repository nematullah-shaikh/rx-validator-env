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

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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

    return json.loads(content)


# ── Task Runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    log_start(task=task_id, env="rx-validator-env", model=MODEL_NAME)

    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        log_step(1, "reset", 0.00, True, str(e))
        log_end(False, 1, [0.0])
        return 0.0

    rewards = []
    final_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        # Call LLM
        try:
            action = call_llm(obs)
            action_summary = f"validate({action.get('drug_name','?')})"
        except Exception as e:
            log_step(step_num, "llm_call", 0.00, True, str(e))
            log_end(False, step_num, rewards + [0.0])
            return final_reward

        # Send action to environment
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            log_step(step_num, action_summary, 0.00, True, str(e))
            log_end(False, step_num, rewards + [0.0])
            return final_reward

        reward = result.get("reward", 0.0)
        done = result.get("done", True)
        rewards.append(reward)
        final_reward = reward

        log_step(step_num, action_summary, reward, done, None)

        if done:
            break

        obs = result.get("observation", obs)

    success = final_reward >= SUCCESS_THRESHOLD
    log_end(success, len(rewards), rewards)
    return final_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Prescription Validator OpenEnv — Baseline Inference")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Router : {API_BASE_URL}")
    print(f"  Env URL: {ENV_URL}")
    print("=" * 60)

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        print(f"\n--- Running {task_id} ---")
        try:
            scores[task_id] = round(run_task(task_id), 4)
        except Exception as e:
            print(f"ERROR: {e}")
            scores[task_id] = 0.0

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for tid, s in scores.items():
        bar = "█" * int(s * 25)
        print(f"  {tid}: {s:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average: {avg:.4f}")
    print("=" * 60)
    return scores


if __name__ == "__main__":
    main()        
