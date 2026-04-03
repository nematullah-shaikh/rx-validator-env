"""
Baseline inference script — Prescription Validator OpenEnv
Required filename: inference.py (placed in project root)

Required environment variables:
  API_BASE_URL  — LLM endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — Hugging Face token (used as API key for HF-hosted models)
  OPENAI_API_KEY — OpenAI API key (fallback)
  ENV_URL       — Deployed HF Space URL (default: http://localhost:7860)
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
API_BASE_URL  = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL       = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
SYSTEM_PROMPT = (
    "You are a highly skilled clinical pharmacist AI. "
    "You validate prescriptions with precision and always respond in valid JSON only."
)

# ── Agent ─────────────────────────────────────────────────────────────────────

def call_llm(obs: dict) -> dict:
    """Send observation to LLM, return Action dict."""

    patient = obs.get("patient", {})
    prescriptions = obs.get("prescriptions", [])
    instruction = obs.get("instruction", "")
    task_id = obs.get("task_id", "task1")

    patient_text = (
        f"Patient: {patient.get('age_years')} years old, "
        f"{patient.get('weight_kg')} kg, {patient.get('sex')}, "
        f"Pediatric: {patient.get('is_pediatric')}"
    )

    if prescriptions:
        pres_lines = []
        for p in prescriptions:
            pres_lines.append(
                f"  Drug: {p['drug_name']}, Dose: {p['dose_mg']}mg, "
                f"Frequency: {p['frequency']}, Route: {p['route']}, "
                f"Duration: {p['duration_days']} days"
            )
        prescriptions_text = "Prescriptions:\n" + "\n".join(pres_lines)
    else:
        prescriptions_text = "No prescriptions provided."

    schema_note = ""
    if task_id == "task1":
        schema_note = (
            "Return JSON with fields: drug_name (string), is_valid (bool), "
            "drug_class (string or null), flag_dangerous (bool), reason (string)."
        )
    elif task_id == "task2":
        schema_note = (
            "Return JSON with fields: drug_name (string), is_valid (bool), "
            "drug_class (string), verdict (one of: safe/overdose/underdose), "
            "recommended_dose_mg (number), flag_dangerous (bool), reason (string)."
        )
    else:
        schema_note = (
            "Return JSON with fields: drug_name (string, use first drug), is_valid (bool), "
            "verdict (safe_to_dispense or do_not_dispense), "
            "flag_dangerous (bool), "
            "interactions_found (list of strings describing each interaction), "
            "reason (string with full clinical justification)."
        )

    prompt = f"""{instruction}

{patient_text}

{prescriptions_text}

{schema_note}

Important rules:
- For pediatric patients (is_pediatric=true), check age-based contraindications (e.g., Aspirin under 12 → Reye's syndrome; Ciprofloxacin under 18 → contraindicated)
- Flag any drug-drug interactions: Warfarin+Aspirin, Warfarin+Ibuprofen, Ibuprofen+Warfarin are HIGH risk
- Dose outside the therapeutic range → verdict overdose/underdose
- flag_dangerous=true whenever there is a contraindication, HIGH interaction, or dose emergency

Respond with JSON only, no markdown, no extra text:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=800,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown fences if present
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
    print(f"  Resetting to {task_id}...")
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    print(f"  Calling LLM...")
    try:
        action = call_llm(obs)
    except Exception as e:
        print(f"  LLM error: {e}")
        return 0.0

    print(f"  Sending action: {json.dumps(action, indent=2)[:300]}...")
    step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    step_resp.raise_for_status()
    result = step_resp.json()

    reward = result.get("reward", 0.0)
    info = result.get("info", {})
    print(f"  Reward: {reward:.4f} | {info.get('grader_feedback', 'no feedback')}")
    return reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Prescription Validator OpenEnv — Baseline Inference")
    print("=" * 60)
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print()

    TASK_IDS = ["task1", "task2", "task3"]
    scores = {}

    for task_id in TASK_IDS:
        print(f"[{task_id.upper()}] Running...")
        try:
            score = run_task(task_id)
            scores[task_id] = round(score, 4)
        except Exception as e:
            print(f"  ERROR: {e}")
            scores[task_id] = 0.0
        print()

    print("=" * 60)
    print("  BASELINE SCORES")
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
