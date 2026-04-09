import os
import json
import requests
from typing import Optional

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY", "") or os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = "You are a clinical pharmacist AI. Respond in valid JSON only."


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


FALLBACK = {
    "drug_name": "Paracetamol",
    "is_valid": True,
    "verdict": "safe",
    "flag_dangerous": False,
    "interactions_found": [],
    "reason": "Fallback response"
}


def call_llm(obs):
    try:
        from openai import OpenAI
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        patient = obs.get("patient", {})
        prescriptions = obs.get("prescriptions", [])
        instruction = obs.get("instruction", "")
        task_id = obs.get("task_id", "task1")

        pres_text = ""
        for p in prescriptions:
            pres_text += f"\nDrug: {p.get('drug_name')}, Dose: {p.get('dose_mg')}mg"

        patient_text = f"Age: {patient.get('age_years')}, Weight: {patient.get('weight_kg')}kg, Pediatric: {patient.get('is_pediatric')}"

        if task_id == "task1":
            schema = "Return JSON: drug_name, is_valid (bool), drug_class, flag_dangerous (bool), reason"
        elif task_id == "task2":
            schema = "Return JSON: drug_name, is_valid (bool), drug_class, verdict (safe/overdose/underdose), recommended_dose_mg, flag_dangerous (bool), reason"
        else:
            schema = "Return JSON: drug_name, is_valid (bool), verdict (safe_to_dispense/do_not_dispense), flag_dangerous (bool), interactions_found (list), reason"

        prompt = f"{instruction}\n{patient_text}\n{pres_text}\n{schema}\nJSON only:"

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        if "```" in content:
            for part in content.split("```"):
                s = part.strip()
                if s.startswith("json"):
                    content = s[4:].strip()
                    break
                elif s.startswith("{"):
                    content = s
                    break

        return json.loads(content)

    except Exception as e:
        print(f"LLM fallback: {e}", flush=True)
    if task_id == "task1":
        return {
        "drug_name": "Paracetamol",
        "is_valid": True,
        "drug_class": "Analgesic",
        "flag_dangerous": False,
        "reason": "Fallback"
    }

    elif task_id == "task2":
        return {
        "drug_name": "Paracetamol",
        "is_valid": True,
        "drug_class": "Analgesic",
        "verdict": "safe",
        "recommended_dose_mg": 500,
        "flag_dangerous": False,
        "reason": "Fallback"
    }

    else:
        return {
        "drug_name": "Paracetamol",
        "is_valid": True,
        "verdict": "safe_to_dispense",
        "flag_dangerous": False,
        "interactions_found": [],
        "reason": "Fallback"
    }


def run_task(task_id):
    log_start(task=task_id, env="rx-validator-env", model=MODEL_NAME)

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
        action = call_llm(obs)
        action_summary = f"validate({action.get('drug_name', '?')})"

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


def main():
    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        try:
            scores[task_id] = round(run_task(task_id), 4)
        except Exception as e:
            print(f"ERROR {task_id}: {e}", flush=True)
            scores[task_id] = 0.0
    return scores


if __name__ == "__main__":
    main()
