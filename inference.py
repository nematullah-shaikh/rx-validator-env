import os
import json
import requests
from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN")

# REQUIRED
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.5
SYSTEM_PROMPT = "You are a clinical pharmacist AI. Respond in valid JSON only."


# LOGS (STRICT FORMAT)

def log_start(task):
    print(f"[START] task={task} env=rx-validator-env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    done_val = str(done).lower()
    error_val = error if error else "null"

    # 🔥 FINAL FIX HERE
    action_str = str(action)

    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )


# FALLBACK

def get_fallback(task_id):
    if task_id == "task1":
        return {"drug_name": "Paracetamol", "is_valid": True, "drug_class": "analgesic_antipyretic", "flag_dangerous": False, "reason": "Fallback"}
    elif task_id == "task2":
        return {"drug_name": "Paracetamol", "is_valid": True, "drug_class": "analgesic_antipyretic", "verdict": "safe", "recommended_dose_mg": 500, "flag_dangerous": False, "reason": "Fallback"}
    else:
        return {"drug_name": "Paracetamol", "is_valid": True, "verdict": "safe_to_dispense", "flag_dangerous": False, "interactions_found": [], "reason": "Fallback"}


# LLM

def call_llm(obs):
    task_id = obs.get("task_id", "task1")

    try:
        patient = obs.get("patient", {})
        prescriptions = obs.get("prescriptions", [])
        instruction = obs.get("instruction", "")

        pres_text = ""
        for p in prescriptions:
            pres_text += f"\nDrug: {p.get('drug_name')}, Dose: {p.get('dose_mg')}mg"

        patient_text = f"Age: {patient.get('age_years')}, Weight: {patient.get('weight_kg')}kg, Pediatric: {patient.get('is_pediatric')}"

        if task_id == "task1":
            schema = "Return JSON: drug_name, is_valid, drug_class, flag_dangerous, reason"
        elif task_id == "task2":
            schema = "Return JSON: drug_name, is_valid, drug_class, verdict, recommended_dose_mg, flag_dangerous, reason"
        else:
            schema = "Return JSON: drug_name, is_valid, verdict, flag_dangerous, interactions_found, reason"

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

        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            content = content[start:end]

        return json.loads(content)

    except Exception:
        return get_fallback(task_id)


# RUN

def run_task(task_id):
    rewards = []
    final_reward = 0.01

    try:
        log_start(task_id)

        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        obs = resp.json()

        for step_num in range(1, MAX_STEPS + 1):
            action = call_llm(obs)

            try:
                step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
                result = step_resp.json()
            except Exception:
                log_step(step_num, action, 0.01, True, "error")
                break

            reward = result.get("reward", 0.01)
            done = result.get("done", True)

            rewards.append(reward)
            final_reward = reward

            log_step(step_num, action, reward, done, None)

            if done:
                break

            obs = result.get("observation", obs)

    finally:
        success = final_reward >= SUCCESS_THRESHOLD
        log_end(success, len(rewards), rewards)

    return final_reward


# MAIN

def main():
    scores = {}

    for task_id in ["task1", "task2", "task3"]:
        try:
            score = run_task(task_id)
            score = max(0.01, min(0.99, score))
            scores[task_id] = round(score, 4)
        except Exception:
            scores[task_id] = 0.01

    return scores


if __name__ == "__main__":
    main()
