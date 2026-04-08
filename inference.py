import os
import json
import requests
from typing import Optional
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "dummy-key")
client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

SYSTEM_PROMPT = (
    "You are a highly skilled clinical pharmacist AI. "
    "You validate prescriptions with precision and always respond in valid JSON only."
)

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.5

# ── Logging ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)

# ── LLM Call (SAFE VERSION) ─────────────────────────────────────────────

def call_llm(obs: dict) -> dict:
    try:
        patient = obs.get("patient", {})
        prescriptions = obs.get("prescriptions", [])
        instruction = obs.get("instruction", "")

        prompt = f"{instruction}\nPatient: {patient}\nPrescriptions: {prescriptions}"

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

        # ✅ fallback (prevents crash → VERY IMPORTANT)
        return {
            "drug_name": "Paracetamol",
            "is_valid": True,
            "verdict": "safe",
            "flag_dangerous": False,
            "reason": "Fallback response due to error"
        }

# ── Task Runner ─────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
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
        try:
            action = call_llm(obs)
            action_summary = f"validate({action.get('drug_name','?')})"
        except Exception as e:
            log_step(step_num, "llm_call", 0.00, True, str(e))
            log_end(False, step_num, rewards + [0.0])
            return final_reward

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

# ── Main ────────────────────────────────────────────────────────────────

def main():
    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        try:
            scores[task_id] = round(run_task(task_id), 4)
        except Exception:
            scores[task_id] = 0.0

    return scores

if __name__ == "__main__":
    main()
