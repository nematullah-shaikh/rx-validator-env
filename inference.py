import os
import json
import requests
from openai import OpenAI

# ================= ENV =================
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

API_KEY = os.environ.get("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.5


# ================= LOG =================
def log_start(task):
    print(f"[START] task={task} env=rx-validator-env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    reward = max(0.0001, min(0.9999, reward))
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True
    )

def log_end(success, steps, rewards):
    clamped = [max(0.0001, min(0.9999, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )



# ================= FALLBACK =================
def fallback_action(task_id):
    if task_id == "task1":
        return {
            "drug_name": "Paracetamol",
            "is_valid": True,
            "drug_class": "analgesic_antipyretic",
            "flag_dangerous": False,
            "reason": "fallback"
        }
    elif task_id == "task2":
        return {
            "drug_name": "Paracetamol",
            "is_valid": True,
            "drug_class": "analgesic_antipyretic",
            "verdict": "safe",
            "recommended_dose_mg": 500,
            "flag_dangerous": False,
            "reason": "fallback"
        }
    else:
        return {
            "drug_name": "Paracetamol",
            "is_valid": True,
            "verdict": "safe_to_dispense",
            "flag_dangerous": False,
            "interactions_found": [],
            "reason": "fallback"
        }


# ================= LLM =================
def call_llm(prompt, task_id):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        content = resp.choices[0].message.content.strip()

        # Extract JSON safely
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])

    except Exception:
        pass

    return fallback_action(task_id)


# ================= SAFE REQUEST =================
def safe_request(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ================= RUN =================
def run_task(task_id):
    rewards = []
    final_reward = 0.01

    log_start(task_id)

    try:
        obs = safe_request(f"{ENV_URL}/reset", {"task_id": task_id})

        if obs is None:
            log_step(1, "reset", 0.01, True, "reset_failed")
            return 0.01

        for step in range(1, MAX_STEPS + 1):
            action = call_llm(json.dumps(obs), task_id)
            action_str = json.dumps(action, separators=(",", ":"))

            result = safe_request(f"{ENV_URL}/step", action)

            if result is None:
                log_step(step, action_str, 0.01, True, "step_failed")
                rewards.append(0.01)
                break

            reward = float(result.get("reward", 0.01))

            # 🔥 REQUIRED: keep reward strictly between (0,1)
            reward = max(0.0001, min(reward, 0.9999))

            done = bool(result.get("done", False))

            rewards.append(reward)
            final_reward = reward

            log_step(step, action_str, reward, done, None)

            if done:
                break

            obs = result.get("observation", obs)

    finally:
        log_end(
            final_reward >= SUCCESS_THRESHOLD,
            len(rewards),
            rewards if rewards else [0.01]
        )

    return final_reward


# ================= MAIN =================
def main():
    for task in ["task1", "task2", "task3"]:
        try:
            run_task(task)
        except Exception:
            pass


if __name__ == "__main__":
    main()
