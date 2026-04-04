from typing import Optional, Dict, Any, List
from models import Observation, Action, StepResponse, Patient, Prescription
from tasks import TASKS


class RxValidatorEnv:
    def __init__(self):
        self._task_id: Optional[str] = None
        self._scenario: Optional[Dict[str, Any]] = None
        self._patient: Optional[Patient] = None
        self._prescriptions: List[Prescription] = []
        self._step_count: int = 0
        self._done: bool = False
        self._last_reward: float = 0.0
        self._episode_rewards: List[float] = []

    def reset(self, task_id: str = "task1") -> Observation:
        if task_id not in TASKS:
            task_id = "task1"

        task_cls = TASKS[task_id]
        self._task_id = task_id
        self._scenario = task_cls.get_scenario()
        self._step_count = 0
        self._done = False
        self._last_reward = 0.0
        self._episode_rewards = []

        # ✅ FIXED (IMPORTANT)
        self._patient = self._scenario.get(
            "patient",
            Patient(age_years=30, weight_kg=70.0, sex="male", is_pediatric=False)
        )

        # prescriptions fix
        pres = self._scenario.get("prescription")
        if pres:
            self._prescriptions = [pres]
        else:
            self._prescriptions = self._scenario.get("prescriptions", [])

        return self._build_obs()

    def step(self, action: Action) -> StepResponse:
        if self._done:
            return StepResponse(
                observation=self._build_obs(),
                reward=0.0,
                done=True,
                info={"message": "Episode complete. Call reset() to start a new episode."},
            )

        self._step_count += 1
        task_cls = TASKS[self._task_id]

        score, feedback = task_cls.grade(action, self._scenario)

        reward = round(min(1.0, score + 0.02 * (1.0 - score)), 4)
        self._last_reward = reward
        self._episode_rewards.append(reward)

        self._done = True

        return StepResponse(
            observation=self._build_obs(),
            reward=reward,
            done=self._done,
            info={
                "grader_score": score,
                "grader_feedback": feedback,
                "step_count": self._step_count,
                "task_id": self._task_id,
                "task_difficulty": task_cls.difficulty,
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "done": self._done,
            "last_reward": self._last_reward,
            "num_prescriptions": len(self._prescriptions),
        }

    def _build_obs(self) -> Observation:
        task_cls = TASKS.get(self._task_id or "task1", TASKS["task1"])
        return Observation(
            task_id=self._task_id or "task1",
            task_name=task_cls.name,
            difficulty=task_cls.difficulty,
            patient=self._patient,
            prescriptions=self._prescriptions,
            instruction=task_cls.instruction,
            step_count=self._step_count,
        )
