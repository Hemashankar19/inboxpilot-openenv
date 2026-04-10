"""Task loader — reads JSON files from data/ and validates structure."""
from __future__ import annotations

import json
import pathlib
from typing import Any

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

TASK_FILES = {
    "easy": DATA_DIR / "task_easy.json",
    "medium": DATA_DIR / "task_medium.json",
    "hard": DATA_DIR / "task_hard.json",
}


def _validate_task_data(task_id: str, data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{task_id}: task file must contain a JSON object")

    required_top_level = ["emails", "gold", "goal"]
    for key in required_top_level:
        if key not in data:
            raise ValueError(f"{task_id}: missing required key {key!r}")

    if not isinstance(data["emails"], list) or not data["emails"]:
        raise ValueError(f"{task_id}: 'emails' must be a non-empty list")

    if not isinstance(data["gold"], list) or not data["gold"]:
        raise ValueError(f"{task_id}: 'gold' must be a non-empty list")

    email_ids = set()
    for i, email in enumerate(data["emails"]):
        if not isinstance(email, dict):
            raise ValueError(f"{task_id}: email #{i} must be an object")
        for key in ("id", "subject", "sender", "body"):
            if key not in email:
                raise ValueError(f"{task_id}: email #{i} missing key {key!r}")
        email_ids.add(email["id"])

    for i, gold in enumerate(data["gold"]):
        if not isinstance(gold, dict):
            raise ValueError(f"{task_id}: gold #{i} must be an object")
        if "email_id" not in gold:
            raise ValueError(f"{task_id}: gold #{i} missing 'email_id'")
        if gold["email_id"] not in email_ids:
            raise ValueError(
                f"{task_id}: gold email_id {gold['email_id']!r} not found in emails"
            )

    data.setdefault("max_steps", 30)
    data.setdefault("soft_step_budget", data["max_steps"])
    data.setdefault(
        "grader_weights",
        {
            "classification": 0.20,
            "priority": 0.15,
            "routing": 0.20,
            "extraction": 0.15,
            "reply": 0.15,
            "resolution": 0.15,
        },
    )

    return data


def load_task(task_id: str) -> dict[str, Any]:
    path = TASK_FILES.get(task_id)
    if path is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_FILES)}")

    if not path.exists():
        raise FileNotFoundError(f"Task file not found for {task_id!r}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _validate_task_data(task_id, data)


def list_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_id, path in TASK_FILES.items():
        if path.exists():
            tasks.append(
                {
                    "id": task_id,
                    "name": f"{task_id.title()} Email Triage",
                    "difficulty": task_id,
                    "grader": "app.graders.grade_task",
                }
            )
    return tasks
