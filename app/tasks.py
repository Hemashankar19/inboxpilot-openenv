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


def load_task(task_id: str) -> dict[str, Any]:
    path = TASK_FILES.get(task_id)
    if path is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_FILES)}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "easy",
            "name": "Easy Email Triage",
            "difficulty": "easy",
            "grader": "app.graders.grade_task",
        },
        {
            "id": "medium",
            "name": "Medium Email Triage",
            "difficulty": "medium",
            "grader": "app.graders.grade_task",
        },
        {
            "id": "hard",
            "name": "Hard Inbox Triage",
            "difficulty": "hard",
            "grader": "app.graders.grade_task",
        },
    ]
