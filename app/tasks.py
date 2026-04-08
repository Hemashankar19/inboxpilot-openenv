"""Task loader — reads JSON files from data/ and validates structure."""
import json, pathlib
from typing import Any

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

TASK_FILES = {
    "easy":   DATA_DIR / "task_easy.json",
    "medium": DATA_DIR / "task_medium.json",
    "hard":   DATA_DIR / "task_hard.json",
}

def load_task(task_id: str) -> dict[str, Any]:
    path = TASK_FILES.get(task_id)
    if path is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_FILES)}")
    with open(path) as f:
        return json.load(f)

def list_tasks() -> list[str]:
    return list(TASK_FILES.keys())
