"""FastAPI server exposing the OpenEnv HTTP interface."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import app.env as env
from app.graders import grade_task
from app.tasks import load_task, list_tasks as _list_tasks

app = FastAPI(title="InboxPilot-OpenEnv", version="1.0.0")


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: dict[str, Any]


class GradeRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(req.task_id)
    return {
        "observation": obs.model_dump(),
        "task_id": req.task_id,
    }


@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    final_score = info.get("final_score")
    if final_score is not None:
        final_score = max(0.01, min(0.99, float(final_score)))
        info["final_score"] = final_score

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/grade")
def grade(req: GradeRequest = GradeRequest()):
    s = env.state()
    if not s:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    task_id = req.task_id or s.get("task_id", "easy")
    task_data = load_task(task_id)

    raw_score = grade_task(env._state, task_data)
    final_score = max(0.01, min(0.99, float(raw_score)))

    return {
        "task_id": task_id,
        "result": {
            "final_score": final_score,
        },
    }


@app.get("/state")
def get_state():
    s = env.state()
    if not s:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return s


@app.get("/tasks")
def list_tasks():
    return {"tasks": _list_tasks()}


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
