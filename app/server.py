"""FastAPI server exposing the OpenEnv HTTP interface."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import app.env as env
from app.graders import grade_task
from app.tasks import load_task, list_tasks as _list_tasks

app = FastAPI(title="InboxPilot-OpenEnv", version="1.0.0")


def _safe_score(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 4)


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: dict[str, Any]


class GradeRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()

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

    safe_reward = _safe_score(reward)

    final_score = info.get("final_score")
    if final_score is not None:
        info["final_score"] = _safe_score(final_score)

    return {
        "observation": obs.model_dump(),
        "reward": safe_reward,
        "done": bool(done),
        "info": info,
    }


@app.post("/grade")
def grade(req: Optional[GradeRequest] = None):
    s = env.state()
    if not s:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    if req is None:
        req = GradeRequest()

    task_id = req.task_id or s.get("task_id", "easy")
    task_data = load_task(task_id)

    raw_score = grade_task(env._state, task_data)
    final_score = _safe_score(raw_score)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "score": final_score,
        "result": {
            "final_score": final_score,
            "score": final_score,
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
    return _list_tasks()


@app.get("/health")
@app.post("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
