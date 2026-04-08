"""FastAPI server exposing the OpenEnv HTTP interface."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import app.env as env

app = FastAPI(title="InboxPilot-OpenEnv", version="1.0.0")


from typing import Optional

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
class StepRequest(BaseModel):
    action: dict[str, Any]


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    obs = env.reset(req.task_id)
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
def get_state():
    s = env.state()
    if not s:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return s


@app.get("/tasks")
def list_tasks():
    from app.tasks import list_tasks as _lt
    return {"tasks": _lt()}


@app.get("/health")
def health():
    return {"status": "ok"}
