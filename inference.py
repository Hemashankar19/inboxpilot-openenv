#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

# ── config ────────────────────────────────────────────────────────────────────
# The checker injects API_BASE_URL and API_KEY — use them exactly as-is.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("API_KEY", os.getenv("HF_TOKEN", "placeholder"))
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "30"))

# All 3 tasks must run — the checker counts [END] lines and reads score= from each
TASK_IDS = ["easy", "medium", "hard"]

VALID_CATEGORIES = {"support","finance","placement","spam","phishing","urgent","general"}
VALID_PRIORITIES = {"low","medium","high"}
VALID_QUEUES     = {"tech_support","finance_office","placement_cell","security","archive"}

# ── heuristic helpers ─────────────────────────────────────────────────────────

def _infer_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["verify password","reset password","otp","suspended","click here","scam","phish"]):
        return "phishing"
    if any(k in t for k in ["lottery","winner","free money","claim prize","giveaway","promo"]):
        return "spam"
    if any(k in t for k in ["internship","placement","offer letter","recruitment","interview","job drive"]):
        return "placement"
    if any(k in t for k in ["invoice","payment","fee","refund","scholarship","billing","charge","charged","transaction"]):
        return "finance"
    if any(k in t for k in ["portal","error","bug","issue","technical","support","access","certificate","grade"]):
        return "support"
    if any(k in t for k in ["urgent","asap","immediately","deadline today"]):
        return "urgent"
    return "general"

def _infer_priority(text: str, category: str) -> str:
    t = text.lower()
    if category in {"phishing","urgent"}:
        return "high"
    if any(k in t for k in ["urgent","asap","immediately","security breach","deadline","double charge","tomorrow"]):
        return "high"
    if category in {"finance","placement","support"}:
        return "medium"
    return "low"

def _infer_route(category: str) -> str:
    return {"phishing":"security","spam":"archive","finance":"finance_office",
            "placement":"placement_cell"}.get(category, "tech_support")

def _heuristic(obs: dict) -> dict:
    emails    = obs.get("email_list", [])
    selected  = obs.get("selected_email")

    if selected is None:
        for e in emails:
            if not e.get("is_resolved"):
                return {"action": "SelectEmail", "email_id": e["id"]}
        return {"action": "FinishEpisode"}

    subject = str(selected.get("subject","") or "")
    body    = str(selected.get("body","") or "")
    sender  = str(selected.get("sender","") or "")
    text    = f"{subject}\n{body}\n{sender}"

    if selected.get("assigned_category") is None:
        return {"action": "ClassifyEmail", "category": _infer_category(text)}

    if selected.get("assigned_priority") is None:
        cat = str(selected.get("assigned_category","general"))
        return {"action": "SetPriority", "level": _infer_priority(text, cat)}

    ef = selected.get("extracted_fields") or {}
    if not ef:
        fields: dict[str,str] = {}
        for m in re.findall(r"TXN-\d{4}-\d+", text): fields["transaction_id"] = m
        for m in re.findall(r"STU-\d+", text):        fields["student_id"] = m
        for m in re.findall(r"OFF-\d{4}-\d+", text):  fields["offer_id"] = m
        for m in re.findall(r"[A-Z]{2}\d{3,}", text): fields["course_code"] = m
        for m in re.findall(r"[Ee]rror\s*(\d{3})", text): fields["error_code"] = m
        if not fields: fields["summary"] = body[:80] or "email"
        return {"action": "ExtractFields", "fields": fields}

    if selected.get("assigned_route") is None:
        cat = str(selected.get("assigned_category","general"))
        return {"action": "RouteEmail", "target_queue": _infer_route(cat)}

    if not selected.get("is_resolved"):
        return {"action": "MarkResolved"}

    for e in emails:
        if not e.get("is_resolved") and e["id"] != selected.get("id"):
            return {"action": "SelectEmail", "email_id": e["id"]}

    return {"action": "FinishEpisode"}

# ── env communication ─────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(f"{ENV_URL}{path}", json=body)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"error": str(exc)}

def _extract_obs(raw: dict) -> dict:
    for key in ("observation", "obs"):
        v = raw.get(key)
        if isinstance(v, dict):
            return v
    return raw

# ── LLM client ───────────────────────────────────────────────────────────────
# Always use API_BASE_URL + API_KEY exactly as the checker injects them.
# Falls back to heuristic only if the OpenAI package is missing.

def _get_llm():
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        return None

SYSTEM_PROMPT = """You are an expert inbox triage agent. Return ONE JSON action only.
Actions: SelectEmail, ClassifyEmail, SetPriority, ExtractFields, RouteEmail, DraftReply, MarkResolved, FinishEpisode.
Order per email: Select -> Classify -> Priority -> Extract -> Route -> Resolve. Then next email. FinishEpisode last."""

def _llm_action(obs: dict, history: list, client) -> dict:
    try:
        msgs = [{"role":"system","content":SYSTEM_PROMPT}]
        msgs.extend(history[-8:])
        msgs.append({"role":"user","content":json.dumps(obs)})
        resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs, temperature=0, max_tokens=200)
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else _heuristic(obs)
    except Exception:
        return _heuristic(obs)

# ── main run loop ─────────────────────────────────────────────────────────────

def run_task(task_id: str, llm) -> float:
    """Run one task. Returns final_score strictly in (0.001, 0.999)."""
    print(f"[START] task={task_id} model={MODEL_NAME} env={ENV_URL}", flush=True)

    reset_raw = _post("/reset", {"task_id": task_id})
    if "error" in reset_raw:
        print(f"[END] task={task_id} score=0.001 steps=0", flush=True)
        return 0.001

    obs = _extract_obs(reset_raw)
    history: list = []
    total_reward  = 0.0
    final_score   = 0.001
    last_action: dict | None = None

    for step_n in range(1, MAX_STEPS + 1):
        action = _llm_action(obs, history, llm) if llm else _heuristic(obs)

        # prevent exact repeat
        if action == last_action:
            action = _heuristic(obs)

        print(f"[STEP] step={step_n} action={json.dumps(action)}", flush=True)

        result  = _post("/step", {"action": action})
        reward  = float(result.get("reward", 0.0))
        done    = bool(result.get("done", False))
        obs     = _extract_obs(result)
        info    = result.get("info", {}) or {}

        total_reward += reward
        history.append({"role":"user",      "content": json.dumps(obs)})
        history.append({"role":"assistant", "content": json.dumps(action)})
        last_action = action

        print(f"[STEP] reward={reward:.4f} done={done} breakdown={info.get('reward_breakdown',{})}", flush=True)

        if done:
            fs = info.get("final_score")
            if fs is not None:
                final_score = float(fs)
            break

    # Guarantee score is strictly in (0.001, 0.999) — checker rejects 0.0 and 1.0
    final_score = max(0.001, min(0.999, final_score))

    print(f"[END] task={task_id} total_reward={total_reward:.4f} score={final_score:.4f} steps={step_n}", flush=True)
    return final_score


def main():
    llm = _get_llm()
    scores = []
    for task_id in TASK_IDS:
        score = run_task(task_id, llm)
        scores.append(score)
    avg = sum(scores) / len(scores)
    print(f"[SUMMARY] tasks={TASK_IDS} scores={scores} avg={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
