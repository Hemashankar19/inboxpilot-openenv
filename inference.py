#!/usr/bin/env python3
"""
Baseline inference script for InboxPilot-OpenEnv.
Logs follow the required [START] / [STEP] / [END] format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

ENV_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

TASK_MAP = {
    "easy": "task_easy_categorize",
    "medium": "task_medium_triage",
    "hard": "task_hard_full_inbox",
    "task_easy_categorize": "task_easy_categorize",
    "task_medium_triage": "task_medium_triage",
    "task_hard_full_inbox": "task_hard_full_inbox",
}

SYSTEM_PROMPT = """You are an expert inbox triage agent for a university operations team.

You will receive the current inbox observation as JSON. Return EXACTLY ONE action as a JSON object.

Available actions and their schemas:
- {"action": "SelectEmail", "email_id": "<id>"}
- {"action": "ClassifyEmail", "category": "<support|finance|placement|spam|phishing|urgent|general>"}
- {"action": "SetPriority", "level": "<low|medium|high>"}
- {"action": "ExtractFields", "fields": {"key": "value", ...}}
- {"action": "RouteEmail", "target_queue": "<tech_support|finance_office|placement_cell|security|archive>"}
- {"action": "DraftReply", "reply_text": "<your reply>"}
- {"action": "MarkResolved"}
- {"action": "RequestMoreInfo", "question": "<your question>"}
- {"action": "FinishEpisode"}

Strict rules:
1. Always SelectEmail before acting on it.
2. ClassifyEmail BEFORE RouteEmail on the same email.
3. Route phishing/suspicious emails to "security". Route spam to "archive".
4. For each email, follow this sequence:
   SelectEmail -> ClassifyEmail -> SetPriority -> ExtractFields -> RouteEmail -> MarkResolved
5. Draft a reply ONLY if asked.
6. NEVER repeat actions.
7. After MarkResolved -> Select next email.
8. When all done -> FinishEpisode.

Return ONLY valid JSON. No explanation.
"""

VALID_CATEGORIES = {"support", "finance", "placement", "spam", "phishing", "urgent", "general"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_QUEUES = {"tech_support", "finance_office", "placement_cell", "security", "archive"}
VALID_ACTIONS = {
    "SelectEmail",
    "ClassifyEmail",
    "SetPriority",
    "ExtractFields",
    "RouteEmail",
    "DraftReply",
    "MarkResolved",
    "RequestMoreInfo",
    "FinishEpisode",
}


def normalize_task_id(task_id: str) -> str:
    return TASK_MAP.get(task_id, task_id)


def call_env(method: str, path: str, body: dict | None = None) -> dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(url, json=body or {}) if method.upper() == "POST" else c.get(url)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {"error": "non_dict_response", "raw": data}
    except Exception as e:
        print(f"[ERROR] call_env failed for {path}: {e}", flush=True)
        return {"error": str(e)}


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        for b in blocks:
            try:
                obj = json.loads(b.strip())
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_step_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("result") if isinstance(result.get("result"), dict) else result

    reward = payload.get("reward", result.get("reward", 0.0))
    done = payload.get("done", result.get("done", False))
    observation = payload.get("observation", payload.get("obs", result.get("observation", {})))
    info = payload.get("info", result.get("info", {}))

    try:
        reward = float(reward)
    except Exception:
        reward = 0.0

    done = bool(done)
    if not isinstance(observation, dict):
        observation = {}
    if not isinstance(info, dict):
        info = {"raw_info": info}

    return {"reward": reward, "done": done, "observation": observation, "info": info}


def infer_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["verify password", "reset password", "otp", "suspended", "login urgently", "click here"]):
        return "phishing"
    if any(k in t for k in ["lottery", "winner", "free money", "claim prize", "unsubscribe", "promo"]):
        return "spam"
    if any(k in t for k in ["internship", "placement", "recruitment", "campus hiring", "interview", "job drive"]):
        return "placement"
    if any(k in t for k in ["invoice", "payment", "fee", "refund", "scholarship", "billing"]):
        return "finance"
    if any(k in t for k in ["help desk", "wifi", "portal error", "bug", "issue", "technical"]):
        return "support"
    if any(k in t for k in ["urgent", "asap", "immediately", "deadline today"]):
        return "urgent"
    return "general"


def infer_priority(text: str, category: str) -> str:
    t = text.lower()
    if category in {"phishing", "urgent"} or any(k in t for k in ["urgent", "asap", "immediately", "security breach"]):
        return "high"
    if category in {"finance", "placement", "support"}:
        return "medium"
    return "low"


def infer_route(category: str) -> str:
    if category == "phishing":
        return "security"
    if category == "spam":
        return "archive"
    if category == "finance":
        return "finance_office"
    if category == "placement":
        return "placement_cell"
    return "tech_support"


def heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    emails = obs.get("email_list", []) or []
    selected = obs.get("selected_email")

    if not selected:
        for e in emails:
            if not e.get("is_resolved", False):
                return {"action": "SelectEmail", "email_id": e.get("id")}
        return {"action": "FinishEpisode"}

    subject = selected.get("subject", "") or ""
    body = selected.get("body", "") or ""
    text = f"{subject}\n{body}"

    if selected.get("assigned_category") is None:
        return {"action": "ClassifyEmail", "category": infer_category(text)}

    if selected.get("assigned_priority") is None:
        return {"action": "SetPriority", "level": infer_priority(text, selected.get("assigned_category", "general"))}

    if not selected.get("extracted_fields"):
        fields = {"subject": subject[:200]} if subject else {"summary": body[:120] or "email"}
        return {"action": "ExtractFields", "fields": fields}

    if selected.get("assigned_route") is None:
        return {"action": "RouteEmail", "target_queue": infer_route(selected.get("assigned_category", "general"))}

    if not selected.get("is_resolved", False):
        return {"action": "MarkResolved"}

    for e in emails:
        if not e.get("is_resolved", False):
            return {"action": "SelectEmail", "email_id": e.get("id")}

    return {"action": "FinishEpisode"}


def sanitize_action(action: dict[str, Any], obs: dict[str, Any], last_action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict) or action.get("action") not in VALID_ACTIONS:
        return heuristic_action(obs)

    selected = obs.get("selected_email")
    emails = obs.get("email_list", []) or []

    act = action["action"]

    if act == "SelectEmail":
        email_id = action.get("email_id")
        valid_ids = {e.get("id") for e in emails if not e.get("is_resolved", False)}
        if email_id not in valid_ids:
            return heuristic_action(obs)
        cleaned = {"action": "SelectEmail", "email_id": email_id}

    elif act == "ClassifyEmail":
        cat = action.get("category")
        if cat not in VALID_CATEGORIES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "ClassifyEmail", "category": cat}

    elif act == "SetPriority":
        lvl = action.get("level")
        if lvl not in VALID_PRIORITIES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "SetPriority", "level": lvl}

    elif act == "ExtractFields":
        fields = action.get("fields")
        if not isinstance(fields, dict) or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "ExtractFields", "fields": fields}

    elif act == "RouteEmail":
        q = action.get("target_queue")
        if q not in VALID_QUEUES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "RouteEmail", "target_queue": q}

    elif act == "DraftReply":
        txt = action.get("reply_text")
        if not isinstance(txt, str) or not txt.strip() or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "DraftReply", "reply_text": txt[:1000]}

    elif act == "RequestMoreInfo":
        q = action.get("question")
        if not isinstance(q, str) or not q.strip() or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "RequestMoreInfo", "question": q[:500]}

    else:
        cleaned = {"action": act}

    if last_action == cleaned:
        return heuristic_action(obs)

    return cleaned


def choose_action(obs: dict[str, Any], history: list[dict[str, str]], last_action: dict[str, Any] | None) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})

    try:
        with httpx.Client(timeout=120) as c:
            resp = c.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("message") or {}).get("content", "")
        parsed = extract_json(raw)
        return sanitize_action(parsed or {}, obs, last_action)
    except Exception as e:
        print(f"[ERROR] Ollama failed, fallback: {e}", flush=True)
        return heuristic_action(obs)


def run(task_id: str = "easy") -> dict[str, Any]:
    resolved_task_id = normalize_task_id(task_id)
    print(f"[START] task={resolved_task_id} model={MODEL_NAME} env={ENV_BASE_URL}", flush=True)

    reset_result = call_env("POST", "/reset", {"task_id": resolved_task_id})
    if "error" in reset_result and not reset_result.get("observation"):
        print("[END] total_reward=0.0000 final_score=0.0000 steps=0", flush=True)
        return {"total_reward": 0.0, "final_score": 0.0, "steps": 0}

    obs_raw = reset_result.get("observation", reset_result.get("obs", {}))
    if not isinstance(obs_raw, dict):
        obs_raw = {}

    history: list[dict[str, str]] = []
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    last_action: dict[str, Any] | None = None
    step_n = 0

    try:
        for step_n in range(1, MAX_STEPS + 1):
            action = choose_action(obs_raw, history, last_action)
            print(f"[STEP] step={step_n} action={json.dumps(action, ensure_ascii=False)}", flush=True)

            result = normalize_step_result(call_env("POST", "/step", {"action": action}))
            reward = result["reward"]
            done = result["done"]
            obs_raw = result["observation"]
            info = result["info"]

            total_reward += reward
            history.append({"role": "user", "content": json.dumps(obs_raw, ensure_ascii=False)})
            history.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            last_action = action

            print(f"[STEP] reward={reward:.4f} done={done} breakdown={info.get('reward_breakdown', {})}", flush=True)

            if done:
                final_info = info
                break

    except Exception as e:
        print(f"[ERROR] run failed: {e}", flush=True)

    final_score = 0.0
    try:
        final_score = float(final_info.get("final_score", 0.0))
    except Exception:
        final_score = 0.0

    print(f"[END] total_reward={total_reward:.4f} final_score={final_score:.4f} steps={step_n}", flush=True)
    return {"total_reward": total_reward, "final_score": final_score, "steps": step_n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    run(args.task)
