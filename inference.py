#!/usr/bin/env python3
"""
Robust inference script for InboxPilot-OpenEnv.
Logs follow the required [START] / [STEP] / [END] format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

ENV_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

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
6. NEVER repeat the exact same action on the same state.
7. After MarkResolved -> Select next email.
8. When all done -> FinishEpisode.

Return ONLY valid JSON. No explanation.
"""

TASK_MAP = {
    "easy": "task_easy_categorize",
    "medium": "task_medium_triage",
    "hard": "task_hard_full_inbox",
    "task_easy_categorize": "task_easy_categorize",
    "task_medium_triage": "task_medium_triage",
    "task_hard_full_inbox": "task_hard_full_inbox",
}

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


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            obj = safe_json_loads(part)
            if obj is not None:
                return obj

    obj = safe_json_loads(text)
    if obj is not None:
        return obj

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        obj = safe_json_loads(match.group(0))
        if obj is not None:
            return obj

    return None


def call_env(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=60) as client:
            if method.upper() == "POST":
                resp = client.post(url, json=body or {})
            else:
                resp = client.get(url)

        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Non-JSON response from env at {path}: {e}", flush=True)
            return {"error": f"non_json_response:{path}", "status_code": resp.status_code, "text": resp.text[:500]}

        if not isinstance(data, dict):
            return {"error": f"unexpected_response_type:{type(data).__name__}", "raw": data}

        return data

    except Exception as e:
        print(f"[ERROR] call_env failed for {path}: {e}", flush=True)
        return {"error": f"env_call_failed:{path}", "exception": str(e)}


def normalize_step_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "reward": 0.0,
            "done": True,
            "observation": {},
            "info": {"error": "step_result_not_dict", "raw_type": str(type(result))},
        }

    payload = result
    if isinstance(result.get("result"), dict):
        payload = result["result"]

    reward = payload.get("reward", 0.0)
    done = payload.get("done", False)
    observation = payload.get("observation", payload.get("obs", {}))
    info = payload.get("info", {})

    if not isinstance(info, dict):
        info = {"raw_info": info}

    if "error" in result and "error" not in info:
        info["error"] = result["error"]
    if "exception" in result and "exception" not in info:
        info["exception"] = result["exception"]

    try:
        reward = float(reward)
    except Exception:
        info["bad_reward"] = reward
        reward = 0.0

    done = bool(done)
    if not isinstance(observation, dict):
        info["bad_observation"] = observation
        observation = {}

    return {
        "reward": reward,
        "done": done,
        "observation": observation,
        "info": info,
    }


def infer_category(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["lottery", "winner", "claim prize", "free money", "click here", "congratulations"]):
        return "spam"
    if any(k in t for k in ["verify password", "reset your password", "login urgently", "suspended account", "bank account", "otp", "credential"]):
        return "phishing"
    if any(k in t for k in ["internship", "placement", "recruitment", "job drive", "interview schedule", "campus hiring"]):
        return "placement"
    if any(k in t for k in ["invoice", "refund", "payment", "fee", "scholarship amount", "billing"]):
        return "finance"
    if any(k in t for k in ["help desk", "wifi", "portal error", "password issue", "technical issue", "bug"]):
        return "support"
    if any(k in t for k in ["urgent", "asap", "immediately", "deadline today"]):
        return "urgent"
    return "general"


def infer_priority(text: str, category: str) -> str:
    t = text.lower()
    if category in {"phishing", "urgent"}:
        return "high"
    if any(k in t for k in ["urgent", "asap", "today", "immediately", "deadline", "security breach"]):
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
    if category in {"support", "urgent", "general"}:
        return "tech_support"
    return "archive"


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
    category = selected.get("assigned_category")
    priority = selected.get("assigned_priority")
    route = selected.get("assigned_route")

    if category is None:
        return {"action": "ClassifyEmail", "category": infer_category(text)}

    if priority is None:
        return {"action": "SetPriority", "level": infer_priority(text, category)}

    extracted_fields = selected.get("extracted_fields")
    if not extracted_fields:
        fields = {}
        if subject:
            fields["subject"] = subject[:200]
        if "@" in body:
            emails_found = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", body)
            if emails_found:
                fields["contact_email"] = emails_found[0]
        return {"action": "ExtractFields", "fields": fields or {"summary": subject[:120] or "email"}}

    if route is None:
        return {"action": "RouteEmail", "target_queue": infer_route(category)}

    if not selected.get("is_resolved", False):
        return {"action": "MarkResolved"}

    for e in emails:
        if not e.get("is_resolved", False):
            return {"action": "SelectEmail", "email_id": e.get("id")}

    return {"action": "FinishEpisode"}


def sanitize_action(action: dict[str, Any], obs: dict[str, Any], last_action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return heuristic_action(obs)

    act = action.get("action")
    if act not in VALID_ACTIONS:
        return heuristic_action(obs)

    selected = obs.get("selected_email")
    emails = obs.get("email_list", []) or []

    if act == "SelectEmail":
        email_id = action.get("email_id")
        valid_ids = {e.get("id") for e in emails if not e.get("is_resolved", False)}
        if email_id not in valid_ids:
            return heuristic_action(obs)
        cleaned = {"action": "SelectEmail", "email_id": email_id}

    elif act == "ClassifyEmail":
        category = action.get("category")
        if category not in VALID_CATEGORIES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "ClassifyEmail", "category": category}

    elif act == "SetPriority":
        level = action.get("level")
        if level not in VALID_PRIORITIES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "SetPriority", "level": level}

    elif act == "ExtractFields":
        fields = action.get("fields")
        if not isinstance(fields, dict) or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "ExtractFields", "fields": fields}

    elif act == "RouteEmail":
        queue = action.get("target_queue")
        if queue not in VALID_QUEUES or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "RouteEmail", "target_queue": queue}

    elif act == "DraftReply":
        reply_text = action.get("reply_text")
        if not isinstance(reply_text, str) or not reply_text.strip() or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "DraftReply", "reply_text": reply_text[:1000]}

    elif act == "RequestMoreInfo":
        question = action.get("question")
        if not isinstance(question, str) or not question.strip() or not selected:
            return heuristic_action(obs)
        cleaned = {"action": "RequestMoreInfo", "question": question[:500]}

    else:
        cleaned = {"action": act}

    if last_action == cleaned:
        return heuristic_action(obs)

    return cleaned


def choose_action(obs: dict[str, Any], history: list[dict[str, str]], last_action: dict[str, Any] | None) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})

    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "email_id": {"type": "string"},
            "category": {"type": "string"},
            "level": {"type": "string"},
            "fields": {"type": "object"},
            "target_queue": {"type": "string"},
            "reply_text": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["action"],
    }

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "options": {"temperature": 0},
                },
            )
        response.raise_for_status()
        data = response.json()

        raw = (data.get("message") or {}).get("content", "")
        parsed = extract_json_object(raw)
        if parsed is None:
            raise ValueError(f"Could not parse model JSON: {raw[:300]}")

        return sanitize_action(parsed, obs, last_action)

    except Exception as e:
        print(f"[ERROR] model failed, using heuristic fallback: {e}", flush=True)
        return heuristic_action(obs)


def run(task_id: str = "easy") -> dict[str, Any]:
    resolved_task_id = normalize_task_id(task_id)
    print(f"[START] task={resolved_task_id} model={MODEL_NAME} env={ENV_BASE_URL}", flush=True)

    reset_result = call_env("POST", "/reset", {"task_id": resolved_task_id})

    if "error" in reset_result:
        print(f"[END] total_reward=0.0000 final_score=0.0000 steps=0", flush=True)
        return {"total_reward": 0.0, "final_score": 0.0, "steps": 0, "error": reset_result}

    obs_raw = reset_result.get("observation", reset_result.get("obs", reset_result))
    if not isinstance(obs_raw, dict):
        obs_raw = {}

    history: list[dict[str, str]] = []
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    last_action: dict[str, Any] | None = None
    step_n = 0

    for step_n in range(1, MAX_STEPS + 1):
        action = choose_action(obs_raw, history, last_action)
        print(f"[STEP] step={step_n} action={json.dumps(action, ensure_ascii=False)}", flush=True)

        raw_result = call_env("POST", "/step", {"action": action})
        parsed = normalize_step_result(raw_result)

        reward = parsed["reward"]
        done = parsed["done"]
        obs_raw = parsed["observation"]
        info = parsed["info"]

        total_reward += reward

        history.append({"role": "user", "content": json.dumps(obs_raw, ensure_ascii=False)})
        history.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        last_action = action

        print(
            f"[STEP] reward={reward:.4f} done={done} breakdown={json.dumps(info.get('reward_breakdown', {}), ensure_ascii=False)}",
            flush=True,
        )

        if done:
            final_info = info if isinstance(info, dict) else {}
            break

    final_score = 0.0
    if isinstance(final_info, dict):
        try:
            final_score = float(final_info.get("final_score", 0.0))
        except Exception:
            final_score = 0.0

    print(f"[END] total_reward={total_reward:.4f} final_score={final_score:.4f} steps={step_n}", flush=True)
    return {
        "total_reward": total_reward,
        "final_score": final_score,
        "steps": step_n,
        "final_info": final_info,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy")
    args = parser.parse_args()
    run(args.task)
