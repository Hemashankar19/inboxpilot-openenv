#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))

VALID_CATEGORIES = {"support", "finance", "placement", "spam", "phishing", "urgent", "general"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_QUEUES = {"tech_support", "finance_office", "placement_cell", "security", "archive"}

SYSTEM_PROMPT = """
You are an expert inbox triage agent for a university operations team.

Return EXACTLY ONE valid JSON object and nothing else.

Available actions:
- {"action": "SelectEmail", "email_id": "<id>"}
- {"action": "ClassifyEmail", "category": "<support|finance|placement|spam|phishing|urgent|general>"}
- {"action": "SetPriority", "level": "<low|medium|high>"}
- {"action": "ExtractFields", "fields": {"key": "value"}}
- {"action": "RouteEmail", "target_queue": "<tech_support|finance_office|placement_cell|security|archive>"}
- {"action": "DraftReply", "reply_text": "<text>"}
- {"action": "MarkResolved"}
- {"action": "RequestMoreInfo", "question": "<text>"}
- {"action": "FinishEpisode"}

Rules:
1. Always SelectEmail first if no email is selected.
2. Prefer this order: SelectEmail -> ClassifyEmail -> SetPriority -> ExtractFields -> RouteEmail -> MarkResolved.
3. Route phishing to security, spam to archive.
4. FinishEpisode only when all emails are resolved.
5. Return JSON only.
""".strip()


def to_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            dumped = x.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def to_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def obs_email_list(obs: Any) -> list[dict[str, Any]]:
    obs = to_dict(obs)
    raw = obs.get("email_list")
    out: list[dict[str, Any]] = []
    for item in to_list(raw):
        d = to_dict(item)
        if d:
            out.append(d)
    return out


def obs_selected(obs: Any) -> dict[str, Any] | None:
    obs = to_dict(obs)
    raw = obs.get("selected_email")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    d = to_dict(raw)
    return d if d else None


def normalize_obs(obs: Any) -> dict[str, Any]:
    obs = to_dict(obs)
    return {
        "inbox_summary": to_dict(obs.get("inbox_summary")),
        "email_list": obs_email_list(obs),
        "selected_email": obs_selected(obs),
        "available_actions": [x for x in to_list(obs.get("available_actions")) if isinstance(x, str)],
        "history": to_list(obs.get("history")),
        "task_goal": obs.get("task_goal") if isinstance(obs.get("task_goal"), str) else "",
        "warnings": to_list(obs.get("warnings")),
        "step": obs.get("step") if isinstance(obs.get("step"), int) else 0,
        "done": bool(obs.get("done", False)),
    }


def call_env(path: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{ENV_URL}{path}", json=body)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else {"raw": data}
    except Exception as e:
        print(f"[ERROR] call_env failed for {path}: {e}", flush=True)
        return {"error": str(e)}


def normalize_step_result(result: Any) -> dict[str, Any]:
    result = to_dict(result)
    payload = to_dict(result.get("result")) if isinstance(result.get("result"), dict) else result

    try:
        reward = float(payload.get("reward", result.get("reward", 0.0)))
    except Exception:
        reward = 0.0

    done = bool(payload.get("done", result.get("done", False)))
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        observation = payload.get("obs")
    if not isinstance(observation, dict):
        observation = result.get("observation")

    return {
        "reward": reward,
        "done": done,
        "observation": normalize_obs(observation),
        "info": to_dict(payload.get("info", result.get("info", {}))),
    }


def infer_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["verify password", "reset password", "otp", "suspended", "login urgently", "click here"]):
        return "phishing"
    if any(k in t for k in ["lottery", "winner", "free money", "claim prize", "unsubscribe", "promo"]):
        return "spam"
    if any(k in t for k in ["internship", "placement", "recruitment", "campus hiring", "interview", "job drive"]):
        return "placement"
    if any(k in t for k in ["invoice", "payment", "fee", "refund", "scholarship", "billing", "charge", "charged"]):
        return "finance"
    if any(k in t for k in ["help desk", "wifi", "portal error", "bug", "issue", "technical", "support"]):
        return "support"
    if any(k in t for k in ["urgent", "asap", "immediately", "deadline today"]):
        return "urgent"
    return "general"


def infer_priority(text: str, category: str) -> str:
    t = text.lower()
    if category in {"phishing", "urgent"}:
        return "high"
    if any(k in t for k in ["urgent", "asap", "immediately", "security breach", "deadline", "double charge"]):
        return "high"
    if category in {"finance", "placement", "support"}:
        return "medium"
    return "low"


def infer_route(category: str) -> str:
    return {
        "phishing": "security",
        "spam": "archive",
        "finance": "finance_office",
        "placement": "placement_cell",
    }.get(category, "tech_support")


def heuristic_action(obs: Any) -> dict[str, Any]:
    obs = normalize_obs(obs)
    selected = obs_selected(obs)
    emails = obs_email_list(obs)

    if selected is None:
        for email in emails:
            if not bool(email.get("is_resolved", False)):
                eid = email.get("id")
                if eid:
                    return {"action": "SelectEmail", "email_id": eid}
        return {"action": "FinishEpisode"}

    subject = str(selected.get("subject", "") or "")
    body = str(selected.get("body", "") or "")
    sender = str(selected.get("sender", "") or "")
    text = f"{subject}\n{body}\n{sender}"

    if selected.get("assigned_category") is None:
        return {"action": "ClassifyEmail", "category": infer_category(text)}

    if selected.get("assigned_priority") is None:
        return {"action": "SetPriority", "level": infer_priority(text, str(selected.get("assigned_category", "general")))}

    ef = selected.get("extracted_fields")
    if not isinstance(ef, dict) or not ef:
        fields: dict[str, str] = {}
        txn = re.findall(r"TXN-\d{4}-\d+", text)
        stu = re.findall(r"STU-\d+", text)

        if txn:
            fields["transaction_id"] = txn[0]
        if stu:
            fields["student_id"] = stu[0]
        if sender:
            fields["sender"] = sender
        if subject:
            fields["subject"] = subject[:200]
        if "double charge" in text.lower():
            fields["issue_type"] = "double_charge"
        if not fields:
            fields["summary"] = body[:120] or "email"

        return {"action": "ExtractFields", "fields": fields}

    if selected.get("assigned_route") is None:
        return {"action": "RouteEmail", "target_queue": infer_route(str(selected.get("assigned_category", "general")))}

    if not bool(selected.get("is_resolved", False)):
        return {"action": "MarkResolved"}

    for email in emails:
        if not bool(email.get("is_resolved", False)) and email.get("id") != selected.get("id"):
            eid = email.get("id")
            if eid:
                return {"action": "SelectEmail", "email_id": eid}

    return {"action": "FinishEpisode"}


def extract_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def sanitize_action(action: Any, obs: Any, last_action: dict[str, Any] | None) -> dict[str, Any]:
    obs = normalize_obs(obs)
    selected = obs_selected(obs)
    unresolved = [e for e in obs_email_list(obs) if not bool(e.get("is_resolved", False))]

    if selected is None and unresolved:
        eid = unresolved[0].get("id")
        return {"action": "SelectEmail", "email_id": eid} if eid else {"action": "FinishEpisode"}

    action = to_dict(action)
    act = action.get("action")

    if act == "FinishEpisode" and (selected is not None or unresolved):
        return heuristic_action(obs)

    if act == "SelectEmail":
        eid = action.get("email_id")
        valid = {e.get("id") for e in unresolved if e.get("id")}
        cleaned = {"action": "SelectEmail", "email_id": eid} if eid in valid else heuristic_action(obs)
    elif act == "ClassifyEmail" and action.get("category") in VALID_CATEGORIES and selected is not None:
        cleaned = {"action": "ClassifyEmail", "category": action.get("category")}
    elif act == "SetPriority" and action.get("level") in VALID_PRIORITIES and selected is not None:
        cleaned = {"action": "SetPriority", "level": action.get("level")}
    elif act == "ExtractFields" and isinstance(action.get("fields"), dict) and selected is not None:
        safe_fields = {str(k): str(v) for k, v in action.get("fields").items()}
        cleaned = {"action": "ExtractFields", "fields": safe_fields}
    elif act == "RouteEmail" and action.get("target_queue") in VALID_QUEUES and selected is not None:
        cleaned = {"action": "RouteEmail", "target_queue": action.get("target_queue")}
    elif act == "DraftReply" and isinstance(action.get("reply_text"), str) and action.get("reply_text").strip() and selected is not None:
        cleaned = {"action": "DraftReply", "reply_text": action.get("reply_text")[:1000]}
    elif act == "RequestMoreInfo" and isinstance(action.get("question"), str) and action.get("question").strip() and selected is not None:
        cleaned = {"action": "RequestMoreInfo", "question": action.get("question")[:500]}
    elif act == "MarkResolved" and selected is not None:
        cleaned = {"action": "MarkResolved"}
    elif act == "FinishEpisode":
        cleaned = {"action": "FinishEpisode"}
    else:
        cleaned = heuristic_action(obs)

    if last_action == cleaned:
        return heuristic_action(obs)

    return cleaned


def get_model_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None


def choose_action(
    obs: Any,
    history: list[dict[str, str]],
    last_action: dict[str, Any] | None,
    client: OpenAI | None,
) -> dict[str, Any]:
    obs = normalize_obs(obs)
    selected = obs_selected(obs)
    unresolved = [e for e in obs_email_list(obs) if not bool(e.get("is_resolved", False))]

    if selected is None and unresolved:
        eid = unresolved[0].get("id")
        return {"action": "SelectEmail", "email_id": eid} if eid else {"action": "FinishEpisode"}

    if client is None:
        return heuristic_action(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
        parsed = extract_json(content)
        return sanitize_action(parsed, obs, last_action)
    except Exception as e:
        print(f"[ERROR] model call failed: {e}", flush=True)
        return heuristic_action(obs)


def run(task_id: str) -> dict[str, Any]:
    print(f"[START] task={task_id} model={MODEL_NAME} env={ENV_URL}", flush=True)

    total_reward = 0.0
    final_score = 0.0
    history: list[dict[str, str]] = []
    last_action: dict[str, Any] | None = None
    step_n = 0
    client = get_model_client()

    try:
        reset_result = call_env("/reset", {"task_id": task_id})
        reset_result = to_dict(reset_result)

        obs = reset_result.get("observation")
        if not isinstance(obs, dict):
            obs = reset_result.get("obs")
        if not isinstance(obs, dict):
            obs = reset_result

        obs = normalize_obs(obs)

        for step_n in range(1, MAX_STEPS + 1):
            action = choose_action(obs, history, last_action, client)
            print(f"[STEP] step={step_n} action={json.dumps(action, ensure_ascii=False)}", flush=True)

            step_result_raw = call_env("/step", {"action": action})
            step_result = normalize_step_result(step_result_raw)

            reward = step_result["reward"]
            done = step_result["done"]
            obs = normalize_obs(step_result["observation"])
            info = to_dict(step_result["info"])

            total_reward += reward
            history.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})
            history.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            last_action = action

            print(
                f"[STEP] reward={reward:.4f} done={done} breakdown={info.get('reward_breakdown', {})}",
                flush=True,
            )

            if done:
                try:
                    final_score = float(info.get("final_score", 0.0))
                except Exception:
                    final_score = 0.0
                break

    except Exception as e:
        print(f"[ERROR] run failed: {e}", flush=True)

    print(f"[END] total_reward={total_reward:.4f} final_score={final_score:.4f} steps={step_n}", flush=True)
    return {"total_reward": total_reward, "final_score": final_score, "steps": step_n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    run(args.task)
