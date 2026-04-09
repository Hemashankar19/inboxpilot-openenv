#!/usr/bin/env python3
"""
Baseline inference script for InboxPilot-OpenEnv.
Logs follow the required [START] / [STEP] / [END] format.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ENV_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "30"))

#  DISABLE OpenAI (force Ollama usage)
client = None

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
4. For each email, follow this sequence: SelectEmail → ClassifyEmail → SetPriority → ExtractFields → RouteEmail → MarkResolved
5. Draft a reply ONLY if asked.
6. NEVER repeat actions.
7. After MarkResolved → Select next email.
8. When all done → FinishEpisode.

Return ONLY valid JSON. No explanation.
"""


def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=60) as c:
            if method == "POST":
                r = c.post(url, json=body)
            else:
                r = c.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] call_env failed for {path}: {e}")
        return {}


#  OLLAMA VERSION (ONLY CHANGE HERE)
def choose_action(obs: dict, history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-10:]:
        messages.append(h)
    messages.append({"role": "user", "content": json.dumps(obs)})

    try:
        response = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        result = response.json()
        raw = result["message"]["content"].strip()

        # clean markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as e:
        print("[ERROR] Ollama failed, fallback:", e)
        return heuristic_action(obs)


def heuristic_action(obs: dict) -> dict:
    emails = obs.get("email_list", [])
    selected = obs.get("selected_email")

    if selected is None:
        for e in emails:
            if not e["is_resolved"]:
                return {"action": "SelectEmail", "email_id": e["id"]}
        return {"action": "FinishEpisode"}

    body_lower = (selected.get("body", "") + selected.get("subject", "")).lower()

    if selected.get("assigned_category") is None:
        if "internship" in body_lower or "placement" in body_lower:
            return {"action": "ClassifyEmail", "category": "placement"}
        return {"action": "ClassifyEmail", "category": "general"}

    if selected.get("assigned_priority") is None:
        return {"action": "SetPriority", "level": "medium"}

    if selected.get("assigned_route") is None:
        return {"action": "RouteEmail", "target_queue": "placement_cell"}

    if not selected.get("is_resolved"):
        return {"action": "MarkResolved"}

    for e in emails:
        if not e["is_resolved"]:
            return {"action": "SelectEmail", "email_id": e["id"]}

    return {"action": "FinishEpisode"}


def run(task_id: str = "easy") -> dict:
    print(f"[START] task={task_id} model={MODEL_NAME} env={ENV_BASE_URL}")
    obs_raw = call_env("POST", "/reset", {"task_id": task_id})
    history: list[dict] = []
    total_reward = 0.0
    final_info: dict = {}

    for step_n in range(1, MAX_STEPS + 1):
        action = choose_action(obs_raw, history)
        print(f"[STEP] step={step_n} action={json.dumps(action)}")

        result = call_env("POST", "/step", {"action": action})
        reward  = result["reward"]
        done    = result["done"]
        obs_raw = result["observation"]
        info    = result["info"]
        total_reward += reward

        history.append({"role": "user",      "content": json.dumps(obs_raw)})
        history.append({"role": "assistant", "content": json.dumps(action)})

        print(f"[STEP] reward={reward:.4f} done={done} breakdown={info.get('reward_breakdown',{})}")

        if done:
            final_info = info
            break

    final_score = final_info.get("final_score", 0.0)
    print(f"[END] total_reward={total_reward:.4f} final_score={final_score:.4f} steps={step_n}")
    return {"total_reward": total_reward, "final_score": final_score, "steps": step_n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy","medium","hard"])
    args = parser.parse_args()
    run(args.task)