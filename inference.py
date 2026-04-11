#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

# ── config ────────────────────────────────────────────────────────────────────
# Checker injects these two — must read them exactly:
API_BASE_URL  = os.getenv("API_BASE_URL", "")          # checker's LiteLLM proxy
API_KEY       = os.getenv("API_KEY", os.getenv("HF_TOKEN", "placeholder"))

# Ollama local config:
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")

# Which model name to report in [START] line:
MODEL_NAME    = os.getenv("MODEL_NAME", OLLAMA_MODEL)

# Environment server URL:
ENV_URL       = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS     = int(os.getenv("MAX_STEPS", "30"))

TASK_IDS = ["easy", "medium", "hard"]

VALID_CATEGORIES = {"support","finance","placement","spam","phishing","urgent","general"}
VALID_PRIORITIES = {"low","medium","high"}
VALID_QUEUES     = {"tech_support","finance_office","placement_cell","security","archive"}

SYSTEM_PROMPT = """You are an expert inbox triage agent for a university operations team.
Return EXACTLY ONE JSON action object and nothing else — no explanation, no markdown.

Available actions:
{"action":"SelectEmail","email_id":"<id>"}
{"action":"ClassifyEmail","category":"<support|finance|placement|spam|phishing|urgent|general>"}
{"action":"SetPriority","level":"<low|medium|high>"}
{"action":"ExtractFields","fields":{"key":"value"}}
{"action":"RouteEmail","target_queue":"<tech_support|finance_office|placement_cell|security|archive>"}
{"action":"DraftReply","reply_text":"<your reply>"}
{"action":"MarkResolved"}
{"action":"FinishEpisode"}

Rules:
1. Per email: SelectEmail → ClassifyEmail → SetPriority → ExtractFields → RouteEmail → DraftReply (if reply needed) → MarkResolved
2. Phishing → route to security. Spam → route to archive.
3. DraftReply: be empathetic, mention the relevant team, ask for details if needed.
4. After all emails resolved → FinishEpisode.
5. NEVER repeat an action on the same email. Check history carefully.
6. Return raw JSON only."""


# ── heuristic agent ───────────────────────────────────────────────────────────

def _infer_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["verify password","reset password","otp","suspended","click here","scam","phish","bank account","password"]):
        return "phishing"
    if any(k in t for k in ["lottery","winner","free money","claim prize","giveaway","promo","selected","free iphone"]):
        return "spam"
    if any(k in t for k in ["internship","placement","offer letter","recruitment","interview","job drive","campus"]):
        return "placement"
    if any(k in t for k in ["invoice","payment","fee","refund","scholarship","billing","charge","charged","transaction","deducted"]):
        return "finance"
    if any(k in t for k in ["portal","error","bug","issue","technical","support","access","certificate","grade","discrepancy","cannot"]):
        return "support"
    if any(k in t for k in ["urgent","asap","immediately","deadline today"]):
        return "urgent"
    return "general"

def _infer_priority(text: str, category: str) -> str:
    t = text.lower()
    if category in {"phishing","urgent"}:
        return "high"
    if any(k in t for k in ["urgent","asap","immediately","security breach","deadline","double charge","tomorrow","today"]):
        return "high"
    if category in {"finance","placement","support"}:
        return "medium"
    return "low"

def _infer_route(category: str) -> str:
    return {
        "phishing":  "security",
        "spam":      "archive",
        "finance":   "finance_office",
        "placement": "placement_cell",
        "support":   "tech_support",
        "urgent":    "tech_support",
        "general":   "archive",
    }.get(category, "tech_support")

def _needs_reply(task_goal: str, category: str) -> bool:
    """Decide if this email warrants a DraftReply."""
    goal = task_goal.lower()
    return "reply" in goal or "draft" in goal or category in {"support","finance"}

def _draft_reply(selected: dict, task_goal: str) -> str:
    cat = str(selected.get("assigned_category","general"))
    subject = str(selected.get("subject","") or "")
    sender  = str(selected.get("sender","") or "").split("@")[0].replace("."," ").title()

    if cat == "finance":
        return (
            f"Dear {sender},\n\nWe sincerely apologize for the inconvenience regarding your payment. "
            "We have noted your concern and forwarded it to our finance office for immediate review. "
            "Please provide your transaction ID and registered email so we can resolve this promptly.\n\n"
            "Best regards,\nUniversity Operations Team"
        )
    if cat == "support":
        return (
            f"Dear {sender},\n\nWe apologize for the trouble you are experiencing. "
            "Our technical support team has been notified and will assist you shortly. "
            "Please provide a screenshot of the error and your registered email so we can resolve this faster.\n\n"
            "Best regards,\nUniversity Operations Team"
        )
    if cat == "placement":
        return (
            f"Dear {sender},\n\nThank you for reaching out to the placement cell. "
            "We have noted your query and our placement team will get back to you within 2 working days. "
            "Please provide your student ID and any relevant offer details.\n\n"
            "Best regards,\nPlacement Cell"
        )
    return (
        f"Dear {sender},\n\nThank you for contacting us regarding: {subject}. "
        "We have received your request and will respond within 2 working days.\n\n"
        "Best regards,\nUniversity Operations Team"
    )

def _heuristic(obs: dict) -> dict:
    emails   = obs.get("email_list", [])
    selected = obs.get("selected_email")
    goal     = str(obs.get("task_goal",""))

    # No email selected → pick first unresolved
    if selected is None:
        for e in emails:
            if not e.get("is_resolved"):
                return {"action": "SelectEmail", "email_id": e["id"]}
        return {"action": "FinishEpisode"}

    subject = str(selected.get("subject","") or "")
    body    = str(selected.get("body","") or "")
    sender  = str(selected.get("sender","") or "")
    text    = f"{subject}\n{body}\n{sender}"

    # Step 1: classify
    if selected.get("assigned_category") is None:
        return {"action": "ClassifyEmail", "category": _infer_category(text)}

    cat = str(selected.get("assigned_category","general"))

    # Step 2: priority
    if selected.get("assigned_priority") is None:
        return {"action": "SetPriority", "level": _infer_priority(text, cat)}

    # Step 3: extract fields
    ef = selected.get("extracted_fields") or {}
    if not ef:
        fields: dict[str,str] = {}
        for m in re.findall(r"TXN-\d{4}-\d+", text):   fields["transaction_id"] = m
        for m in re.findall(r"STU-\d+", text):            fields["student_id"]     = m
        for m in re.findall(r"OFF-\d{4}-\d+", text):    fields["offer_id"]        = m
        for m in re.findall(r"[A-Z]{2}\d{3,}", text):    fields["course_code"]     = m
        for m in re.findall(r"[Ee]rror\s*(\d{3})", text): fields["error_code"]    = m
        if not fields:
            fields["summary"] = body[:80] or "email"
        return {"action": "ExtractFields", "fields": fields}

    # Step 4: route
    if selected.get("assigned_route") is None:
        return {"action": "RouteEmail", "target_queue": _infer_route(cat)}

    # Step 5: draft reply if needed AND not already drafted AND not spam/phishing
    if (selected.get("draft_reply") is None
            and cat not in {"spam","phishing","general"}
            and _needs_reply(goal, cat)):
        return {"action": "DraftReply", "reply_text": _draft_reply(selected, goal)}

    # Step 6: resolve
    if not selected.get("is_resolved"):
        return {"action": "MarkResolved"}

    # Move to next unresolved email
    for e in emails:
        if not e.get("is_resolved") and e.get("id") != selected.get("id"):
            return {"action": "SelectEmail", "email_id": e["id"]}

    return {"action": "FinishEpisode"}


# ── LLM client ────────────────────────────────────────────────────────────────

def _get_llm():
    """
    Returns (client, model_name) or (None, None).

    Priority:
      1. Checker's proxy  — API_BASE_URL env var is set and non-empty
      2. Local Ollama     — reachable at OLLAMA_URL
      3. None             — heuristic fallback
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, None

    # 1. Checker proxy
    if API_BASE_URL:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            return client, MODEL_NAME
        except Exception:
            pass

    # 2. Local Ollama (OpenAI-compatible endpoint)
    ollama_base = f"{OLLAMA_URL}/v1"
    try:
        # Quick reachability check
        with httpx.Client(timeout=3) as c:
            r = c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                client = OpenAI(base_url=ollama_base, api_key="ollama")
                return client, OLLAMA_MODEL
    except Exception:
        pass

    return None, None


def _parse_action(raw: str, obs: dict) -> dict:
    """Parse LLM output into a valid action dict."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    # Try direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except Exception:
            pass
    return _heuristic(obs)


def _llm_action(obs: dict, history: list, client, model: str) -> dict:
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs.extend(history[-10:])
        msgs.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_action(raw, obs)
    except Exception as e:
        return _heuristic(obs)


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
    return raw if isinstance(raw, dict) else {}


# ── main run loop ─────────────────────────────────────────────────────────────

def run_task(task_id: str, client, model: str | None) -> float:
    print(f"[START] task={task_id} model={model or 'heuristic'} env={ENV_URL}", flush=True)

    reset_raw = _post("/reset", {"task_id": task_id})
    if "error" in reset_raw:
        print(f"[END] task={task_id} score=0.001 steps=0", flush=True)
        return 0.001

    obs         = _extract_obs(reset_raw)
    history: list = []
    total_reward  = 0.0
    final_score   = 0.001
    last_action: dict | None = None

    for step_n in range(1, MAX_STEPS + 1):
        # Choose action: LLM if available, else heuristic
        if client and model:
            action = _llm_action(obs, history, client, model)
        else:
            action = _heuristic(obs)

        # Prevent exact repeat (infinite loop guard)
        if action == last_action:
            action = _heuristic(obs)
            if action == last_action:
                action = {"action": "FinishEpisode"}

        print(f"[STEP] step={step_n} action={json.dumps(action)}", flush=True)

        result       = _post("/step", {"action": action})
        reward       = float(result.get("reward", 0.0))
        done         = bool(result.get("done", False))
        new_obs      = _extract_obs(result)
        info         = result.get("info", {}) or {}

        # Update obs only if we got a real observation back
        if new_obs:
            obs = new_obs

        total_reward += reward
        history.append({"role": "user",      "content": json.dumps(obs, ensure_ascii=False)})
        history.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        last_action = action

        print(f"[STEP] reward={reward:.4f} done={done} breakdown={info.get('reward_breakdown',{})}", flush=True)

        if done:
            fs = info.get("final_score")
            if fs is not None:
                final_score = float(fs)
            break

    # Always strictly in (0.001, 0.999) — checker rejects 0.0 and 1.0
    final_score = max(0.001, min(0.999, final_score))

    print(f"[END] task={task_id} total_reward={total_reward:.4f} score={final_score:.4f} steps={step_n}", flush=True)
    return final_score


def main():
    client, model = _get_llm()

    if client and model:
        source = f"LLM proxy ({API_BASE_URL})" if API_BASE_URL else f"Ollama ({OLLAMA_URL}, model={model})"
        print(f"[INFO] Using {source}", flush=True)
    else:
        print("[INFO] No LLM reachable — using heuristic agent", flush=True)

    scores = []
    for task_id in TASK_IDS:
        score = run_task(task_id, client, model)
        scores.append(score)

    avg = sum(scores) / len(scores)
    print(f"[SUMMARY] tasks={TASK_IDS} scores={scores} avg={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
