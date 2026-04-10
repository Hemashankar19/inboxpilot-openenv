"""Deterministic graders for each task type. No LLM involved."""
from __future__ import annotations

import re
from typing import Any

from app.models import EmailMessage, EpisodeState


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_phrase_from_group(text: str, group: list[str]) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in group)


def _extract_id_from_text(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text or "", re.I))


def _clamp_open_unit(x: float) -> float:
    """
    Clamp score strictly into (0, 1), never allowing exact 0.0 or 1.0.
    """
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 4)


# ── reply grader (rule-based semantic templates) ──────────────────────────────

def grade_reply(reply: str, requirements: dict) -> float:
    """
    requirements keys (all optional):
      required_groups: list[list[str]]  — at least one phrase per group must appear
      forbidden_phrases: list[str]
      required_ids: list[str]  — regex patterns that must match
    Returns score strictly in (0, 1).
    """
    if not reply:
        return 0.01

    score = 1.0
    groups = requirements.get("required_groups", [])
    if groups:
        per_group = 1.0 / len(groups)
        for group in groups:
            if not _has_phrase_from_group(reply, group):
                score -= per_group

    forbidden = requirements.get("forbidden_phrases", [])
    for phrase in forbidden:
        if phrase.lower() in reply.lower():
            score -= 0.3

    for pattern in requirements.get("required_ids", []):
        if not _extract_id_from_text(reply, pattern):
            score -= 0.2

    return _clamp_open_unit(score)


# ── per-email grader ──────────────────────────────────────────────────────────

def grade_email(email: EmailMessage, gold: dict[str, Any]) -> dict[str, float]:
    """Grade a single email against gold standard. Returns component scores."""
    result: dict[str, float] = {
        "classification": 0.0,
        "priority": 0.0,
        "routing": 0.0,
        "extraction": 0.0,
        "reply": 0.0,
    }

    if gold.get("category") and email.assigned_category == gold["category"]:
        result["classification"] = 1.0

    if gold.get("priority") and email.assigned_priority == gold["priority"]:
        result["priority"] = 1.0

    if gold.get("route") and email.assigned_route == gold["route"]:
        result["routing"] = 1.0

    required_fields: dict[str, str] = gold.get("required_fields", {})
    if required_fields:
        correct = sum(
            1
            for k, v in required_fields.items()
            if str(email.extracted_fields.get(k, "")).strip().lower() == str(v).strip().lower()
        )
        result["extraction"] = correct / len(required_fields)

    reply_reqs = gold.get("reply_requirements")
    if reply_reqs:
        result["reply"] = grade_reply(email.draft_reply or "", reply_reqs)

    return result


# ── task-level graders ────────────────────────────────────────────────────────

def grade_task(state: EpisodeState, task_data: dict[str, Any]) -> float:
    """
    Master grader. Returns a normalised score strictly in (0, 1).
    task_data["gold"] is a list of gold objects, one per email.
    """
    gold_map: dict[str, dict[str, Any]] = {
        g["email_id"]: g for g in task_data.get("gold", []) if "email_id" in g
    }
    email_map: dict[str, EmailMessage] = {e.id: e for e in state.emails}

    if not gold_map:
        return 0.5

    total_possible = 0.0
    total_earned = 0.0

    weights = task_data.get(
        "grader_weights",
        {
            "classification": 0.20,
            "priority": 0.15,
            "routing": 0.25,
            "extraction": 0.20,
            "reply": 0.20,
        },
    )

    for email_id, gold in gold_map.items():
        email = email_map.get(email_id)
        if email is None:
            continue

        scores = grade_email(email, gold)

        active = {
            k
            for k in weights
            if gold.get(_gold_key(k)) is not None or k in ("extraction", "reply")
        }

        if not gold.get("reply_requirements"):
            active.discard("reply")
        if not gold.get("required_fields"):
            active.discard("extraction")

        for dim in active:
            w = float(weights.get(dim, 0.0))
            total_possible += w
            total_earned += w * float(scores.get(dim, 0.0))

    if total_possible == 0:
        raw = 0.5
    else:
        raw = total_earned / total_possible

    soft_budget = task_data.get("soft_step_budget", state.max_steps)
    over = max(0, state.step - soft_budget)
    penalty = min(over * 0.01, 0.10)

    final_score = raw - penalty
    return _clamp_open_unit(final_score)


def _gold_key(dim: str) -> str:
    return {
        "classification": "category",
        "priority": "priority",
        "routing": "route",
    }.get(dim, dim)
