"""Deterministic graders for each task type. No LLM involved."""
from __future__ import annotations

import re
from typing import Any

from app.models import EmailMessage, EpisodeState


def _has_phrase_from_group(text: str, group: list[str]) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in group)


def _extract_id_from_text(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text or "", re.I))


def _safe_score(raw: float) -> float:
    """Guarantee a score strictly inside (0, 1)."""
    x = float(raw)
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 2)


def grade_reply(reply: str, requirements: dict[str, Any]) -> float:
    if not reply:
        return 0.25

    score = 0.75

    groups = requirements.get("required_groups", [])
    if groups:
        per_group = 0.75 / len(groups)
        for group in groups:
            if _has_phrase_from_group(reply, group):
                score += per_group

    for phrase in requirements.get("forbidden_phrases", []):
        if phrase.lower() in reply.lower():
            score = max(0.01, score - 0.25)

    for pattern in requirements.get("required_ids", []):
        if _extract_id_from_text(reply, pattern):
            score = min(0.99, score + 0.15)

    return _safe_score(score)


def grade_email(email: EmailMessage, gold: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {
        "classification": 0.25,
        "priority": 0.25,
        "routing": 0.25,
        "extraction": 0.25,
        "reply": 0.25,
    }

    if gold.get("category") is not None:
        result["classification"] = 0.75 if email.assigned_category == gold["category"] else 0.25

    if gold.get("priority") is not None:
        result["priority"] = 0.75 if email.assigned_priority == gold["priority"] else 0.25

    if gold.get("route") is not None:
        result["routing"] = 0.75 if email.assigned_route == gold["route"] else 0.25

    required_fields: dict[str, str] = gold.get("required_fields", {})
    if required_fields:
        correct = sum(
            1
            for k, v in required_fields.items()
            if str(email.extracted_fields.get(k, "")).strip().lower() == str(v).strip().lower()
        )
        pct_correct = correct / len(required_fields)
        result["extraction"] = _safe_score(pct_correct * 0.75 + 0.25)

    reply_reqs = gold.get("reply_requirements")
    if reply_reqs:
        result["reply"] = grade_reply(email.draft_reply or "", reply_reqs)

    return result


def grade_task(state: EpisodeState, task_data: dict[str, Any]) -> float:
    try:
        gold_items = task_data.get("gold", [])
        if not isinstance(gold_items, list) or len(gold_items) == 0:
            return 0.50

        gold_map: dict[str, dict[str, Any]] = {
            g["email_id"]: g for g in gold_items
            if isinstance(g, dict) and "email_id" in g
        }
        email_map: dict[str, EmailMessage] = {e.id: e for e in state.emails}

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

        total_possible = 0.0
        total_earned = 0.0

        for email_id, gold in gold_map.items():
            email = email_map.get(email_id)
            if email is None:
                continue

            scores = grade_email(email, gold)

            active = set()
            if gold.get("category") is not None:
                active.add("classification")
            if gold.get("priority") is not None:
                active.add("priority")
            if gold.get("route") is not None:
                active.add("routing")
            if gold.get("required_fields"):
                active.add("extraction")
            if gold.get("reply_requirements"):
                active.add("reply")

            for dim in active:
                w = float(weights.get(dim, 0.0))
                total_possible += w
                total_earned += w * float(scores.get(dim, 0.25))

        if total_possible == 0:
            return 0.50

        raw = total_earned / total_possible

        soft_budget = task_data.get("soft_step_budget", getattr(state, "max_steps", 30))
        over = max(0, getattr(state, "step", 0) - soft_budget)
        penalty = min(over * 0.01, 0.10)

        final = raw - penalty
        return _safe_score(final)

    except Exception:
        return 0.50
