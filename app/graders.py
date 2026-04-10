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


def _clamp(score: float) -> float:
    """Clamp to strictly open interval (0.001, 0.999)."""
    return round(max(0.001, min(0.999, float(score))), 4)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _soft_match(actual: Any, expected: Any) -> float:
    a = _normalize_text(actual)
    e = _normalize_text(expected)

    if not e:
        return 0.0
    if a == e:
        return 1.0
    if not a:
        return 0.0
    if a in e or e in a:
        return 0.5
    return 0.0


def grade_reply(reply: str, requirements: dict[str, Any]) -> float:
    """
    requirements keys (all optional):
      required_groups: list[list[str]]
      forbidden_phrases: list[str]
      required_ids: list[str]
    Returns score in (0.001, 0.999).
    """
    if not reply:
        return 0.001

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

    return _clamp(score)


def grade_email(email: EmailMessage, gold: dict[str, Any]) -> dict[str, float]:
    """Grade a single email against gold standard."""
    result: dict[str, float] = {
        "classification": 0.0,
        "priority": 0.0,
        "routing": 0.0,
        "extraction": 0.0,
        "reply": 0.0,
        "resolution": 0.0,
    }

    if gold.get("category") is not None:
        result["classification"] = _soft_match(
            email.assigned_category, gold.get("category")
        )

    if gold.get("priority") is not None:
        result["priority"] = _soft_match(
            email.assigned_priority, gold.get("priority")
        )

    if gold.get("route") is not None:
        result["routing"] = _soft_match(
            email.assigned_route, gold.get("route")
        )

    required_fields: dict[str, str] = gold.get("required_fields", {})
    if required_fields:
        correct = 0.0
        total = len(required_fields)

        for k, v in required_fields.items():
            actual = email.extracted_fields.get(k, "")
            if _normalize_text(actual) == _normalize_text(v):
                correct += 1.0
            elif _normalize_text(actual):
                correct += 0.5

        result["extraction"] = correct / total

    reply_reqs = gold.get("reply_requirements")
    if reply_reqs:
        result["reply"] = grade_reply(email.draft_reply or "", reply_reqs)

    if "resolved" in gold:
        expected = bool(gold.get("resolved"))
        actual = bool(email.is_resolved)
        result["resolution"] = 1.0 if actual == expected else 0.0

    return result


def _gold_key(dim: str) -> str:
    return {
        "classification": "category",
        "priority": "priority",
        "routing": "route",
        "resolution": "resolved",
    }.get(dim, dim)


def grade_task(state: EpisodeState, task_data: dict[str, Any]) -> float:
    """
    Master grader. Returns a score strictly in (0.001, 0.999).
    """
    gold_map: dict[str, dict[str, Any]] = {
        g["email_id"]: g for g in task_data.get("gold", []) if "email_id" in g
    }
    email_map: dict[str, EmailMessage] = {e.id: e for e in state.emails}

    if not gold_map:
        return 0.001

    weights = task_data.get(
        "grader_weights",
        {
            "classification": 0.20,
            "priority": 0.15,
            "routing": 0.20,
            "extraction": 0.15,
            "reply": 0.15,
            "resolution": 0.15,
        },
    )

    total_possible = 0.0
    total_earned = 0.0

    for email_id, gold in gold_map.items():
        email = email_map.get(email_id)
        if email is None:
            continue

        scores = grade_email(email, gold)

        active: set[str] = set()

        for dim in ("classification", "priority", "routing", "resolution"):
            if _gold_key(dim) in gold:
                active.add(dim)

        if gold.get("required_fields"):
            active.add("extraction")

        if gold.get("reply_requirements"):
            active.add("reply")

        for dim in active:
            w = float(weights.get(dim, 0.0))
            if w <= 0:
                continue
            total_possible += w
            total_earned += w * float(scores.get(dim, 0.0))

    if total_possible <= 0:
        return 0.001

    raw = total_earned / total_possible

    soft_budget = int(task_data.get("soft_step_budget", state.max_steps))
    over = max(0, int(state.step) - soft_budget)
    penalty = min(over * 0.01, 0.10)

    invalid_action_penalty = min(len(getattr(state, "warnings", [])) * 0.005, 0.05)

    final_score = raw - penalty - invalid_action_penalty
    return _clamp(final_score)
