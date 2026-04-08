"""Shaped reward computation called after every step."""
from __future__ import annotations
from typing import Optional
from app.models import EpisodeState, RewardBreakdown


# incremental reward deltas applied the moment a correct sub-decision is made
_DELTA = {
    "classification_correct": 0.15,
    "priority_correct":       0.10,
    "routing_correct":        0.20,
    "field_extracted":        0.05,   # per field
    "reply_slot_hit":         0.04,   # per required phrase group hit
    "email_resolved":         0.10,
    "invalid_action":        -0.05,
    "repeat_action":         -0.08,
    "phishing_mislabeled":   -0.12,
}


def compute_step_reward(
    state: EpisodeState,
    action_type: str,
    action_result: dict,
    gold_map: dict,
) -> float:
    """
    Returns a shaped reward for the current step.
    action_result is a dict produced by env._apply_action().
    """
    r = 0.0
    email_id = state.selected_email_id

    if action_result.get("invalid"):
        return _DELTA["invalid_action"]

    if action_result.get("repeat"):
        return _DELTA["repeat_action"]

    if action_type == "ClassifyEmail" and email_id:
        gold = gold_map.get(email_id, {})
        if gold.get("category") == action_result.get("category"):
            r += _DELTA["classification_correct"]
            # extra penalty if phishing is misclassified anywhere else
        elif gold.get("category") == "phishing":
            r += _DELTA["phishing_mislabeled"]

    if action_type == "SetPriority" and email_id:
        gold = gold_map.get(email_id, {})
        if gold.get("priority") == action_result.get("level"):
            r += _DELTA["priority_correct"]

    if action_type == "RouteEmail" and email_id:
        gold = gold_map.get(email_id, {})
        if gold.get("route") == action_result.get("target_queue"):
            r += _DELTA["routing_correct"]
        elif gold.get("category") == "phishing":
            r += _DELTA["phishing_mislabeled"]

    if action_type == "ExtractFields" and email_id:
        gold = gold_map.get(email_id, {})
        required = gold.get("required_fields", {})
        for k, v in action_result.get("fields", {}).items():
            if required.get(k, "").lower() == v.lower():
                r += _DELTA["field_extracted"]

    if action_type == "MarkResolved":
        r += _DELTA["email_resolved"]

    # step cost after soft budget
    soft = action_result.get("soft_budget", 30)
    if state.step > soft:
        r -= 0.02

    return round(r, 4)
