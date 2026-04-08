"""Core InboxPilot environment."""
from __future__ import annotations
import copy, uuid
from typing import Any, Optional
from app.models import (
    EpisodeState, EmailMessage, Observation, InboxSummary,
    EmailPreview, RewardBreakdown,
    SelectEmail, ClassifyEmail, SetPriority, ExtractFields,
    RouteEmail, DraftReply, MarkResolved, RequestMoreInfo, FinishEpisode,
)
from app.tasks import load_task
from app.reward import compute_step_reward
from app.graders import grade_task

AVAILABLE_ACTIONS = [
    "SelectEmail", "ClassifyEmail", "SetPriority", "ExtractFields",
    "RouteEmail", "DraftReply", "MarkResolved", "RequestMoreInfo", "FinishEpisode",
]

_state: Optional[EpisodeState] = None
_task_data: Optional[dict] = None


def _gold_map() -> dict[str, dict]:
    if _task_data is None:
        return {}
    return {g["email_id"]: g for g in _task_data.get("gold", [])}


def reset(task_id: str = "easy") -> Observation:
    global _state, _task_data
    _task_data = load_task(task_id)
    emails = [EmailMessage(**e) for e in _task_data["emails"]]
    _state = EpisodeState(
        episode_id=str(uuid.uuid4()),
        task_id=task_id,
        emails=emails,
        max_steps=_task_data.get("max_steps", 30),
    )
    return _build_obs()


def step(action: dict) -> tuple[Observation, float, bool, dict]:
    """Returns (observation, reward, done, info)."""
    if _state is None:
        raise RuntimeError("Call reset() first.")

    _state.step += 1
    warnings: list[str] = []

    # Check max steps
    if _state.step >= _state.max_steps:
        _state.done = True
        _state.warnings.append("Max steps reached.")
        final_score = grade_task(_state, _task_data)
        obs = _build_obs()
        return obs, 0.0, True, {"final_score": final_score,
                                 "reward_breakdown": _state.reward_breakdown.model_dump()}

    action_type = action.get("action", "")
    # Key includes the currently selected email so same action type on a
    # different email is NOT treated as a repeat.
    selected_ctx = _state.selected_email_id or ""
    action_key = f"{selected_ctx}:{action_type}:{action.get('category') or action.get('level') or action.get('target_queue') or ''}"

    is_repeat = _state.action_counts.get(action_key, 0) > 0
    _state.action_counts[action_key] = _state.action_counts.get(action_key, 0) + 1

    result = _apply_action(action, is_repeat, warnings)
    result["soft_budget"] = _task_data.get("soft_step_budget", _state.max_steps)

    reward = compute_step_reward(_state, action_type, result, _gold_map())
    _update_reward_breakdown(action_type, reward)

    history_entry = f"Step {_state.step}: {action_type}"
    if result.get("invalid"):
        history_entry += " [INVALID]"
    elif result.get("repeat"):
        history_entry += " [REPEAT]"
    _state.history.append(history_entry)
    _state.warnings = warnings

    done = _state.done or action_type == "FinishEpisode"
    if done:
        _state.done = True

    final_score = grade_task(_state, _task_data) if done else None
    info = {"reward_breakdown": _state.reward_breakdown.model_dump()}
    if final_score is not None:
        info["final_score"] = final_score

    return _build_obs(), reward, done, info


def state() -> dict:
    if _state is None:
        return {}
    return {
        "episode_id": _state.episode_id,
        "task_id": _state.task_id,
        "step": _state.step,
        "max_steps": _state.max_steps,
        "selected_email_id": _state.selected_email_id,
        "done": _state.done,
        "reward_breakdown": _state.reward_breakdown.model_dump(),
        "triage_status": {
            e.id: {
                "category": e.assigned_category,
                "priority": e.assigned_priority,
                "route": e.assigned_route,
                "resolved": e.is_resolved,
            }
            for e in _state.emails
        },
    }


# ── internal ──────────────────────────────────────────────────────────────────

def _get_selected() -> Optional[EmailMessage]:
    if _state is None or _state.selected_email_id is None:
        return None
    return next((e for e in _state.emails if e.id == _state.selected_email_id), None)


def _apply_action(action: dict, is_repeat: bool, warnings: list[str]) -> dict:
    atype = action.get("action", "")

    if is_repeat:
        warnings.append(f"Repeated action: {atype}")
        return {"repeat": True}

    if atype == "SelectEmail":
        eid = action.get("email_id")
        email = next((e for e in _state.emails if e.id == eid), None)
        if email is None:
            warnings.append(f"No email with id {eid!r}")
            return {"invalid": True}
        _state.selected_email_id = eid
        email.is_read = True
        return {"email_id": eid}

    email = _get_selected()

    if atype == "ClassifyEmail":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        cat = action.get("category")
        email.assigned_category = cat
        return {"category": cat}

    if atype == "SetPriority":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        lvl = action.get("level")
        email.assigned_priority = lvl
        return {"level": lvl}

    if atype == "ExtractFields":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        fields = action.get("fields", {})
        email.extracted_fields.update(fields)
        return {"fields": fields}

    if atype == "RouteEmail":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        if email.assigned_category is None:
            warnings.append("Classify the email before routing.")
            return {"invalid": True}
        q = action.get("target_queue")
        email.assigned_route = q
        return {"target_queue": q}

    if atype == "DraftReply":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        email.draft_reply = action.get("reply_text", "")
        return {"reply_text": email.draft_reply}

    if atype == "MarkResolved":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        email.is_resolved = True
        return {"resolved": True}

    if atype == "RequestMoreInfo":
        if email is None:
            warnings.append("Select an email first.")
            return {"invalid": True}
        return {"question": action.get("question", "")}

    if atype == "FinishEpisode":
        _state.done = True
        return {"finished": True}

    warnings.append(f"Unknown action type: {atype!r}")
    return {"invalid": True}


def _build_obs() -> Observation:
    urgent_cats = {"urgent", "phishing"}
    unread  = sum(1 for e in _state.emails if not e.is_read)
    urgent  = sum(1 for e in _state.emails if e.assigned_category in urgent_cats)
    resolved= sum(1 for e in _state.emails if e.is_resolved)

    selected = _get_selected()

    return Observation(
        inbox_summary=InboxSummary(
            total=len(_state.emails),
            unread=unread,
            urgent=urgent,
            resolved=resolved,
        ),
        email_list=[
            EmailPreview(id=e.id, subject=e.subject, sender=e.sender,
                         is_read=e.is_read, is_resolved=e.is_resolved)
            for e in _state.emails
        ],
        selected_email=selected,
        available_actions=AVAILABLE_ACTIONS,
        history=list(_state.history[-10:]),
        task_goal=_task_data.get("goal", ""),
        warnings=list(_state.warnings),
        step=_state.step,
        done=_state.done,
    )


def _update_reward_breakdown(action_type: str, reward: float) -> None:
    rb = _state.reward_breakdown
    if reward < 0:
        rb.efficiency_penalty += reward
    elif action_type == "ClassifyEmail":
        rb.classification += reward
    elif action_type == "SetPriority":
        rb.priority += reward
    elif action_type == "RouteEmail":
        rb.routing += reward
    elif action_type == "ExtractFields":
        rb.extraction += reward
    elif action_type == "DraftReply":
        rb.reply += reward
    elif action_type == "MarkResolved":
        rb.resolution += reward
