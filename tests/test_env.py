"""Smoke tests for the InboxPilot environment."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.env as env


def test_reset_easy():
    obs = env.reset("easy")
    assert obs.inbox_summary.total == 1
    assert obs.step == 0
    assert not obs.done

def test_full_easy_episode():
    obs = env.reset("easy")
    # Select
    obs, r, done, info = env.step({"action": "SelectEmail", "email_id": "email_001"})
    assert obs.selected_email is not None
    # Classify
    obs, r, done, info = env.step({"action": "ClassifyEmail", "category": "finance"})
    assert r > 0
    # Priority
    obs, r, done, info = env.step({"action": "SetPriority", "level": "high"})
    assert r > 0
    # Extract
    obs, r, done, info = env.step({"action": "ExtractFields", "fields": {"transaction_id": "TXN-2024-88341", "student_id": "STU-4421"}})
    # Route
    obs, r, done, info = env.step({"action": "RouteEmail", "target_queue": "finance_office"})
    assert r > 0
    # Resolve
    obs, r, done, info = env.step({"action": "MarkResolved"})
    # Finish
    obs, r, done, info = env.step({"action": "FinishEpisode"})
    assert done
    assert info["final_score"] > 0.5

def test_invalid_action_penalty():
    env.reset("easy")
    obs, r, done, info = env.step({"action": "ClassifyEmail", "category": "finance"})
    assert r < 0  # no email selected

def test_repeat_action_penalty():
    env.reset("easy")
    env.step({"action": "SelectEmail", "email_id": "email_001"})
    env.step({"action": "ClassifyEmail", "category": "finance"})
    obs, r, done, info = env.step({"action": "ClassifyEmail", "category": "finance"})
    assert r < 0

def test_reset_hard():
    obs = env.reset("hard")
    assert obs.inbox_summary.total == 6

def test_state():
    env.reset("easy")
    s = env.state()
    assert "episode_id" in s
    assert s["task_id"] == "easy"

if __name__ == "__main__":
    test_reset_easy()
    test_full_easy_episode()
    test_invalid_action_penalty()
    test_repeat_action_penalty()
    test_reset_hard()
    test_state()
    print("All tests passed!")
