"""Tests for deterministic graders."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graders import grade_reply, grade_email
from app.models import EmailMessage


def test_grade_reply_perfect():
    reply = "We apologize for the inconvenience. Please provide a screenshot of the portal error. Our support team will assist you shortly."
    reqs = {
        "required_groups": [
            ["sorry","understand","apologize"],
            ["portal","certificate","access"],
            ["please provide","provide","screenshot"],
            ["support team","tech support"]
        ],
        "forbidden_phrases": ["guaranteed","instantly"],
    }
    score = grade_reply(reply, reqs)
    assert score == 1.0

def test_grade_reply_missing_group():
    reply = "We apologize. Our support team will assist you."
    reqs = {
        "required_groups": [
            ["sorry","apologize"],
            ["portal","certificate"],
            ["screenshot","attach"],
        ]
    }
    score = grade_reply(reply, reqs)
    assert 0 < score < 1.0

def test_grade_reply_forbidden():
    reply = "We apologize. The issue will be instantly resolved by our support team."
    reqs = {
        "required_groups": [["apologize"]],
        "forbidden_phrases": ["instantly"],
    }
    score = grade_reply(reply, reqs)
    assert score < 1.0

def test_grade_email_all_correct():
    email = EmailMessage(
        id="e1", subject="test", sender="a@b.com", sender_domain="b.com", body="",
        assigned_category="finance", assigned_priority="high",
        assigned_route="finance_office",
        extracted_fields={"transaction_id": "TXN-2024-88341", "student_id": "STU-4421"}
    )
    gold = {
        "category": "finance", "priority": "high", "route": "finance_office",
        "required_fields": {"transaction_id": "TXN-2024-88341", "student_id": "STU-4421"}
    }
    scores = grade_email(email, gold)
    assert scores["classification"] == 1.0
    assert scores["priority"] == 1.0
    assert scores["routing"] == 1.0
    assert scores["extraction"] == 1.0

if __name__ == "__main__":
    test_grade_reply_perfect()
    test_grade_reply_missing_group()
    test_grade_reply_forbidden()
    test_grade_email_all_correct()
    print("All grader tests passed!")
