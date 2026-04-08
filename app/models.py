from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class SelectEmail(BaseModel):
    action: Literal["SelectEmail"] = "SelectEmail"
    email_id: str

class ClassifyEmail(BaseModel):
    action: Literal["ClassifyEmail"] = "ClassifyEmail"
    category: Literal["support","finance","placement","spam","phishing","urgent","general"]

class SetPriority(BaseModel):
    action: Literal["SetPriority"] = "SetPriority"
    level: Literal["low","medium","high"]

class ExtractFields(BaseModel):
    action: Literal["ExtractFields"] = "ExtractFields"
    fields: dict[str,str]

class RouteEmail(BaseModel):
    action: Literal["RouteEmail"] = "RouteEmail"
    target_queue: Literal["tech_support","finance_office","placement_cell","security","archive"]

class DraftReply(BaseModel):
    action: Literal["DraftReply"] = "DraftReply"
    reply_text: str

class MarkResolved(BaseModel):
    action: Literal["MarkResolved"] = "MarkResolved"

class RequestMoreInfo(BaseModel):
    action: Literal["RequestMoreInfo"] = "RequestMoreInfo"
    question: str

class FinishEpisode(BaseModel):
    action: Literal["FinishEpisode"] = "FinishEpisode"


class EmailMessage(BaseModel):
    id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    has_attachment: bool = False
    is_read: bool = False
    is_resolved: bool = False
    assigned_category: Optional[str] = None
    assigned_priority: Optional[str] = None
    assigned_route: Optional[str] = None
    extracted_fields: dict[str,str] = Field(default_factory=dict)
    draft_reply: Optional[str] = None


class EmailPreview(BaseModel):
    id: str
    subject: str
    sender: str
    is_read: bool
    is_resolved: bool

class InboxSummary(BaseModel):
    total: int
    unread: int
    urgent: int
    resolved: int

class Observation(BaseModel):
    inbox_summary: InboxSummary
    email_list: list[EmailPreview]
    selected_email: Optional[EmailMessage] = None
    available_actions: list[str]
    history: list[str] = Field(default_factory=list)
    task_goal: str
    warnings: list[str] = Field(default_factory=list)
    step: int = 0
    done: bool = False


class RewardBreakdown(BaseModel):
    classification: float = 0.0
    priority: float = 0.0
    routing: float = 0.0
    extraction: float = 0.0
    reply: float = 0.0
    resolution: float = 0.0
    efficiency_penalty: float = 0.0

    def total(self) -> float:
        return (self.classification + self.priority + self.routing +
                self.extraction + self.reply + self.resolution + self.efficiency_penalty)


class EpisodeState(BaseModel):
    episode_id: str
    task_id: str
    step: int = 0
    max_steps: int = 30
    selected_email_id: Optional[str] = None
    emails: list[EmailMessage]
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    history: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    done: bool = False
    action_counts: dict[str,int] = Field(default_factory=dict)
