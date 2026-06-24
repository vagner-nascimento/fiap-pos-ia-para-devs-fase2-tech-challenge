"""Schemas Pydantic para endpoints do agente LLM."""

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    initial_report: str
    row_count: int


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
