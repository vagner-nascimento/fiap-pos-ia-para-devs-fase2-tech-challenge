"""Store in-memory para sessões do agente LLM."""

from __future__ import annotations

import uuid

from src.agents.nutritional_agent import NutritionalHealthAgent

_sessions: dict[str, NutritionalHealthAgent] = {}


def create_session(agent: NutritionalHealthAgent) -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = agent
    return session_id


def get_session(session_id: str) -> NutritionalHealthAgent | None:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None
