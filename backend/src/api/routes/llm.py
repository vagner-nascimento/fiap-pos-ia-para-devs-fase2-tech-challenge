"""Rotas do agente LLM nutricional."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.agents.nutritional_agent import NutritionalHealthAgent
from src.api.schemas.llm import ChatRequest, ChatResponse, SessionCreateResponse
from src.api.session_store import create_session, delete_session, get_session

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/session", response_model=SessionCreateResponse)
async def create_llm_session(
    file: UploadFile = File(...),
    mappings: str = Form("{}"),
):
    try:
        mappings_dict = json.loads(mappings) if mappings else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="mappings deve ser JSON válido") from exc

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {exc}") from exc

    try:
        agent = NutritionalHealthAgent(df, mappings_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao inicializar agente: {exc}") from exc

    session_id = create_session(agent)
    return SessionCreateResponse(
        session_id=session_id,
        initial_report=agent.initial_report,
        row_count=len(df),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    agent = get_session(request.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Sessão não encontrada: {request.session_id}")

    try:
        answer = agent.ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer)


@router.delete("/session/{session_id}")
def close_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Sessão não encontrada: {session_id}")
    return {"status": "closed", "session_id": session_id}
