"""Aplicação FastAPI — ponto de entrada da API REST."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, llm, pipeline, tuning

load_dotenv()

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)

app = FastAPI(
    title="SISVAN Nutricional API",
    description="API REST para tuning genético e agente de saúde nutricional",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(tuning.router)
app.include_router(pipeline.router)
app.include_router(llm.router)
