"""Schemas Pydantic para endpoints de tuning."""

from typing import Literal

from pydantic import BaseModel, Field


class TuningRunRequest(BaseModel):
    dataset: str = Field(..., description="Nome ou caminho do CSV processado")
    target_col: str = "TARGET"
    pop_size: int = Field(20, ge=2, le=100)
    max_generations: int = Field(10, ge=2, le=50)
    patience: int = Field(5, ge=1, le=20)
    k_folds: Literal[3, 5, 10] = 5
    aggressiveness: Literal["low", "medium", "high"] = "medium"
    elitism: bool = True
    cxpb: float = Field(0.7, ge=0.1, le=1.0)
    mutpb: float = Field(0.3, ge=0.05, le=1.0)
    indpb: float = Field(0.5, ge=0.1, le=1.0)
    random_seed: int = Field(42, ge=0)
    async_mode: bool = Field(False, description="Se True, retorna job_id para polling")


class TuningRunResponse(BaseModel):
    generations_stats: list
    best_individual: dict | None
    stopped_at: int | None
    reason: str | None
    params: dict


class TuningJobCreated(BaseModel):
    job_id: str
    status: str = "pending"


class TuningJobStatus(BaseModel):
    job_id: str
    status: str
    result: TuningRunResponse | None = None
    error: str | None = None


class LatestLogsResponse(BaseModel):
    history: dict
    stats_csv: str
