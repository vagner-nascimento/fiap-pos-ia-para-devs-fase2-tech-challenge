"""Rotas do pipeline de preprocessamento, tuning e predições."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.api.job_store import (
    create_job,
    get_job,
    set_job_completed,
    set_job_failed,
    set_job_running,
)
from src.api.pipeline_store import (
    check_predictions_completed,
    check_preprocessing_completed,
    check_tuning_completed,
    get_pipeline_state,
    reset_pipeline,
    set_predictions_completed,
    set_preprocessing_completed,
    set_tuning_completed,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class TuningRequest(BaseModel):
    """Request schema for tuning parameters."""
    pop_size: int = Field(default=4, ge=2, description="Population size per algorithm type")
    max_generations: int = Field(default=2, ge=1, description="Maximum number of generations")
    patience: int = Field(default=3, ge=1, description="Patience for early stopping")
    k_folds: int = Field(default=3, ge=2, description="Number of folds for cross-validation")
    aggressiveness: Literal["low", "medium", "high"] = Field(default="medium", description="Mutation aggressiveness")
    elitism: bool = Field(default=True, description="Enable elitism")
    crossover_probability: float = Field(default=0.7, ge=0.0, le=1.0, description="Crossover probability")
    mutation_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="Mutation probability")
    individual_mutation_probability: float = Field(default=0.5, ge=0.0, le=1.0, description="Individual gene mutation probability")
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")


# Fixed paths
RAW_CSV_PATH = "data/raw/estado_nutricional_sao_paulo.csv"
PROCESSED_CSV_PATH = "data/processed/estado_nutricional_clean.csv"
MAPPINGS_PATH = "models/artifacts/mappings.json"
MODEL_PATH = "models/artifacts/best_model.joblib"
PREDICTIONS_PATH = "models/artifacts/predictions.csv"


def _run_preprocessing_job(job_id: str) -> None:
    """Executa o script de preprocessing em background."""
    set_job_running(job_id)
    try:
        # Reset pipeline state when preprocessing starts
        reset_pipeline()
        
        cmd = [
            sys.executable,
            "scripts/run_preprocessing.py",
            "--input", RAW_CSV_PATH,
            "--output", PROCESSED_CSV_PATH,
            "--output-mappings", MAPPINGS_PATH,
        ]
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent.parent,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"Preprocessing failed: {result.stderr}")
        
        # Read the first 10 rows and mappings for response
        df = pd.read_csv(PROCESSED_CSV_PATH)
        first_10_rows = df.head(10).to_dict(orient="records")
        
        with open(MAPPINGS_PATH, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        set_preprocessing_completed()
        
        set_job_completed(job_id, {
            "first_10_rows": first_10_rows,
            "mappings": mappings,
            "processed_csv_path": PROCESSED_CSV_PATH,
            "mappings_path": MAPPINGS_PATH,
            "total_rows": len(df),
        })
    except Exception as exc:
        logger.exception("Preprocessing job %s failed", job_id)
        set_job_failed(job_id, str(exc))


def _run_tuning_job(job_id: str, params: TuningRequest) -> None:
    """Executa o script de tuning em background."""
    set_job_running(job_id)
    try:
        cmd = [
            sys.executable,
            "scripts/run_tuning.py",
            "--input", PROCESSED_CSV_PATH,
            "--output-model", MODEL_PATH,
            "--target", "TARGET",
            "--pop-size", str(params.pop_size),
            "--max-generations", str(params.max_generations),
            "--patience", str(params.patience),
            "--k-folds", str(params.k_folds),
            "--aggressiveness", params.aggressiveness,
            "--elitism", str(params.elitism).lower(),
            "--cxpb", str(params.crossover_probability),
            "--mutpb", str(params.mutation_probability),
            "--indpb", str(params.individual_mutation_probability),
            "--random-seed", str(params.random_seed),
        ]
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent.parent,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"Tuning failed: {result.stderr}")
        
        set_tuning_completed()
        
        set_job_completed(job_id, {
            "model_path": MODEL_PATH,
        })
    except Exception as exc:
        logger.exception("Tuning job %s failed", job_id)
        set_job_failed(job_id, str(exc))


def _run_predictions_job(job_id: str) -> None:
    """Executa o script de predictions em background."""
    set_job_running(job_id)
    try:
        cmd = [
            sys.executable,
            "scripts/run_predictions.py",
            "--input", PROCESSED_CSV_PATH,
            "--model", MODEL_PATH,
            "--output", PREDICTIONS_PATH,
            "--target", "TARGET",
        ]
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent.parent,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"Predictions failed: {result.stderr}")
        
        set_predictions_completed()
        
        set_job_completed(job_id, {
            "predictions_path": PREDICTIONS_PATH,
        })
    except Exception as exc:
        logger.exception("Predictions job %s failed", job_id)
        set_job_failed(job_id, str(exc))


@router.post("/preprocess")
def run_preprocessing(background_tasks: BackgroundTasks):
    """
    Executa o preprocessing dos dados brutos.
    
    Requer que o arquivo data/raw/estado_nutricional_sao_paulo.csv exista.
    Retorna um job_id para consultar o status.
    """
    # Check if raw file exists
    raw_path = Path(RAW_CSV_PATH)
    if not raw_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo bruto não encontrado: {RAW_CSV_PATH}. "
                   f"Por favor, coloque o arquivo CSV em data/raw/ antes de executar o preprocessing."
        )
    
    job_id = create_job()
    background_tasks.add_task(_run_preprocessing_job, job_id)
    return {"job_id": job_id, "status": "pending"}


@router.post("/tune")
def run_tuning(params: TuningRequest, background_tasks: BackgroundTasks):
    """
    Executa o tuning genético com parâmetros configuráveis.
    
    Requer que o preprocessing tenha sido concluído anteriormente.
    Retorna um job_id para consultar o status.
    """
    if not check_preprocessing_completed():
        raise HTTPException(
            status_code=400,
            detail="Preprocessing deve ser concluído antes do tuning. "
                   "Chame POST /pipeline/preprocess primeiro."
        )
    
    job_id = create_job()
    background_tasks.add_task(_run_tuning_job, job_id, params)
    return {"job_id": job_id, "status": "pending"}


@router.post("/predict")
def run_predictions(background_tasks: BackgroundTasks):
    """
    Executa as predições usando o modelo treinado.
    
    Requer que o tuning tenha sido concluído anteriormente.
    Retorna um job_id para consultar o status.
    """
    if not check_tuning_completed():
        raise HTTPException(
            status_code=400,
            detail="Tuning deve ser concluído antes das predições. "
                   "Chame POST /pipeline/tune primeiro."
        )
    
    job_id = create_job()
    background_tasks.add_task(_run_predictions_job, job_id)
    return {"job_id": job_id, "status": "pending"}


@router.get("/status")
def get_pipeline_status():
    """Retorna o estado atual do pipeline."""
    return get_pipeline_state()


@router.get("/jobs/{job_id}")
def get_pipeline_job(job_id: str):
    """Retorna o status de um job do pipeline."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }
