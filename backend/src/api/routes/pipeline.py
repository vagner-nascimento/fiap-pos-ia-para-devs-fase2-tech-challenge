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
from src.data.ingest import extract_rar_file
from src.services import tuning_service

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
RAR_PATH = "data/raw/estado_nutricional_sao_paulo.rar"
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

        # Extract .rar file if CSV doesn't exist
        csv_path = Path(RAW_CSV_PATH)
        rar_path = Path(RAR_PATH)
        if not csv_path.exists() and rar_path.exists():
            logger.info(f"CSV não encontrado, extraindo de {RAR_PATH}")
            extract_rar_file(RAR_PATH, RAW_CSV_PATH)
        elif not csv_path.exists() and not rar_path.exists():
            raise FileNotFoundError(
                f"Arquivo bruto não encontrado: nem {RAW_CSV_PATH} nem {RAR_PATH} existem"
            )

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
    """Executa o tuning via tuning_service (direto, sem subprocess)."""
    set_job_running(job_id)
    try:
        result = tuning_service.run_tuning(
            dataset=PROCESSED_CSV_PATH,
            target_col="TARGET",
            pop_size=params.pop_size,
            max_generations=params.max_generations,
            patience=params.patience,
            k_folds=params.k_folds,
            aggressiveness=params.aggressiveness,
            elitism=params.elitism,
            cxpb=params.crossover_probability,
            mutpb=params.mutation_probability,
            indpb=params.individual_mutation_probability,
            random_seed=params.random_seed,
            save_logs=True,
            job_id=job_id,
        )
        set_tuning_completed()
        set_job_completed(job_id, {"model_path": MODEL_PATH, **result})
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

    Requer que o arquivo data/raw/estado_nutricional_sao_paulo.csv ou
    data/raw/estado_nutricional_sao_paulo.rar exista.
    Se fornecido .rar, o arquivo será extraído automaticamente.
    Retorna um job_id para consultar o status.
    """
    # Check if raw file exists (CSV or RAR)
    csv_path = Path(RAW_CSV_PATH)
    rar_path = Path(RAR_PATH)
    if not csv_path.exists() and not rar_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo bruto não encontrado: nem {RAW_CSV_PATH} nem {RAR_PATH} existem. "
                   f"Por favor, coloque o arquivo CSV ou .rar em data/raw/ antes de executar o preprocessing."
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
