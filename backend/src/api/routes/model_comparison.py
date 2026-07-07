"""Rotas para comparação de modelos de predição."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.job_store import (
    create_job,
    get_job,
    set_job_completed,
    set_job_failed,
    set_job_running,
)
from src.api.pipeline_store import check_predictions_completed, set_comparison_completed
from src.services.model_comparison_service import compare_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model-comparison", tags=["model-comparison"])


def _run_comparison_job(job_id: str) -> None:
    """Executa a comparação de modelos em background."""
    set_job_running(job_id)
    try:
        report = compare_models()
        set_comparison_completed()
        set_job_completed(job_id, report)
    except Exception as exc:
        logger.exception("Model comparison job %s failed", job_id)
        set_job_failed(job_id, str(exc))


@router.post("/compare")
def run_comparison(background_tasks: BackgroundTasks):
    """
    Executa a comparação de modelos de predição.
    
    Requer que as predições tenham sido concluídas anteriormente.
    Calcula métricas (accuracy, precision, recall, F1-score) e gera gráficos.
    Retorna um job_id para consultar o status.
    """
    if not check_predictions_completed():
        raise HTTPException(
            status_code=400,
            detail="Predições devem ser concluídas antes da comparação. "
                   "Chame POST /pipeline/predict primeiro."
        )
    
    job_id = create_job()
    background_tasks.add_task(_run_comparison_job, job_id)
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
def get_comparison_job(job_id: str):
    """Retorna o status de um job de comparação."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }


@router.get("/report")
def get_comparison_report():
    """
    Retorna o relatório de comparação mais recente.
    
    Se não houver relatório, retorna 404.
    """
    report_path = Path("reports/model_comparison_report.json")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Relatório de comparação não encontrado. "
                   "Execute POST /model-comparison/compare primeiro."
        )
    
    import json
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao ler relatório: {exc}")
