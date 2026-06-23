"""Rotas de tuning genético."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.job_store import (
    create_job,
    get_job,
    set_job_completed,
    set_job_failed,
    set_job_running,
)
from src.api.schemas.tuning import (
    LatestLogsResponse,
    TuningJobCreated,
    TuningJobStatus,
    TuningRunRequest,
    TuningRunResponse,
)
from src.services import tuning_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tuning", tags=["tuning"])


def _run_tuning_job(job_id: str, params: dict) -> None:
    set_job_running(job_id)
    try:
        result = tuning_service.run_tuning(**params)
        set_job_completed(job_id, result)
    except Exception as exc:
        logger.exception("Job %s falhou", job_id)
        set_job_failed(job_id, str(exc))


@router.get("/datasets")
def list_datasets():
    return {"datasets": tuning_service.list_datasets()}


@router.post("/run", response_model=TuningRunResponse | TuningJobCreated)
def run_tuning(request: TuningRunRequest, background_tasks: BackgroundTasks):
    params = request.model_dump(exclude={"async_mode"})

    if request.async_mode:
        job_id = create_job()
        background_tasks.add_task(_run_tuning_job, job_id, params)
        return TuningJobCreated(job_id=job_id)

    try:
        result = tuning_service.run_tuning(**params)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TuningRunResponse(**result)


@router.get("/jobs/{job_id}", response_model=TuningJobStatus)
def get_tuning_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")

    result = None
    if job["result"] is not None:
        result = TuningRunResponse(**job["result"])

    return TuningJobStatus(
        job_id=job_id,
        status=job["status"],
        result=result,
        error=job["error"],
    )


@router.get("/logs/latest", response_model=LatestLogsResponse)
def get_latest_logs():
    try:
        logs = tuning_service.get_latest_logs()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LatestLogsResponse(**logs)
