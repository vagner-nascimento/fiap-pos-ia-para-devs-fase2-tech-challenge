"""Store in-memory para jobs assíncronos de tuning."""

from __future__ import annotations

import threading
import uuid
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def set_job_running(job_id: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"


def set_job_completed(job_id: str, result: dict) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result


def set_job_failed(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = error
