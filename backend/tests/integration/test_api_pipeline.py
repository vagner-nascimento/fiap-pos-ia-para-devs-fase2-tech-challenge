"""Testes de integração da API REST de pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.job_store import create_job

@pytest.fixture
def api_client():
    return TestClient(app)

class TestPipelineEndpoints:
    def test_pipeline_status(self, api_client):
        response = api_client.get("/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert "preprocessing_completed" in data
        assert "tuning_completed" in data
        assert "predictions_completed" in data

    def test_get_job_not_found(self, api_client):
        response = api_client.get("/pipeline/jobs/nonexistent")
        assert response.status_code == 404

    def test_get_logs_job_not_found(self, api_client):
        response = api_client.get("/pipeline/jobs/nonexistent/logs")
        assert response.status_code == 404

    def test_get_logs_file_not_found(self, api_client):
        job_id = create_job()
        response = api_client.get(f"/pipeline/jobs/{job_id}/logs")
        assert response.status_code == 200
        assert response.json() == {"job_id": job_id, "logs": ""}

    def test_get_logs_success(self, api_client):
        job_id = create_job()
        log_path = Path(f"/tmp/preprocessing_{job_id}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Line 1 of preprocessing logs\nLine 2 of preprocessing logs")
            
            response = api_client.get(f"/pipeline/jobs/{job_id}/logs")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == job_id
            assert "Line 1 of preprocessing logs" in data["logs"]
            assert "Line 2 of preprocessing logs" in data["logs"]
        finally:
            if log_path.exists():
                log_path.unlink()
