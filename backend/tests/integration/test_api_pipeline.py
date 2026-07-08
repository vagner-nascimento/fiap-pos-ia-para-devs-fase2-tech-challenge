"""Testes de integração da API REST de pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.job_store import create_job
from src.api.pipeline_store import set_preprocessing_completed, set_tuning_completed

@pytest.fixture
def api_client():
    return TestClient(app)

class TestPipelineEndpoints:
    def test_pipeline_status(self, api_client):
        response = api_client.get("/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert "preprocessing_completed" in data

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
                f.write("Line 1\nLine 2")
            response = api_client.get(f"/pipeline/jobs/{job_id}/logs")
            assert response.status_code == 200
            assert "Line 1" in response.json()["logs"]
        finally:
            if log_path.exists():
                log_path.unlink()

    @patch('src.api.routes.pipeline.Path.exists')
    def test_preprocess_raw_file_not_found(self, mock_exists, api_client):
        # Simulate RAW_CSV_PATH and RAR_PATH not existing
        mock_exists.return_value = False
        response = api_client.post("/pipeline/preprocess")
        assert response.status_code == 404

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('src.api.routes.pipeline.Path.exists')
    @patch('subprocess.Popen')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"mappings": {}}')
    def test_preprocess_success(self, mock_file, mock_read_csv, mock_popen, mock_exists, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        mock_exists.side_effect = lambda: True
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        mock_df = MagicMock()
        mock_df.head.return_value.to_dict.return_value = []
        mock_read_csv.return_value = mock_df
        
        response = api_client.post("/pipeline/preprocess")
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        
        # Verify job is completed
        job_id = data["job_id"]
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.status_code == 200
        assert job_resp.json()["status"] == "completed"

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('src.api.routes.pipeline.Path.exists')
    @patch('subprocess.Popen')
    def test_preprocess_failed(self, mock_popen, mock_exists, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        mock_exists.side_effect = lambda: True
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_process.returncode = 1
        mock_popen.return_value = mock_process
        
        response = api_client.post("/pipeline/preprocess")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.json()["status"] == "failed"

    def test_tune_without_preprocess_failed(self, api_client):
        from src.api.pipeline_store import reset_pipeline
        reset_pipeline()
        response = api_client.post("/pipeline/tune", json={})
        assert response.status_code == 400

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('src.api.routes.pipeline.tuning_service.run_tuning')
    def test_tune_success(self, mock_run_tuning, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        set_preprocessing_completed()
        
        mock_run_tuning.return_value = {"best_individual": {}}
        
        response = api_client.post("/pipeline/tune", json={
            "pop_size": 2,
            "max_generations": 1,
            "patience": 1,
            "k_folds": 2,
        })
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.json()["status"] == "completed"

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('src.api.routes.pipeline.tuning_service.run_tuning')
    def test_tune_failed(self, mock_run_tuning, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        set_preprocessing_completed()
        
        mock_run_tuning.side_effect = Exception("Tuning error")
        
        response = api_client.post("/pipeline/tune", json={})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.json()["status"] == "failed"

    def test_predict_without_tuning_failed(self, api_client):
        from src.api.pipeline_store import reset_pipeline
        reset_pipeline()
        response = api_client.post("/pipeline/predict")
        assert response.status_code == 400

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('subprocess.run')
    def test_predict_success(self, mock_sub_run, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        set_tuning_completed()
        
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_sub_run.return_value = mock_res
        
        response = api_client.post("/pipeline/predict")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.json()["status"] == "completed"

    @patch('fastapi.BackgroundTasks.add_task')
    @patch('subprocess.run')
    def test_predict_failed(self, mock_sub_run, mock_add_task, api_client):
        mock_add_task.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        set_tuning_completed()
        
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = "Prediction script error"
        mock_sub_run.return_value = mock_res
        
        response = api_client.post("/pipeline/predict")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        job_resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert job_resp.json()["status"] == "failed"
