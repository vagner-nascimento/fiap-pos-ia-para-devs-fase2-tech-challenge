"""Testes de integração da API REST de tuning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification

from src.api.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "processed"
    log_dir = tmp_path / "models" / "logs"
    data_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["TARGET"] = y
    csv_path = data_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_PATH", str(log_dir))

    import src.services.tuning_service as tuning_service

    monkeypatch.setattr(tuning_service, "DATA_PATH", tmp_path / "data")
    monkeypatch.setattr(tuning_service, "LOG_PATH", log_dir)
    monkeypatch.setattr(tuning_service, "PROCESSED_DIR", data_dir)

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestTuningEndpoints:
    def test_list_datasets(self, api_client):
        response = api_client.get("/tuning/datasets")
        assert response.status_code == 200
        assert "test_data.csv" in response.json()["datasets"]

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    def test_run_tuning_sync(self, mock_run_tuning, api_client):
        mock_run_tuning.return_value = {
            "generations_stats": [
                {"generation": 1, "rf": {}, "knn": {}, "global_best_f1": 0.8},
                {"generation": 2, "rf": {}, "knn": {}, "global_best_f1": 0.85}
            ],
            "best_individual": {"type": "RF", "hyperparams": {}, "fitness_f1": 0.85, "fitness_acc": 0.9},
            "stopped_at": 2,
            "reason": "max_generations",
            "params": {}
        }
        payload = {
            "dataset": "test_data.csv",
            "target_col": "TARGET",
            "pop_size": 4,
            "max_generations": 2,
            "patience": 10,
            "k_folds": 3,
            "aggressiveness": "low",
            "elitism": True,
            "random_seed": 42,
            "async_mode": False,
        }
        response = api_client.post("/tuning/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "generations_stats" in data
        assert len(data["generations_stats"]) == 2
        assert data["best_individual"] is not None

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    def test_run_tuning_with_sample_size(self, mock_run_tuning, api_client):
        mock_run_tuning.return_value = {
            "generations_stats": [
                {"generation": 1, "rf": {}, "knn": {}, "global_best_f1": 0.8},
                {"generation": 2, "rf": {}, "knn": {}, "global_best_f1": 0.85}
            ],
            "best_individual": {"type": "RF", "hyperparams": {}, "fitness_f1": 0.85, "fitness_acc": 0.9},
            "stopped_at": 2,
            "reason": "max_generations",
            "params": {}
        }
        payload = {
            "dataset": "test_data.csv",
            "target_col": "TARGET",
            "pop_size": 4,
            "max_generations": 2,
            "patience": 10,
            "k_folds": 3,
            "aggressiveness": "low",
            "elitism": True,
            "random_seed": 42,
            "async_mode": False,
            "sample_size": 30,
        }
        response = api_client.post("/tuning/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "generations_stats" in data
        assert len(data["generations_stats"]) == 2
        assert data["best_individual"] is not None

    def test_run_tuning_dataset_not_found(self, api_client):
        response = api_client.post(
            "/tuning/run",
            json={"dataset": "inexistente.csv", "async_mode": False},
        )
        assert response.status_code == 404

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    @patch('src.api.routes.tuning.tuning_service.get_latest_logs')
    def test_get_latest_logs_after_run(self, mock_get_logs, mock_run_tuning, api_client):
        mock_run_tuning.return_value = {
            "generations_stats": [{"generation": 1, "rf": {}, "knn": {}, "global_best_f1": 0.8}],
            "best_individual": {"type": "RF", "hyperparams": {}, "fitness_f1": 0.8, "fitness_acc": 0.85},
            "stopped_at": 1,
            "reason": "max_generations",
            "params": {}
        }
        mock_get_logs.return_value = {
            "history": {
                "generations_stats": [{"generation": 1, "rf": {}, "knn": {}, "global_best_f1": 0.8}]
            },
            "stats_csv": "csv content"
        }
        payload = {
            "dataset": "test_data.csv",
            "target_col": "TARGET",
            "pop_size": 4,
            "max_generations": 2,
            "patience": 10,
            "k_folds": 3,
            "async_mode": False,
        }
        api_client.post("/tuning/run", json=payload)
        response = api_client.get("/tuning/logs/latest")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]["generations_stats"]) == 1

    @patch('src.api.routes.tuning._run_tuning_job')
    def test_async_job_lifecycle(self, mock_run_job, api_client):
        # Mock the background task to complete immediately
        def mock_job_func(job_id, params):
            from src.api.job_store import set_job_completed
            result = {
                "generations_stats": [{"generation": 1, "rf": {}, "knn": {}, "global_best_f1": 0.8}],
                "best_individual": {"type": "RF", "hyperparams": {}, "fitness_f1": 0.8, "fitness_acc": 0.85},
                "stopped_at": 1,
                "reason": "max_generations",
                "params": {}
            }
            set_job_completed(job_id, result)
        mock_run_job.side_effect = mock_job_func
        
        payload = {
            "dataset": "test_data.csv",
            "target_col": "TARGET",
            "pop_size": 4,
            "max_generations": 2,
            "patience": 10,
            "k_folds": 3,
            "async_mode": True,
        }
        create_resp = api_client.post("/tuning/run", json=payload)
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        status_resp = api_client.get(f"/tuning/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] in ("pending", "running", "completed", "failed")
        if status_data["status"] == "completed":
            assert status_data["result"] is not None
