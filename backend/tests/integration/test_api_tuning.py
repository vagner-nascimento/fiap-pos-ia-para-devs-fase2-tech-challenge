"""Testes de integração da API REST de tuning."""

from __future__ import annotations

from pathlib import Path

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

    def test_run_tuning_sync(self, api_client):
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

    def test_run_tuning_dataset_not_found(self, api_client):
        response = api_client.post(
            "/tuning/run",
            json={"dataset": "inexistente.csv", "async_mode": False},
        )
        assert response.status_code == 404

    def test_get_latest_logs_after_run(self, api_client):
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
        assert len(data["history"]["generations_stats"]) == 2

    def test_async_job_lifecycle(self, api_client):
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
