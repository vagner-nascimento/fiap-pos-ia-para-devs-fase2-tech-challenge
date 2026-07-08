"""Testes de integração para as rotas da API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from src.api.main import app
from src.api.pipeline_store import (
    set_preprocessing_completed,
    set_tuning_completed,
    set_predictions_completed,
    reset_pipeline,
)
from src.api.job_store import create_job


@pytest.fixture
def api_client():
    return TestClient(app)


class TestHealthRoute:
    def test_health_endpoint(self, api_client):
        """Testa endpoint de health check."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestLLMRoutes:
    @patch('src.api.routes.llm.NutritionalHealthAgent')
    @patch('src.api.routes.llm.create_session')
    def test_create_session_success(self, mock_create_session, mock_agent_class, api_client):
        """Testa criação de sessão LLM com sucesso."""
        mock_agent = MagicMock()
        mock_agent.initial_report = "Relatório inicial"
        mock_agent_class.return_value = mock_agent
        mock_create_session.return_value = "session_123"
        
        # Create a mock CSV file
        csv_content = "col1,col2\n1,2\n"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        data = {"mappings": "{}"}
        
        response = api_client.post("/llm/session", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["session_id"] == "session_123"
        assert result["initial_report"] == "Relatório inicial"

    def test_create_session_invalid_json(self, api_client):
        """Testa erro quando mappings JSON é inválido."""
        csv_content = "col1,col2\n1,2\n"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        data = {"mappings": "invalid json"}
        
        response = api_client.post("/llm/session", files=files, data=data)
        
        assert response.status_code == 400
        assert "mappings deve ser JSON válido" in response.json()["detail"]

    @patch('src.api.routes.llm.NutritionalHealthAgent')
    @patch('src.api.routes.llm.create_session')
    def test_create_session_invalid_csv(self, mock_create_session, mock_agent_class, api_client):
        """Testa erro quando CSV é inválido."""
        mock_agent_class.side_effect = Exception("Erro ao ler CSV")
        
        csv_content = "invalid csv content"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        data = {"mappings": "{}"}
        
        response = api_client.post("/llm/session", files=files, data=data)
        
        assert response.status_code == 400

    @patch('src.api.routes.llm.get_session')
    def test_chat_success(self, mock_get_session, api_client):
        """Testa endpoint de chat com sucesso."""
        mock_agent = MagicMock()
        mock_agent.ask.return_value = "Resposta do agente"
        mock_get_session.return_value = mock_agent
        
        response = api_client.post(
            "/llm/chat",
            json={"session_id": "session_123", "question": "Qual a média de idade?"}
        )
        
        assert response.status_code == 200
        assert response.json()["answer"] == "Resposta do agente"

    @patch('src.api.routes.llm.get_session')
    def test_chat_session_not_found(self, mock_get_session, api_client):
        """Testa erro quando sessão não existe."""
        mock_get_session.return_value = None
        
        response = api_client.post(
            "/llm/chat",
            json={"session_id": "nonexistent", "question": "Teste"}
        )
        
        assert response.status_code == 404

    @patch('src.api.routes.llm.delete_session')
    def test_close_session_success(self, mock_delete_session, api_client):
        """Testa fechamento de sessão com sucesso."""
        mock_delete_session.return_value = True
        
        response = api_client.delete("/llm/session/session_123")
        
        assert response.status_code == 200
        assert response.json()["status"] == "closed"

    @patch('src.api.routes.llm.delete_session')
    def test_close_session_not_found(self, mock_delete_session, api_client):
        """Testa erro ao fechar sessão inexistente."""
        mock_delete_session.return_value = False
        
        response = api_client.delete("/llm/session/nonexistent")
        
        assert response.status_code == 404


class TestModelComparisonRoutes:
    @patch('src.api.routes.model_comparison.compare_models')
    @patch('src.api.routes.model_comparison.check_predictions_completed')
    @patch('src.api.routes.model_comparison.create_job')
    def test_run_comparison_success(
        self, mock_create_job, mock_check_completed, mock_compare, api_client
    ):
        """Testa execução de comparação com sucesso."""
        mock_check_completed.return_value = True
        mock_create_job.return_value = "job_123"
        mock_compare.return_value = {"report": "comparison report"}
        
        response = api_client.post("/model-comparison/compare")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == "job_123"
        assert response.json()["status"] == "pending"

    @patch('src.api.routes.model_comparison.check_predictions_completed')
    def test_run_comparison_predictions_not_completed(self, mock_check_completed, api_client):
        """Testa erro quando predições não foram concluídas."""
        mock_check_completed.return_value = False
        
        response = api_client.post("/model-comparison/compare")
        
        assert response.status_code == 400
        assert "Predições devem ser concluídas" in response.json()["detail"]

    def test_get_comparison_job_not_found(self, api_client):
        """Testa busca de job de comparação inexistente."""
        response = api_client.get("/model-comparison/jobs/nonexistent")
        
        assert response.status_code == 404

    def test_get_comparison_job_success(self, api_client):
        """Testa busca de job de comparação existente."""
        job_id = create_job()
        
        response = api_client.get(f"/model-comparison/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id

    def test_get_comparison_report_not_found(self, api_client):
        """Testa erro quando relatório não existe."""
        response = api_client.get("/model-comparison/report")
        
        assert response.status_code == 404

    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_get_comparison_report_success(self, mock_exists, mock_open, api_client):
        """Testa busca de relatório existente."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = '{"report": "test"}'
        
        response = api_client.get("/model-comparison/report")
        
        assert response.status_code == 200
        assert response.json()["report"] == "test"


class TestPipelineRoutes:
    def test_get_pipeline_status(self, api_client):
        """Testa busca de status do pipeline."""
        response = api_client.get("/pipeline/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "preprocessing_completed" in data
        assert "tuning_completed" in data
        assert "predictions_completed" in data

    @patch('src.api.routes.pipeline.Path.exists')
    def test_run_preprocessing_no_files(self, mock_exists, api_client):
        """Testa erro quando não há arquivos de dados."""
        mock_exists.return_value = False
        
        response = api_client.post("/pipeline/preprocess")
        
        assert response.status_code == 404
        assert "Arquivo bruto não encontrado" in response.json()["detail"]

    @patch('src.api.routes.pipeline.check_preprocessing_completed')
    def test_run_tuning_preprocessing_not_completed(self, mock_check_completed, api_client):
        """Testa erro quando preprocessing não foi concluído."""
        mock_check_completed.return_value = False
        
        response = api_client.post("/pipeline/tune", json={
            "pop_size": 4,
            "max_generations": 2,
            "patience": 3,
            "k_folds": 3,
            "aggressiveness": "medium",
            "elitism": True,
            "crossover_probability": 0.7,
            "mutation_probability": 0.3,
            "individual_mutation_probability": 0.5,
            "random_seed": 42,
            "sample_size": 50000
        })
        
        assert response.status_code == 400
        assert "Preprocessing deve ser concluído" in response.json()["detail"]

    @patch('src.api.routes.pipeline.check_tuning_completed')
    def test_run_predictions_tuning_not_completed(self, mock_check_completed, api_client):
        """Testa erro quando tuning não foi concluído."""
        mock_check_completed.return_value = False
        
        response = api_client.post("/pipeline/predict")
        
        assert response.status_code == 400
        assert "Tuning deve ser concluído" in response.json()["detail"]

    def test_get_pipeline_job_not_found(self, api_client):
        """Testa busca de job de pipeline inexistente."""
        response = api_client.get("/pipeline/jobs/nonexistent")
        
        assert response.status_code == 404

    def test_get_pipeline_job_success(self, api_client):
        """Testa busca de job de pipeline existente."""
        job_id = create_job()
        
        response = api_client.get(f"/pipeline/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id

    def test_get_pipeline_job_logs_not_found(self, api_client):
        """Testa busca de logs de job inexistente."""
        response = api_client.get("/pipeline/jobs/nonexistent/logs")
        
        assert response.status_code == 404


class TestTuningRoutes:
    @patch('src.api.routes.tuning.tuning_service.list_datasets')
    def test_list_datasets(self, mock_list_datasets, api_client):
        """Testa listagem de datasets disponíveis."""
        mock_list_datasets.return_value = ["dataset1.csv", "dataset2.csv"]
        
        response = api_client.get("/tuning/datasets")
        
        assert response.status_code == 200
        assert response.json()["datasets"] == ["dataset1.csv", "dataset2.csv"]

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    @patch('src.api.routes.tuning.create_job')
    def test_run_tuning_async(self, mock_create_job, mock_run_tuning, api_client):
        """Testa execução de tuning em modo assíncrono."""
        mock_create_job.return_value = "job_123"
        mock_run_tuning.return_value = {"result": "tuning result"}
        
        response = api_client.post(
            "/tuning/run",
            json={
                "dataset": "test.csv",
                "target_col": "TARGET",
                "pop_size": 4,
                "max_generations": 2,
                "async_mode": True
            }
        )
        
        assert response.status_code == 200
        assert response.json()["job_id"] == "job_123"

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    def test_run_tuning_sync(self, mock_run_tuning, api_client):
        """Testa execução de tuning em modo síncrono."""
        mock_run_tuning.return_value = {
            "best_model": "model.joblib",
            "best_fitness": 0.95
        }
        
        response = api_client.post(
            "/tuning/run",
            json={
                "dataset": "test.csv",
                "target_col": "TARGET",
                "pop_size": 4,
                "max_generations": 2,
                "async_mode": False
            }
        )
        
        assert response.status_code == 200
        assert response.json()["best_model"] == "model.joblib"

    @patch('src.api.routes.tuning.tuning_service.run_tuning')
    def test_run_tuning_file_not_found(self, mock_run_tuning, api_client):
        """Testa erro quando dataset não existe."""
        mock_run_tuning.side_effect = FileNotFoundError("Dataset não encontrado")
        
        response = api_client.post(
            "/tuning/run",
            json={
                "dataset": "nonexistent.csv",
                "target_col": "TARGET",
                "pop_size": 4,
                "max_generations": 2,
                "async_mode": False
            }
        )
        
        assert response.status_code == 404

    def test_get_tuning_job_not_found(self, api_client):
        """Testa busca de job de tuning inexistente."""
        response = api_client.get("/tuning/jobs/nonexistent")
        
        assert response.status_code == 404

    def test_get_tuning_job_success(self, api_client):
        """Testa busca de job de tuning existente."""
        job_id = create_job()
        
        response = api_client.get(f"/tuning/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id

    @patch('src.api.routes.tuning.tuning_service.get_latest_logs')
    def test_get_latest_logs(self, mock_get_logs, api_client):
        """Testa busca de logs mais recentes."""
        mock_get_logs.return_value = {"logs": "log content"}
        
        response = api_client.get("/tuning/logs/latest")
        
        assert response.status_code == 200
        assert response.json()["logs"] == "log content"

    @patch('src.api.routes.tuning.tuning_service.get_latest_logs')
    def test_get_latest_logs_not_found(self, mock_get_logs, api_client):
        """Testa erro quando logs não existem."""
        mock_get_logs.side_effect = FileNotFoundError("Logs não encontrados")
        
        response = api_client.get("/tuning/logs/latest")
        
        assert response.status_code == 404

    @patch('src.api.routes.tuning.tuning_service.get_generation_snapshots')
    def test_get_generation_snapshots(self, mock_get_snapshots, api_client):
        """Testa busca de snapshots de gerações."""
        mock_get_snapshots.return_value = [
            {"generation": 1, "best_fitness": 0.8},
            {"generation": 2, "best_fitness": 0.9}
        ]
        
        response = api_client.get("/tuning/jobs/job_123/generations?since=0")
        
        assert response.status_code == 200
        assert len(response.json()["snapshots"]) == 2
        assert response.json()["last_generation"] == 2

    def test_get_generation_snapshots_job_not_found(self, api_client):
        """Testa erro quando job não existe."""
        response = api_client.get("/tuning/jobs/nonexistent/generations")
        
        assert response.status_code == 404
