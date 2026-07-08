"""Testes unitários para o cliente API do frontend."""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from src.api_client import (
    ApiError,
    TuningClient,
    PipelineClient,
    LLMClient,
    ModelComparisonClient,
)


class TestApiError:
    def test_api_error_init(self):
        """Testa inicialização de ApiError."""
        error = ApiError("Test error", status_code=404)
        
        assert str(error) == "Test error"
        assert error.status_code == 404

    def test_api_error_without_status(self):
        """Testa ApiError sem status code."""
        error = ApiError("Test error")
        
        assert error.status_code is None


class TestTuningClient:
    def test_tuning_client_init_default(self):
        """Testa inicialização com URL padrão."""
        client = TuningClient()
        
        assert client.base_url == "http://localhost:8000"

    @patch('src.api_client.os.getenv')
    def test_tuning_client_init_custom_url(self, mock_getenv):
        """Testa inicialização com URL customizada."""
        mock_getenv.return_value = "http://custom:8000"
        client = TuningClient()
        
        assert client.base_url == "http://custom:8000"

    def test_tuning_client_init_explicit_url(self):
        """Testa inicialização com URL explícita."""
        client = TuningClient(base_url="http://explicit:9000")
        
        assert client.base_url == "http://explicit:9000"

    @patch('src.api_client.httpx.Client')
    def test_health_check_success(self, mock_client_class):
        """Testa health check com sucesso."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"status": "ok"}
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.health_check()
        
        assert result == {"status": "ok"}
        mock_client_instance.get.assert_called_once_with("/health")

    @patch('src.api_client.httpx.Client')
    def test_health_check_failure(self, mock_client_class):
        """Testa health check com falha."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        
        with pytest.raises(ApiError):
            client.health_check()

    @patch('src.api_client.httpx.Client')
    def test_list_datasets(self, mock_client_class):
        """Testa listagem de datasets."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"datasets": ["dataset1.csv", "dataset2.csv"]}
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.list_datasets()
        
        assert result == ["dataset1.csv", "dataset2.csv"]

    @patch('src.api_client.httpx.Client')
    def test_run_tuning_sync(self, mock_client_class):
        """Testa execução de tuning em modo síncrono."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"best_model": "model.joblib"}
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.run_tuning(async_mode=False, dataset="test.csv")
        
        assert result["best_model"] == "model.joblib"

    @patch('src.api_client.httpx.Client')
    @patch('src.api_client.time.sleep')
    def test_run_tuning_async_completed(self, mock_sleep, mock_client_class):
        """Testa execução de tuning em modo assíncrono com sucesso."""
        # Primeira chamada POST retorna job_id
        post_response = MagicMock()
        post_response.is_success = True
        post_response.json.return_value = {"job_id": "job_123"}
        
        # Primeira chamada GET retorna running
        get_response_running = MagicMock()
        get_response_running.is_success = True
        get_response_running.json.return_value = {"status": "running"}
        
        # Segunda chamada GET retorna completed
        get_response_completed = MagicMock()
        get_response_completed.is_success = True
        get_response_completed.json.return_value = {
            "status": "completed",
            "result": {"best_model": "model.joblib"}
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = post_response
        mock_client_instance.get.side_effect = [get_response_running, get_response_completed]
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.run_tuning(async_mode=True, dataset="test.csv")
        
        assert result["best_model"] == "model.joblib"

    @patch('src.api_client.httpx.Client')
    def test_start_tuning_async(self, mock_client_class):
        """Testa início de tuning assíncrono sem bloqueio."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"job_id": "job_123"}
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        job_id = client.start_tuning_async(dataset="test.csv")
        
        assert job_id == "job_123"

    @patch('src.api_client.httpx.Client')
    def test_get_generation_snapshots(self, mock_client_class):
        """Testa busca de snapshots de gerações."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "snapshots": [{"generation": 1, "best_fitness": 0.8}],
            "last_generation": 1,
            "job_status": "running"
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.get_generation_snapshots("job_123", since=0)
        
        assert len(result["snapshots"]) == 1
        assert result["last_generation"] == 1

    @patch('src.api_client.httpx.Client')
    def test_get_latest_logs(self, mock_client_class):
        """Testa busca de logs mais recentes."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"history": {}, "stats_csv": "csv content"}
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = TuningClient()
        result = client.get_latest_logs()
        
        assert result["stats_csv"] == "csv content"


class TestPipelineClient:
    def test_pipeline_client_init_default(self):
        """Testa inicialização com URL padrão."""
        client = PipelineClient()
        
        assert client.base_url == "http://localhost:8000"

    @patch('src.api_client.httpx.Client')
    @patch('src.api_client.time.sleep')
    def test_run_preprocessing(self, mock_sleep, mock_client_class):
        """Testa execução de preprocessing."""
        post_response = MagicMock()
        post_response.is_success = True
        post_response.json.return_value = {"job_id": "job_123"}
        
        get_response_completed = MagicMock()
        get_response_completed.is_success = True
        get_response_completed.json.return_value = {
            "status": "completed",
            "result": {"processed_csv_path": "test.csv"}
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = post_response
        mock_client_instance.get.return_value = get_response_completed
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = PipelineClient()
        result = client.run_preprocessing()
        
        assert result["status"] == "completed"

    @patch('src.api_client.httpx.Client')
    def test_start_preprocessing_async(self, mock_client_class):
        """Testa início de preprocessing assíncrono."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"job_id": "job_123"}
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = PipelineClient()
        job_id = client.start_preprocessing_async()
        
        assert job_id == "job_123"

    @patch('src.api_client.httpx.Client')
    def test_get_job_logs(self, mock_client_class):
        """Testa busca de logs de job."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"job_id": "job_123", "logs": "log content"}
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = PipelineClient()
        result = client.get_job_logs("job_123")
        
        assert result["logs"] == "log content"

    @patch('src.api_client.httpx.Client')
    def test_get_pipeline_status(self, mock_client_class):
        """Testa busca de status do pipeline."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "preprocessing_completed": True,
            "tuning_completed": False,
            "predictions_completed": False
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = PipelineClient()
        result = client.get_pipeline_status()
        
        assert result["preprocessing_completed"] is True

    @patch('src.api_client.httpx.Client')
    def test_run_predictions(self, mock_client_class):
        """Testa execução de predições."""
        post_response = MagicMock()
        post_response.is_success = True
        post_response.json.return_value = {"job_id": "job_123"}
        
        get_response_completed = MagicMock()
        get_response_completed.is_success = True
        get_response_completed.json.return_value = {
            "status": "completed",
            "result": {"predictions_path": "pred.csv"}
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = post_response
        mock_client_instance.get.return_value = get_response_completed
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = PipelineClient()
        result = client.run_predictions()
        
        assert result["status"] == "completed"


class TestLLMClient:
    def test_llm_client_init_default(self):
        """Testa inicialização com URL padrão."""
        client = LLMClient()
        
        assert client.base_url == "http://localhost:8000"

    @patch('src.api_client.httpx.Client')
    def test_create_session(self, mock_client_class):
        """Testa criação de sessão LLM."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "session_id": "session_123",
            "initial_report": "Report",
            "row_count": 100
        }
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = LLMClient()
        result = client.create_session(b"csv data", "test.csv")
        
        assert result["session_id"] == "session_123"

    @patch('src.api_client.httpx.Client')
    def test_chat(self, mock_client_class):
        """Testa envio de pergunta ao chat."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"answer": "Resposta do agente"}
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = LLMClient()
        result = client.chat("session_123", "Qual a média de idade?")
        
        assert result == "Resposta do agente"

    @patch('src.api_client.httpx.Client')
    def test_close_session(self, mock_client_class):
        """Testa fechamento de sessão."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"status": "closed", "session_id": "session_123"}
        mock_client_instance = MagicMock()
        mock_client_instance.delete.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = LLMClient()
        client.close_session("session_123")
        
        mock_client_instance.delete.assert_called_once_with("/llm/session/session_123")

    @patch('src.api_client.httpx.Client')
    def test_create_session_from_files(self, mock_client_class):
        """Testa criação de sessão a partir de arquivos."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "session_id": "session_123",
            "initial_report": "Report",
            "row_count": 100
        }
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = LLMClient()
        result = client.create_session_from_files()
        
        assert result["session_id"] == "session_123"


class TestModelComparisonClient:
    def test_model_comparison_client_init_default(self):
        """Testa inicialização com URL padrão."""
        client = ModelComparisonClient()
        
        assert client.base_url == "http://localhost:8000"

    @patch('src.api_client.httpx.Client')
    def test_run_comparison(self, mock_client_class):
        """Testa execução de comparação de modelos."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"job_id": "job_123", "status": "pending"}
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = ModelComparisonClient()
        result = client.run_comparison()
        
        assert result["job_id"] == "job_123"

    @patch('src.api_client.httpx.Client')
    def test_get_job_status(self, mock_client_class):
        """Testa busca de status de job de comparação."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "job_id": "job_123",
            "status": "completed",
            "result": {"metrics": {}}
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = ModelComparisonClient()
        result = client.get_job_status("job_123")
        
        assert result["status"] == "completed"

    @patch('src.api_client.httpx.Client')
    def test_get_report(self, mock_client_class):
        """Testa busca de relatório de comparação."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "metrics": {"model1": {"accuracy": 0.9}},
            "plots": {}
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance
        
        client = ModelComparisonClient()
        result = client.get_report()
        
        assert "metrics" in result
