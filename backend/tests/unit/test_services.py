"""Testes unitários para os serviços (model_comparison_service, tuning_service)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from src.services.model_comparison_service import (
    ensure_directories,
    load_predictions,
    calculate_metrics,
    generate_confusion_matrix_plot,
    generate_metrics_comparison_plot,
    generate_class_distribution_plot,
    compare_models,
)
from src.services.tuning_service import (
    list_datasets,
    resolve_dataset_path,
    load_training_data,
    stratified_sample,
    serialize_ga_results,
    train_original_models,
    run_tuning,
    get_generation_snapshots,
    get_latest_logs,
)


class TestModelComparisonService:
    @patch('src.services.model_comparison_service.REPORTS_DIR')
    @patch('src.services.model_comparison_service.PLOTS_DIR')
    def test_ensure_directories(self, mock_plots_dir, mock_reports_dir):
        """Testa criação de diretórios."""
        mock_reports_dir.mkdir = MagicMock()
        mock_plots_dir.mkdir = MagicMock()
        
        ensure_directories()
        
        mock_reports_dir.mkdir.assert_called_once_with(exist_ok=True)
        mock_plots_dir.mkdir.assert_called_once_with(exist_ok=True)

    @patch('src.services.model_comparison_service.ARTIFACTS_DIR')
    def test_load_predictions_success(self, mock_artifacts_dir):
        """Testa carregamento de predições com sucesso."""
        mock_artifacts_dir.__truediv__ = MagicMock(return_value=MagicMock(exists=lambda: True))
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"ESTADO_NUTRI": [1, 2], "Prediction": [1, 2]})
            
            predictions = load_predictions()
            
            assert len(predictions) > 0

    @patch('src.services.model_comparison_service.ARTIFACTS_DIR')
    def test_load_predictions_file_not_found(self, mock_artifacts_dir):
        """Testa erro quando arquivo de predição não existe."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_artifacts_dir.__truediv__.return_value = mock_path
        
        with pytest.raises(FileNotFoundError, match="Prediction file not found"):
            load_predictions()

    def test_calculate_metrics(self):
        """Testa cálculo de métricas."""
        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        y_pred = pd.Series([0, 1, 1, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_calculate_metrics_perfect(self):
        """Testa cálculo de métricas com predição perfeita."""
        y_true = pd.Series([0, 1, 2])
        y_pred = pd.Series([0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    @patch('src.services.model_comparison_service.plt')
    @patch('src.services.model_comparison_service.PLOTS_DIR')
    def test_generate_confusion_matrix_plot(self, mock_plots_dir, mock_plt):
        """Testa geração de plot de matriz de confusão."""
        mock_plots_dir.__truediv__.return_value = "test_path.png"
        y_true = pd.Series([0, 1, 2])
        y_pred = pd.Series([0, 1, 1])
        
        plot_path = generate_confusion_matrix_plot(y_true, y_pred, "test_model")
        
        assert plot_path == "/reports/plots/confusion_matrix_test_model.png"

    @patch('src.services.model_comparison_service.plt')
    @patch('src.services.model_comparison_service.PLOTS_DIR')
    def test_generate_metrics_comparison_plot(self, mock_plots_dir, mock_plt):
        """Testa geração de plot de comparação de métricas."""
        mock_plots_dir.__truediv__.return_value = "test_path.png"
        metrics_dict = {
            "model1": {"accuracy": 0.9, "precision_macro": 0.85, "recall_macro": 0.88, "f1_macro": 0.86},
            "model2": {"accuracy": 0.85, "precision_macro": 0.82, "recall_macro": 0.84, "f1_macro": 0.83}
        }
        
        plot_path = generate_metrics_comparison_plot(metrics_dict)
        
        assert plot_path == "/reports/plots/metrics_comparison.png"

    @patch('src.services.model_comparison_service.plt')
    @patch('src.services.model_comparison_service.PLOTS_DIR')
    def test_generate_class_distribution_plot(self, mock_plots_dir, mock_plt):
        """Testa geração de plot de distribuição de classes."""
        mock_plots_dir.__truediv__.return_value = "test_path.png"
        y_true = pd.Series([0, 1, 2])
        predictions_dict = {
            "model1": pd.Series([0, 1, 1]),
            "model2": pd.Series([0, 1, 2])
        }
        
        plot_path = generate_class_distribution_plot(y_true, predictions_dict)
        
        assert plot_path == "/reports/plots/class_distribution.png"


class TestTuningService:
    @patch('src.services.tuning_service.PROCESSED_DIR')
    def test_list_datasets_empty(self, mock_processed_dir):
        """Testa listagem quando não há datasets."""
        mock_processed_dir.exists.return_value = False
        
        datasets = list_datasets()
        
        assert datasets == []

    @patch('src.services.tuning_service.PROCESSED_DIR')
    def test_list_datasets_with_files(self, mock_processed_dir):
        """Testa listagem com arquivos CSV."""
        mock_file1 = MagicMock()
        mock_file1.name = "dataset1.csv"
        mock_file2 = MagicMock()
        mock_file2.name = "dataset2.csv"
        mock_processed_dir.exists.return_value = True
        mock_processed_dir.glob.return_value = [mock_file1, mock_file2]
        
        datasets = list_datasets()
        
        assert len(datasets) == 2
        assert "dataset1.csv" in datasets
        assert "dataset2.csv" in datasets

    def test_resolve_dataset_path_absolute(self):
        """Testa resolução de caminho absoluto."""
        path = resolve_dataset_path("/absolute/path/to/dataset.csv")
        
        assert path == Path("/absolute/path/to/dataset.csv")

    def test_resolve_dataset_path_relative(self):
        """Testa resolução de caminho relativo."""
        path = resolve_dataset_path("dataset.csv")
        
        assert path.name == "dataset.csv"

    @patch('src.services.tuning_service.resolve_dataset_path')
    def test_load_training_data_success(self, mock_resolve_path):
        """Testa carregamento de dados com sucesso."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_resolve_path.return_value = mock_path
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                "TARGET": [0, 1, 2],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0]
            })
            
            X, y = load_training_data("test.csv", "TARGET")
            
            assert X.shape[1] == 2
            assert len(y) == 3

    @patch('src.services.tuning_service.resolve_dataset_path')
    def test_load_training_data_file_not_found(self, mock_resolve_path):
        """Testa erro quando arquivo não existe."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_resolve_path.return_value = mock_path
        
        with pytest.raises(FileNotFoundError, match="Dataset não encontrado"):
            load_training_data("nonexistent.csv", "TARGET")

    @patch('src.services.tuning_service.resolve_dataset_path')
    def test_load_training_data_invalid_target(self, mock_resolve_path):
        """Testa erro quando coluna target não existe."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_resolve_path.return_value = mock_path
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                "feature1": [1.0, 2.0],
                "feature2": [4.0, 5.0]
            })
            
            with pytest.raises(ValueError, match="Coluna `TARGET` não encontrada"):
                load_training_data("test.csv", "TARGET")

    def test_stratified_sample_no_sampling_needed(self):
        """Testa amostragem quando não é necessária (sample_size >= total)."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        X_s, y_s = stratified_sample(X, y, n_samples=10)
        
        assert X_s.shape == X.shape
        assert y_s.shape == y.shape

    def test_stratified_sample_with_sampling(self):
        """Testa amostragem estratificada."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        X_s, y_s = stratified_sample(X, y, n_samples=2, random_seed=42)
        
        assert len(X_s) == 2
        assert len(y_s) == 2

    def test_serialize_ga_results(self):
        """Testa serialização de resultados do GA."""
        results = {
            "generations_stats": [{"generation": 1, "best_fitness": 0.8}],
            "best_individual": MagicMock(),
            "stopped_at": 5,
            "reason": "convergence",
            "params": {"pop_size": 10}
        }
        
        with patch('src.services.tuning_service._individuo_to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"params": {"n_estimators": 100}}
            
            serialized = serialize_ga_results(results)
            
            assert "generations_stats" in serialized
            assert "best_individual" in serialized
            assert serialized["stopped_at"] == 5
            assert serialized["reason"] == "convergence"

    @patch('src.services.tuning_service.joblib')
    @patch('src.services.tuning_service.PIPELINE')
    @patch('src.services.tuning_service.ORIGINALS_PATH')
    def test_train_original_models(self, mock_originals_dir, mock_pipeline, mock_joblib):
        """Testa treinamento de modelos originais."""
        mock_originals_dir.mkdir = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.fit = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        train_original_models(X, y, mock_originals_dir)
        
        mock_originals_dir.mkdir.assert_called_once()
        assert mock_pipeline_instance.fit.call_count == 2

    @patch('src.services.tuning_service.load_training_data')
    @patch('src.services.tuning_service.stratified_sample')
    @patch('src.services.tuning_service.train_original_models')
    @patch('src.services.tuning_service.GeneticAlgorithm')
    @patch('src.services.tuning_service.save_ga_results')
    @patch('src.services.tuning_service.save_best_model')
    def test_run_tuning(
        self, mock_save_best, mock_save_logs, mock_ga, mock_train_orig,
        mock_sample, mock_load_data
    ):
        """Testa execução do tuning."""
        mock_load_data.return_value = (np.array([[1, 2]]), np.array([0]))
        mock_sample.return_value = (np.array([[1, 2]]), np.array([0]))
        
        mock_ga_instance = MagicMock()
        mock_ga_instance.run.return_value = {
            "best_individual": MagicMock(),
            "generations_stats": [],
            "stopped_at": 5,
            "reason": "convergence",
            "params": {}
        }
        mock_ga.return_value = mock_ga_instance
        
        with patch('src.services.tuning_service.serialize_ga_results') as mock_serialize:
            mock_serialize.return_value = {"result": "ok"}
            
            result = run_tuning(
                dataset="test.csv",
                target_col="TARGET",
                pop_size=4,
                max_generations=2,
                sample_size=100
            )
            
            assert result["result"] == "ok"

    @patch('src.services.tuning_service.read_generation_snapshots')
    def test_get_generation_snapshots(self, mock_read_snapshots):
        """Testa busca de snapshots de gerações."""
        mock_read_snapshots.return_value = [
            {"generation": 1, "best_fitness": 0.8},
            {"generation": 2, "best_fitness": 0.9}
        ]
        
        snapshots = get_generation_snapshots("job_123", since_generation=0)
        
        assert len(snapshots) == 2
        mock_read_snapshots.assert_called_once()

    @patch('src.services.tuning_service.load_ga_history')
    @patch('src.services.tuning_service.LOG_PATH')
    def test_get_latest_logs(self, mock_log_path, mock_load_history):
        """Testa busca de logs mais recentes."""
        mock_json_path = MagicMock()
        mock_json_path.exists.return_value = True
        mock_csv_path = MagicMock()
        mock_csv_path.exists.return_value = True
        mock_csv_path.read_text.return_value = "csv content"
        mock_log_path.__truediv__.side_effect = [mock_json_path, mock_csv_path]
        
        mock_load_history.return_value = {"history": "test"}
        
        logs = get_latest_logs()
        
        assert logs["history"] == {"history": "test"}
        assert logs["stats_csv"] == "csv content"

    @patch('src.services.tuning_service.LOG_PATH')
    def test_get_latest_logs_not_found(self, mock_log_path):
        """Testa erro quando logs não existem."""
        mock_json_path = MagicMock()
        mock_json_path.exists.return_value = False
        mock_log_path.__truediv__.return_value = mock_json_path
        
        with pytest.raises(FileNotFoundError, match="Nenhum histórico GA encontrado"):
            get_latest_logs()
