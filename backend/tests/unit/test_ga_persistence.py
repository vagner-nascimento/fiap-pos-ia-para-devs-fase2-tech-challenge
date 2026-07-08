"""Testes unitários para o módulo de persistência do GA."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.models.ga_persistence import (
    _individuo_to_dict,
    save_ga_results,
    save_best_model,
    load_ga_history,
)
from src.models.individuo import IndividuoRF, IndividuoKNN


class TestIndividuoToDict:
    def test_individuo_to_dict_none(self):
        """Testa conversão de indivíduo None."""
        result = _individuo_to_dict(None)
        
        assert result is None

    def test_individuo_to_dict_rf(self):
        """Testa conversão de indivíduo RF para dict."""
        individual = IndividuoRF({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        })
        individual.fitness_values = (0.85, 0.90)
        
        result = _individuo_to_dict(individual)
        
        assert result["type"] == "RF"
        assert result["hyperparams"]["n_estimators"] == 100
        assert result["fitness_f1"] == 0.85
        assert result["fitness_acc"] == 0.90

    def test_individuo_to_dict_knn(self):
        """Testa conversão de indivíduo KNN para dict."""
        individual = IndividuoKNN({
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "algorithm": "auto"
        })
        individual.fitness_values = (0.80, 0.85)
        
        result = _individuo_to_dict(individual)
        
        assert result["type"] == "KNN"
        assert result["hyperparams"]["n_neighbors"] == 5
        assert result["fitness_f1"] == 0.80
        assert result["fitness_acc"] == 0.85

    def test_individuo_to_dict_no_fitness(self):
        """Testa conversão de indivíduo sem fitness."""
        individual = IndividuoRF({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        })
        
        result = _individuo_to_dict(individual)
        
        assert result["fitness_f1"] is None
        assert result["fitness_acc"] is None


class TestSaveGAResults:
    def test_save_ga_results(self, tmp_path):
        """Testa salvamento de resultados do GA."""
        results = {
            "params": {"pop_size": 10, "max_generations": 5},
            "stopped_at": 5,
            "reason": "convergence",
            "best_individual": IndividuoRF({
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "criterion": "gini"
            }),
            "generations_stats": [
                {
                    "generation": 1,
                    "rf": {"count": 5, "best_f1": 0.8, "avg_f1": 0.75},
                    "knn": {"count": 5, "best_f1": 0.78, "avg_f1": 0.73},
                    "global_best_f1": 0.8,
                    "global_best_score": 0.82,
                    "global_best_type": "RF"
                }
            ]
        }
        
        save_ga_results(results, str(tmp_path))
        
        json_path = tmp_path / "ga_history.json"
        csv_path = tmp_path / "ga_generation_stats.csv"
        
        assert json_path.exists()
        assert csv_path.exists()

    def test_save_ga_results_json_content(self, tmp_path):
        """Testa conteúdo do JSON salvo."""
        results = {
            "params": {"pop_size": 10},
            "stopped_at": 5,
            "reason": "convergence",
            "best_individual": None,
            "generations_stats": []
        }
        
        save_ga_results(results, str(tmp_path))
        
        json_path = tmp_path / "ga_history.json"
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["params"]["pop_size"] == 10
        assert data["stopped_at"] == 5
        assert data["reason"] == "convergence"

    def test_save_ga_results_csv_content(self, tmp_path):
        """Testa conteúdo do CSV salvo."""
        results = {
            "params": {},
            "stopped_at": 1,
            "reason": "max_generations",
            "best_individual": None,
            "generations_stats": [
                {
                    "generation": 1,
                    "rf": {"count": 5, "best_f1": 0.8, "avg_f1": 0.75, "best_acc": 0.82, "best_score": 0.81, "avg_score": 0.78},
                    "knn": {"count": 5, "best_f1": 0.78, "avg_f1": 0.73, "best_acc": 0.80, "best_score": 0.79, "avg_score": 0.76},
                    "global_best_f1": 0.8,
                    "global_best_score": 0.81,
                    "global_best_type": "RF",
                    "stopped_early": False
                }
            ]
        }
        
        save_ga_results(results, str(tmp_path))
        
        csv_path = tmp_path / "ga_generation_stats.csv"
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        assert len(df) == 1
        assert df.loc[0, "generation"] == 1
        assert df.loc[0, "rf_count"] == 5
        assert df.loc[0, "knn_count"] == 5

    def test_save_ga_results_empty_stats(self, tmp_path):
        """Testa salvamento com stats vazias."""
        results = {
            "params": {},
            "stopped_at": 0,
            "reason": "error",
            "best_individual": None,
            "generations_stats": []
        }
        
        save_ga_results(results, str(tmp_path))
        
        json_path = tmp_path / "ga_history.json"
        csv_path = tmp_path / "ga_generation_stats.csv"
        
        assert json_path.exists()
        # CSV não deve ser criado se stats estiver vazio
        assert not csv_path.exists()


class TestSaveBestModel:
    @patch('src.models.ga_persistence.save_model')
    def test_save_best_model_rf(self, mock_save_model):
        """Testa salvamento de melhor modelo RF."""
        individual = IndividuoRF({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        })
        
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        
        save_best_model(individual, X, y, "test_model.joblib")
        
        mock_save_model.assert_called_once()

    @patch('src.models.ga_persistence.save_model')
    def test_save_best_model_knn(self, mock_save_model):
        """Testa salvamento de melhor modelo KNN."""
        individual = IndividuoKNN({
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "algorithm": "auto"
        })
        
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        
        save_best_model(individual, X, y, "test_model.joblib")
        
        mock_save_model.assert_called_once()


class TestLoadGAHistory:
    def test_load_ga_history_success(self, tmp_path):
        """Testa carregamento de histórico com sucesso."""
        test_data = {
            "params": {"pop_size": 10},
            "stopped_at": 5,
            "reason": "convergence",
            "best_individual": None,
            "generations_stats": []
        }
        
        json_path = tmp_path / "ga_history.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        loaded_data = load_ga_history(str(json_path))
        
        assert loaded_data["params"]["pop_size"] == 10
        assert loaded_data["stopped_at"] == 5
        assert loaded_data["reason"] == "convergence"

    def test_load_ga_history_file_not_found(self):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError, match="Histórico GA não encontrado"):
            load_ga_history("nonexistent.json")

    def test_load_ga_history_invalid_json(self, tmp_path):
        """Testa erro quando JSON é inválido."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            load_ga_history(str(json_path))
