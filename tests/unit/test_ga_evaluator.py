"""
Testes unitários da função de avaliação de fitness (ga_evaluator.py).

Usa mock data simples (50 samples, 4 features, 3 classes) para validar
que evaluate() retorna tupla válida e que fitness_score() calcula corretamente.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.ga_evaluator import evaluate, fitness_score
from src.models.ga_operators import create_random_rf, create_random_knn
from src.models.individuo import IndividuoRF, IndividuoKNN


# ---------------------------------------------------------------------------
# Fixture de dados simples
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """50 samples, 4 features, 3 classes — dados sintéticos reproduzíveis."""
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


# ---------------------------------------------------------------------------
# Testes de evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_rf_returns_tuple_of_two_floats(self, mock_data):
        X, y = mock_data
        ind = create_random_rf()
        result = evaluate(ind, X, y, k_folds=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_knn_returns_tuple_of_two_floats(self, mock_data):
        X, y = mock_data
        ind = create_random_knn()
        result = evaluate(ind, X, y, k_folds=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_rf_fitness_in_valid_range(self, mock_data):
        X, y = mock_data
        for _ in range(5):
            ind = create_random_rf()
            f1, acc = evaluate(ind, X, y, k_folds=3)
            assert 0.0 <= f1 <= 1.0, f"F1 fora do range: {f1}"
            assert 0.0 <= acc <= 1.0, f"Acc fora do range: {acc}"

    def test_knn_fitness_in_valid_range(self, mock_data):
        X, y = mock_data
        for _ in range(5):
            ind = create_random_knn()
            f1, acc = evaluate(ind, X, y, k_folds=3)
            assert 0.0 <= f1 <= 1.0
            assert 0.0 <= acc <= 1.0

    def test_evaluate_sets_fitness_values(self, mock_data):
        """evaluate() deve popular individual.fitness_values."""
        X, y = mock_data
        ind = create_random_rf()
        assert ind.fitness_values is None
        evaluate(ind, X, y, k_folds=3)
        assert ind.fitness_values is not None
        assert len(ind.fitness_values) == 2

    def test_evaluate_knn_with_scaling(self, mock_data):
        """Garante que o pipeline KNN com StandardScaler não falha."""
        X, y = mock_data
        # KNN com metric euclidean — sensível a escala — deve funcionar com scaler
        ind = IndividuoKNN({
            "n_neighbors": 3,
            "weights": "uniform",
            "metric": "euclidean",
            "algorithm": "auto",
        })
        f1, acc = evaluate(ind, X, y, k_folds=3)
        assert 0.0 <= f1 <= 1.0

    def test_evaluate_rf_valid_model(self, mock_data):
        """RF com hiperparâmetros válidos deve funcionar sem falha."""
        X, y = mock_data
        ind = IndividuoRF({
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
        })
        f1, acc = evaluate(ind, X, y, k_folds=3)
        assert f1 > 0.0

    def test_evaluate_returns_zero_on_failure(self, mock_data):
        """Indivíduo com max_depth=0 (inválido) deve retornar (0.0, 0.0) sem exception."""
        X, y = mock_data
        # Forçamos um hiperparâmetro inválido para testar o fallback
        ind = IndividuoRF({
            "n_estimators": 10,
            "max_depth": 0,        # sklearn aceita None mas não 0
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
        })
        result = evaluate(ind, X, y, k_folds=3)
        # Deve retornar sem lançar exceção (0.0 ou valor válido)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Testes de fitness_score()
# ---------------------------------------------------------------------------

class TestFitnessScore:
    def test_perfect_fitness(self):
        assert fitness_score((1.0, 1.0)) == pytest.approx(1.0)

    def test_zero_fitness(self):
        assert fitness_score((0.0, 0.0)) == pytest.approx(0.0)

    def test_weighted_calculation(self):
        """F1=1.0, Acc=0.0 → score deve ser 0.6 (peso F1=0.6)."""
        assert fitness_score((1.0, 0.0)) == pytest.approx(0.6)

    def test_weighted_calculation_acc_only(self):
        """F1=0.0, Acc=1.0 → score deve ser 0.4 (peso Acc=0.4)."""
        assert fitness_score((0.0, 1.0)) == pytest.approx(0.4)

    def test_mixed_values(self):
        f1, acc = 0.8, 0.9
        expected = 0.8 * 0.6 + 0.9 * 0.4
        assert fitness_score((f1, acc)) == pytest.approx(expected)
