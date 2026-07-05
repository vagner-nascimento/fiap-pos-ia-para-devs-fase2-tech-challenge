"""
Testes unitários da função de avaliação de fitness (ga_evaluator.py).

Usa mock data simples (50 samples, 4 features, 3 classes) para validar
que evaluate() retorna tupla válida e que fitness_score() calcula corretamente.

Reforços adicionados (plano-implementacao-testes-ga.md):
    - test_evaluate_returns_zero_on_failure_real_exception: usa unittest.mock
      para forçar exceção real no build_model(), garantindo que o bloco except
      em evaluate() seja realmente exercitado (não apenas o error_score do sklearn).
    - test_evaluate_uses_k_folds_correctly: verifica que k_folds é propagado
      corretamente para cross_val_score via mock.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
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
        """Indivíduo com max_depth=0 (inválido) deve retornar sem exception."""
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

    def test_evaluate_returns_zero_on_failure_real_exception(self, mock_data):
        """Garante que o fallback (0.0, 0.0) é acionado quando build_model() falha.

        O sklearn captura erros internamente via error_score=0.0 em cross_val_score,
        então usar max_depth=0 não aciona o bloco except de evaluate().
        Aqui usamos mock para forçar uma exceção real no pipeline.fit(),
        garantindo que o path de exceção em evaluate() seja realmente exercitado.
        """
        X, y = mock_data
        ind = create_random_rf()
        bad_pipeline = MagicMock()
        bad_pipeline.fit.side_effect = RuntimeError("modelo quebrado intencionalmente")
        with patch.object(ind, "build_model", return_value=bad_pipeline):
            result = evaluate(ind, X, y, k_folds=3)
        assert result == (0.0, 0.0), f"Esperado (0.0, 0.0), obteve {result}"
        assert ind.fitness_values == (0.0, 0.0), "fitness_values não foi setado para (0.0, 0.0)"

    def test_evaluate_uses_k_folds_correctly(self, mock_data):
        """Verifica que evaluate() propaga k_folds corretamente para cross_val_score.

        Garante que alterar k_folds no evaluate() afeta de fato o número de folds
        usado internamente, e não é ignorado silenciosamente.
        """
        X, y = mock_data
        ind = create_random_rf()
        mock_scores = np.array([0.8, 0.75, 0.82, 0.79, 0.81])
        with patch("src.models.ga_evaluator.cross_val_score", return_value=mock_scores) as mock_cv:
            evaluate(ind, X, y, k_folds=5)
        # Verifica que cross_val_score foi chamado com cv=5
        calls = mock_cv.call_args_list
        assert len(calls) >= 1, "cross_val_score não foi chamado"
        # Pelo menos uma das chamadas deve ter cv=5
        cv_values = [
            call.kwargs.get("cv") or (call.args[3] if len(call.args) > 3 else None)
            for call in calls
        ]
        assert 5 in cv_values, f"k_folds=5 não foi passado para cross_val_score; cv recebidos: {cv_values}"


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
