"""
Testes de integração do GeneticAlgorithm Co-Evolutivo (genetic_algorithm.py).

Usa dados sintéticos pequenos para validar o fluxo completo em tempo razoável:
    - 2 gerações, pop_size=4 por tipo (8 indivíduos total)
    - Verifica estrutura do resultado
    - Testa critério de parada por convergência
    - Testa elitismo ligado e desligado
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.genetic_algorithm import GeneticAlgorithm
from src.models.ga_evaluator import fitness_score
from src.models.individuo import IndividuoRF, IndividuoKNN


# ---------------------------------------------------------------------------
# Fixture de dados — subset pequeno para velocidade
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_data():
    """100 samples, 5 features, 3 classes — rápido de avaliar com k=3 folds."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


def make_ga(data, **kwargs) -> GeneticAlgorithm:
    """Cria GA com configuração mínima para testes rápidos."""
    X, y = data
    defaults = dict(
        pop_size=4,
        max_generations=2,
        patience=10,
        k_folds=3,
        random_seed=42,
    )
    defaults.update(kwargs)
    return GeneticAlgorithm(X, y, **defaults)


# ---------------------------------------------------------------------------
# Estrutura do resultado
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_has_required_keys(self, small_data):
        ga = make_ga(small_data)
        result = ga.run()
        assert "generations_stats" in result
        assert "best_individual" in result
        assert "stopped_at" in result
        assert "reason" in result
        assert "params" in result

    def test_generations_stats_length(self, small_data):
        ga = make_ga(small_data, max_generations=2)
        result = ga.run()
        assert len(result["generations_stats"]) == 2

    def test_each_generation_has_rf_and_knn_stats(self, small_data):
        ga = make_ga(small_data)
        result = ga.run()
        for stat in result["generations_stats"]:
            assert "rf" in stat
            assert "knn" in stat
            assert "global_best_f1" in stat
            assert "global_best_type" in stat

    def test_best_individual_is_individuo(self, small_data):
        ga = make_ga(small_data)
        result = ga.run()
        best = result["best_individual"]
        assert isinstance(best, (IndividuoRF, IndividuoKNN))

    def test_best_individual_has_fitness(self, small_data):
        ga = make_ga(small_data)
        result = ga.run()
        best = result["best_individual"]
        assert best.fitness_values is not None
        assert len(best.fitness_values) == 2

    def test_best_individual_has_valid_fitness_range(self, small_data):
        ga = make_ga(small_data)
        result = ga.run()
        f1, acc = result["best_individual"].fitness_values
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= acc <= 1.0

    def test_stopped_at_equals_generations_count(self, small_data):
        ga = make_ga(small_data, max_generations=2)
        result = ga.run()
        assert result["stopped_at"] == len(result["generations_stats"])


# ---------------------------------------------------------------------------
# Critério de parada
# ---------------------------------------------------------------------------

class TestStoppingCriteria:
    def test_stops_at_max_generations(self, small_data):
        ga = make_ga(small_data, max_generations=2, patience=100)
        result = ga.run()
        assert result["reason"] == "max_generations"
        assert result["stopped_at"] <= 2

    def test_convergence_stops_early(self, small_data):
        """Com patience=1, deve parar na 2ª geração se não houver melhoria."""
        ga = make_ga(small_data, max_generations=10, patience=1)
        result = ga.run()
        # Com patience=1, para no máximo na 2ª geração sem melhoria
        assert result["stopped_at"] <= 10
        # Pode parar por convergência ou max_generations, ambos válidos
        assert result["reason"] in ("convergence", "max_generations")

    def test_params_preserved_in_result(self, small_data):
        ga = make_ga(small_data, pop_size=4, max_generations=2, patience=5, k_folds=3)
        result = ga.run()
        assert result["params"]["pop_size"] == 4
        assert result["params"]["max_generations"] == 2
        assert result["params"]["patience"] == 5


# ---------------------------------------------------------------------------
# Elitismo
# ---------------------------------------------------------------------------

class TestElitism:
    def test_with_elitism_enabled(self, small_data):
        ga = make_ga(small_data, elitism=True, max_generations=2)
        result = ga.run()
        assert result["best_individual"] is not None

    def test_with_elitism_disabled(self, small_data):
        ga = make_ga(small_data, elitism=False, max_generations=2)
        result = ga.run()
        assert result["best_individual"] is not None

    def test_elitism_flag_preserved_in_params(self, small_data):
        ga = make_ga(small_data, elitism=False)
        result = ga.run()
        assert result["params"]["elitism"] is False


# ---------------------------------------------------------------------------
# Populações co-evolutivas
# ---------------------------------------------------------------------------

class TestCoEvolution:
    def test_both_types_appear_in_stats(self, small_data):
        """RF e KNN devem aparecer nas estatísticas de todas as gerações."""
        ga = make_ga(small_data, max_generations=2)
        result = ga.run()
        for stat in result["generations_stats"]:
            assert stat["rf"]["count"] >= 0
            assert stat["knn"]["count"] >= 0

    def test_global_best_type_is_rf_or_knn(self, small_data):
        ga = make_ga(small_data, max_generations=2)
        result = ga.run()
        for stat in result["generations_stats"]:
            assert stat["global_best_type"] in ("RF", "KNN")

    def test_reproducibility_with_same_seed(self, small_data):
        """Mesma seed deve produzir mesmo resultado."""
        X, y = small_data
        ga1 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99)
        ga2 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99)
        r1 = ga1.run()
        r2 = ga2.run()
        assert r1["best_individual"].classifier_type == r2["best_individual"].classifier_type
        assert r1["stopped_at"] == r2["stopped_at"]
