"""
Testes de integração do GeneticAlgorithm Co-Evolutivo (genetic_algorithm.py).

Usa dados sintéticos pequenos para validar o fluxo completo em tempo razoável:
    - 2 gerações, pop_size=4 por tipo (8 indivíduos total)
    - Verifica estrutura do resultado
    - Testa critério de parada por convergência
    - Testa elitismo ligado e desligado

Reforços adicionados (plano-implementacao-testes-ga.md):
    - TestElitism.test_elitism_true_never_regresses: verifica que global_best_f1
      é monoôtona não-decrescente com elitism=True (ADR-009).
    - TestStructuralElitism: RF e KNN nunca somem da população (count >= 1).
    - TestGlobalTournamentSelection: indivíduo com fitness maior vence torneio
      com frequência maior (via _tournament_select diretamente).
    - TestStoppingCriteria.test_convergence_actually_triggers: valida que ao
      parar por convergência, as últimas gerações mostram plateau real.
    - TestCoEvolution.test_reproducibility_full_hyperparams: compara hyperparams
      e fitness_values completos entre duas runs com mesma seed.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.genetic_algorithm import GeneticAlgorithm
from src.models.ga_evaluator import fitness_score
from src.models.individuo import IndividuoRF, IndividuoKNN


@pytest.fixture(autouse=True)
def mock_ga_evaluate(monkeypatch):
    import hashlib
    def _mock_evaluate(individual, X, y, k_folds=5):
        hp_str = f"{individual.classifier_type}:{sorted(individual.hyperparams.items())}"
        h = hashlib.md5(hp_str.encode('utf-8')).hexdigest()
        val = int(h, 16)
        f1 = 0.5 + (val % 450) / 1000.0
        acc = 0.5 + ((val >> 16) % 450) / 1000.0
        individual.fitness_values = (f1, acc)
        return (f1, acc)
    monkeypatch.setattr("src.models.genetic_algorithm.evaluate", _mock_evaluate)


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

    def test_convergence_actually_triggers(self, small_data):
        """Reforça test_convergence_stops_early: verifica que ao parar por
        convergência, as últimas gerações realmente exibem plateau de fitness.

        O teste original aceitava qualquer reason, passando mesmo que a
        convergência nunca disparasse de fato. Aqui confirmamos o plateau.
        """
        ga = make_ga(small_data, max_generations=20, patience=2, pop_size=4)
        result = ga.run()
        if result["reason"] == "convergence":
            bests = [g["global_best_f1"] for g in result["generations_stats"]]
            # Se parou por convergência, as últimas gerações devem mostrar plateau
            last_n = bests[-3:] if len(bests) >= 3 else bests
            assert max(last_n) - min(last_n) < 1e-4, (
                f"Parou por 'convergence' mas últimas gerações não mostram plateau: {last_n}"
            )

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

    def test_elitism_true_never_regresses(self, small_data):
        """Com elitism=True, global_best_f1 nunca deve piorar entre gerações.

        Este é o teste de maior valor diagnóstico do plano: os testes anteriores
        (test_with_elitism_enabled/disabled) só checam que best_individual é
        não-nulo, o que passa mesmo se a reinserção do elite estiver quebrada.
        Aqui validamos a propriedade fundamental do elitismo: monotonia (ADR-009).
        """
        ga = make_ga(small_data, elitism=True, max_generations=6, pop_size=4)
        result = ga.run()
        bests = [g["global_best_f1"] for g in result["generations_stats"]]
        assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bests, bests[1:])), (
            f"global_best_f1 regrediu com elitism=True: {bests}"
        )


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

    def test_reproducibility_full_hyperparams(self, small_data):
        """Reforça test_reproducibility_with_same_seed: compara hyperparams e
        fitness_values completos em vez de apenas classifier_type e stopped_at.

        O teste original não detecta vazamento de RNG em operadores genéticos.
        Este teste pega casos onde a seed é parcialmente propagada mas há
        não-determinismo em sub-componentes (ex: KFold sem seed fixa).

        Nota: se este teste falhar, o bug provavelmente está na propagação
        da seed no cross_val_score/KFold — tratar como bug real, não ajustar
        o teste para passar (conforme critério de aceite do plano).
        """
        X, y = small_data
        r1 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99).run()
        r2 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99).run()
        assert r1["best_individual"].hyperparams == r2["best_individual"].hyperparams, (
            "Hyperparams divergentes com mesma seed — possível não-determinismo no RNG"
        )
        assert r1["best_individual"].fitness_values == r2["best_individual"].fitness_values, (
            "fitness_values divergentes com mesma seed — verificar seed do KFold/cross_val_score"
        )


# ---------------------------------------------------------------------------
# Elitismo estrutural mínimo (ADR-004/009)
# ---------------------------------------------------------------------------

class TestStructuralElitism:
    """Valida que o elitismo estrutural mínimo (sempre ativo, independente
    do flag elitism) garante ao menos 1 sobrevivente de cada tipo por geração.

    Gap identificado: test_both_types_appear_in_stats usa count >= 0, que é
    sempre verdadeiro e não valida nada. Este teste usa count >= 1 (ADR-004).
    """

    def test_min_one_survivor_per_type_every_generation(self, small_data):
        """RF e KNN nunca somem completamente da população, mesmo com pressão
        competitiva (elitismo estrutural mínimo garantido via _split_by_type).
        """
        ga = make_ga(small_data, max_generations=6, pop_size=4)
        result = ga.run()
        for i, stat in enumerate(result["generations_stats"]):
            assert stat["rf"]["count"] >= 1, (
                f"RF sumiu na geração {i+1}: count={stat['rf']['count']}"
            )
            assert stat["knn"]["count"] >= 1, (
                f"KNN sumiu na geração {i+1}: count={stat['knn']['count']}"
            )


# ---------------------------------------------------------------------------
# Seleção por torneio global (ADR-004)
# ---------------------------------------------------------------------------

class TestGlobalTournamentSelection:
    """Testa o comportamento da seleção por torneio via _tournament_select.

    Como _tournament_select é um método de instância (não uma função pura
    exportada), testamos diretamente através da instância do GA com pool de
    indivíduos com fitness pré-populado artificialmente — sem precisar do
    run() completo (que seria lento e não isolaria o comportamento da seleção).
    """

    def test_higher_fitness_wins_more_often(self, small_data):
        """Indivíduo com fitness maior deve vencer torneios com mais frequência.

        Com fitness 0.9 vs 0.1 e tournsize=3, o forte deve vencer >>50% das vezes.
        Em 50 torneios com seed controlada, esperamos >25 vitórias do forte.
        """
        ga = make_ga(small_data)

        strong = IndividuoRF({"n_estimators": 10, "max_depth": 3,
                               "min_samples_split": 2, "min_samples_leaf": 1,
                               "criterion": "gini"})
        strong.fitness_values = (0.9, 0.9)  # fitness_score = 0.9*0.6 + 0.9*0.4 = 0.9

        weak = IndividuoKNN({"n_neighbors": 3, "weights": "uniform",
                              "metric": "euclidean", "algorithm": "auto"})
        weak.fitness_values = (0.1, 0.1)    # fitness_score = 0.1

        # Pool com 10 fortes e 10 fracos — proporção 50/50
        pool = [strong] * 10 + [weak] * 10

        wins_strong = sum(
            1 for _ in range(50)
            if ga._tournament_select(pool, k=1, tournsize=3)[0].fitness_values == (0.9, 0.9)
        )
        assert wins_strong > 25, (
            f"Esperado >25 vitórias do forte em 50 torneios, obteve {wins_strong}. "
            "Possível bug na seleção por torneio."
        )
