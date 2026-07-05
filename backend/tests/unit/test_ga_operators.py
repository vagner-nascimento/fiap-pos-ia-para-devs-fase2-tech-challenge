"""
Testes unitários dos operadores genéticos (ga_operators.py).

Cobre:
    - create_random_rf / create_random_knn: ranges válidos
    - crossover_rf / crossover_knn: tipos preservados, params válidos
    - mutate_rf / mutate_knn: tipos preservados, params dentro do range
    - Todos os níveis de aggressiveness

Reforços adicionados (plano-implementacao-testes-ga.md):
    - test_indpb_half_swaps_approximately_half (RF e KNN): valida que indpb=0.5
      produz ~50% de troca de genes em média ao longo de repetições.
    - TestMutateRFAggressiveness / TestMutateKNNAggressiveness: valida que
      'high' produz deltas médios maiores que 'medium', que são maiores que
      'low' (ADR-008 promete ±10%/±30%/±60% do range).
"""

import pytest
import random as _random
import statistics
from src.models.individuo import (
    RF_HYPERPARAM_SPACE,
    KNN_HYPERPARAM_SPACE,
    IndividuoRF,
    IndividuoKNN,
)
from src.models.ga_operators import (
    create_random_rf,
    create_random_knn,
    crossover_rf,
    crossover_knn,
    mutate_rf,
    mutate_knn,
)


# ---------------------------------------------------------------------------
# Helpers de validação
# ---------------------------------------------------------------------------

def assert_rf_params_valid(hp: dict) -> None:
    space = RF_HYPERPARAM_SPACE
    assert isinstance(hp["n_estimators"], int)
    assert space["n_estimators"][0] <= hp["n_estimators"] <= space["n_estimators"][1]

    assert hp["max_depth"] is None or (
        isinstance(hp["max_depth"], int)
        and space["max_depth"][0] <= hp["max_depth"] <= space["max_depth"][1]
    )

    assert isinstance(hp["min_samples_split"], int)
    assert space["min_samples_split"][0] <= hp["min_samples_split"] <= space["min_samples_split"][1]

    assert isinstance(hp["min_samples_leaf"], int)
    assert space["min_samples_leaf"][0] <= hp["min_samples_leaf"] <= space["min_samples_leaf"][1]

    assert hp["criterion"] in space["criterion"]


def assert_knn_params_valid(hp: dict) -> None:
    space = KNN_HYPERPARAM_SPACE
    assert isinstance(hp["n_neighbors"], int)
    assert space["n_neighbors"][0] <= hp["n_neighbors"] <= space["n_neighbors"][1]

    assert hp["weights"] in space["weights"]
    assert hp["metric"] in space["metric"]
    assert hp["algorithm"] in space["algorithm"]


# ---------------------------------------------------------------------------
# Geração aleatória
# ---------------------------------------------------------------------------

class TestCreateRandom:
    def test_create_random_rf_type(self):
        ind = create_random_rf()
        assert isinstance(ind, IndividuoRF)

    def test_create_random_rf_params_valid(self):
        for _ in range(20):
            ind = create_random_rf()
            assert_rf_params_valid(ind.hyperparams)

    def test_create_random_rf_no_fitness(self):
        ind = create_random_rf()
        assert ind.fitness_values is None

    def test_create_random_knn_type(self):
        ind = create_random_knn()
        assert isinstance(ind, IndividuoKNN)

    def test_create_random_knn_params_valid(self):
        for _ in range(20):
            ind = create_random_knn()
            assert_knn_params_valid(ind.hyperparams)

    def test_create_random_knn_no_fitness(self):
        ind = create_random_knn()
        assert ind.fitness_values is None


# ---------------------------------------------------------------------------
# Crossover RF
# ---------------------------------------------------------------------------

class TestCrossoverRF:
    def test_returns_two_rf_individuals(self):
        ind1 = create_random_rf()
        ind2 = create_random_rf()
        c1, c2 = crossover_rf(ind1, ind2)
        assert isinstance(c1, IndividuoRF)
        assert isinstance(c2, IndividuoRF)

    def test_children_have_valid_params(self):
        for _ in range(10):
            ind1, ind2 = create_random_rf(), create_random_rf()
            c1, c2 = crossover_rf(ind1, ind2)
            assert_rf_params_valid(c1.hyperparams)
            assert_rf_params_valid(c2.hyperparams)

    def test_children_fitness_is_none(self):
        ind1, ind2 = create_random_rf(), create_random_rf()
        ind1.fitness_values = (0.9, 0.85)
        ind2.fitness_values = (0.8, 0.75)
        c1, c2 = crossover_rf(ind1, ind2)
        assert c1.fitness_values is None
        assert c2.fitness_values is None

    def test_indpb_zero_preserves_parents(self):
        """Com indpb=0, nenhum gene é trocado: filhos idênticos aos pais."""
        ind1 = create_random_rf()
        ind2 = create_random_rf()
        c1, c2 = crossover_rf(ind1, ind2, indpb=0.0)
        assert c1.hyperparams == ind1.hyperparams
        assert c2.hyperparams == ind2.hyperparams

    def test_indpb_one_swaps_all(self):
        """Com indpb=1, todos os genes são trocados: filhos com params cruzados."""
        ind1 = create_random_rf()
        ind2 = create_random_rf()
        c1, c2 = crossover_rf(ind1, ind2, indpb=1.0)
        assert c1.hyperparams == ind2.hyperparams
        assert c2.hyperparams == ind1.hyperparams

    def test_indpb_half_swaps_approximately_half(self):
        """indpb=0.5 → ~50% dos genes RF trocados em média ao longo de repetições.

        Valida o comportamento estatístico do cxUniform adaptado para dicts
        (ADR-003): cada gene tem 50% de chance de ser trocado independentemente.
        Com N=200 repetições, a taxa média deve convergir para [0.3, 0.7].
        """
        _random.seed(42)
        ind1_hp = {"n_estimators": 10, "max_depth": 3,
                   "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
        ind2_hp = {"n_estimators": 100, "max_depth": 10,
                   "min_samples_split": 5, "min_samples_leaf": 3, "criterion": "entropy"}
        keys = list(ind1_hp.keys())
        n_genes = len(keys)
        swap_counts = []
        for _ in range(200):
            c1, _ = crossover_rf(IndividuoRF(ind1_hp), IndividuoRF(ind2_hp), indpb=0.5)
            swaps = sum(1 for k in keys if c1.hyperparams[k] != ind1_hp[k])
            swap_counts.append(swaps)
        avg_swap_rate = sum(swap_counts) / (len(swap_counts) * n_genes)
        assert 0.3 <= avg_swap_rate <= 0.7, (
            f"Taxa de swap RF com indpb=0.5 foi {avg_swap_rate:.2f}, esperado entre 0.3 e 0.7"
        )


# ---------------------------------------------------------------------------
# Crossover KNN
# ---------------------------------------------------------------------------

class TestCrossoverKNN:
    def test_returns_two_knn_individuals(self):
        ind1 = create_random_knn()
        ind2 = create_random_knn()
        c1, c2 = crossover_knn(ind1, ind2)
        assert isinstance(c1, IndividuoKNN)
        assert isinstance(c2, IndividuoKNN)

    def test_children_have_valid_params(self):
        for _ in range(10):
            ind1, ind2 = create_random_knn(), create_random_knn()
            c1, c2 = crossover_knn(ind1, ind2)
            assert_knn_params_valid(c1.hyperparams)
            assert_knn_params_valid(c2.hyperparams)

    def test_indpb_zero_preserves_parents(self):
        ind1 = create_random_knn()
        ind2 = create_random_knn()
        c1, c2 = crossover_knn(ind1, ind2, indpb=0.0)
        assert c1.hyperparams == ind1.hyperparams
        assert c2.hyperparams == ind2.hyperparams

    def test_indpb_one_swaps_all(self):
        ind1 = create_random_knn()
        ind2 = create_random_knn()
        c1, c2 = crossover_knn(ind1, ind2, indpb=1.0)
        assert c1.hyperparams == ind2.hyperparams
        assert c2.hyperparams == ind1.hyperparams

    def test_indpb_half_swaps_approximately_half(self):
        """indpb=0.5 → ~50% dos genes KNN trocados em média ao longo de repetições.

        Análogo ao teste RF: valida o comportamento estatístico do cxUniform
        adaptado para dicts KNN (ADR-003).
        """
        _random.seed(42)
        ind1_hp = {"n_neighbors": 3, "weights": "uniform",
                   "metric": "euclidean", "algorithm": "auto"}
        ind2_hp = {"n_neighbors": 15, "weights": "distance",
                   "metric": "manhattan", "algorithm": "kd_tree"}
        keys = list(ind1_hp.keys())
        n_genes = len(keys)
        swap_counts = []
        for _ in range(200):
            c1, _ = crossover_knn(IndividuoKNN(ind1_hp), IndividuoKNN(ind2_hp), indpb=0.5)
            swaps = sum(1 for k in keys if c1.hyperparams[k] != ind1_hp[k])
            swap_counts.append(swaps)
        avg_swap_rate = sum(swap_counts) / (len(swap_counts) * n_genes)
        assert 0.3 <= avg_swap_rate <= 0.7, (
            f"Taxa de swap KNN com indpb=0.5 foi {avg_swap_rate:.2f}, esperado entre 0.3 e 0.7"
        )


# ---------------------------------------------------------------------------
# Mutação RF — todos os níveis de aggressiveness
# ---------------------------------------------------------------------------

class TestMutateRF:
    @pytest.mark.parametrize("aggressiveness", ["low", "medium", "high"])
    def test_returns_rf_individual(self, aggressiveness):
        ind = create_random_rf()
        (mutated,) = mutate_rf(ind, aggressiveness)
        assert isinstance(mutated, IndividuoRF)

    @pytest.mark.parametrize("aggressiveness", ["low", "medium", "high"])
    def test_params_remain_valid(self, aggressiveness):
        for _ in range(20):
            ind = create_random_rf()
            (mutated,) = mutate_rf(ind, aggressiveness)
            assert_rf_params_valid(mutated.hyperparams)

    def test_fitness_is_none_after_mutation(self):
        ind = create_random_rf()
        ind.fitness_values = (0.9, 0.85)
        (mutated,) = mutate_rf(ind)
        assert mutated.fitness_values is None

    def test_does_not_mutate_original(self):
        ind = create_random_rf()
        original_hp = dict(ind.hyperparams)
        mutate_rf(ind, "high")
        assert ind.hyperparams == original_hp


# ---------------------------------------------------------------------------
# Mutação KNN — todos os níveis de aggressiveness
# ---------------------------------------------------------------------------

class TestMutateKNN:
    @pytest.mark.parametrize("aggressiveness", ["low", "medium", "high"])
    def test_returns_knn_individual(self, aggressiveness):
        ind = create_random_knn()
        (mutated,) = mutate_knn(ind, aggressiveness)
        assert isinstance(mutated, IndividuoKNN)

    @pytest.mark.parametrize("aggressiveness", ["low", "medium", "high"])
    def test_params_remain_valid(self, aggressiveness):
        for _ in range(20):
            ind = create_random_knn()
            (mutated,) = mutate_knn(ind, aggressiveness)
            assert_knn_params_valid(mutated.hyperparams)

    def test_fitness_is_none_after_mutation(self):
        ind = create_random_knn()
        ind.fitness_values = (0.9, 0.85)
        (mutated,) = mutate_knn(ind)
        assert mutated.fitness_values is None

    def test_does_not_mutate_original(self):
        ind = create_random_knn()
        original_hp = dict(ind.hyperparams)
        mutate_knn(ind, "high")
        assert ind.hyperparams == original_hp


# ---------------------------------------------------------------------------
# Magnitude de mutação RF por nível de agressividade (ADR-008)
# ---------------------------------------------------------------------------

class TestMutateRFAggressiveness:
    """Valida que os 3 níveis de aggressiveness produzem magnitudes de mutação
    distintas e ordenadas (high >= medium >= low) para n_estimators RF.

    ADR-008 define: low=±10%, medium=±30%, high=±60% do range.
    Com N=100 amostras e seed fixa, a ordenação deve ser consistente.
    """

    def _avg_delta_rf(self, aggressiveness: str, n: int = 100) -> float:
        """Retorna delta médio absoluto em n_estimators para o nível dado."""
        _random.seed(42)
        base = IndividuoRF({"n_estimators": 100, "max_depth": 5,
                             "min_samples_split": 5, "min_samples_leaf": 2,
                             "criterion": "gini"})
        deltas = []
        for _ in range(n):
            (mutated,) = mutate_rf(base, aggressiveness)
            deltas.append(abs(mutated.hyperparams["n_estimators"] - base.hyperparams["n_estimators"]))
        return statistics.mean(deltas)

    def test_aggressiveness_ordering_n_estimators(self):
        """Verifica que high >= medium >= low em delta médio de n_estimators."""
        delta_low = self._avg_delta_rf("low")
        delta_medium = self._avg_delta_rf("medium")
        delta_high = self._avg_delta_rf("high")
        assert delta_high >= delta_medium >= delta_low, (
            f"Ordenação RF violada: low={delta_low:.2f}, "
            f"medium={delta_medium:.2f}, high={delta_high:.2f}"
        )


# ---------------------------------------------------------------------------
# Magnitude de mutação KNN por nível de agressividade (ADR-008)
# ---------------------------------------------------------------------------

class TestMutateKNNAggressiveness:
    """Valida que os 3 níveis de aggressiveness produzem magnitudes de mutação
    distintas e ordenadas (high >= medium >= low) para n_neighbors KNN.

    ADR-008 define: low=±10%, medium=±30%, high=±60% do range.
    """

    def _avg_delta_knn(self, aggressiveness: str, n: int = 100) -> float:
        """Retorna delta médio absoluto em n_neighbors para o nível dado."""
        _random.seed(42)
        base = IndividuoKNN({"n_neighbors": 15, "weights": "uniform",
                              "metric": "euclidean", "algorithm": "auto"})
        deltas = []
        for _ in range(n):
            (mutated,) = mutate_knn(base, aggressiveness)
            deltas.append(abs(mutated.hyperparams["n_neighbors"] - base.hyperparams["n_neighbors"]))
        return statistics.mean(deltas)

    def test_aggressiveness_ordering_n_neighbors(self):
        """Verifica que high >= medium >= low em delta médio de n_neighbors."""
        delta_low = self._avg_delta_knn("low")
        delta_medium = self._avg_delta_knn("medium")
        delta_high = self._avg_delta_knn("high")
        assert delta_high >= delta_medium >= delta_low, (
            f"Ordenação KNN violada: low={delta_low:.2f}, "
            f"medium={delta_medium:.2f}, high={delta_high:.2f}"
        )
