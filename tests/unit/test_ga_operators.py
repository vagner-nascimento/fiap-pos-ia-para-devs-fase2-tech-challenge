"""
Testes unitários dos operadores genéticos (ga_operators.py).

Cobre:
    - create_random_rf / create_random_knn: ranges válidos
    - crossover_rf / crossover_knn: tipos preservados, params válidos
    - mutate_rf / mutate_knn: tipos preservados, params dentro do range
    - Todos os níveis de aggressiveness
"""

import pytest
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
