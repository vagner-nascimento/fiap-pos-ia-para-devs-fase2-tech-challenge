"""Testes unitários para o módulo de indivíduos do GA."""

import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.models.individuo import (
    Individuo,
    IndividuoRF,
    IndividuoKNN,
    RF_HYPERPARAM_SPACE,
    KNN_HYPERPARAM_SPACE,
)


class TestIndividuoBase:
    def test_individuo_abstract(self):
        """Testa que Individuo é abstrato."""
        with pytest.raises(TypeError):
            Individuo({"param": "value"})

    def test_individuo_init(self):
        """Testa inicialização de IndividuoRF."""
        individual = IndividuoRF({"n_estimators": 100})
        
        assert individual.hyperparams == {"n_estimators": 100}
        assert individual.fitness_values is None

    def test_individuo_fitness_assignment(self):
        """Testa atribuição de fitness."""
        individual = IndividuoRF({"n_estimators": 100})
        individual.fitness_values = (0.85, 0.90)
        
        assert individual.fitness_values == (0.85, 0.90)

    def test_individuo_clone(self):
        """Testa clonagem de indivíduo."""
        individual = IndividuoRF({"n_estimators": 100})
        individual.fitness_values = (0.85, 0.90)
        
        clone = individual.clone()
        
        assert clone.hyperparams == individual.hyperparams
        assert clone.fitness_values is None  # Fitness não é copiado
        assert clone is not individual

    def test_individuo_classifier_type_rf(self):
        """Testa propriedade classifier_type para RF."""
        individual = IndividuoRF({"n_estimators": 100})
        
        assert individual.classifier_type == "RF"

    def test_individuo_classifier_type_knn(self):
        """Testa propriedade classifier_type para KNN."""
        individual = IndividuoKNN({"n_neighbors": 5})
        
        assert individual.classifier_type == "KNN"

    def test_individuo_repr_with_fitness(self):
        """Testa representação com fitness."""
        individual = IndividuoRF({"n_estimators": 100})
        individual.fitness_values = (0.85, 0.90)
        
        repr_str = repr(individual)
        
        assert "RF" in repr_str
        assert "F1=0.8500" in repr_str
        assert "Acc=0.9000" in repr_str

    def test_individuo_repr_without_fitness(self):
        """Testa representação sem fitness."""
        individual = IndividuoRF({"n_estimators": 100})
        
        repr_str = repr(individual)
        
        assert "RF" in repr_str
        assert "não avaliado" in repr_str


class TestIndividuoRF:
    def test_rf_build_model(self):
        """Testa construção de modelo RF."""
        individual = IndividuoRF({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        })
        
        pipeline = individual.build_model()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "clf"
        assert isinstance(pipeline.steps[0][1], RandomForestClassifier)

    def test_rf_build_model_params(self):
        """Testa que hiperparâmetros são passados corretamente."""
        individual = IndividuoRF({
            "n_estimators": 150,
            "max_depth": 15,
            "criterion": "entropy"
        })
        
        pipeline = individual.build_model()
        clf = pipeline.steps[0][1]
        
        assert clf.n_estimators == 150
        assert clf.max_depth == 15
        assert clf.criterion == "entropy"
        assert clf.random_state == 42
        assert clf.n_jobs == -1

    def test_rf_hyperparam_space(self):
        """Testa definição do espaço de hiperparâmetros RF."""
        assert "n_estimators" in RF_HYPERPARAM_SPACE
        assert "max_depth" in RF_HYPERPARAM_SPACE
        assert "min_samples_split" in RF_HYPERPARAM_SPACE
        assert "min_samples_leaf" in RF_HYPERPARAM_SPACE
        assert "criterion" in RF_HYPERPARAM_SPACE

    def test_rf_hyperparam_space_values(self):
        """Testa valores do espaço de hiperparâmetros RF."""
        assert RF_HYPERPARAM_SPACE["n_estimators"] == (10, 500)
        assert RF_HYPERPARAM_SPACE["max_depth"] == (2, 30)
        assert RF_HYPERPARAM_SPACE["criterion"] == ["gini", "entropy"]


class TestIndividuoKNN:
    def test_knn_build_model(self):
        """Testa construção de modelo KNN."""
        individual = IndividuoKNN({
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "algorithm": "auto"
        })
        
        pipeline = individual.build_model()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "scaler"
        assert isinstance(pipeline.steps[0][1], StandardScaler)
        assert pipeline.steps[1][0] == "clf"
        assert isinstance(pipeline.steps[1][1], KNeighborsClassifier)

    def test_knn_build_model_params(self):
        """Testa que hiperparâmetros são passados corretamente."""
        individual = IndividuoKNN({
            "n_neighbors": 10,
            "weights": "distance",
            "metric": "manhattan",
            "algorithm": "ball_tree"
        })
        
        pipeline = individual.build_model()
        clf = pipeline.steps[1][1]
        
        assert clf.n_neighbors == 10
        assert clf.weights == "distance"
        assert clf.metric == "manhattan"
        assert clf.algorithm == "ball_tree"

    def test_knn_hyperparam_space(self):
        """Testa definição do espaço de hiperparâmetros KNN."""
        assert "n_neighbors" in KNN_HYPERPARAM_SPACE
        assert "weights" in KNN_HYPERPARAM_SPACE
        assert "metric" in KNN_HYPERPARAM_SPACE
        assert "algorithm" in KNN_HYPERPARAM_SPACE

    def test_knn_hyperparam_space_values(self):
        """Testa valores do espaço de hiperparâmetros KNN."""
        assert KNN_HYPERPARAM_SPACE["n_neighbors"] == (1, 30)
        assert KNN_HYPERPARAM_SPACE["weights"] == ["uniform", "distance"]
        assert KNN_HYPERPARAM_SPACE["metric"] == ["euclidean", "manhattan", "minkowski"]
        assert KNN_HYPERPARAM_SPACE["algorithm"] == ["auto", "ball_tree", "kd_tree"]


class TestIndividuoIntegration:
    def test_rf_vs_knn_pipeline_structure(self):
        """Testa que RF e KNN têm estruturas de pipeline diferentes."""
        rf = IndividuoRF({"n_estimators": 100})
        knn = IndividuoKNN({"n_neighbors": 5})
        
        rf_pipeline = rf.build_model()
        knn_pipeline = knn.build_model()
        
        # RF não tem scaler
        assert len(rf_pipeline.steps) == 1
        # KNN tem scaler
        assert len(knn_pipeline.steps) == 2

    def test_multiple_clones(self):
        """Testa múltiplas clonagens do mesmo indivíduo."""
        original = IndividuoRF({"n_estimators": 100})
        original.fitness_values = (0.85, 0.90)
        
        clone1 = original.clone()
        clone2 = original.clone()
        
        assert clone1 is not clone2
        assert clone2 is not original
        assert clone1.hyperparams == original.hyperparams
        assert clone2.hyperparams == original.hyperparams
