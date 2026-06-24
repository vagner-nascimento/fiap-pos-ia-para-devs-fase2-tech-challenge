"""
Módulo de definição dos indivíduos do Algoritmo Genético Co-Evolutivo.

Hierarquia:
    Individuo (ABC)
    ├── IndividuoRF  — RandomForestClassifier (sem scaling)
    └── IndividuoKNN — KNeighborsClassifier   (com StandardScaler no pipeline)
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Espaços de hiperparâmetros válidos (usados em ga_operators.py)
# ---------------------------------------------------------------------------

RF_HYPERPARAM_SPACE: dict = {
    "n_estimators": (10, 500),          # int  — número de árvores
    "max_depth": (2, 30),               # int  — profundidade máxima (None = ilimitado)
    "min_samples_split": (2, 20),       # int  — mínimo de samples para split
    "min_samples_leaf": (1, 10),        # int  — mínimo de samples na folha
    "criterion": ["gini", "entropy"],   # str  — critério de impureza
}

KNN_HYPERPARAM_SPACE: dict = {
    "n_neighbors": (1, 30),                              # int
    "weights": ["uniform", "distance"],                  # str
    "metric": ["euclidean", "manhattan", "minkowski"],   # str
    "algorithm": ["auto", "ball_tree", "kd_tree"],       # str
}


# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------

class Individuo(ABC):
    """
    Contrato base para todos os indivíduos do GA.

    Attributes:
        hyperparams (dict): Hiperparâmetros do modelo sklearn encapsulado.
        fitness (tuple | None): Tupla (f1_weighted, accuracy) atribuída após
            avaliação. None enquanto não avaliado.
    """

    def __init__(self, hyperparams: dict) -> None:
        self.hyperparams: dict = hyperparams
        # fitness é gerenciado pelo DEAP via FitnessMax, mas mantemos aqui
        # uma cópia para acesso direto sem depender do DEAP fora do GA.
        self.fitness_values: tuple[float, float] | None = None

    @abstractmethod
    def build_model(self) -> Pipeline:
        """Constrói e retorna o pipeline sklearn do indivíduo."""
        ...

    def clone(self) -> "Individuo":
        """Retorna uma cópia profunda do indivíduo (fitness não copiado)."""
        clone = copy.deepcopy(self)
        clone.fitness_values = None
        return clone

    @property
    def classifier_type(self) -> str:
        """Identificador do tipo ('RF' ou 'KNN')."""
        return self.__class__.__name__.replace("Individuo", "")

    def __repr__(self) -> str:
        fitness_str = (
            f"F1={self.fitness_values[0]:.4f}, Acc={self.fitness_values[1]:.4f}"
            if self.fitness_values
            else "não avaliado"
        )
        return f"{self.classifier_type}({self.hyperparams} | {fitness_str})"


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------

class IndividuoRF(Individuo):
    """
    Indivíduo RandomForest.

    Hiperparâmetros controlados:
        - n_estimators  : int  [10, 500]
        - max_depth     : int  [2, 30] | None
        - min_samples_split : int [2, 20]
        - min_samples_leaf  : int [1, 10]
        - criterion     : str  {'gini', 'entropy'}

    O pipeline não inclui scaling pois RandomForest é invariante a escala.
    """

    def build_model(self) -> Pipeline:
        """
        Constrói Pipeline com RandomForestClassifier.

        Returns:
            Pipeline: sklearn pipeline pronto para fit/predict.
        """
        clf = RandomForestClassifier(
            **self.hyperparams,
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class IndividuoKNN(Individuo):
    """
    Indivíduo KNeighborsClassifier.

    Hiperparâmetros controlados:
        - n_neighbors : int [1, 30]
        - weights     : str {'uniform', 'distance'}
        - metric      : str {'euclidean', 'manhattan', 'minkowski'}
        - algorithm   : str {'auto', 'ball_tree', 'kd_tree'}

    O pipeline inclui StandardScaler obrigatório, pois KNN é baseado em
    distância e features de escalas diferentes distorcem os resultados.
    """

    def build_model(self) -> Pipeline:
        """
        Constrói Pipeline com StandardScaler + KNeighborsClassifier.

        Returns:
            Pipeline: sklearn pipeline pronto para fit/predict.
        """
        clf = KNeighborsClassifier(**self.hyperparams)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
