"""
Módulo de avaliação de fitness do Algoritmo Genético Co-Evolutivo.

A função `evaluate` é agnóstica ao tipo de indivíduo (RF ou KNN):
    - Constrói o pipeline via `individual.build_model()`
    - Executa k-Fold Cross Validation
    - Retorna fitness ponderado: F1_weighted * 0.6 + Accuracy * 0.4

Decisão de design:
    F1_weighted foi escolhido como métrica primária pois os dados do SISVAN
    podem apresentar desbalanceamento de classes entre os estados nutricionais.
    A Acurácia entra com peso menor para penalizar modelos que ignoram classes
    minoritárias.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import cross_val_score

from src.models.individuo import Individuo

logger = logging.getLogger(__name__)

# Pesos do fitness composto
_WEIGHT_F1 = 0.6
_WEIGHT_ACC = 0.4


def evaluate(
    individual: Individuo,
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
) -> tuple[float, float]:
    """
    Avalia um indivíduo do GA via k-Fold Cross Validation.

    O score de fitness é calculado como:
        fitness_score = F1_weighted * 0.6 + Accuracy * 0.4

    Args:
        individual: Qualquer instância de Individuo (IndividuoRF ou IndividuoKNN).
        X: Features (numpy array ou DataFrame).
        y: Target (numpy array ou Series).
        k_folds: Número de folds para cross-validation. Default=5.

    Returns:
        Tupla (f1_weighted, accuracy) com médias dos folds.
        Ambos os valores estão no intervalo [0.0, 1.0].

    Raises:
        Exception: Se o modelo falhar durante o cross_val_score, retorna (0.0, 0.0)
            e loga o erro para não interromper o loop do GA.
    """
    pipeline = individual.build_model()

    try:
        f1_scores = cross_val_score(
            pipeline, X, y,
            cv=k_folds,
            scoring="f1_weighted",
            error_score=0.0,
        )
        acc_scores = cross_val_score(
            pipeline, X, y,
            cv=k_folds,
            scoring="accuracy",
            error_score=0.0,
        )

        f1_mean = float(np.mean(f1_scores))
        acc_mean = float(np.mean(acc_scores))

        individual.fitness_values = (f1_mean, acc_mean)

        logger.debug(
            "evaluate [%s]: F1=%.4f, Acc=%.4f, fitness=%.4f",
            individual.classifier_type,
            f1_mean,
            acc_mean,
            f1_mean * _WEIGHT_F1 + acc_mean * _WEIGHT_ACC,
        )
        return (f1_mean, acc_mean)

    except Exception as exc:
        logger.warning(
            "evaluate [%s] falhou com params %s: %s",
            individual.classifier_type,
            individual.hyperparams,
            exc,
        )
        individual.fitness_values = (0.0, 0.0)
        return (0.0, 0.0)


def fitness_score(fitness_tuple: tuple[float, float]) -> float:
    """
    Calcula o score escalar de fitness ponderado a partir de uma tupla.

    Útil para comparações diretas sem depender da estrutura DEAP.

    Args:
        fitness_tuple: Tupla (f1_weighted, accuracy).

    Returns:
        float: F1 * 0.6 + Accuracy * 0.4
    """
    f1, acc = fitness_tuple
    return f1 * _WEIGHT_F1 + acc * _WEIGHT_ACC
