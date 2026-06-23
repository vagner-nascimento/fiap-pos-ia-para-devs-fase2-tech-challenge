"""
Operadores genéticos do GA Co-Evolutivo.

Cada operador é específico ao tipo de indivíduo (RF ou KNN).
Não há operadores mistos entre tipos.

Operadores disponíveis:
    Geração aleatória : create_random_rf(), create_random_knn()
    Crossover uniforme: crossover_rf(ind1, ind2, indpb), crossover_knn(ind1, ind2, indpb)
    Mutação           : mutate_rf(ind, aggressiveness), mutate_knn(ind, aggressiveness)
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Literal

from src.models.individuo import (
    RF_HYPERPARAM_SPACE,
    KNN_HYPERPARAM_SPACE,
    IndividuoRF,
    IndividuoKNN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tabela de agressividade de mutação
# ---------------------------------------------------------------------------

AggressivenessLevel = Literal["low", "medium", "high"]

_AGGRESSIVENESS_MAP: dict[str, dict] = {
    "low": {
        "mutation_rate": 0.1,   # prob de mutar cada hiperparâmetro
        "delta_pct": 0.10,      # perturbação gaussiana: ±10% do range
    },
    "medium": {
        "mutation_rate": 0.2,
        "delta_pct": 0.30,
    },
    "high": {
        "mutation_rate": 0.4,
        "delta_pct": 0.60,
    },
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _mutate_int(value: int, low: int, high: int, delta_pct: float) -> int:
    """Perturba um hiperparâmetro inteiro com ruído gaussiano escalado."""
    span = high - low
    sigma = max(1, int(span * delta_pct))
    new_val = value + random.randint(-sigma, sigma)
    return int(max(low, min(high, new_val)))


def _mutate_categorical(value: str, choices: list[str]) -> str:
    """Substitui um hiperparâmetro categórico por outro valor aleatório."""
    options = [c for c in choices if c != value]
    return random.choice(options) if options else value


def _uniform_crossover_dicts(
    dict1: dict, dict2: dict, indpb: float
) -> tuple[dict, dict]:
    """
    Crossover uniforme sobre dois dicionários de hiperparâmetros.

    Para cada chave, com probabilidade `indpb`, os valores dos dois pais
    são trocados. Equivale ao cxUniform do DEAP aplicado sobre dicts nomeados
    em vez de listas posicionais — necessário pois os hiperparâmetros não são
    intercambiáveis por posição.

    Args:
        dict1: Hiperparâmetros do pai 1.
        dict2: Hiperparâmetros do pai 2.
        indpb: Probabilidade de swap por gene (chave).

    Returns:
        Dois dicionários após crossover (in-place nos dicts copiados).
    """
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    for key in d1:
        if random.random() < indpb:
            d1[key], d2[key] = d2[key], d1[key]
    return d1, d2


# ---------------------------------------------------------------------------
# Geração aleatória
# ---------------------------------------------------------------------------

def create_random_rf() -> IndividuoRF:
    """
    Cria um IndividuoRF com hiperparâmetros aleatórios dentro dos ranges válidos.

    Returns:
        IndividuoRF com hyperparams amostrados uniformemente.
    """
    space = RF_HYPERPARAM_SPACE
    hyperparams = {
        "n_estimators": random.randint(*space["n_estimators"]),
        "max_depth": random.choice(
            [None] + list(range(space["max_depth"][0], space["max_depth"][1] + 1))
        ),
        "min_samples_split": random.randint(*space["min_samples_split"]),
        "min_samples_leaf": random.randint(*space["min_samples_leaf"]),
        "criterion": random.choice(space["criterion"]),
    }
    return IndividuoRF(hyperparams)


def create_random_knn() -> IndividuoKNN:
    """
    Cria um IndividuoKNN com hiperparâmetros aleatórios dentro dos ranges válidos.

    Returns:
        IndividuoKNN com hyperparams amostrados uniformemente.
    """
    space = KNN_HYPERPARAM_SPACE
    hyperparams = {
        "n_neighbors": random.randint(*space["n_neighbors"]),
        "weights": random.choice(space["weights"]),
        "metric": random.choice(space["metric"]),
        "algorithm": random.choice(space["algorithm"]),
    }
    return IndividuoKNN(hyperparams)


# ---------------------------------------------------------------------------
# Crossover uniforme (cxUniform adaptado para dicts nomeados)
# ---------------------------------------------------------------------------

def crossover_rf(
    ind1: IndividuoRF,
    ind2: IndividuoRF,
    indpb: float = 0.5,
) -> tuple[IndividuoRF, IndividuoRF]:
    """
    Crossover uniforme entre dois IndividuoRF.

    Para cada hiperparâmetro RF, com probabilidade `indpb`, os valores são
    trocados entre os dois pais. Os filhos resultantes têm fitness invalidado.

    Args:
        ind1: Pai 1 (IndividuoRF).
        ind2: Pai 2 (IndividuoRF).
        indpb: Probabilidade de swap por gene. Default=0.5.

    Returns:
        Tupla (filho1, filho2) — novos IndividuoRF com hiperparâmetros cruzados.
    """
    new_hp1, new_hp2 = _uniform_crossover_dicts(ind1.hyperparams, ind2.hyperparams, indpb)
    child1 = IndividuoRF(new_hp1)
    child2 = IndividuoRF(new_hp2)
    logger.debug("crossover_rf: %s × %s → %s, %s", ind1, ind2, child1, child2)
    return child1, child2


def crossover_knn(
    ind1: IndividuoKNN,
    ind2: IndividuoKNN,
    indpb: float = 0.5,
) -> tuple[IndividuoKNN, IndividuoKNN]:
    """
    Crossover uniforme entre dois IndividuoKNN.

    Para cada hiperparâmetro KNN, com probabilidade `indpb`, os valores são
    trocados entre os dois pais.

    Args:
        ind1: Pai 1 (IndividuoKNN).
        ind2: Pai 2 (IndividuoKNN).
        indpb: Probabilidade de swap por gene. Default=0.5.

    Returns:
        Tupla (filho1, filho2) — novos IndividuoKNN com hiperparâmetros cruzados.
    """
    new_hp1, new_hp2 = _uniform_crossover_dicts(ind1.hyperparams, ind2.hyperparams, indpb)
    child1 = IndividuoKNN(new_hp1)
    child2 = IndividuoKNN(new_hp2)
    logger.debug("crossover_knn: %s × %s → %s, %s", ind1, ind2, child1, child2)
    return child1, child2


# ---------------------------------------------------------------------------
# Mutação com controle de agressividade
# ---------------------------------------------------------------------------

def mutate_rf(
    individual: IndividuoRF,
    aggressiveness: AggressivenessLevel = "medium",
) -> tuple[IndividuoRF]:
    """
    Mutação de um IndividuoRF com intensidade configurável.

    Parâmetros numéricos (n_estimators, max_depth, min_samples_split,
    min_samples_leaf) sofrem perturbação gaussiana escalada pelo delta_pct.
    Parâmetro categórico (criterion) é substituído aleatoriamente com prob=mutation_rate.
    O indivíduo original é copiado; a mutação é aplicada na cópia.

    Args:
        individual: IndividuoRF a mutar.
        aggressiveness: Nível de agressividade ('low', 'medium', 'high').

    Returns:
        Tupla com um único IndividuoRF mutado (convenção DEAP).
    """
    cfg = _AGGRESSIVENESS_MAP[aggressiveness]
    rate = cfg["mutation_rate"]
    delta = cfg["delta_pct"]

    hp = copy.deepcopy(individual.hyperparams)
    space = RF_HYPERPARAM_SPACE

    if random.random() < rate:
        hp["n_estimators"] = _mutate_int(
            hp["n_estimators"], *space["n_estimators"], delta
        )

    if random.random() < rate:
        if hp["max_depth"] is None:
            hp["max_depth"] = random.randint(*space["max_depth"])
        elif random.random() < 0.1:
            hp["max_depth"] = None          # 10% chance de voltar para None
        else:
            hp["max_depth"] = _mutate_int(
                hp["max_depth"], *space["max_depth"], delta
            )

    if random.random() < rate:
        hp["min_samples_split"] = _mutate_int(
            hp["min_samples_split"], *space["min_samples_split"], delta
        )

    if random.random() < rate:
        hp["min_samples_leaf"] = _mutate_int(
            hp["min_samples_leaf"], *space["min_samples_leaf"], delta
        )

    if random.random() < rate:
        hp["criterion"] = _mutate_categorical(hp["criterion"], space["criterion"])

    mutated = IndividuoRF(hp)
    logger.debug("mutate_rf [%s]: %s → %s", aggressiveness, individual.hyperparams, hp)
    return (mutated,)


def mutate_knn(
    individual: IndividuoKNN,
    aggressiveness: AggressivenessLevel = "medium",
) -> tuple[IndividuoKNN]:
    """
    Mutação de um IndividuoKNN com intensidade configurável.

    n_neighbors sofre perturbação inteira escalada.
    weights, metric e algorithm são substituídos aleatoriamente com prob=mutation_rate.
    O indivíduo original é copiado; a mutação é aplicada na cópia.

    Args:
        individual: IndividuoKNN a mutar.
        aggressiveness: Nível de agressividade ('low', 'medium', 'high').

    Returns:
        Tupla com um único IndividuoKNN mutado (convenção DEAP).
    """
    cfg = _AGGRESSIVENESS_MAP[aggressiveness]
    rate = cfg["mutation_rate"]
    delta = cfg["delta_pct"]

    hp = copy.deepcopy(individual.hyperparams)
    space = KNN_HYPERPARAM_SPACE

    if random.random() < rate:
        hp["n_neighbors"] = _mutate_int(
            hp["n_neighbors"], *space["n_neighbors"], delta
        )

    if random.random() < rate:
        hp["weights"] = _mutate_categorical(hp["weights"], space["weights"])

    if random.random() < rate:
        hp["metric"] = _mutate_categorical(hp["metric"], space["metric"])

    if random.random() < rate:
        hp["algorithm"] = _mutate_categorical(hp["algorithm"], space["algorithm"])

    mutated = IndividuoKNN(hp)
    logger.debug("mutate_knn [%s]: %s → %s", aggressiveness, individual.hyperparams, hp)
    return (mutated,)
