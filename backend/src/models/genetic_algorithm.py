"""
Algoritmo Genético Co-Evolutivo para otimização de hiperparâmetros.

Mantém duas populações independentes (RF e KNN) que:
  - Competem pelo fitness global (pressão seletiva compartilhada)
  - Cruzam apenas dentro do mesmo tipo (operadores específicos por tipo)

Critérios de parada:
  1. max_generations: número máximo de gerações
  2. patience: gerações consecutivas sem melhoria no best fitness global

Parâmetros configuráveis:
  - pop_size, max_generations, patience, k_folds
  - mutation_aggressiveness: 'low' | 'medium' | 'high'
  - elitism: bool — preserva o melhor indivíduo global a cada geração
  - indpb, cxpb, mutpb
  - random_seed
"""

from __future__ import annotations

import copy
import logging
import random
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from src.models.ga_evaluator import evaluate, fitness_score
from src.models.ga_operators import (
    create_random_knn,
    create_random_rf,
    crossover_knn,
    crossover_rf,
    mutate_knn,
    mutate_rf,
)
from src.models.ga_snapshot import save_generation_snapshot
from src.models.individuo import Individuo, IndividuoKNN, IndividuoRF

logger = logging.getLogger(__name__)

AggressivenessLevel = Literal["low", "medium", "high"]

# Tolerância mínima para considerar melhoria no fitness (evitar ruído numérico)
_CONVERGENCE_EPS = 1e-6


class GeneticAlgorithm:
    """
    Orquestrador do GA Co-Evolutivo.

    Attributes:
        X: Features de treinamento.
        y: Target de treinamento.
        pop_size: Número de indivíduos por tipo (total = 2 * pop_size).
        max_generations: Limite máximo de gerações (critério de parada 1).
        patience: Gerações sem melhoria para parada antecipada (critério 2).
        k_folds: Número de folds para k-Fold Cross Validation.
        mutation_aggressiveness: Intensidade das mutações.
        elitism: Se True, preserva o melhor indivíduo global a cada geração.
        indpb: Probabilidade de swap por gene no crossover uniforme.
        cxpb: Probabilidade de crossover de um par.
        mutpb: Probabilidade de mutação de um indivíduo.
        random_seed: Semente para reprodutibilidade.
    """

    def __init__(
        self,
        X,
        y,
        pop_size: int = 20,
        max_generations: int = 10,
        patience: int = 5,
        k_folds: int = 5,
        mutation_aggressiveness: AggressivenessLevel = "medium",
        elitism: bool = True,
        indpb: float = 0.5,
        cxpb: float = 0.7,
        mutpb: float = 0.3,
        random_seed: int = 42,
        job_id: str | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.patience = patience
        self.k_folds = k_folds
        self.mutation_aggressiveness = mutation_aggressiveness
        self.elitism = elitism
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.random_seed = random_seed
        self.job_id = job_id

        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(
            "GeneticAlgorithm inicializado | pop=%d/tipo | max_gen=%d | "
            "patience=%d | k_folds=%d | aggressiveness=%s | elitism=%s | job_id=%s",
            pop_size, max_generations, patience, k_folds,
            mutation_aggressiveness, elitism, job_id,
        )

    # ------------------------------------------------------------------
    # Inicialização de populações
    # ------------------------------------------------------------------

    def _init_populations(self) -> tuple[list[IndividuoRF], list[IndividuoKNN]]:
        """Cria as populações iniciais de RF e KNN."""
        pop_rf = [create_random_rf() for _ in range(self.pop_size)]
        pop_knn = [create_random_knn() for _ in range(self.pop_size)]
        return pop_rf, pop_knn

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def _evaluate_unevaluated(self, pool: list[Individuo]) -> None:
        """Avalia apenas os indivíduos ainda sem fitness_values."""
        unevaluated = [ind for ind in pool if ind.fitness_values is None]
        for ind in unevaluated:
            evaluate(ind, self.X, self.y, k_folds=self.k_folds)

    # ------------------------------------------------------------------
    # Seleção por torneio (global — RF e KNN competem juntos)
    # ------------------------------------------------------------------

    def _tournament_select(
        self, pool: list[Individuo], k: int, tournsize: int = 3
    ) -> list[Individuo]:
        """
        Seleção por torneio sobre o pool global.

        Seleciona `k` indivíduos do pool realizando torneios de `tournsize`
        participantes. O vencedor de cada torneio é o de maior fitness escalar.

        Args:
            pool: Pool global (RF + KNN misturados).
            k: Número de indivíduos a selecionar.
            tournsize: Tamanho do torneio.

        Returns:
            Lista de k indivíduos selecionados (cópias).
        """
        selected = []
        for _ in range(k):
            contestants = random.sample(pool, min(tournsize, len(pool)))
            winner = max(contestants, key=lambda i: fitness_score(i.fitness_values or (0.0, 0.0)))
            selected.append(copy.deepcopy(winner))
        return selected

    # ------------------------------------------------------------------
    # Separação por tipo após seleção global
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_type(
        selected: list[Individuo],
        min_rf: int = 1,
        min_knn: int = 1,
    ) -> tuple[list[IndividuoRF], list[IndividuoKNN]]:
        """
        Separa indivíduos selecionados por tipo, garantindo sobrevivência mínima.

        Mesmo que a seleção global favoreça fortemente um tipo, garante ao
        menos `min_rf` e `min_knn` sobreviventes de cada (elitismo estrutural
        mínimo — independente do flag `elitism`).

        Args:
            selected: Lista misturada de indivíduos após seleção global.
            min_rf: Mínimo de IndividuoRF a preservar.
            min_knn: Mínimo de IndividuoKNN a preservar.

        Returns:
            Tupla (pop_rf, pop_knn) separadas.
        """
        pop_rf = [i for i in selected if isinstance(i, IndividuoRF)]
        pop_knn = [i for i in selected if isinstance(i, IndividuoKNN)]

        # Garante sobrevivência mínima de cada tipo
        if len(pop_rf) < min_rf:
            extras = [create_random_rf() for _ in range(min_rf - len(pop_rf))]
            pop_rf.extend(extras)
            logger.debug("Elitismo estrutural RF: adicionados %d indivíduos aleatórios", len(extras))

        if len(pop_knn) < min_knn:
            extras = [create_random_knn() for _ in range(min_knn - len(pop_knn))]
            pop_knn.extend(extras)
            logger.debug("Elitismo estrutural KNN: adicionados %d indivíduos aleatórios", len(extras))

        return pop_rf, pop_knn

    # ------------------------------------------------------------------
    # Crossover e mutação por tipo
    # ------------------------------------------------------------------

    def _apply_operators_rf(
        self, pop_rf: list[IndividuoRF]
    ) -> list[IndividuoRF]:
        """Aplica crossover uniforme e mutação na população RF."""
        offspring: list[IndividuoRF] = []
        individuals = list(pop_rf)
        random.shuffle(individuals)

        # Crossover em pares
        for i in range(0, len(individuals) - 1, 2):
            a, b = individuals[i], individuals[i + 1]
            if random.random() < self.cxpb:
                c1, c2 = crossover_rf(a, b, indpb=self.indpb)
                offspring.extend([c1, c2])
            else:
                offspring.extend([copy.deepcopy(a), copy.deepcopy(b)])

        if len(individuals) % 2 == 1:
            offspring.append(copy.deepcopy(individuals[-1]))

        # Mutação
        mutated: list[IndividuoRF] = []
        for ind in offspring:
            if random.random() < self.mutpb:
                (new_ind,) = mutate_rf(ind, self.mutation_aggressiveness)
                mutated.append(new_ind)
            else:
                mutated.append(ind)

        return mutated

    def _apply_operators_knn(
        self, pop_knn: list[IndividuoKNN]
    ) -> list[IndividuoKNN]:
        """Aplica crossover uniforme e mutação na população KNN."""
        offspring: list[IndividuoKNN] = []
        individuals = list(pop_knn)
        random.shuffle(individuals)

        for i in range(0, len(individuals) - 1, 2):
            a, b = individuals[i], individuals[i + 1]
            if random.random() < self.cxpb:
                c1, c2 = crossover_knn(a, b, indpb=self.indpb)
                offspring.extend([c1, c2])
            else:
                offspring.extend([copy.deepcopy(a), copy.deepcopy(b)])

        if len(individuals) % 2 == 1:
            offspring.append(copy.deepcopy(individuals[-1]))

        mutated: list[IndividuoKNN] = []
        for ind in offspring:
            if random.random() < self.mutpb:
                (new_ind,) = mutate_knn(ind, self.mutation_aggressiveness)
                mutated.append(new_ind)
            else:
                mutated.append(ind)

        return mutated

    # ------------------------------------------------------------------
    # Logging por geração
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_stats(
        gen: int,
        pop_rf: list[IndividuoRF],
        pop_knn: list[IndividuoKNN],
        stopped_early: bool = False,
    ) -> dict:
        """Coleta estatísticas de uma geração para logging e persistência."""

        def _stats(pop: list[Individuo]) -> dict:
            scores = [fitness_score(i.fitness_values or (0.0, 0.0)) for i in pop]
            f1s = [(i.fitness_values or (0.0, 0.0))[0] for i in pop]
            accs = [(i.fitness_values or (0.0, 0.0))[1] for i in pop]
            best = max(pop, key=lambda i: fitness_score(i.fitness_values or (0.0, 0.0)))
            return {
                "count": len(pop),
                "best_f1": round(max(f1s), 6),
                "avg_f1": round(float(np.mean(f1s)), 6),
                "best_acc": round(max(accs), 6),
                "avg_acc": round(float(np.mean(accs)), 6),
                "best_score": round(max(scores), 6),
                "avg_score": round(float(np.mean(scores)), 6),
                "best_hyperparams": best.hyperparams,
            }

        pool = pop_rf + pop_knn
        global_best = max(
            pool,
            key=lambda i: fitness_score(i.fitness_values or (0.0, 0.0)),
        )

        return {
            "generation": gen,
            "rf": _stats(pop_rf),
            "knn": _stats(pop_knn),
            "global_best_f1": round((global_best.fitness_values or (0.0, 0.0))[0], 6),
            "global_best_score": round(fitness_score(global_best.fitness_values or (0.0, 0.0)), 6),
            "global_best_type": global_best.classifier_type,
            "stopped_early": stopped_early,
        }

    # ------------------------------------------------------------------
    # Loop principal co-evolutivo
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Executa o Algoritmo Genético Co-Evolutivo.

        Fluxo por geração:
            1. Avaliação global (RF + KNN juntos)
            2. Verificação de critério de parada por convergência
            3. Seleção por torneio global (RF e KNN competem)
            4. Separação por tipo (com elitismo estrutural mínimo)
            5. Elitismo opcional (preserva o best global)
            6. Crossover + Mutação por tipo
            7. Logging de estatísticas

        Returns:
            dict com chaves:
                - "generations_stats": lista de dicts por geração
                - "best_individual": Individuo com maior fitness_score
                - "stopped_at": geração em que parou (0-indexed)
                - "reason": "max_generations" | "convergence"
                - "params": parâmetros usados na execução
        """
        logger.info("Iniciando GA co-evolutivo...")
        pop_rf, pop_knn = self._init_populations()

        best_fitness_global: float = -float("inf")
        no_improvement_count: int = 0
        generations_stats: list[dict] = []
        overall_best: Individuo | None = None
        stop_reason = "max_generations"

        for gen in range(self.max_generations):
            pool = pop_rf + pop_knn

            # ---- Avaliação global ----
            self._evaluate_unevaluated(pool)

            # ---- Critério de parada por convergência ----
            current_best = max(
                pool,
                key=lambda i: fitness_score(i.fitness_values or (0.0, 0.0)),
            )
            current_score = fitness_score(current_best.fitness_values or (0.0, 0.0))

            if current_score > best_fitness_global + _CONVERGENCE_EPS:
                best_fitness_global = current_score
                overall_best = copy.deepcopy(current_best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            logger.info(
                "[Gen %02d/%02d] Best=%s(score=%.4f) | RF=%d KNN=%d | no_improve=%d/%d",
                gen + 1, self.max_generations,
                current_best.classifier_type, current_score,
                len(pop_rf), len(pop_knn),
                no_improvement_count, self.patience,
            )

            # Estatísticas ANTES dos operadores (representa o estado da geração)
            stopped_early = no_improvement_count >= self.patience
            gen_stat = self._gen_stats(gen + 1, pop_rf, pop_knn, stopped_early)
            generations_stats.append(gen_stat)

            # --- Snapshot em tempo real (Opção C — JSONL interno ao backend) ---
            if self.job_id is not None:
                pool_scores = [
                    fitness_score(i.fitness_values or (0.0, 0.0)) for i in pool
                ]
                pool_accs = [(i.fitness_values or (0.0, 0.0))[1] for i in pool]
                snapshot = {
                    "generation": gen + 1,
                    "best_fitness": gen_stat["global_best_score"],
                    "avg_fitness": round(float(np.mean(pool_scores)), 6),
                    "std_fitness": round(float(np.std(pool_scores)), 6),
                    "best_f2": gen_stat["global_best_f1"],
                    "best_accuracy": round(max(pool_accs), 6),
                    "best_params": (
                        overall_best.hyperparams if overall_best is not None else {}
                    ),
                    "model_type": gen_stat["global_best_type"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_generation_snapshot(self.job_id, snapshot)

            if stopped_early:
                stop_reason = "convergence"
                logger.info(
                    "Parada antecipada: %d gerações sem melhoria (patience=%d)",
                    no_improvement_count, self.patience,
                )
                break

            # ---- Seleção global (RF e KNN competem juntos) ----
            selected = self._tournament_select(pool, k=len(pool), tournsize=3)
            pop_rf, pop_knn = self._split_by_type(selected, min_rf=1, min_knn=1)

            # ---- Elitismo opcional ----
            elite: Individuo | None = None
            if self.elitism and overall_best is not None:
                elite = copy.deepcopy(overall_best)

            # ---- Crossover + Mutação por tipo ----
            pop_rf = self._apply_operators_rf(pop_rf)
            pop_knn = self._apply_operators_knn(pop_knn)

            # ---- Reinsere elite substituindo o pior da próxima geração ----
            if elite is not None:
                # Determinar em qual sub-pop reinserir pelo tipo
                if isinstance(elite, IndividuoRF):
                    if pop_rf:
                        worst_idx = min(
                            range(len(pop_rf)),
                            key=lambda i: fitness_score(pop_rf[i].fitness_values or (0.0, 0.0)),
                        )
                        pop_rf[worst_idx] = elite
                else:
                    if pop_knn:
                        worst_idx = min(
                            range(len(pop_knn)),
                            key=lambda i: fitness_score(pop_knn[i].fitness_values or (0.0, 0.0)),
                        )
                        pop_knn[worst_idx] = elite

        # Avaliação final do pool restante (para garantir fitness nos filhos)
        final_pool = pop_rf + pop_knn
        self._evaluate_unevaluated(final_pool)

        if overall_best is None and final_pool:
            overall_best = max(
                final_pool,
                key=lambda i: fitness_score(i.fitness_values or (0.0, 0.0)),
            )

        logger.info(
            "GA finalizado | Motivo: %s | Melhor: %s (score=%.4f)",
            stop_reason,
            overall_best,
            fitness_score(overall_best.fitness_values or (0.0, 0.0)) if overall_best else 0.0,
        )

        return {
            "generations_stats": generations_stats,
            "best_individual": overall_best,
            "stopped_at": len(generations_stats),
            "reason": stop_reason,
            "params": {
                "pop_size": self.pop_size,
                "max_generations": self.max_generations,
                "patience": self.patience,
                "k_folds": self.k_folds,
                "mutation_aggressiveness": self.mutation_aggressiveness,
                "elitism": self.elitism,
                "indpb": self.indpb,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "random_seed": self.random_seed,
            },
        }
