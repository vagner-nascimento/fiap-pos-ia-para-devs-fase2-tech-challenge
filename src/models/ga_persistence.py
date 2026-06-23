"""
Persistência de resultados do Algoritmo Genético Co-Evolutivo.

Salva:
    - ga_history.json          : histórico completo (todas as gerações + metadados)
    - ga_generation_stats.csv  : tabela por geração com stats de RF, KNN e global
    - best_model.joblib        : modelo sklearn do melhor indivíduo treinado no dataset completo

Carrega:
    - load_ga_history(path)    : devolve dict do JSON salvo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.models.individuo import Individuo
from src.utils.persistence import save_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialização do indivíduo (JSON-safe)
# ---------------------------------------------------------------------------

def _individuo_to_dict(individual: Individuo | None) -> dict | None:
    """Converte um indivíduo para dict serializável em JSON."""
    if individual is None:
        return None
    return {
        "type": individual.classifier_type,
        "hyperparams": individual.hyperparams,
        "fitness_f1": (individual.fitness_values or (None, None))[0],
        "fitness_acc": (individual.fitness_values or (None, None))[1],
    }


# ---------------------------------------------------------------------------
# Salvar resultados do GA
# ---------------------------------------------------------------------------

def save_ga_results(results: dict, output_dir: str) -> None:
    """
    Persiste os resultados do GA em JSON e CSV.

    Arquivos gerados:
        <output_dir>/ga_history.json         — histórico completo
        <output_dir>/ga_generation_stats.csv — tabela por geração

    Estrutura do CSV (uma linha por geração):
        generation | rf_count | rf_best_f1 | rf_avg_f1 | rf_best_score
                   | knn_count | knn_best_f1 | knn_avg_f1 | knn_best_score
                   | global_best_f1 | global_best_score | global_best_type
                   | stopped_early

    Args:
        results: Dict retornado por GeneticAlgorithm.run().
        output_dir: Diretório onde os arquivos serão gravados.

    Raises:
        OSError: Se não for possível criar o diretório ou gravar os arquivos.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- JSON completo ----
    json_path = out / "ga_history.json"
    history_payload = {
        "params": results.get("params", {}),
        "stopped_at": results.get("stopped_at"),
        "reason": results.get("reason"),
        "best_individual": _individuo_to_dict(results.get("best_individual")),
        "generations_stats": results.get("generations_stats", []),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Histórico GA salvo em: %s", json_path)

    # ---- CSV por geração ----
    stats = results.get("generations_stats", [])
    if stats:
        rows = []
        for s in stats:
            rf = s.get("rf", {})
            knn = s.get("knn", {})
            rows.append({
                "generation": s["generation"],
                # RF
                "rf_count": rf.get("count", 0),
                "rf_best_f1": rf.get("best_f1", 0.0),
                "rf_avg_f1": rf.get("avg_f1", 0.0),
                "rf_best_acc": rf.get("best_acc", 0.0),
                "rf_best_score": rf.get("best_score", 0.0),
                "rf_avg_score": rf.get("avg_score", 0.0),
                # KNN
                "knn_count": knn.get("count", 0),
                "knn_best_f1": knn.get("best_f1", 0.0),
                "knn_avg_f1": knn.get("avg_f1", 0.0),
                "knn_best_acc": knn.get("best_acc", 0.0),
                "knn_best_score": knn.get("best_score", 0.0),
                "knn_avg_score": knn.get("avg_score", 0.0),
                # Global
                "global_best_f1": s.get("global_best_f1", 0.0),
                "global_best_score": s.get("global_best_score", 0.0),
                "global_best_type": s.get("global_best_type", ""),
                "stopped_early": s.get("stopped_early", False),
            })

        csv_path = out / "ga_generation_stats.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("Stats por geração salvas em: %s", csv_path)


# ---------------------------------------------------------------------------
# Salvar melhor modelo
# ---------------------------------------------------------------------------

def save_best_model(
    best_individual: Individuo,
    X,
    y,
    model_path: str,
) -> None:
    """
    Treina o melhor indivíduo no dataset completo e persiste via joblib.

    O modelo é treinado no X/y completo (sem CV) antes de ser salvo,
    pois durante o GA ele só foi avaliado via cross_val_score.

    Args:
        best_individual: Indivíduo com melhor fitness do GA.
        X: Features completas para treino final.
        y: Target completo para treino final.
        model_path: Caminho completo para o arquivo .joblib de saída.

    Raises:
        Exception: Propaga exceções de fit ou joblib.dump.
    """
    pipeline = best_individual.build_model()
    logger.info(
        "Treinando melhor modelo [%s] no dataset completo (shape=%s)...",
        best_individual.classifier_type,
        getattr(X, "shape", "?"),
    )
    pipeline.fit(X, y)
    save_model(pipeline, model_path)
    logger.info("Melhor modelo salvo em: %s", model_path)


# ---------------------------------------------------------------------------
# Carregar histórico
# ---------------------------------------------------------------------------

def load_ga_history(json_path: str) -> dict:
    """
    Carrega o histórico de um run do GA a partir do JSON salvo.

    Args:
        json_path: Caminho para o arquivo ga_history.json.

    Returns:
        Dict com as chaves: params, stopped_at, reason, best_individual,
        generations_stats.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        json.JSONDecodeError: Se o arquivo não for JSON válido.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Histórico GA não encontrado: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "Histórico GA carregado: %d gerações, motivo de parada='%s'",
        len(data.get("generations_stats", [])),
        data.get("reason", "?"),
    )
    return data
