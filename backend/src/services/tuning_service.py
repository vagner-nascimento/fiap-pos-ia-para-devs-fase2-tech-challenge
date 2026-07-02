"""Serviço de tuning genético — carregamento de dados e execução do GA."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.models.ga_persistence import _individuo_to_dict, load_ga_history, save_ga_results
from src.models.ga_snapshot import cleanup_snapshot_file, read_generation_snapshots
from src.models.genetic_algorithm import GeneticAlgorithm

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
LOG_PATH = Path(os.getenv("LOG_PATH", "./models/logs"))
PROCESSED_DIR = DATA_PATH / "processed"


def list_datasets() -> list[str]:
    """Lista arquivos CSV disponíveis em data/processed/."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted(p.name for p in PROCESSED_DIR.glob("*.csv"))


def resolve_dataset_path(dataset: str) -> Path:
    """Resolve caminho do dataset (nome ou path relativo)."""
    path = Path(dataset)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return PROCESSED_DIR / dataset


def load_training_data(dataset: str, target_col: str) -> tuple:
    """Carrega X, y a partir de um CSV processado."""
    path = resolve_dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna `{target_col}` não encontrada. Colunas: {list(df.columns)}"
        )

    y = df[target_col].values

    # Seleciona apenas colunas numéricas para evitar erros no StandardScaler/KNN
    feature_df = df.drop(columns=[target_col])
    numeric_df = feature_df.select_dtypes(include="number")

    dropped = set(feature_df.columns) - set(numeric_df.columns)
    if dropped:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "load_training_data: colunas não-numéricas ignoradas: %s", sorted(dropped)
        )

    X = numeric_df.values
    return X, y


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_seed: int = 42,
) -> tuple:
    """
    Retorna um subconjunto estratificado de (X, y) com no máximo `n_samples` linhas.

    Preserva a distribuição de classes do target via train_test_split(stratify=y).
    Se `n_samples >= len(y)`, retorna o dataset completo sem amostragem.
    """
    logger = logging.getLogger(__name__)
    total = len(y)
    if n_samples >= total:
        logger.info("sample_size=%d >= total=%d: usando dataset completo.", n_samples, total)
        return X, y

    sample_ratio = n_samples / total
    _, X_s, _, y_s = train_test_split(
        X, y,
        test_size=sample_ratio,
        stratify=y,
        random_state=random_seed,
    )
    logger.info(
        "Amostragem estratificada: %d → %d linhas (%.1f%%)",
        total, len(y_s), 100 * sample_ratio,
    )
    return X_s, y_s


def serialize_ga_results(results: dict) -> dict:
    """Converte resultado do GA para dict JSON-serializável."""
    return {
        "generations_stats": results.get("generations_stats", []),
        "best_individual": _individuo_to_dict(results.get("best_individual")),
        "stopped_at": results.get("stopped_at"),
        "reason": results.get("reason"),
        "params": results.get("params", {}),
    }


def run_tuning(
    *,
    dataset: str,
    target_col: str = "TARGET",
    pop_size: int = 20,
    max_generations: int = 10,
    patience: int = 5,
    k_folds: int = 5,
    aggressiveness: str = "medium",
    elitism: bool = True,
    cxpb: float = 0.7,
    mutpb: float = 0.3,
    indpb: float = 0.5,
    random_seed: int = 42,
    save_logs: bool = True,
    job_id: str | None = None,
    sample_size: int = 50_000,
) -> dict:
    """Executa o GA Co-Evolutivo e opcionalmente persiste logs.

    Args:
        sample_size: Número máximo de linhas para treino (amostragem estratificada).
                     Use 0 para dataset completo (muito mais lento).
    """
    X, y = load_training_data(dataset, target_col)

    if sample_size > 0:
        X, y = stratified_sample(X, y, n_samples=sample_size, random_seed=random_seed)

    # Limpa snapshots de run anterior para este job_id
    if job_id is not None:
        cleanup_snapshot_file(job_id)

    ga = GeneticAlgorithm(
        X=X,
        y=y,
        pop_size=pop_size,
        max_generations=max_generations,
        patience=patience,
        k_folds=k_folds,
        mutation_aggressiveness=aggressiveness,
        elitism=elitism,
        indpb=indpb,
        cxpb=cxpb,
        mutpb=mutpb,
        random_seed=random_seed,
        job_id=job_id,
    )
    results = ga.run()

    if save_logs:
        save_ga_results(results, str(LOG_PATH))

    return serialize_ga_results(results)


def get_generation_snapshots(job_id: str, since_generation: int = 0) -> list[dict]:
    """
    Retorna snapshots de gerações posteriores a `since_generation`.

    Args:
        job_id: Identificador único do job de tuning.
        since_generation: Cursor — apenas gerações acima deste valor são retornadas.

    Returns:
        Lista de dicts de snapshot ordenada por geração.
    """
    return read_generation_snapshots(job_id, since_generation)


def get_latest_logs() -> dict:
    """Retorna o histórico GA mais recente (JSON) e stats CSV."""
    json_path = LOG_PATH / "ga_history.json"
    csv_path = LOG_PATH / "ga_generation_stats.csv"

    if not json_path.exists():
        raise FileNotFoundError("Nenhum histórico GA encontrado.")

    history = load_ga_history(str(json_path))
    csv_content = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""

    return {"history": history, "stats_csv": csv_content}
