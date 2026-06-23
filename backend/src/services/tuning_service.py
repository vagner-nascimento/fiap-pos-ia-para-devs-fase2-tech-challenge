"""Serviço de tuning genético — carregamento de dados e execução do GA."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.models.ga_persistence import _individuo_to_dict, load_ga_history, save_ga_results
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

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y


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
) -> dict:
    """Executa o GA Co-Evolutivo e opcionalmente persiste logs."""
    X, y = load_training_data(dataset, target_col)

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
    )
    results = ga.run()

    if save_logs:
        save_ga_results(results, str(LOG_PATH))

    return serialize_ga_results(results)


def get_latest_logs() -> dict:
    """Retorna o histórico GA mais recente (JSON) e stats CSV."""
    json_path = LOG_PATH / "ga_history.json"
    csv_path = LOG_PATH / "ga_generation_stats.csv"

    if not json_path.exists():
        raise FileNotFoundError("Nenhum histórico GA encontrado.")

    history = load_ga_history(str(json_path))
    csv_content = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""

    return {"history": history, "stats_csv": csv_content}
