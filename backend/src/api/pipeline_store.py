"""Store in-memory para estado de execução do pipeline."""

from __future__ import annotations

import threading
from typing import Any

_pipeline_state: dict[str, Any] = {
    "preprocessing_completed": False,
    "tuning_completed": False,
    "predictions_completed": False,
}
_lock = threading.Lock()


def reset_pipeline() -> None:
    """Reseta o estado do pipeline (chamado quando preprocessing inicia)."""
    with _lock:
        _pipeline_state["preprocessing_completed"] = False
        _pipeline_state["tuning_completed"] = False
        _pipeline_state["predictions_completed"] = False


def set_preprocessing_completed() -> None:
    """Marca preprocessing como concluído."""
    with _lock:
        _pipeline_state["preprocessing_completed"] = True


def set_tuning_completed() -> None:
    """Marca tuning como concluído."""
    with _lock:
        _pipeline_state["tuning_completed"] = True


def set_predictions_completed() -> None:
    """Marca predictions como concluído."""
    with _lock:
        _pipeline_state["predictions_completed"] = True


def get_pipeline_state() -> dict[str, Any]:
    """Retorna o estado atual do pipeline."""
    with _lock:
        return dict(_pipeline_state)


def check_preprocessing_completed() -> bool:
    """Verifica se preprocessing foi concluído."""
    with _lock:
        return _pipeline_state["preprocessing_completed"]


def check_tuning_completed() -> bool:
    """Verifica se tuning foi concluído."""
    with _lock:
        return _pipeline_state["tuning_completed"]


def check_predictions_completed() -> bool:
    """Verifica se predictions foi concluído."""
    with _lock:
        return _pipeline_state["predictions_completed"]
