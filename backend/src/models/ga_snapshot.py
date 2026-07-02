"""
Persistência incremental de snapshots por geração do Algoritmo Genético.

Usa um arquivo JSONL em /tmp (interno ao backend).
O frontend acessa os dados exclusivamente via endpoint REST.

Opção C do plano: desenvolvimento/prototipagem.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TMP_DIR = Path(os.getenv("GA_SNAPSHOT_DIR", "/tmp"))


def _snapshot_path(job_id: str) -> Path:
    return _TMP_DIR / f"ag_job_{job_id}_generations.jsonl"


def save_generation_snapshot(job_id: str, snapshot: dict) -> None:
    """
    Persiste um snapshot de geração em JSONL.

    Cada chamada adiciona uma linha ao arquivo.
    Seguro para ser chamado de uma thread de background.

    Args:
        job_id: Identificador único do job de tuning.
        snapshot: Dict com os dados da geração (deve ser JSON-serializável).
    """
    path = _snapshot_path(job_id)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")
        logger.debug("Snapshot gen=%s persistido em %s", snapshot.get("generation"), path)
    except OSError as exc:
        logger.warning("Falha ao persistir snapshot (job=%s): %s", job_id, exc)


def cleanup_snapshot_file(job_id: str) -> None:
    """
    Remove o arquivo JSONL de um job anterior.

    Deve ser chamado ao iniciar um novo job para evitar leitura de dados antigos.

    Args:
        job_id: Identificador único do job de tuning.
    """
    path = _snapshot_path(job_id)
    try:
        path.unlink(missing_ok=True)
        logger.debug("Arquivo de snapshots removido: %s", path)
    except OSError as exc:
        logger.warning("Falha ao remover arquivo de snapshots (job=%s): %s", job_id, exc)


def read_generation_snapshots(job_id: str, since_generation: int = 0) -> list[dict]:
    """
    Lê snapshots de gerações posteriores a `since_generation`.

    Implementa leitura incremental: apenas as linhas com
    generation > since_generation são retornadas.

    Args:
        job_id: Identificador único do job de tuning.
        since_generation: Cursor — apenas gerações acima deste valor são retornadas.
                          Use 0 para retornar todas as gerações.

    Returns:
        Lista de dicts de snapshot, ordenada por geração crescente.
        Retorna lista vazia se o arquivo não existir ou não houver novas gerações.
    """
    path = _snapshot_path(job_id)
    if not path.exists():
        return []

    snapshots: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    if s.get("generation", 0) > since_generation:
                        snapshots.append(s)
                except json.JSONDecodeError as exc:
                    logger.warning("Linha inválida no JSONL (job=%s): %s", job_id, exc)
    except OSError as exc:
        logger.warning("Falha ao ler snapshots (job=%s): %s", job_id, exc)

    return snapshots
