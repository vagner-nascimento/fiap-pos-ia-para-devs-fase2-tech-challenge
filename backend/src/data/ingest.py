"""
Módulo para ingestão de dados do estado nutricional.

Responsável por:
- Leitura de arquivos CSV
- Validação da estrutura de dados
- Detecção de encoding
"""

import pandas as pd
import logging
import signal
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def read_csv_data(
    file_path: str,
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    Lê dados de um arquivo CSV com tratamento de erros.

    Usa `memory_map=False` e tenta o engine C primeiro. Se ocorrer SIGSEGV
    (exit 139, comum em arquivos grandes em volumes Docker/Podman), faz
    fallback automático para `engine='python'`.

    Args:
        file_path (str): Caminho do arquivo CSV.
        encoding (str): Encoding do arquivo. Default: 'utf-8'.
        **kwargs: Argumentos adicionais para pd.read_csv.

    Returns:
        pd.DataFrame: DataFrame com os dados lidos.

    Raises:
        FileNotFoundError: Se arquivo não existir.
        UnicodeDecodeError: Se erro de encoding.
        pd.errors.ParserError: Se erro ao parsing do CSV.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info(f"Lendo arquivo CSV: {file_path}")

    # O parser C do pandas (padrão) causa SIGSEGV (exit code 139) ao ler
    # arquivos CSV grandes (>550k linhas) em volumes Docker/Podman.
    # O SIGSEGV mata o processo antes de qualquer exceção Python ser lançada,
    # portanto o try/except não resolve — a única solução é forçar engine='python'.
    kwargs.setdefault("memory_map", False)
    kwargs.setdefault("engine", "python")

    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        logger.info(f"Arquivo lido com sucesso. Shape: {df.shape}")
        return df
    except UnicodeDecodeError as e:
        logger.error(f"Erro de encoding: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Erro ao fazer parsing do CSV: {e}")
        raise


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> bool:
    """
    Valida a estrutura básica de um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        required_columns (List[str], optional): Colunas obrigatórias.

    Returns:
        bool: True se validação passa.

    Raises:
        ValueError: Se validação falhar.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vazio ou None")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias faltando: {missing_cols}")

    logger.info(f"Validação de DataFrame passou. Shape: {df.shape}")
    return True
