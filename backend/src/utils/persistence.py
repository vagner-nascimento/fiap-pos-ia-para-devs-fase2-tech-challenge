"""
Módulo para persistência de dados e modelos.

Este módulo é responsável por:
- Salvar e carregar DataFrames em diferentes formatos
- Salvar e carregar modelos treinados
- Gerenciar versões de arquivos
"""

import pandas as pd
import pickle
import joblib
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = "csv",
    **kwargs,
) -> None:
    """
    Salva um DataFrame em arquivo.

    Args:
        df (pd.DataFrame): DataFrame a salvar.
        path (str): Caminho para salvar o arquivo.
        format (str): Formato de arquivo ('csv', 'parquet', 'excel'). Default: 'csv'.
        **kwargs: Argumentos adicionais para o método de salvamento.

    Raises:
        ValueError: Se formato não suportado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Salvando DataFrame em {format}: {path}")

    try:
        if format == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif format == "parquet":
            df.to_parquet(path, index=False, **kwargs)
        elif format == "excel":
            df.to_excel(path, index=False, **kwargs)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info(f"DataFrame salvo com sucesso. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Erro ao salvar DataFrame: {e}")
        raise


def load_dataframe(
    path: str,
    format: str = "csv",
    **kwargs,
) -> pd.DataFrame:
    """
    Carrega um DataFrame de arquivo.

    Args:
        path (str): Caminho do arquivo.
        format (str): Formato de arquivo ('csv', 'parquet', 'excel'). Default: 'csv'.
        **kwargs: Argumentos adicionais para o método de carregamento.

    Returns:
        pd.DataFrame: DataFrame carregado.

    Raises:
        FileNotFoundError: Se arquivo não existir.
        ValueError: Se formato não suportado.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    logger.info(f"Carregando DataFrame em {format}: {path}")

    try:
        if format == "csv":
            df = pd.read_csv(path, **kwargs)
        elif format == "parquet":
            df = pd.read_parquet(path, **kwargs)
        elif format == "excel":
            df = pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info(f"DataFrame carregado com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar DataFrame: {e}")
        raise


def save_model(
    model: Any,
    path: str,
    format: str = "joblib",
) -> None:
    """
    Salva um modelo treinado.

    Args:
        model (Any): Modelo a salvar.
        path (str): Caminho para salvar o modelo.
        format (str): Formato ('joblib' ou 'pickle'). Default: 'joblib'.

    Raises:
        ValueError: Se formato não suportado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Salvando modelo em {format}: {path}")

    try:
        if format == "joblib":
            joblib.dump(model, path)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info(f"Modelo salvo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        raise


def load_model(
    path: str,
    format: str = "joblib",
) -> Any:
    """
    Carrega um modelo treinado.

    Args:
        path (str): Caminho do modelo.
        format (str): Formato ('joblib' ou 'pickle'). Default: 'joblib'.

    Returns:
        Any: Modelo carregado.

    Raises:
        FileNotFoundError: Se arquivo não existir.
        ValueError: Se formato não suportado.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")

    logger.info(f"Carregando modelo em {format}: {path}")

    try:
        if format == "joblib":
            model = joblib.load(path)
        elif format == "pickle":
            with open(path, "rb") as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info(f"Modelo carregado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise


def save_dict(
    data: dict,
    path: str,
    format: str = "joblib",
) -> None:
    """
    Salva um dicionário (ex: encoders, parâmetros).

    Args:
        data (dict): Dicionário a salvar.
        path (str): Caminho para salvar.
        format (str): Formato ('joblib' ou 'pickle'). Default: 'joblib'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Salvando dicionário: {path}")

    try:
        if format == "joblib":
            joblib.dump(data, path)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info("Dicionário salvo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar dicionário: {e}")
        raise


def load_dict(
    path: str,
    format: str = "joblib",
) -> dict:
    """
    Carrega um dicionário.

    Args:
        path (str): Caminho do dicionário.
        format (str): Formato ('joblib' ou 'pickle'). Default: 'joblib'.

    Returns:
        dict: Dicionário carregado.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    logger.info(f"Carregando dicionário: {path}")

    try:
        if format == "joblib":
            data = joblib.load(path)
        elif format == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info("Dicionário carregado com sucesso")
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar dicionário: {e}")
        raise
