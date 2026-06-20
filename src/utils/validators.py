"""
Módulo para validação de dados.

Este módulo fornece funções para validar dados de entrada e estrutura.
"""

import pandas as pd
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def validate_columns_exist(
    df: pd.DataFrame,
    required_columns: List[str],
) -> Tuple[bool, List[str]]:
    """
    Valida se as colunas obrigatórias existem no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        required_columns (List[str]): Lista de colunas obrigatórias.

    Returns:
        Tuple[bool, List[str]]: (validação passou, colunas faltando)
    """
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        logger.error(f"Colunas obrigatórias faltando: {missing_columns}")
        return False, list(missing_columns)

    logger.info(f"Validação de colunas passou: {len(required_columns)} colunas encontradas")
    return True, []


def validate_no_missing_values(
    df: pd.DataFrame,
    subset: List[str] = None,
    threshold: float = 0.0,
) -> Tuple[bool, Dict]:
    """
    Valida valores ausentes no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        subset (List[str], optional): Colunas a verificar. Se None, valida todas.
        threshold (float): Porcentagem máxima de valores ausentes permitida (0-1).

    Returns:
        Tuple[bool, Dict]: (validação passou, detalhes de valores ausentes)
    """
    if subset is None:
        subset = df.columns.tolist()

    missing_info = {}
    validation_passed = True

    for col in subset:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)

            missing_info[col] = {
                "count": missing_count,
                "percentage": missing_pct,
            }

            if missing_pct > threshold:
                logger.warning(
                    f"Coluna '{col}' tem {missing_pct:.2%} de valores ausentes "
                    f"(limite: {threshold:.2%})"
                )
                validation_passed = False

    return validation_passed, missing_info


def validate_data_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str],
) -> Tuple[bool, Dict]:
    """
    Valida os tipos de dados das colunas.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        expected_types (Dict[str, str]): Mapeamento de coluna para tipo esperado.

    Returns:
        Tuple[bool, Dict]: (validação passou, detalhes de tipos)
    """
    type_info = {}
    validation_passed = True

    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            matches = expected_type in actual_type.lower()

            type_info[col] = {
                "expected": expected_type,
                "actual": actual_type,
                "matches": matches,
            }

            if not matches:
                logger.warning(
                    f"Coluna '{col}' esperada '{expected_type}', "
                    f"encontrada '{actual_type}'"
                )
                validation_passed = False

    return validation_passed, type_info


def validate_value_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]],
) -> Tuple[bool, Dict]:
    """
    Valida se valores estão dentro de ranges esperados.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        ranges (Dict[str, Tuple[float, float]]): Mapeamento de coluna para (min, max).

    Returns:
        Tuple[bool, Dict]: (validação passou, detalhes de ranges)
    """
    range_info = {}
    validation_passed = True

    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()

            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            out_of_range_count = out_of_range.sum()

            range_info[col] = {
                "min_expected": min_val,
                "max_expected": max_val,
                "min_actual": col_min,
                "max_actual": col_max,
                "out_of_range_count": out_of_range_count,
            }

            if out_of_range_count > 0:
                logger.warning(
                    f"Coluna '{col}' tem {out_of_range_count} valores fora do range "
                    f"esperado [{min_val}, {max_val}]"
                )
                validation_passed = False

    return validation_passed, range_info


def validate_nutritional_data(df: pd.DataFrame) -> Dict[str, Tuple[bool, any]]:
    """
    Executa validações específicas para dados nutricionais.

    Args:
        df (pd.DataFrame): DataFrame de dados nutricionais.

    Returns:
        Dict[str, Tuple[bool, any]]: Resultados de validação.
    """
    logger.info("Executando validações específicas para dados nutricionais...")

    results = {}

    # Validar colunas obrigatórias
    required_cols = ["NU_IDADE_ANO", "NU_PESO", "NU_ALTURA", "DS_IMC", "ESTADO_NUTRI"]
    cols_valid, missing_cols = validate_columns_exist(df, required_cols)
    results["columns"] = (cols_valid, missing_cols)

    # Validar valores ausentes
    missing_valid, missing_info = validate_no_missing_values(df, subset=required_cols, threshold=0.01)
    results["missing_values"] = (missing_valid, missing_info)

    # Validar tipos de dados
    expected_types = {
        "NU_IDADE_ANO": "int",
        "NU_PESO": "float",
        "NU_ALTURA": "float",
        "DS_IMC": "float",
        "ESTADO_NUTRI": "object",
    }
    types_valid, type_info = validate_data_types(df, expected_types)
    results["data_types"] = (types_valid, type_info)

    # Validar ranges
    ranges = {
        "NU_IDADE_ANO": (0, 120),
        "NU_PESO": (1, 300),
        "NU_ALTURA": (50, 220),
        "DS_IMC": (10, 100),
    }
    ranges_valid, range_info = validate_value_ranges(df, ranges)
    results["ranges"] = (ranges_valid, range_info)

    # Validar categorias de ESTADO_NUTRI
    valid_categories = {
        "Baixo peso",
        "Eutrofia",
        "Risco/Sobrepeso",
        "Obesidade",
        "Obesidade Grave",
    }
    actual_categories = set(df["ESTADO_NUTRI"].unique())
    invalid_categories = actual_categories - valid_categories

    if invalid_categories:
        logger.warning(f"Categorias inválidas encontradas: {invalid_categories}")
        results["categories"] = (False, invalid_categories)
    else:
        logger.info(f"Todas as categorias estão válidas")
        results["categories"] = (True, valid_categories)

    return results
