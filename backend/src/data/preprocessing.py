"""
Módulo para pré-processamento de dados de estado nutricional.

Responsável por:
- Remoção de registros de gestantes
- Remoção de colunas irrelevantes
- Consolidação de targets por faixa etária
- Padronização de categorias de estado nutricional
- Conversão de tipos numéricos
- Remoção de coluna de raça/cor
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Colunas que serão removidas por serem irrelevantes ou redundantes
COLUMNS_TO_DROP = [
    "ST_PARTICIPA_ANDI",      # Participação no programa ANDI
    "CO_POVO_COMUNIDADE",     # Majoritariamente nulo
    "DS_IMC_PRE_GESTACIONAL", # Todos nulos
    "CO_ESCOLARIDADE",        # Majoritariamente nulo
    "DS_ESCOLARIDADE",        # Maioria SEM INFORMAÇÃO
    "CO_ACOMPANHAMENTO",      # Código técnico irrelevante
    "CO_PESSOA_SISVAN",       # Código de identificação
    "CO_MUNICIPIO_IBGE",      # Valor constante (SÃO PAULO)
    "SG_UF",                  # Valor constante (SÃO PAULO)
    "NO_MUNICIPIO",           # Valor constante (SÃO PAULO)
    "CO_CNES",                # Código de estabelecimento
    "DS_POVO_COMUNIDADE",     # Maioria NÃO INFORMADO
    "NU_COMPETENCIA",         # Referência temporal irrelevante
    "CO_SISTEMA_ORIGEM_ACOMP",# Código de sistema
    "SISTEMA_ORIGEM_ACOMP",   # Informação de origem irrelevante
    "DT_ACOMPANHAMENTO",      # Data irrelevante
    "PESO X IDADE",           # Específico para 0-10 anos
    "CRI. ALTURA X IDADE",    # Específico para 0-10 anos
    "ADO. ALTURA X IDADE",    # Específico para 10-20 anos
    "PESO X ALTURA",          # Informação redundante
    "CO_ESTADO_NUTRI_IMC_SEMGEST",  # Gestantes
    "NU_FASE_VIDA",           # Código da fase (usamos DS_FASE_VIDA)
    "CO_RACA_COR",            # Código da raça (removemos também DS_RACA_COR depois)
]

# Colunas com targets por faixa etária
TARGET_COLUMNS = [
    "CRI. IMC X IDADE",            # Crianças 0-10 anos
    "ADO. IMC X IDADE",            # Adolescentes 10-20 anos
    "CO_ESTADO_NUTRI_ADULTO",      # Adultos 20-60 anos
    "CO_ESTADO_NUTRI_IDOSO",       # Idosos 60+ anos
]


def remove_pregnant_women(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros de gestantes baseado na coluna CO_ESTADO_NUTRI_IMC_SEMGEST.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos.

    Returns:
        pd.DataFrame: DataFrame sem registros de gestantes.
    """
    logger.info("Removendo registros de gestantes...")
    
    initial_count = len(df)
    df_clean = df[df["CO_ESTADO_NUTRI_IMC_SEMGEST"].isna()].copy()
    removed_count = initial_count - len(df_clean)
    
    logger.info(f"Removidos {removed_count} registros de gestantes")
    return df_clean


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas irrelevantes para o modelo.

    Args:
        df (pd.DataFrame): DataFrame com colunas a remover.

    Returns:
        pd.DataFrame: DataFrame sem colunas irrelevantes.
    """
    logger.info("Removendo colunas irrelevantes...")
    
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop).copy()
    
    logger.info(f"Removidas {len(cols_to_drop)} colunas irrelevantes")
    return df_clean


def consolidate_nutritional_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida targets de 4 colunas (por faixa etária) em uma única coluna.

    Args:
        df (pd.DataFrame): DataFrame com targets dispersos.

    Returns:
        pd.DataFrame: DataFrame com target consolidado.
    """
    logger.info("Consolidando status nutricional...")
    
    df_clean = df.copy()
    
    # Usar bfill para obter o primeiro valor não-nulo da esquerda para direita
    df_clean['ESTADO_NUTRI'] = df_clean[TARGET_COLUMNS].bfill(axis=1).iloc[:, 0]
    
    logger.info(f"Status consolidado. Registros com valor: {df_clean['ESTADO_NUTRI'].notna().sum()}")
    return df_clean


def remove_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros sem valor no target consolidado.

    Args:
        df (pd.DataFrame): DataFrame com target consolidado.

    Returns:
        pd.DataFrame: DataFrame sem registros sem target.
    """
    logger.info("Removendo registros sem target...")
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['ESTADO_NUTRI']).copy()
    removed_count = initial_count - len(df_clean)
    
    logger.info(f"Removidos {removed_count} registros sem target")
    return df_clean


def drop_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove as colunas de target por faixa etária após consolidação.

    Args:
        df (pd.DataFrame): DataFrame com targets consolidados.

    Returns:
        pd.DataFrame: DataFrame sem colunas de target dispersas.
    """
    logger.info("Removendo colunas de target por faixa etária...")
    
    cols_to_drop = [col for col in TARGET_COLUMNS if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop).copy()
    
    return df_clean


def standardize_nutritional_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza categorias de estado nutricional para nomenclatura unificada.

    Args:
        df (pd.DataFrame): DataFrame com estado nutricional em diferentes formatos.

    Returns:
        pd.DataFrame: DataFrame com estado nutricional padronizado.
    """
    logger.info("Padronizando categorias de estado nutricional...")
    
    df_clean = df.copy()
    
    # Aplicar replacements para unificar nomenclatura
    df_clean["ESTADO_NUTRI"] = (
        df_clean["ESTADO_NUTRI"]
        .str.strip()
        .str.replace(r"Obesidade Grau III", "Obesidade Grave")
        .str.replace(r"Obesidade Grau II", "Obesidade Grave")
        .str.replace(r"Obesidade Grau I", "Obesidade")
        .str.replace(r"Magreza acentuada", "Baixo peso")
        .str.replace(r"Magreza", "Baixo peso")
        .str.replace(r"Adequado ou Eutrófico", "Eutrofia")
        .str.replace(r"Adequado ou eutrófico", "Eutrofia")
        .str.replace(r"Sobrepeso", "Risco/Sobrepeso")
        .str.replace(r"Risco de sobrepeso", "Risco/Sobrepeso")
    )
    
    logger.info(f"Categorias padronizadas. Categorias únicas: {df_clean['ESTADO_NUTRI'].nunique()}")
    return df_clean


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas de string (com vírgulas) para float.

    Args:
        df (pd.DataFrame): DataFrame com colunas numéricas em formato string.

    Returns:
        pd.DataFrame: DataFrame com colunas numéricas em formato float.
    """
    logger.info("Convertendo colunas numéricas...")
    
    df_clean = df.copy()
    
    numeric_columns = ["NU_PESO", "NU_ALTURA", "DS_IMC"]
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".")
                .astype(float)
            )
    
    logger.info(f"Colunas numéricas convertidas: {numeric_columns}")
    return df_clean


def remove_race_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove coluna DS_RACA_COR por ter baixa relevância no modelo.

    Args:
        df (pd.DataFrame): DataFrame com coluna DS_RACA_COR.

    Returns:
        pd.DataFrame: DataFrame sem coluna DS_RACA_COR.
    """
    logger.info("Removendo coluna DS_RACA_COR...")
    
    df_clean = df.copy()
    if "DS_RACA_COR" in df_clean.columns:
        df_clean = df_clean.drop(columns=["DS_RACA_COR"])
        logger.info("Coluna DS_RACA_COR removida")
    else:
        logger.warning("Coluna DS_RACA_COR não encontrada")
    
    return df_clean


def run_preprocessing(
    df: pd.DataFrame,
    remove_pregnant: bool = True,
    drop_cols: bool = True,
    consolidate_target: bool = True,
    remove_missing: bool = True,
    standardize: bool = True,
    convert_numeric: bool = True,
    remove_race: bool = True,
) -> pd.DataFrame:
    """
    Orquestra todo o pipeline de pré-processamento.

    Args:
        df (pd.DataFrame): DataFrame bruto de entrada.
        remove_pregnant (bool): Remover gestantes. Default: True.
        drop_cols (bool): Remover colunas irrelevantes. Default: True.
        consolidate_target (bool): Consolidar target. Default: True.
        remove_missing (bool): Remover registros sem target. Default: True.
        standardize (bool): Padronizar categorias. Default: True.
        convert_numeric (bool): Converter colunas numéricas. Default: True.
        remove_race (bool): Remover coluna de raça. Default: True.

    Returns:
        pd.DataFrame: DataFrame completamente pré-processado.
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
    logger.info("=" * 80)
    logger.info(f"Registros iniciais: {len(df)}")

    df_clean = df.copy()

    if remove_pregnant:
        df_clean = remove_pregnant_women(df_clean)

    if drop_cols:
        df_clean = drop_irrelevant_columns(df_clean)

    if consolidate_target:
        df_clean = consolidate_nutritional_status(df_clean)

    if remove_missing:
        df_clean = remove_missing_target(df_clean)

    # Remover colunas de target por faixa etária
    df_clean = drop_target_columns(df_clean)

    if standardize:
        df_clean = standardize_nutritional_categories(df_clean)

    if convert_numeric:
        df_clean = convert_numeric_columns(df_clean)

    if remove_race:
        df_clean = remove_race_column(df_clean)

    logger.info("=" * 80)
    logger.info(f"PRÉ-PROCESSAMENTO CONCLUÍDO")
    logger.info(f"Registros finais: {len(df_clean)}")
    logger.info(f"Colunas finais: {df_clean.columns.tolist()}")
    logger.info("=" * 80)

    return df_clean
