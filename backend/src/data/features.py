"""
Módulo para engenharia de features do estado nutricional.

Responsável por:
- Codificação de variáveis categóricas
- Criação de target numérico
- Cálculo do percentual de gordura corporal (Deurenberg)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Mapeamento de estado nutricional para valores numéricos
NUTRITIONAL_STATE_MAP = {
    "Baixo peso": 0,
    "Eutrofia": 1,
    "Risco/Sobrepeso": 2,
    "Obesidade": 3,
    "Obesidade Grave": 4,
}


def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_cols: list = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Codifica variáveis categóricas usando LabelEncoder.

    Args:
        df (pd.DataFrame): DataFrame com variáveis categóricas.
        categorical_cols (list, optional): Colunas a codificar.
                                          Default: ["DS_FASE_VIDA", "SG_SEXO"]

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame codificado e dicionário de encoders.
    """
    if categorical_cols is None:
        categorical_cols = ["DS_FASE_VIDA", "SG_SEXO"]

    logger.info(f"Codificando variáveis categóricas: {categorical_cols}")

    df_encoded = df.copy()
    encoders = {}

    for col in categorical_cols:
        if col in df_encoded.columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
            encoders[col] = encoder

    logger.info(f"Variáveis categóricas codificadas. Encoders salvos: {list(encoders.keys())}")
    return df_encoded, encoders


def create_numeric_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna TARGET com mapeamento numérico de ESTADO_NUTRI.

    Args:
        df (pd.DataFrame): DataFrame com coluna ESTADO_NUTRI.

    Returns:
        pd.DataFrame: DataFrame com coluna TARGET numérica.
    """
    logger.info("Criando target numérico...")

    df_clean = df.copy()
    df_clean["TARGET"] = df_clean["ESTADO_NUTRI"].map(NUTRITIONAL_STATE_MAP)

    if df_clean["TARGET"].isna().sum() > 0:
        logger.warning(f"Valores não mapeados no target: {df_clean['TARGET'].isna().sum()}")

    logger.info("Target numérico criado")
    return df_clean


def calculate_body_fat_percentage(
    df: pd.DataFrame,
    age_col: str = "NU_IDADE_ANO",
    imc_col: str = "DS_IMC",
    sex_col: str = "SG_SEXO",
) -> pd.DataFrame:
    """
    Calcula percentual de gordura corporal usando fórmula de Deurenberg.

    Fórmula por faixa etária:
    - Crianças (≤15 anos): PERC_GORDURA = 1.20*IMC + 0.23*idade - 10.8*sexo - 5.4
    - Adultos (>15 anos):  PERC_GORDURA = 1.51*IMC - 0.70*idade - 3.6*sexo + 1.4

    Onde: sexo = 0 (feminino), 1 (masculino)

    Ref: DEURENBERG, P.; WESTSTRATE, J. A.; SEIDELL, J. C.
    British Journal of Nutrition, v. 65, n. 2, p. 105–114, mar. 1991.

    Args:
        df (pd.DataFrame): DataFrame com dados necessários.
        age_col (str): Nome da coluna de idade.
        imc_col (str): Nome da coluna de IMC.
        sex_col (str): Nome da coluna de sexo (0=F, 1=M).

    Returns:
        pd.DataFrame: DataFrame com coluna PERC_GORDURA adicionada.
    """
    logger.info("Calculando percentual de gordura (Deurenberg)...")

    df_clean = df.copy()

    # Crianças (idade <= 15 anos)
    children_mask = df_clean[age_col] <= 15
    df_clean.loc[children_mask, "PERC_GORDURA"] = (
        1.20 * df_clean.loc[children_mask, imc_col]
        + 0.23 * df_clean.loc[children_mask, age_col]
        - 10.8 * df_clean.loc[children_mask, sex_col]
        - 5.4
    )

    # Adultos (idade > 15 anos)
    adults_mask = df_clean[age_col] > 15
    df_clean.loc[adults_mask, "PERC_GORDURA"] = (
        1.51 * df_clean.loc[adults_mask, imc_col]
        - 0.70 * df_clean.loc[adults_mask, age_col]
        - 3.6 * df_clean.loc[adults_mask, sex_col]
        + 1.4
    )

    # Arredondar para 2 casas decimais
    df_clean["PERC_GORDURA"] = df_clean["PERC_GORDURA"].round(2)

    logger.info(
        f"Percentual de gordura calculado. "
        f"Crianças: {children_mask.sum()}, Adultos: {adults_mask.sum()}"
    )

    return df_clean

def run_feature_engineering(
    df: pd.DataFrame,
    create_target: bool = True,
    encode_cats: bool = True,
    calculate_fat: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Orquestra todo o pipeline de engenharia de features.

    Args:
        df (pd.DataFrame): DataFrame pré-processado.
        create_target (bool): Criar target numérico. Default: True.
        encode_cats (bool): Codificar categóricas. Default: True.
        calculate_fat (bool): Calcular % gordura. Default: True.

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame com features e dicionário de encoders.
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE ENGENHARIA DE FEATURES")
    logger.info("=" * 80)

    df_features = df.copy()
    encoders = {}

    if create_target:
        df_features = create_numeric_target(df_features)

    if encode_cats:
        df_features, encoders = encode_categorical_variables(df_features)

    if calculate_fat:
        df_features = calculate_body_fat_percentage(df_features)

    logger.info("=" * 80)
    logger.info("ENGENHARIA DE FEATURES CONCLUÍDA")
    logger.info(f"Features finais: {df_features.columns.tolist()}")
    logger.info("=" * 80)

    return df_features, encoders
