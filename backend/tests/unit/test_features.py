"""Testes unitários para o módulo de engenharia de features."""

import pandas as pd
import numpy as np
import pytest
from src.data.features import (
    encode_categorical_variables,
    create_numeric_target,
    calculate_body_fat_percentage,
    run_feature_engineering,
    NUTRITIONAL_STATE_MAP,
)


class TestEncodeCategoricalVariables:
    def test_encode_default_columns(self):
        """Testa codificação com colunas padrão."""
        df = pd.DataFrame({
            "DS_FASE_VIDA": ["Adulto", "Idoso", "Adulto"],
            "SG_SEXO": ["Masculino", "Feminino", "Masculino"],
            "NU_PESO": [70.0, 65.0, 80.0]
        })
        
        df_encoded, encoders = encode_categorical_variables(df)
        
        assert "DS_FASE_VIDA" in encoders
        assert "SG_SEXO" in encoders
        assert df_encoded["DS_FASE_VIDA"].dtype in [np.int32, np.int64]
        assert df_encoded["SG_SEXO"].dtype in [np.int32, np.int64]

    def test_encode_custom_columns(self):
        """Testa codificação com colunas customizadas."""
        df = pd.DataFrame({
            "TEST_COL": ["A", "B", "A"],
            "NU_PESO": [70.0, 65.0, 80.0]
        })
        
        df_encoded, encoders = encode_categorical_variables(df, categorical_cols=["TEST_COL"])
        
        assert "TEST_COL" in encoders
        assert df_encoded["TEST_COL"].dtype in [np.int32, np.int64]

    def test_encode_missing_column(self):
        """Testa comportamento quando coluna não existe."""
        df = pd.DataFrame({"NU_PESO": [70.0, 65.0, 80.0]})
        
        df_encoded, encoders = encode_categorical_variables(df, categorical_cols=["NON_EXISTENT"])
        
        assert len(encoders) == 0
        assert df_encoded.shape == df.shape

    def test_encode_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "DS_FASE_VIDA": ["Adulto", "Idoso"],
            "SG_SEXO": ["Masculino", "Feminino"]
        })
        original_values = df["DS_FASE_VIDA"].copy()
        
        encode_categorical_variables(df)
        
        pd.testing.assert_series_equal(df["DS_FASE_VIDA"], original_values)


class TestCreateNumericTarget:
    def test_create_target_valid_mapping(self):
        """Testa criação de target com mapeamento válido."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", "Obesidade", "Baixo peso"],
            "NU_PESO": [70.0, 90.0, 50.0]
        })
        
        df_result = create_numeric_target(df)
        
        assert "TARGET" in df_result.columns
        assert df_result.loc[0, "TARGET"] == NUTRITIONAL_STATE_MAP["Eutrofia"]
        assert df_result.loc[1, "TARGET"] == NUTRITIONAL_STATE_MAP["Obesidade"]
        assert df_result.loc[2, "TARGET"] == NUTRITIONAL_STATE_MAP["Baixo peso"]

    def test_create_target_unmapped_values(self):
        """Testa comportamento com valores não mapeados."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", "Valor Desconhecido"],
            "NU_PESO": [70.0, 80.0]
        })
        
        df_result = create_numeric_target(df)
        
        assert "TARGET" in df_result.columns
        assert df_result.loc[0, "TARGET"] == NUTRITIONAL_STATE_MAP["Eutrofia"]
        assert pd.isna(df_result.loc[1, "TARGET"])

    def test_create_target_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia"],
            "NU_PESO": [70.0]
        })
        original_cols = set(df.columns)
        
        create_numeric_target(df)
        
        assert set(df.columns) == original_cols


class TestCalculateBodyFatPercentage:
    def test_calculate_children(self):
        """Testa cálculo para crianças (idade <= 15)."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [10, 12, 15],
            "DS_IMC": [18.0, 20.0, 22.0],
            "SG_SEXO": [0, 1, 0],  # 0=F, 1=M
        })
        
        df_result = calculate_body_fat_percentage(df)
        
        assert "PERC_GORDURA" in df_result.columns
        # Fórmula crianças: 1.20*IMC + 0.23*idade - 10.8*sexo - 5.4
        expected_0 = 1.20 * 18.0 + 0.23 * 10 - 10.8 * 0 - 5.4
        assert df_result.loc[0, "PERC_GORDURA"] == pytest.approx(expected_0, rel=0.01)

    def test_calculate_adults(self):
        """Testa cálculo para adultos (idade > 15)."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [20, 30, 40],
            "DS_IMC": [22.0, 25.0, 28.0],
            "SG_SEXO": [0, 1, 0],  # 0=F, 1=M
        })
        
        df_result = calculate_body_fat_percentage(df)
        
        assert "PERC_GORDURA" in df_result.columns
        # Fórmula adultos: 1.51*IMC - 0.70*idade - 3.6*sexo + 1.4
        expected_0 = 1.51 * 22.0 - 0.70 * 20 - 3.6 * 0 + 1.4
        assert df_result.loc[0, "PERC_GORDURA"] == pytest.approx(expected_0, rel=0.01)

    def test_calculate_mixed_ages(self):
        """Testa cálculo com mistura de crianças e adultos."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [10, 20],
            "DS_IMC": [18.0, 22.0],
            "SG_SEXO": [0, 0],
        })
        
        df_result = calculate_body_fat_percentage(df)
        
        assert "PERC_GORDURA" in df_result.columns
        assert len(df_result) == 2

    def test_calculate_custom_columns(self):
        """Testa cálculo com nomes de colunas customizados."""
        df = pd.DataFrame({
            "IDADE": [20],
            "IMC": [22.0],
            "SEXO": [0],
        })
        
        df_result = calculate_body_fat_percentage(
            df,
            age_col="IDADE",
            imc_col="IMC",
            sex_col="SEXO"
        )
        
        assert "PERC_GORDURA" in df_result.columns

    def test_calculate_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [20],
            "DS_IMC": [22.0],
            "SG_SEXO": [0],
        })
        original_cols = set(df.columns)
        
        calculate_body_fat_percentage(df)
        
        assert set(df.columns) == original_cols


class TestRunFeatureEngineering:
    def test_full_pipeline(self):
        """Testa pipeline completo de engenharia de features."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", "Obesidade"],
            "DS_FASE_VIDA": ["Adulto", "Idoso"],
            "SG_SEXO": ["Masculino", "Feminino"],
            "NU_IDADE_ANO": [25, 65],
            "DS_IMC": [22.0, 28.0],
            "NU_PESO": [70.0, 80.0],
        })
        
        df_result, encoders = run_feature_engineering(df)
        
        assert "TARGET" in df_result.columns
        assert "PERC_GORDURA" in df_result.columns
        assert len(encoders) > 0

    def test_pipeline_selective_steps(self):
        """Testa pipeline com passos selecionados."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia"],
            "DS_FASE_VIDA": ["Adulto"],
            "SG_SEXO": ["Masculino"],
            "NU_IDADE_ANO": [25],
            "DS_IMC": [22.0],
        })
        
        df_result, encoders = run_feature_engineering(
            df,
            create_target=True,
            encode_cats=False,
            calculate_fat=False
        )
        
        assert "TARGET" in df_result.columns
        assert "PERC_GORDURA" not in df_result.columns
        assert len(encoders) == 0

    def test_pipeline_all_disabled(self):
        """Testa pipeline com todos os passos desabilitados."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia"],
            "NU_PESO": [70.0],
        })
        
        df_result, encoders = run_feature_engineering(
            df,
            create_target=False,
            encode_cats=False,
            calculate_fat=False
        )
        
        assert "TARGET" not in df_result.columns
        assert "PERC_GORDURA" not in df_result.columns
        assert len(encoders) == 0
