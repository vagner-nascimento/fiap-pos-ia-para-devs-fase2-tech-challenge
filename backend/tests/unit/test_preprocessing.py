"""Testes unitários para o módulo de pré-processamento."""

import pandas as pd
import pytest
from src.data.preprocessing import (
    remove_pregnant_women,
    drop_irrelevant_columns,
    consolidate_nutritional_status,
    remove_missing_target,
    drop_target_columns,
    standardize_nutritional_categories,
    convert_numeric_columns,
    remove_race_column,
    run_preprocessing,
    COLUMNS_TO_DROP,
    TARGET_COLUMNS,
)


class TestRemovePregnantWomen:
    def test_remove_pregnant_records(self):
        """Testa remoção de registros de gestantes."""
        df = pd.DataFrame({
            "CO_ESTADO_NUTRI_IMC_SEMGEST": [None, "Gestante", None, None],
            "NU_PESO": [70.0, 65.0, 80.0, 75.0]
        })
        
        df_result = remove_pregnant_women(df)
        
        assert len(df_result) == 3
        assert all(df_result["CO_ESTADO_NUTRI_IMC_SEMGEST"].isna())

    def test_remove_no_pregnant(self):
        """Testa quando não há gestantes."""
        df = pd.DataFrame({
            "CO_ESTADO_NUTRI_IMC_SEMGEST": [None, None, None],
            "NU_PESO": [70.0, 65.0, 80.0]
        })
        
        df_result = remove_pregnant_women(df)
        
        assert len(df_result) == 3

    def test_remove_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "CO_ESTADO_NUTRI_IMC_SEMGEST": [None, "Gestante"],
            "NU_PESO": [70.0, 65.0]
        })
        original_len = len(df)
        
        remove_pregnant_women(df)
        
        assert len(df) == original_len


class TestDropIrrelevantColumns:
    def test_drop_default_columns(self):
        """Testa remoção de colunas irrelevantes padrão."""
        df = pd.DataFrame({
            "NU_PESO": [70.0, 65.0],
            "ST_PARTICIPA_ANDI": ["A", "B"],
            "CO_POVO_COMUNIDADE": [1, 2],
            "NU_ALTURA": [170.0, 175.0]
        })
        
        df_result = drop_irrelevant_columns(df)
        
        assert "ST_PARTICIPA_ANDI" not in df_result.columns
        assert "CO_POVO_COMUNIDADE" not in df_result.columns
        assert "NU_PESO" in df_result.columns
        assert "NU_ALTURA" in df_result.columns

    def test_drop_columns_not_in_dataframe(self):
        """Testa remoção quando colunas não existem no DataFrame."""
        df = pd.DataFrame({
            "NU_PESO": [70.0, 65.0],
            "NU_ALTURA": [170.0, 175.0]
        })
        
        df_result = drop_irrelevant_columns(df)
        
        assert len(df_result.columns) == 2

    def test_drop_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "NU_PESO": [70.0],
            "ST_PARTICIPA_ANDI": ["A"]
        })
        original_cols = set(df.columns)
        
        drop_irrelevant_columns(df)
        
        assert set(df.columns) == original_cols


class TestConsolidateNutritionalStatus:
    def test_consolidate_targets(self):
        """Testa consolidação de targets por faixa etária."""
        df = pd.DataFrame({
            "CRI. IMC X IDADE": ["Eutrofia", None, None],
            "ADO. IMC X IDADE": [None, "Obesidade", None],
            "CO_ESTADO_NUTRI_ADULTO": [None, None, "Baixo peso"],
            "CO_ESTADO_NUTRI_IDOSO": [None, None, None],
        })
        
        df_result = consolidate_nutritional_status(df)
        
        assert "ESTADO_NUTRI" in df_result.columns
        assert df_result.loc[0, "ESTADO_NUTRI"] == "Eutrofia"
        assert df_result.loc[1, "ESTADO_NUTRI"] == "Obesidade"
        assert df_result.loc[2, "ESTADO_NUTRI"] == "Baixo peso"

    def test_consolidate_with_all_none(self):
        """Testa consolidação quando todos são None."""
        df = pd.DataFrame({
            "CRI. IMC X IDADE": [None, None],
            "ADO. IMC X IDADE": [None, None],
            "CO_ESTADO_NUTRI_ADULTO": [None, None],
            "CO_ESTADO_NUTRI_IDOSO": [None, None],
        })
        
        df_result = consolidate_nutritional_status(df)
        
        assert "ESTADO_NUTRI" in df_result.columns
        assert df_result["ESTADO_NUTRI"].isna().all()

    def test_consolidate_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "CRI. IMC X IDADE": ["Eutrofia"],
            "ADO. IMC X IDADE": [None],
            "CO_ESTADO_NUTRI_ADULTO": [None],
            "CO_ESTADO_NUTRI_IDOSO": [None],
        })
        original_cols = set(df.columns)
        
        consolidate_nutritional_status(df)
        
        assert set(df.columns) == original_cols


class TestRemoveMissingTarget:
    def test_remove_missing_target(self):
        """Testa remoção de registros sem target."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", None, "Obesidade"],
            "NU_PESO": [70.0, 65.0, 80.0]
        })
        
        df_result = remove_missing_target(df)
        
        assert len(df_result) == 2
        assert df_result["ESTADO_NUTRI"].notna().all()

    def test_remove_no_missing(self):
        """Testa quando não há missing targets."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", "Obesidade"],
            "NU_PESO": [70.0, 80.0]
        })
        
        df_result = remove_missing_target(df)
        
        assert len(df_result) == 2

    def test_remove_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", None],
            "NU_PESO": [70.0, 65.0]
        })
        original_len = len(df)
        
        remove_missing_target(df)
        
        assert len(df) == original_len


class TestDropTargetColumns:
    def test_drop_target_columns(self):
        """Testa remoção de colunas de target por faixa etária."""
        df = pd.DataFrame({
            "CRI. IMC X IDADE": ["Eutrofia"],
            "ADO. IMC X IDADE": ["Obesidade"],
            "NU_PESO": [70.0]
        })
        
        df_result = drop_target_columns(df)
        
        assert "CRI. IMC X IDADE" not in df_result.columns
        assert "ADO. IMC X IDADE" not in df_result.columns
        assert "NU_PESO" in df_result.columns

    def test_drop_no_target_columns(self):
        """Testa quando não há colunas de target."""
        df = pd.DataFrame({
            "NU_PESO": [70.0],
            "NU_ALTURA": [170.0]
        })
        
        df_result = drop_target_columns(df)
        
        assert len(df_result.columns) == 2


class TestStandardizeNutritionalCategories:
    def test_standardize_categories(self):
        """Testa padronização de categorias."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": [
                "Obesidade Grau III",
                "Obesidade Grau II",
                "Obesidade Grau I",
                "Magreza acentuada",
                "Magreza",
                "Adequado ou Eutrófico",
                "Sobrepeso",
                "Risco de sobrepeso"
            ]
        })
        
        df_result = standardize_nutritional_categories(df)
        
        assert df_result.loc[0, "ESTADO_NUTRI"] == "Obesidade Grave"
        assert df_result.loc[1, "ESTADO_NUTRI"] == "Obesidade Grave"
        assert df_result.loc[2, "ESTADO_NUTRI"] == "Obesidade"
        assert df_result.loc[3, "ESTADO_NUTRI"] == "Baixo peso"
        assert df_result.loc[4, "ESTADO_NUTRI"] == "Baixo peso"
        assert df_result.loc[5, "ESTADO_NUTRI"] == "Eutrofia"
        assert df_result.loc[6, "ESTADO_NUTRI"] == "Risco/Sobrepeso"
        assert df_result.loc[7, "ESTADO_NUTRI"] == "Risco/Sobrepeso"

    def test_standardize_already_standard(self):
        """Testa quando categorias já estão padronizadas."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Eutrofia", "Obesidade", "Baixo peso"]
        })
        
        df_result = standardize_nutritional_categories(df)
        
        assert df_result.loc[0, "ESTADO_NUTRI"] == "Eutrofia"
        assert df_result.loc[1, "ESTADO_NUTRI"] == "Obesidade"
        assert df_result.loc[2, "ESTADO_NUTRI"] == "Baixo peso"

    def test_standardize_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "ESTADO_NUTRI": ["Obesidade Grau III"]
        })
        original_value = df.loc[0, "ESTADO_NUTRI"]
        
        standardize_nutritional_categories(df)
        
        assert df.loc[0, "ESTADO_NUTRI"] == original_value


class TestConvertNumericColumns:
    def test_convert_comma_to_dot(self):
        """Testa conversão de vírgula para ponto."""
        df = pd.DataFrame({
            "NU_PESO": ["70,5", "65,3"],
            "NU_ALTURA": ["170,0", "175,5"],
            "DS_IMC": ["24,4", "21,2"]
        })
        
        df_result = convert_numeric_columns(df)
        
        assert df_result["NU_PESO"].dtype == float
        assert df_result.loc[0, "NU_PESO"] == 70.5
        assert df_result.loc[1, "NU_PESO"] == 65.3

    def test_convert_already_float(self):
        """Testa conversão quando já são float."""
        df = pd.DataFrame({
            "NU_PESO": [70.5, 65.3],
            "NU_ALTURA": [170.0, 175.5],
            "DS_IMC": [24.4, 21.2]
        })
        
        df_result = convert_numeric_columns(df)
        
        assert df_result["NU_PESO"].dtype == float

    def test_convert_missing_columns(self):
        """Testa conversão quando colunas não existem."""
        df = pd.DataFrame({
            "OTHER_COL": [1, 2, 3]
        })
        
        df_result = convert_numeric_columns(df)
        
        assert len(df_result.columns) == 1

    def test_convert_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "NU_PESO": ["70,5"]
        })
        original_dtype = df["NU_PESO"].dtype
        
        convert_numeric_columns(df)
        
        assert df["NU_PESO"].dtype == original_dtype


class TestRemoveRaceColumn:
    def test_remove_race_column(self):
        """Testa remoção da coluna DS_RACA_COR."""
        df = pd.DataFrame({
            "DS_RACA_COR": ["Branca", "Parda"],
            "NU_PESO": [70.0, 65.0]
        })
        
        df_result = remove_race_column(df)
        
        assert "DS_RACA_COR" not in df_result.columns
        assert "NU_PESO" in df_result.columns

    def test_remove_race_column_not_exists(self):
        """Testa quando coluna não existe."""
        df = pd.DataFrame({
            "NU_PESO": [70.0, 65.0]
        })
        
        df_result = remove_race_column(df)
        
        assert len(df_result.columns) == 1

    def test_remove_race_returns_copy(self):
        """Testa que o DataFrame original não é modificado."""
        df = pd.DataFrame({
            "DS_RACA_COR": ["Branca"],
            "NU_PESO": [70.0]
        })
        original_cols = set(df.columns)
        
        remove_race_column(df)
        
        assert set(df.columns) == original_cols


class TestRunPreprocessing:
    def test_full_pipeline(self):
        """Testa pipeline completo de pré-processamento."""
        df = pd.DataFrame({
            "CO_ESTADO_NUTRI_IMC_SEMGEST": [None, None, None],
            "ST_PARTICIPA_ANDI": ["A", "B", "C"],
            "CRI. IMC X IDADE": ["Eutrofia", None, None],
            "ADO. IMC X IDADE": [None, "Obesidade", None],
            "CO_ESTADO_NUTRI_ADULTO": [None, None, "Baixo peso"],
            "CO_ESTADO_NUTRI_IDOSO": [None, None, None],
            "DS_RACA_COR": ["Branca", "Parda", "Preta"],
            "NU_PESO": ["70,5", "65,3", "80,0"],
            "NU_ALTURA": ["170,0", "175,5", "180,0"],
            "DS_IMC": ["24,4", "21,2", "24,7"],
        })
        
        df_result = run_preprocessing(df)
        
        assert "ESTADO_NUTRI" in df_result.columns
        assert "ST_PARTICIPA_ANDI" not in df_result.columns
        assert "DS_RACA_COR" not in df_result.columns
        assert df_result["NU_PESO"].dtype == float

    def test_pipeline_selective_steps(self):
        """Testa pipeline com passos selecionados."""
        df = pd.DataFrame({
            "CO_ESTADO_NUTRI_IMC_SEMGEST": [None],
            "CRI. IMC X IDADE": ["Eutrofia"],
            "ADO. IMC X IDADE": [None],
            "CO_ESTADO_NUTRI_ADULTO": [None],
            "CO_ESTADO_NUTRI_IDOSO": [None],
            "NU_PESO": ["70,5"],
        })
        
        df_result = run_preprocessing(
            df,
            remove_pregnant=True,
            drop_cols=False,
            consolidate_target=True,
            remove_missing=True,
            standardize=False,
            convert_numeric=False,
            remove_race=False
        )
        
        assert "ESTADO_NUTRI" in df_result.columns
        assert "CO_ESTADO_NUTRI_IMC_SEMGEST" in df_result.columns
        assert df_result["NU_PESO"].dtype == object

    def test_pipeline_all_disabled(self):
        """Testa pipeline com todos os passos desabilitados."""
        df = pd.DataFrame({
            "NU_PESO": [70.0],
            "NU_ALTURA": [170.0]
        })
        
        df_result = run_preprocessing(
            df,
            remove_pregnant=False,
            drop_cols=False,
            consolidate_target=False,
            remove_missing=False,
            standardize=False,
            convert_numeric=False,
            remove_race=False
        )
        
        assert df_result.shape == df.shape
