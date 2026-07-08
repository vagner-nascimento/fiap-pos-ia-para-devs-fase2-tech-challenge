"""Testes unitários para os utilitários (logger, persistence, validators)."""

import pytest
import pandas as pd
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.utils.logger import setup_logger, get_logger
from src.utils.persistence import (
    save_dataframe,
    load_dataframe,
    save_model,
    load_model,
    save_dict,
    load_dict,
)
from src.utils.validators import (
    validate_columns_exist,
    validate_no_missing_values,
    validate_data_types,
    validate_value_ranges,
    validate_nutritional_data,
)


class TestLogger:
    def test_setup_logger_console_only(self):
        """Testa configuração de logger apenas com console."""
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

    def test_setup_logger_with_file(self, tmp_path):
        """Testa configuração de logger com arquivo."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_logger", log_file=str(log_file))
        
        assert len(logger.handlers) == 2

    def test_setup_logger_custom_level(self):
        """Testa configuração de logger com nível customizado."""
        logger = setup_logger("test_logger", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG

    def test_setup_logger_custom_format(self):
        """Testa configuração de logger com formato customizado."""
        custom_format = "%(name)s - %(message)s"
        logger = setup_logger("test_logger", format_string=custom_format)
        
        assert logger.handlers[0].formatter._fmt == custom_format

    def test_get_logger_existing(self):
        """Testa busca de logger existente."""
        logger1 = setup_logger("test_get")
        logger2 = get_logger("test_get")
        
        assert logger1.name == logger2.name

    def test_get_logger_new(self):
        """Testa criação de novo logger."""
        logger = get_logger("new_logger")
        
        assert logger.name == "new_logger"


class TestPersistence:
    def test_save_dataframe_csv(self, tmp_path):
        """Testa salvamento de DataFrame em CSV."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        file_path = tmp_path / "test.csv"
        
        save_dataframe(df, str(file_path), format="csv")
        
        assert file_path.exists()

    def test_save_dataframe_unsupported_format(self, tmp_path):
        """Testa erro ao salvar em formato não suportado."""
        df = pd.DataFrame({"col1": [1, 2]})
        file_path = tmp_path / "test.xyz"
        
        with pytest.raises(ValueError, match="Formato não suportado"):
            save_dataframe(df, str(file_path), format="xyz")

    def test_load_dataframe_csv(self, tmp_path):
        """Testa carregamento de DataFrame de CSV."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        loaded_df = load_dataframe(str(file_path), format="csv")
        
        assert loaded_df.shape == df.shape
        assert list(loaded_df.columns) == list(df.columns)

    def test_load_dataframe_file_not_found(self):
        """Testa erro ao carregar arquivo inexistente."""
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            load_dataframe("nonexistent.csv")

    def test_load_dataframe_unsupported_format(self, tmp_path):
        """Testa erro ao carregar formato não suportado."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("test")
        
        with pytest.raises(ValueError, match="Formato não suportado"):
            load_dataframe(str(file_path), format="xyz")

    def test_save_model_joblib(self, tmp_path):
        """Testa salvamento de modelo em joblib."""
        model = {"param": "value"}
        file_path = tmp_path / "model.joblib"
        
        save_model(model, str(file_path), format="joblib")
        
        assert file_path.exists()

    def test_save_model_pickle(self, tmp_path):
        """Testa salvamento de modelo em pickle."""
        model = {"param": "value"}
        file_path = tmp_path / "model.pkl"
        
        save_model(model, str(file_path), format="pickle")
        
        assert file_path.exists()

    def test_save_model_unsupported_format(self, tmp_path):
        """Testa erro ao salvar em formato não suportado."""
        model = {"param": "value"}
        file_path = tmp_path / "model.xyz"
        
        with pytest.raises(ValueError, match="Formato não suportado"):
            save_model(model, str(file_path), format="xyz")

    def test_load_model_joblib(self, tmp_path):
        """Testa carregamento de modelo em joblib."""
        model = {"param": "value"}
        file_path = tmp_path / "model.joblib"
        
        save_model(model, str(file_path), format="joblib")
        loaded_model = load_model(str(file_path), format="joblib")
        
        assert loaded_model == model

    def test_load_model_file_not_found(self):
        """Testa erro ao carregar modelo inexistente."""
        with pytest.raises(FileNotFoundError, match="Arquivo de modelo não encontrado"):
            load_model("nonexistent.joblib")

    def test_save_dict_joblib(self, tmp_path):
        """Testa salvamento de dicionário em joblib."""
        data = {"key": "value"}
        file_path = tmp_path / "dict.joblib"
        
        save_dict(data, str(file_path), format="joblib")
        
        assert file_path.exists()

    def test_load_dict_joblib(self, tmp_path):
        """Testa carregamento de dicionário em joblib."""
        data = {"key": "value"}
        file_path = tmp_path / "dict.joblib"
        
        save_dict(data, str(file_path), format="joblib")
        loaded_data = load_dict(str(file_path), format="joblib")
        
        assert loaded_data == data

    def test_load_dict_file_not_found(self):
        """Testa erro ao carregar dicionário inexistente."""
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            load_dict("nonexistent.joblib")


class TestValidators:
    def test_validate_columns_exist_success(self):
        """Testa validação de colunas com sucesso."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        valid, missing = validate_columns_exist(df, ["col1", "col2"])
        
        assert valid is True
        assert missing == []

    def test_validate_columns_exist_missing(self):
        """Testa validação de colunas com colunas faltando."""
        df = pd.DataFrame({"col1": [1, 2]})
        
        valid, missing = validate_columns_exist(df, ["col1", "col2", "col3"])
        
        assert valid is False
        assert set(missing) == {"col2", "col3"}

    def test_validate_no_missing_values_success(self):
        """Testa validação de valores ausentes com sucesso."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        valid, info = validate_no_missing_values(df, threshold=0.0)
        
        assert valid is True
        assert all(info[col]["count"] == 0 for col in info)

    def test_validate_no_missing_values_with_missing(self):
        """Testa validação de valores ausentes com dados faltando."""
        df = pd.DataFrame({"col1": [1, None, 3], "col2": [4, 5, 6]})
        
        valid, info = validate_no_missing_values(df, threshold=0.0)
        
        assert valid is False
        assert info["col1"]["count"] == 1

    def test_validate_no_missing_values_subset(self):
        """Testa validação de valores ausentes em subset de colunas."""
        df = pd.DataFrame({"col1": [1, None], "col2": [3, 4], "col3": [5, None]})
        
        valid, info = validate_no_missing_values(df, subset=["col1", "col2"])
        
        assert "col1" in info
        assert "col2" in info
        assert "col3" not in info

    def test_validate_data_types_success(self):
        """Testa validação de tipos de dados com sucesso."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [1.5, 2.5]})
        
        valid, info = validate_data_types(df, {"col1": "int", "col2": "float"})
        
        assert valid is True

    def test_validate_data_types_mismatch(self):
        """Testa validação de tipos de dados com mismatch."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [1.5, 2.5]})
        
        valid, info = validate_data_types(df, {"col1": "float", "col2": "int"})
        
        assert valid is False
        assert info["col1"]["matches"] is False

    def test_validate_value_ranges_success(self):
        """Testa validação de ranges com sucesso."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30]})
        
        valid, info = validate_value_ranges(df, {"col1": (0, 5), "col2": (5, 35)})
        
        assert valid is True

    def test_validate_value_ranges_out_of_range(self):
        """Testa validação de ranges com valores fora do range."""
        df = pd.DataFrame({"col1": [1, 2, 10], "col2": [10, 20, 30]})
        
        valid, info = validate_value_ranges(df, {"col1": (0, 5)})
        
        assert valid is False
        assert info["col1"]["out_of_range_count"] > 0

    def test_validate_nutritional_data_success(self):
        """Testa validação completa de dados nutricionais com sucesso."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [25, 30, 35],
            "NU_PESO": [70.0, 75.0, 80.0],
            "NU_ALTURA": [170.0, 175.0, 180.0],
            "DS_IMC": [24.2, 24.5, 24.7],
            "ESTADO_NUTRI": ["Eutrofia", "Eutrofia", "Obesidade"]
        })
        
        results = validate_nutritional_data(df)
        
        assert results["columns"][0] is True
        assert results["missing_values"][0] is True
        assert results["data_types"][0] is True
        assert results["ranges"][0] is True
        assert results["categories"][0] is True

    def test_validate_nutritional_data_missing_columns(self):
        """Testa validação de dados nutricionais com colunas faltando."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [25, 30],
            "NU_PESO": [70.0, 75.0]
        })
        
        results = validate_nutritional_data(df)
        
        assert results["columns"][0] is False
        assert len(results["columns"][1]) > 0

    def test_validate_nutritional_data_invalid_categories(self):
        """Testa validação de dados nutricionais com categorias inválidas."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [25, 30],
            "NU_PESO": [70.0, 75.0],
            "NU_ALTURA": [170.0, 175.0],
            "DS_IMC": [24.2, 24.5],
            "ESTADO_NUTRI": ["Eutrofia", "Categoria Invalida"]
        })
        
        results = validate_nutritional_data(df)
        
        assert results["categories"][0] is False
        assert len(results["categories"][1]) > 0

    def test_validate_nutritional_data_out_of_range(self):
        """Testa validação de dados nutricionais com valores fora de range."""
        df = pd.DataFrame({
            "NU_IDADE_ANO": [25, 150],  # 150 fora do range (0-120)
            "NU_PESO": [70.0, 75.0],
            "NU_ALTURA": [170.0, 175.0],
            "DS_IMC": [24.2, 24.5],
            "ESTADO_NUTRI": ["Eutrofia", "Eutrofia"]
        })
        
        results = validate_nutritional_data(df)
        
        assert results["ranges"][0] is False
