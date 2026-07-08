"""Testes unitários para o módulo de ingestão de dados."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.ingest import (
    extract_rar_file,
    read_csv_data,
    validate_dataframe,
)


class TestExtractRarFile:
    @patch('src.data.ingest.patoolib.extract_archive')
    @patch('pathlib.Path.exists')
    def test_extract_when_csv_exists(self, mock_exists, mock_extract):
        """Testa que extração é pulada quando CSV já existe."""
        mock_exists.return_value = True
        
        extract_rar_file("test.rar", "test.csv")
        
        mock_extract.assert_not_called()

    @patch('src.data.ingest.patoolib.extract_archive')
    @patch('src.data.ingest.Path')
    def test_extract_when_csv_not_exists(self, mock_path, mock_extract):
        """Testa extração quando CSV não existe."""
        # RAR exists, CSV doesn't
        mock_rar_file = MagicMock()
        mock_rar_file.exists.return_value = True
        mock_csv_file = MagicMock()
        mock_csv_file.exists.return_value = False
        
        def path_side_effect(path_str):
            if "rar" in path_str:
                return mock_rar_file
            return mock_csv_file
        
        mock_path.side_effect = path_side_effect
        
        extract_rar_file("test.rar", "test.csv")
        
        mock_extract.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_extract_rar_not_found(self, mock_exists):
        """Testa erro quando arquivo RAR não existe."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Arquivo .rar não encontrado"):
            extract_rar_file("nonexistent.rar", "test.csv")

    @patch('src.data.ingest.patoolib.extract_archive')
    @patch('src.data.ingest.Path')
    def test_extract_error_handling(self, mock_path, mock_extract):
        """Testa tratamento de erro na extração."""
        # RAR exists, CSV doesn't
        mock_rar_file = MagicMock()
        mock_rar_file.exists.return_value = True
        mock_csv_file = MagicMock()
        mock_csv_file.exists.return_value = False
        
        def path_side_effect(path_str):
            if "rar" in path_str:
                return mock_rar_file
            return mock_csv_file
        
        mock_path.side_effect = path_side_effect
        mock_extract.side_effect = Exception("Erro de extração")
        
        with pytest.raises(Exception, match="Erro de extração"):
            extract_rar_file("test.rar", "test.csv")


class TestReadCsvData:
    def test_read_csv_success(self, tmp_path):
        """Testa leitura bem-sucedida de CSV."""
        csv_file = tmp_path / "test.csv"
        test_data = "col1,col2\n1,2\n3,4\n"
        csv_file.write_text(test_data, encoding='utf-8')
        
        df = read_csv_data(str(csv_file))
        
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_read_csv_file_not_found(self):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            read_csv_data("nonexistent.csv")

    def test_read_csv_with_custom_encoding(self, tmp_path):
        """Testa leitura com encoding customizado."""
        csv_file = tmp_path / "test.csv"
        test_data = "col1,col2\n1,2\n"
        csv_file.write_text(test_data, encoding='latin-1')
        
        df = read_csv_data(str(csv_file), encoding='latin-1')
        
        assert df.shape == (1, 2)

    def test_read_csv_with_kwargs(self, tmp_path):
        """Testa leitura com kwargs adicionais."""
        csv_file = tmp_path / "test.csv"
        test_data = "col1,col2,col3\n1,2,3\n"
        csv_file.write_text(test_data)
        
        df = read_csv_data(str(csv_file), usecols=["col1", "col2"])
        
        assert list(df.columns) == ["col1", "col2"]

    def test_read_csv_uses_python_engine(self, tmp_path):
        """Testa que engine='python' é usado por padrão."""
        csv_file = tmp_path / "test.csv"
        test_data = "col1,col2\n1,2\n"
        csv_file.write_text(test_data)
        
        with patch('pandas.read_csv') as mock_read:
            read_csv_data(str(csv_file))
            
            call_kwargs = mock_read.call_args[1]
            assert call_kwargs.get('engine') == 'python'
            assert call_kwargs.get('memory_map') == False


class TestValidateDataFrame:
    def test_validate_valid_dataframe(self):
        """Testa validação de DataFrame válido."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        result = validate_dataframe(df)
        
        assert result is True

    def test_validate_none_dataframe(self):
        """Testa erro quando DataFrame é None."""
        with pytest.raises(ValueError, match="DataFrame vazio ou None"):
            validate_dataframe(None)

    def test_validate_empty_dataframe(self):
        """Testa erro quando DataFrame está vazio."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame vazio ou None"):
            validate_dataframe(df)

    def test_validate_missing_required_columns(self):
        """Testa erro quando faltam colunas obrigatórias."""
        df = pd.DataFrame({"col1": [1, 2]})
        
        with pytest.raises(ValueError, match="Colunas obrigatórias faltando"):
            validate_dataframe(df, required_columns=["col1", "col2"])

    def test_validate_with_required_columns(self):
        """Testa validação com colunas obrigatórias presentes."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        result = validate_dataframe(df, required_columns=["col1", "col2"])
        
        assert result is True

    def test_validate_empty_required_columns(self):
        """Testa validação sem colunas obrigatórias especificadas."""
        df = pd.DataFrame({"col1": [1, 2]})
        
        result = validate_dataframe(df, required_columns=None)
        
        assert result is True
