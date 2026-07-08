"""Testes unitários para o módulo de snapshots do GA."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
import tempfile

from src.models.ga_snapshot import (
    _snapshot_path,
    save_generation_snapshot,
    cleanup_snapshot_file,
    read_generation_snapshots,
)


class TestSnapshotPath:
    @patch('src.models.ga_snapshot._TMP_DIR')
    def test_snapshot_path(self, mock_tmp_dir):
        """Testa geração de caminho de snapshot."""
        mock_tmp_dir.__truediv__.return_value = Path("/tmp/ag_job_test123_generations.jsonl")
        
        path = _snapshot_path("test123")
        
        assert "test123" in str(path)
        assert path.suffix == ".jsonl"


class TestSaveGenerationSnapshot:
    def test_save_generation_snapshot(self, tmp_path):
        """Testa salvamento de snapshot de geração."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot = {
                "generation": 1,
                "best_fitness": 0.85,
                "avg_fitness": 0.75
            }
            
            save_generation_snapshot("job_123", snapshot)
            
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            assert path.exists()

    def test_save_generation_snapshot_append(self, tmp_path):
        """Testa que snapshots são adicionados (append) ao arquivo."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot1 = {"generation": 1, "best_fitness": 0.85}
            snapshot2 = {"generation": 2, "best_fitness": 0.90}
            
            save_generation_snapshot("job_123", snapshot1)
            save_generation_snapshot("job_123", snapshot2)
            
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            with open(path, encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == 2

    def test_save_generation_snapshot_content(self, tmp_path):
        """Testa conteúdo do snapshot salvo."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot = {
                "generation": 1,
                "best_fitness": 0.85,
                "avg_fitness": 0.75,
                "population_size": 10
            }
            
            save_generation_snapshot("job_123", snapshot)
            
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            with open(path, encoding='utf-8') as f:
                content = f.read().strip()
            
            loaded = json.loads(content)
            assert loaded["generation"] == 1
            assert loaded["best_fitness"] == 0.85


class TestCleanupSnapshotFile:
    def test_cleanup_snapshot_file_exists(self, tmp_path):
        """Testa remoção de arquivo de snapshot existente."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            # Criar arquivo
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            path.write_text("test content")
            
            cleanup_snapshot_file("job_123")
            
            assert not path.exists()

    def test_cleanup_snapshot_file_not_exists(self, tmp_path):
        """Testa cleanup quando arquivo não existe."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            # Não criar arquivo
            cleanup_snapshot_file("job_123")
            
            # Não deve lançar erro
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            assert not path.exists()


class TestReadGenerationSnapshots:
    def test_read_generation_snapshots_file_not_exists(self, tmp_path):
        """Testa leitura quando arquivo não existe."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshots = read_generation_snapshots("job_123")
            
            assert snapshots == []

    def test_read_generation_snapshots_all(self, tmp_path):
        """Testa leitura de todos os snapshots."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot1 = {"generation": 1, "best_fitness": 0.85}
            snapshot2 = {"generation": 2, "best_fitness": 0.90}
            snapshot3 = {"generation": 3, "best_fitness": 0.92}
            
            save_generation_snapshot("job_123", snapshot1)
            save_generation_snapshot("job_123", snapshot2)
            save_generation_snapshot("job_123", snapshot3)
            
            snapshots = read_generation_snapshots("job_123", since_generation=0)
            
            assert len(snapshots) == 3
            assert snapshots[0]["generation"] == 1
            assert snapshots[1]["generation"] == 2
            assert snapshots[2]["generation"] == 3

    def test_read_generation_snapshots_since(self, tmp_path):
        """Testa leitura incremental com since_generation."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot1 = {"generation": 1, "best_fitness": 0.85}
            snapshot2 = {"generation": 2, "best_fitness": 0.90}
            snapshot3 = {"generation": 3, "best_fitness": 0.92}
            
            save_generation_snapshot("job_123", snapshot1)
            save_generation_snapshot("job_123", snapshot2)
            save_generation_snapshot("job_123", snapshot3)
            
            snapshots = read_generation_snapshots("job_123", since_generation=1)
            
            assert len(snapshots) == 2
            assert snapshots[0]["generation"] == 2
            assert snapshots[1]["generation"] == 3

    def test_read_generation_snapshots_empty_lines(self, tmp_path):
        """Testa leitura com linhas vazias."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            path.write_text('{"generation": 1}\n\n{"generation": 2}\n')
            
            snapshots = read_generation_snapshots("job_123")
            
            assert len(snapshots) == 2

    def test_read_generation_snapshots_invalid_json(self, tmp_path):
        """Testa leitura com JSON inválido (deve ignorar linha inválida)."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            path = tmp_path / "ag_job_job_123_generations.jsonl"
            path.write_text('{"generation": 1}\ninvalid json\n{"generation": 2}\n')
            
            snapshots = read_generation_snapshots("job_123")
            
            # Deve ignorar a linha inválida
            assert len(snapshots) == 2

    def test_read_generation_snapshots_no_new_snapshots(self, tmp_path):
        """Testa quando não há novos snapshots após since_generation."""
        with patch('src.models.ga_snapshot._TMP_DIR', tmp_path):
            snapshot1 = {"generation": 1, "best_fitness": 0.85}
            
            save_generation_snapshot("job_123", snapshot1)
            
            snapshots = read_generation_snapshots("job_123", since_generation=5)
            
            assert len(snapshots) == 0
