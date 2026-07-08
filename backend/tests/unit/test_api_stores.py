"""Testes unitários para os stores da API (job_store, pipeline_store, session_store)."""

import pytest
from src.api.job_store import (
    create_job,
    get_job,
    set_job_running,
    set_job_completed,
    set_job_failed,
)
from src.api.pipeline_store import (
    reset_pipeline,
    set_preprocessing_completed,
    set_tuning_completed,
    set_predictions_completed,
    set_comparison_completed,
    get_pipeline_state,
    check_preprocessing_completed,
    check_tuning_completed,
    check_predictions_completed,
    check_comparison_completed,
)
from src.api.session_store import (
    create_session,
    get_session,
    delete_session,
)


class TestJobStore:
    def test_create_job(self):
        """Testa criação de um novo job."""
        job_id = create_job()
        
        assert job_id is not None
        assert isinstance(job_id, str)
        
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert job["result"] is None
        assert job["error"] is None

    def test_get_job_not_found(self):
        """Testa busca de job inexistente."""
        job = get_job("nonexistent_job_id")
        assert job is None

    def test_set_job_running(self):
        """Testa mudança de status para running."""
        job_id = create_job()
        set_job_running(job_id)
        
        job = get_job(job_id)
        assert job["status"] == "running"

    def test_set_job_completed(self):
        """Testa mudança de status para completed."""
        job_id = create_job()
        result = {"metric": 0.95}
        set_job_completed(job_id, result)
        
        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["result"] == result

    def test_set_job_failed(self):
        """Testa mudança de status para failed."""
        job_id = create_job()
        error = "Erro no processamento"
        set_job_failed(job_id, error)
        
        job = get_job(job_id)
        assert job["status"] == "failed"
        assert job["error"] == error

    def test_job_lifecycle(self):
        """Testa ciclo de vida completo de um job."""
        job_id = create_job()
        
        assert get_job(job_id)["status"] == "pending"
        
        set_job_running(job_id)
        assert get_job(job_id)["status"] == "running"
        
        set_job_completed(job_id, {"result": "ok"})
        assert get_job(job_id)["status"] == "completed"
        assert get_job(job_id)["result"] == {"result": "ok"}

    def test_multiple_jobs(self):
        """Testa criação de múltiplos jobs."""
        job_id1 = create_job()
        job_id2 = create_job()
        
        assert job_id1 != job_id2
        assert get_job(job_id1) is not None
        assert get_job(job_id2) is not None


class TestPipelineStore:
    def test_reset_pipeline(self):
        """Testa reset do estado do pipeline."""
        set_preprocessing_completed()
        set_tuning_completed()
        set_predictions_completed()
        
        reset_pipeline()
        
        state = get_pipeline_state()
        assert state["preprocessing_completed"] is False
        assert state["tuning_completed"] is False
        assert state["predictions_completed"] is False
        assert state["comparison_completed"] is False

    def test_set_preprocessing_completed(self):
        """Testa marcação de preprocessing como concluído."""
        reset_pipeline()
        set_preprocessing_completed()
        
        assert check_preprocessing_completed() is True

    def test_set_tuning_completed(self):
        """Testa marcação de tuning como concluído."""
        reset_pipeline()
        set_tuning_completed()
        
        assert check_tuning_completed() is True

    def test_set_predictions_completed(self):
        """Testa marcação de predictions como concluído."""
        reset_pipeline()
        set_predictions_completed()
        
        assert check_predictions_completed() is True

    def test_set_comparison_completed(self):
        """Testa marcação de comparison como concluído."""
        reset_pipeline()
        set_comparison_completed()
        
        assert check_comparison_completed() is True

    def test_get_pipeline_state(self):
        """Testa recuperação do estado do pipeline."""
        reset_pipeline()
        set_preprocessing_completed()
        set_tuning_completed()
        
        state = get_pipeline_state()
        
        assert state["preprocessing_completed"] is True
        assert state["tuning_completed"] is True
        assert state["predictions_completed"] is False
        assert state["comparison_completed"] is False

    def test_check_functions_initial_state(self):
        """Testa verificações no estado inicial."""
        reset_pipeline()
        
        assert check_preprocessing_completed() is False
        assert check_tuning_completed() is False
        assert check_predictions_completed() is False
        assert check_comparison_completed() is False

    def test_pipeline_state_returns_copy(self):
        """Testa que get_pipeline_state retorna uma cópia."""
        reset_pipeline()
        state1 = get_pipeline_state()
        state1["preprocessing_completed"] = True
        
        state2 = get_pipeline_state()
        
        assert state2["preprocessing_completed"] is False


class TestSessionStore:
    def test_create_session(self):
        """Testa criação de uma sessão."""
        mock_agent = {"data": "mock"}
        session_id = create_session(mock_agent)
        
        assert session_id is not None
        assert isinstance(session_id, str)
        
        agent = get_session(session_id)
        assert agent is not None
        assert agent == mock_agent

    def test_get_session_not_found(self):
        """Testa busca de sessão inexistente."""
        agent = get_session("nonexistent_session_id")
        assert agent is None

    def test_delete_session(self):
        """Testa deleção de uma sessão."""
        mock_agent = {"data": "mock"}
        session_id = create_session(mock_agent)
        
        result = delete_session(session_id)
        assert result is True
        
        agent = get_session(session_id)
        assert agent is None

    def test_delete_session_not_found(self):
        """Testa deleção de sessão inexistente."""
        result = delete_session("nonexistent_session_id")
        assert result is False

    def test_session_lifecycle(self):
        """Testa ciclo de vida completo de uma sessão."""
        mock_agent = {"data": "mock"}
        session_id = create_session(mock_agent)
        
        agent = get_session(session_id)
        assert agent == mock_agent
        
        delete_session(session_id)
        agent = get_session(session_id)
        assert agent is None

    def test_multiple_sessions(self):
        """Testa criação de múltiplas sessões."""
        agent1 = {"data": "agent1"}
        agent2 = {"data": "agent2"}
        
        session_id1 = create_session(agent1)
        session_id2 = create_session(agent2)
        
        assert session_id1 != session_id2
        assert get_session(session_id1) == agent1
        assert get_session(session_id2) == agent2
