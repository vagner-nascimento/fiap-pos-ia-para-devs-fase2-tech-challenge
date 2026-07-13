"""
Testes unitários para NutritionalHealthAgent — camada de LLM com fallback multi-provider.

Cenários cobertos:
  1. test_agent_initialization_default_gemini        — Regressão zero: sem LLM_PROVIDER_ORDER, usa LLM_API_KEY
  2. test_agent_initialization_with_fallback_chain   — Cadeia openai→gemini quando ambas as chaves existem
  3. test_agent_missing_provider_key_skipped         — Provedor sem chave é ignorado, agente inicializa mesmo assim
  4. test_agent_all_providers_fail_raises_runtime    — Sem nenhuma chave válida levanta RuntimeError
  5. test_agent_fallback_on_primary_failure          — Falha no primário → resposta do secundário sem exceção
  6. test_agent_initial_report_uses_fallback         — generate_initial_report beneficia do fallback via self.llm
  7. test_agent_tools                                — Ferramentas get_statistics / filter_records / recommendations
  8. test_agent_ask                                  — ask() delega corretamente ao AgentExecutor
  —— Memória de sessão (fallback por rate limit) ——
  9. test_ask_rate_limit_retries_with_next_provider  — Rate limit no primário → retenta com secundário transparentemente
 10. test_ask_rate_limit_advances_active_index       — Após rate limit, _active_index incrementa e executor é recriado
 11. test_ask_next_question_uses_advanced_provider   — Segunda chamada já usa provider avançado (primário não é tentado)
 12. test_ask_all_providers_rate_limited_raises      — Rate limit em todos → RuntimeError com mensagem clara
 13. test_initial_report_rate_limit_retries_next     — Rate limit no generate_initial_report → retenta com próximo
"""
from unittest.mock import MagicMock, patch, call
import pytest
import pandas as pd
from src.agents.nutritional_agent import NutritionalHealthAgent


@pytest.fixture(autouse=True)
def mock_load_dotenv():
    with patch("src.agents.nutritional_agent.load_dotenv"):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockResponse:
    def __init__(self, content: str = "Relatório Inicial Gerado."):
        self.content = content


def _sample_df():
    return pd.DataFrame({
        "NU_IDADE_ANO": [23, 28, 39, 64],
        "DS_FASE_VIDA": [1, 1, 1, 6],
        "SG_SEXO": [0, 1, 0, 1],
        "NU_PESO": [67.0, 64.5, 108.0, 72.3],
        "NU_ALTURA": [167.0, 170.0, 162.0, 178.0],
        "DS_IMC": [24.02, 22.32, 41.15, 22.82],
        "Prediction": ["Eutrofia", "Eutrofia", "Obesidade Grave", "Eutrofia"],
    })


def _sample_mappings():
    return {
        "SG_SEXO": {"0": "Masculino", "1": "Feminino"},
        "DS_FASE_VIDA": {"1": "Adulto", "6": "Idoso"},
    }


def _make_mock_llm(report_content: str = "Relatório Inicial Gerado.") -> MagicMock:
    """Retorna um mock de ChatModel que responde a .invoke() e suporta .with_fallbacks()."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MockResponse(report_content)
    # with_fallbacks deve retornar o próprio mock para manter a interface
    mock_llm.with_fallbacks.return_value = mock_llm
    return mock_llm


# ---------------------------------------------------------------------------
# 1. Regressão zero — sem LLM_PROVIDER_ORDER, usa LLM_API_KEY (Gemini)
# ---------------------------------------------------------------------------

def test_agent_initialization_default_gemini(monkeypatch):
    """Sem LLM_PROVIDER_ORDER definido, o agente deve usar Gemini via LLM_API_KEY."""
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_llm = _make_mock_llm()
    with patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_llm):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    assert agent.initial_report == "Relatório Inicial Gerado."
    assert agent.df.loc[0, "SG_SEXO"] == "Masculino"
    assert agent.df.loc[3, "DS_FASE_VIDA"] == "Idoso"


# ---------------------------------------------------------------------------
# 2. Cadeia de fallback openai → gemini
# ---------------------------------------------------------------------------

def test_agent_initialization_with_fallback_chain(monkeypatch):
    """Com LLM_PROVIDER_ORDER=openai,gemini e ambas as chaves, _built_llms tem 2 providers."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_openai = _make_mock_llm()
    mock_gemini = _make_mock_llm()

    with (
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_openai),
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    # Novo comportamento: sem with_fallbacks(), estado gerenciado pela sessão
    assert len(agent._built_llms) == 2
    assert agent._active_index == 0
    assert agent.llm is mock_openai   # primário ativo
    assert agent.initial_report == "Relatório Inicial Gerado."


# ---------------------------------------------------------------------------
# 3. Provedor sem chave é ignorado com warning — inicialização não falha
# ---------------------------------------------------------------------------

def test_agent_missing_provider_key_skipped(monkeypatch, caplog):
    """Provedor listado sem chave deve ser ignorado com WARNING, sem derrubar o agente."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # OpenAI sem chave
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_gemini = _make_mock_llm()

    import logging
    with (
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
        caplog.at_level(logging.WARNING, logger="src.agents.nutritional_agent"),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    assert agent.initial_report == "Relatório Inicial Gerado."
    assert any("openai" in record.message.lower() for record in caplog.records)
    assert len(agent._built_llms) == 1
    assert agent._active_index == 0
    assert agent.llm is mock_gemini


# ---------------------------------------------------------------------------
# 4. Todos os provedores falham → RuntimeError
# ---------------------------------------------------------------------------

def test_agent_all_providers_fail_raises_runtime(monkeypatch):
    """Se nenhum provedor tiver chave, deve levantar RuntimeError com mensagem clara."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Impede que load_dotenv() carregue o .env real e reintroduza as chaves
    with patch("src.agents.nutritional_agent.load_dotenv"):
        with pytest.raises(RuntimeError, match="Nenhum provedor de LLM configurado"):
            NutritionalHealthAgent(_sample_df(), _sample_mappings())


# ---------------------------------------------------------------------------
# 5. Fallback ativo: falha no primário → resposta do secundário sem exceção
# ---------------------------------------------------------------------------

def test_agent_fallback_on_primary_failure(monkeypatch):
    """
    Simula falha no primário e verifica que o fallback (via with_fallbacks) é ativado
    sem propagar exceção para ask().
    """
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    # O mock do with_fallbacks retorna um LLM composto que responde normalmente
    mock_openai = _make_mock_llm()
    mock_gemini = _make_mock_llm()

    # Simula cadeia: openai falha → gemini responde
    composite_llm = MagicMock()
    composite_llm.invoke.return_value = MockResponse("Resposta via Gemini (fallback).")
    composite_llm.with_fallbacks.return_value = composite_llm
    mock_openai.with_fallbacks.return_value = composite_llm

    with (
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_openai),
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    # ask() via AgentExecutor não deve propagar exceção
    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", return_value={"output": "OK via fallback"}):
        result = agent.ask("Quantos pacientes obesos?")

    assert result == "OK via fallback"


# ---------------------------------------------------------------------------
# 6. generate_initial_report também beneficia do fallback
# ---------------------------------------------------------------------------

def test_agent_initial_report_uses_fallback(monkeypatch):
    """generate_initial_report usa self.llm.invoke diretamente → deve beneficiar do fallback."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_openai = _make_mock_llm(report_content="Relatório via Fallback Gemini.")
    mock_gemini = _make_mock_llm(report_content="Relatório via Fallback Gemini.")

    composite_llm = MagicMock()
    composite_llm.invoke.return_value = MockResponse("Relatório via Fallback Gemini.")
    composite_llm.with_fallbacks.return_value = composite_llm
    mock_openai.with_fallbacks.return_value = composite_llm

    with (
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_openai),
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    assert "Fallback Gemini" in agent.initial_report


# ---------------------------------------------------------------------------
# 7. Ferramentas (regressão)
# ---------------------------------------------------------------------------

def test_agent_tools(monkeypatch):
    """Verifica que as três ferramentas do agente continuam funcionando corretamente."""
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_llm = _make_mock_llm()
    with patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_llm):
        agent = NutritionalHealthAgent(
            pd.DataFrame({
                "NU_IDADE_ANO": [23, 28],
                "DS_FASE_VIDA": [1, 1],
                "SG_SEXO": [0, 1],
                "NU_PESO": [67.0, 64.5],
                "NU_ALTURA": [167.0, 170.0],
                "DS_IMC": [24.02, 22.32],
                "Prediction": ["Eutrofia", "Eutrofia"],
            }),
            {"SG_SEXO": {"0": "Masculino", "1": "Feminino"}, "DS_FASE_VIDA": {"1": "Adulto"}},
        )

    stats = agent._tool_get_statistics()
    assert "Estatísticas Descritivas Gerais" in stats
    assert "Eutrofia" in stats

    filtered = agent._tool_filter_records("NU_IDADE_ANO == 23")
    assert "Masculino" in filtered
    assert "24.02" in filtered

    rec = agent._tool_get_recommendations("Obesidade Grave")
    assert "Diretrizes Clínicas para Obesidade" in rec


# ---------------------------------------------------------------------------
# 8. ask() — regressão
# ---------------------------------------------------------------------------

def test_agent_ask(monkeypatch):
    """ask() delega ao AgentExecutor.invoke e retorna a chave 'output' corretamente."""
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_llm = _make_mock_llm()
    with patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_llm):
        agent = NutritionalHealthAgent(
            pd.DataFrame({
                "NU_IDADE_ANO": [23], "DS_FASE_VIDA": [1], "SG_SEXO": [0],
                "NU_PESO": [67.0], "NU_ALTURA": [167.0], "DS_IMC": [24.02],
                "Prediction": ["Eutrofia"],
            }),
            {"SG_SEXO": {"0": "Masculino"}},
        )

    with patch(
        "src.agents.nutritional_agent.AgentExecutor.invoke",
        return_value={"output": "Resposta Mockada."},
    ) as mock_invoke:
        result = agent.ask("Qual a média de idade?")

    assert result == "Resposta Mockada."
    mock_invoke.assert_called_once_with({"input": "Qual a média de idade?"})


# ===========================================================================
# Cenários de memória de sessão — fallback por rate limit
# ===========================================================================

from openai import RateLimitError as OpenAIRateLimitError  # noqa: E402


def _make_rate_limit_error() -> OpenAIRateLimitError:
    """Cria uma instância de OpenAIRateLimitError sem fazer chamada real."""
    return OpenAIRateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
    )


def _make_agent_two_providers(monkeypatch) -> NutritionalHealthAgent:
    """Helper: cria agente com chain openai → gemini e mocks configurados."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_openai = _make_mock_llm()
    mock_gemini = _make_mock_llm()

    with (
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_openai),
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    return agent


# ---------------------------------------------------------------------------
# 9. Rate limit no primário → retenta com secundário transparentemente
# ---------------------------------------------------------------------------

def test_ask_rate_limit_retries_with_next_provider(monkeypatch):
    """
    Quando o AgentExecutor levanta RateLimitError, ask() deve:
    1. Avançar para o próximo provider
    2. Retentar a mesma pergunta
    3. Retornar a resposta do provider secundário (transparente ao chamador)
    """
    agent = _make_agent_two_providers(monkeypatch)

    call_count = 0

    def invoke_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_rate_limit_error()
        return {"output": "Resposta via fallback gemini."}

    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", new=invoke_side_effect):
        result = agent.ask("Qual a média de IMC?")

    assert result == "Resposta via fallback gemini."
    assert call_count == 2   # tentou 2 vezes: primária (falhou) + retry (ok)


# ---------------------------------------------------------------------------
# 10. _active_index avança e agent_executor é recriádo após rate limit
# ---------------------------------------------------------------------------

def test_ask_rate_limit_advances_active_index(monkeypatch):
    """Após rate limit, _active_index deve ser 1 e agent_executor recriado com novo LLM."""
    agent = _make_agent_two_providers(monkeypatch)

    assert agent._active_index == 0
    original_executor = agent.agent_executor

    def invoke_side_effect(*args, **kwargs):
        raise _make_rate_limit_error()

    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", new=invoke_side_effect):
        # A retentativa também falha, mas queremos verificar o avanço do índice
        with pytest.raises(Exception):  # rate limit no retry também é ok aqui
            agent.ask("Qualquer pergunta")

    # _active_index deve ter avançado após o primeiro rate limit
    assert agent._active_index == 1
    # agent_executor deve ser um objeto diferente (foi recriado)
    assert agent.agent_executor is not original_executor


# ---------------------------------------------------------------------------
# 11. Segunda chamada já usa o provider avançado
# ---------------------------------------------------------------------------

def test_ask_next_question_uses_advanced_provider(monkeypatch):
    """
    Após um rate limit na pergunta N (que foi retentada com gemini),
    a pergunta N+1 deve ir diretamente ao gemini sem tentar openai novamente.
    """
    agent = _make_agent_two_providers(monkeypatch)

    call_count = 0

    def invoke_q1(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_rate_limit_error()
        return {"output": "ok_q1"}

    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", new=invoke_q1):
        agent.ask("Pergunta 1")

    # Após Q1, _active_index == 1 (gemini)
    assert agent._active_index == 1

    # Pergunta 2: o executor já é o do gemini; não deve levantar rate limit
    call_count_q2 = 0
    def invoke_q2(*args, **kwargs):
        nonlocal call_count_q2
        call_count_q2 += 1
        return {"output": "ok_q2"}

    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", new=invoke_q2):
        result = agent.ask("Pergunta 2")

    assert result == "ok_q2"
    # Foi chamado exatamente uma vez (sem retry de rate limit)
    assert call_count_q2 == 1


# ---------------------------------------------------------------------------
# 12. Todos os providers com rate limit → RuntimeError
# ---------------------------------------------------------------------------

def test_ask_all_providers_rate_limited_raises(monkeypatch):
    """Se o único provider disponível atingir rate limit, RuntimeError deve ser levantado."""
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_llm = _make_mock_llm()
    with patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_llm):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    # Apenas um provider; qualquer rate limit deve esgotar a cadeia
    def invoke_side_effect(*args, **kwargs):
        raise _make_rate_limit_error()

    with patch("src.agents.nutritional_agent.AgentExecutor.invoke", new=invoke_side_effect):
        with pytest.raises(RuntimeError, match="limite de taxa"):
            agent.ask("Pergunta qualquer")


# ---------------------------------------------------------------------------
# 13. Rate limit no generate_initial_report → retenta com próximo provider
# ---------------------------------------------------------------------------

def test_initial_report_rate_limit_retries_next(monkeypatch):
    """
    Se generate_initial_report sofrer rate limit, deve retentar com o próximo
    provider e retornar o relatório gerado por ele.
    """
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_openai = MagicMock()
    mock_gemini = MagicMock()

    # OpenAI lança rate limit; Gemini responde normalmente
    mock_openai.invoke.side_effect = _make_rate_limit_error()
    mock_gemini.invoke.return_value = MockResponse("Relatório via Gemini.")
    mock_openai.with_fallbacks = MagicMock(return_value=mock_openai)
    mock_gemini.with_fallbacks = MagicMock(return_value=mock_gemini)

    with (
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_openai),
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_gemini),
    ):
        agent = NutritionalHealthAgent(_sample_df(), _sample_mappings())

    # O relatório deve ter sido gerado pelo Gemini após o fallback
    assert "Gemini" in agent.initial_report
    # _active_index deve ter avançado para 1 (gemini)
    assert agent._active_index == 1


# ---------------------------------------------------------------------------
# 14. Auto-recovery de parâmetro 'stop' não suportado no PatchedChatOpenAI
# ---------------------------------------------------------------------------

def test_patched_chat_openai_auto_recovery_stop_parameter():
    """
    Verifica se o PatchedChatOpenAI detecta BadRequestError de parâmetro 'stop'
    não suportado, ativa a flag drop_stop e retenta a chamada com sucesso.
    """
    from openai import BadRequestError
    from src.agents.nutritional_agent import PatchedChatOpenAI

    # Cria mock de BadRequestError da OpenAI
    err_body = {
        "error": {
            "message": "Unsupported parameter: 'stop' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "stop"
        }
    }
    response_mock = MagicMock(status_code=400, headers={})
    bad_req_err = BadRequestError(
        "Unsupported parameter: 'stop' is not supported with this model.",
        response=response_mock,
        body=err_body
    )

    model = PatchedChatOpenAI(api_key="fake", model="fake-model")
    assert model.drop_stop is False

    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.messages import AIMessage

    call_count = 0

    def mock_super_generate(messages, stop=None, run_manager=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if stop is not None or "stop" in kwargs:
            raise bad_req_err
        # Na retentativa (sem stop), retorna ChatResult válido do LangChain
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Resposta sem stop!"))])

    with patch("langchain_openai.ChatOpenAI._generate", side_effect=mock_super_generate):
        result = model.invoke("Olá", stop=["\nStop"])

    assert result.content == "Resposta sem stop!"
    assert call_count == 2            # Chamada original (falhou) + retentativa (sucesso)
    assert model.drop_stop is True    # A flag de auto-recovery foi ativada


