from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.llm import NutritionalHealthAgent

# Mock response class for LangChain invoke
class MockResponse:
    def __init__(self, content: str):
        self.content = content


@patch('src.app.llm.ChatGoogleGenerativeAI')
def test_nutritional_health_agent_initialization(mock_chat_llm):
    # Set up mock behavior for LLM
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MockResponse("Relatório Inicial Gerado.")
    mock_chat_llm.return_value = mock_instance

    # Prepare sample data
    data = {
        "NU_IDADE_ANO": [23, 28, 39, 64],
        "DS_FASE_VIDA": [1, 1, 1, 6],
        "SG_SEXO": [0, 1, 0, 1],
        "NU_PESO": [67.0, 64.5, 108.0, 72.3],
        "NU_ALTURA": [167.0, 170.0, 162.0, 178.0],
        "DS_IMC": [24.02, 22.32, 41.15, 22.82],
        "Prediction": ["Eutrofia", "Eutrofia", "Obesidade Grave", "Eutrofia"]
    }
    df = pd.DataFrame(data)
    mappings = {
        "SG_SEXO": {"0": "Masculino", "1": "Feminino"},
        "DS_FASE_VIDA": {"1": "Adulto", "6": "Idoso"}
    }

    # Initialize agent
    agent = NutritionalHealthAgent(df, mappings)

    # Verify column decoding
    assert agent.df.loc[0, "SG_SEXO"] == "Masculino"
    assert agent.df.loc[1, "SG_SEXO"] == "Feminino"
    assert agent.df.loc[0, "DS_FASE_VIDA"] == "Adulto"
    assert agent.df.loc[3, "DS_FASE_VIDA"] == "Idoso"

    # Verify initial report call
    assert agent.initial_report == "Relatório Inicial Gerado."
    assert mock_instance.invoke.called


@patch('src.app.llm.ChatGoogleGenerativeAI')
def test_agent_tools(mock_chat_llm):
    # Set up mock behavior for LLM
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MockResponse("Relatório Inicial Gerado.")
    mock_chat_llm.return_value = mock_instance

    # Prepare sample data
    data = {
        "NU_IDADE_ANO": [23, 28],
        "DS_FASE_VIDA": [1, 1],
        "SG_SEXO": [0, 1],
        "NU_PESO": [67.0, 64.5],
        "NU_ALTURA": [167.0, 170.0],
        "DS_IMC": [24.02, 22.32],
        "Prediction": ["Eutrofia", "Eutrofia"]
    }
    df = pd.DataFrame(data)
    mappings = {
        "SG_SEXO": {"0": "Masculino", "1": "Feminino"},
        "DS_FASE_VIDA": {"1": "Adulto"}
    }

    agent = NutritionalHealthAgent(df, mappings)

    # Test get_nutrition_statistics tool
    stats_output = agent._tool_get_statistics()
    assert "Estatísticas Descritivas Gerais" in stats_output
    assert "Eutrofia" in stats_output

    # Test filter_nutrition_records tool
    filter_output = agent._tool_filter_records("NU_IDADE_ANO == 23")
    assert "Masculino" in filter_output
    assert "24.02" in filter_output

    # Test get_clinical_recommendations tool
    rec_output = agent._tool_get_recommendations("Obesidade Grave")
    assert "Diretrizes Clínicas para Obesidade / Obesidade Grave" in rec_output


@patch('src.app.llm.ChatGoogleGenerativeAI')
def test_agent_ask(mock_chat_llm):
    # Set up mock behavior for LLM
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MockResponse("Relatório Inicial Gerado.")
    mock_chat_llm.return_value = mock_instance

    # Prepare sample data
    data = {
        "NU_IDADE_ANO": [23, 28],
        "DS_FASE_VIDA": [1, 1],
        "SG_SEXO": [0, 1],
        "NU_PESO": [67.0, 64.5],
        "NU_ALTURA": [167.0, 170.0],
        "DS_IMC": [24.02, 22.32],
        "Prediction": ["Eutrofia", "Eutrofia"]
    }
    df = pd.DataFrame(data)
    mappings = {"SG_SEXO": {"0": "Masculino"}}

    agent = NutritionalHealthAgent(df, mappings)

    # Mock AgentExecutor.invoke using patch
    with patch('src.app.llm.AgentExecutor.invoke', return_value={"output": "Resposta Mockada."}) as mock_invoke:
        res = agent.ask("Qual a média de idade?")
        assert res == "Resposta Mockada."
        mock_invoke.assert_called_once_with({"input": "Qual a média de idade?"})
