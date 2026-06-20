import os
import pandas as pd
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Local custom prompt template for classic ReAct Agent
REACT_PROMPT_TEMPLATE = """Você é o Agente de Saúde Nutricional, um especialista em análise estatística de dados nutricionais e interpretação de predições de modelos de Aprendizado de Máquina (ML).
Você é atencioso, profissional e foca em gerar insights clínicos de alta qualidade.

O usuário carregou um conjunto de dados de pacientes. O resumo da base de dados e suas descrições de colunas e categorias decodificadas estão disponíveis para você.
Você deve usar esse contexto e as ferramentas disponíveis para responder às perguntas do usuário com precisão clínica e estatística.

Você tem acesso às seguintes ferramentas para auxiliá-lo a responder perguntas sobre os pacientes e as predições de estado nutricional:

{tools}

Para usar uma ferramenta, você DEVE seguir exatamente o seguinte formato (com "Thought", "Action", "Action Input" e "Observation"):

Thought: Você deve sempre pensar sobre o que fazer e qual passo tomar.
Action: A ação a ser tomada, deve ser uma das seguintes ferramentas: [{tool_names}]
Action Input: A entrada para a ação/ferramenta.
Observation: O resultado retornado pela ferramenta.

Esse ciclo de Thought/Action/Action Input/Observation pode se repetir quantas vezes forem necessárias até que você tenha informações suficientes.
Quando você tiver a resposta final, você DEVE responder no formato:

Thought: Eu sei a resposta final.
Final Answer: [sua resposta detalhada aqui em português]

Histórico de Conversação:
{chat_history}

Pergunta do Usuário: {input}
Thought: {agent_scratchpad}"""


class NutritionalHealthAgent:
    def __init__(self, df: pd.DataFrame, mappings: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Inicializa o agente de saúde nutricional com os dados de pacientes (DataFrame)
        e o dicionário de mapeamento para decodificação.

        Args:
            df (pd.DataFrame): DataFrame contendo os dados nutricionais.
            mappings (dict, optional): Dicionário de mapeamento para decodificação de categorias.
                Exemplo: {"SG_SEXO": {"0": "Masculino", "1": "Feminino"}}
        """
        # Carrega variáveis do arquivo .env
        load_dotenv()

        # Configurações do LLM
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        temperature_val = os.getenv("LLM_TEMPERATURE", "0.7")
        
        try:
            temperature = float(temperature_val)
        except (TypeError, ValueError):
            temperature = 0.7

        # Configura a chave do Google no environment para que o LangChain acesse
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Inicializa o modelo da Google
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

        self.raw_df = df.copy()
        self.mappings = mappings or {}

        # Decodifica as colunas do DataFrame com base nos mapeamentos
        self.df = self._decode_dataframe(self.raw_df, self.mappings)

        # Gera o relatório inicial descritivo sobre o estado nutricional dos dados
        self.initial_report = self.generate_initial_report()

        # Configura a memória local em buffer InMemory para o histórico de conversação
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False  # Mantém como string formatada para injetar no prompt
        )

        # Configura as ferramentas de análise de dados
        self.tools = self._setup_tools()

        # Configura o agente ReAct e seu executor
        self.agent_executor = self._setup_agent()

    def _decode_dataframe(self, df: pd.DataFrame, mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Decodifica valores categóricos codificados usando o dicionário fornecido.
        """
        decoded_df = df.copy()
        for col, mapping in mappings.items():
            if col in decoded_df.columns:
                # Converte os valores da coluna para string sem '.0' (caso sejam lidos como float)
                # para garantir compatibilidade com as chaves string do dicionário
                decoded_df[col] = (
                    decoded_df[col]
                    .astype(str)
                    .str.split('.')
                    .str[0]
                    .map(mapping)
                    .fillna(decoded_df[col])
                )
        return decoded_df

    def generate_initial_report(self) -> str:
        """
        Gera um relatório inicial descritivo do conjunto de dados utilizando o LLM.
        """
        summary_stats = self.df.describe(include='all').to_string()
        
        # Contagem das classes de predição do modelo de ML se existirem
        prediction_counts = "N/A"
        if 'Prediction' in self.df.columns:
            prediction_counts = self.df['Prediction'].value_counts().to_string()
        elif 'prediction' in self.df.columns:
            prediction_counts = self.df['prediction'].value_counts().to_string()

        sample_rows = self.df.head(10).to_markdown()

        prompt = f"""
Você é um Agente de Saúde Nutricional e analista clínico de dados de ML.
Aqui está a base de dados de estado nutricional que foi importada e decodificada para análise:

### Estatísticas Gerais da Base:
{summary_stats}

### Distribuição das Classes Preditas pelo Modelo de ML (Prediction):
{prediction_counts}

### Amostra dos Dados (Primeiras 10 linhas):
{sample_rows}

Por favor, analise esses dados e gere um relatório clínico inicial robusto e descritivo em português. 
O relatório deve cobrir:
1. Resumo demográfico da amostra (faixas etárias predominantes, distribuição de sexo, fases da vida).
2. Perfil antropométrico geral (médias de peso, altura e IMC).
3. Distribuição e diagnóstico das predições de estado nutricional (quantos pacientes eutróficos, obesos, etc.).
4. Principais insights clínicos, alertas sobre possíveis riscos à saúde nutricional encontrados e sugestões de foco de intervenção.
"""
        response = self.llm.invoke(prompt)
        return response.content

    def _setup_tools(self) -> List[Tool]:
        """
        Define as ferramentas que o agente ReAct pode utilizar para interagir com os dados.
        """
        return [
            Tool(
                name="get_nutrition_statistics",
                func=self._tool_get_statistics,
                description="Útil para obter estatísticas descritivas rápidas e distribuição das predições de estado nutricional na base de dados de pacientes."
            ),
            Tool(
                name="filter_nutrition_records",
                func=self._tool_filter_records,
                description="Útil para filtrar e consultar registros específicos dos pacientes utilizando sintaxe pandas. Exemplo de entrada: `Prediction == 'Obesidade Grave'` ou `NU_IDADE_ANO > 60`."
            ),
            Tool(
                name="get_clinical_recommendations",
                func=self._tool_get_recommendations,
                description="Útil para obter as diretrizes clínicas e recomendações nutricionais de referência baseadas no estado nutricional (ex: 'Obesidade Grave', 'Eutrofia', 'Baixo Peso')."
            )
        ]

    def _tool_get_statistics(self, query: str = "") -> str:
        """
        Função de ferramenta para gerar resumo descritivo.
        """
        desc = self.df.describe(include='all').to_string()
        pred_col = 'Prediction' if 'Prediction' in self.df.columns else ('prediction' if 'prediction' in self.df.columns else None)
        counts = self.df[pred_col].value_counts().to_string() if pred_col else "N/A"
        return f"Estatísticas Descritivas Gerais:\n{desc}\n\nDistribuição do Estado Nutricional Predito:\n{counts}"

    def _tool_filter_records(self, query: str) -> str:
        """
        Função de ferramenta para filtrar os pacientes baseados em expressões python pandas.
        """
        # Limpa aspas extras que o modelo às vezes coloca
        query_clean = query.strip().strip("'").strip('"').strip('`')
        try:
            result = self.df.query(query_clean)
            if result.empty:
                return f"Nenhum registro encontrado para o filtro: {query_clean}"
            # Limita a resposta às primeiras 30 linhas para não estourar o contexto do agente
            return f"Registros encontrados (exibindo até 30 de {len(result)} resultados):\n" + result.head(30).to_markdown()
        except Exception as e:
            return f"Erro ao executar o filtro '{query_clean}': {str(e)}. Por favor, utilize a sintaxe de query do pandas (ex: `SG_SEXO == 'Masculino' & NU_IDADE_ANO > 30`)."

    def _tool_get_recommendations(self, category: str) -> str:
        """
        Função de ferramenta para buscar diretrizes clínicas associadas a um diagnóstico.
        """
        category_clean = category.strip().strip("'").strip('"').lower()
        if "obesidade" in category_clean:
            return (
                "Diretrizes Clínicas para Obesidade / Obesidade Grave:\n"
                "- Encaminhamento prioritário para equipe multidisciplinar (médico endocrinologista, nutricionista clínico e psicólogo).\n"
                "- Terapia comportamental e de reeducação alimentar focada em restrição calórica leve a moderada adaptada individualmente.\n"
                "- Acompanhamento de comorbidades metabólicas e cardiovasculares associadas (perfil lipídico, glicemia, pressão arterial).\n"
                "- Incentivo seguro a atividades físicas."
            )
        elif "eutrofia" in category_clean:
            return (
                "Diretrizes Clínicas para Eutrofia:\n"
                "- Manutenção de hábitos alimentares saudáveis com foco em alimentos in natura e minimamente processados.\n"
                "- Incentivo a check-ups nutricionais periódicos preventivos.\n"
                "- Manutenção de nível de atividade física regular."
            )
        elif "sobrepeso" in category_clean or "pré-obesidade" in category_clean or "pre-obesidade" in category_clean:
            return (
                "Diretrizes Clínicas para Sobrepeso:\n"
                "- Intervenção preventiva precoce na dieta com redução de açúcares refinados, gorduras trans e alimentos ultraprocessados.\n"
                "- Aumento do consumo de fibras alimentares solúveis e insolúveis.\n"
                "- Promoção de estilo de vida ativo para evitar o ganho ponderal contínuo."
            )
        elif "baixo peso" in category_clean or "desnutri" in category_clean or "magreza" in category_clean:
            return (
                "Diretrizes Clínicas para Baixo Peso / Desnutrição:\n"
                "- Investigação clínica e laboratorial detalhada para identificar etiologias subjacentes (carências nutricionais, infecções ou distúrbios absortivos/psicológicos).\n"
                "- Implementação de plano alimentar hipercalórico e hiperproteico fracionado.\n"
                "- Monitoramento próximo e suplementação de micronutrientes deficientes."
            )
        else:
            return "Diretrizes Gerais: Promover alimentação equilibrada, variada, ingestão hídrica ideal e estilo de vida ativo em conformidade com as diretrizes do Guia Alimentar para a População Brasileira."

    def _setup_agent(self) -> AgentExecutor:
        """
        Configura o agente ReAct clássico e seu executor.
        """
        prompt = PromptTemplate(
            template=REACT_PROMPT_TEMPLATE,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"]
        )

        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def ask(self, question: str) -> str:
        """
        Envia uma pergunta ao agente ReAct, mantendo o histórico de conversa.
        """
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
