import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Fallback e Rate Limit Exceptions — importadas com guard para evitar falha se os pacotes não estiverem instalados
from openai import (
    RateLimitError as OpenAIRateLimitError,
    APIConnectionError as OpenAIConnectionError,
    APITimeoutError as OpenAITimeoutError,
    InternalServerError as OpenAIInternalServerError,
    BadRequestError as OpenAIBadRequestError,
)
try:
    from google.api_core.exceptions import GoogleAPICallError
except ImportError:
    GoogleAPICallError = None  # type: ignore[assignment,misc]

from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

# Tupla unificada de exceções que disparam transição de provider (rate limit, indisponibilidade ou rede)
_FALLBACK_EXCEPTIONS: tuple = tuple(filter(None, [
    OpenAIRateLimitError,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIInternalServerError,
    GoogleAPICallError,
]))

# Logger deve ser declarado antes de PatchedChatOpenAI para evitar NameError no bloco de auto-recovery
logger = logging.getLogger(__name__)

class PatchedChatOpenAI(ChatOpenAI):
    """
    Subclasse de ChatOpenAI que remove o parâmetro 'stop' se configurado
    por variáveis de ambiente, para evitar erros em modelos que não o suportam.

    NOTA TÉCNICA: O LangChain ReAct Agent injeta 'stop' via _get_request_payload,
    que é chamado internamente por _generate e _stream. Portanto, para remover
    o parâmetro 'stop' do payload da API de forma efetiva, precisamos sobrescrever
    _get_request_payload — não apenas _generate.

    Fluxo de auto-recovery:
    1. Na primeira chamada, a flag drop_stop=False. O stop passa normalmente.
    2. A API rejeita com BadRequestError ('stop' not supported).
    3. _generate intercepta o erro, ativa drop_stop=True e retenta.
    4. A retentativa chama _get_request_payload que agora remove o stop.
    5. Todas as chamadas futuras já omitem o stop via _get_request_payload.
    """
    drop_stop: bool = False

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Sobrescreve o payload de requisição para remover 'stop' quando drop_stop=True."""
        if self.drop_stop:
            stop = None
            kwargs.pop("stop", None)
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Intercepta BadRequestError de 'stop' e retenta com drop_stop=True."""
        if self.drop_stop:
            stop = None
            kwargs.pop("stop", None)
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except OpenAIBadRequestError as exc:
            msg = str(exc).lower()
            if "stop" in msg and ("parameter" in msg or "not supported" in msg or "unsupported" in msg):
                logger.warning(
                    "O modelo '%s' indicou que o parâmetro 'stop' não é suportado. "
                    "Ativando auto-recovery (drop_stop=True) e retentando a chamada...",
                    self.model_name,
                )
                self.drop_stop = True
                kwargs.pop("stop", None)
                return super()._generate(messages, stop=None, run_manager=run_manager, **kwargs)
            raise exc

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

Mapeamentos de Colunas e Categorias na Base (Mapeamentos do Encoder):
{mappings_context}

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

        # Inicializa o LLM com suporte a fallback multi-provider
        self.llm = self._build_llm_with_fallbacks()

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

    @classmethod
    def from_files(cls, csv_path: str, mappings_path: str):
        """
        Inicializa o agente a partir de arquivos CSV e JSON de mapeamentos salvos.

        Args:
            csv_path (str): Caminho para o arquivo CSV com dados processados.
            mappings_path (str): Caminho para o arquivo JSON com mapeamentos.

        Returns:
            NutritionalHealthAgent: Instância do agente inicializada.
        """
        csv_path = Path(csv_path)
        mappings_path = Path(mappings_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")
        if not mappings_path.exists():
            raise FileNotFoundError(f"Arquivo de mapeamentos não encontrado: {mappings_path}")

        # Carregar CSV
        df = pd.read_csv(csv_path)

        # Carregar mapeamentos JSON
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        return cls(df, mappings)

    def _build_llm_provider(self, provider: str) -> BaseChatModel:
        """
        Instancia um único ChatModel a partir do nome do provedor,
        lendo as variáveis de ambiente correspondentes.

        Args:
            provider (str): Nome do provedor ('gemini', 'openai', 'openrouter').

        Returns:
            BaseChatModel: Instância do modelo de linguagem.

        Raises:
            ValueError: Se a chave de API não estiver configurada ou o provedor for desconhecido.
        """
        provider = provider.strip().lower()

        if provider == "gemini":
            # Tenta GEMINI_API_KEY; retrocompatibilidade: cai para LLM_API_KEY
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "Provedor 'gemini' requer GEMINI_API_KEY (ou LLM_API_KEY para retrocompatibilidade)."
                )
            model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-2.5-flash")
            temp_val = os.getenv("GEMINI_TEMPERATURE") or os.getenv("LLM_TEMPERATURE", "0.7")
            # Expõe a chave como GOOGLE_API_KEY para que o LangChain acesse
            os.environ["GOOGLE_API_KEY"] = api_key
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=float(temp_val),
                google_api_key=api_key,
                max_retries=1,
            )

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Provedor 'openai' requer OPENAI_API_KEY.")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            temp_val = os.getenv("OPENAI_TEMPERATURE", "0.7")
            drop_stop = os.getenv("OPENAI_DROP_STOP", "false").lower() == "true"
            return PatchedChatOpenAI(
                model=model,
                temperature=float(temp_val),
                api_key=api_key,
                drop_stop=drop_stop,
                max_retries=1,
            )

        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("Provedor 'openrouter' requer OPENROUTER_API_KEY.")
            model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            temp_val = os.getenv("OPENROUTER_TEMPERATURE", "0.7")
            drop_stop = os.getenv("OPENROUTER_DROP_STOP", "false").lower() == "true"
            return PatchedChatOpenAI(
                model=model,
                temperature=float(temp_val),
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                drop_stop=drop_stop,
                max_retries=1,
            )

        raise ValueError(f"Provedor desconhecido: '{provider}'. Use 'gemini', 'openai' ou 'openrouter'.")

    def _build_llm_with_fallbacks(self) -> BaseChatModel:
        """
        Constrói a lista de provedores LLM disponíveis e retorna o primário.

        Lê LLM_PROVIDER_ORDER do ambiente (ex: 'openai,gemini,openrouter').
        Se não definido, usa ['gemini'] para retrocompatibilidade.
        Provedores sem chave configurada são ignorados com aviso de log.

        Popula self._built_llms e self._active_index para uso pelo mecanismo
        de fallback com memória de sessão em ask() e generate_initial_report().

        Returns:
            BaseChatModel: LLM do provider primário (sem RunnableWithFallbacks —
            o fallback é gerenciado manualmente via _advance_provider).

        Raises:
            RuntimeError: Se nenhum provedor puder ser construído.
        """
        provider_order_env = os.getenv("LLM_PROVIDER_ORDER", "gemini")
        providers = [p.strip() for p in provider_order_env.split(",") if p.strip()]

        self._built_llms: List[BaseChatModel] = []
        self._active_index: int = 0

        for provider in providers:
            try:
                llm = self._build_llm_provider(provider)
                self._built_llms.append(llm)
            except ValueError as exc:
                logger.warning("Provedor '%s' ignorado: %s", provider, exc)

        if not self._built_llms:
            raise RuntimeError(
                "Nenhum provedor de LLM configurado corretamente — verifique as API keys. "
                f"Provedores tentados: {providers}"
            )

        resolved_names = providers[: len(self._built_llms)]
        logger.info("LLM session chain: %s", " -> ".join(resolved_names))

        # Retorna apenas o primário — o fallback é gerenciado pela sessão
        return self._built_llms[0]

    def _advance_provider(self) -> bool:
        """
        Avança permanentemente para o próximo provider disponível nesta sessão.

        Deve ser chamado apenas após um erro de rate limit. O provider descartado
        nunca mais é tentado na mesma sessão (instância do agente).
        Atualiza self.llm e reconstrói self.agent_executor com o novo LLM.

        Returns:
            bool: True se havia um próximo provider; False se a cadeia se esgotou.
        """
        next_index = self._active_index + 1
        if next_index >= len(self._built_llms):
            logger.error(
                "Rate limit atingido no último provider disponível (índice %d). "
                "Nenhum fallback restante na sessão.",
                self._active_index,
            )
            return False

        self._active_index = next_index
        self.llm = self._built_llms[self._active_index]
        # Reconstrói o AgentExecutor pois create_react_agent bake o LLM no objeto
        if hasattr(self, "tools"):
            self.agent_executor = self._setup_agent()
        logger.warning(
            "Provider demovido por rate limit. Sessão agora usa provider índice %d.",
            self._active_index,
        )
        return True

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
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except _FALLBACK_EXCEPTIONS as exc:
            if self._advance_provider():
                logger.warning(
                    "Falha ou rate limit no relatório inicial; retentando com provider índice %d.",
                    self._active_index,
                )
                response = self.llm.invoke(prompt)
                return response.content
            raise RuntimeError(
                "Erro ou rate limit em todos os providers ao gerar o relatório inicial."
            ) from exc

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
        # Limpa apenas um nível de aspas externas redundantes (iguais) que o modelo às vezes coloca
        query_clean = query.strip()
        for quote in ["'", '"', '`']:
            if query_clean.startswith(quote) and query_clean.endswith(quote) and len(query_clean) >= 2:
                query_clean = query_clean[1:-1]
                break
        try:
            print(f"Filtrando registros com query: {query_clean}")
            result = self.df.query(query_clean)
            if result.empty:
                return f"Nenhum registro encontrado para o filtro: {query_clean}"
            # Limita a resposta às primeiras 30 linhas para não estourar o contexto do agente
            return f"Registros encontrados (exibindo até 30 de {len(result)} resultados):\n" + result.head(30).to_markdown()
        except Exception as e:
            return f"Erro ao executar o filtro '{query_clean}': {str(e)}. Por favor, utilize a sintaxe de query do pandas (ex: `SG_SEXO == 'M' & NU_IDADE_ANO > 30`)."

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
        mappings_lines = []
        for col, values in self.mappings.items():
            mappings_lines.append(f"- Coluna '{col}':")
            for k, v in values.items():
                mappings_lines.append(f"  O valor '{k}' na base representa a categoria '{v}'")
        
        mappings_str = "\n".join(mappings_lines) if mappings_lines else "Nenhum mapeamento categórico disponível."
        custom_template = REACT_PROMPT_TEMPLATE.replace("{mappings_context}", mappings_str)

        prompt = PromptTemplate(
            template=custom_template,
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

        Em caso de rate limit do provider ativo:
        - O provider é descartado permanentemente nesta sessão.
        - A pergunta é retentada automaticamente com o próximo provider.
        - Se todos os providers estiverem esgotados, levanta RuntimeError.
        """
        try:
            response = self.agent_executor.invoke({"input": question})
            return response["output"]
        except _FALLBACK_EXCEPTIONS as exc:
            if self._advance_provider():
                logger.warning(
                    "Retentando pergunta com provider índice %d após falha de API ou rate limit.",
                    self._active_index,
                )
                response = self.agent_executor.invoke({"input": question})
                return response["output"]
            raise RuntimeError(
                "Todos os provedores de LLM falharam ou atingiram o limite de taxa. "
                "Tente novamente mais tarde."
            ) from exc
