# Fallback Multi-Provider para `NutritionalHealthAgent`

> **Status:** ✅ Implementado e testado (8/8 testes passando)  
> **Spec de referência:** [`spec_fallback_llm_nutritional_agent.md`](./spec_fallback_llm_nutritional_agent.md)

Adicionar resiliência ao `NutritionalHealthAgent` com suporte a múltiplos provedores de LLM via `RunnableWithFallbacks` do LangChain. Caso o provedor principal falhe (rate limit, indisponibilidade, erro de rede), o agente tenta automaticamente os próximos provedores configurados — sem impacto nas tools, na memória ou no prompt ReAct existente.

---

## Decisões de design

| Questão | Decisão |
|---|---|
| Exceções que disparam fallback | `with_fallbacks()` sem `exceptions_to_handle` — LangChain captura qualquer `Exception` |
| Gemini sem `GEMINI_API_KEY` | Tenta `LLM_API_KEY` como retrocompatibilidade antes de lançar `ValueError` |

---

## Alterações realizadas

### 1. Dependências

#### [MODIFY] [`pyproject.toml`](../backend/pyproject.toml)

Adicionada `langchain-openai>=0.2.0,<1.0.0` (cobre tanto OpenAI quanto OpenRouter, que usa interface OpenAI-compatível):

```diff
  "langchain-google-genai>=2.0.0,<3.0.0",
+ "langchain-openai>=0.2.0,<1.0.0",
```

---

### 2. Agente principal

#### [MODIFY] [`nutritional_agent.py`](../backend/src/agents/nutritional_agent.py)

**2.1. Novos imports e logger de módulo**

```python
import logging
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)
```

**2.2. Novo método `_build_llm_provider(provider: str) -> BaseChatModel`**

Instancia um único `ChatModel` com base no nome do provedor, lendo as variáveis de ambiente correspondentes. Lança `ValueError` se a chave não estiver configurada.

| provider | Classe | Env vars lidas |
|---|---|---|
| `gemini` | `ChatGoogleGenerativeAI` | `GEMINI_API_KEY` (fallback: `LLM_API_KEY`), `GEMINI_MODEL` (fallback: `LLM_MODEL`), `GEMINI_TEMPERATURE` |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE` |
| `openrouter` | `ChatOpenAI` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE` + `base_url="https://openrouter.ai/api/v1"` |

**2.3. Novo método `_build_llm_with_fallbacks() -> BaseChatModel`**

1. Lê `LLM_PROVIDER_ORDER` (default: `"gemini"` para retrocompatibilidade).
2. Para cada provedor, chama `_build_llm_provider`. Se falhar (chave ausente), loga `WARNING` e passa para o próximo.
3. Se nenhum provedor for construído, levanta `RuntimeError` com mensagem clara.
4. Constrói a cadeia: `primary.with_fallbacks(fallbacks)` (sem `exceptions_to_handle`).
5. Loga `INFO` com a cadeia resolvida (ex: `"LLM fallback chain: openai -> gemini -> openrouter"`).

**2.4. Alteração em `__init__`**

Substituir o bloco hardcoded do Gemini:
```python
# ANTES
api_key = os.getenv("LLM_API_KEY")
model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
temperature_val = os.getenv("LLM_TEMPERATURE", "0.7")
...
self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
```

por:
```python
# DEPOIS
self.llm = self._build_llm_with_fallbacks()
```

As demais linhas do `__init__` (`_decode_dataframe`, `generate_initial_report`, `_setup_tools`, `_setup_agent`, `memory`) **permanecem inalteradas** — todas já consomem `self.llm` pronto.

---

### 3. Variáveis de ambiente

#### [MODIFY] [`.env.example`](../backend/.env.example)

Substituído o bloco `LLM Configuration` pelo novo esquema por provedor, com retrocompatibilidade:

```env
# ── LLM Configuration ─────────────────────────────────────────────────────────
# Ordem dos provedores de LLM (separados por vírgula). O primeiro é o principal.
# Se omitido, usa apenas Gemini com LLM_API_KEY/LLM_MODEL (retrocompatibilidade).
# LLM_PROVIDER_ORDER=gemini,openai,openrouter

# Retrocompatibilidade: usado quando LLM_PROVIDER_ORDER não está definido,
# ou quando 'gemini' está na ordem mas GEMINI_API_KEY não foi configurada.
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.7

# --- Google Gemini ---
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7

# --- OpenAI ---
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o
# OPENAI_TEMPERATURE=0.7

# --- OpenRouter ---
# OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
# OPENROUTER_TEMPERATURE=0.7
```

---

### 4. Testes

#### [MODIFY] [`test_llm_agent.py`](../backend/tests/unit/test_llm_agent.py)

8 cenários cobrindo todos os critérios de aceitação da spec:

| Teste | Critério |
|---|---|
| `test_agent_initialization_default_gemini` | Regressão zero: sem `LLM_PROVIDER_ORDER`, usa `LLM_API_KEY` |
| `test_agent_initialization_with_fallback_chain` | Cadeia com `.with_fallbacks()` quando ambas as chaves existem |
| `test_agent_missing_provider_key_skipped` | Provedor sem chave ignorado com `WARNING`, inicialização não falha |
| `test_agent_all_providers_fail_raises_runtime` | Sem nenhuma chave válida → `RuntimeError` claro |
| `test_agent_fallback_on_primary_failure` | Falha no primário → resposta do secundário sem exceção em `ask()` |
| `test_agent_initial_report_uses_fallback` | `generate_initial_report` beneficia do fallback via `self.llm.invoke` |
| `test_agent_tools` | Regressão: ferramentas continuam funcionando |
| `test_agent_ask` | Regressão: `ask()` delega corretamente ao `AgentExecutor` |

---

## Resultado da verificação

```
uv run pytest tests/unit/test_llm_agent.py -v
========================= 8 passed, 1 warning in 2.92s =========================
```

> O warning (`LangChainDeprecationWarning: ConversationBufferMemory`) é pré-existente no projeto e está fora do escopo desta tarefa.

---

## Considerações adicionais (não bloqueantes — v2+)

- **OpenRouter com `models` array interno:** `model_kwargs={"extra_body": {"models": [...]}}` adiciona segunda camada de resiliência sem custo de código.
- **README:** documentar que modelos de fallback gratuitos (`:free`) podem gerar relatórios clínicos de qualidade inferior — usar apenas como último recurso.
- **Rate limit OpenRouter free tier:** 20 req/min — adequado apenas como fallback de baixo volume.
