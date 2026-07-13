# Especificação: Fallback Multi-Provider para o `NutritionalHealthAgent`

## 1. Objetivo

Adicionar resiliência ao `NutritionalHealthAgent` para que, caso o provedor de LLM
principal (atualmente Google Gemini, via `ChatGoogleGenerativeAI`) falhe — por rate
limit, indisponibilidade, erro de rede ou erro de API — o agente tente
automaticamente outro(s) provedor(es) configurados (ex: OpenAI e/ou OpenRouter),
sem que o restante da aplicação (tools, memória, prompt ReAct) precise mudar.

**Fora de escopo:** trocar a arquitetura do agente (ReAct + `AgentExecutor`), a
lógica das tools, o `_decode_dataframe`, ou o formato do relatório inicial. O
fallback deve ser uma mudança isolada na camada de criação do LLM.

---

## 2. Situação atual

Hoje o LLM é criado uma única vez, de um único provedor, dentro de `__init__`:

```python
self.llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=temperature,
    google_api_key=api_key
)
```

Configuração vem de três variáveis de ambiente: `LLM_API_KEY`, `LLM_MODEL`,
`LLM_TEMPERATURE`. Não há tratamento de erro nem alternativa caso a chamada ao
Gemini falhe — o erro simplesmente propaga para `ask()` e derruba a interação.

---

## 3. Estratégia proposta

Usar o mecanismo **nativo do LangChain** para fallback entre modelos:
`RunnableWithFallbacks`, exposto pelo método `.with_fallbacks()` disponível em
qualquer `ChatModel` do LangChain (`ChatOpenAI`, `ChatGoogleGenerativeAI`, etc).

Isso evita reimplementar loops de try/except manuais (como fizemos nos exemplos
em Python puro) — o LangChain já resolve isso de forma declarativa e compatível
com `AgentExecutor`, `Tool`, e streaming.

```python
self.llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])
```

O objeto resultante se comporta como um `ChatModel` normal — `create_react_agent`
e `AgentExecutor` não precisam saber que existe fallback por trás.

### 3.1. Provedores suportados

| Provider     | Classe LangChain         | base_url custom                                              |
|--------------|---------------------------|---------------------------------------------------------------|
| OpenAI       | `ChatOpenAI`               | padrão (não precisa)                                          |
| Google Gemini| `ChatGoogleGenerativeAI` **ou** `ChatOpenAI` com base_url compatível | `https://generativelanguage.googleapis.com/v1beta/openai/`   |
| OpenRouter   | `ChatOpenAI`               | `https://openrouter.ai/api/v1`                                 |

> Nota: Gemini pode continuar usando `ChatGoogleGenerativeAI` nativo (como já está
> implementado) — não é obrigatório migrar para `ChatOpenAI`. A tabela mostra a
> alternativa apenas para referência caso um único cliente unificado seja
> preferido no futuro.

---

## 4. Configuração (variáveis de ambiente)

Substituir as 3 variáveis atuais (`LLM_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE`)
por uma configuração por provedor, mantendo retrocompatibilidade.

```env
# Ordem de fallback, separada por vírgula. O primeiro é o principal.
LLM_PROVIDER_ORDER=openai,gemini,openrouter

# --- OpenAI ---
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7

# --- Gemini ---
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7

# --- OpenRouter ---
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
OPENROUTER_TEMPERATURE=0.7

# Retrocompatibilidade: se LLM_PROVIDER_ORDER não existir,
# comportamento atual é preservado (somente Gemini, usando LLM_API_KEY/LLM_MODEL)
```

**Regra de retrocompatibilidade:** se `LLM_PROVIDER_ORDER` não estiver definida,
o agente deve se comportar exatamente como hoje (só Gemini, sem fallback), lendo
`LLM_API_KEY` / `LLM_MODEL` / `LLM_TEMPERATURE` como já faz. Isso evita quebrar
quem já usa o agente em produção.

**Regra de disponibilidade:** um provedor só entra na cadeia de fallback se sua
API key correspondente estiver presente no ambiente. Se `LLM_PROVIDER_ORDER`
incluir um provedor sem chave configurada, ele deve ser **ignorado com um aviso
de log**, não deve derrubar a inicialização do agente.

---

## 5. Alterações no código

### 5.1. Novo método: `_build_llm_provider(provider: str)`

Responsável por instanciar um único `ChatModel` a partir do nome do provedor,
lendo as variáveis de ambiente correspondentes. Deve levantar uma exceção clara
(`ValueError`) se a chave da API não estiver presente, para que
`_build_llm_with_fallbacks` possa capturar e pular esse provedor.

### 5.2. Novo método: `_build_llm_with_fallbacks() -> BaseChatModel`

1. Lê `LLM_PROVIDER_ORDER` (ou usa `["gemini"]` como default de retrocompatibilidade).
2. Para cada provedor da lista, tenta construir via `_build_llm_provider`.
   - Se falhar (chave ausente), loga aviso e continua para o próximo.
3. Se nenhum provedor puder ser construído, levanta `RuntimeError` explícito
   ("Nenhum provedor de LLM configurado corretamente — verifique as API keys").
4. Constrói a cadeia com `.with_fallbacks()`:
   ```python
   primary, *fallbacks = built_llms
   return primary.with_fallbacks(fallbacks) if fallbacks else primary
   ```

### 5.3. Alteração em `__init__`

Substituir o bloco atual:

```python
self.llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=temperature,
    google_api_key=api_key
)
```

por:

```python
self.llm = self._build_llm_with_fallbacks()
```

Nenhuma outra parte de `__init__` muda (`_decode_dataframe`, `generate_initial_report`,
`_setup_tools`, `_setup_agent`, `memory`, tudo permanece igual, pois todos
recebem `self.llm` já pronto).

### 5.4. Tratamento de exceções que disparam o fallback

Por padrão, `.with_fallbacks()` do LangChain captura **qualquer** `Exception`
levantada pela chamada ao modelo principal e tenta o próximo da lista. Para um
controle mais fino (recomendado em produção), especificar explicitamente quais
exceções devem disparar fallback via o parâmetro `exceptions_to_handle`:

```python
primary.with_fallbacks(
    fallbacks,
    exceptions_to_handle=(RateLimitError, APIError, APIConnectionError, APITimeoutError),
)
```

Isso evita mascarar bugs reais do próprio agente (ex: erro de parsing das tools)
como se fossem falha de provedor.

### 5.5. Logging / observabilidade

Adicionar log (nível `INFO`) sempre que:
- O agente é inicializado, mostrando a ordem de fallback resolvida
  (ex: `"LLM fallback chain: openai -> gemini -> openrouter"`).
- Uma chamada precisar recorrer ao fallback (o LangChain expõe isso via
  callback/`response_metadata`, ou pode ser inferido comparando o modelo
  esperado com o retornado, dependendo da versão do LangChain instalada).

Isso é importante clinicamente: se o agente está respondendo com um modelo mais
fraco (ex: `:free` do OpenRouter) por causa de uma falha em cascade, a equipe
deve conseguir perceber isso nos logs, já que a qualidade da análise clínica
pode variar entre modelos.

---

## 6. Compatibilidade com as tools existentes

Nenhuma tool (`get_nutrition_statistics`, `filter_nutrition_records`,
`get_clinical_recommendations`) precisa de alteração — elas dependem apenas do
`AgentExecutor`, que por sua vez depende apenas de `self.llm` ter a interface
padrão de `ChatModel`. Isso é validado pelo próprio `.with_fallbacks()`, que
retorna um objeto com a mesma interface (`invoke`, `stream`, `bind_tools` etc).

**Atenção com `create_react_agent`:** o agente ReAct atual usa parsing baseado em
texto (`Thought/Action/Action Input`), não tool calling nativo — então o
comportamento deve ser idêntico entre provedores, desde que cada modelo escolhido
consiga seguir o formato de prompt ReAct definido em `REACT_PROMPT_TEMPLATE`.
**Recomendação:** ao escolher modelos de fallback (principalmente gratuitos via
OpenRouter), validar manualmente se o modelo consegue seguir esse formato de
Thought/Action antes de colocá-lo em produção — nem todo modelo pequeno segue
bem instruções de formato rígidas.

---

## 7. Dependências novas

```
langchain-openai>=0.2.0
```

(`langchain-google-genai` já está no projeto; `langchain-openai` cobre tanto
OpenAI quanto OpenRouter, já que ambos são compatíveis com a interface OpenAI.)

---

## 8. Critérios de aceitação

1. Com apenas `GEMINI_API_KEY` configurada (sem `LLM_PROVIDER_ORDER`), o agente
   deve funcionar exatamente como antes (regressão zero).
2. Com `LLM_PROVIDER_ORDER=openai,gemini` e ambas as chaves presentes, uma falha
   simulada no OpenAI (ex: key inválida) deve resultar em resposta bem-sucedida
   vinda do Gemini, sem exceção propagada para `ask()`.
3. Se todos os provedores da cadeia falharem, `ask()` deve propagar um erro claro
   e identificável (não um erro genérico do LangChain de baixo nível).
4. Se um provedor da lista em `LLM_PROVIDER_ORDER` não tiver API key, a
   inicialização não deve falhar — apenas logar aviso e seguir sem ele.
5. O relatório inicial (`generate_initial_report`) também deve se beneficiar do
   fallback, já que usa `self.llm.invoke(prompt)` diretamente.

---

## 9. Considerações adicionais (não bloqueantes para a v1)

- **OpenRouter com `models` array interno:** o provedor OpenRouter, dentro da
  cadeia, pode por sua vez ter seu próprio fallback interno (ex:
  `extra_body={"models": [...]}`), adicionando uma segunda camada de resiliência
  sem custo adicional de código no agente. Pode ser adicionado depois, via
  `model_kwargs={"extra_body": {...}}` no `ChatOpenAI`.
- **Custo/qualidade:** documentar claramente em `README` do projeto que modelos
  de fallback gratuitos (`:free`) podem gerar relatórios clínicos de qualidade
  inferior — não usar como modelo principal em ambiente de produção clínica real,
  apenas como último recurso de disponibilidade.
- **Rate limit do OpenRouter free tier:** 20 req/min, 50-1000/dia — adequado
  apenas como fallback de baixo volume, não como camada primária.
