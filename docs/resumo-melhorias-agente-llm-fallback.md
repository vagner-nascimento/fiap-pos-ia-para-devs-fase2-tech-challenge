# Resumo de Melhorias e Decisões de Design — Agente LLM & Fallback Stateful

Este documento consolida as principais discussões, decisões técnicas e melhorias implementadas no `NutritionalHealthAgent` e na API do backend para fornecer alta resiliência e tratamento correto de erros na camada de LLM e ferramentas.

---

## 1. Fallback Multi-Provedor Stateful (Com Memória de Sessão)

### Discussão e Problema
O LangChain fornece o mecanismo `RunnableWithFallbacks` (via `.with_fallbacks()`), mas ele é **stateless** por design. A cada pergunta (`ask()`), ele sempre começa a tentar pelo primeiro provedor da lista. Se o provedor primário estivesse sofrendo rate limit, a aplicação ficaria travando a cada nova pergunta tentando o primário antes de cair para o fallback.

### Decisões e Solução
- **Estado de Sessão Manual:** Removemos o `RunnableWithFallbacks` e passamos a gerenciar o provedor ativo na própria instância do agente através dos atributos `_built_llms` e `_active_index`.
- **Avanço Permanente:** Quando o provedor ativo falha com um erro de cota ou rede persistente, o método `_advance_provider()` é invocado, promovendo o próximo modelo da cadeia permanentemente para a sessão ativa.
- **Reconstrução Dinâmica:** O `AgentExecutor` é reconstruído dinamicamente no avanço com o novo LLM (protegido por um check `hasattr(self, 'tools')` para evitar falhas durante o `__init__` no relatório inicial).
- **Retry Transparente:** A pergunta que sofreu a falha é retentada uma única vez de forma automática e transparente com o novo provedor antes de reportar qualquer erro ao usuário.
- **Resposta HTTP 503:** Caso todos os provedores se esgotem, a rota FastAPI `/chat` intercepta o erro e retorna um status **HTTP 503 (Service Unavailable)** em vez de HTTP 500 genérico.

---

## 2. Otimização de Retentativas (max_retries) e Erros de Rede

### Discussão e Problema
O agente continuava fazendo 5 tentativas lentas com backoff longo no Google Gemini antes de propagar o erro `ResourceExhausted` (rate limit) para a nossa camada de fallback de sessão. 

### Decisões e Solução
- **max_retries=1:** Configuramos `max_retries=1` em todos os construtores de provedores (`ChatGoogleGenerativeAI` e `PatchedChatOpenAI`). Isso permite **uma única retentativa rápida** a nível de HTTP para instabilidades pontuais de rede, mas desiste de imediato no caso de rate limit persistente.
- **Fallback para Erros de Rede/Servidor:** Expandimos a tupla de captura de erros de `_RATE_LIMIT_EXCEPTIONS` para `_FALLBACK_EXCEPTIONS`. Agora, ela captura timeouts (`APITimeoutError`), falhas de conexão de rede (`APIConnectionError`) e erros 5xx de servidor (`InternalServerError` e `GoogleAPICallError`) de ambos os SDKs, acionando a mudança de provedor para qualquer indisponibilidade de API.

---

## 3. Correção na Limpeza de Queries do Pandas (Filtro de Registros)

### Discussão e Problema
A ferramenta `filter_nutrition_records` usava `.strip("'").strip('"')` para limpar aspas extras inseridas pelo LLM. Porém, quando o LLM enviava uma expressão envolta por aspas simples externas, como `'SG_SEXO == 'Masculino''`, o `.strip("'")` removia recursivamente as aspas da ponta direita, engolindo a aspa de fechamento interna e gerando a query truncada `SG_SEXO == 'Masculino` (causando o erro `unterminated string literal` no Pandas).

### Decisões e Solução
- Substituímos o `.strip()` por um loop de remoção de aspas emparelhadas. Apenas um nível de aspas externas idênticas (`'`, `"`, ou `` ` ``) é fatiado de ponta a ponta (`[1:-1]`), preservando inteiramente as aspas internas da query.

---

## 4. Ensino de Mapeamento Categórico no Prompt (SG_SEXO como M/F)

### Discussão e Problema
Na base original do SISVAN, a coluna `SG_SEXO` possui os valores `M` ou `F`. O agente, no entanto, tentava adivinhar os valores enviando a query `SG_SEXO == 'Masculino'`. O dicionário de mapeamentos do encoder converte de `0/1` para `F/M`, impossibilitando a tradução direta automática para categorias descritivas por limitações técnicas do pipeline do projeto.

### Decisões e Solução
- **Prompt Enriquecido:** Adicionamos a seção `Mapeamentos de Colunas e Categorias na Base` no prompt principal `REACT_PROMPT_TEMPLATE`.
- **Formatação Plana (Sem Chaves):** Para evitar que o parser do LangChain interpretasse as chaves `{}` do JSON de mappings como tags de f-string aninhadas inválidas (o que quebrava a inicialização do prompt no Pydantic), os mapeamentos são formatados estaticamente como uma lista descritiva plana de linhas no `_setup_agent`:
  ```
  - Coluna 'SG_SEXO':
    O valor '0' na base representa a categoria 'F'
    O valor '1' na base representa a categoria 'M'
  ```
  Isso ensina o agente de forma clara quais são as abreviações vigentes na base carregada sem quebrar o template.

---

## 5. Auto-Recovery Dinâmico do Parâmetro 'stop'

### Discussão e Problema
Modelos de raciocínio (como `o1`, `o3-mini`, `o1-mini` da OpenAI, seja diretamente ou via OpenRouter) não aceitam stop sequences em sua API, levantando erro HTTP 400 (`Unsupported parameter: 'stop'`). O LangChain ReAct Agent passa stop sequences por padrão.

### Decisões e Solução
- **PatchedChatOpenAI:** Implementamos a classe customizada `PatchedChatOpenAI` que herda de `ChatOpenAI`.
- **Auto-Recovery em Runtime:** Se a chamada à API falhar com `BadRequestError` e a mensagem contiver o termo `'stop'`, a classe intercepta o erro, exibe um `WARNING` no console, ativa a flag `self.drop_stop = True` e **refaz a chamada de forma transparente sem o parâmetro stop** na mesma hora. As chamadas seguintes usam a flag persistida para omitir o parâmetro de antemão.

---

## 6. Ferramenta de Validação e Testes Herméticos

- **validate_tools.py:** Criamos o utilitário interativo [`scripts/validate_tools.py`](../backend/scripts/validate_tools.py) que mocka o LLM no startup para permitir ao desenvolvedor testar estatísticas, filtros Pandas e recomendações clínicas localmente via console com custo zero de API.
- **mock_load_dotenv:** Adicionamos um fixture de autouse com patch em `load_dotenv` em [`test_llm_agent.py`](../backend/tests/unit/test_llm_agent.py) para garantir que arquivos `.env` locais com chaves de API reais do desenvolvedor não vazem/interfiram na execução das asserções de testes unitários do pipeline.
