# Architecture Decision Records (ADR)

Registro das principais decisões de arquitetura e de algoritmo tomadas no projeto SISVAN Fase 2.
Cada ADR documenta o contexto, as alternativas consideradas e a justificativa da escolha feita.

---

## ADR-001 — Biblioteca de Algoritmo Genético: DEAP

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O projeto precisa de uma biblioteca de GA que suporte representação de indivíduos não-homogênea (RF e KNN possuem hiperparâmetros de tipos e ranges distintos) e que permita operadores customizados.

### Alternativas Consideradas
| Biblioteca | Prós | Contras |
|---|---|---|
| **DEAP** | Flexibilidade total; suporte a representações heterogêneas; amplamente usado em pesquisa | API mais complexa |
| `scikit-optimize` (BayesOpt) | Simples de usar | Sem suporte a população híbrida; sem crossover customizável |
| `optuna` | Moderno, ótima interface | Baseado em Bayesiana, não em GA; sem controle do loop evolutivo |
| Implementação própria | Controle total | Alto custo de desenvolvimento e manutenção |

### Decisão
**DEAP** — pelo suporte nativo a indivíduos customizados, operadores intercambiáveis (`cxUniform`, `selTournament`) e pela maturidade da biblioteca.

### Consequências
- Dependência adicionada: `deap>=1.4.0,<2.0.0`
- Operadores DEAP usados como referência, mas adaptados para trabalhar com `dict` em vez de listas posicionais

---

## ADR-002 — Representação do Indivíduo: Classe Python vs. Lista DEAP

**Data**: 2026-06  
**Status**: Aceito

### Contexto
DEAP tradicionalmente representa indivíduos como listas (`list`) com atributo `fitness`. Nosso espaço de hiperparâmetros é heterogêneo (inteiros, floats, categorias) e nomeado — parâmetros não são intercambiáveis por posição.

### Alternativas Consideradas
1. **Lista DEAP pura** com índices fixos para cada hiperparâmetro
2. **Classe Python customizada** com `dict` de hiperparâmetros

### Decisão
**Classe Python (`Individuo` ABC + subclasses)** com `hyperparams: dict`. Motivos:
- Hiperparâmetros são nomeados semanticamente (ex: `n_estimators`, não `params[0]`)
- Subclasses permitem encapsular a lógica de `build_model()` por tipo
- `IndividuoKNN` inclui `StandardScaler` automaticamente sem necessidade de lógica externa
- Melhor legibilidade e testabilidade (cada classe tem responsabilidade única)

### Consequências
- Operadores DEAP (`cxUniform`, `mutGaussian`) não são aplicáveis diretamente — adaptamos a semântica em `ga_operators.py`
- O atributo `fitness` do DEAP não é usado; usamos `fitness_values: tuple` direto no objeto

---

## ADR-003 — Crossover: cxUniform Adaptado para Dicts Nomeados

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O `cxUniform` do DEAP troca genes por índice (`ind1[i] ↔ ind2[i]`). Com hiperparâmetros em `dict`, não há índice posicional — os genes são identificados por chave.

### Alternativas Consideradas
1. **cxUniform do DEAP** — requer conversão dict→list, aplicação, list→dict. Propenso a erros de mapeamento.
2. **cxTwoPoint** — swap de bloco contíguo. Sem semântica para dicts.
3. **cxUniform adaptado** — para cada `key` do dict, sorteia com `prob=indpb` se troca ou não.

### Decisão
**cxUniform adaptado** implementado em `_uniform_crossover_dicts()` no `ga_operators.py`:
```python
for key in dict1:
    if random.random() < indpb:
        dict1[key], dict2[key] = dict2[key], dict1[key]
```

### Consequências
- Comportamento equivalente ao `cxUniform` do DEAP, mas semanticamente correto para dicts nomeados
- `indpb=0.0` → filhos idênticos aos pais (verificado em testes)
- `indpb=1.0` → troca total de parâmetros (verificado em testes)
- `indpb=0.5` → default — cada gene tem 50% de chance de ser trocado

> **Validação:** `TestCrossoverRF.test_indpb_half_swaps_approximately_half` e `TestCrossoverKNN.test_indpb_half_swaps_approximately_half` confirmam taxa média de swap ~0.49–0.51 em 200 repetições com seed fixa.

---

## ADR-004 — Populações Separadas com Competição Global (Co-Evolução)

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Precisamos otimizar hiperparâmetros de dois tipos de modelo (RF e KNN). As opções arquiteturais principais são:

1. **GA híbrido** — uma única população mista com crossover entre tipos
2. **GAs sequenciais** — dois loops independentes, sem interação
3. **GA co-evolutivo** — populações separadas, mas avaliação e seleção compartilhadas

### Decisão
**GA Co-Evolutivo** (opção 3):
- RF e KNN **não cruzam entre si** (hiperparâmetros não são intercambiáveis entre modelos)
- A **seleção por torneio é global** (pool = RF + KNN) — pressão competitiva real entre tipos
- Cada tipo gera filhos apenas dentro do próprio tipo

### Justificativa
- **Vs. GA híbrido**: Crossover entre RF e KNN não faz sentido semântico (não existe `n_estimators` em KNN)
- **Vs. GAs sequenciais**: A seleção global cria pressão competitiva dinâmica — o tipo mais fraco precisa evoluir ou é eliminado, sem custo computacional adicional
- Produz um resultado único e comparável ao final (melhor de RF vs. melhor de KNN)

### Consequências
- A proporção RF/KNN na população pode variar a cada geração (dinâmica emergente)
- **Elitismo estrutural mínimo**: garantia de ao menos 1 sobrevivente por tipo (evita extinção prematura)
- Saída unificada: `results["overall_best"]` é o vencedor global

> **Validação:** `TestStructuralElitism.test_min_one_survivor_per_type_every_generation` confirma `count >= 1` para RF e KNN em todas as gerações de um run com 6 gerações. `TestGlobalTournamentSelection.test_higher_fitness_wins_more_often` confirma pressão seletiva real: indivíduo com fitness 0.9 vence >25 de 50 torneios vs. fitness 0.1.

---

## ADR-005 — Fitness: F1-Weighted × 0.6 + Accuracy × 0.4

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O dataset SISVAN possui múltiplas classes de estado nutricional (Obesidade, Sobrepeso, Eutrofia, Baixo Peso, etc.) com possível desbalanceamento.

### Alternativas Consideradas
| Métrica | Problema |
|---|---|
| Apenas Acurácia | Favorece classes majoritárias; ignora desbalanceamento |
| Apenas F1-macro | Igual peso a todas as classes, inclusive as raríssimas |
| **F1-weighted** | Pondera pelo suporte de cada classe — adequado para desbalanceamento |
| F1-weighted × 0.6 + Acc × 0.4 | Mantém F1 como primário mas penaliza queda geral de acurácia |

### Decisão
**`fitness = F1_weighted × 0.6 + Accuracy × 0.4`**  
Implementado em `ga_evaluator.fitness_score()`.

### Consequências
- Modelos que ignoram classes minoritárias terão F1 penalizado mesmo com acurácia alta
- A dupla métrica evita overfitting em relação a uma única dimensão
- `fitness_score()` é exposto como função pública para uso no dashboard

---

## ADR-006 — Critérios de Parada Duplos: max_generations + patience

**Data**: 2026-06  
**Status**: Aceito

### Contexto
GAs podem ficar estagnados (plateau de fitness) muito antes de atingir o número máximo de gerações, desperdiçando tempo computacional.

### Decisão
Dois critérios independentes — para quando o **primeiro** for satisfeito:

1. **`max_generations`**: limite absoluto (equivalente ao critério padrão de GAs)
2. **`patience`**: N gerações consecutivas sem melhoria de `_CONVERGENCE_EPS = 1e-6` no best fitness global

### Consequências
- O resultado inclui `reason: "max_generations" | "convergence"` para rastreabilidade
- Para dados grandes (SISVAN, ~50MB+), a parada por convergência pode economizar horas de processamento
- `patience=5` (default) é conservador — ajustar para `patience=3` em runs rápidos de validação

---

## ADR-007 — Pipeline sklearn para IndividuoKNN: StandardScaler Obrigatório

**Data**: 2026-06  
**Status**: Aceito

### Contexto
KNN é um algoritmo baseado em distância. Features com escalas muito diferentes (ex: IMC em kg/m² vs. idade em anos) dominam o cálculo de distância, distorcendo os resultados.

### Decisão
`IndividuoKNN.build_model()` sempre retorna:
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(**self.hyperparams))
])
```
O `StandardScaler` é **parte do indivíduo**, não do pipeline externo.

### Justificativa
- RF é invariante à escala (baseado em splits de árvore) → não precisa de scaler
- Encapsular o scaler no Pipeline do KNN evita data leakage (o scaler é fit apenas no treino de cada fold via `cross_val_score`)
- O `ga_evaluator.evaluate()` pode tratar ambos os tipos uniformemente sem lógica condicional

### Consequências
- `IndividuoRF.build_model()` → `Pipeline([("clf", RF(...))])` — sem scaler
- `IndividuoKNN.build_model()` → `Pipeline([("scaler", StandardScaler()), ("clf", KNN(...))])` — scaler incluso
- Ambos retornam `Pipeline` — interface unificada para `cross_val_score` e `fit/predict`

---

## ADR-008 — Mutação com 3 Níveis de Agressividade

**Data**: 2026-06  
**Status**: Aceito

### Contexto
A taxa de mutação fixa (`mutation_rate=0.2`) não é adequada para todos os cenários:
- Exploração inicial → mutação mais agressiva pode ser desejável
- Refinamento final → mutações menores preservam boas soluções

### Decisão
Parâmetro `mutation_aggressiveness: "low" | "medium" | "high"` com mapeamento:

| Nível | `mutation_rate` | `delta_pct` (numérico) |
|---|---|---|
| `low` | 0.1 | ±10% do range |
| `medium` | 0.2 | ±30% do range |
| `high` | 0.4 | ±60% do range |

Parâmetros categóricos (ex: `criterion`, `weights`) usam apenas `mutation_rate` (substituição aleatória).

### Consequências
- Usuário expõe `--aggressiveness` no CLI e no sidebar Streamlit
- Perturbação gaussiana em parâmetros numéricos: `_mutate_int()` com sigma proporcional ao range
- Original nunca é modificado (cópia profunda antes de mutar) — validado por `test_does_not_mutate_original`

> **Validação:** `TestMutateRFAggressiveness.test_aggressiveness_ordering_n_estimators` e `TestMutateKNNAggressiveness.test_aggressiveness_ordering_n_neighbors` confirmam a ordenação `high >= medium >= low` em delta médio absoluto com N=100 amostras e seed fixa.

---

## ADR-009 — Elitismo Opcional com Elitismo Estrutural Mínimo

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Elitismo garante que o melhor indivíduo nunca é perdido, mas pode reduzir diversidade e causar convergência prematura.

### Decisão
**Duas camadas de elitismo**:

1. **Elitismo opcional** (`elitism: bool`):
   - `True` (padrão): deepcopy do melhor global é reinserido substituindo o pior da sub-pop correspondente após crossover/mutação
   - `False`: nenhuma garantia de preservação do melhor — GA mais exploratório

2. **Elitismo estrutural mínimo** (sempre ativo, independente do flag `elitism`):
   - Após seleção global, garante ao menos 1 sobrevivente de cada tipo (`max(1, ...)`)
   - Evita extinção prematura de um tipo inteiro pela pressão competitiva

### Consequências
- `--no-elitism` no CLI ativa modo exploratório puro
- O elitismo estrutural mínimo é uma salvaguarda implícita — não exposto como parâmetro
- `test_with_elitism_enabled` e `test_with_elitism_disabled` validam ambos os cenários

> **Validação aprofundada (2026-07-05):**
> - `TestElitism.test_elitism_true_never_regresses`: confirma que `global_best_f1` é monotônica não-decrescente ao longo de 6 gerações com `elitism=True`. Este teste detecta bugs na lógica de reinserção do elite que os testes anteriores não pegariam.
> - `TestCoEvolution.test_reproducibility_full_hyperparams`: confirma que o GA é completamente determinístico (mesma seed → mesmos `hyperparams` e `fitness_values`), incluindo o `KFold` interno ao `cross_val_score`.

---

## ADR-010 — Persistência em 3 Formatos: JSON + CSV + joblib

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Os resultados do GA precisam ser consumidos por diferentes audiências:
- **Desenvolvimento**: análise do histórico de evolução via Python/Pandas
- **Visualização**: gráficos no dashboard Streamlit
- **Produção**: modelo sklearn carregável para predições

### Decisão
**3 formatos de saída** via `ga_persistence.save_ga_results()`:

| Arquivo | Formato | Uso |
|---|---|---|
| `ga_history.json` | JSON | Histórico completo estruturado (params + all generations + best_individual) |
| `ga_generation_stats.csv` | CSV | Tabela unificada por geração (rf_* + knn_* + global_*) para análise/visualização |
| `best_model.joblib` | joblib | Pipeline sklearn treinado no dataset completo, pronto para `predict()` |

### Consequências
- `save_best_model()` executa um `fit()` final no dataset completo antes de persistir (o GA avalia via CV, não treina o modelo final)
- `load_ga_history()` permite recarregar resultados sem re-executar o GA
- O CSV tem uma linha por geração com todas as métricas — adequado para `pd.read_csv()` direto no Streamlit

---

## ADR-011 — Biblioteca de Extração de Archives: patoolib

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O dataset SISVAN é distribuído em formato `.rar`. O backend precisa extrair o arquivo automaticamente antes do pipeline de pré-processamento sem exigir intervenção manual.

### Alternativas Consideradas
| Biblioteca | Prós | Contras |
|---|---|---|
| **`patoolib`** | Suporta múltiplos formatos (rar, zip, 7z, tar, etc.); delega a ferramenta externa disponível no SO; interface unificada | Depende de executável externo instalado no sistema |
| `rarfile` | Específico para RAR; não precisa de binário externo para leitura | Apenas RAR; requer `unrar` para escrita; menos flexível |
| `zipfile` (stdlib) | Sem dependências externas | Não suporta `.rar` |
| `py7zr` | Suporta 7z nativamente em Python | Não suporta `.rar` nativamente |

### Decisão
**`patoolib>=2.0.0`** — pela interface unificada que abstrai o formato e delega automaticamente para o extrator disponível no sistema (`unrar`, `7z`, `unar`, etc.). Isso permite que o mesmo código funcione em diferentes ambientes sem alteração.

### Consequências
- Dependência adicionada: `patool>=2.0.0,<3.0.0`
- O container Docker precisa ter ao menos um extrator compatível instalado via `apt-get`
- A função `extract_rar_file()` em `src/data/ingest.py` usa `patoolib.extract_archive()` como ponto de entrada único

---

## ADR-012 — Pacote de Extração RAR no Docker: `unrar-free` vs. `unrar`

**Data**: 2026-06  
**Status**: Aceito

### Contexto
A imagem base `python:3.13-slim` usa **Debian trixie**. O pacote `unrar` (versão proprietária da RARLab) **não está disponível nos repositórios `main` do Debian trixie** — requer habilitar o repositório `non-free`, o que não é feito por padrão na imagem base.

Sintoma observado:
```
could not find an executable program to extract format rar;
candidates are rar,unrar,7z,7zz,7zzs,unar
```

### Alternativas Consideradas
| Opção | Disponibilidade | Complexidade |
|---|---|---|
| `unrar` (proprietário) | Requer habilitar `non-free` no apt | Alta — modifica fontes do apt |
| **`unrar-free`** | Disponível no repositório `main` | Baixa — `apt-get install unrar-free` |
| `p7zip-rar` | Não disponível em Debian trixie | — |
| `unar` | Disponível no `main` | Baixa, mas menos testado |

### Decisão
**`unrar-free`** — disponível nos repositórios padrão do Debian trixie (sem necessidade de habilitar `non-free`), provê o binário `/usr/bin/unrar` reconhecido pelo `patoolib`.

```dockerfile
RUN apt-get update && apt-get install -y p7zip-full unrar-free && rm -rf /var/lib/apt/lists/*
```

### Consequências
- Nenhuma mudança nas fontes do apt — imagem mais simples e segura
- `unrar-free` suporta RAR 2.x e 3.x; arquivos RAR5 podem ter limitações (o dataset SISVAN usa RAR3)
- Documentação atualizada: instrução Linux passa de `apt-get install unrar` para `apt-get install unrar-free`

---

## ADR-013 — Orquestração do Pipeline via API REST com Jobs Assíncronos

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O pipeline completo (pré-processamento → tuning → predições) envolve operações de longa duração (minutos a horas). O frontend Streamlit precisa de uma forma de iniciar cada etapa e monitorar o progresso sem bloquear a interface.

### Alternativas Consideradas
| Abordagem | Prós | Contras |
|---|---|---|
| Execução síncrona (resposta direta) | Simples | HTTP timeout; frontend trava |
| **FastAPI BackgroundTasks + job store em memória** | Sem dependências externas; simples de implementar | Job perdido em restart do container |
| Celery + Redis/RabbitMQ | Persistente; escalável; retry automático | Alta complexidade operacional; requer infraestrutura adicional |
| SSE / WebSockets | Streaming em tempo real | Maior complexidade no frontend |

### Decisão
**FastAPI `BackgroundTasks` + job store em memória** (`src/api/job_store.py`), exposto pela rota `src/api/routes/pipeline.py`.

Fluxo de cada etapa:
1. `POST /pipeline/preprocess|tune|predict` → cria job, inicia tarefa em background, retorna `{job_id}`
2. Frontend faz polling em `GET /pipeline/jobs/{job_id}` até `status == "completed"` ou `"failed"`
3. Estado do pipeline é rastreado em `src/api/pipeline_store.py` para garantir ordenação das etapas

### Endpoints do Pipeline

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/pipeline/preprocess` | Inicia pré-processamento (extrai .rar se necessário) |
| `POST` | `/pipeline/tune` | Inicia tuning genético (requer preprocess concluído) |
| `POST` | `/pipeline/predict` | Gera predições (requer tune concluído) |
| `GET` | `/pipeline/status` | Estado atual do pipeline |
| `GET` | `/pipeline/jobs/{id}` | Status e resultado de um job |

### Consequências
- Jobs são perdidos em restart do container (aceitável para o escopo do Tech Challenge)
- A ordenação das etapas é validada pelo `pipeline_store` — não é possível executar `tune` antes de `preprocess`
- Cada etapa delega a execução a um subprocess (`scripts/run_*.py`) para isolamento de memória e compatibilidade com o engine Python do pandas (necessário para evitar SIGSEGV em arquivos grandes)

---

## ADR-014 — Monitoramento do Algoritmo Genético em Tempo Real (Snapshots + Polling Incremental)

**Data**: 2026-07  
**Status**: Aceito

### Contexto
O algoritmo genético roda em segundo plano e pode levar vários minutos para concluir todas as gerações. O usuário precisa acompanhar a evolução das métricas (fitness, F1, acurácia e parâmetros) em tempo real no dashboard, sem que a página Streamlit precise ser atualizada manualmente ou pisque de forma desagradável.

### Alternativas Consideradas
1. **Compartilhamento de Arquivo Local**: O frontend ler diretamente o arquivo JSONL de logs gerado pelo backend.
   - *Contra*: Não funciona em arquiteturas de containers distribuídas onde backend e frontend rodam em hosts separados sem volumes compartilhados.
2. **Mecanismo de WebSockets / SSE**: Enviar atualizações via streaming do backend para o frontend.
   - *Contra*: Alta complexidade de integração com o modelo de execução síncrono/passivo do Streamlit.
3. **Persistência Incremental por Geração (JSONL) + Polling via REST API**:
   - *Decisão*: O backend grava um snapshot estruturado em um arquivo local JSONL (`/tmp/ag_job_{job_id}_generations.jsonl`) a cada geração do GA. O frontend faz chamadas periódicas via HTTP (`GET /tuning/jobs/{job_id}/generations?since=N`) buscando apenas gerações novas.

### Justificativa
- **Isolamento de Infraestrutura**: O frontend nunca acessa o filesystem do backend; a comunicação é 100% via API REST.
- **Eficiência**: O parâmetro `since=N` garante uma busca incremental rápida, sem trafegar o histórico inteiro a cada requisição.
- **Resiliência do Streamlit**: Em vez de loops `while True` bloqueantes que causam colisões de elementos (`StreamlitDuplicateElementKey`), usamos o ciclo de vida nativo com `st.rerun()` e `time.sleep()`. A reconciliação do React no Streamlit garante que apenas as informações alteradas atualizem, de forma extremamente fluida e sem piscar.

### Consequências
- A assinatura do `GeneticAlgorithm.run()` agora aceita `job_id` para persistência em tempo real.
- Novo endpoint implementado no backend: `GET /tuning/jobs/{job_id}/generations`.
- Os jobs de tuning do pipeline principal (`/pipeline/tune`) foram migrados de subprocessos isolados para chamadas diretas através do `tuning_service`, permitindo o compartilhamento de contexto de execução e gravação de snapshots em tempo real.
- Para evitar a lentidão de processamento do GA em datasets muito grandes (1.5M+ linhas), implementou-se amostragem estratificada (`sample_size=50_000`) como padrão no `tuning_service`.

---

## ADR-015 — Logs de Pré-processamento via Arquivo Local e Polling REST

**Data**: 2026-07  
**Status**: Aceito

### Contexto
O pipeline de pré-processamento executa como um subprocesso em segundo plano e pode levar de alguns segundos a minutos. No cenário original, a tela de pré-processamento no Streamlit "congelava" enquanto aguardava de forma síncrona a conclusão do job. Desejava-se retornar o status de inicialização imediatamente, exibir logs detalhados do progresso e atualizar a interface automaticamente ao finalizar.

### Alternativas Consideradas
1. **Streaming via WebSockets / SSE**: Transmitir logs em tempo real por fluxo contínuo.
   - *Contra*: O modelo de execução do Streamlit é puramente síncrono e baseado em execuções de cima para baixo. Integrar WebSockets/SSE adiciona grande complexidade no controle de estado do frontend.
2. **Armazenamento de Logs em Memória (Job Store)**: Capturar a saída do processo e salvá-la em uma variável em memória.
   - *Contra*: Ingestão de logs em variáveis pode consumir memória excessiva e corre o risco de perda de informações em restarts de container ou crashs de rotina.
3. **Escrita em Arquivo Temporário Local + Polling REST**: Redirecionar `stdout`/`stderr` do subprocesso diretamente para um arquivo local (`/tmp/preprocessing_{job_id}.log`) e expor uma rota GET na API do backend para retornar o conteúdo do arquivo. O frontend faz polling reativo consultando esse arquivo.

### Decisão
**Escrita em Arquivo Temporário Local + Polling REST** (opção 3):
- Abstração simples e robusta, suportada nativamente pelo sistema operacional.
- O backend inicia o pré-processamento via `subprocess.Popen` em uma `BackgroundTask`, liberando a resposta HTTP inicial imediatamente.
- Todo o output do script de pré-processamento é escrito no arquivo `/tmp/preprocessing_{job_id}.log`.
- O endpoint `/pipeline/jobs/{job_id}/logs` retorna o conteúdo desse arquivo em tempo real.
- O frontend faz o polling de forma reativa a cada 2 segundos, renderizando os logs no console da página de forma não-bloqueante e atualizando a interface quando o status do job for `"completed"` ou `"failed"`.

### Consequências
- A tela do pré-processamento não congela mais e mostra feedback instantâneo da execução.
- O arquivo de log permanece disponível para consulta mesmo após a conclusão do processamento.
- Sem necessidade de dependências complexas de streaming bidirecional no Streamlit.

---

## ADR-016 — Fallback Multi-Provedor Stateful para o Agente LLM

**Data**: 2026-07  
**Status**: Aceito

### Contexto

O agente `NutritionalHealthAgent` precisa ser resiliente a falhas de rate limit e indisponibilidade da API do provedor LLM primário. O LangChain oferece o mecanismo `RunnableWithFallbacks` (via `.with_fallbacks()`), mas ele é **stateless** por design: a cada pergunta, sempre recomeça tentando pelo primeiro provedor da lista. Isso causa comportamento ineficiente quando o provedor primário está sofrendo rate limit — cada mensagem do usuário acumularia a latência de falha do primário antes de cair para o fallback.

### Alternativas Consideradas

| Alternativa | Prós | Contras |
|---|---|---|
| `RunnableWithFallbacks` do LangChain | API simples | Stateless — sempre retenta o primário; latência acumulada em rate limit |
| Retry com backoff no provedor primário | Simples | Bloqueia a interface do usuário por longos períodos |
| **Fallback stateful manual (sessão)** | Avanço permanente no rate limit; retry transparente | Requer gestão de estado explícita na instância do agente |

### Decisão

**Fallback stateful gerenciado manualmente** via atributos `_built_llms` (lista de provedores instanciados) e `_active_index` (índice do provedor ativo na sessão):

- `_advance_provider()`: promove permanentemente para o próximo provedor ao detectar falha, sem voltar ao primário na mesma sessão
- Retry transparente: a pergunta que sofreu falha é retentada automaticamente com o novo provedor
- Se todos os provedores falharem, a rota `/llm/chat` retorna HTTP 503 (Service Unavailable)
- Exceções que disparam avanço: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError` (OpenAI) e `GoogleAPICallError` (Google)

### Consequências

- O `AgentExecutor` é reconstruído dinamicamente a cada avanço de provedor (pois `create_react_agent` bake o LLM no objeto)
- A reconstrução é protegida por `hasattr(self, 'tools')` para evitar falhas durante o `__init__` (antes das ferramentas serem definidas)
- Provedores sem chave configurada são automaticamente ignorados na inicialização com aviso de log
- A ordem dos provedores é configurável via `LLM_PROVIDER_ORDER` no `.env`

---

## ADR-017 — PatchedChatOpenAI: Auto-Recovery do Parâmetro `stop`

**Data**: 2026-07  
**Status**: Aceito

### Contexto

Modelos de raciocínio da OpenAI (ex: `o1`, `o3-mini`, `gpt-5-nano`) e via OpenRouter não aceitam o parâmetro `stop` na API, retornando `HTTP 400 BadRequestError` com a mensagem `"Unsupported parameter: 'stop' is not supported with this model."`. O LangChain ReAct Agent injeta `stop=["Observation:"]` por padrão em todas as chamadas.

### Diagnóstico Técnico

O parâmetro `stop` é injetado no payload JSON da API pelo método interno `_get_request_payload`, que é chamado *dentro* de `_generate` e `_stream`. Interceptar apenas `_generate` sobrescrevendo o argumento `stop=None` **não é suficiente**, pois `super()._generate()` chama `_get_request_payload` internamente, que reinsere o `stop` via `kwargs` ou `_default_params`.

```
LangChain ReAct Agent
  └─► _generate(messages, stop=["Observation:"])   ← argumento interceptável
        └─► _get_request_payload(messages, stop=stop)  ← ponto real de injeção
              └─► {"messages": [...], "stop": [...]}   ← enviado à API
```

### Alternativas Consideradas

| Alternativa | Prós | Contras |
|---|---|---|
| Sobrescrever apenas `_generate` | Simples | Insuficiente: `super()._generate()` reinjecta o `stop` via `_get_request_payload` |
| Configuração manual `OPENAI_DROP_STOP=true` | Elimina o primeiro erro | Requer configuração explícita por modelo |
| **Sobrescrever `_get_request_payload` + auto-recovery em `_generate`** | Elimina o `stop` no ponto real; funciona automaticamente | Dependência do método privado do LangChain (mas estável) |

### Decisão

Classe `PatchedChatOpenAI(ChatOpenAI)` com duas camadas de proteção:

1. **`_get_request_payload`** (sobrescrito): quando `drop_stop=True`, remove `stop=None` e `kwargs.pop("stop")` *antes* do payload ser construído — elimina o parâmetro no ponto de origem
2. **`_generate`** (sobrescrito): intercepta `BadRequestError` com mensagem contendo `"stop"`, ativa `drop_stop=True` e retenta de forma transparente — auto-recovery na primeira falha

Para modelos conhecidos que não suportam `stop`, configure `OPENAI_DROP_STOP=true` ou `OPENROUTER_DROP_STOP=true` no `.env` para eliminar o erro mesmo na primeira chamada.

### Consequências

- `logger` deve ser declarado antes da classe `PatchedChatOpenAI` para evitar `NameError` no bloco de auto-recovery (exceção capturada antes do logger ser definido)
- A flag `drop_stop` é persistida na instância do modelo: uma vez ativada, permanece ativa para todas as chamadas subsequentes naquela sessão
- A classe é reutilizável para qualquer provedor OpenAI-compatível (incluindo OpenRouter)
- Adicionado ao `_FALLBACK_EXCEPTIONS` apenas erros de disponibilidade; `BadRequestError` de `stop` é tratado localmente sem acionar o fallback de provedor

