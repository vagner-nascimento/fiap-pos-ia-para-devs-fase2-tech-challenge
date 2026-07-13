# FIAP Pós-IA - Fase 2: Tech Challenge — Backend

Este repositório contém o **backend** do projeto desenvolvido para o **Tech Challenge da Fase 2** da Pós-Graduação em Inteligência Artificial para Desenvolvedores da **FIAP**.

O sistema analisa e prevê o estado nutricional a partir de dados do SISVAN (Sistema de Vigilância Alimentar e Nutricional), utilizando:

- **Algoritmo Genético Co-Evolutivo** para otimização de hiperparâmetros (RF e KNN)
- **Pipeline de Machine Learning** com RandomForest e KNeighborsClassifier
- **Agente LLM ReAct** com fallback multi-provedor para análise clínica interativa
- **API REST** (FastAPI) para orquestração assíncrona de todos os serviços

---

## 🚀 Requisitos e Dependências

- **Python >= 3.13**
- **Gerenciador de dependências**: [`uv`](https://docs.astral.sh/uv/) (recomendado) ou `pip`
- **Pelo menos uma chave de API LLM** (Google Gemini, OpenAI ou OpenRouter)
- **`unrar`** para extração de arquivos `.rar`:
  - **Windows**: Instale o [WinRAR](https://www.win-rar.com/) ou adicione `unrar` ao PATH
  - **Linux/macOS**: `sudo apt-get install unrar-free` ou `brew install unar`

  > **Docker**: A imagem já inclui `unrar-free` automaticamente.

---

## 🛠️ Configuração do Ambiente

### Passo 1: Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd fiap-pos-ia-para-devs-fase2-tech-challenge/backend
```

### Passo 2: Configurar Variáveis de Ambiente

```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Edite o `.env` com sua(s) chave(s) de API:

```env
# ── Ordem dos provedores (o primeiro é o principal, os demais são fallbacks) ──
LLM_PROVIDER_ORDER=gemini,openai,openrouter

# ── Google Gemini ──────────────────────────────────────────────────────────────
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7

# ── OpenAI (opcional — fallback) ───────────────────────────────────────────────
# OPENAI_API_KEY=sk-proj-...
# OPENAI_MODEL=gpt-4o
# OPENAI_TEMPERATURE=0.7
# OPENAI_DROP_STOP=true   # Obrigatório para modelos de raciocínio (o1, o3-mini, gpt-5-nano)

# ── OpenRouter (opcional — fallback) ───────────────────────────────────────────
# OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
# OPENROUTER_TEMPERATURE=0.7
# OPENROUTER_DROP_STOP=true   # Obrigatório para modelos de raciocínio via OpenRouter
```

> **Nota**: Provedores sem chave configurada são automaticamente ignorados com aviso de log.

### Passo 3: Instalar Dependências

**Com `uv` (recomendado):**
```bash
uv sync
```

**Com `pip` (alternativa):**
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install .
```

---

## 🔑 Como Obter Chaves de API

### Google Gemini (gratuito)

1. Acesse o [Google AI Studio](https://aistudio.google.com/)
2. Clique em **"Get API key"** → **"Create API key"**
3. Copie a chave (começa com `AIzaSy...`) e adicione ao `.env` como `GEMINI_API_KEY`

### OpenAI

1. Acesse [platform.openai.com](https://platform.openai.com/)
2. Vá em **API keys** → **Create new secret key**
3. Adicione ao `.env` como `OPENAI_API_KEY`

### OpenRouter (acesso a múltiplos modelos — inclui gratuitos)

1. Acesse [openrouter.ai](https://openrouter.ai/)
2. Vá em **Keys** → **Create Key**
3. Adicione ao `.env` como `OPENROUTER_API_KEY`

---

## 🖥️ Como Executar

### 1. Pré-processamento de Dados

```bash
# Arquivo CSV
uv run python scripts/run_preprocessing.py --input data/raw/estado_nutricional_sao_paulo.csv

# Arquivo .rar (extração automática)
uv run python scripts/run_preprocessing.py --input data/raw/estado_nutricional_sao_paulo.rar
```

### 2. Tuning de Hiperparâmetros (GA Co-Evolutivo)

```bash
# Execução rápida (~5-10 min) — 50k amostras, pop=4, 2 gerações
uv run python scripts/run_tuning.py \
  --input data/processed/estado_nutricional_clean.csv

# Produção (~30-60 min) — dataset completo, pop=20, 10 gerações
uv run python scripts/run_tuning.py \
  --input data/processed/estado_nutricional_clean.csv \
  --sample 200000 \
  --pop-size 20 --max-generations 10 --patience 5 --k-folds 5
```

> [!TIP]
> Use `--no-elitism --aggressiveness high` para forçar maior exploração quando os resultados convergirem prematuramente.

### 3. Geração de Predições

```bash
uv run python scripts/run_predictions.py
```

### 4. API REST (FastAPI)

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

Documentação interativa: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Validação Interativa das Ferramentas do Agente (sem custo de API)

```bash
uv run python scripts/validate_tools.py
```

Este utilitário faz mock do LLM e permite testar as ferramentas `get_nutrition_statistics`, `filter_nutrition_records` e `get_clinical_recommendations` via console.

---

## 📊 Endpoints Principais da API

| Método | Rota | Pré-requisito | Descrição |
|--------|------|----------------|-----------|
| `GET` | `/health` | — | Health check |
| `POST` | `/pipeline/preprocess` | `.csv`/`.rar` em `data/raw/` | Pré-processamento assíncrono |
| `POST` | `/pipeline/tune` | `preprocess` concluído | GA Co-Evolutivo com snapshots em tempo real |
| `POST` | `/pipeline/predict` | `tune` concluído | Gera `predictions.csv` |
| `GET` | `/pipeline/status` | — | Estado atual do pipeline |
| `GET` | `/pipeline/jobs/{id}` | — | Status e resultado de um job |
| `GET` | `/pipeline/jobs/{id}/logs` | — | Logs em tempo real do pré-processamento |
| `GET` | `/tuning/datasets` | — | Lista CSVs disponíveis em `data/processed/` |
| `POST` | `/tuning/run` | — | Executa GA Co-Evolutivo (modo legado) |
| `GET` | `/tuning/jobs/{id}` | — | Status de job assíncrono |
| `GET` | `/tuning/jobs/{id}/generations` | — | Snapshots incrementais de gerações |
| `GET` | `/tuning/logs/latest` | — | Último histórico GA |
| `POST` | `/llm/session` | — | Cria sessão do agente (upload CSV) |
| `POST` | `/llm/chat` | Sessão ativa | Pergunta ao agente ReAct |

---

## 🤖 Agente LLM — Fallback Multi-Provedor

O `NutritionalHealthAgent` implementa um sistema de **fallback stateful** para alta disponibilidade:

- **Ordem configurável**: `LLM_PROVIDER_ORDER=gemini,openai,openrouter`
- **Avanço permanente**: ao detectar rate limit, timeout ou erro de servidor, o provedor falho é descartado e o próximo é promovido **permanentemente** na sessão (sem retry ao primário)
- **Retry transparente**: a pergunta que sofreu falha é retentada automaticamente com o novo provedor
- **HTTP 503**: se todos os provedores falharem, a API retorna 503 (Service Unavailable)

### Auto-Recovery do Parâmetro `stop`

Modelos de raciocínio (ex: `o1`, `o3-mini`, `gpt-5-nano`) não aceitam o parâmetro `stop`, mas o LangChain ReAct Agent o injeta por padrão. A classe `PatchedChatOpenAI` resolve isso automaticamente:

1. Na primeira chamada, se a API retornar `BadRequestError` com mensagem contendo `"stop"`, a flag `drop_stop=True` é ativada e a chamada é retentada sem o parâmetro
2. Nas chamadas seguintes, o `stop` é removido em `_get_request_payload` (ponto real de injeção no payload JSON)
3. Para evitar o primeiro erro, configure `OPENAI_DROP_STOP=true` ou `OPENROUTER_DROP_STOP=true` no `.env`

---

## 🧪 Testes Automatizados

A suíte cobre **313 testes** no total, validando de forma abrangente o comportamento correto do sistema. A execução completa leva menos de **25 segundos** graças ao mocking otimizado.

```bash
# Executar todos os testes
uv run pytest

# Com relatório de cobertura detalhado
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

### Cobertura por Área

| Suite | Testes | Área |
|---|---|---|
| `test_ga_operators.py` | ~26 | Crossover uniforme, mutação por magnitude |
| `test_ga_evaluator.py` | ~12 | Fitness, fallback mock, propagação k_folds |
| `test_ga_persistence.py` | ~15 | Save/load JSON+CSV do histórico GA |
| `test_ga_snapshot.py` | ~10 | Snapshots incrementais JSONL em tempo real |
| `test_individuo.py` | ~15 | IndividuoRF, IndividuoKNN, build_model |
| `test_features.py` | ~20 | Engenharia de features, codificação |
| `test_preprocessing.py` | ~25 | Limpeza, remoção de gestantes, padronização |
| `test_ingest.py` | ~10 | Leitura CSV, extração .rar |
| `test_api_stores.py` | ~20 | JobStore, PipelineStore, SessionStore |
| `test_services.py` | ~20 | TuningService, lógica de negócio |
| `test_utils.py` | ~20 | Logger, persistence, validators |
| `test_llm_agent.py` | ~14 | Fallback multi-provedor, PatchedChatOpenAI, ferramentas ReAct |
| `test_ga_optimizer.py` | ~15 | GA end-to-end: elitismo, convergência, reprodutibilidade |
| `test_api_pipeline.py` | ~10 | Jobs assíncronos, logs, orquestração |
| `test_api_routes.py` | ~25 | Rotas gerais da API |
| `test_api_tuning.py` | ~10 | Rotas de tuning, polling de gerações |

---

## 📁 Estrutura de Diretórios

```
backend/
├── README.md               # Este arquivo
├── pyproject.toml          # Definições de empacotamento e dependências
├── .env.example            # Modelo de variáveis de ambiente
│
├── docs/                   # Documentação técnica
│   ├── architecture.md     # Arquitetura técnica detalhada
│   ├── adr.md              # Architecture Decision Records
│   └── resumo-melhorias-agente-llm-fallback.md  # Histórico do agente LLM
│
├── src/
│   ├── agents/
│   │   └── nutritional_agent.py    # NutritionalHealthAgent + PatchedChatOpenAI
│   ├── api/
│   │   ├── main.py                 # FastAPI app + CORS
│   │   ├── job_store.py            # Store de jobs assíncronos
│   │   ├── pipeline_store.py       # Estado das etapas do pipeline
│   │   ├── session_store.py        # Sessões do agente LLM
│   │   └── routes/
│   │       ├── health.py           # GET /health
│   │       ├── pipeline.py         # /pipeline/*
│   │       ├── tuning.py           # /tuning/*
│   │       └── llm.py              # /llm/*
│   ├── data/
│   │   ├── ingest.py               # Ingestão CSV + extração .rar
│   │   ├── features.py             # Engenharia de features
│   │   └── preprocessing.py        # Pipeline de pré-processamento
│   ├── models/
│   │   ├── individuo.py            # Hierarquia de indivíduos GA
│   │   ├── ga_operators.py         # Operadores genéticos
│   │   ├── ga_evaluator.py         # Função de fitness k-Fold
│   │   ├── genetic_algorithm.py    # Loop co-evolutivo
│   │   └── ga_persistence.py       # Save/load de artefatos
│   ├── services/
│   │   └── tuning_service.py
│   └── utils/
│       ├── logger.py
│       ├── persistence.py
│       └── validators.py
│
├── scripts/
│   ├── run_preprocessing.py        # CLI pré-processamento
│   ├── run_tuning.py               # CLI GA Co-Evolutivo
│   ├── run_predictions.py          # CLI geração de predições
│   └── validate_tools.py           # Validação interativa das ferramentas do agente
│
├── models/
│   ├── artifacts/                  # best_model.joblib, mappings.json, encoders
│   └── logs/                       # ga_history.json, ga_generation_stats.csv
│
└── tests/
    ├── unit/                       # 13 arquivos de testes unitários
    └── integration/                # 4 arquivos de testes de integração
```

---

## 📖 Documentação Adicional

- [Arquitetura Técnica](docs/architecture.md) — estrutura completa, fluxo do GA e do agente LLM
- [Architecture Decision Records](docs/adr.md) — justificativas das decisões de design
- [Melhorias do Agente LLM](docs/resumo-melhorias-agente-llm-fallback.md) — histórico de decisões sobre fallback e auto-recovery
- [Avaliação de Qualidade do LLM](../experiments/llm_quality_eval.md) — rubrica e análise de respostas do agente