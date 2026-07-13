# FIAP Pós-IA - Fase 2: Tech Challenge

Sistema completo de análise e previsão de estado nutricional a partir de dados do SISVAN (Sistema de Vigilância Alimentar e Nutricional), utilizando:

- **Algoritmo Genético Co-Evolutivo** para otimização de hiperparâmetros (RF e KNN)
- **Pipeline de Machine Learning** (RandomForest e KNeighborsClassifier)
- **Agente LLM ReAct** com fallback multi-provedor (Gemini, OpenAI, OpenRouter) para análise clínica interativa
- **API REST** (FastAPI) com jobs assíncronos e monitoramento em tempo real
- **Interface Visual** (Streamlit) com dashboard do GA e chat com o agente

---

## 🏗️ Arquitetura do Projeto

```
├── backend/          # API FastAPI + GA Co-Evolutivo + Agente LLM
├── frontend/         # Interface Streamlit
├── docs/             # Documentação técnica
├── experiments/      # Avaliação qualitativa do agente LLM
└── docker-compose.yml
```

### Backend

- **API REST** (FastAPI) com jobs assíncronos, polling de gerações e logs em tempo real
- **GA Co-Evolutivo**: duas populações (RF e KNN) competindo por fitness global (F1×0.6 + Acc×0.4)
- **Agente ReAct** com fallback stateful entre múltiplos provedores LLM (Gemini → OpenAI → OpenRouter)
- **Auto-recovery** do parâmetro `stop` para modelos de raciocínio que não o suportam

### Frontend

- **🗂 Pré-processamento**: console de logs em tempo real
- **🧬 Tuning Genético**: dashboard com gráficos Plotly atualizados incrementalmente por geração
- **📊 Predições**: visualização do CSV com o estado nutricional predito
- **💬 Agente Nutricional**: chat clínico interativo com o agente ReAct
- **🔬 Comparação de Modelos**: métricas comparativas RF vs KNN
- **📖 Explicação do Pipeline**: guia didático para o usuário

Para mais detalhes, consulte:
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [Arquitetura Técnica](docs/architecture.md)

---

## 🚀 Como Executar com Docker Compose

### Pré-requisitos

- **Docker** e **Docker Compose** instalados
- **Chave de API de pelo menos um provedor LLM** (Gemini é o padrão e gratuito)
- **`unrar`** para extração de `.rar` (a imagem Docker já inclui automaticamente)

### Passo 1: Configurar Variáveis de Ambiente

**Backend:**
```bash
cd backend
cp .env.example .env   # Linux/macOS
# copy .env.example .env  # Windows
```

Edite `backend/.env` com sua chave de API:

```env
# Provedor primário: Google Gemini (gratuito)
GEMINI_API_KEY=AIzaSySuaChaveGeradaAqui...
GEMINI_MODEL=gemini-2.5-flash

# Opcional — fallbacks para alta disponibilidade
# OPENAI_API_KEY=sk-proj-...
# OPENAI_MODEL=gpt-4o
# OPENAI_DROP_STOP=true   # Obrigatório para modelos de raciocínio (o1, gpt-5-nano, etc.)

# OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
# OPENROUTER_DROP_STOP=true

# Ordem de fallback (o primeiro é o principal)
LLM_PROVIDER_ORDER=gemini,openai,openrouter
```

**Frontend:**
```bash
cd frontend
cp .env.example .env
```

`frontend/.env`:
```env
BACKEND_URL=http://backend:8000
```

### Passo 2: Executar

Na raiz do projeto:

```bash
docker-compose up --build
```

Serviços expostos:
- **Frontend Streamlit**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **Documentação interativa**: http://localhost:8000/docs

### Scripts de Reinicialização Rápida

```bash
# Linux/macOS
chmod +x restart_containers.sh
./restart_containers.sh

# Windows
restart_containers.bat
```

---

## 🔑 Como Obter a Chave da API do Google AI (Gemini)

1. Acesse o [Google AI Studio](https://aistudio.google.com/)
2. Clique em **"Get API key"** → **"Create API key"**
3. Copie a chave gerada (começa com `AIzaSy...`)
4. Adicione ao `backend/.env`:
   ```env
   GEMINI_API_KEY=AIzaSySuaChaveGeradaAqui...
   ```

---

## 📁 Estrutura de Diretórios

```
├── backend/
│   ├── src/
│   │   ├── agents/           # NutritionalHealthAgent + PatchedChatOpenAI
│   │   ├── api/              # FastAPI: rotas, stores, schemas
│   │   ├── data/             # Ingestão, features, pré-processamento
│   │   ├── models/           # GA Co-Evolutivo (individuo, operadores, evaluator)
│   │   ├── services/         # Lógica de negócio (TuningService)
│   │   └── utils/            # Logger, persistence, validators
│   ├── scripts/              # CLIs: preprocessing, tuning, predictions, validate_tools
│   ├── data/                 # Dados brutos e processados
│   ├── models/               # Artefatos de ML (best_model.joblib, logs GA)
│   ├── docs/                 # Documentação técnica do backend
│   └── tests/                # 313 testes (unit + integration)
│
├── frontend/
│   ├── app/
│   │   ├── main.py           # Página inicial
│   │   └── pages/            # 6 páginas Streamlit
│   └── src/
│       └── api_client.py     # Cliente HTTP para o backend
│
├── docs/                     # Documentação raiz
│   ├── architecture.md       # Arquitetura técnica completa
│   ├── adr.md                # Architecture Decision Records
│   └── gap_analysis.md       # Análise de gaps
│
├── experiments/
│   └── llm_quality_eval.md   # Avaliação qualitativa do agente LLM
│
└── docker-compose.yml        # Orquestração Docker
```

---

## 🛠️ Comandos Úteis

### Docker Compose

```bash
# Iniciar
docker-compose up --build

# Iniciar em background
docker-compose up -d --build

# Parar
docker-compose down

# Ver logs
docker-compose logs -f

# Reiniciar um serviço
docker-compose restart backend
docker-compose restart frontend
```

### Execução Local (Sem Docker)

```bash
# Backend
cd backend
uv sync
uv run uvicorn src.api.main:app --reload --port 8000

# Frontend (novo terminal)
cd frontend
uv sync
uv run streamlit run app/main.py --server.port 8501
```

---

## 📊 Endpoints Principais da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Health check |
| `POST` | `/pipeline/preprocess` | Pré-processamento assíncrono (extrai .rar se necessário) |
| `POST` | `/pipeline/tune` | GA Co-Evolutivo com snapshots em tempo real |
| `POST` | `/pipeline/predict` | Gera predições com o melhor modelo |
| `GET` | `/pipeline/status` | Estado atual do pipeline |
| `GET` | `/pipeline/jobs/{id}` | Status e resultado de um job |
| `GET` | `/pipeline/jobs/{id}/logs` | Logs em tempo real do pré-processamento |
| `GET` | `/tuning/datasets` | Lista CSVs disponíveis |
| `POST` | `/tuning/run` | Executa GA Co-Evolutivo (modo legado) |
| `GET` | `/tuning/jobs/{id}/generations` | Snapshots incrementais de gerações (polling) |
| `GET` | `/tuning/logs/latest` | Último histórico GA |
| `POST` | `/llm/session` | Cria sessão do agente (upload CSV) |
| `POST` | `/llm/chat` | Pergunta ao agente ReAct |

Documentação interativa completa: http://localhost:8000/docs

---

## 🧪 Testes

```bash
# Backend (313 testes — executa em < 25s)
cd backend
uv run pytest

# Com cobertura de código
uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# Frontend
cd ../frontend
uv run pytest
```

---

## 📖 Documentação Adicional

| Documento | Descrição |
|-----------|-----------|
| [Arquitetura Técnica](docs/architecture.md) | Estrutura completa, fluxo do GA e do agente LLM |
| [Architecture Decision Records](docs/adr.md) | Justificativas de decisões de design |
| [Backend README](backend/README.md) | Instruções detalhadas do backend |
| [Frontend README](frontend/README.md) | Instruções detalhadas do frontend |
| [Avaliação do Agente LLM](experiments/llm_quality_eval.md) | Rubrica, perguntas-teste e análise de falhas |
| [Melhorias do Agente LLM](docs/resumo-melhorias-agente-llm-fallback.md) | Histórico de decisões sobre fallback e auto-recovery |

---

## 🤝 Contribuição

Este projeto foi desenvolvido como parte do Tech Challenge da Fase 2 da Pós-Graduação em Inteligência Artificial para Desenvolvedores da FIAP.
