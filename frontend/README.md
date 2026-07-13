# Frontend Streamlit — SISVAN Nutricional

Interface visual do sistema de análise nutricional. Consome a API REST do backend via HTTP.

---

## Requisitos

- **Python >= 3.13**
- **Backend API** rodando (ver [`../backend/README.md`](../backend/README.md))

---

## Configuração

```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Edite `.env`:

```env
BACKEND_URL=http://localhost:8000
```

---

## Instalação

```bash
uv sync
```

---

## Executar

```bash
# Terminal 1 — backend (na pasta backend/)
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend (nesta pasta)
uv run streamlit run app/main.py --server.port 8501
```

Acesse: [http://localhost:8501](http://localhost:8501)

---

## Páginas

| # | Página | Arquivo | Descrição |
|---|--------|---------|-----------|
| — | **Home** | `app/main.py` | Status da API, links de navegação e visão geral |
| 01 | **🗂 Pré-processamento** | `app/pages/01_preprocessing.py` | Console com logs em tempo real do pré-processamento (`/pipeline/preprocess`) |
| 02 | **🧬 Tuning Genético** | `app/pages/02_tuning.py` | Dashboard do GA Co-Evolutivo com gráficos Plotly atualizados incrementalmente via polling de gerações |
| 03 | **📊 Predições** | `app/pages/03_predictions.py` | Visualização do CSV de predições gerado pelo modelo vencedor |
| 04 | **💬 Agente Nutricional** | `app/pages/04_llm_chat.py` | Chat interativo com o agente ReAct (via `/llm/session` e `/llm/chat`) |
| 04 | **🔬 Comparação de Modelos** | `app/pages/04_model_comparison.py` | Comparativo de métricas RF vs KNN |
| 05 | **📖 Explicação do Pipeline** | `app/pages/05_pipeline_explanation.py` | Guia didático do pipeline para o usuário |

---

## Fluxo de Uso Recomendado

1. **Pré-processamento** → carregue o arquivo `.csv` ou `.rar` do SISVAN e aguarde a limpeza dos dados
2. **Tuning Genético** → inicie o GA Co-Evolutivo e acompanhe a evolução em tempo real
3. **Predições** → gere as predições com o melhor modelo encontrado
4. **Agente Nutricional** → faça upload do CSV de predições e inicie o chat clínico

---

## Deploy

### Com Docker Compose (recomendado)

Na raiz do repositório:

```bash
docker-compose up --build
```

Serviços expostos:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000

### Docker Manual

```bash
docker build -t sisvan-frontend .
docker run -p 8501:8501 -e BACKEND_URL=https://api.seu-dominio.com sisvan-frontend
```

---

## Testes

```bash
uv run pytest

# Com relatório de cobertura
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

---

## Documentação Adicional

- [Backend README](../backend/README.md) — configuração do backend e da API
- [Arquitetura Técnica](../docs/architecture.md) — arquitetura completa do sistema
