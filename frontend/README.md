# Front-end Streamlit — SISVAN Nutricional

Interface de usuário separada do backend. Consome a API REST via HTTP.

## Requisitos

- Python >= 3.13
- Backend API rodando (ver `../backend/README.md`)

## Configuração

```powershell
cd frontend
copy .env.example .env
```

Edite `.env`:

```env
BACKEND_URL=http://localhost:8000
```

## Instalação

```bash
uv sync
```

## Executar

```bash
# Terminal 1 — backend (na pasta backend/)
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
uv run streamlit run app/main.py --server.port 8501
```

## Páginas

| Página | Arquivo | Descrição |
|--------|---------|-----------|
| Home | `app/main.py` | Status da API e navegação |
| 🧬 Tuning Genético | `app/pages/tuning_monitor.py` | Dashboard do GA Co-Evolutivo |
| 💬 Agente Nutricional | `app/pages/chat_agent.py` | Chat ReAct com Gemini |

## Deploy

Build e execução via Docker:

```bash
docker build -t sisvan-frontend .
docker run -p 8501:8501 -e BACKEND_URL=https://api.seu-dominio.com sisvan-frontend
```

Ou use `docker-compose up` na raiz do repositório.
