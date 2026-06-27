# FIAP Pós-IA - Fase 2: Tech Challenge

Sistema completo de análise e previsão de estado nutricional a partir de dados do SISVAN (Sistema de Vigilância Alimentar e Nutricional), utilizando:

- **Algoritmo Genético Co-Evolutivo** para otimização de hiperparâmetros de classificação
- **Pipeline de Machine Learning** (RandomForest e KNN)
- **Agente de Inteligência Artificial** (LLM) baseado no padrão ReAct para interpretação clínica e análise estatística interativa
- **API REST** (FastAPI) para orquestração dos serviços
- **Interface Visual** (Streamlit) para monitoramento e interação

---

## 🏗️ Arquitetura do Projeto

O projeto é dividido em dois componentes principais:

```
├── backend/          # API FastAPI + Algoritmo Genético + Agente LLM
├── frontend/         # Interface Streamlit
└── docker-compose.yml # Orquestração dos containers
```

### Backend
- **API REST** com FastAPI para exposição de endpoints
- **Algoritmo Genético Co-Evolutivo** para tuning de hiperparâmetros (RF e KNN)
- **Agente ReAct** integrado com Google Gemini para análise clínica
- **Scripts** de pré-processamento e tuning de dados

### Frontend
- **Dashboard Streamlit** para monitoramento do GA Co-Evolutivo
- **Chat interativo** com o agente de saúde nutricional
- **Visualização** de resultados e estatísticas

Para mais detalhes sobre a arquitetura, consulte:
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [Arquitetura Técnica](backend/docs/architecture.md)

---

## 🚀 Como Executar com Docker Compose

### Pré-requisitos

- **Docker** e **Docker Compose** instalados
- **Chave de API do Google AI Studio** (Gemini)

### Passo 1: Configurar Variáveis de Ambiente

Configure os arquivos `.env` em ambos os diretórios:

**Backend:**
```bash
cd backend
copy .env.example .env
```

Edite `backend/.env` e adicione sua chave da API:
```env
LLM_API_KEY=AIzaSySuaChaveGeradaAqui...
LLM_MODEL=gemini-2.5-flash
```

**Frontend:**
```bash
cd frontend
copy .env.example .env
```

Edite `frontend/.env`:
```env
BACKEND_URL=http://backend:8000
```

### Passo 2: Executar com Docker Compose

Na raiz do projeto:

```bash
docker-compose up --build
```

Isso irá:
1. Construir as imagens Docker do backend e frontend
2. Iniciar os containers
3. Expor os serviços nas portas:
   - **Backend API**: http://localhost:8000
   - **Frontend**: http://localhost:8501

### Passo 3: Acessar a Aplicação

- **Frontend Streamlit**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🔑 Como Obter a Chave da API do Google AI (Gemini)

1. Acesse o [Google AI Studio](https://aistudio.google.com/)
2. Faça login com sua conta Google
3. Clique em **"Get API key"** → **"Create API key"**
4. Copie a chave gerada (começa com `AIzaSy...`)
5. Adicione ao arquivo `backend/.env`:
   ```env
   LLM_API_KEY=AIzaSySuaChaveGeradaAqui...
   ```

---

## 📁 Estrutura de Diretórios

```
├── backend/
│   ├── src/
│   │   ├── api/              # FastAPI routes
│   │   ├── agents/           # Agente LLM ReAct
│   │   ├── models/           # Algoritmo Genético
│   │   ├── services/         # Lógica de negócio
│   │   └── utils/            # Utilitários
│   ├── scripts/              # Scripts CLI
│   ├── data/                 # Dados brutos e processados
│   ├── models/               # Artefatos de ML
│   ├── docs/                 # Documentação técnica
│   └── tests/                # Testes automatizados
│
├── frontend/
│   ├── app/
│   │   ├── main.py           # Página principal
│   │   └── pages/            # Páginas do Streamlit
│   └── src/
│       └── api_client.py     # Cliente HTTP
│
└── docker-compose.yml        # Orquestração Docker
```

---

## 🛠️ Comandos Úteis

### Docker Compose

```bash
# Iniciar os serviços
docker-compose up --build

# Iniciar em background
docker-compose up -d --build

# Parar os serviços
docker-compose down

# Ver logs
docker-compose logs -f

# Reiniciar um serviço específico
docker-compose restart backend
docker-compose restart frontend
```

### Script de Reinicialização

Para reiniciar todos os containers (parar, reconstruir e iniciar em background), use o script `restar_containers.sh`:

```bash
# No Linux/Mac (torne o script executável primeiro)
chmod +x restar_containers.sh
./restar_containers.sh

# No Windows (via Git Bash ou WSL)
bash restar_containers.sh
```

Este script executa:
1. `docker compose down` - Para todos os containers
2. `docker compose up --build -d` - Reconstrói as imagens e inicia em background

### Execução Local (Sem Docker)

Para desenvolvimento local, consulte as instruções detalhadas em:
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)

---

## 📊 Endpoints Principais da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Health check |
| `GET` | `/tuning/datasets` | Lista CSVs disponíveis |
| `POST` | `/tuning/run` | Executa GA Co-Evolutivo |
| `GET` | `/tuning/jobs/{id}` | Status de job assíncrono |
| `GET` | `/tuning/logs/latest` | Último histórico GA |
| `POST` | `/llm/session` | Cria sessão do agente |
| `POST` | `/llm/chat` | Pergunta ao agente ReAct |

Documentação interativa completa em: http://localhost:8000/docs

---

## 🧪 Testes

Para executar os testes automatizados:

```bash
# Via Docker
docker-compose exec backend uv run pytest

# Localmente (no diretório backend)
cd backend
uv run pytest
```

---

## 📖 Documentação Adicional

- [Arquitetura Técnica](backend/docs/architecture.md) - Detalhes da arquitetura e fluxo do GA
- [Architecture Decision Records](backend/docs/adr.md) - Decisões de design
- [Backend README](backend/README.md) - Instruções detalhadas do backend
- [Frontend README](frontend/README.md) - Instruções detalhadas do frontend

---

## 🤝 Contribuição

Este projeto foi desenvolvido como parte do Tech Challenge da Fase 2 da Pós-Graduação em Inteligência Artificial para Desenvolvedores da FIAP.
