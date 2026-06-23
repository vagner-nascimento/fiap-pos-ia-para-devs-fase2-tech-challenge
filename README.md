# FIAP Pós-IA - Fase 2: Tech Challenge

Este repositório contém o projeto desenvolvido para o **Tech Challenge da Fase 2** da Pós-Graduação em Inteligência Artificial para Desenvolvedores da **FIAP**.

O objetivo do projeto é construir um sistema robusto de análise e previsão de estado nutricional a partir de dados do SISVAN (Sistema de Vigilância Alimentar e Nutricional), utilizando:
- Um **Algoritmo Genético Co-Evolutivo** para otimização de hiperparâmetros de classificação
- Um pipeline clássico de **Machine Learning** (RandomForest e KNN)
- Um **Agente de Inteligência Artificial** (LLM) baseado no padrão ReAct para interpretação clínica e análise estatística interativa

---

## 🚀 Requisitos e Dependências

Para rodar a aplicação, você precisará de:

1. **Python >= 3.13** (conforme especificado no [pyproject.toml](file:///c:/code/fiap-pos-ia/fase-2/fiap-pos-ia-para-devs-fase2-tech-challenge/pyproject.toml)).
2. **Gerenciador de Dependências**: Recomendamos o **`uv`** (pois o projeto já inclui um arquivo de trava `uv.lock`), mas você também pode utilizar o `pip` clássico com um ambiente virtual (`venv`).
3. **Chave de API do Google AI Studio** (Gemini) para alimentar o Agente de Saúde Nutricional.

---

## 🛠️ Configuração do Ambiente e Instalação

Siga os passos abaixo para preparar o ambiente e rodar o projeto localmente:

### Passo 1: Clonar o Repositório
Abra o seu terminal e execute:
```bash
git clone <url-do-repositorio>
cd fiap-pos-ia-para-devs-fase2-tech-challenge
```

### Passo 2: Renomear e Configurar o arquivo `.env`
O projeto necessita de variáveis de ambiente configuradas para funcionar corretamente. Fornecemos um arquivo modelo chamado `.env.example`. Você precisa criar uma cópia dele chamada `.env`.

* **No Windows (PowerShell):**
  ```powershell
  copy .env.example .env
  ```
* **No Windows (Prompt de Comando - CMD):**
  ```cmd
  copy .env.example .env
  ```
* **No Linux / macOS:**
  ```bash
  cp .env.example .env
  ```

---

## 🔑 Como Obter a Chave da API do Google AI (Gemini)

O agente inteligente utiliza modelos da família Gemini. Siga este passo a passo detalhado para gerar sua chave de acesso gratuita:

1. Acesse o site do **[Google AI Studio](https://aistudio.google.com/)**.
2. Faça login com a sua conta Google corporativa ou pessoal.
3. No painel lateral esquerdo (ou no topo da página), clique em **"Get API key"** (Obter chave de API).
4. Clique no botão **"Create API key"** (Criar chave de API).
5. O sistema exibirá uma janela de diálogo:
   - Você pode selecionar a opção para gerar a chave em um **novo projeto do Google Cloud** (opção padrão e mais rápida) ou associar a um projeto do Google Cloud existente.
6. Após a criação, uma chave alfanumérica será exibida (geralmente começando com `AIzaSy...`). Clique em **"Copy"** (Copiar) para salvá-la em sua área de transferência.
7. Abra o arquivo `.env` recém-criado em seu editor de texto (como o VS Code ou bloco de notas).
8. Procure pela linha `LLM_API_KEY=your_api_key_here` e substitua pelo valor copiado:
   ```env
   LLM_API_KEY=AIzaSySuaChaveGeradaAqui...
   ```
9. Aproveite para verificar e ajustar a variável `LLM_MODEL` se desejar. Por padrão, a aplicação é compatível com modelos como `gemini-2.5-flash` ou `gemini-1.5-pro`. Certifique-se de usar um modelo Gemini válido suportado pela biblioteca `langchain-google-genai` (exemplo no `.env`):
   ```env
   LLM_MODEL=gemini-2.5-flash
   ```

---

## 📦 Instalação de Dependências

### Opção A: Usando o `uv` (Recomendado)
O `uv` é um instalador e gerenciador de pacotes extremamente rápido escrito em Rust.
1. Se você já possui o `uv` instalado, basta rodar o comando abaixo para criar o ambiente virtual e instalar as dependências de forma otimizada:
   ```bash
   uv sync
   ```
2. Para rodar comandos ou scripts usando este ambiente virtual gerenciado pelo `uv`, utilize o prefixo `uv run`:
   ```bash
   uv run python scripts/run_preprocessing.py --help
   ```

### Opção B: Usando `pip` + `venv` (Tradicional)
Se preferir usar o ambiente virtual nativo do Python:
1. Crie o ambiente virtual:
   ```bash
   python -m venv .venv
   ```
2. Ative o ambiente virtual:
   * **No Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **No Windows (CMD):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **No Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
3. Instale as dependências:
   ```bash
   pip install --upgrade pip
   # Usando o pyproject.toml
   pip install .
   # Ou instalando o arquivo de requisitos direto
   pip install -r requirements.txt
   ```

---

## 🖥️ Como Executar a Aplicação

### 1. Pré-processamento de Dados (`scripts/run_preprocessing.py`)
Antes de utilizar o modelo de Machine Learning ou o painel de análise, é necessário pré-processar os dados brutos obtidos do SISVAN.

Exemplo de execução básica:
```bash
python scripts/run_preprocessing.py --input data/raw/estado_nutricional_sao_paulo.csv
```
> [!NOTE]
> Caso os dados de entrada estejam em outra pasta ou com outro nome, forneça o caminho correspondente no parâmetro `--input`.

### 2. Tuning de Hiperparâmetros com GA Co-Evolutivo (`scripts/run_tuning.py`)
Após o pré-processamento, execute o Algoritmo Genético para encontrar os melhores hiperparâmetros:

```bash
# Execução padrão
uv run python scripts/run_tuning.py \
  --input data/processed/estado_nutricional_clean.csv

# Com parâmetros customizados
uv run python scripts/run_tuning.py \
  --input data/processed/estado_nutricional_clean.csv \
  --pop-size 20 --max-generations 10 --patience 5 \
  --aggressiveness medium --elitism \
  --cxpb 0.7 --mutpb 0.3 --indpb 0.5 --random-seed 42

# Sem elitismo
uv run python scripts/run_tuning.py \
  --input data/processed/estado_nutricional_clean.csv \
  --no-elitism --aggressiveness high
```

O script gera:
- `models/artifacts/best_model.joblib` — pipeline sklearn do modelo vencedor (RF ou KNN)
- `models/logs/ga_history.json` — histórico completo do GA
- `models/logs/ga_generation_stats.csv` — tabela de evolutção por geração

> [!NOTE]
> Com `pop_size=20` e `max_generations=10` (defaults), o tuning pode levar 1–2 horas nos dados completos do SISVAN. Use `--pop-size 4 --max-generations 2` para uma validação rápida.

### 3. Testes Automatizados
Para garantir que toda a lógica de negócio e os agentes do LangChain estão funcionando:

```bash
# Todos os testes (60 no total)
uv run pytest

# Apenas unitários (rápidos)
uv run pytest tests/unit/ -v

# Apenas integração (GA co-evolutivo end-to-end)
uv run pytest tests/integration/ -v
```

---

## 📁 Estrutura de Diretórios Principal

```
├── README.md               # Este arquivo com instruções gerais
├── pyproject.toml          # Definições de empacotamento e dependências
├── .env.example            # Modelo de variáveis de ambiente
├── docs/
│   ├── architecture.md     # Arquitetura técnica detalhada e fluxo do GA
│   └── adr.md              # Architecture Decision Records (decisões de design)
├── src/
│   ├── models/             # GA Co-Evolutivo
│   │   ├── individuo.py        # IndividuoRF e IndividuoKNN (pipeline sklearn)
│   │   ├── ga_operators.py     # Crossover uniforme e mutação por tipo
│   │   ├── ga_evaluator.py     # Fitness k-Fold CV (F1×0.6 + Acc×0.4)
│   │   ├── genetic_algorithm.py # Orquestrador co-evolutivo
│   │   └── ga_persistence.py   # Save/load JSON, CSV, joblib
│   ├── app/                # Streamlit + Agente LLM ReAct
│   │   ├── llm.py
│   │   └── pages/tuning_monitor.py  # Dashboard 🧬 Tuning Genético
│   └── utils/              # logger, persistence, validators
├── scripts/
│   ├── run_preprocessing.py
│   └── run_tuning.py       # CLI do GA Co-Evolutivo
└── tests/
    ├── unit/               # 44 testes (operadores + evaluator)
    └── integration/        # 16 testes (GA end-to-end)
```

Para uma visão detalhada da arquitetura técnica e do comportamento do Agente ReAct, consulte o documento [docs/architecture.md](docs/architecture.md).

Para as decisões de design do Algoritmo Genético e da arquitetura geral, consulte o [docs/adr.md](docs/adr.md).