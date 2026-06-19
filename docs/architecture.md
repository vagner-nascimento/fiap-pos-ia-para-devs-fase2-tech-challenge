# Arquitetura do Projeto - Fase 1 Refatorada

## Estrutura de Pastas

```
project-root/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml

├── config/
│   ├── __init__.py
│   ├── settings.py          # configurações centralizadas do projeto
│   ├── ga_params.yaml       # espaço de hiperparâmetros para GA
│   └── models_config.yaml   # configuração dos modelos candidatos

├── data/
│   ├── raw/                 # dados brutos originais
│   ├── processed/           # dataset limpo
│   └── .gitkeep

├── models/
│   ├── artifacts/           # modelos treinados, métricas, histórico de GA
│   ├── logs/                # logs de execução do tuning
│   ├── cache/               # cache de datasets intermediários
│   └── .gitkeep

├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── preprocessing.py
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── genetic_search.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   ├── pages/           # páginas adicionais do Streamlit
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing_monitor.py
│   │   │   ├── tuning_dashboard.py
│   │   │   ├── prediction.py
│   │   │   └── llm_interpretation.py
│   │   └── llm.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_converter.py
│       ├── persistence.py
│       ├── logger.py        # centralizar logging
│       └── validators.py    # validação de dados

├── scripts/
│   ├── run_preprocessing.py
│   ├── run_tuning.py
│   ├── run_app.py
│   └── run_all.py           # executa pipeline completo

├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_persistence.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_genetic_search.py
│   │   └── test_training_pipeline.py
│   └── conftest.py          # fixtures compartilhadas

└── docs/
    ├── architecture.md      # este arquivo
    ├── setup.md             # instruções de setup
    ├── preprocessing.md     # documentação do pipeline de dados
    └── api.md               # documentação da API interna
```

## Descrição dos Componentes

### `config/`
- **Propósito**: centralizar todas as configurações do projeto.
- **Arquivos principais**:
  - `settings.py` - variáveis de ambiente e paths do projeto.
  - `ga_params.yaml` - espaço de hiperparâmetros para o algoritmo genético.
  - `models_config.yaml` - definição dos modelos candidatos.

### `data/`
- **`raw/`** - dados brutos originais em formato original (CSV, rar, etc.).
- **`processed/`** - dataset limpo e pronto para treinamento.

### `models/`
- **`artifacts/`** - modelos treinados finais, métricas e histórico de evolução do GA.
- **`logs/`** - logs de execução do tuning genético.
- **`cache/`** - cache de datasets intermediários para evitar reprocessamento.

### `src/data/`
- **`ingest.py`** - leitura e conversão de arquivos de entrada.
- **`preprocessing.py`** - limpeza de dados, remoção de colunas irrelevantes, exclusão de gestantes.
- **`features.py`** - criação de `ESTADO_NUTRI`, `TARGET`, `PERC_GORDURA` e codificação de variáveis categóricas.

### `src/models/`
- **`registry.py`** - define os modelos candidatos e seus espaços de hiperparâmetros.
- **`genetic_search.py`** - implementação do algoritmo genético para buscar a melhor combinação de modelo e hiperparâmetros.
- **`train.py`** - treino do melhor modelo encontrado e persistência do artefato.
- **`evaluate.py`** - cálculo de métricas e geração de relatórios.

### `src/app/`
- **`streamlit_app.py`** - aplicação principal Streamlit.
- **`pages/`** - páginas adicionais:
  - `preprocessing_monitor.py` - monitoramento do pré-processamento de dados.
  - `tuning_dashboard.py` - dashboard com curva de fitness em tempo real.
  - `prediction.py` - painel de previsão interativa.
  - `llm_interpretation.py` - seção de interpretação com LLM.
- **`llm.py`** - adaptador para integração com modelo de linguagem.

### `src/utils/`
- **`file_converter.py`** - conversão de CSV para Parquet e tratamento de encoding.
- **`persistence.py`** - carregamento e salvamento de datasets e modelos.
- **`logger.py`** - logging centralizado para todo o projeto.
- **`validators.py`** - validação de dados de entrada.

### `scripts/`
- **`run_preprocessing.py`** - executa o pipeline de tratamento de dados.
- **`run_tuning.py`** - executa a busca genética e salva o melhor modelo.
- **`run_app.py`** - inicia a aplicação Streamlit.
- **`run_all.py`** - orquestra todo o pipeline em uma única chamada.

### `tests/`
- **`unit/`** - testes isolados de cada módulo.
- **`integration/`** - testes de pipeline completo.
- **`conftest.py`** - fixtures compartilhadas entre testes.

### `docs/`
- **`architecture.md`** - este arquivo com detalhamento da arquitetura.
- **`setup.md`** - instruções de setup e instalação.
- **`preprocessing.md`** - documentação do pipeline de dados.
- **`api.md`** - documentação da API interna.

## Fluxo de Execução

1. **Pré-processamento** (`run_preprocessing.py`)
   - Lê dados brutos de `data/raw/`
   - Aplica limpeza e engenharia de features
   - Salva dataset processado em `data/processed/`

2. **Tuning Genético** (`run_tuning.py`)
   - Carrega dataset processado
   - Executa algoritmo genético para otimizar modelo + hiperparâmetros
   - Salva melhor modelo em `models/artifacts/`
   - Gera logs em `models/logs/`

3. **Aplicação Streamlit** (`run_app.py`)
   - Inicia interface web
   - Carrega modelo treinado
   - Oferece dashboard de tuning, previsão interativa e interpretação via LLM

## Configurações e Variáveis de Ambiente

Usar `.env.example` como template para criar `.env` local com:
- `LLM_API_KEY` - chave de acesso ao modelo de linguagem
- `DATA_PATH` - caminho raiz para dados
- `MODEL_PATH` - caminho para salvar modelos
- `LOG_LEVEL` - nível de logging

## Decisões de Design

- **Sem notebooks**: toda lógica em módulos Python reutilizáveis.
- **Multi-página Streamlit**: separação clara de responsabilidades (tuning, previsão, interpretação).
- **Centralização de configurações**: facilitando ajustes e reproduzibilidade.
- **GA customizado**: implementação em `src/models/genetic_search.py` para evitar dependências extras.
- **Logging estruturado**: rastreabilidade completa do pipeline.
- **Testes divididos**: unit e integration para cobertura abrangente.
