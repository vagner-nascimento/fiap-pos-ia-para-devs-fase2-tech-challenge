# FIAP Pós-IA - Fase 2: Tech Challenge

Este repositório contém o projeto desenvolvido para o **Tech Challenge da Fase 2** da Pós-Graduação em Inteligência Artificial para Desenvolvedores da **FIAP**.

O objetivo do projeto é construir um sistema robusto de análise e previsão de estado nutricional a partir de dados do SISVAN (Sistema de Vigilância Alimentar e Nutricional), utilizando um pipeline clássico de Machine Learning combinado com um agente de inteligência artificial (LLM) baseado no padrão ReAct para interpretação clínica e análise estatística interativa.

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
Antes de utilizar o modelo de Machine Learning ou o painel de análise, é necessário pré-processar os dados brutos obtidos do SISVAN. O script lê os dados brutos em `data/raw`, realiza limpezas e engenharia de características e salva o resultado.

Exemplo de execução básica:
```bash
python scripts/run_preprocessing.py --input data/raw/estado_nutricional_sao_paulo.csv
```
> [!NOTE]  
> Caso os dados de entrada estejam em outra pasta ou com outro nome, forneça o caminho correspondente no parâmetro `--input`.

### 2. Testes Unitários e de Integração
Para garantir que toda a lógica de negócio e os agentes do LangChain estão funcionando sem erros na sua máquina, execute os testes unitários:

```bash
# Se estiver usando o ambiente virtual ativado
pytest

# Se estiver usando o uv
uv run pytest
```

---

## 📁 Estrutura de Diretórios Principal

A estrutura básica de arquivos do repositório é organizada da seguinte maneira:

```
├── README.md               # Este arquivo com instruções gerais
├── pyproject.toml          # Definições de empacotamento e dependências
├── .env.example            # Modelo de variáveis de ambiente
├── docs/                   # Documentação detalhada do projeto
│   └── architecture.md     # Detalhamento da arquitetura técnica e do Agente
├── src/                    # Código-fonte principal da aplicação
│   ├── app/                # Camada da aplicação (Streamlit e Integração LLM)
│   │   ├── llm.py          # Implementação do Agente de Saúde Nutricional ReAct
│   │   └── pages/          # Páginas e dashboards
│   ├── data/               # Ingestão e processamento de dados do SISVAN
│   └── utils/              # Ferramentas auxiliares (logger, validators, persistência)
├── scripts/                # Scripts executáveis do pipeline
│   └── run_preprocessing.py
└── tests/                  # Testes automatizados do projeto
```

Para uma visão detalhada das decisões de arquitetura e do comportamento do Agente ReAct, consulte o documento [docs/architecture.md](file:///c:/code/fiap-pos-ia/fase-2/fiap-pos-ia-para-devs-fase2-tech-challenge/docs/architecture.md).