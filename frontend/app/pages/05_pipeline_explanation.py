"""Página Streamlit — Explicação do Pipeline de Processamento."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="05 Pipeline | SISVAN",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Pipeline de Processamento de Dados")
st.markdown(
    """
    Esta página explica o fluxo completo para processar os dados nutricionais, treinar o modelo
    e criar uma sessão com o agente de saúde nutricional.
    """
)

st.divider()

st.header("📁 Pré-requisito: Arquivo de Dados Brutos")

st.warning(
    """
    **IMPORTANTE:** Antes de iniciar o pipeline, você deve colocar o arquivo CSV bruto 
    no diretório do backend:
    
    ```
    backend/data/raw/estado_nutricional_sao_paulo.csv
    ```
    
    Este arquivo deve conter os dados brutos do SISVAN que serão processados.
    """
)

st.divider()

st.header("🔄 Fluxo do Pipeline")

st.markdown(
    """
    O pipeline consiste em **4 etapas sequenciais** que devem ser executadas em ordem.
    Cada etapa depende da conclusão da anterior.
    """
)

# Step 1
st.subheader("1️⃣ Preprocessing")
st.markdown(
    """
    **Endpoint:** `POST /pipeline/preprocess`
    
    **O que faz:** Limpa e processa os dados brutos, criando:
    - Dataset processado: `data/processed/estado_nutricional_clean.csv`
    - Mapeamentos de features: `models/artifacts/mappings.json`
    
    **Parâmetros:** Não recebe parâmetros (usa caminhos fixos)
    
    **Retorno:** `{"job_id": "...", "status": "pending"}`
    
    **Como verificar status:** `GET /pipeline/jobs/{job_id}`
    
    **Resultado quando concluído:** Retorna os 10 primeiros registros do arquivo processado,
    os mapeamentos feitos e os caminhos dos arquivos gerados.
    """
)

with st.expander("Ver exemplo de resposta do preprocessing"):
    st.json({
        "job_id": "abc-123-def",
        "status": "pending"
    })

st.divider()

# Step 2
st.subheader("2️⃣ Tuning Genético")
st.markdown(
    """
    **Endpoint:** `POST /pipeline/tune`
    
    **O que faz:** Executa o Algoritmo Genético Co-Evolutivo para tunar hiperparâmetros
    do modelo (Random Forest + KNN) e treina o melhor modelo.
    
    **Parâmetros fixos:**
    - `--input data/processed/estado_nutricional_clean.csv`
    - `--output-model models/artifacts/best_model.joblib`
    - `--target TARGET`
    - `--pop-size 4`
    - `--max-generations 2`
    - `--patience 3`
    - `--k-folds 3`
    
    **Retorno:** `{"job_id": "...", "status": "pending"}`
    
    **Como verificar status:** `GET /pipeline/jobs/{job_id}`
    
    **Resultado quando concluído:** Modelo treinado salvo em `models/artifacts/best_model.joblib`
    
    ⚠️ **Validação:** Só pode ser chamado após o preprocessing ser concluído.
    """
)

st.divider()

# Step 3
st.subheader("3️⃣ Predições")
st.markdown(
    """
    **Endpoint:** `POST /pipeline/predict`
    
    **O que faz:** Usa o modelo treinado para gerar predições sobre o dataset processado.
    
    **Parâmetros fixos:**
    - `--input data/processed/estado_nutricional_clean.csv`
    - `--model models/artifacts/best_model.joblib`
    - `--output models/artifacts/predictions.csv`
    - `--target TARGET`
    
    **Retorno:** `{"job_id": "...", "status": "pending"}`
    
    **Como verificar status:** `GET /pipeline/jobs/{job_id}`
    
    **Resultado quando concluído:** CSV com predições salvo em `models/artifacts/predictions.csv`
    
    ⚠️ **Validação:** Só pode ser chamado após o tuning ser concluído.
    """
)

st.divider()

# Step 4
st.subheader("4️⃣ Criar Sessão do Agente LLM")
st.markdown(
    """
    **Endpoint:** `POST /llm/session/from-files`
    
    **O que faz:** Cria uma sessão do agente de saúde nutricional usando os arquivos
    gerados pelo pipeline.
    
    **Parâmetros:** Não recebe parâmetros (usa caminhos fixos):
    - CSV: `models/artifacts/predictions.csv`
    - Mappings: `models/artifacts/mappings.json`
    
    **Retorno:**
    ```json
    {
        "session_id": "...",
        "initial_report": "...",
        "row_count": 12345
    }
    ```
    
    ⚠️ **Validação:** Só pode ser chamado após as predições serem concluídas.
    """
)

st.divider()

st.header("📊 Status do Pipeline")

st.markdown(
    """
    Você pode verificar o estado atual do pipeline a qualquer momento:
    
    **Endpoint:** `GET /pipeline/status`
    
    **Retorno:**
    ```json
    {
        "preprocessing_completed": true,
        "tuning_completed": true,
        "predictions_completed": false
    }
    ```
    """
)

st.divider()

st.header("🚀 Exemplo de Fluxo Completo")

st.code("""
# 1. Coloque o arquivo bruto em:
backend/data/raw/estado_nutricional_sao_paulo.csv

# 2. Execute o preprocessing
curl -X POST http://localhost:8000/pipeline/preprocess
# Resposta: {"job_id": "abc-123", "status": "pending"}

# 3. Verifique o status do preprocessing
curl http://localhost:8000/pipeline/jobs/abc-123
# Aguarde até status ser "completed"

# 4. Execute o tuning
curl -X POST http://localhost:8000/pipeline/tune
# Resposta: {"job_id": "def-456", "status": "pending"}

# 5. Verifique o status do tuning
curl http://localhost:8000/pipeline/jobs/def-456
# Aguarde até status ser "completed"

# 6. Execute as predições
curl -X POST http://localhost:8000/pipeline/predict
# Resposta: {"job_id": "ghi-789", "status": "pending"}

# 7. Verifique o status das predições
curl http://localhost:8000/pipeline/jobs/ghi-789
# Aguarde até status ser "completed"

# 8. Crie a sessão do agente LLM
curl -X POST http://localhost:8000/llm/session/from-files
# Resposta: {"session_id": "...", "initial_report": "...", "row_count": 12345}

# 9. Use o session_id para conversar com o agente
curl -X POST http://localhost:8000/llm/chat \\
  -H "Content-Type: application/json" \\
  -d '{"session_id": "...", "question": "Qual a distribuição por idade?"}'
""", language="bash")

st.divider()

st.header("⚠️ Importante")

st.info(
    """
    - **Ordem obrigatória:** As etapas devem ser executadas na ordem mostrada acima.
    - **Validação automática:** A API retornará erro 400 se tentar pular etapas.
    - **Arquivos fixos:** Não é necessário especificar caminhos - eles são pré-definidos.
    - **Jobs assíncronos:** Preprocessing, tuning e predictions são assíncronos - use o job_id para acompanhar o progresso.
    - **Reset do pipeline:** Ao executar o preprocessing novamente, o estado do pipeline é resetado.
    """
)

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
