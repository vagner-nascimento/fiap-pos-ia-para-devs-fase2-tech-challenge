import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, PipelineClient, TuningClient

load_dotenv()

st.set_page_config(
    page_title="Home | SISVAN Nutricional",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 SISVAN — Análise Nutricional")
st.markdown(
    "Sistema de análise e previsão de estado nutricional com **Algoritmo Genético Co-Evolutivo** "
    "e **Agente de Saúde Nutricional** (ReAct + Gemini)."
)

st.divider()

# Backend health check
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")

tuning_client = TuningClient()
try:
    health = tuning_client.health_check()
    st.success(f"✅ API online — status: {health.get('status', 'ok')}")
except ApiError as exc:
    st.error(f"❌ API indisponível: {exc}")
except Exception as exc:
    st.error(f"❌ Erro de conexão: {exc}")

st.divider()

# Pipeline status
st.header("📊 Status do Pipeline")
pipeline_client = PipelineClient()
try:
    status = pipeline_client.get_pipeline_status()
    col1, col2, col3 = st.columns(3)
    
    if status.get("preprocessing_completed"):
        col1.success("✅ Preprocessing")
    else:
        col1.warning("⏳ Preprocessing")
    
    if status.get("tuning_completed"):
        col2.success("✅ Tuning")
    else:
        col2.warning("⏳ Tuning")
    
    if status.get("predictions_completed"):
        col3.success("✅ Predições")
    else:
        col3.warning("⏳ Predições")
except ApiError:
    st.warning("Não foi possível verificar o status do pipeline.")

st.divider()

# Navigation cards
st.header("🚀 Navegação do Pipeline")

st.markdown("""
Execute as etapas do pipeline em ordem sequencial:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 1️⃣ 🔄 Preprocessing
    
    Limpa e processa os dados brutos do SISVAN.
    
    **Pré-requisito:** Arquivo CSV em `backend/data/raw/estado_nutricional_sao_paulo.csv`
    """)

with col2:
    st.markdown("""
    ### 2️⃣ 🧬 Tuning
    
    Executa o Algoritmo Genético Co-Evolutivo para tunar hiperparâmetros.
    
    **Pré-requisito:** Preprocessing concluído
    """)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    ### 3️⃣ 📊 Predições
    
    Gera predições usando o modelo treinado.
    
    **Pré-requisito:** Tuning concluído
    """)

with col4:
    st.markdown("""
    ### 4️⃣ 💬 Agente LLM
    
    Chat com o agente para análise dos dados de predição.
    
    **Pré-requisito:** Predições concluídas
    """)

st.divider()

st.markdown("""
### 📋 Documentação

Consulte a página **Pipeline** para obter detalhes técnicos sobre cada etapa do fluxo.
""")

st.divider()

st.caption(
    "Configure `BACKEND_URL` no `.env` apontando para a URL pública do backend em produção."
)
