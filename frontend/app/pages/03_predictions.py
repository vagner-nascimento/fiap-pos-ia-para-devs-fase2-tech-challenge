"""Página Streamlit — Predições do Pipeline."""

import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, PipelineClient

load_dotenv()

st.set_page_config(
    page_title="03 Predições | SISVAN",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Predições")
st.markdown(
    """
    Esta página executa as predições usando o modelo treinado.
    O modelo é aplicado ao dataset processado para gerar predições de estado nutricional.
    """
)

st.divider()

client = PipelineClient()

if "predictions_job_id" not in st.session_state:
    st.session_state.predictions_job_id = None
if "predictions_result" not in st.session_state:
    st.session_state.predictions_result = None

st.header("Executar Predições")

st.info(
    """
    ⚠️ **Pré-requisitos:**
    - O preprocessing deve ter sido concluído
    - O tuning deve ter sido concluído
    """
)

if st.button("🚀 Iniciar Predições", use_container_width=True, type="primary"):
    try:
        with st.spinner("Iniciando job de predições..."):
            response = client.run_predictions()
            st.session_state.predictions_job_id = response["job_id"]
            st.success(f"Job iniciado! Job ID: `{response['job_id']}`")
    except ApiError as exc:
        st.error(f"❌ Erro ao iniciar predições: {exc}")

st.divider()

st.header("Status do Job")

if st.session_state.predictions_job_id:
    job_id = st.session_state.predictions_job_id
    st.info(f"Job ID: `{job_id}`")
    
    if st.button("🔄 Atualizar Status", use_container_width=True):
        try:
            status = client.get_job_status(job_id)
            st.session_state.predictions_result = status
            
            if status["status"] == "completed":
                st.success("✅ Predições concluídas com sucesso!")
            elif status["status"] == "failed":
                st.error(f"❌ Job falhou: {status.get('error', 'Erro desconhecido')}")
            else:
                st.info(f"⏳ Status atual: {status['status']}")
        except ApiError as exc:
            st.error(f"❌ Erro ao consultar status: {exc}")
else:
    st.info("Clique em **🚀 Iniciar Predições** para começar.")

st.divider()

if st.session_state.predictions_result:
    result = st.session_state.predictions_result
    
    if result["status"] == "completed":
        st.subheader("📁 Arquivos Gerados")
        
        result_data = result.get("result", {})
        predictions_path = result_data.get("predictions_path", "N/A")
        
        st.markdown(f"""
        - **CSV de Predições:** `{predictions_path}`
        """)
        
        st.success("As predições foram geradas e salvas com sucesso. Você pode agora criar uma sessão com o agente LLM para analisar os resultados.")

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
