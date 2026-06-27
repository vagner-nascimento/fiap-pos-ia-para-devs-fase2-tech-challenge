"""Página Streamlit — Preprocessing do Pipeline."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, PipelineClient

load_dotenv()

st.set_page_config(
    page_title="01 Preprocessing | SISVAN",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Preprocessing dos Dados")
st.markdown(
    """
    Esta página executa o preprocessing dos dados brutos do SISVAN.
    O processo limpa e processa os dados, criando arquivos para as etapas subsequentes.
    """
)

st.divider()

st.header("📁 Pré-requisito")

st.warning(
    """
    **IMPORTANTE:** Antes de iniciar o preprocessing, você deve colocar o arquivo bruto 
    no diretório do backend (suporta .csv ou .rar):
    
    ```
    backend/data/raw/estado_nutricional_sao_paulo.csv
    ```
    
    ou
    
    ```
    backend/data/raw/estado_nutricional_sao_paulo.rar
    ```
    
    Se fornecido um arquivo .rar, ele será automaticamente extraído antes do processamento.
    """
)

st.divider()

client = PipelineClient()

if "preprocessing_job_id" not in st.session_state:
    st.session_state.preprocessing_job_id = None
if "preprocessing_result" not in st.session_state:
    st.session_state.preprocessing_result = None

st.subheader("Executar Preprocessing")

if st.button("🚀 Iniciar Preprocessing", use_container_width=True, type="primary"):
    try:
        with st.spinner("Iniciando job de preprocessing..."):
            response = client.run_preprocessing()
            st.session_state.preprocessing_job_id = response["job_id"]
            st.success(f"Job iniciado! Job ID: `{response['job_id']}`")
    except ApiError as exc:
        st.error(f"❌ Erro ao iniciar preprocessing: {exc}")

st.divider()

if st.session_state.preprocessing_job_id:
    st.subheader("Status do Job")
    
    job_id = st.session_state.preprocessing_job_id
    st.info(f"Job ID: `{job_id}`")
    
    if st.button("🔄 Atualizar Status", use_container_width=True):
        try:
            status = client.get_job_status(job_id)
            st.session_state.preprocessing_result = status
            
            if status["status"] == "completed":
                st.success("✅ Preprocessing concluído com sucesso!")
            elif status["status"] == "failed":
                st.error(f"❌ Job falhou: {status.get('error', 'Erro desconhecido')}")
            else:
                st.info(f"⏳ Status atual: {status['status']}")
        except ApiError as exc:
            st.error(f"❌ Erro ao consultar status: {exc}")

st.divider()

if st.session_state.preprocessing_result:
    result = st.session_state.preprocessing_result
    
    if result["status"] == "completed":
        st.subheader("📊 Resultados")
        
        result_data = result.get("result", {})
        
        col1, col2 = st.columns(2)
        col1.metric("Total de Registros", result_data.get("total_rows", 0))
        col2.metric("Arquivo Processado", result_data.get("processed_csv_path", "N/A"))
        
        st.divider()
        
        st.subheader("📋 Primeiros 10 Registros")
        first_10 = result_data.get("first_10_rows", [])
        if first_10:
            df = pd.DataFrame(first_10)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Nenhum registro disponível.")
        
        st.divider()
        
        st.subheader("🗺️ Mapeamentos de Features")
        mappings = result_data.get("mappings", {})
        if mappings:
            st.json(mappings)
        else:
            st.warning("Nenhum mapeamento disponível.")
        
        st.divider()
        
        st.subheader("📁 Arquivos Gerados")
        st.markdown(f"""
        - **CSV Processado:** `{result_data.get('processed_csv_path')}`
        - **Mapeamentos:** `{result_data.get('mappings_path')}`
        """)

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
