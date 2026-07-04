"""Página Streamlit — Preprocessing do Pipeline."""

import os
import time

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

# Inicializar session state
if "preprocessing_job_id" not in st.session_state:
    st.session_state.preprocessing_job_id = None
if "preprocessing_result" not in st.session_state:
    st.session_state.preprocessing_result = None
if "preprocessing_running" not in st.session_state:
    st.session_state.preprocessing_running = False

job_id = st.session_state.preprocessing_job_id

# ---------------------------------------------------------------------------
# Loop de Polling e Exibição de Execução em Andamento
# ---------------------------------------------------------------------------
if st.session_state.preprocessing_running and job_id:
    st.header("📡 Execução em andamento")
    st.caption(f"Job ID: `{job_id}`")
    
    # 1. Obter e exibir logs em tempo real
    try:
        logs_data = client.get_job_logs(job_id)
        logs = logs_data.get("logs", "")
        if logs:
            st.text_area("Logs de Execução", value=logs, height=350, disabled=True)
        else:
            st.info("⏳ Inicializando logs...")
    except ApiError as exc:
        st.warning(f"⚠️ Não foi possível carregar os logs: {exc}")
        
    # 2. Obter status do job e verificar se concluiu
    try:
        status_data = client.get_job_status(job_id)
        status = status_data["status"]
        if status in ("completed", "failed"):
            st.session_state.preprocessing_running = False
            st.session_state.preprocessing_result = status_data
            st.rerun()
        else:
            st.info("⏳ Executando pré-processamento no backend... A tela será atualizada automaticamente.")
            time.sleep(2)
            st.rerun()
    except ApiError as exc:
        st.error(f"❌ Erro ao consultar status do job: {exc}")
        time.sleep(2)
        st.rerun()

# ---------------------------------------------------------------------------
# Tela Inicial: Botão de disparo
# ---------------------------------------------------------------------------
else:
    st.subheader("Executar Preprocessing")
    
    start_disabled = st.session_state.preprocessing_running
    if st.button("🚀 Iniciar Preprocessing", use_container_width=True, type="primary", disabled=start_disabled):
        try:
            with st.spinner("Iniciando job de preprocessing..."):
                new_job_id = client.start_preprocessing_async()
                st.session_state.preprocessing_job_id = new_job_id
                st.session_state.preprocessing_running = True
                st.session_state.preprocessing_result = None
                st.success(f"Job iniciado! Job ID: `{new_job_id}`")
                st.rerun()
        except ApiError as exc:
            st.error(f"❌ Erro ao iniciar preprocessing: {exc}")

# ---------------------------------------------------------------------------
# Exibição de Resultados (Finalizado)
# ---------------------------------------------------------------------------
if st.session_state.preprocessing_result:
    st.divider()
    result = st.session_state.preprocessing_result
    
    if result["status"] == "completed":
        st.success("✅ Preprocessing concluído com sucesso!")
        
        # Mostrar Logs do Job em um Expander para auditoria
        try:
            logs_data = client.get_job_logs(st.session_state.preprocessing_job_id)
            logs = logs_data.get("logs", "")
            if logs:
                with st.expander("📝 Ver logs completos do processamento", expanded=False):
                    st.text_area("Logs Finais", value=logs, height=250, disabled=True)
        except Exception:
            pass
            
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
        
    elif result["status"] == "failed":
        st.error(f"❌ O job de pré-processamento falhou!")
        st.error(f"Erro: {result.get('error', 'Erro desconhecido')}")
        
        # Mostrar logs para ajudar a debugar
        try:
            logs_data = client.get_job_logs(st.session_state.preprocessing_job_id)
            logs = logs_data.get("logs", "")
            if logs:
                st.subheader("📝 Logs de Erro")
                st.text_area("Logs de Execução", value=logs, height=350, disabled=True)
        except Exception:
            pass

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
