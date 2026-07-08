"""Página Streamlit — Comparação de Modelos."""

import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, ModelComparisonClient

load_dotenv()

POLL_INTERVAL = 2  # segundos entre cada polling

st.set_page_config(
    page_title="04 Comparação de Modelos | SISVAN",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Comparação de Modelos")
st.markdown(
    """
    Esta página compara a assertividade de três modelos de predição:
    - **Best Model** (modelo tunado pelo algoritmo genético)
    - **KNN Original** (modelo KNN sem tuning)
    - **Random Forest Original** (modelo RF sem tuning)
    
    Métricas calculadas: accuracy, precision, recall e F1-score.
    """
)

st.divider()

client = ModelComparisonClient()

if "comparison_job_id" not in st.session_state:
    st.session_state.comparison_job_id = None
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

st.header("Executar Comparação")

st.info(
    """
    ⚠️ **Pré-requisito:**
    - As predições devem ter sido concluídas (execute na página "03 Predições")
    """
)

if st.button("🔍 Iniciar Comparação de Modelos", use_container_width=True, type="primary"):
    try:
        with st.spinner("Iniciando job de comparação..."):
            response = client.run_comparison()
            st.session_state.comparison_job_id = response["job_id"]
            st.session_state.comparison_result = None
            st.success(f"Job iniciado! Job ID: `{response['job_id']}`")
            st.rerun()
    except ApiError as exc:
        st.error(f"❌ Erro ao iniciar comparação: {exc}")

st.divider()

st.header("Status do Job")

if st.session_state.comparison_job_id:
    job_id = st.session_state.comparison_job_id
    st.info(f"Job ID: `{job_id}`")
    
    # Automatic polling if job is running or pending
    if st.session_state.comparison_result is None or st.session_state.comparison_result["status"] in ["pending", "running"]:
        try:
            with st.spinner(f"⏳ Processando comparação... (atualizando a cada {POLL_INTERVAL}s)"):
                status = client.get_job_status(job_id)
                st.session_state.comparison_result = status
                
                if status["status"] == "completed":
                    st.success("✅ Comparação concluída com sucesso!")
                elif status["status"] == "failed":
                    st.error(f"❌ Job falhou: {status.get('error', 'Erro desconhecido')}")
                else:
                    # Continue polling
                    time.sleep(POLL_INTERVAL)
                    st.rerun()
        except ApiError as exc:
            st.error(f"❌ Erro ao consultar status: {exc}")
    else:
        # Job already completed or failed, show manual refresh option
        status = st.session_state.comparison_result
        if status["status"] == "completed":
            st.success("✅ Comparação concluída com sucesso!")
        elif status["status"] == "failed":
            st.error(f"❌ Job falhou: {status.get('error', 'Erro desconhecido')}")
        
        if st.button("🔄 Atualizar Status", use_container_width=True):
            try:
                status = client.get_job_status(job_id)
                st.session_state.comparison_result = status
                st.rerun()
            except ApiError as exc:
                st.error(f"❌ Erro ao consultar status: {exc}")
else:
    st.info("Clique em **🔍 Iniciar Comparação de Modelos** para começar.")

st.divider()

# Display results when job is completed
if st.session_state.comparison_result and st.session_state.comparison_result["status"] == "completed":
    result = st.session_state.comparison_result["result"]
    
    st.header("📊 Resultados da Comparação")
    
    # Metrics table
    st.subheader("Métricas por Modelo")
    
    metrics_data = result["metrics"]
    
    # Prepare data for display
    display_data = []
    for model_name, metrics in metrics_data.items():
        display_data.append({
            "Modelo": model_name.replace("_", " ").title(),
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision (Macro)": f"{metrics['precision_macro']:.4f}",
            "Recall (Macro)": f"{metrics['recall_macro']:.4f}",
            "F1-Score (Macro)": f"{metrics['f1_macro']:.4f}",
            "Precision (Weighted)": f"{metrics['precision_weighted']:.4f}",
            "Recall (Weighted)": f"{metrics['recall_weighted']:.4f}",
            "F1-Score (Weighted)": f"{metrics['f1_weighted']:.4f}",
        })
    
    df_metrics = pd.DataFrame(display_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Plots
    st.subheader("Gráficos")
    
    plots = result["plots"]
    
    # Construct full URLs for plot images
    # Use localhost for browser-accessible URLs (st.image runs in browser, not container)
    backend_url = "http://localhost:8000"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Comparação de Métricas")
        if "metrics_comparison" in plots:
            plot_url = f"{backend_url}{plots['metrics_comparison']}"
            st.image(plot_url, use_container_width=True)
    
    with col2:
        st.markdown("### Distribuição de Classes")
        if "class_distribution" in plots:
            plot_url = f"{backend_url}{plots['class_distribution']}"
            st.image(plot_url, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Matrizes de Confusão")
    
    confusion_matrices = plots.get("confusion_matrices", {})
    
    if confusion_matrices:
        cols_cm = st.columns(len(confusion_matrices))
        for idx, (model_name, cm_path) in enumerate(confusion_matrices.items()):
            with cols_cm[idx]:
                st.markdown(f"### {model_name.replace('_', ' ').title()}")
                cm_url = f"{backend_url}{cm_path}"
                st.image(cm_url, use_container_width=True)
    
    # Class names
    st.subheader("Classes Identificadas")
    class_names = result.get("class_names", [])
    st.write(f"Classes: {', '.join(str(c) for c in class_names)}")
    
    # Timestamp
    st.caption(f"Relatório gerado em: {result.get('timestamp', 'N/A')}")

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
