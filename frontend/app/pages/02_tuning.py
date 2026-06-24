"""Página Streamlit — Tuning Genético do Pipeline."""

import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, PipelineClient

load_dotenv()

st.set_page_config(
    page_title="02 Tuning | SISVAN",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Tuning Genético")
st.markdown(
    """
    Esta página executa o tuning genético com parâmetros configuráveis.
    O Algoritmo Genético Co-Evolutivo otimiza hiperparâmetros do modelo (Random Forest + KNN).
    """
)

st.divider()

client = PipelineClient()

if "tuning_job_id" not in st.session_state:
    st.session_state.tuning_job_id = None
if "tuning_result" not in st.session_state:
    st.session_state.tuning_result = None

st.header("⚙️ Parâmetros do Tuning")

with st.sidebar:
    st.header("Parâmetros do GA")
    
    st.subheader("População")
    pop_size = st.slider(
        "População por tipo",
        min_value=2,
        max_value=100,
        value=4,
        step=2,
        help="Número de indivíduos por algoritmo (RF e KNN)"
    )
    
    st.subheader("Gerações")
    max_generations = st.slider(
        "Máximo de gerações",
        min_value=1,
        max_value=50,
        value=2,
        help="Número máximo de gerações do algoritmo genético"
    )
    patience = st.slider(
        "Patience (convergência)",
        min_value=1,
        max_value=20,
        value=3,
        help="Número de gerações sem melhoria antes de parar"
    )
    
    st.subheader("Validação")
    k_folds = st.selectbox(
        "K-Folds Cross-Validation",
        options=[2, 3, 5, 10],
        index=1,
        help="Número de folds para validação cruzada"
    )
    
    st.subheader("Operadores Genéticos")
    aggressiveness = st.select_slider(
        "Agressividade da mutação",
        options=["low", "medium", "high"],
        value="medium",
        help="Intensidade da mutação dos hiperparâmetros"
    )
    elitism = st.toggle(
        "Elitismo",
        value=True,
        help="Preservar o melhor indivíduo de cada geração"
    )
    crossover_probability = st.slider(
        "Probabilidade de Crossover",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Probabilidade de ocorrer crossover entre indivíduos"
    )
    mutation_probability = st.slider(
        "Probabilidade de Mutação",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Probabilidade de ocorrer mutação em um indivíduo"
    )
    individual_mutation_probability = st.slider(
        "Probabilidade de Mutação por Gene",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Probabilidade de um gene específico sofrer mutação"
    )
    
    st.subheader("Reprodutibilidade")
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        value=42,
        step=1,
        help="Semente aleatória para reprodutibilidade dos resultados"
    )
    
    st.divider()
    
    if st.button("🚀 Iniciar Tuning", use_container_width=True, type="primary"):
        params = {
            "pop_size": pop_size,
            "max_generations": max_generations,
            "patience": patience,
            "k_folds": k_folds,
            "aggressiveness": aggressiveness,
            "elitism": elitism,
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "individual_mutation_probability": individual_mutation_probability,
            "random_seed": random_seed,
        }
        
        try:
            with st.spinner("Iniciando job de tuning..."):
                response = client.run_tuning(**params)
                st.session_state.tuning_job_id = response["job_id"]
                st.success(f"Job iniciado! Job ID: `{response['job_id']}`")
        except ApiError as exc:
            st.error(f"❌ Erro ao iniciar tuning: {exc}")

st.divider()

st.header("📊 Status do Job")

if st.session_state.tuning_job_id:
    job_id = st.session_state.tuning_job_id
    st.info(f"Job ID: `{job_id}`")
    
    if st.button("🔄 Atualizar Status", use_container_width=True):
        try:
            status = client.get_job_status(job_id)
            st.session_state.tuning_result = status
            
            if status["status"] == "completed":
                st.success("✅ Tuning concluído com sucesso!")
            elif status["status"] == "failed":
                st.error(f"❌ Job falhou: {status.get('error', 'Erro desconhecido')}")
            else:
                st.info(f"⏳ Status atual: {status['status']}")
        except ApiError as exc:
            st.error(f"❌ Erro ao consultar status: {exc}")
else:
    st.info("Configure os parâmetros na barra lateral e clique em **🚀 Iniciar Tuning**.")

st.divider()

if st.session_state.tuning_result:
    result = st.session_state.tuning_result
    
    if result["status"] == "completed":
        st.subheader("📁 Arquivos Gerados")
        
        result_data = result.get("result", {})
        model_path = result_data.get("model_path", "N/A")
        
        st.markdown(f"""
        - **Modelo Treinado:** `{model_path}`
        """)
        
        st.success("O modelo foi treinado e salvo com sucesso. Você pode agora executar as predições.")

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
