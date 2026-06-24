import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, TuningClient

load_dotenv()

st.set_page_config(
    page_title="SISVAN Nutricional",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 SISVAN — Análise Nutricional")
st.markdown(
    "Sistema de análise e previsão de estado nutricional com **Algoritmo Genético Co-Evolutivo** "
    "e **Agente de Saúde Nutricional** (ReAct + Gemini)."
)

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")

client = TuningClient()

try:
    health = client.health_check()
    st.success(f"✅ API online — status: {health.get('status', 'ok')}")
except ApiError as exc:
    st.error(f"❌ API indisponível: {exc}")
except Exception as exc:
    st.error(f"❌ Erro de conexão: {exc}")

st.divider()

st.markdown("""
### Páginas disponíveis

Use o menu lateral para navegar:

| Página | Descrição |
|--------|-----------|
| **🧬 Tuning Genético** | Dashboard interativo do GA Co-Evolutivo (RF vs KNN) |
| **💬 Agente Nutricional** | Chat com o agente ReAct para análise clínica dos dados |
""")

st.markdown(
    "Configure `BACKEND_URL` no `.env` apontando para a URL pública do backend em produção."
)
