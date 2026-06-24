"""Página Streamlit — Chat com Agente LLM."""

import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, LLMClient

load_dotenv()

st.set_page_config(
    page_title="04 Agente LLM | SISVAN",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Agente de Saúde Nutricional")
st.markdown(
    "Converse com o agente **ReAct** (Gemini) para obter insights clínicos "
    "sobre os dados de predição de estado nutricional."
)

st.divider()

client = LLMClient()

if "llm_session_id" not in st.session_state:
    st.session_state.llm_session_id = None
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []
if "llm_initial_report" not in st.session_state:
    st.session_state.llm_initial_report = None
if "llm_row_count" not in st.session_state:
    st.session_state.llm_row_count = None

st.header("Sessão do Agente")

st.info(
    """
    ⚠️ **Pré-requisitos:**
    - O preprocessing deve ter sido concluído
    - O tuning deve ter sido concluído
    - As predições devem ter sido concluídas
    
    O agente usará os arquivos gerados pelo pipeline:
    - `models/artifacts/predictions.csv` (dados com predições)
    - `models/artifacts/mappings.json` (mapeamentos de features)
    """
)

if st.button("🚀 Criar Sessão do Agente", use_container_width=True, type="primary"):
    try:
        with st.spinner("Criando sessão do agente..."):
            session = client.create_session_from_files()
            st.session_state.llm_session_id = session["session_id"]
            st.session_state.llm_initial_report = session["initial_report"]
            st.session_state.llm_row_count = session["row_count"]
            st.session_state.llm_messages = [
                {"role": "assistant", "content": session["initial_report"]}
            ]
            st.success(f"Sessão criada! {session['row_count']} registros carregados.")
    except ApiError as exc:
        st.error(f"❌ Erro ao criar sessão: {exc}")

if st.session_state.llm_session_id:
    st.divider()
    
    col1, col2 = st.columns(2)
    col1.metric("Registros", st.session_state.llm_row_count)
    col2.metric("Session ID", st.session_state.llm_session_id[:8] + "...")
    
    if st.button("🔄 Fechar Sessão", use_container_width=True):
        try:
            client.close_session(st.session_state.llm_session_id)
            st.session_state.llm_session_id = None
            st.session_state.llm_messages = []
            st.session_state.llm_initial_report = None
            st.session_state.llm_row_count = None
            st.success("Sessão fechada!")
            st.rerun()
        except ApiError as exc:
            st.error(f"❌ Erro ao fechar sessão: {exc}")

st.divider()

if not st.session_state.llm_session_id:
    st.info("Clique em **🚀 Criar Sessão do Agente** para começar.")
    st.stop()

st.header("💬 Chat")

for msg in st.session_state.llm_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Faça uma pergunta sobre os dados nutricionais..."):
    st.session_state.llm_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                answer = client.chat(st.session_state.llm_session_id, question)
                st.markdown(answer)
                st.session_state.llm_messages.append({"role": "assistant", "content": answer})
            except ApiError as exc:
                st.error(str(exc))

st.divider()

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.info(f"Backend API: `{backend_url}`")
