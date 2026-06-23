"""Página Streamlit — Chat com Agente de Saúde Nutricional."""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, LLMClient

load_dotenv()

st.set_page_config(
    page_title="💬 Agente Nutricional | SISVAN",
    page_icon="💬",
    layout="wide",
)

client = LLMClient()

st.title("💬 Agente de Saúde Nutricional")
st.markdown(
    "Converse com o agente **ReAct** (Gemini) para obter insights clínicos "
    "sobre os dados de estado nutricional."
)

if "llm_session_id" not in st.session_state:
    st.session_state.llm_session_id = None
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

with st.sidebar:
    st.header("📂 Dados")
    uploaded = st.file_uploader("CSV de pacientes", type=["csv"])
    mappings_text = st.text_area(
        "Mapeamentos JSON (opcional)",
        value='{"SG_SEXO": {"0": "Masculino", "1": "Feminino"}}',
        height=100,
    )

    if st.button("🔄 Nova sessão", use_container_width=True):
        if st.session_state.llm_session_id:
            try:
                client.close_session(st.session_state.llm_session_id)
            except ApiError:
                pass
        st.session_state.llm_session_id = None
        st.session_state.llm_messages = []
        st.rerun()

    if uploaded and st.button("🚀 Iniciar agente", use_container_width=True, type="primary"):
        try:
            mappings = json.loads(mappings_text) if mappings_text.strip() else {}
        except json.JSONDecodeError:
            st.error("Mapeamentos JSON inválidos.")
            st.stop()

        with st.spinner("Inicializando agente..."):
            try:
                session = client.create_session(
                    csv_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    mappings=mappings,
                )
                st.session_state.llm_session_id = session["session_id"]
                st.session_state.llm_messages = [
                    {"role": "assistant", "content": session["initial_report"]}
                ]
                st.success(f"Sessão criada — {session['row_count']} registros.")
            except ApiError as exc:
                st.error(str(exc))

    if st.session_state.llm_session_id:
        st.caption(f"Sessão: `{st.session_state.llm_session_id[:8]}...`")

st.divider()

if not st.session_state.llm_session_id:
    st.info("Faça upload de um CSV na barra lateral e clique em **🚀 Iniciar agente**.")
    st.stop()

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
