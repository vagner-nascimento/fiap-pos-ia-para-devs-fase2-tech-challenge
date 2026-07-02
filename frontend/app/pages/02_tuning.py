"""Página Streamlit — Tuning Genético com Dashboard em Tempo Real."""

from __future__ import annotations

import os
import time

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.api_client import ApiError, PipelineClient

load_dotenv()

POLL_INTERVAL = 3  # segundos entre cada polling

st.set_page_config(
    page_title="02 Tuning | SISVAN",
    page_icon="🧬",
    layout="wide",
)

client = PipelineClient()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "tuning_job_id": None,
        "tuning_job_running": False,
        "tuning_result": None,
        "generations": [],        # lista acumulada de snapshots
        "last_generation": 0,     # cursor para leitura incremental
        "ga_params": {},          # parâmetros usados na execução atual
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar — configuração e disparo
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Parâmetros do GA")

    st.subheader("População")
    pop_size = st.slider(
        "População por tipo",
        min_value=2, max_value=100, value=4, step=2,
        help="Número de indivíduos por algoritmo (RF e KNN)",
    )

    st.subheader("Gerações")
    max_generations = st.slider(
        "Máximo de gerações",
        min_value=1, max_value=50, value=2,
        help="Número máximo de gerações do algoritmo genético",
    )
    patience = st.slider(
        "Patience (convergência)",
        min_value=1, max_value=20, value=3,
        help="Número de gerações sem melhoria antes de parar",
    )

    st.subheader("Validação")
    k_folds = st.selectbox(
        "K-Folds Cross-Validation",
        options=[2, 3, 5, 10], index=1,
        help="Número de folds para validação cruzada",
    )

    st.subheader("Operadores Genéticos")
    aggressiveness = st.select_slider(
        "Agressividade da mutação",
        options=["low", "medium", "high"], value="medium",
        help="Intensidade da mutação dos hiperparâmetros",
    )
    elitism = st.toggle(
        "Elitismo", value=True,
        help="Preservar o melhor indivíduo de cada geração",
    )
    crossover_probability = st.slider(
        "Probabilidade de Crossover",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Probabilidade de ocorrer crossover entre indivíduos",
    )
    mutation_probability = st.slider(
        "Probabilidade de Mutação",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Probabilidade de ocorrer mutação em um indivíduo",
    )
    individual_mutation_probability = st.slider(
        "Probabilidade de Mutação por Gene",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="Probabilidade de um gene específico sofrer mutação",
    )

    st.subheader("Reprodutibilidade")
    random_seed = st.number_input(
        "Random Seed", min_value=0, value=42, step=1,
        help="Semente aleatória para reprodutibilidade dos resultados",
    )

    st.divider()

    start_clicked = st.button(
        "🚀 Iniciar Tuning",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.tuning_job_running,
    )

    if start_clicked:
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
                job_id = client.start_pipeline_tuning_async(**params)
                st.session_state.tuning_job_id = job_id
                st.session_state.tuning_job_running = True
                st.session_state.tuning_result = None
                st.session_state.generations = []
                st.session_state.last_generation = 0
                st.session_state.ga_params = {
                    "Tamanho da população (por tipo)": pop_size,
                    "Gerações totais (máx)": max_generations,
                    "Patience (convergência)": patience,
                    "K-Folds": k_folds,
                    "Agressividade da mutação": aggressiveness,
                    "Elitismo": "✅ Ativo" if elitism else "❌ Inativo",
                    "P(crossover)": f"{crossover_probability:.2f}",
                    "P(mutação indivíduo)": f"{mutation_probability:.2f}",
                    "P(mutação gene)": f"{individual_mutation_probability:.2f}",
                    "Random seed": random_seed,
                }
                st.success(f"✅ Job iniciado! ID: `{job_id}`")
        except ApiError as exc:
            st.error(f"❌ Erro ao iniciar tuning: {exc}")


# ---------------------------------------------------------------------------
# Cabeçalho da página
# ---------------------------------------------------------------------------

st.title("🧬 Tuning Genético")
st.markdown(
    "Acompanhe a evolução do **Algoritmo Genético Co-Evolutivo** em tempo real. "
    "Configure os parâmetros na barra lateral e clique em **🚀 Iniciar Tuning**."
)
st.divider()


# ---------------------------------------------------------------------------
# Componentes visuais do dashboard
# ---------------------------------------------------------------------------

def render_metric_cards(generations: list[dict]) -> None:
    """Cartões de métricas no topo do dashboard."""
    current = generations[-1]
    previous = generations[-2] if len(generations) > 1 else None

    def delta(key: str) -> float | None:
        if previous is None:
            return None
        return current[key] - previous[key]

    cols = st.columns(5)
    metrics = [
        ("🏆 Melhor fitness",    "best_fitness",  4),
        ("📐 F1-weighted",       "best_f2",        4),
        ("🎯 Melhor acurácia",   "best_accuracy", 4),
        ("📊 Fitness médio",     "avg_fitness",   4),
        ("🔢 Geração atual",     "generation",    0),
    ]
    for col, (label, key, decimals) in zip(cols, metrics):
        val = current[key]
        d = delta(key)
        fmt = f"{val:.{decimals}f}" if decimals > 0 else str(int(val))
        d_fmt = (f"{d:+.4f}" if decimals > 0 else f"{d:+.0f}") if d is not None else None
        col.metric(label=label, value=fmt, delta=d_fmt)


def build_fitness_evolution_chart(generations: list[dict]) -> go.Figure:
    """Gráfico de linha com evolução do fitness por geração."""
    gens  = [g["generation"]   for g in generations]
    best  = [g["best_fitness"] for g in generations]
    avg   = [g["avg_fitness"]  for g in generations]
    std   = [g["std_fitness"]  for g in generations]

    upper = [a + s for a, s in zip(avg, std)]
    lower = [a - s for a, s in zip(avg, std)]

    fig = go.Figure()

    # Banda ±1σ
    fig.add_trace(go.Scatter(
        x=gens + gens[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(160, 200, 120, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ população",
        hoverinfo="skip",
    ))

    # Fitness médio
    fig.add_trace(go.Scatter(
        x=gens, y=avg,
        mode="lines",
        line=dict(color="#7aa3d4", width=1.5, dash="dot"),
        name="Fitness médio",
    ))

    # Melhor fitness
    fig.add_trace(go.Scatter(
        x=gens, y=best,
        mode="lines+markers",
        line=dict(color="#3266ad", width=2),
        marker=dict(size=5, color="#3266ad"),
        name="Melhor fitness",
    ))

    fig.update_layout(
        title="Evolução do fitness por geração",
        xaxis_title="Geração",
        yaxis_title="Fitness",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
        height=300,
        hovermode="x unified",
    )
    return fig


def build_f2_accuracy_scatter(generations: list[dict]) -> go.Figure:
    """Scatter F1-weighted vs acurácia colorido por geração."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[g["best_accuracy"] for g in generations],
        y=[g["best_f2"]       for g in generations],
        mode="markers",
        marker=dict(
            size=8,
            color=[g["generation"] for g in generations],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Geração", thickness=12),
            line=dict(width=0.5, color="white"),
        ),
        text=[f"Geração {g['generation']}" for g in generations],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Acurácia: %{x:.4f}<br>"
            "F1-weighted: %{y:.4f}<extra></extra>"
        ),
        name="",
    ))

    # Linha diagonal de referência
    fig.add_shape(
        type="line", x0=0.5, y0=0.5, x1=1.0, y1=1.0,
        line=dict(color="rgba(100,100,100,0.3)", dash="dot", width=1),
    )

    fig.update_layout(
        title="F1-weighted vs Acurácia por geração",
        xaxis=dict(title="Acurácia", range=[0.5, 1.0]),
        yaxis=dict(title="F1-weighted", range=[0.5, 1.0]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    return fig


def render_best_params_table(latest: dict) -> None:
    """Tabela com os melhores hiperparâmetros encontrados até agora."""
    st.markdown("**🔬 Melhores parâmetros encontrados**")
    model = latest.get("model_type", "—")
    params = latest.get("best_params", {})
    st.caption(f"Modelo: `{model}` — Geração {latest['generation']}")
    if params:
        st.dataframe(
            data={"Parâmetro": list(params.keys()), "Valor": list(params.values())},
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Parâmetros não disponíveis.")


def render_run_config() -> None:
    """Tabela com os parâmetros da execução atual."""
    st.markdown("**⚙️ Configuração da execução**")
    config = st.session_state.get("ga_params", {})
    if config:
        st.dataframe(
            data={"Parâmetro": list(config.keys()), "Valor": list(config.values())},
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Inicie o tuning para ver a configuração.")


def render_dashboard(generations: list[dict]) -> None:
    """Renderiza o dashboard completo com todos os dados acumulados."""
    if not generations:
        st.info("⏳ Aguardando dados da primeira geração...")
        return

    render_metric_cards(generations)
    st.divider()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            build_fitness_evolution_chart(generations),
            use_container_width=True,
            config={"displayModeBar": False},
            key="chart_fitness_evolution",
        )
    with col_right:
        st.plotly_chart(
            build_f2_accuracy_scatter(generations),
            use_container_width=True,
            config={"displayModeBar": False},
            key="chart_f2_accuracy",
        )

    col_params, col_config = st.columns(2)
    with col_params:
        render_best_params_table(generations[-1])
    with col_config:
        render_run_config()


# ---------------------------------------------------------------------------
# Loop de polling
# ---------------------------------------------------------------------------

def fetch_new_snapshots(job_id: str) -> tuple[list[dict], str]:
    """
    Busca snapshots novos via API e retorna (novos_snapshots, job_status).
    """
    try:
        data = client.get_generation_snapshots(
            job_id, since=st.session_state.last_generation
        )
        return data.get("snapshots", []), data.get("job_status", "running")
    except ApiError as exc:
        st.warning(f"⚠️ Erro ao buscar dados do AG: {exc}")
        return [], "running"


def check_and_update_job(job_id: str) -> bool:
    """
    Busca novos dados de gerações do AG e atualiza o session_state.
    Retorna True se o job ainda estiver executando, False caso contrário.
    """
    new_snapshots, job_status = fetch_new_snapshots(job_id)

    if new_snapshots:
        st.session_state.generations.extend(new_snapshots)
        st.session_state.last_generation = new_snapshots[-1]["generation"]

    if job_status in ("completed", "failed"):
        st.session_state.tuning_job_running = False
        st.session_state.tuning_result = job_status
        return False

    return True


# ---------------------------------------------------------------------------
# Corpo da página
# ---------------------------------------------------------------------------

job_id = st.session_state.tuning_job_id

if st.session_state.tuning_job_running and job_id:
    st.header("📡 Execução em andamento")
    st.caption(f"Job ID: `{job_id}`")

    # 1. Renderiza o dashboard com os dados já obtidos até o momento
    render_dashboard(st.session_state.generations)

    # 2. Verifica se há novos dados da API e atualiza o estado
    still_running = check_and_update_job(job_id)

    # 3. Informa o progresso e agenda o próximo ciclo com rerun()
    gen_count = len(st.session_state.generations)
    if still_running:
        if gen_count > 0:
            st.info(
                f"⏳ Executando... **{gen_count} geração(ões)** concluída(s) "
                f"| Próxima verificação em {POLL_INTERVAL}s"
            )
        else:
            st.info("⏳ Inicializando o algoritmo genético no backend...")
        
        time.sleep(POLL_INTERVAL)
        st.rerun()
    else:
        # Se terminou nesta rodada, forçamos o rerun imediato para entrar no fluxo de finalizado
        st.rerun()

elif st.session_state.generations:
    st.header("✅ Resultado final")
    st.caption(f"Job ID: `{job_id}`")
    render_dashboard(st.session_state.generations)

    if st.session_state.get("tuning_result") == "failed":
        st.error("❌ O job falhou. Verifique os logs do backend.")
    else:
        st.success(
            f"Otimização concluída com {len(st.session_state.generations)} gerações. "
            "Você pode agora executar as predições."
        )

else:
    st.header("📊 Dashboard")
    st.info(
        "Configure os parâmetros na barra lateral e clique em "
        "**🚀 Iniciar Tuning** para iniciar o algoritmo genético."
    )
    render_run_config()

st.divider()
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.caption(f"Backend API: `{backend_url}`")
