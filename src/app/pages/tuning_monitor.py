"""
Dashboard Streamlit — Monitor do Algoritmo Genético Co-Evolutivo.

Página: 🧬 Tuning Genético

Funcionalidades:
    - Configuração dos parâmetros do GA via widgets
    - Execução do GA com spinner de progresso
    - Gráficos co-evolutivos RF vs KNN na mesma linha do tempo
    - Tabela unificada de estatísticas por geração
    - Card de destaque com o melhor indivíduo encontrado
    - Download dos resultados (JSON + CSV)
"""

import json
import io
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🧬 Tuning Genético | SISVAN",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS customizado
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { font-size: 0.85rem; opacity: 0.75; margin: 0; }
    .metric-card h2 { font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0 0; }
    .badge-rf {
        background: #1565C0; color: white;
        border-radius: 6px; padding: 2px 10px; font-weight: 700;
    }
    .badge-knn {
        background: #E65100; color: white;
        border-radius: 6px; padding: 2px 10px; font-weight: 700;
    }
    .winner-card {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        border-radius: 12px; padding: 1.5rem; color: white;
    }
    .stopped-early {
        background: #f57f17; color: white;
        border-radius: 6px; padding: 3px 10px; font-weight: 600;
        display: inline-block; margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Título
# ---------------------------------------------------------------------------

st.title("🧬 Tuning Genético Co-Evolutivo")
st.markdown(
    "Otimização de hiperparâmetros via **GA Co-Evolutivo** com populações independentes de "
    "**RandomForest** e **KNN** competindo pelo fitness global."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — Configuração dos parâmetros
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Parâmetros do GA")

    st.subheader("Dados")
    data_path = st.text_input(
        "Caminho do CSV processado",
        value="data/processed/estado_nutricional_clean.csv",
    )
    target_col = st.text_input("Coluna target", value="TARGET")

    st.subheader("Populações")
    pop_size = st.slider("Pop. size por tipo", 4, 100, 20, step=2,
                         help="Número de indivíduos RF e KNN (total = 2×)")
    max_generations = st.slider("Máx. de gerações", 2, 50, 10)
    patience = st.slider("Patience (convergência)", 1, 20, 5,
                         help="Gerações sem melhoria antes de parar")
    k_folds = st.selectbox("K-Folds CV", [3, 5, 10], index=1)

    st.subheader("Operadores")
    aggressiveness = st.select_slider(
        "Agressividade da mutação",
        options=["low", "medium", "high"],
        value="medium",
    )
    elitism = st.toggle("Elitismo", value=True,
                        help="Preserva o melhor indivíduo a cada geração")
    cxpb = st.slider("P(crossover)", 0.1, 1.0, 0.7, step=0.05)
    mutpb = st.slider("P(mutação)", 0.05, 1.0, 0.3, step=0.05)
    indpb = st.slider("P(swap/gene) — cxUniform", 0.1, 1.0, 0.5, step=0.1)
    random_seed = st.number_input("Random seed", value=42, min_value=0)

    st.divider()
    run_button = st.button("🚀 Iniciar Tuning", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Helpers de visualização
# ---------------------------------------------------------------------------

def _make_evolution_chart(df: pd.DataFrame) -> go.Figure:
    """Gráfico co-evolutivo: F1 de RF e KNN por geração + melhor global."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["rf_best_f1"],
        name="RF — Best F1", mode="lines+markers",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=7, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["rf_avg_f1"],
        name="RF — Avg F1", mode="lines",
        line=dict(color="#1565C0", width=1, dash="dot"),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["knn_best_f1"],
        name="KNN — Best F1", mode="lines+markers",
        line=dict(color="#E65100", width=2),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["knn_avg_f1"],
        name="KNN — Avg F1", mode="lines",
        line=dict(color="#E65100", width=1, dash="dot"),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["global_best_f1"],
        name="⭐ Best Global F1", mode="lines+markers",
        line=dict(color="#FFD600", width=2.5),
        marker=dict(size=9, symbol="star"),
    ))

    # Marcação de parada antecipada
    early = df[df["stopped_early"]]
    if not early.empty:
        fig.add_vline(
            x=early.iloc[0]["generation"],
            line_dash="dash", line_color="#f57f17",
            annotation_text="Convergência", annotation_position="top right",
        )

    fig.update_layout(
        title="Evolução do F1 por Geração",
        xaxis_title="Geração",
        yaxis_title="F1 Weighted",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=380,
    )
    return fig


def _make_population_chart(df: pd.DataFrame) -> go.Figure:
    """Gráfico de proporção de sobreviventes RF vs KNN por geração."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["rf_count"],
        name="RF", marker_color="#1565C0",
    ))
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["knn_count"],
        name="KNN", marker_color="#E65100",
    ))
    fig.update_layout(
        barmode="stack",
        title="Proporção de Sobreviventes por Geração",
        xaxis_title="Geração",
        yaxis_title="Nº de indivíduos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=300,
    )
    return fig


def _best_card(result: dict) -> None:
    """Exibe card do melhor indivíduo encontrado."""
    best = result.get("best_individual")
    if not best:
        return

    tipo = best.get("type", "?")
    hp = best.get("hyperparams", {})
    f1 = best.get("fitness_f1", 0.0) or 0.0
    acc = best.get("fitness_acc", 0.0) or 0.0
    score = f1 * 0.6 + acc * 0.4

    badge = f'<span class="badge-rf">RF</span>' if tipo == "RF" else f'<span class="badge-knn">KNN</span>'
    stopped = result.get("reason", "")
    early_badge = '<span class="stopped-early">⚡ Convergência</span>' if stopped == "convergence" else ""

    st.markdown(f"""
    <div class="winner-card">
        <h3>🏆 Melhor Indivíduo Global {badge} {early_badge}</h3>
        <p style="font-size:1.4rem; font-weight:700; margin:0.5rem 0">
            F1 = {f1:.4f} &nbsp;|&nbsp; Acc = {acc:.4f} &nbsp;|&nbsp; Score = {score:.4f}
        </p>
        <p style="opacity:0.85; font-size:0.9rem; margin:0">
            Parâmetros: {json.dumps(hp, default=str)}
        </p>
        <p style="opacity:0.7; font-size:0.85rem; margin:0.4rem 0 0">
            Parou na geração {result.get("stopped_at", "?")} — motivo: <b>{stopped}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Execução do GA
# ---------------------------------------------------------------------------

if run_button:
    import pandas as pd_local
    from pathlib import Path

    # Valida dados
    if not Path(data_path).exists():
        st.error(f"❌ Arquivo não encontrado: `{data_path}`")
        st.stop()

    try:
        df_raw = pd_local.read_csv(data_path)
        if target_col not in df_raw.columns:
            st.error(f"❌ Coluna `{target_col}` não encontrada. Colunas disponíveis: {list(df_raw.columns)}")
            st.stop()

        X = df_raw.drop(columns=[target_col]).values
        y = df_raw[target_col].values
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

    # Importa GA após validação
    from src.models.genetic_algorithm import GeneticAlgorithm
    from src.models.ga_evaluator import fitness_score as _fitness_score
    from src.models.ga_persistence import save_ga_results

    st.info(
        f"🚀 Iniciando GA | Pop: {pop_size}/tipo ({pop_size*2} total) | "
        f"Máx. {max_generations} gerações | Patience: {patience} | "
        f"Mutação: {aggressiveness} | Elitismo: {'✅' if elitism else '❌'}"
    )

    with st.spinner("⏳ Executando Algoritmo Genético Co-Evolutivo..."):
        ga = GeneticAlgorithm(
            X=X, y=y,
            pop_size=pop_size,
            max_generations=max_generations,
            patience=patience,
            k_folds=k_folds,
            mutation_aggressiveness=aggressiveness,
            elitism=elitism,
            indpb=indpb,
            cxpb=cxpb,
            mutpb=mutpb,
            random_seed=random_seed,
        )
        results = ga.run()

    # Salva no session_state para reutilização
    st.session_state["ga_results"] = results
    st.success("✅ Tuning concluído!")

# ---------------------------------------------------------------------------
# Exibição dos resultados (persiste via session_state)
# ---------------------------------------------------------------------------

if "ga_results" in st.session_state:
    results = st.session_state["ga_results"]
    stats = results.get("generations_stats", [])

    if not stats:
        st.warning("Nenhuma estatística disponível.")
        st.stop()

    df_stats = pd.DataFrame([{
        "generation": s["generation"],
        "rf_count": s["rf"]["count"],
        "rf_best_f1": s["rf"]["best_f1"],
        "rf_avg_f1": s["rf"]["avg_f1"],
        "rf_best_score": s["rf"]["best_score"],
        "knn_count": s["knn"]["count"],
        "knn_best_f1": s["knn"]["best_f1"],
        "knn_avg_f1": s["knn"]["avg_f1"],
        "knn_best_score": s["knn"]["best_score"],
        "global_best_f1": s["global_best_f1"],
        "global_best_type": s["global_best_type"],
        "stopped_early": s.get("stopped_early", False),
    } for s in stats])

    st.divider()

    # ---- Métricas resumo ----
    best_ind = results.get("best_individual")
    best_f1 = best_ind.get("fitness_f1", 0.0) if best_ind else 0.0
    best_acc = best_ind.get("fitness_acc", 0.0) if best_ind else 0.0
    best_score = (best_f1 or 0.0) * 0.6 + (best_acc or 0.0) * 0.4
    best_type = best_ind.get("type", "?") if best_ind else "?"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏆 Melhor F1", f"{best_f1:.4f}")
    col2.metric("🎯 Melhor Acurácia", f"{best_acc:.4f}")
    col3.metric("⭐ Fitness Score", f"{best_score:.4f}")
    col4.metric("🔬 Tipo Vencedor", best_type)

    st.divider()

    # ---- Card do melhor ----
    _best_card(results)

    st.divider()

    # ---- Gráficos co-evolutivos ----
    st.subheader("📈 Evolução Co-Evolutiva")
    st.plotly_chart(_make_evolution_chart(df_stats), use_container_width=True)
    st.plotly_chart(_make_population_chart(df_stats), use_container_width=True)

    # ---- Tabela de estatísticas ----
    st.subheader("📊 Estatísticas por Geração")
    st.dataframe(
        df_stats.style.format({
            "rf_best_f1": "{:.4f}", "rf_avg_f1": "{:.4f}", "rf_best_score": "{:.4f}",
            "knn_best_f1": "{:.4f}", "knn_avg_f1": "{:.4f}", "knn_best_score": "{:.4f}",
            "global_best_f1": "{:.4f}",
        }).background_gradient(subset=["global_best_f1"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Downloads ----
    st.divider()
    st.subheader("⬇️ Download dos Resultados")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        json_payload = {
            "params": results.get("params", {}),
            "stopped_at": results.get("stopped_at"),
            "reason": results.get("reason"),
            "best_individual": results.get("best_individual"),
            "generations_stats": stats,
        }
        st.download_button(
            "📄 ga_history.json",
            data=json.dumps(json_payload, indent=2, default=str),
            file_name="ga_history.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        csv_buffer = io.StringIO()
        df_stats.to_csv(csv_buffer, index=False)
        st.download_button(
            "📊 ga_generation_stats.csv",
            data=csv_buffer.getvalue(),
            file_name="ga_generation_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # Estado inicial — nenhum run ainda
    st.info(
        "👈 Configure os parâmetros na barra lateral e clique em **🚀 Iniciar Tuning** para começar."
    )
    st.markdown("""
    ### Como funciona o GA Co-Evolutivo

    | Etapa | Descrição |
    |---|---|
    | **Inicialização** | `pop_size` indivíduos RF + `pop_size` indivíduos KNN gerados aleatoriamente |
    | **Avaliação global** | Todos avaliados via k-Fold CV (F1×0.6 + Acc×0.4) |
    | **Seleção por torneio** | RF e KNN **competem juntos** pelo fitness global |
    | **Crossover (cxUniform)** | Swap de hiperparâmetros só dentro do mesmo tipo |
    | **Mutação** | Perturbação gaussiana/categórica com intensidade configurável |
    | **Elitismo** | Preserva o melhor indivíduo global (opcional) |
    | **Critérios de parada** | `max_generations` ou `patience` gerações sem melhoria |
    """)
