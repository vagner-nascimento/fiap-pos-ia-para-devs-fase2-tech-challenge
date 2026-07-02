# Plano de implementação: dashboard em tempo real para algoritmo genético

## Contexto

A aplicação atual executa um algoritmo genético para tunar hiperparâmetros de modelos `KNNClassifier` e `RandomForestClassifier`. O loop do AG roda em um job assíncrono. A interface é feita em Streamlit, que já faz polling periódico para verificar se o job concluiu.

O objetivo deste plano é estender o código existente para que o dashboard exiba, em tempo real, a evolução de cada geração do algoritmo — incluindo fitness, F2 score e acurácia — sem alterar a lógica do AG nem o mecanismo de disparo do job.

O layout alvo do dashboard é o seguinte:

<!-- PLACEHOLDER: cole aqui a imagem ou descrição do mockup do dashboard -->
<!-- Exemplo: ![Mockup do dashboard](./mockup_dashboard.png) -->

## [Dashboard mockup](genetic_algo_dashboard_mockup.html)

## Visão geral das mudanças

As alterações se dividem em três camadas independentes que podem ser implementadas em sequência:

1. **Persistência por geração** — o job grava um snapshot ao final de cada geração
2. **Leitura incremental no Streamlit** — o loop de polling acumula snapshots em `st.session_state`
3. **Renderização do dashboard** — novos componentes visuais substituem o indicador de status atual

---

## Camada 1 — Persistência dos dados de geração no job assíncrono

### Estrutura do snapshot por geração

Ao final do loop de cada geração, o job deve persistir um objeto com a seguinte estrutura:

```python
{
    "generation": int,           # número da geração atual (começa em 1)
    "best_fitness": float,       # melhor valor de fitness da geração
    "avg_fitness": float,        # média de fitness da população
    "std_fitness": float,        # desvio padrão do fitness da população
    "best_f2": float,            # F2 score do melhor indivíduo
    "best_accuracy": float,      # acurácia do melhor indivíduo
    "best_params": dict,         # hiperparâmetros do melhor indivíduo
    "model_type": str,           # "KNN" ou "RandomForest"
    "timestamp": str             # ISO 8601, ex: "2026-07-01T14:23:11Z"
}
```

### Onde persistir

Use o mesmo mecanismo de estado que o job já utiliza para sinalizar conclusão. As opções em ordem de preferência:

**Opção A — Redis (recomendada se já estiver em uso):**

```python
import json, redis

r = redis.Redis(...)

def save_generation_snapshot(job_id: str, snapshot: dict):
    key = f"ag_job:{job_id}:generations"
    r.rpush(key, json.dumps(snapshot))
    r.expire(key, 86400)  # TTL de 24h
```

**Opção B — Banco de dados relacional:**

```sql
CREATE TABLE ag_generation_snapshots (
    id          SERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL,
    generation  INTEGER NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON ag_generation_snapshots (job_id, generation);
```

```python
def save_generation_snapshot(job_id: str, snapshot: dict):
    db.execute(
        "INSERT INTO ag_generation_snapshots (job_id, generation, payload) VALUES (%s, %s, %s)",
        (job_id, snapshot["generation"], json.dumps(snapshot))
    )
```

**Opção C — Arquivo JSON (desenvolvimento/prototipagem):**

```python
import json, os

def save_generation_snapshot(job_id: str, snapshot: dict):
    path = f"/tmp/ag_job_{job_id}_generations.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
```

### Onde chamar no loop do AG

```python
# Dentro do loop do algoritmo genético, ao final de cada geração:

for generation in range(1, max_generations + 1):
    # ... lógica existente do AG ...

    snapshot = {
        "generation": generation,
        "best_fitness": float(best_fitness),
        "avg_fitness": float(np.mean(fitness_scores)),
        "std_fitness": float(np.std(fitness_scores)),
        "best_f2": float(best_f2),
        "best_accuracy": float(best_accuracy),
        "best_params": best_individual.params,
        "model_type": model_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    save_generation_snapshot(job_id, snapshot)
```

---

## Camada 2 — Leitura incremental no Streamlit

### Função de leitura dos snapshots

Adicionar uma função que busca apenas as gerações ainda não carregadas, usando o índice da última geração conhecida como cursor:

```python
def fetch_new_generation_snapshots(job_id: str, since_generation: int = 0) -> list[dict]:
    """
    Retorna apenas os snapshots de gerações posteriores a `since_generation`.
    Adaptar o corpo conforme a opção de persistência escolhida na Camada 1.
    """
    # Opção A — Redis:
    # raw = r.lrange(f"ag_job:{job_id}:generations", since_generation, -1)
    # return [json.loads(item) for item in raw]

    # Opção B — Banco:
    # rows = db.fetchall(
    #     "SELECT payload FROM ag_generation_snapshots "
    #     "WHERE job_id = %s AND generation > %s ORDER BY generation",
    #     (job_id, since_generation)
    # )
    # return [row["payload"] for row in rows]

    # Opção C — Arquivo JSONL:
    # snapshots = []
    # with open(f"/tmp/ag_job_{job_id}_generations.jsonl") as f:
    #     for line in f:
    #         s = json.loads(line)
    #         if s["generation"] > since_generation:
    #             snapshots.append(s)
    # return snapshots
    pass
```

### Inicialização do estado da sessão

```python
def init_session_state():
    defaults = {
        "generations": [],          # lista acumulada de snapshots
        "last_generation": 0,       # cursor para leitura incremental
        "job_running": False,
        "job_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

### Loop de polling estendido

Substituir o loop de polling existente por este padrão. O `st.empty()` garante que o dashboard seja atualizado no lugar, sem piscar:

```python
def run_polling_loop(job_id: str, poll_interval: int = 3):
    dashboard_placeholder = st.empty()

    while True:
        # Busca apenas as gerações novas desde o último poll
        new_snapshots = fetch_new_generation_snapshots(
            job_id,
            since_generation=st.session_state.last_generation
        )

        if new_snapshots:
            st.session_state.generations.extend(new_snapshots)
            st.session_state.last_generation = new_snapshots[-1]["generation"]

        # Renderiza o dashboard com todos os dados acumulados
        with dashboard_placeholder.container():
            render_dashboard(st.session_state.generations)

        # Verifica se o job concluiu
        if is_job_done(job_id):
            st.session_state.job_running = False
            break

        time.sleep(poll_interval)
```

---

## Camada 3 — Componentes visuais do dashboard

### Estrutura geral da função `render_dashboard`

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_dashboard(generations: list[dict]):
    if not generations:
        st.info("Aguardando dados da primeira geração...")
        return

    # --- Métricas resumo (topo) ---
    render_metric_cards(generations)

    # --- Gráficos principais ---
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            build_fitness_evolution_chart(generations),
            use_container_width=True,
            config={"displayModeBar": False}
        )
    with col_right:
        st.plotly_chart(
            build_f2_accuracy_scatter(generations),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # --- Linha inferior ---
    col_params, col_config = st.columns(2)
    with col_params:
        render_best_params_table(generations[-1])
    with col_config:
        render_run_config()
```

### Cartões de métricas

```python
def render_metric_cards(generations: list[dict]):
    current = generations[-1]
    previous = generations[-2] if len(generations) > 1 else None

    def delta(key):
        if previous is None:
            return None
        return current[key] - previous[key]

    cols = st.columns(4)
    metrics = [
        ("Melhor fitness",     "best_fitness",  4),
        ("Melhor F2 score",    "best_f2",       4),
        ("Melhor acurácia",    "best_accuracy", 4),
        ("Fitness médio pop.", "avg_fitness",   4),
    ]

    for col, (label, key, decimals) in zip(cols, metrics):
        d = delta(key)
        col.metric(
            label=label,
            value=f"{current[key]:.{decimals}f}",
            delta=f"{d:+.4f}" if d is not None else None,
        )
```

### Gráfico de evolução do fitness

```python
def build_fitness_evolution_chart(generations: list[dict]) -> go.Figure:
    gens   = [g["generation"]   for g in generations]
    best   = [g["best_fitness"] for g in generations]
    avg    = [g["avg_fitness"]  for g in generations]
    std    = [g["std_fitness"]  for g in generations]

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
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
        hovermode="x unified",
    )

    return fig
```

### Scatter F2 score vs acurácia

```python
def build_f2_accuracy_scatter(generations: list[dict]) -> go.Figure:
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
            "F2 score: %{y:.4f}<extra></extra>"
        ),
        name="",
    ))

    # Linha diagonal de referência (F2 = acurácia)
    fig.add_shape(
        type="line", x0=0.5, y0=0.5, x1=1.0, y1=1.0,
        line=dict(color="rgba(100,100,100,0.3)", dash="dot", width=1),
    )

    fig.update_layout(
        title="F2 score vs acurácia",
        xaxis=dict(title="Acurácia", range=[0.5, 1.0]),
        yaxis=dict(title="F2 score", range=[0.5, 1.0]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )

    return fig
```

### Tabela de melhores parâmetros

```python
def render_best_params_table(latest: dict):
    st.markdown("**Melhores parâmetros encontrados**")

    model = latest.get("model_type", "—")
    params = latest.get("best_params", {})

    st.caption(f"Modelo: `{model}` — Geração {latest['generation']}")
    st.dataframe(
        data={"Parâmetro": list(params.keys()), "Valor": list(params.values())},
        use_container_width=True,
        hide_index=True,
    )
```

### Tabela de configuração da execução

```python
def render_run_config():
    """
    Preencher com os parâmetros reais do AG da sua aplicação.
    Os valores abaixo são placeholders.
    """
    st.markdown("**Configuração da execução**")

    # PLACEHOLDER: substituir pelos valores reais lidos da sua config/job
    config = {
        "Tamanho da população": "<!-- PLACEHOLDER: population_size -->",
        "Gerações totais":      "<!-- PLACEHOLDER: max_generations -->",
        "Taxa de mutação":      "<!-- PLACEHOLDER: mutation_rate -->",
        "Taxa de crossover":    "<!-- PLACEHOLDER: crossover_rate -->",
        "Peso F2 no fitness":   "<!-- PLACEHOLDER: f2_weight -->",
        "Peso acurácia":        "<!-- PLACEHOLDER: accuracy_weight -->",
    }

    st.dataframe(
        data={"Parâmetro": list(config.keys()), "Valor": list(config.values())},
        use_container_width=True,
        hide_index=True,
    )
```

---

## Ponto de entrada — integração com o código existente

```python
# app.py (ou o arquivo principal do Streamlit)

import time
import streamlit as st

def main():
    st.set_page_config(page_title="AG Tuner", layout="wide")
    init_session_state()

    # --- Painel de disparo (existente, manter como está) ---
    with st.sidebar:
        st.header("Configuração")
        # PLACEHOLDER: seus inputs de configuração existentes aqui

        if st.button("Iniciar otimização"):
            job_id = dispatch_job(...)   # PLACEHOLDER: sua função de disparo
            st.session_state.job_id = job_id
            st.session_state.job_running = True
            st.session_state.generations = []
            st.session_state.last_generation = 0

    # --- Dashboard ---
    if st.session_state.job_running:
        run_polling_loop(
            job_id=st.session_state.job_id,
            poll_interval=3,            # PLACEHOLDER: ajustar conforme necessário
        )
    elif st.session_state.generations:
        # Job concluído — exibe o dashboard final estático
        render_dashboard(st.session_state.generations)
        st.success("Otimização concluída.")
    else:
        st.info("Configure os parâmetros e inicie a otimização.")

if __name__ == "__main__":
    main()
```

---

## Checklist de implementação

- [ ] Escolher a opção de persistência (A, B ou C) e implementar `save_generation_snapshot`
- [ ] Adicionar a chamada a `save_generation_snapshot` no loop do AG, ao final de cada geração
- [ ] Implementar `fetch_new_generation_snapshots` para a opção de persistência escolhida
- [ ] Adicionar `init_session_state` ao início do `main()`
- [ ] Substituir o loop de polling atual por `run_polling_loop`
- [ ] Implementar `render_dashboard` e todas as subfunções de visualização
- [ ] Preencher os PLACEHOLDERs de configuração em `render_run_config` com os valores reais
- [ ] Ajustar `poll_interval` conforme o tempo médio de cada geração
- [ ] Testar com um job curto (5–10 gerações) antes de rodar o experimento completo

---

## Observações para o agente

- Não alterar a lógica interna do algoritmo genético nem a função de fitness
- Não alterar o mecanismo de disparo e monitoramento de conclusão do job existente
- Adaptar os nomes de variáveis (`best_fitness`, `best_f2`, `best_accuracy`, `best_params`) aos nomes reais usados no loop do AG
- A função `fetch_new_generation_snapshots` deve implementar apenas a opção de persistência já em uso na aplicação
- O campo `best_params` deve ser um dicionário serializável em JSON; se o AG usar objetos não serializáveis, converter antes de gravar
