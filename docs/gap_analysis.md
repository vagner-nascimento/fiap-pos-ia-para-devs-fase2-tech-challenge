# 📋 Relatório de Gaps — Tech Challenge Fase 2 (Projeto 1)

> Avaliação do projeto [`fiap-pos-ia-para-devs-fase2-tech-challenge`](file:///home/luizbaroni/projetos/fiap/fiap-pos-ia-para-devs-fase2-tech-challenge) em relação ao edital [IADT - Fase 2 - Tech challenge-1.pdf](file:///home/luizbaroni/projetos/fiap/fiap-pos-ia-para-devs-fase2-tech-challenge/docs/IADT%20-%20Fase%202%20-%20Tech%20challenge-1.pdf).

---

## ✅ Pontuação por Critério

| Seção do Edital | Status | Observação |
|----------------|:------:|-----------|
| 1. Algoritmo Genético — implementação | ✅ Completo | GA Co-Evolutivo RF+KNN totalmente implementado |
| 1. Codificação de genes (representação) | ✅ Completo | ADR-002: `IndividuoRF`/`IndividuoKNN` com `dict` tipado |
| 1. Operadores: seleção, cruzamento, mutação | ✅ Completo | ADR-003/008/009: torneio global, cxUniform dict, 3 níveis de mutação |
| 1. Função fitness | ✅ Completo | ADR-005: `F1_weighted×0.6 + Acc×0.4` via k-Fold CV |
| 1. ≥ 3 experimentos com configurações diferentes do GA | ⚠️ Parcial | Infra documentada (parâmetros CLI), mas **sem relatório documentado de 3 runs com resultados comparativos** |
| 1. Comparação GA-otimizado vs. modelos originais | ⚠️ Parcial | Código existe (`model_comparison_service.py`, `run_predictions_all.py`), mas **artefatos e relatório de comparação não estão no repositório** (pasta `reports/` e `models/artifacts/` vazias) |
| 2. Monitoramento e logging | ✅ Completo | ADR-014/015: logs por geração, polling REST, console de pré-processamento |
| 2. Documentação de arquitetura | ✅ Completo | `docs/architecture.md`, 15 ADRs, diagramas Mermaid |
| 2. Escalabilidade (monitoramento de demanda) | ✅ Aceitável | Docker Compose, jobs assíncronos, amostragem estratificada |
| 2. Implementação em nuvem (opcional) | ❌ Não entregue | Nenhum IaC, Dockerfile apenas local. Marcado como opcional, sem penalidade obrigatória |
| 3. Integração com LLM | ✅ Completo | `NutritionalHealthAgent` com Google Gemini via LangChain ReAct |
| 3. Explicações em linguagem natural | ✅ Completo | 3 ferramentas: estatísticas, filtro pandas, recomendações clínicas |
| 3. Prompt engineering documentado | ✅ Completo | `REACT_PROMPT_TEMPLATE` definido em `nutritional_agent.py`; abordagem discutida em `llm_quality_eval.md` |
| 3. Avaliação da qualidade das interpretações | ✅ Completo | `experiments/llm_quality_eval.md`: rubrica 1–5, 10 Q&A anotadas, análise de falhas, média 4.80/5 |
| 4. Projeto Python bem estruturado + venv | ✅ Completo | `uv` / `pyproject.toml`, `Dockerfile`, `docker-compose.yml` |
| 4. Documentação detalhada + diagramas | ✅ Completo | `README.md`, `docs/architecture.md`, `docs/adr.md` |
| 4. Testes automatizados | ✅ Completo | 302 testes (unit + integração), `pytest --cov` |
| **Repositório Git** | ✅ Completo | Código-fonte completo no repositório |
| **Documentação da API** | ✅ Completo | FastAPI `/docs` (Swagger), endpoints descritos no README |
| **Scripts/notebooks de demonstração** | ⚠️ Parcial | Scripts CLI presentes (`run_*.py`), `llm_eval.py`; **sem notebook de demonstração end-to-end com saídas executadas** |
| **Relatório técnico** | ❌ Gap crítico | **Não existe documento consolidado** (PDF/MD) com os 4 tópicos exigidos pelo edital |
| **Vídeo de demonstração** | ❌ Gap crítico | **Nenhum vídeo** (YouTube/Vimeo) mencionado ou linkado em qualquer arquivo |

---

## 🔴 Gaps Críticos (Obrigatórios)

### GAP 1 — Vídeo de Demonstração (até 15 min)
**Exigência do edital (pág. 6–7):**
- Upload no YouTube ou Vimeo (público ou não listado)
- Demonstrar o sistema em execução
- Explicar os diferentes componentes
- Apresentar resultados da otimização via algoritmos genéticos
- Demonstrar integração com LLMs

**Status atual:** Nenhum link, arquivo ou referência a vídeo foi encontrado em qualquer documento do projeto.

> [!CAUTION]
> Este é o entregável mais fácil de ser rejeitado automaticamente. Sem o link do vídeo, a avaliação pode ser zerada. **Prioridade máxima.**

---

### GAP 2 — Relatório Técnico Consolidado
**Exigência do edital (pág. 6):** documento explicando:
1. Implementação do algoritmo genético e resultados da otimização de hiperparâmetros
2. Integração com LLMs: abordagem, prompts utilizados e avaliação da qualidade
3. **Comparativo de desempenho entre os modelos originais e otimizados**
4. Desafios enfrentados e soluções implementadas

**Status atual:** A informação existe **dispersa** em vários arquivos:
- ADRs cobrem decisões de design ✅
- `llm_quality_eval.md` cobre avaliação do LLM ✅
- `architecture.md` cobre arquitetura ✅
- **Mas não existe um documento único e consolidado** como "Relatório Técnico" que organize todos esses pontos
- **Faltam completamente os resultados numéricos** da otimização (tabelas com fitness, F1, Accuracy antes/depois)

> [!IMPORTANT]
> O relatório técnico deve ser um documento **standalone** que um avaliador pode ler sem precisar navegar pelo código-fonte.

---

### GAP 3 — Comparativo de Desempenho: Modelos Originais vs. Otimizados
**Exigência do edital (pág. 3):** _"Comparar o desempenho dos modelos otimizados com os modelos originais"_

**Status atual:**
- O código de comparação **existe** (`model_comparison_service.py`, `run_predictions_all.py`)
- Há uma API para isso (`POST /model-comparison/compare`)
- O frontend provavelmente tem uma página para isso
- **Mas os artefatos de modelo não estão no repositório** (`models/artifacts/` e `models/originals/` estão vazios — apenas `.gitkeep`)
- Não há relatório gerado, tabela ou gráfico publicado no repositório mostrando métricas comparativas
- Não foi encontrado nenhum documento que mostre "RF Original: F1=X → RF GA-Otimizado: F1=Y (+Z%)"

> [!WARNING]
> O repositório está rastreando arquivos `.joblib` via Git LFS (commit menciona isso), mas as pastas estão vazias no workspace local. Verifique se os arquivos foram de fato enviados com `git lfs push`.

---

## 🟡 Gaps Moderados

### GAP 4 — 3 Experimentos Documentados com Configurações Diferentes do GA
**Exigência do edital (pág. 3):** _"Realizar ao menos 3 experimentos com diferentes configurações do algoritmo genético (tamanho da população, taxas de mutação, etc.)"_

**Status atual:**
- O CLI permite configurar todos os parâmetros (`--pop-size`, `--aggressiveness`, `--mutation-rate`, etc.)
- O README mostra um exemplo de execução rápida e um de produção
- **Mas não há 3 runs documentados com resultados concretos** (tabela com configurações e métricas de cada run)
- O `llm_quality_eval.md` documenta experimentos do LLM, mas **não do GA**

**O que falta:**
```
| Experimento | pop_size | max_gen | mutation | k_folds | Best F1 | Best Acc | Modelo vencedor |
|-------------|---------|---------|---------|---------|---------|---------|-----------------|
| Exp. 1 (rápido) | 4 | 2 | medium | 3 | 0.XX | 0.XX | RF/KNN |
| Exp. 2 (padrão) | 20 | 10 | medium | 5 | 0.XX | 0.XX | RF/KNN |
| Exp. 3 (agressivo) | 20 | 10 | high | 5 | 0.XX | 0.XX | RF/KNN |
```

---

### GAP 5 — Notebook de Demonstração End-to-End
**Exigência do edital (pág. 6):** _"Scripts ou notebooks de demonstração"_

**Status atual:**
- Há scripts CLI (`run_preprocessing.py`, `run_tuning.py`, `run_predictions.py`)
- Há um `validations.ipynb` básico (só tem 2 células de markdown: "validações rápidas" e "validando preprocessamento")
- **Não há um Jupyter Notebook** demonstrando o pipeline completo com saídas executadas e visualizações
- O `llm_quality_eval.md` serve como "demonstração" do agente, mas é estático (sem execução reproduzível)

---

## 🟢 O que está bem (Pontos Fortes)

| Área | Detalhe |
|------|---------|
| **Algoritmo Genético** | Co-evolutivo RF+KNN com 15 ADRs documentando cada decisão de design |
| **Testes** | 302 testes cobrindo unitários, integração e API — executam em <20s |
| **Agente LLM** | ReAct com Gemini, 3 ferramentas customizadas, memória multi-turno |
| **Prompt Engineering** | `REACT_PROMPT_TEMPLATE` documentado + análise de falhas (F1–F4) |
| **Avaliação LLM** | Rubrica 1–5, 10 perguntas-teste com Q&A anotadas, média 4.80/5 |
| **API REST** | FastAPI com Swagger, jobs assíncronos, polling em tempo real |
| **Frontend** | Streamlit com dashboard GA em tempo real e chat LLM |
| **DevOps** | Docker Compose, scripts de restart, `uv` para deps |
| **Documentação técnica** | `architecture.md` + `adr.md` de alta qualidade, diagramas Mermaid |
| **Monitoramento** | Logs por geração, polling REST, console de pré-processamento |

---

## 📋 Plano de Ação Priorizado

| Prioridade | Ação | Esforço estimado |
|:----------:|------|:---------------:|
| 🔴 P1 | **Gravar e publicar o vídeo de demonstração** (YouTube/Vimeo, até 15 min) | 2–4h |
| 🔴 P2 | **Criar o Relatório Técnico consolidado** (MD ou PDF) com os 4 tópicos do edital | 3–5h |
| 🔴 P3 | **Executar e publicar o comparativo** modelos originais vs. GA-otimizado (garantir que `.joblib` estão no LFS e `reports/` tem saída) | 1–2h |
| 🟡 P4 | **Documentar 3 experimentos GA** com configurações e métricas em tabela | 2–3h |
| 🟡 P5 | **Adicionar link do vídeo** ao `README.md` e ao relatório técnico | 30 min |
| 🟢 P6 | Criar notebook de demonstração end-to-end (opcional, melhora apresentação) | 2–3h |

---

## 📊 Sumário Executivo

```
Critérios obrigatórios satisfeitos:     11 / 14   (79%)
Critérios com gaps parciais:             2 / 14   (14%)
Critérios com gaps críticos:             1 / 14   (7%)

Entregáveis finais satisfeitos:          3 / 5    (60%)
  ✅ Repositório Git
  ✅ Documentação da API
  ⚠️  Scripts/notebooks de demonstração (parcial)
  ❌ Relatório técnico consolidado
  ❌ Vídeo de demonstração
```

> [!NOTE]
> O projeto tem **base técnica muito sólida** — algoritmo genético bem implementado, testes extensos, boa documentação de arquitetura e avaliação do LLM de qualidade. Os gaps são predominantemente de **apresentação e empacotamento** dos resultados, não de implementação. Com 1–2 dias de esforço, é possível fechar todos os gaps críticos.
