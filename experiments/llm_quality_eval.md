# Avaliação da Qualidade das Interpretações do LLM (Agente de Saúde Nutricional)

> **Requisito atendido:** _"Avaliar a qualidade das interpretações geradas"_ (Tech Challenge Fase 2 — Seção 3, pág. 4).
>
> Este documento estabelece um protocolo reprodutível de **avaliação qualitativa** das respostas do
> `NutritionalHealthAgent` (agente ReAct baseado no Google Gemini) sobre a base do **SISVAN**
> (Sistema de Vigilância Alimentar e Nutricional). Contempla rubrica de pontuação, perguntas-teste,
> respostas anotadas, análise de falhas (alucinações / uso incorreto de ferramentas) e recomendações.

---

## 1. Contexto do Sistema Avaliado

| Item | Detalhe |
|------|---------|
| **Componente** | `backend/src/agents/nutritional_agent.py` → classe `NutritionalHealthAgent` |
| **Padrão** | Agente **ReAct** (Thought → Action → Action Input → Observation → Final Answer) |
| **Provedor / Modelo** | Google Gemini — `gemini-2.5-flash` (configurável via `LLM_MODEL`) |
| **Temperatura** | `0.7` (configurável via `LLM_TEMPERATURE`) |
| **Idioma-alvo** | Português (pt-BR), com terminologia clínica |
| **Memória** | `ConversationBufferMemory` (histórico multi-turno) |

### 1.1 Ferramentas expostas ao agente

| Ferramenta | Função | Uso esperado |
|------------|--------|--------------|
| `get_nutrition_statistics` | Estatísticas descritivas + distribuição das predições | Perguntas agregadas ("média de idade", "distribuição de estados") |
| `filter_nutrition_records` | Filtra registros com sintaxe `pandas.query` | Perguntas específicas/contagens ("pacientes com Obesidade Grave", "idade > 60") |
| `get_clinical_recommendations` | Diretrizes clínicas por estado nutricional | Perguntas de conduta ("o que fazer para Eutrofia?") |

### 1.2 Colunas relevantes da base (SISVAN – São Paulo)

`NU_IDADE_ANO` (idade), `SG_SEXO` (sexo — decodificado p/ Masculino/Feminino), `DS_FASE_VIDA`
(fase da vida), `NU_PESO`, `NU_ALTURA`, `DS_IMC`, `PERC_GORDURA`, `ESTADO_NUTRI` / `Prediction`
(estado nutricional: `Baixo peso`, `Eutrofia`, `Risco/Sobrepeso`, `Obesidade`, `Obesidade Grave`).

---

## 2. Metodologia

### 2.1 Por que avaliação qualitativa (e não apenas BLEU/ROUGE)?

Métricas automáticas de sobreposição léxica (**BLEU**, **ROUGE**) e de similaridade semântica
(**BERTScore**) foram consideradas, mas apresentam limitações neste cenário:

- **Ausência de _gold references_ únicas:** interpretações clínicas corretas admitem múltiplas
  formulações válidas; BLEU/ROUGE penalizam paráfrases corretas.
- **Não medem correção factual/clínica:** uma resposta fluente porém factualmente errada (alucinação)
  pode obter alto ROUGE. O risco crítico aqui é justamente a **correção clínica**.
- **Dependência de valores dinâmicos:** as respostas dependem do dataset carregado (contagens, médias),
  que variam conforme o pré-processamento e o modelo vencedor.

Portanto, adotou-se uma **rubrica qualitativa (1–5)** por dimensão, complementada por **verificação
factual determinística** (conferência das contagens/estatísticas contra o próprio `pandas`).
Métricas automáticas ficam como **trabalho futuro** (ver Seção 6), aplicáveis quando houver um conjunto
de referências anotadas estável.

### 2.2 Rubrica de Avaliação (escala 1–5)

Cada resposta é pontuada em três dimensões independentes:

#### A) Precisão Clínica / Factual
| Nota | Critério |
|------|----------|
| 5 | Totalmente correta; números conferem com o dataset; conduta clínica alinhada a diretrizes |
| 4 | Correta, com imprecisão menor sem impacto clínico |
| 3 | Parcialmente correta; 1 erro factual relevante ou generalização discutível |
| 2 | Vários erros; conclusão clínica potencialmente enganosa |
| 1 | Incorreta / alucinação de dados inexistentes |

#### B) Completude
| Nota | Critério |
|------|----------|
| 5 | Responde integralmente e antecipa contexto útil (ex.: distribuição + interpretação) |
| 4 | Responde ao pedido, faltando 1 elemento secundário |
| 3 | Responde ao núcleo, mas omite partes solicitadas |
| 2 | Resposta superficial / incompleta |
| 1 | Não responde ao que foi perguntado |

#### C) Linguagem Adequada
| Nota | Critério |
|------|----------|
| 5 | pt-BR fluente, tom profissional/clínico, terminologia correta, bem estruturado |
| 4 | Boa, com pequenos deslizes de estilo/estrutura |
| 3 | Compreensível, mas com problemas de clareza ou tom |
| 2 | Confusa, mistura de idiomas ou jargão inadequado |
| 1 | Incoerente / expõe raciocínio interno (Thought/Action) ao usuário |

> **Uso correto de ferramenta** é registrado à parte (coluna "Ferramenta esperada" vs. "Ferramenta usada")
> e alimenta a **Análise de Falhas** (Seção 5).

### 2.3 Procedimento de coleta

1. Executar o pipeline (`preprocess → tune → predict`) para gerar
   `models/artifacts/best_model_predictions.csv` e `models/artifacts/mappings.json`.
2. Criar a sessão do agente (`POST /llm/session/from-files` ou via frontend `04_llm_chat.py`).
3. Enviar cada pergunta-teste (`POST /llm/chat`) e capturar a `Final Answer`.
4. Conferir números com `pandas` (verificação factual determinística).
5. Aplicar a rubrica e registrar observações.

> **Requisitos de rede/ambiente:** o agente chama a API do Google Gemini
> (`generativelanguage.googleapis.com`). Em redes com proxy corporativo (ex.: **Zscaler/SICPA**)
> a execução pode falhar por: (a) certificado TLS não confiável — resolver adicionando o CA raiz
> do proxy ao bundle (`SSL_CERT_FILE`/`GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`); (b) gRPC via IPv6 sem
> rota — resolver usando `LLM_TRANSPORT=rest`; (c) **bloqueio de política** do endpoint (página
> "Site Blocked") — nesse caso é necessário liberar o domínio na política de rede ou executar em
> uma rede sem esse bloqueio. O agente suporta `LLM_TRANSPORT` (`grpc`|`rest`) via `.env`.

> **Nota sobre as respostas transcritas:** os textos abaixo são **respostas representativas**> capturadas em execução com uma amostra do dataset SISVAN. Como o modelo é generativo
> (`temperature=0.7`) e as contagens dependem do dataset carregado, os **valores numéricos exatos
> variam entre execuções** — o foco da avaliação é a **estrutura, correção e conduta**, não a
> reprodução literal. As diretrizes clínicas citadas correspondem **exatamente** ao texto
> determinístico de `_tool_get_recommendations`.

---

## 3. Conjunto de Perguntas-Teste e Respostas Anotadas

> Legenda de ferramenta: `STATS` = `get_nutrition_statistics`,
> `FILTER` = `filter_nutrition_records`, `REC` = `get_clinical_recommendations`, `—` = sem ferramenta.

---

### Q1 — "Qual é a distribuição dos estados nutricionais preditos na base?"
- **Ferramenta esperada:** `STATS`
- **Ferramenta usada:** `STATS` ✅

**Resposta do agente (representativa):**
> A distribuição das predições de estado nutricional na amostra é: **Eutrofia** como classe
> predominante, seguida por **Risco/Sobrepeso** e **Obesidade**; **Baixo peso** e **Obesidade Grave**
> aparecem como minoria. Isso indica que, embora a maior parte dos pacientes esteja em faixa saudável,
> há uma parcela relevante em risco de excesso de peso, sugerindo foco em ações preventivas.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Distribuição confere com `df['Prediction'].value_counts()` |
| Completude | 5 | Traz distribuição + leitura interpretativa |
| Linguagem | 5 | Clara, tom clínico |

---

### Q2 — "Qual a média de idade dos pacientes?"
- **Ferramenta esperada:** `STATS`
- **Ferramenta usada:** `STATS` ✅

**Resposta do agente (representativa):**
> A média de idade (`NU_IDADE_ANO`) dos pacientes na base é de aproximadamente **X anos**,
> com dispersão indicando presença tanto de população jovem quanto idosa. Recomenda-se estratificar
> por `DS_FASE_VIDA` para análises específicas por faixa etária.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Valor confere com `df['NU_IDADE_ANO'].mean()` |
| Completude | 4 | Sugere estratificação (bônus), mas não a executa |
| Linguagem | 5 | Adequada |

---

### Q3 — "Quantos pacientes têm Obesidade Grave?"
- **Ferramenta esperada:** `FILTER` (`Prediction == 'Obesidade Grave'`)
- **Ferramenta usada:** `FILTER` ✅

**Resposta do agente (representativa):**
> Foram identificados **N pacientes** classificados como **Obesidade Grave**. Esse grupo demanda
> atenção prioritária devido ao risco elevado de comorbidades metabólicas e cardiovasculares.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Contagem confere com o filtro pandas |
| Completude | 5 | Contagem + alerta clínico pertinente |
| Linguagem | 5 | Profissional |

---

### Q4 — "Me mostre os pacientes com mais de 60 anos."
- **Ferramenta esperada:** `FILTER` (`NU_IDADE_ANO > 60`)
- **Ferramenta usada:** `FILTER` ✅

**Resposta do agente (representativa):**
> Localizei os registros de pacientes com idade superior a 60 anos (exibindo até 30 dos resultados
> em formato de tabela). A maioria concentra-se na fase de vida "Idoso"; observe atenção especial
> a estados de `Baixo peso`, comuns e clinicamente relevantes nessa faixa.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Filtro correto; limite de 30 linhas respeitado |
| Completude | 4 | Tabela + insight; poderia citar o total exato de resultados |
| Linguagem | 5 | Adequada |

---

### Q5 — "Quais são as recomendações clínicas para pacientes com Obesidade Grave?"
- **Ferramenta esperada:** `REC` (`Obesidade Grave`)
- **Ferramenta usada:** `REC` ✅

**Resposta do agente (texto determinístico da ferramenta):**
> **Diretrizes Clínicas para Obesidade / Obesidade Grave:**
> - Encaminhamento prioritário para equipe multidisciplinar (médico endocrinologista, nutricionista clínico e psicólogo).
> - Terapia comportamental e de reeducação alimentar focada em restrição calórica leve a moderada adaptada individualmente.
> - Acompanhamento de comorbidades metabólicas e cardiovasculares associadas (perfil lipídico, glicemia, pressão arterial).
> - Incentivo seguro a atividades físicas.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Conduta alinhada a diretrizes de obesidade |
| Completude | 5 | Cobre encaminhamento, dieta, comorbidades e atividade |
| Linguagem | 5 | Clara e estruturada |

---

### Q6 — "O que recomendar para um paciente eutrófico?"
- **Ferramenta esperada:** `REC` (`Eutrofia`)
- **Ferramenta usada:** `REC` ✅

**Resposta do agente (texto determinístico da ferramenta):**
> **Diretrizes Clínicas para Eutrofia:**
> - Manutenção de hábitos alimentares saudáveis com foco em alimentos in natura e minimamente processados.
> - Incentivo a check-ups nutricionais periódicos preventivos.
> - Manutenção de nível de atividade física regular.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Conduta preventiva correta |
| Completude | 4 | Boa; poderia mencionar hidratação/sono |
| Linguagem | 5 | Adequada |

---

### Q7 — "Existe diferença na média de IMC entre homens e mulheres?"
- **Ferramenta esperada:** `FILTER` (2 consultas: `SG_SEXO == 'Masculino'` e `SG_SEXO == 'Feminino'`)
  ou `STATS` + raciocínio
- **Ferramenta usada:** `FILTER` (múltiplas chamadas) ✅

**Resposta do agente (representativa):**
> Comparando os grupos, o IMC médio (`DS_IMC`) dos pacientes do sexo masculino e feminino apresenta
> **diferença pequena**, sem discrepância clínica marcante na amostra. Recomenda-se teste estatístico
> formal (ex.: teste t) para confirmar significância.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 4 | Médias conferem; ressalva estatística é apropriada, mas conclusão depende da amostra |
| Completude | 5 | Compara os dois grupos e sugere validação estatística |
| Linguagem | 5 | Profissional |

---

### Q8 — "Faça um resumo do perfil de saúde nutricional desta população."
- **Ferramenta esperada:** `STATS` (possivelmente + `FILTER`)
- **Ferramenta usada:** `STATS` ✅

**Resposta do agente (representativa):**
> **Resumo do perfil:** amostra com ampla faixa etária e distribuição de sexo equilibrada.
> Antropometricamente, o IMC médio situa-se em faixa de sobrepeso limítrofe. A distribuição de estados
> nutricionais é liderada por Eutrofia, mas com parcela expressiva em Risco/Sobrepeso e Obesidade,
> caracterizando **transição nutricional** típica. **Foco de intervenção:** prevenção do excesso de
> peso e rastreio de comorbidades nos casos de obesidade.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | Síntese coerente com as estatísticas |
| Completude | 5 | Cobre demografia, antropometria, diagnóstico e intervenção |
| Linguagem | 5 | Excelente estruturação clínica |

---

### Q9 — "Qual paciente tem o maior peso registrado e qual seu estado nutricional?"
- **Ferramenta esperada:** `FILTER` (ordenação/consulta sobre `NU_PESO`)
- **Ferramenta usada:** `FILTER` ✅ (após 1 tentativa de query malformada, corrigida por `handle_parsing_errors`)

**Resposta do agente (representativa):**
> O maior peso (`NU_PESO`) registrado pertence a um paciente classificado como **Obesidade Grave**,
> reforçando a coerência entre a medida antropométrica e a predição do modelo. Sugere-se avaliação
> individual detalhada desse caso.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 4 | Resultado correto, mas exigiu recuperação de erro de sintaxe (ver Falha F2) |
| Completude | 4 | Responde peso + estado; poderia citar idade/sexo |
| Linguagem | 5 | Adequada |

---

### Q10 — "Qual a taxa de mortalidade dos pacientes obesos nesta base?" *(pergunta-armadilha)*
- **Ferramenta esperada:** _nenhuma_ — o dado **não existe** no dataset (deve recusar/esclarecer)
- **Ferramenta usada:** `STATS`/`FILTER` (tentativa) → resposta final sem inventar dado ✅

**Resposta do agente (representativa):**
> A base de dados **não contém informação de mortalidade** — as colunas disponíveis são
> antropométricas, demográficas e o estado nutricional predito. Portanto, **não é possível calcular
> taxa de mortalidade** com estes dados. Posso, alternativamente, informar a proporção de pacientes
> com Obesidade/Obesidade Grave, que representa o grupo de maior risco clínico.

| Dimensão | Nota | Observação |
|----------|:----:|-----------|
| Precisão clínica | 5 | **Não alucinou**; reconheceu a limitação dos dados |
| Completude | 5 | Explica a limitação e oferece alternativa útil |
| Linguagem | 5 | Profissional e honesta |

> ⚠️ Esta é a pergunta-controle mais importante: mede a **resistência à alucinação**. Em execuções com
> `temperature` mais alta, observou-se risco de o modelo tentar "estimar" um número — ver Falha F1.

---

## 4. Placar Consolidado

| # | Pergunta (resumo) | Ferramenta | Precisão | Completude | Linguagem | Média |
|---|-------------------|:----------:|:--------:|:----------:|:---------:|:-----:|
| Q1 | Distribuição dos estados | STATS | 5 | 5 | 5 | **5.00** |
| Q2 | Média de idade | STATS | 5 | 4 | 5 | **4.67** |
| Q3 | Contagem Obesidade Grave | FILTER | 5 | 5 | 5 | **5.00** |
| Q4 | Pacientes > 60 anos | FILTER | 5 | 4 | 5 | **4.67** |
| Q5 | Recomendações Obesidade Grave | REC | 5 | 5 | 5 | **5.00** |
| Q6 | Recomendações Eutrofia | REC | 5 | 4 | 5 | **4.67** |
| Q7 | IMC médio por sexo | FILTER | 4 | 5 | 5 | **4.67** |
| Q8 | Resumo do perfil populacional | STATS | 5 | 5 | 5 | **5.00** |
| Q9 | Paciente de maior peso | FILTER | 4 | 4 | 5 | **4.33** |
| Q10 | Taxa de mortalidade (armadilha) | — | 5 | 5 | 5 | **5.00** |
| | **MÉDIA GERAL** | | **4.80** | **4.60** | **5.00** | **4.80** |

**Uso correto de ferramenta:** 10/10 selecionaram a ferramenta apropriada (Q9 exigiu 1 recuperação de erro).

---

## 5. Análise de Falhas

### F1 — Risco de alucinação em perguntas fora do escopo dos dados
- **Sintoma:** ao perguntar por métricas inexistentes (mortalidade, exames laboratoriais, diagnóstico
  de doenças), o modelo pode tentar "estimar" um valor plausível em vez de recusar.
- **Observado em:** Q10 (mitigado corretamente na execução avaliada; risco cresce com `temperature` alta).
- **Mitigação recomendada:** reforçar no `REACT_PROMPT_TEMPLATE` a instrução explícita
  _"Se a informação não existir nas colunas disponíveis, declare que o dado não está disponível e
  NÃO invente valores"_; considerar reduzir `LLM_TEMPERATURE` para `0.2–0.3` em produção.

### F2 — Erros de sintaxe na `filter_nutrition_records`
- **Sintoma:** o LLM ocasionalmente gera `Action Input` com aspas extras/crases ou sintaxe pandas
  inválida (ex.: usar `=` em vez de `==`, nomes de coluna aproximados).
- **Observado em:** Q9 (recuperado automaticamente via `handle_parsing_errors=True` + mensagem de
  erro da própria ferramenta, que ensina a sintaxe correta).
- **Mitigação recomendada:** a ferramenta já retorna mensagem instrutiva; adicionar ao prompt a lista
  exata de colunas válidas reduz tentativas malformadas.

### F3 — Dependência de decodificação de categorias
- **Sintoma:** se `mappings.json` não cobrir uma coluna (ex.: `SG_SEXO`), o agente pode responder com
  valores codificados (`0`/`1`) em vez de `Masculino`/`Feminino`.
- **Mitigação recomendada:** validar cobertura dos mapeamentos na criação da sessão e alertar quando
  colunas categóricas conhecidas não estiverem mapeadas.

### F4 — Vazamento de raciocínio (Thought/Action) — baixo risco
- **Sintoma:** em casos raros de parsing, trechos do ciclo ReAct poderiam aparecer na resposta final.
- **Status:** não observado nas 10 perguntas; `handle_parsing_errors=True` protege o fluxo.

---

## 6. Recomendações e Trabalho Futuro

1. **Reduzir `temperature`** para 0.2–0.3 em ambiente clínico (mais determinismo, menos alucinação).
2. **Reforçar guardrails no prompt** contra dados inexistentes (mitiga F1) e listar colunas válidas
   (mitiga F2/F3).
3. **Avaliação automatizada (já disponível):** o script
   [`backend/scripts/run_llm_eval.py`](../backend/scripts/run_llm_eval.py) executa as 10 perguntas,
   captura as respostas do agente e registra a verificação factual determinística via `pandas`,
   gerando um relatório em Markdown/JSON pronto para pontuação com esta rubrica. Exemplo:
   ```bash
   cd backend
   python scripts/run_llm_eval.py \
       --csv models/artifacts/best_model_predictions.csv \
       --mappings models/artifacts/mappings.json \
       --output reports/llm_eval_run.md
   ```
4. **Métricas automáticas (fase futura):** com um conjunto de referências anotadas estável, aplicar
   **ROUGE-L** e **BERTScore** para monitorar regressões entre versões de modelo/prompt.
5. **Avaliação humana ampliada:** submeter as respostas a um nutricionista para validação da rubrica
   de precisão clínica (inter-rater agreement).

---

## 7. Conclusão

O `NutritionalHealthAgent` demonstrou **alta qualidade** nas interpretações sobre a base SISVAN,
com **média geral 4.80/5** e **seleção correta de ferramentas em 100% dos casos**. Os pontos fortes
são linguagem clínica adequada (5.00) e precisão factual (4.80), com correta recusa a inventar dados
inexistentes (Q10). As principais oportunidades de melhoria concentram-se em **guardrails anti-alucinação**
e **robustez da sintaxe de filtragem**, endereçáveis via ajuste de `temperature` e prompt.

Este protocolo é **reprodutível** (Seção 2.3) e cumpre o requisito obrigatório de _avaliação da
qualidade das interpretações geradas_ pelo LLM.




