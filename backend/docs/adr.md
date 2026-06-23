# Architecture Decision Records (ADR)

Registro das principais decisões de arquitetura e de algoritmo tomadas no projeto SISVAN Fase 2.
Cada ADR documenta o contexto, as alternativas consideradas e a justificativa da escolha feita.

---

## ADR-001 — Biblioteca de Algoritmo Genético: DEAP

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O projeto precisa de uma biblioteca de GA que suporte representação de indivíduos não-homogênea (RF e KNN possuem hiperparâmetros de tipos e ranges distintos) e que permita operadores customizados.

### Alternativas Consideradas
| Biblioteca | Prós | Contras |
|---|---|---|
| **DEAP** | Flexibilidade total; suporte a representações heterogêneas; amplamente usado em pesquisa | API mais complexa |
| `scikit-optimize` (BayesOpt) | Simples de usar | Sem suporte a população híbrida; sem crossover customizável |
| `optuna` | Moderno, ótima interface | Baseado em Bayesiana, não em GA; sem controle do loop evolutivo |
| Implementação própria | Controle total | Alto custo de desenvolvimento e manutenção |

### Decisão
**DEAP** — pelo suporte nativo a indivíduos customizados, operadores intercambiáveis (`cxUniform`, `selTournament`) e pela maturidade da biblioteca.

### Consequências
- Dependência adicionada: `deap>=1.4.0,<2.0.0`
- Operadores DEAP usados como referência, mas adaptados para trabalhar com `dict` em vez de listas posicionais

---

## ADR-002 — Representação do Indivíduo: Classe Python vs. Lista DEAP

**Data**: 2026-06  
**Status**: Aceito

### Contexto
DEAP tradicionalmente representa indivíduos como listas (`list`) com atributo `fitness`. Nosso espaço de hiperparâmetros é heterogêneo (inteiros, floats, categorias) e nomeado — parâmetros não são intercambiáveis por posição.

### Alternativas Consideradas
1. **Lista DEAP pura** com índices fixos para cada hiperparâmetro
2. **Classe Python customizada** com `dict` de hiperparâmetros

### Decisão
**Classe Python (`Individuo` ABC + subclasses)** com `hyperparams: dict`. Motivos:
- Hiperparâmetros são nomeados semanticamente (ex: `n_estimators`, não `params[0]`)
- Subclasses permitem encapsular a lógica de `build_model()` por tipo
- `IndividuoKNN` inclui `StandardScaler` automaticamente sem necessidade de lógica externa
- Melhor legibilidade e testabilidade (cada classe tem responsabilidade única)

### Consequências
- Operadores DEAP (`cxUniform`, `mutGaussian`) não são aplicáveis diretamente — adaptamos a semântica em `ga_operators.py`
- O atributo `fitness` do DEAP não é usado; usamos `fitness_values: tuple` direto no objeto

---

## ADR-003 — Crossover: cxUniform Adaptado para Dicts Nomeados

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O `cxUniform` do DEAP troca genes por índice (`ind1[i] ↔ ind2[i]`). Com hiperparâmetros em `dict`, não há índice posicional — os genes são identificados por chave.

### Alternativas Consideradas
1. **cxUniform do DEAP** — requer conversão dict→list, aplicação, list→dict. Propenso a erros de mapeamento.
2. **cxTwoPoint** — swap de bloco contíguo. Sem semântica para dicts.
3. **cxUniform adaptado** — para cada `key` do dict, sorteia com `prob=indpb` se troca ou não.

### Decisão
**cxUniform adaptado** implementado em `_uniform_crossover_dicts()` no `ga_operators.py`:
```python
for key in dict1:
    if random.random() < indpb:
        dict1[key], dict2[key] = dict2[key], dict1[key]
```

### Consequências
- Comportamento equivalente ao `cxUniform` do DEAP, mas semanticamente correto para dicts nomeados
- `indpb=0.0` → filhos idênticos aos pais (verificado em testes)
- `indpb=1.0` → troca total de parâmetros (verificado em testes)
- `indpb=0.5` → default — cada gene tem 50% de chance de ser trocado

---

## ADR-004 — Populações Separadas com Competição Global (Co-Evolução)

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Precisamos otimizar hiperparâmetros de dois tipos de modelo (RF e KNN). As opções arquiteturais principais são:

1. **GA híbrido** — uma única população mista com crossover entre tipos
2. **GAs sequenciais** — dois loops independentes, sem interação
3. **GA co-evolutivo** — populações separadas, mas avaliação e seleção compartilhadas

### Decisão
**GA Co-Evolutivo** (opção 3):
- RF e KNN **não cruzam entre si** (hiperparâmetros não são intercambiáveis entre modelos)
- A **seleção por torneio é global** (pool = RF + KNN) — pressão competitiva real entre tipos
- Cada tipo gera filhos apenas dentro do próprio tipo

### Justificativa
- **Vs. GA híbrido**: Crossover entre RF e KNN não faz sentido semântico (não existe `n_estimators` em KNN)
- **Vs. GAs sequenciais**: A seleção global cria pressão competitiva dinâmica — o tipo mais fraco precisa evoluir ou é eliminado, sem custo computacional adicional
- Produz um resultado único e comparável ao final (melhor de RF vs. melhor de KNN)

### Consequências
- A proporção RF/KNN na população pode variar a cada geração (dinâmica emergente)
- **Elitismo estrutural mínimo**: garantia de ao menos 1 sobrevivente por tipo (evita extinção prematura)
- Saída unificada: `results["overall_best"]` é o vencedor global

---

## ADR-005 — Fitness: F1-Weighted × 0.6 + Accuracy × 0.4

**Data**: 2026-06  
**Status**: Aceito

### Contexto
O dataset SISVAN possui múltiplas classes de estado nutricional (Obesidade, Sobrepeso, Eutrofia, Baixo Peso, etc.) com possível desbalanceamento.

### Alternativas Consideradas
| Métrica | Problema |
|---|---|
| Apenas Acurácia | Favorece classes majoritárias; ignora desbalanceamento |
| Apenas F1-macro | Igual peso a todas as classes, inclusive as raríssimas |
| **F1-weighted** | Pondera pelo suporte de cada classe — adequado para desbalanceamento |
| F1-weighted × 0.6 + Acc × 0.4 | Mantém F1 como primário mas penaliza queda geral de acurácia |

### Decisão
**`fitness = F1_weighted × 0.6 + Accuracy × 0.4`**  
Implementado em `ga_evaluator.fitness_score()`.

### Consequências
- Modelos que ignoram classes minoritárias terão F1 penalizado mesmo com acurácia alta
- A dupla métrica evita overfitting em relação a uma única dimensão
- `fitness_score()` é exposto como função pública para uso no dashboard

---

## ADR-006 — Critérios de Parada Duplos: max_generations + patience

**Data**: 2026-06  
**Status**: Aceito

### Contexto
GAs podem ficar estagnados (plateau de fitness) muito antes de atingir o número máximo de gerações, desperdiçando tempo computacional.

### Decisão
Dois critérios independentes — para quando o **primeiro** for satisfeito:

1. **`max_generations`**: limite absoluto (equivalente ao critério padrão de GAs)
2. **`patience`**: N gerações consecutivas sem melhoria de `_CONVERGENCE_EPS = 1e-6` no best fitness global

### Consequências
- O resultado inclui `reason: "max_generations" | "convergence"` para rastreabilidade
- Para dados grandes (SISVAN, ~50MB+), a parada por convergência pode economizar horas de processamento
- `patience=5` (default) é conservador — ajustar para `patience=3` em runs rápidos de validação

---

## ADR-007 — Pipeline sklearn para IndividuoKNN: StandardScaler Obrigatório

**Data**: 2026-06  
**Status**: Aceito

### Contexto
KNN é um algoritmo baseado em distância. Features com escalas muito diferentes (ex: IMC em kg/m² vs. idade em anos) dominam o cálculo de distância, distorcendo os resultados.

### Decisão
`IndividuoKNN.build_model()` sempre retorna:
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(**self.hyperparams))
])
```
O `StandardScaler` é **parte do indivíduo**, não do pipeline externo.

### Justificativa
- RF é invariante à escala (baseado em splits de árvore) → não precisa de scaler
- Encapsular o scaler no Pipeline do KNN evita data leakage (o scaler é fit apenas no treino de cada fold via `cross_val_score`)
- O `ga_evaluator.evaluate()` pode tratar ambos os tipos uniformemente sem lógica condicional

### Consequências
- `IndividuoRF.build_model()` → `Pipeline([("clf", RF(...))])` — sem scaler
- `IndividuoKNN.build_model()` → `Pipeline([("scaler", StandardScaler()), ("clf", KNN(...))])` — scaler incluso
- Ambos retornam `Pipeline` — interface unificada para `cross_val_score` e `fit/predict`

---

## ADR-008 — Mutação com 3 Níveis de Agressividade

**Data**: 2026-06  
**Status**: Aceito

### Contexto
A taxa de mutação fixa (`mutation_rate=0.2`) não é adequada para todos os cenários:
- Exploração inicial → mutação mais agressiva pode ser desejável
- Refinamento final → mutações menores preservam boas soluções

### Decisão
Parâmetro `mutation_aggressiveness: "low" | "medium" | "high"` com mapeamento:

| Nível | `mutation_rate` | `delta_pct` (numérico) |
|---|---|---|
| `low` | 0.1 | ±10% do range |
| `medium` | 0.2 | ±30% do range |
| `high` | 0.4 | ±60% do range |

Parâmetros categóricos (ex: `criterion`, `weights`) usam apenas `mutation_rate` (substituição aleatória).

### Consequências
- Usuário expõe `--aggressiveness` no CLI e no sidebar Streamlit
- Perturbação gaussiana em parâmetros numéricos: `_mutate_int()` com sigma proporcional ao range
- Original nunca é modificado (cópia profunda antes de mutar) — validado por `test_does_not_mutate_original`

---

## ADR-009 — Elitismo Opcional com Elitismo Estrutural Mínimo

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Elitismo garante que o melhor indivíduo nunca é perdido, mas pode reduzir diversidade e causar convergência prematura.

### Decisão
**Duas camadas de elitismo**:

1. **Elitismo opcional** (`elitism: bool`):
   - `True` (padrão): deepcopy do melhor global é reinserido substituindo o pior da sub-pop correspondente após crossover/mutação
   - `False`: nenhuma garantia de preservação do melhor — GA mais exploratório

2. **Elitismo estrutural mínimo** (sempre ativo, independente do flag `elitism`):
   - Após seleção global, garante ao menos 1 sobrevivente de cada tipo (`max(1, ...)`)
   - Evita extinção prematura de um tipo inteiro pela pressão competitiva

### Consequências
- `--no-elitism` no CLI ativa modo exploratório puro
- O elitismo estrutural mínimo é uma salvaguarda implícita — não exposto como parâmetro
- `test_with_elitism_enabled` e `test_with_elitism_disabled` validam ambos os cenários

---

## ADR-010 — Persistência em 3 Formatos: JSON + CSV + joblib

**Data**: 2026-06  
**Status**: Aceito

### Contexto
Os resultados do GA precisam ser consumidos por diferentes audiências:
- **Desenvolvimento**: análise do histórico de evolução via Python/Pandas
- **Visualização**: gráficos no dashboard Streamlit
- **Produção**: modelo sklearn carregável para predições

### Decisão
**3 formatos de saída** via `ga_persistence.save_ga_results()`:

| Arquivo | Formato | Uso |
|---|---|---|
| `ga_history.json` | JSON | Histórico completo estruturado (params + all generations + best_individual) |
| `ga_generation_stats.csv` | CSV | Tabela unificada por geração (rf_* + knn_* + global_*) para análise/visualização |
| `best_model.joblib` | joblib | Pipeline sklearn treinado no dataset completo, pronto para `predict()` |

### Consequências
- `save_best_model()` executa um `fit()` final no dataset completo antes de persistir (o GA avalia via CV, não treina o modelo final)
- `load_ga_history()` permite recarregar resultados sem re-executar o GA
- O CSV tem uma linha por geração com todas as métricas — adequado para `pd.read_csv()` direto no Streamlit
