# Plano de Implementação — Cobertura de Testes do Algoritmo Genético

**Objetivo:** fechar as lacunas de validação identificadas nos testes atuais (`test_ga_operators.py`, `test_ga_evaluator.py`, `tests/integration/test_ga_optimizer.py`), cobrindo especialmente **elitismo**, **elitismo estrutural mínimo**, **seleção por torneio global** e **magnitude estatística** de crossover/mutação — partes do GA hoje sem validação de comportamento correto (só validação de "não quebra").

**Escopo revisado (v2):** ao descobrir `tests/integration/test_ga_optimizer.py`, o plano deixou de precisar de um arquivo novo de integração — quase tudo que faltava na camada `GeneticAlgorithm.run()` já tem uma classe de teste correspondente lá. O trabalho agora é **adicionar métodos novos dentro desse arquivo existente** (não criar `tests/unit/test_genetic_algorithm.py`), mais os reforços já planejados em `test_ga_operators.py`/`test_ga_evaluator.py`.

---

## 0. Pré-requisitos — ✅ resolvido pelo `test_ga_optimizer.py`

A assinatura real, confirmada pelo arquivo de integração já existente:

```python
GeneticAlgorithm(X, y, pop_size=..., max_generations=..., patience=..., k_folds=..., random_seed=..., elitism=...)
result = ga.run()   # sem argumentos — X, y já vão no construtor
```

Formato de retorno confirmado:
```python
{
    "generations_stats": [
        {"rf": {"count": ...}, "knn": {"count": ...}, "global_best_f1": ..., "global_best_type": "RF"|"KNN"},
        ...
    ],
    "best_individual": IndividuoRF | IndividuoKNN,   # com .hyperparams, .fitness_values, .classifier_type
    "stopped_at": int,
    "reason": "max_generations" | "convergence",
    "params": {"pop_size": ..., "max_generations": ..., "patience": ..., "elitism": ..., ...},
}
```

- [ ] Ainda vale confirmar se existe uma função de seleção exportada separadamente (ex: `select_tournament_global()`), necessária para o teste de seleção isolada (item 1.3 abaixo). Se não existir isolada, o teste de seleção precisa rodar via `run()` completo com fitness pré-populado artificialmente, ou ser descartado em favor só do teste de elitismo estrutural (que cobre o efeito prático da seleção).
- [ ] Rodar a suíte atual (`pytest tests/unit tests/integration -q`) e confirmar baseline verde antes de adicionar testes novos.

---

## 1. Extensão de `tests/integration/test_ga_optimizer.py` (arquivo existente)

Nenhum arquivo novo — adicionar métodos dentro das classes já existentes, reaproveitando a fixture `small_data` e o helper `make_ga()`.

### 1.1 `TestElitism` — adicionar

```python
def test_elitism_true_never_regresses(self, small_data):
    """Com elitism=True, global_best_f1 nunca deve piorar entre gerações (ADR-009)."""
    ga = make_ga(small_data, elitism=True, max_generations=6, pop_size=4)
    result = ga.run()
    bests = [g["global_best_f1"] for g in result["generations_stats"]]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bests, bests[1:]))
```

- **Por que importa**: é o teste de maior valor de detecção de bug do plano inteiro — os testes atuais (`test_with_elitism_enabled/disabled`) só checam `best_individual is not None`, o que passa mesmo se a reinserção do elite estiver quebrada.

```python
def test_elitism_false_can_regress(self, small_data):
    """Sem elitismo, não há garantia de não-regressão — probabilístico,
    roda várias seeds e espera observar ao menos 1 regressão em algum run."""
    X, y = small_data
    regressed = False
    for seed in range(5):
        ga = GeneticAlgorithm(X, y, pop_size=4, max_generations=8, k_folds=3,
                               elitism=False, random_seed=seed)
        bests = [g["global_best_f1"] for g in ga.run()["generations_stats"]]
        if any(b2 < b1 - 1e-9 for b1, b2 in zip(bests, bests[1:])):
            regressed = True
            break
    assert regressed, "Esperava-se ao menos uma regressão sem elitismo em 5 seeds"
```

- **Nota de flakiness**: documentar no docstring que é probabilístico. Se ficar instável na CI, trocar por uma checagem mais fraca (variância dos bests com `elitism=False` maior que com `elitism=True`).

### 1.2 Nova classe `TestStructuralElitism` — adicionar

```python
class TestStructuralElitism:
    def test_min_one_survivor_per_type_every_generation(self, small_data):
        """Elitismo estrutural mínimo (ADR-004/009): RF e KNN nunca somem
        completamente da população, mesmo com pressão competitiva."""
        ga = make_ga(small_data, max_generations=6, pop_size=4)
        result = ga.run()
        for stat in result["generations_stats"]:
            assert stat["rf"]["count"] >= 1
            assert stat["knn"]["count"] >= 1
```

- **Gap real**: o teste existente `test_both_types_appear_in_stats` checa `count >= 0`, que é sempre verdadeiro e não valida nada. Esse substitui/reforça essa checagem com `>= 1`.

### 1.3 Nova classe `TestGlobalTournamentSelection` — adicionar (se a seleção for isolável)

```python
class TestGlobalTournamentSelection:
    def test_higher_fitness_wins_more_often(self):
        """Fitness maior deve vencer o torneio com mais frequência ao longo de repetições."""
        # depende de localizar a função de seleção real (ex: select_tournament_global)
        # criar 2 indivíduos com fitness_values fixos (ex: 0.9 vs 0.3), rodar N vezes,
        # contar vitórias do de fitness 0.9; assert proporção > 0.5
```

- **Bloqueado até confirmar** se a seleção está isolada em função própria (ver item 0). Se não estiver, marcar como "não aplicável nesta iteração" e revisitar após um pequeno refactor de extração da função.

### 1.4 `TestStoppingCriteria` — adicionar/reforçar

```python
def test_convergence_actually_triggers(self, small_data):
    """Reforça test_convergence_stops_early: hoje ele aceita qualquer reason,
    então passa mesmo que a convergência nunca dispare de verdade."""
    ga = make_ga(small_data, max_generations=20, patience=2, pop_size=4)
    result = ga.run()
    if result["reason"] == "convergence":
        bests = [g["global_best_f1"] for g in result["generations_stats"]]
        last_n = bests[-3:]
        assert max(last_n) - min(last_n) < 1e-5
    # se nunca convergir em 20 gerações com patience=2, considerar dataset
    # ainda mais trivial (menos features informativas) para forçar plateau
```

### 1.5 `TestCoEvolution` — reforçar reprodutibilidade

```python
def test_reproducibility_full_hyperparams(self, small_data):
    """Reforça test_reproducibility_with_same_seed: hoje só compara
    classifier_type e stopped_at, o que não pega vazamento de RNG em operadores."""
    X, y = small_data
    r1 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99).run()
    r2 = GeneticAlgorithm(X, y, pop_size=3, max_generations=2, k_folds=3, random_seed=99).run()
    assert r1["best_individual"].hyperparams == r2["best_individual"].hyperparams
    assert r1["best_individual"].fitness_values == r2["best_individual"].fitness_values
```

- **Risco conhecido**: se `evaluate()` usa `cross_val_score`/`KFold(shuffle=True)` sem seed fixa por fold, esse teste pode falhar mesmo com o GA determinístico — sinal de que a semente não propaga para o k-Fold. Investigar separadamente se falhar; não ajustar o teste para "passar".

---

## 2. Reforço em `test_ga_operators.py` (arquivo existente)

Adicionar sem remover nada:

- [ ] `TestCrossoverRF.test_indpb_half_swaps_approximately_half` — validação estatística de que `indpb=0.5` produz ~50% de troca de genes ao longo de repetições (hoje só os extremos 0.0/1.0 são testados).
- [ ] `TestMutateRF.test_aggressiveness_ordering` — confirma que `high` produz deltas médios maiores que `medium`, que por sua vez são maiores que `low`, para hiperparâmetros numéricos (ex: `n_estimators`). Hoje nenhum teste verifica a *magnitude* da mutação por nível, só que o resultado fica dentro do range válido (ADR-008 promete ±10%/±30%/±60%).
- [ ] Repetir os dois itens acima para KNN (`TestCrossoverKNN`, `TestMutateKNN`), usando `n_neighbors` como hiperparâmetro numérico de referência.

---

## 3. Reforço em `test_ga_evaluator.py` (arquivo existente)

- [ ] Fortalecer `test_evaluate_returns_zero_on_failure`: hoje o assert só checa `isinstance(result, tuple)` e `len == 2`, o que passa mesmo se o fallback nunca foi acionado de fato. Trocar para `assert (f1, acc) == (0.0, 0.0)` para confirmar que o path de exceção foi realmente exercitado.
- [ ] (Opcional) Adicionar teste de que `evaluate()` propaga `k_folds` corretamente — ex: mockar `cross_val_score` e verificar que é chamado com `cv=k_folds`.

---

## 4. Ordem de implementação sugerida

1. Investigar assinatura real de `genetic_algorithm.py` (item 0) — bloqueia todo o resto.
2. `TestStoppingCriteria` e `TestResultStructure` — mais simples, sem dependência de mock de fitness, bom aquecimento.
3. `TestElitism` (não-regressão) — maior valor de detecção de bug, prioridade alta.
4. `TestReproducibility` — segunda maior prioridade; pode revelar problema de seed no k-Fold.
5. `TestStructuralElitism` e `TestGlobalTournamentSelection` — dependem de como a seleção está exposta; podem exigir pequeno refactor (extrair função de seleção) se hoje estiver 100% inline no `run()`.
6. Reforços em `test_ga_operators.py` e `test_ga_evaluator.py` — baixo risco, podem ser feitos em paralelo a qualquer momento.

---

## 5. Critérios de aceite do plano

- [ ] `pytest tests/unit tests/integration -q` passa 100% (baseline + novos testes).
- [ ] Nenhum teste novo é flaky em 10 execuções consecutivas locais (`pytest --count=10` ou loop manual), especialmente os estatísticos (`indpb=0.5`, `elitism=False`, torneio).
- [ ] Rodando o GA real (`scripts/run_tuning.py` com `--pop-size` pequeno) e inspecionando `models/logs/ga_generation_stats.csv`, a coluna `global_best_fitness` é monotônica não-decrescente quando `--elitism` está ativo — validação manual cruzada com os testes automatizados.
- [ ] Se algum teste novo falhar, documentar o bug encontrado em um ADR ou issue antes de "consertar o teste" — o objetivo é validar o comportamento real, não fazer o teste passar a qualquer custo.

---

## 6. Riscos e pontos de atenção

| Risco | Mitigação |
|---|---|
| Testes estatísticos (torneio, indpb=0.5, aggressiveness) ficam flaky em CI | Usar `random_seed` fixo nesses testes sempre que possível; se ainda assim variar, aumentar N de repetições ou relaxar o threshold |
| Seleção/elitismo não estão isolados em funções testáveis | Pequeno refactor: extrair `select_tournament_global(pool, k)` e `apply_elitism(pop, elite)` como funções puras testáveis, sem mudar o comportamento do `run()` |
| k-Fold não seedado corretamente | O teste de reprodutibilidade (`test_same_seed_same_history`) vai expor isso — tratar como bug real, não ajustar o teste |
| Testes de `run()` completo são lentos (várias gerações × k-Fold) | Manter `pop_size` e `max_generations` pequenos (5 e 3-6) e usar o mesmo `mock_data` fixture já existente em `test_ga_evaluator.py` |
