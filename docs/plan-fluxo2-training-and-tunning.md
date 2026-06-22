# Plan: Item 2 - Treinamento e Tuning com Algoritmo Genético Híbrido

## TL;DR
Implementar um sistema de **Algoritmo Genético com população híbrida** (RandomForest + KNeighborsClassifier) que otimiza hiperparâmetros usando fitness ponderado (F1-Score 60% + Acurácia 40%) com k-Fold Cross Validation (5 folds). O sistema deve persistir histórico de evolução, modelos e integrar-se com Streamlit para visualização em tempo real. Usar DEAP como biblioteca base.

---

## Steps

### **Fase 1: Preparação e Instalação de Dependências** *(~30min)*
1. Adicionar `deap>=1.4.0` ao `pyproject.toml` e instalar via UV
2. Adicionar bibliotecas auxiliares: `matplotlib>=3.7.0` (gráficos de evolução), `plotly>=5.14.0` (visualização interativa)
3. Verificar que `scikit-learn`, `pandas`, `numpy`, `joblib` estão disponíveis (já no projeto)

### **Fase 2: Implementação do Núcleo GA** *(paralelo com Fase 1, ~2-3 horas)*
4. Criar `src/models/hybrid_classifier.py` com classe `HybridClassifier`:
   - Atributos: `classifier_type` (str: "RF" ou "KNN"), `hyperparams` (dict), `model` (objeto sklearn)
   - Método `build_model()`: constrói o modelo sklearn baseado no tipo e hiperparâmetros
   - Método `__repr__()` para debug legível

5. Criar `src/models/ga_evaluator.py` com função `evaluate_hybrid()`:
   - Recebe `individual: HybridClassifier`, `X`, `y`, `k_folds=5`, `weight_f1=0.6`, `weight_acc=0.4`
   - Usa `cross_val_score()` com scoring='f1_weighted' e scoring='accuracy'
   - Retorna tupla `(weighted_f1, weighted_accuracy)` como fitness

6. Criar `src/models/ga_operators.py` com operadores genéticos:
   - `create_random_hyperparams(classifier_type)`: gera hiperparâmetros aleatórios por tipo
   - `crossover_hybrid(ind1, ind2)`: crossover inteligente (igual tipo → swap params; tipo diferente → herda do melhor)
   - `mutate_hybrid(individual, mutation_rate=0.2)`: mutação de hiperparâmetros + possibilidade de mudar tipo (10%)

7. Criar `src/models/ga_optimizer.py` com classe `HybridGAOptimizer`:
   - Inicialização: recebe `X, y, pop_size=20, generations=10, k_folds=5`
   - Método `setup_toolbox()`: configura DEAP toolbox com indivíduos, fitness, operadores
   - Método `run()`: executa loop de GA (seleção tournsize=3, crossover prob=0.7, mutação prob=0.3)
   - Método `_log_generation()`: coleta estatísticas por geração (best_f1, best_acc, avg_f1, avg_acc, melhor indivíduo)
   - Método `_finalize()`: retorna dict com histórico, identifica melhor geração

8. Criar `src/models/persistence.py` (complementar ao existente em `src/utils/`):
   - Função `save_ga_results(optimizer_results, output_dir)`: salva em JSON + CSV
   - Função `save_best_model(best_classifier, model_path)`: salva modelo via joblib
   - Função `load_ga_history(json_path)`: carrega histórico para análise

### **Fase 3: Script de Execução** *(~1.5 horas, *depende da Fase 2*)*
9. Criar `scripts/run_tuning.py`:
   - Argumentos: `--input` (path dados processados), `--output-model` (path modelo), `--output-history` (path JSON), `--pop-size`, `--generations`, `--random-seed`
   - Carrega CSV processado de `data/processed/`
   - Separa features (X) e target (y) → usa TARGET (coluna numérica 0-4)
   - Instancia `HybridGAOptimizer` com parâmetros do CLI
   - Executa `.run()` e exibe progress a cada geração
   - Salva modelo best_model.joblib em `models/artifacts/`
   - Salva histórico em `models/logs/` (ga_history.json + ga_generation_stats.csv)
   - Printa resumo final (best generation, best F1, best individual type/hyperparams)

### **Fase 4: Testes** *(~1 hora, *paralelo com Fase 3*)*
10. Criar `tests/unit/test_ga_operators.py`:
    - Teste `create_random_hyperparams()` valida ranges (ex: n_estimators 10-500, max_depth 2-30)
    - Teste `crossover_hybrid()` com mesmo tipo vs. tipos diferentes
    - Teste `mutate_hybrid()` garante que hiperparâmetros permanecem válidos pós-mutação

11. Criar `tests/unit/test_ga_evaluator.py`:
    - Mock data simples (50 samples, 4 features, 3 classes)
    - Teste `evaluate_hybrid()` com RandomForest e KNeighbors
    - Valida que fitness é tupla de 2 floats entre 0-1

12. Criar `tests/integration/test_ga_optimizer.py`:
    - Dados reais pequenos (subset de 500 linhas)
    - Teste `HybridGAOptimizer.run()` com 2 gerações, pop_size=4
    - Valida que generations_stats tem 2 entradas
    - Verifica best_individual contém 'type' e 'hyperparams'

### **Fase 5: Integração Streamlit** *(~2 horas, *depende de Fase 3 & 4*)*
13. Atualizar `src/app/pages/` (criar nova página ou expandir existente):
    - Criar `src/app/pages/tuning_monitor.py` com:
      - Input widgets: pop_size (slider 10-100), generations (slider 5-50), k_folds (select 3/5/10)
      - Botão "Iniciar Tuning" que executa `HybridGAOptimizer.run()` (com `st.spinner()`)
      - Gráfico 1: Evolução F1 vs. Geração (line chart)
      - Gráfico 2: Evolução Acurácia vs. Geração (line chart)
      - Tabela: Estatísticas por geração (best_f1, best_acc, avg_f1, avg_acc)
      - Metadata: Melhor indivíduo (tipo, hyperparams, geração encontrada)
      - Salva histórico em cache/session_state para reutilização

14. Atualizar `main.py` (ou arquivo principal Streamlit):
    - Adicionar navegação para nova página "🧬 Tuning Genético"
    - Integração com agente LLM para interpretar resultados (opcional: "Resumir melhor modelo")

### **Fase 6: Verificação e Documentação** *(~1 hora, *após todas as fases*)*
15. Adicionar docstrings em todos os módulos (Fase 2 & 3)
16. Atualizar `docs/architecture.md` seção "Fluxo de Execução":
    - Expandir item 2 com diagramas mermaid do fluxo GA
    - Documentar estrutura de inputs/outputs
17. Atualizar `scripts/README.md`:
    - Adicionar instruções para `run_tuning.py` com exemplos de CLI

---

## Relevant files

- `pyproject.toml` — Adicionar `deap>=1.4.0`, `matplotlib>=3.7.0`, `plotly>=5.14.0`
- **[Nova]** `src/models/hybrid_classifier.py` — Classe HybridClassifier (wrapper para RF + KNN)
- **[Nova]** `src/models/ga_evaluator.py` — Função evaluate_hybrid() com k-Fold CV
- **[Nova]** `src/models/ga_operators.py` — Operadores genéticos (crossover, mutação, geração aleatória)
- **[Nova]** `src/models/ga_optimizer.py` — Classe HybridGAOptimizer orquestradora + persistência
- **[Nova]** `src/models/persistence.py` — Save/load resultados GA + modelos (extensão de src/utils/persistence.py)
- **[Nova]** `scripts/run_tuning.py` — Script CLI executável com argparse
- **[Nova]** `tests/unit/test_ga_operators.py` — Testes dos operadores
- **[Nova]** `tests/unit/test_ga_evaluator.py` — Testes da função fitness
- **[Nova]** `tests/integration/test_ga_optimizer.py` — Testes end-to-end do GA
- **[Nova]** `src/app/pages/tuning_monitor.py` — Dashboard Streamlit interativo
- `src/app/__init__.py` — Registrar página tuning_monitor
- `main.py` — Adicionar botão de navegação para página Tuning
- `docs/architecture.md` — Expandir seção 2 do pipeline com detalhes GA
- `scripts/README.md` — Documentar run_tuning.py

---

## Verification

1. **Execução CLI**: `python scripts/run_tuning.py --input data/processed/estado_nutricional_clean.csv --generations 5 --pop-size 10` deve completar sem erros e gerar:
   - `models/artifacts/best_model.joblib` (modelo sklearn carregável)
   - `models/logs/ga_history.json` (histórico com 5 gerações)
   - `models/logs/ga_generation_stats.csv` (CSV com 5 linhas)

2. **Testes Unitários**: `pytest tests/unit/test_ga_*.py -v` deve passar todas as 6 suites
   - Validação de ranges de hiperparâmetros
   - Fitness retorna tupla válida (0-1)
   - Crossover/mutação mantêm integridade

3. **Teste Integração**: `pytest tests/integration/test_ga_optimizer.py -v` deve:
   - Executar 2 gerações com pop_size=4 em <10 segundos (mock)
   - Gerar 2 entradas em generation_stats
   - best_individual ter 'type' (RF ou KNN) e 'hyperparams' (dict)

4. **Streamlit Page**: Executar `streamlit run main.py`:
   - Verificar que nova página "🧬 Tuning Genético" aparece na sidebar
   - Valores padrão: pop_size=20, generations=10, k_folds=5
   - Botão "Iniciar Tuning" é clicável
   - Gráficos aparecem após execução
   - Tabela exibe estatísticas com 10 linhas (se 10 gerações)

5. **Persistência**: Verificar que:
   - Modelo salvo pode ser carregado: `joblib.load('models/artifacts/best_model.joblib')`
   - JSON é válido: `json.load(open('models/logs/ga_history.json'))`
   - CSV tem 10+ colunas (generation, best_f1, best_acc, ...)

---

## Decisions

- **Biblioteca GA**: Escolhido **DEAP** por suporte nativo a população heterogênea e flexibilidade
- **Representação Híbrida**: Cada indivíduo é wrapper `HybridClassifier` que encapsula tipo + hiperparâmetros
- **Fitness Ponderado**: F1-Score 60% + Acurácia 40% (não single-metric) porque dados podem ser desbalanceados
- **k-Fold CV**: 5-folds obrigatório (evita overfitting em GA)
- **Operadores Customizados**: Crossover inteligente diferencia por tipo; mutação permite switches de tipo (10%)
- **Persistência em 3 formatos**: JSON (histórico estruturado) + CSV (análise Pandas) + joblib (modelo reutilizável)
- **Integração Streamlit**: Visualização interativa permite acompanhar evolução em tempo real (requisito do usuário)

---

## Further Considerations

1. **Performance vs. Qualidade**: pop_size=20 + generations=10 com k-Fold=5 vai custar ~100 CV folds. Em dados grandes (estado_nutricional_clean.csv tem 50MB+), pode levar 1-2 horas. Recomendação: permitir CLI tuning offline + cache resultados em Streamlit.

2. **Ensemble Futuro**: A população híbrida atualmente seleciona o MELHOR indivíduo. Considerar depois ensemble dos TOP-N modelos (ex: top 3) para predições mais robustas.

3. **Reprodutibilidade**: `random_seed` é argumentos crítico. Garantir que DEAP seed + numpy seed + sklearn seed sejam sincronizados para dados científicos.

