"""
Script CLI para execução do Algoritmo Genético Co-Evolutivo de tuning.

Uso:
    python scripts/run_tuning.py \\
        --input data/processed/estado_nutricional_clean.csv \\
        --target TARGET \\
        --output-model models/artifacts/best_model.joblib \\
        --output-history models/logs/ \\
        --pop-size 20 \\
        --max-generations 10 \\
        --patience 5 \\
        --k-folds 5 \\
        --aggressiveness medium \\
        --elitism \\
        --cxpb 0.7 \\
        --mutpb 0.3 \\
        --indpb 0.5 \\
        --random-seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

# Garante que o root do projeto está no path para imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.ga_evaluator import fitness_score
from src.models.ga_persistence import load_ga_history, save_best_model, save_ga_results
from src.models.genetic_algorithm import GeneticAlgorithm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning de hiperparâmetros via Algoritmo Genético Co-Evolutivo (RF + KNN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dados ----
    parser.add_argument(
        "--input", required=True,
        help="Caminho para o CSV processado (ex: data/processed/estado_nutricional_clean.csv).",
    )
    parser.add_argument(
        "--target", default="TARGET",
        help="Nome da coluna alvo no CSV.",
    )
    parser.add_argument(
        "--drop-cols", nargs="*", default=[],
        help="Colunas a remover do CSV antes do treino (além da coluna target).",
    )
    parser.add_argument(
        "--sample", type=int, default=50000,
        help=(
            "Número máximo de amostras para treino. A amostragem é ESTRATIFICADA "
            "(proporcional por classe) para preservar a distribuição do target. "
            "Use None para dataset completo (muito mais lento). "
            "Default: 50000 (validação rápida ~10-15min)."
        ),
    )

    # ---- Saídas ----
    parser.add_argument(
        "--output-model", default="models/artifacts/best_model.joblib",
        help="Caminho de saída para o modelo .joblib.",
    )
    parser.add_argument(
        "--output-history", default="models/logs/",
        help="Diretório de saída para ga_history.json e ga_generation_stats.csv.",
    )

    # ---- Parâmetros do GA ----
    parser.add_argument("--pop-size", type=int, default=4,
                        help="Indivíduos por tipo (RF e KNN terão pop_size cada). "
                             "Default: 4 (rápido). Produção recomendado: 20.")
    parser.add_argument("--max-generations", type=int, default=2,
                        help="Número máximo de gerações. "
                             "Default: 2 (rápido). Produção recomendado: 10.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Gerações sem melhoria para parada antecipada. "
                             "Default: 3 (rápido). Produção recomendado: 5.")
    parser.add_argument("--k-folds", type=int, default=3,
                        help="Número de folds para k-Fold CV. "
                             "Default: 3 (rápido). Produção recomendado: 5.")
    parser.add_argument(
        "--aggressiveness", choices=["low", "medium", "high"], default="medium",
        help="Agressividade das mutações.",
    )
    parser.add_argument(
        "--elitism", action="store_true", default=True,
        help="Ativa elitismo (preserva melhor indivíduo global a cada geração).",
    )
    parser.add_argument(
        "--no-elitism", dest="elitism", action="store_false",
        help="Desativa elitismo.",
    )
    parser.add_argument("--cxpb", type=float, default=0.7,
                        help="Probabilidade de crossover.")
    parser.add_argument("--mutpb", type=float, default=0.3,
                        help="Probabilidade de mutação.")
    parser.add_argument("--indpb", type=float, default=0.5,
                        help="Probabilidade de swap por gene no cxUniform.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Semente para reprodutibilidade.")

    # ---- Misc ----
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nível de log.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Amostragem estratificada
# ---------------------------------------------------------------------------

def stratified_sample(X, y, n_samples: int, random_seed: int = 42):
    """
    Retorna um subconjunto estratificado de (X, y) com exatamente `n_samples` linhas.

    A proporção de cada classe é preservada via train_test_split(stratify=y),
    garantindo que o dataset amostrado representa a distribuição original.

    Args:
        X: Features (numpy array).
        y: Target (numpy array).
        n_samples: Número de amostras desejado.
        random_seed: Semente para reprodutibilidade.

    Returns:
        Tupla (X_sample, y_sample) com shape (n_samples, n_features).
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    total = len(y)
    if n_samples >= total:
        logger.info("--sample=%d >= total=%d: usando dataset completo.", n_samples, total)
        return X, y

    sample_ratio = n_samples / total

    # Verifica se alguma classe ficaria com menos amostras do que o k-fold
    classes, counts = np.unique(y, return_counts=True)
    min_class_count = int(min(counts) * sample_ratio)
    if min_class_count < 2:
        logger.warning(
            "Após amostragem, classe minoritária teria apenas %d amostra(s). "
            "Considere aumentar --sample ou reduzir --k-folds.",
            min_class_count,
        )

    # train_test_split com stratify preserva proporções por classe
    _, X_sample, _, y_sample = train_test_split(
        X, y,
        test_size=sample_ratio,
        stratify=y,
        random_state=random_seed,
    )

    logger.info(
        "Amostragem estratificada: %d → %d linhas (%.1f%% do total)",
        total, len(y_sample), 100 * sample_ratio,
    )

    # Log da distribuição por classe após amostragem
    sampled_classes, sampled_counts = np.unique(y_sample, return_counts=True)
    dist = {int(c): int(n) for c, n in zip(sampled_classes, sampled_counts)}
    logger.info("Distribuição por classe após amostragem: %s", dist)

    return X_sample, y_sample


# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------

def load_data(input_path: str, target_col: str, drop_cols: list[str], sample: int = None):
    """Carrega CSV processado e separa X (apenas numéricas, sem leakage) e y."""
    path = Path(input_path)
    if not path.exists():
        logger.error("Arquivo não encontrado: %s", path)
        sys.exit(1)

    logger.info("Carregando dados de: %s", path)
    df = pd.read_csv(path)
    logger.info("Shape original: %s", df.shape)

    if target_col not in df.columns:
        logger.error("Coluna target '%s' não encontrada. Colunas: %s", target_col, list(df.columns))
        sys.exit(1)

    # Remove colunas explicitamente solicitadas e a target
    cols_to_drop = {target_col} | set(c for c in drop_cols if c in df.columns)
    df_features = df.drop(columns=list(cols_to_drop))

    # Detecta colunas que são representação textual do target (leakage direto).
    # Critério: coluna object cujo conjunto de valores únicos tem o mesmo tamanho
    # que os valores únicos do target — ex: ESTADO_NUTRI é o label de TARGET.
    target_n_classes = df[target_col].nunique()
    leakage_cols = [
        c for c in df_features.select_dtypes(include="object").columns
        if df_features[c].nunique() == target_n_classes
    ]
    if leakage_cols:
        logger.warning(
            "Colunas detectadas como representação textual do target (leakage) e removidas: %s",
            leakage_cols,
        )
        print(f"  🚫 Leakage detectado e removido: {leakage_cols}")

    # Filtra apenas colunas numéricas — colunas string/object causam erro no sklearn
    numeric_cols = df_features.select_dtypes(include="number").columns.tolist()
    dropped_non_numeric = [
        c for c in df_features.columns
        if c not in numeric_cols and c not in leakage_cols
    ]
    if dropped_non_numeric:
        logger.warning(
            "Colunas não-numéricas ignoradas (não suportadas pelo sklearn): %s",
            dropped_non_numeric,
        )
        print(f"  ⚠️  Colunas ignoradas (não numéricas): {dropped_non_numeric}")

    X = df_features[numeric_cols].values
    y = df[target_col].values

    logger.info(
        "X shape: %s (%d features numéricas) | y classes: %s",
        X.shape,
        len(numeric_cols),
        sorted(set(y)),
    )
    print(f"  📊 Features usadas ({len(numeric_cols)}): {numeric_cols}")

    # Amostragem estratificada (se solicitada)
    if sample is not None:
        X, y = stratified_sample(X, y, n_samples=sample)

    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    print("\n" + "=" * 60)
    print("  🧬 GA Co-Evolutivo — Tuning de Hiperparâmetros (SISVAN)")
    print("=" * 60)
    print(f"  Input       : {args.input}")
    print(f"  Target      : {args.target}")
    print(f"  Pop size    : {args.pop_size}/tipo  (total={args.pop_size * 2})")
    print(f"  Max ger.    : {args.max_generations}")
    print(f"  Patience    : {args.patience}")
    print(f"  k-Folds     : {args.k_folds}")
    print(f"  Mutação     : {args.aggressiveness}")
    print(f"  Elitismo    : {'✅' if args.elitism else '❌'}")
    print(f"  cxpb/mutpb  : {args.cxpb}/{args.mutpb}  indpb={args.indpb}")
    print(f"  Seed        : {args.random_seed}")
    sample_info = f"{args.sample:,} (estratificado)" if args.sample else "completo (1.5M+)"
    print(f"  Sample      : {sample_info}")
    print("=" * 60 + "\n")

    # Dados
    X, y = load_data(args.input, args.target, args.drop_cols, sample=args.sample)

    # GA
    ga = GeneticAlgorithm(
        X=X,
        y=y,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        patience=args.patience,
        k_folds=args.k_folds,
        mutation_aggressiveness=args.aggressiveness,
        elitism=args.elitism,
        indpb=args.indpb,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        random_seed=args.random_seed,
    )

    results = ga.run()

    # Resumo final
    best = results["best_individual"]
    best_score = fitness_score(best.fitness_values or (0.0, 0.0)) if best else 0.0
    best_f1 = (best.fitness_values or (0.0, 0.0))[0] if best else 0.0
    best_acc = (best.fitness_values or (0.0, 0.0))[1] if best else 0.0

    print("\n" + "=" * 60)
    print("  ✅ Resultado Final")
    print("=" * 60)
    print(f"  Gerações executadas : {results['stopped_at']}")
    print(f"  Motivo de parada    : {results['reason']}")
    print(f"  Melhor tipo         : {best.classifier_type if best else 'N/A'}")
    print(f"  Melhor F1           : {best_f1:.4f}")
    print(f"  Melhor Acurácia     : {best_acc:.4f}")
    print(f"  Fitness score       : {best_score:.4f}")
    print(f"  Hiperparâmetros     : {best.hyperparams if best else {}}")
    print("=" * 60 + "\n")

    # Persistência
    save_ga_results(results, args.output_history)

    if best:
        save_best_model(best, X, y, args.output_model)
        print(f"  📦 Modelo salvo em  : {args.output_model}")
        print(f"  📊 Histórico salvo em: {args.output_history}")

    print("\n  🎉 Tuning concluído!\n")


if __name__ == "__main__":
    main()
