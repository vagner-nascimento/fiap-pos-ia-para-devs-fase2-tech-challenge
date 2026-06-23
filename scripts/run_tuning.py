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
    parser.add_argument("--pop-size", type=int, default=20,
                        help="Indivíduos por tipo (RF e KNN terão pop_size cada).")
    parser.add_argument("--max-generations", type=int, default=10,
                        help="Número máximo de gerações.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Gerações sem melhoria para parada antecipada.")
    parser.add_argument("--k-folds", type=int, default=5,
                        help="Número de folds para k-Fold CV.")
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
# Carregamento dos dados
# ---------------------------------------------------------------------------

def load_data(input_path: str, target_col: str, drop_cols: list[str]):
    """Carrega CSV processado e separa X e y."""
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

    cols_to_drop = [target_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop).values
    y = df[target_col].values

    logger.info("X shape: %s | y classes: %s", X.shape, sorted(set(y)))
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
    print("=" * 60 + "\n")

    # Dados
    X, y = load_data(args.input, args.target, args.drop_cols)

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
