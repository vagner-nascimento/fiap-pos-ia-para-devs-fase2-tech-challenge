#!/usr/bin/env python3
"""
Script para gerar predições usando o melhor modelo tunado.

Gera predições para:
- Melhor modelo do tuning genético

Uso:
    python scripts/run_predictions_all.py \
        --input data/processed/estado_nutricional_clean.csv \
        --output-dir models/artifacts
"""

import argparse
import logging
import sys
from pathlib import Path

# Garante que o root do projeto está no path para imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib

from src.data.features import NUTRITIONAL_STATE_MAP

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera predições usando modelo tunado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", required=True,
        help="Caminho para o CSV processado (ex: data/processed/estado_nutricional_clean.csv).",
    )
    parser.add_argument(
        "--output-dir", default="models/artifacts",
        help="Diretório de saída para os CSVs com predições.",
    )
    parser.add_argument(
        "--best-model", default="models/artifacts/best_model.joblib",
        help="Caminho para o melhor modelo tunado.",
    )
    parser.add_argument(
        "--target", default="TARGET",
        help="Nome da coluna alvo no CSV.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log.",
    )

    return parser.parse_args()


def load_model(model_path: str, model_name: str):
    """Carrega um modelo treinado."""
    path = Path(model_path)
    if not path.exists():
        logger.error("Modelo %s não encontrado: %s", model_name, path)
        sys.exit(1)

    logger.info("Carregando modelo %s de: %s", model_name, path)
    model = joblib.load(path)
    return model


def load_data(input_path: str, target_col: str):
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

    # Colunas a excluir: target numérico e a coluna textual original (ESTADO_NUTRI),
    # pois os modelos esperam apenas features numéricas.
    cols_to_drop = [target_col]
    if "ESTADO_NUTRI" in df.columns:
        cols_to_drop.append("ESTADO_NUTRI")
        logger.info("Removendo coluna textual 'ESTADO_NUTRI' das features de entrada.")

    # Separa features e target
    X = df.drop(columns=cols_to_drop).values
    y = df[target_col].values

    logger.info("X shape: %s | y shape: %s", X.shape, y.shape)
    return df, X, y


def decode_predictions(y_pred_numeric):
    """Decodifica predições numéricas para labels textuais."""
    # Inverte o mapeamento: {0: "Baixo peso", 1: "Eutrofia", ...}
    reverse_map = {v: k for k, v in NUTRITIONAL_STATE_MAP.items()}
    return [reverse_map.get(pred, "Desconhecido") for pred in y_pred_numeric]


def generate_predictions(model, X, model_name: str):
    """Gera predições usando um modelo específico."""
    logger.info("Gerando predições com modelo %s...", model_name)
    y_pred = model.predict(X)
    y_pred_labels = decode_predictions(y_pred)
    return y_pred_labels


def save_predictions(df, y_pred_labels, output_path: str, model_name: str):
    """Salva predições em CSV."""
    df_result = df.copy()
    df_result["Prediction"] = y_pred_labels

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path_obj, index=False)

    logger.info("Predições do modelo %s salvas em: %s", model_name, output_path_obj)

    # Estatísticas das predições
    pred_counts = df_result["Prediction"].value_counts()
    print(f"\n  Distribuição das Predições ({model_name}):")
    for label, count in pred_counts.items():
        print(f"    {label}: {count} ({100 * count / len(df_result):.1f}%)")

    return output_path_obj


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    print("\n" + "=" * 60)
    print("  🔮 Gerando Predições com Modelo Tunado e Originais")
    print("=" * 60)
    print(f"  Input         : {args.input}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Best model    : {args.best_model}")
    print(f"  Target        : {args.target}")
    print("=" * 60 + "\n")

    # Carregar dados
    df, X, y = load_data(args.input, args.target)

    # Carregar modelos
    print("\nCarregando modelos...")
    best_model = load_model(args.best_model, "Melhor Modelo Tunado")
    
    originals_dir = Path(args.output_dir).parent / "originals"
    original_rf_path = originals_dir / "original_rf.joblib"
    original_knn_path = originals_dir / "original_knn.joblib"
    
    original_models = {}
    if original_rf_path.exists():
        original_models["RF"] = load_model(original_rf_path, "RandomForest Original")
    else:
        logger.warning("Modelo original RF não encontrado: %s", original_rf_path)
    
    if original_knn_path.exists():
        original_models["KNN"] = load_model(original_knn_path, "KNeighborsClassifier Original")
    else:
        logger.warning("Modelo original KNN não encontrado: %s", original_knn_path)

    # Gerar predições
    print("\nGerando predições...")
    best_pred_labels = generate_predictions(best_model, X, "Melhor Modelo Tunado")
    
    original_predictions = {}
    for model_key, model in original_models.items():
        original_predictions[model_key] = generate_predictions(model, X, f"Modelo Original {model_key}")

    # Salvar predições
    print("\nSalvando predições...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_predictions(
        df, best_pred_labels,
        output_dir / "best_model_predictions.csv",
        "Melhor Modelo Tunado"
    )
    
    saved_files = [best_path.name]
    
    for model_key, pred_labels in original_predictions.items():
        model_filename = f"original_{model_key.lower()}_predictions.csv"
        path = save_predictions(
            df, pred_labels,
            output_dir / model_filename,
            f"Modelo Original {model_key}"
        )
        saved_files.append(path.name)

    print("\n" + "=" * 60)
    print("  ✅ Predições Concluídas")
    print("=" * 60)
    print(f"  Total de registros: {len(df)}")
    print(f"  Arquivos salvos em: {output_dir}")
    for filename in saved_files:
        print(f"    - {filename}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
