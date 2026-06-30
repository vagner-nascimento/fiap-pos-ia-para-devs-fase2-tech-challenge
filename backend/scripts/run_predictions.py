#!/usr/bin/env python3
"""
Script para gerar predições usando o modelo treinado e salvar CSV com coluna Prediction.

Uso:
    python scripts/run_predictions.py \\
        --input data/processed/estado_nutricional_clean.csv \\
        --model models/artifacts/best_model.joblib \\
        --output models/artifacts/predictions.csv
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
        description="Gera predições usando modelo treinado e salva CSV com coluna Prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", required=True,
        help="Caminho para o CSV processado (ex: data/processed/estado_nutricional_clean.csv).",
    )
    parser.add_argument(
        "--model", required=True,
        help="Caminho para o modelo treinado (ex: models/artifacts/best_model.joblib).",
    )
    parser.add_argument(
        "--output", default="models/artifacts/predictions.csv",
        help="Caminho de saída para o CSV com predições.",
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


def load_model(model_path: str):
    """Carrega o modelo treinado."""
    path = Path(model_path)
    if not path.exists():
        logger.error("Modelo não encontrado: %s", path)
        sys.exit(1)

    logger.info("Carregando modelo de: %s", path)
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
    # pois o modelo espera apenas features numéricas.
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    print("\n" + "=" * 60)
    print("  🔮 Gerando Predições com Modelo Treinado")
    print("=" * 60)
    print(f"  Input       : {args.input}")
    print(f"  Model       : {args.model}")
    print(f"  Output      : {args.output}")
    print(f"  Target      : {args.target}")
    print("=" * 60 + "\n")

    # Carregar modelo
    model = load_model(args.model)

    # Carregar dados
    df, X, y = load_data(args.input, args.target)

    # Gerar predições
    logger.info("Gerando predições...")
    y_pred = model.predict(X)

    # Decodificar predições para labels textuais
    y_pred_labels = decode_predictions(y_pred)

    # Adicionar coluna Prediction ao DataFrame original
    df_result = df.copy()
    df_result["Prediction"] = y_pred_labels

    # Salvar CSV com predições
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)

    logger.info("CSV com predições salvo em: %s", output_path)

    # Estatísticas das predições
    pred_counts = df_result["Prediction"].value_counts()
    print("\n" + "=" * 60)
    print("  ✅ Predições Geradas")
    print("=" * 60)
    print(f"  Total de registros: {len(df_result)}")
    print(f"  Arquivo salvo em: {output_path}")
    print("\n  Distribuição das Predições:")
    for label, count in pred_counts.items():
        print(f"    {label}: {count} ({100 * count / len(df_result):.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
