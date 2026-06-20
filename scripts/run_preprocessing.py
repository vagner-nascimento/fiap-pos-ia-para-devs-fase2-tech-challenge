#!/usr/bin/env python3
"""
Script para executar o pipeline completo de pré-processamento e engenharia de features.

Fluxo:
1. Ingesta de dados brutos (CSV)
2. Pré-processamento (limpeza, consolidação de target)
3. Engenharia de features (codificação, cálculo de percentual de gordura)
4. Salvamento do dataset processado

Uso:
    python scripts/run_preprocessing.py --input data/raw/estado_nutricional_sao_paulo.csv \
                                        --output data/processed/estado_nutricional_clean.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import read_csv_data, validate_dataframe
from src.data.preprocessing import run_preprocessing
from src.data.features import run_feature_engineering
from src.utils.persistence import save_dataframe, save_dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Executa o pipeline completo."""
    parser = argparse.ArgumentParser(
        description="Pipeline de pré-processamento de dados de estado nutricional"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Caminho do arquivo CSV de entrada",
    )
    parser.add_argument(
        "--output",
        default="data/processed/estado_nutricional_clean.csv",
        help="Caminho para salvar o dataset processado",
    )
    parser.add_argument(
        "--output-encoders",
        default="models/artifacts/encoders.joblib",
        help="Caminho para salvar os encoders",
    )
    parser.add_argument(
        "--format-output",
        choices=["csv", "parquet"],
        default="csv",
        help="Formato do arquivo de saída",
    )
    parser.add_argument(
        "--remove-pregnant",
        action="store_true",
        default=True,
        help="Remover registros de gestantes",
    )
    parser.add_argument(
        "--no-remove-pregnant",
        dest="remove_pregnant",
        action="store_false",
        help="Não remover registros de gestantes",
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 80)
        logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        logger.info("=" * 80)

        # 1. Ingestão de dados
        logger.info(f"\n[1/4] Ingestão de dados: {args.input}")
        df = read_csv_data(args.input)
        
        # Validar estrutura básica
        required_columns = ["NU_IDADE_ANO", "NU_PESO", "NU_ALTURA", "DS_IMC"]
        validate_dataframe(df, required_columns)
        logger.info(f"✓ Dados ingeridos com sucesso. Shape: {df.shape}")

        # 2. Pré-processamento
        logger.info(f"\n[2/4] Pré-processamento de dados")
        df_clean = run_preprocessing(
            df,
            remove_pregnant=args.remove_pregnant,
            drop_cols=True,
            standardize=True,
            convert_numeric=True,
        )
        logger.info(f"✓ Pré-processamento concluído. Shape: {df_clean.shape}")

        # 3. Engenharia de features
        logger.info(f"\n[3/4] Engenharia de features")
        df_features, encoders = run_feature_engineering(
            df_clean,
            create_target=True,
            encode_cats=True,
            calculate_fat=True,
        )
        logger.info(f"✓ Features engenheiradas. Shape: {df_features.shape}")

        # 4. Salvamento de resultados
        logger.info(f"\n[4/4] Salvamento de resultados")
        output_path = args.output
        
        save_dataframe(
            df_features,
            output_path,
            format=args.format_output,
        )
        logger.info(f"✓ Dataset salvo em: {output_path}")

        # Salvar encoders para uso posterior
        save_dict(
            encoders,
            args.output_encoders,
            format="joblib",
        )
        logger.info(f"✓ Encoders salvos em: {args.output_encoders}")

        # Relatório final
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
        logger.info("=" * 80)
        logger.info(f"Dataset final:")
        logger.info(f"  - Registros: {len(df_features)}")
        logger.info(f"  - Features: {len(df_features.columns)}")
        logger.info(f"  - Colunas: {df_features.columns.tolist()}")
        logger.info(f"  - Arquivo: {output_path}")
        logger.info(f"  - Encoders: {args.output_encoders}")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("ERRO NO PIPELINE")
        logger.error("=" * 80)
        logger.error(f"Erro: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
