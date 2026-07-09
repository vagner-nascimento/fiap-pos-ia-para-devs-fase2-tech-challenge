#!/usr/bin/env python3
"""
Gera uma AMOSTRA representativa (sintética) dos dados SISVAN já com a coluna `Prediction`,
apenas para demonstrar/rodar o harness de avaliação do LLM (`run_llm_eval.py`) sem precisar
executar o pipeline completo (preprocess + tuning genético + predict).

NÃO substitui os dados reais: use somente para validar o fluxo de avaliação.

Saída:
    data/sample/sample_predictions.csv
    data/sample/sample_mappings.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 500

FASES = ["Criança", "Adolescente", "Adulto", "Idoso"]
CLASSES = ["Baixo peso", "Eutrofia", "Risco/Sobrepeso", "Obesidade", "Obesidade Grave"]


def classify(imc: float) -> str:
    if imc < 18.5:
        return "Baixo peso"
    if imc < 25:
        return "Eutrofia"
    if imc < 30:
        return "Risco/Sobrepeso"
    if imc < 35:
        return "Obesidade"
    return "Obesidade Grave"


def deurenberg(imc: float, idade: int, sexo: int) -> float:
    # sexo: 1 = Masculino, 0 = Feminino
    if idade <= 15:
        return round(1.20 * imc + 0.23 * idade - 10.8 * sexo - 5.4, 1)
    return round(1.51 * imc - 0.70 * idade - 3.6 * sexo + 1.4, 1)


def main() -> None:
    idade = RNG.integers(2, 90, size=N)
    sexo = RNG.integers(0, 2, size=N)  # 0 = Feminino, 1 = Masculino (ordem LabelEncoder)
    altura = np.clip(RNG.normal(1.55, 0.20, size=N), 0.80, 1.95).round(2)
    # peso correlacionado à altura + ruído para gerar variedade de IMC
    peso = np.clip(altura * altura * RNG.normal(24, 6, size=N), 10, 160).round(1)
    imc = (peso / (altura ** 2)).round(1)

    fase = np.select(
        [idade < 10, idade < 20, idade < 60],
        [FASES[0], FASES[1], FASES[2]],
        default=FASES[3],
    )

    perc_gordura = [deurenberg(i, a, s) for i, a, s in zip(imc, idade, sexo)]
    prediction = [classify(i) for i in imc]

    df = pd.DataFrame({
        "NU_IDADE_ANO": idade,
        "SG_SEXO": sexo,
        "DS_FASE_VIDA": fase,
        "NU_PESO": peso,
        "NU_ALTURA": altura,
        "DS_IMC": imc,
        "PERC_GORDURA": perc_gordura,
        "Prediction": prediction,
    })

    out_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sample_predictions.csv"
    mappings_path = out_dir / "sample_mappings.json"

    df.to_csv(csv_path, index=False)

    mappings = {"SG_SEXO": {"0": "Feminino", "1": "Masculino"}}
    mappings_path.write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Amostra gerada: {csv_path} ({len(df)} linhas)")
    print(f"Mapeamentos:    {mappings_path}")
    print("\nDistribuição de Prediction:")
    print(df["Prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()

