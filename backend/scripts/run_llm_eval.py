#!/usr/bin/env python3
"""
Script de avaliação da qualidade das interpretações do LLM (Agente de Saúde Nutricional).

Executa o conjunto de perguntas-teste definido em `experiments/llm_quality_eval.md`,
captura as respostas do agente ReAct e realiza uma VERIFICAÇÃO FACTUAL DETERMINÍSTICA
(via pandas) para dar suporte à pontuação qualitativa da rubrica.

O objetivo é tornar a avaliação REPRODUTÍVEL: gera um relatório em Markdown/JSON com
as respostas geradas e os valores de referência calculados diretamente do dataset.

Uso:
    python scripts/run_llm_eval.py \\
        --csv models/artifacts/best_model_predictions.csv \\
        --mappings models/artifacts/mappings.json \\
        --output reports/llm_eval_run.md

Requer LLM_API_KEY configurada no .env (chave do Google AI Studio / Gemini).
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Garante que o root do projeto está no path para imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Evita UnicodeEncodeError no Windows ao printar emojis/caracteres UTF-8
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Conjunto de perguntas-teste (espelha experiments/llm_quality_eval.md)
# Cada item: (id, pergunta, ferramenta_esperada, fato_esperado_fn)
# `fato_esperado_fn` calcula o valor de referência determinístico a partir do df.
# --------------------------------------------------------------------------- #
def _pred_col(df: pd.DataFrame) -> str | None:
    for c in ("Prediction", "prediction", "ESTADO_NUTRI"):
        if c in df.columns:
            return c
    return None


def build_questions(df: pd.DataFrame):
    pred = _pred_col(df)

    def dist(_df):
        return _df[pred].value_counts().to_dict() if pred else "N/A"

    def mean_age(_df):
        return round(float(_df["NU_IDADE_ANO"].mean()), 2) if "NU_IDADE_ANO" in _df else "N/A"

    def count_severe(_df):
        if pred is None:
            return "N/A"
        return int((_df[pred].astype(str).str.lower() == "obesidade grave").sum())

    def count_elderly(_df):
        return int((_df["NU_IDADE_ANO"] > 60).sum()) if "NU_IDADE_ANO" in _df else "N/A"

    def imc_by_sex(_df):
        if "DS_IMC" not in _df or "SG_SEXO" not in _df:
            return "N/A"
        return _df.groupby("SG_SEXO")["DS_IMC"].mean().round(2).to_dict()

    def max_weight(_df):
        if "NU_PESO" not in _df:
            return "N/A"
        idx = _df["NU_PESO"].idxmax()
        state = _df.loc[idx, pred] if pred else "N/A"
        return {"NU_PESO": float(_df.loc[idx, "NU_PESO"]), "estado": str(state)}

    return [
        ("Q1", "Qual é a distribuição dos estados nutricionais preditos na base?",
         "get_nutrition_statistics", dist),
        ("Q2", "Qual a média de idade dos pacientes?",
         "get_nutrition_statistics", mean_age),
        ("Q3", "Quantos pacientes têm Obesidade Grave?",
         "filter_nutrition_records", count_severe),
        ("Q4", "Me mostre os pacientes com mais de 60 anos.",
         "filter_nutrition_records", count_elderly),
        ("Q5", "Quais são as recomendações clínicas para pacientes com Obesidade Grave?",
         "get_clinical_recommendations", lambda _df: "Diretriz determinística (ver ferramenta)"),
        ("Q6", "O que recomendar para um paciente eutrófico?",
         "get_clinical_recommendations", lambda _df: "Diretriz determinística (ver ferramenta)"),
        ("Q7", "Existe diferença na média de IMC entre homens e mulheres?",
         "filter_nutrition_records", imc_by_sex),
        ("Q8", "Faça um resumo do perfil de saúde nutricional desta população.",
         "get_nutrition_statistics", dist),
        ("Q9", "Qual paciente tem o maior peso registrado e qual seu estado nutricional?",
         "filter_nutrition_records", max_weight),
        ("Q10", "Qual a taxa de mortalidade dos pacientes obesos nesta base?",
         "nenhuma (pergunta-armadilha)", lambda _df: "N/A — coluna de mortalidade inexistente"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a qualidade das interpretações do LLM sobre dados nutricionais.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", default="models/artifacts/best_model_predictions.csv",
        help="CSV com predições para alimentar o agente.",
    )
    parser.add_argument(
        "--mappings", default="models/artifacts/mappings.json",
        help="JSON de mapeamentos de decodificação de categorias.",
    )
    parser.add_argument(
        "--output", default="reports/llm_eval_run.md",
        help="Caminho de saída do relatório em Markdown.",
    )
    parser.add_argument(
        "--json-output", default=None,
        help="(Opcional) Caminho para salvar o resultado bruto em JSON.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    csv_path = Path(args.csv)
    mappings_path = Path(args.mappings)

    if not csv_path.exists():
        logger.error("CSV não encontrado: %s (execute o pipeline predict antes).", csv_path)
        sys.exit(1)
    if not mappings_path.exists():
        logger.error("Mapeamentos não encontrados: %s", mappings_path)
        sys.exit(1)

    # Import tardio: só precisa do LangChain/Gemini ao executar de fato.
    from src.agents.nutritional_agent import NutritionalHealthAgent

    print("\n" + "=" * 60)
    print("  🧪 Avaliação de Qualidade das Interpretações do LLM")
    print("=" * 60)
    print(f"  CSV        : {csv_path}")
    print(f"  Mappings   : {mappings_path}")
    print(f"  Relatório  : {args.output}")
    print("=" * 60 + "\n")

    logger.info("Inicializando o agente (isso gera o relatório inicial via LLM)...")
    agent = NutritionalHealthAgent.from_files(str(csv_path), str(mappings_path))

    # DataFrame decodificado usado como referência factual
    df = agent.df
    questions = build_questions(df)

    results = []
    for qid, question, expected_tool, fact_fn in questions:
        logger.info("[%s] %s", qid, question)
        try:
            answer = agent.ask(question)
        except Exception as exc:  # noqa: BLE001
            answer = f"[ERRO ao consultar o agente: {exc}]"
            logger.exception("Falha na pergunta %s", qid)

        try:
            reference = fact_fn(df)
        except Exception as exc:  # noqa: BLE001
            reference = f"[ERRO ao calcular referência: {exc}]"

        results.append({
            "id": qid,
            "pergunta": question,
            "ferramenta_esperada": expected_tool,
            "referencia_deterministica": reference,
            "resposta_agente": answer,
        })

    # ------------------------------------------------------------------- #
    # Escreve relatório Markdown
    # ------------------------------------------------------------------- #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Relatório de Execução — Avaliação de Qualidade do LLM",
        "",
        f"- **Data/hora:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Dataset:** `{csv_path}` ({len(df)} registros)",
        f"- **Modelo:** definido em `LLM_MODEL` (padrão `gemini-3.5-flash`)",
        "",
        "> Este relatório é gerado automaticamente. Use-o em conjunto com a rubrica de",
        "> `experiments/llm_quality_eval.md` para atribuir as notas (1–5) por dimensão.",
        "",
        "---",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['id']} — {r['pergunta']}",
            "",
            f"- **Ferramenta esperada:** `{r['ferramenta_esperada']}`",
            f"- **Referência determinística (pandas):** `{r['referencia_deterministica']}`",
            "",
            "**Resposta do agente:**",
            "",
            "> " + str(r["resposta_agente"]).replace("\n", "\n> "),
            "",
            "| Precisão | Completude | Linguagem | Uso de ferramenta correto? |",
            "|:--------:|:----------:|:---------:|:--------------------------:|",
            "|          |            |           |                            |",
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Relatório Markdown salvo em: %s", out_path)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Resultado bruto (JSON) salvo em: %s", json_path)

    print("\n" + "=" * 60)
    print("  ✅ Avaliação concluída")
    print("=" * 60)
    print(f"  Perguntas executadas : {len(results)}")
    print(f"  Relatório            : {out_path}")
    print("  Próximo passo: preencher as notas da rubrica no relatório gerado.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

