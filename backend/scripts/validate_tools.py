"""
Script para validação interativa das ferramentas do NutritionalHealthAgent
sem realizar chamadas de rede ou consumir créditos de LLMs.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adiciona o diretório backend ao path para importação local
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.agents.nutritional_agent import NutritionalHealthAgent

# Mock de resposta estática para evitar chamadas de LLM durante o __init__
class MockResponse:
    content = "Relatório Inicial Mockado para Validação Local."

def main():
    print("=" * 70)
    print("  Validador Local de Ferramentas - NutritionalHealthAgent (Sem LLM)")
    print("=" * 70)

    csv_path = Path("models/artifacts/best_model_predictions.csv")
    mappings_path = Path("models/artifacts/mappings.json")

    # Caso os arquivos padrão não existam, gera dados sintéticos de teste
    if not csv_path.exists() or not mappings_path.exists():
        print("[!] Arquivos de modelo/predição reais não encontrados na pasta models/artifacts.")
        print("[*] Usando base de dados sintética de validação...")
        df = pd.DataFrame({
            "NU_IDADE_ANO": [23, 28, 45, 62, 12, 8],
            "DS_FASE_VIDA": ["Adulto", "Adulto", "Adulto", "Idoso", "Adolescente", "Criança"],
            "SG_SEXO": ["Masculino", "Feminino", "Masculino", "Feminino", "Feminino", "Masculino"],
            "NU_PESO": [72.0, 58.5, 95.0, 64.0, 48.0, 26.0],
            "NU_ALTURA": [175.0, 163.0, 172.0, 158.0, 155.0, 122.0],
            "DS_IMC": [23.51, 22.02, 32.11, 25.64, 19.98, 17.47],
            "Prediction": ["Eutrofia", "Eutrofia", "Obesidade", "Sobrepeso", "Eutrofia", "Baixo Peso"]
        })
        mappings = {}
    else:
        print(f"[*] Carregando base de dados real: {csv_path}")
        df = pd.read_csv(csv_path)
        import json
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)

    # Inicializa o agente aplicando patch no invoke do LLM para evitar chamadas de API
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MockResponse()

    print("[*] Inicializando o NutritionalHealthAgent com LLM mockado...")
    with (
        patch("src.agents.nutritional_agent.ChatGoogleGenerativeAI", return_value=mock_llm),
        patch("src.agents.nutritional_agent.PatchedChatOpenAI", return_value=mock_llm),
    ):
        agent = NutritionalHealthAgent(df, mappings)

    print("\n[+] Agente carregado com sucesso!")
    print(f"    - Linhas na base: {len(agent.df)}")
    print(f"    - Colunas disponíveis: {list(agent.df.columns)}")
    print("-" * 70)

    while True:
        print("\nEscolha a ferramenta para testar:")
        print("1. get_nutrition_statistics (Resumo estatístico da base)")
        print("2. filter_nutrition_records (Filtrar linhas com sintaxe Pandas)")
        print("3. get_clinical_recommendations (Diretrizes por categoria)")
        print("4. Sair")

        try:
            opcao = input("\nDigite o número da opção (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaindo...")
            break

        if opcao == "1":
            print("\n--- get_nutrition_statistics ---")
            resultado = agent._tool_get_statistics()
            print(resultado)

        elif opcao == "2":
            print("\n--- filter_nutrition_records ---")
            print("Digite sua query no formato Pandas query. Exemplos:")
            print("  - SG_SEXO == 'Masculino' & NU_IDADE_ANO > 30")
            print("  - Prediction == 'Obesidade Grave'")
            print("  - DS_IMC >= 30")
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                resultado = agent._tool_filter_records(query)
                print("\n" + resultado)
            except Exception as e:
                print(f"\n[Erro]: {e}")

        elif opcao == "3":
            print("\n--- get_clinical_recommendations ---")
            print("Exemplos de categorias: 'Obesidade Grave', 'Eutrofia', 'Sobrepeso', 'Baixo Peso'")
            try:
                categoria = input("\nCategoria: ").strip()
                if not categoria:
                    continue
                resultado = agent._tool_get_recommendations(categoria)
                print("\n" + resultado)
            except Exception as e:
                print(f"\n[Erro]: {e}")

        elif opcao == "4" or opcao.lower() == "sair":
            print("\nSaindo...")
            break
        else:
            print("\n[Opção Inválida] Escolha de 1 a 4.")

if __name__ == "__main__":
    main()
