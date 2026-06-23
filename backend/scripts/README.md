# Scripts de Execução

Este diretório contém scripts executáveis para o pipeline de processamento de dados e treinamento de modelos.

## Scripts Disponíveis

### `run_preprocessing.py`

Executa o pipeline completo de ingestão, pré-processamento e engenharia de features.

**Uso:**

```bash
python scripts/run_preprocessing.py \
    --input data/raw/estado_nutricional_sao_paulo.csv \
    --output data/processed/estado_nutricional_clean.csv \
    --output-encoders models/artifacts/encoders.joblib \
    --format-output csv
```

**Argumentos:**

- `--input` (obrigatório): Caminho do arquivo CSV de entrada
- `--output`: Caminho para salvar o dataset processado (default: `data/processed/estado_nutricional_clean.csv`)
- `--output-encoders`: Caminho para salvar os encoders (default: `models/artifacts/encoders.joblib`)
- `--format-output`: Formato do arquivo de saída: `csv` ou `parquet` (default: `csv`)
- `--remove-pregnant`: Remove registros de gestantes (default: True)
- `--no-remove-pregnant`: Não remove registros de gestantes

**Saída:**

- `data/processed/estado_nutricional_clean.csv`: Dataset limpo e processado
- `models/artifacts/encoders.joblib`: Dicionário com encoders para uso posterior
- Logs detalhados do processo

**Exemplo com Parquet:**

```bash
python scripts/run_preprocessing.py \
    --input data/raw/estado_nutricional_sao_paulo.csv \
    --output data/processed/estado_nutricional_clean.parquet \
    --format-output parquet
```

### `run_tuning.py` (Futuro)

Executará o algoritmo genético para otimização de hiperparâmetros.

### `run_app.py` (Futuro)

Iniciará a aplicação Streamlit.

## Estrutura de Dados

### Entrada

O arquivo de entrada esperado deve ser um CSV com as seguintes colunas principais:

- `NU_IDADE_ANO`: Idade em anos
- `NU_PESO`: Peso em kg
- `NU_ALTURA`: Altura em cm
- `DS_IMC`: Índice de Massa Corporal
- `DS_FASE_VIDA`: Fase da vida (Criança, Adolescente, Adulto, Idoso)
- `SG_SEXO`: Sexo (M/F)
- Colunas de target por faixa etária

### Saída

O arquivo processado contém:

- `NU_IDADE_ANO`: Idade em anos
- `DS_FASE_VIDA`: Fase da vida (codificada numericamente)
- `SG_SEXO`: Sexo (codificado numericamente)
- `NU_PESO`: Peso em kg
- `NU_ALTURA`: Altura em cm
- `DS_IMC`: Índice de Massa Corporal
- `ESTADO_NUTRI`: Estado nutricional consolidado
- `PERC_GORDURA`: Percentual de gordura corporal (calculado)
- `TARGET`: Estado nutricional em formato numérico

## Logs

Os logs são exibidos no console e podem ser salvos em arquivo. Use a variável de ambiente `LOG_LEVEL` para controlar o nível de logging:

```bash
LOG_LEVEL=DEBUG python scripts/run_preprocessing.py ...
```

Níveis: DEBUG, INFO (default), WARNING, ERROR, CRITICAL

## Tratamento de Erros

Os scripts realizam validações de entrada e fornecem mensagens de erro informativas. Verifique:

1. Arquivo de entrada existe e tem caminho correto
2. Arquivo tem permissões de leitura
3. Caminho de saída tem permissões de escrita
4. Espaço em disco suficiente

## Reprodutibilidade

Os scripts usam `random_state=42` para garantir reprodutibilidade dos resultados.
