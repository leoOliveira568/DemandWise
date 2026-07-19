"""Prepara os dados brutos do projeto DemandWise.

O script valida os arquivos de entrada, converte datas, cria atributos
temporais e grava as bases de treino e teste na pasta ``data/processed``.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

TEMPORAL_COLUMNS = [
    "year",
    "month",
    "day",
    "day_of_week",
    "day_name",
    "week_of_year",
    "quarter",
    "is_weekend",
]


def _validate_input_file(path: Path, required_columns: set[str]) -> None:
    """Verifica se um CSV existe e contém as colunas esperadas."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. "
            "Coloque os arquivos do Kaggle em data/raw/."
        )

    columns = set(pd.read_csv(path, nrows=0).columns)
    missing_columns = required_columns - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path.name} não contém as colunas obrigatórias: {missing}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega e valida treino, teste e modelo de submissão."""
    _validate_input_file(TRAIN_PATH, {"date", "store", "item", "sales"})
    _validate_input_file(TEST_PATH, {"id", "date", "store", "item"})
    _validate_input_file(SAMPLE_SUBMISSION_PATH, {"id", "sales"})

    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["date"])
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    return train, test, sample_submission


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos de calendário a partir da coluna ``date``."""
    if "date" not in data.columns:
        raise ValueError("A base precisa conter a coluna 'date'.")

    enriched = data.copy()
    enriched["date"] = pd.to_datetime(enriched["date"], errors="raise")
    enriched["year"] = enriched["date"].dt.year
    enriched["month"] = enriched["date"].dt.month
    enriched["day"] = enriched["date"].dt.day
    enriched["day_of_week"] = enriched["date"].dt.dayofweek
    enriched["day_name"] = enriched["date"].dt.day_name()
    enriched["week_of_year"] = enriched["date"].dt.isocalendar().week.astype("int16")
    enriched["quarter"] = enriched["date"].dt.quarter
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype("int8")
    return enriched


def print_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Exibe um resumo das bases processadas."""
    print("\nResumo do processamento")
    print("-" * 50)
    print(f"Linhas do treino: {len(train):,}")
    print(f"Linhas do teste: {len(test):,}")
    print(
        "Intervalo de datas do treino: "
        f"{train['date'].min():%d/%m/%Y} a {train['date'].max():%d/%m/%Y}"
    )
    print(
        "Intervalo de datas do teste: "
        f"{test['date'].min():%d/%m/%Y} a {test['date'].max():%d/%m/%Y}"
    )
    print(f"Quantidade de lojas: {train['store'].nunique()}")
    print(f"Quantidade de produtos: {train['item'].nunique()}")
    print("-" * 50)


def main() -> None:
    """Executa o pipeline de preparação das bases."""
    train, test, sample_submission = load_raw_data()

    if len(test) != len(sample_submission):
        raise ValueError(
            "test.csv e sample_submission.csv possuem quantidades de linhas diferentes."
        )

    train_processed = add_temporal_features(train).sort_values(
        ["date", "store", "item"], ignore_index=True
    )
    test_processed = add_temporal_features(test).sort_values(
        ["date", "store", "item"], ignore_index=True
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(TRAIN_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    test_processed.to_csv(TEST_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    print_summary(train_processed, test_processed)
    print(f"Treino processado salvo em: {TRAIN_OUTPUT_PATH}")
    print(f"Teste processado salvo em: {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
