"""Cria features históricas para previsão de demanda.

Todas as variáveis derivadas de ``sales`` usam somente observações de datas
anteriores à linha atual. Essa regra evita vazamento temporal e permite que a
base resultante seja usada posteriormente em validação temporal e modelagem.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_INPUT_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "train_features.csv"

SERIES_KEYS = ["store", "item"]
LAG_PERIODS = (7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)

MODEL_FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "sales_by_store_mean",
    "sales_by_item_mean",
    "sales_by_month_mean",
]


def load_processed_train(path: Path = TRAIN_INPUT_PATH) -> pd.DataFrame:
    """Carrega a base de treino processada e valida sua estrutura."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. "
            "Execute primeiro 'python src/make_dataset.py'."
        )

    train = pd.read_csv(path, parse_dates=["date"])
    required_columns = {"date", "store", "item", "sales", "month"}
    missing_columns = required_columns - set(train.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"A base processada não contém as colunas: {missing}")

    if train[["date", *SERIES_KEYS]].duplicated().any():
        raise ValueError("Existem registros duplicados para date, store e item.")

    if train["sales"].isna().any():
        raise ValueError("A coluna sales contém valores ausentes.")

    return train.sort_values([*SERIES_KEYS, "date"], ignore_index=True)


def add_lag_features(
    data: pd.DataFrame,
    lag_periods: Sequence[int] = LAG_PERIODS,
) -> pd.DataFrame:
    """Adiciona vendas defasadas por loja e produto."""
    featured = data.copy()
    grouped_sales = featured.groupby(SERIES_KEYS, sort=False)["sales"]

    for period in lag_periods:
        if period <= 0:
            raise ValueError("Os períodos de lag precisam ser positivos.")
        featured[f"lag_{period}"] = grouped_sales.shift(period).astype("float32")

    return featured


def add_rolling_features(
    data: pd.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Cria estatísticas móveis a partir da venda do dia anterior."""
    featured = data.copy()
    grouped_sales = featured.groupby(SERIES_KEYS, sort=False)["sales"]

    for window in windows:
        if window <= 0:
            raise ValueError("As janelas móveis precisam ser positivas.")
        featured[f"rolling_mean_{window}"] = grouped_sales.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        ).astype("float32")

    featured["rolling_std_7"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).std()
    ).astype("float32")
    return featured


def _prior_date_mean(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    feature_name: str,
) -> pd.DataFrame:
    """Calcula média expansiva usando somente datas anteriores do grupo.

    Primeiro, as vendas são agregadas por data. Assim, registros da mesma data
    nunca entram no histórico uns dos outros, independentemente da ordenação das
    linhas dentro daquele dia.
    """
    date_groups = [*group_columns, "date"]
    daily_statistics = (
        data.groupby(date_groups, as_index=False, sort=False)["sales"]
        .agg(daily_sum="sum", daily_count="count")
        .sort_values(date_groups, ignore_index=True)
    )

    grouped_daily = daily_statistics.groupby(list(group_columns), sort=False)
    previous_sum = (
        grouped_daily["daily_sum"].cumsum() - daily_statistics["daily_sum"]
    )
    previous_count = (
        grouped_daily["daily_count"].cumsum() - daily_statistics["daily_count"]
    )

    daily_statistics[feature_name] = np.where(
        previous_count > 0,
        previous_sum / previous_count,
        np.nan,
    ).astype("float32")

    return data.merge(
        daily_statistics[[*date_groups, feature_name]],
        on=date_groups,
        how="left",
        validate="many_to_one",
        sort=False,
    )


def add_historical_mean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona médias históricas por loja, produto e mês do ano."""
    featured = _prior_date_mean(data, ["store"], "sales_by_store_mean")
    featured = _prior_date_mean(featured, ["item"], "sales_by_item_mean")
    featured = _prior_date_mean(featured, ["month"], "sales_by_month_mean")
    return featured


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Executa todas as etapas de feature engineering do treino."""
    ordered = data.sort_values([*SERIES_KEYS, "date"], ignore_index=True)
    featured = add_lag_features(ordered)
    featured = add_rolling_features(featured)
    featured = add_historical_mean_features(featured)
    return featured.sort_values(["date", *SERIES_KEYS], ignore_index=True)


def validate_features(featured: pd.DataFrame) -> None:
    """Valida estrutura, defasagens e regras básicas contra vazamento."""
    missing_features = set(MODEL_FEATURE_COLUMNS) - set(featured.columns)
    if missing_features:
        missing = ", ".join(sorted(missing_features))
        raise ValueError(f"Features não criadas: {missing}")

    if not featured.equals(
        featured.sort_values(["date", *SERIES_KEYS]).reset_index(drop=True)
    ):
        raise ValueError("A saída não está ordenada por data, loja e produto.")

    first_series = featured[
        (featured["store"] == featured["store"].min())
        & (featured["item"] == featured["item"].min())
    ].sort_values("date")

    for period in LAG_PERIODS:
        expected = first_series["sales"].shift(period)
        actual = first_series[f"lag_{period}"]
        if not np.allclose(actual, expected, equal_nan=True):
            raise ValueError(f"A validação de lag_{period} falhou.")

    if featured.loc[featured["date"] == featured["date"].min(), MODEL_FEATURE_COLUMNS].notna().any().any():
        raise ValueError("A primeira data contém features históricas, indicando vazamento.")


def print_summary(featured: pd.DataFrame) -> None:
    """Exibe um resumo da base de features."""
    print("\nResumo do feature engineering")
    print("-" * 64)
    print(f"Linhas: {len(featured):,}")
    print(f"Colunas: {featured.shape[1]}")
    print(f"Séries loja-produto: {featured[SERIES_KEYS].drop_duplicates().shape[0]:,}")
    print(f"Intervalo: {featured['date'].min():%d/%m/%Y} a {featured['date'].max():%d/%m/%Y}")
    print("\nValores ausentes esperados no início das séries:")
    for feature in MODEL_FEATURE_COLUMNS:
        print(f"  {feature}: {featured[feature].isna().sum():,}")
    print("-" * 64)


def main() -> None:
    """Gera a base de treino com features históricas."""
    train = load_processed_train()
    train_features = create_features(train)
    validate_features(train_features)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(TRAIN_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    print_summary(train_features)
    print(f"Base de features salva em: {TRAIN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
