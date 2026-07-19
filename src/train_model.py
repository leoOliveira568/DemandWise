"""Cria e avalia baselines de previsão com validação temporal.

O período final de 90 dias é reservado integralmente para validação, espelhando
o horizonte do ``test.csv``. Nenhum valor desse período é usado para construir
os baselines, o que simula uma previsão feita em uma única data de origem.
"""

from pathlib import Path

import pandas as pd

try:
    from src.evaluate_model import evaluate_by_group, evaluate_models
except ModuleNotFoundError:
    from evaluate_model import evaluate_by_group, evaluate_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_INPUT_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
PREDICTIONS_OUTPUT_PATH = PROCESSED_DATA_DIR / "baseline_validation_predictions.csv"
METRICS_OUTPUT_PATH = REPORTS_DIR / "baseline_metrics.csv"
STORE_METRICS_OUTPUT_PATH = REPORTS_DIR / "baseline_metrics_by_store.csv"
ITEM_METRICS_OUTPUT_PATH = REPORTS_DIR / "baseline_metrics_by_item.csv"
SUMMARY_OUTPUT_PATH = REPORTS_DIR / "baseline_summary.md"

VALIDATION_DAYS = 90
SERIES_KEYS = ["store", "item"]

BASELINE_COLUMNS = {
    "Média global histórica": "prediction_global_mean",
    "Média histórica loja-produto": "prediction_store_item_mean",
    "Média dos últimos 7 dias": "prediction_last_7_mean",
    "Média dos últimos 28 dias": "prediction_last_28_mean",
}


def load_training_data(path: Path = TRAIN_INPUT_PATH) -> pd.DataFrame:
    """Carrega e valida a base processada usada pelos baselines."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. "
            "Execute primeiro 'python src/make_dataset.py'."
        )

    data = pd.read_csv(
        path,
        usecols=["date", "store", "item", "sales"],
        parse_dates=["date"],
    )
    if data.empty:
        raise ValueError("A base de treino está vazia.")
    if data.isna().any().any():
        raise ValueError("A base contém valores ausentes nas colunas essenciais.")
    if data[["date", *SERIES_KEYS]].duplicated().any():
        raise ValueError("Existem registros duplicados para date, store e item.")

    return data.sort_values(["date", *SERIES_KEYS], ignore_index=True)


def temporal_split(
    data: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Reserva os últimos ``validation_days`` dias para validação."""
    if validation_days <= 0:
        raise ValueError("validation_days precisa ser positivo.")

    validation_end = data["date"].max()
    validation_start = validation_end - pd.Timedelta(days=validation_days - 1)
    train = data[data["date"] < validation_start].copy()
    validation = data[data["date"] >= validation_start].copy()

    if train.empty or validation.empty:
        raise ValueError("O corte temporal produziu treino ou validação vazios.")
    if validation["date"].nunique() != validation_days:
        raise ValueError(
            "A base não possui um registro contínuo para todos os dias da validação."
        )
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Os períodos de treino e validação se sobrepõem.")

    return train, validation, validation_start


def _series_statistic(
    values: pd.Series,
    column_name: str,
) -> pd.DataFrame:
    """Converte uma estatística por série em uma tabela para junção."""
    statistics = values.rename(column_name).reset_index()
    expected_columns = {*SERIES_KEYS, column_name}
    if set(statistics.columns) != expected_columns:
        raise ValueError(f"Estrutura inválida para a estatística {column_name}.")
    return statistics


def create_baseline_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Gera previsões fixas na origem usando somente o período de treino."""
    predictions = validation[["date", *SERIES_KEYS, "sales"]].copy()
    global_mean = float(train["sales"].mean())
    predictions["prediction_global_mean"] = global_mean

    series_mean = _series_statistic(
        train.groupby(SERIES_KEYS, sort=False)["sales"].mean(),
        "prediction_store_item_mean",
    )

    ordered_train = train.sort_values([*SERIES_KEYS, "date"])
    last_7_mean = _series_statistic(
        ordered_train.groupby(SERIES_KEYS, sort=False).tail(7)
        .groupby(SERIES_KEYS, sort=False)["sales"]
        .mean(),
        "prediction_last_7_mean",
    )
    last_28_mean = _series_statistic(
        ordered_train.groupby(SERIES_KEYS, sort=False).tail(28)
        .groupby(SERIES_KEYS, sort=False)["sales"]
        .mean(),
        "prediction_last_28_mean",
    )

    for statistics in [series_mean, last_7_mean, last_28_mean]:
        predictions = predictions.merge(
            statistics,
            on=SERIES_KEYS,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    for prediction_column in BASELINE_COLUMNS.values():
        predictions[prediction_column] = (
            predictions[prediction_column]
            .fillna(global_mean)
            .clip(lower=0)
            .astype("float32")
        )

    return predictions.sort_values(["date", *SERIES_KEYS], ignore_index=True)


def validate_baseline_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Confirma integridade do split e das previsões geradas."""
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Foi detectada sobreposição temporal.")
    if len(predictions) != len(validation):
        raise ValueError("A quantidade de previsões não corresponde à validação.")
    if predictions[list(BASELINE_COLUMNS.values())].isna().any().any():
        raise ValueError("Existem previsões ausentes.")
    if (predictions[list(BASELINE_COLUMNS.values())] < 0).any().any():
        raise ValueError("Existem previsões negativas.")

    unique_predictions = predictions.groupby(SERIES_KEYS)[
        list(BASELINE_COLUMNS.values())
    ].nunique()
    if (unique_predictions > 1).any().any():
        raise ValueError(
            "Os baselines mudaram dentro do horizonte; isso indicaria uso da validação."
        )


def build_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    """Cria um relatório Markdown com o resultado da avaliação."""
    best = metrics.iloc[0]
    metric_rows = "\n".join(
        f"| {row.model} | {row.MAE:.3f} | {row.RMSE:.3f} | {row.MAPE:.2f}% | {row.SMAPE:.2f}% |"
        for row in metrics.itertuples(index=False)
    )
    return f"""# DemandWise — Avaliação dos baselines

## Estratégia de validação

- Treino: {train['date'].min():%d/%m/%Y} a {train['date'].max():%d/%m/%Y}
- Validação: {validation['date'].min():%d/%m/%Y} a {validation['date'].max():%d/%m/%Y}
- Horizonte: {validation['date'].nunique()} dias
- Observações de validação: {len(validation):,}
- Origem das previsões: fechamento de {train['date'].max():%d/%m/%Y}

Todas as previsões são calculadas exclusivamente com o período de treino e
permanecem fixas durante os 90 dias de validação.

## Resultados

| Baseline | MAE | RMSE | MAPE | SMAPE |
| --- | ---: | ---: | ---: | ---: |
{metric_rows}

## Melhor baseline

**{best['model']}**, com MAE de **{best['MAE']:.3f}** unidades e SMAPE de
**{best['SMAPE']:.2f}%**.

Esses resultados formam o patamar mínimo que os modelos de machine learning
deverão superar usando o mesmo corte temporal.
"""


def run_baseline_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o pipeline completo de criação e avaliação dos baselines."""
    data = load_training_data()
    train, validation, validation_start = temporal_split(data)
    predictions = create_baseline_predictions(train, validation)
    validate_baseline_predictions(train, validation, predictions)

    metrics = evaluate_models(predictions, BASELINE_COLUMNS)
    best_model_name = str(metrics.iloc[0]["model"])
    best_prediction_column = BASELINE_COLUMNS[best_model_name]

    store_metrics = evaluate_by_group(
        predictions,
        best_prediction_column,
        "store",
        best_model_name,
    )
    item_metrics = evaluate_by_group(
        predictions,
        best_prediction_column,
        "item",
        best_model_name,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    store_metrics.to_csv(STORE_METRICS_OUTPUT_PATH, index=False)
    item_metrics.to_csv(ITEM_METRICS_OUTPUT_PATH, index=False)
    SUMMARY_OUTPUT_PATH.write_text(
        build_summary(train, validation, metrics), encoding="utf-8"
    )

    print("\nAvaliação temporal dos baselines")
    print("-" * 72)
    print(f"Treino: até {train['date'].max():%d/%m/%Y} ({len(train):,} linhas)")
    print(
        f"Validação: {validation_start:%d/%m/%Y} a "
        f"{validation['date'].max():%d/%m/%Y} ({len(validation):,} linhas)"
    )
    print("\n" + metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMelhor baseline: {best_model_name}")
    print(f"Previsões salvas em: {PREDICTIONS_OUTPUT_PATH}")
    print(f"Métricas salvas em: {METRICS_OUTPUT_PATH}")
    print(f"Resumo salvo em: {SUMMARY_OUTPUT_PATH}")
    print("-" * 72)
    return predictions, metrics


def main() -> None:
    """Executa a avaliação de baselines pela linha de comando."""
    run_baseline_evaluation()


if __name__ == "__main__":
    main()
