"""Retreina o melhor modelo e gera a submissão final do DemandWise.

O Random Forest é ajustado novamente com a janela temporal mais recente,
encerrada em 31/12/2017. As vendas de 01/01/2018 a 31/03/2018 são previstas
recursivamente: cada dia futuro usa as previsões dos dias anteriores para
atualizar lags, médias móveis e médias históricas.
"""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

try:
    from src.train_ml_models import (
        FEATURE_COLUMNS,
        MAX_TRAINING_ROWS,
        MODEL_NAMES,
        build_models,
        load_modeling_data,
        load_optimized_training_rows,
        recursive_forecast,
        select_training_rows,
    )
except ModuleNotFoundError:
    from train_ml_models import (
        FEATURE_COLUMNS,
        MAX_TRAINING_ROWS,
        MODEL_NAMES,
        build_models,
        load_modeling_data,
        load_optimized_training_rows,
        recursive_forecast,
        select_training_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TEST_INPUT_PATH = PROCESSED_DATA_DIR / "test_processed.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

TEST_PREDICTIONS_OUTPUT_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"
SUBMISSION_OUTPUT_PATH = SUBMISSIONS_DIR / "demandwise_submission.csv"
FINAL_MODEL_OUTPUT_PATH = MODELS_DIR / "random_forest_final.joblib"
FORECAST_SUMMARY_OUTPUT_PATH = REPORTS_DIR / "future_forecast_summary.md"
MONTHLY_FORECAST_OUTPUT_PATH = REPORTS_DIR / "future_forecast_by_month.csv"
STORE_FORECAST_OUTPUT_PATH = REPORTS_DIR / "future_forecast_by_store.csv"
ITEM_FORECAST_OUTPUT_PATH = REPORTS_DIR / "future_forecast_by_item.csv"

MODEL_KEY = "random_forest"
PREDICTION_COLUMN = "prediction_random_forest"
EXPECTED_FORECAST_DAYS = 90


def load_future_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o teste processado e o modelo de submissão."""
    if not TEST_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {TEST_INPUT_PATH}. "
            "Execute primeiro 'python src/make_dataset.py'."
        )
    if not SAMPLE_SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {SAMPLE_SUBMISSION_PATH}.")

    test = pd.read_csv(TEST_INPUT_PATH, parse_dates=["date"])
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    required_test_columns = {"id", "date", "store", "item"}
    missing_test_columns = required_test_columns - set(test.columns)
    if missing_test_columns:
        raise ValueError(
            f"Colunas ausentes no teste: {sorted(missing_test_columns)}"
        )
    if list(sample_submission.columns) != ["id", "sales"]:
        raise ValueError("sample_submission.csv precisa conter exatamente id,sales.")
    if test["id"].duplicated().any() or sample_submission["id"].duplicated().any():
        raise ValueError("Existem IDs duplicados no teste ou na submissão de exemplo.")
    if set(test["id"]) != set(sample_submission["id"]):
        raise ValueError("Os IDs do teste e da submissão de exemplo não coincidem.")
    if test["date"].nunique() != EXPECTED_FORECAST_DAYS:
        raise ValueError("O teste não possui o horizonte esperado de 90 dias.")
    if test[["date", "store", "item"]].duplicated().any():
        raise ValueError("Existem séries duplicadas na mesma data do teste.")

    return test, sample_submission


def train_final_model(
    features: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> tuple[object, pd.DataFrame, float]:
    """Retreina o Random Forest com dados disponíveis até o início do teste."""
    training = select_training_rows(
        features,
        validation_start=forecast_start,
        max_rows=load_optimized_training_rows(),
    )
    if training["date"].max() >= forecast_start:
        raise ValueError("O treino final contém datas do horizonte futuro.")

    model = build_models()[MODEL_KEY]
    started_at = perf_counter()
    model.fit(
        training[FEATURE_COLUMNS].astype("float32"),
        training["sales"].astype("float32"),
    )
    training_seconds = perf_counter() - started_at
    return model, training, training_seconds


def create_future_forecast(
    model: object,
    history: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Gera o horizonte futuro e associa cada previsão ao ID original."""
    recursive_predictions = recursive_forecast(
        model=model,
        history=history,
        future=test[["date", "store", "item"]],
        prediction_column=PREDICTION_COLUMN,
    )
    forecast = test[["id", "date", "store", "item"]].merge(
        recursive_predictions,
        on=["date", "store", "item"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    forecast = forecast.rename(columns={PREDICTION_COLUMN: "sales"})
    return forecast.sort_values("id", ignore_index=True)


def build_submission(
    forecast: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Monta a entrega na mesma ordem e estrutura da submissão de exemplo."""
    submission = sample_submission[["id"]].merge(
        forecast[["id", "sales"]],
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    submission["sales"] = submission["sales"].round(4)
    return submission


def validate_outputs(
    history: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    forecast: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    """Valida integridade temporal e formato dos arquivos finais."""
    if history["date"].max() >= test["date"].min():
        raise ValueError("O histórico se sobrepõe ao horizonte de teste.")
    if len(forecast) != len(test) or len(submission) != len(sample_submission):
        raise ValueError("A quantidade de previsões não corresponde ao teste.")
    if list(submission.columns) != ["id", "sales"]:
        raise ValueError("A submissão final precisa conter exatamente id,sales.")
    if not submission["id"].equals(sample_submission["id"]):
        raise ValueError("A ordem dos IDs difere do sample_submission.csv.")
    if forecast["id"].nunique() != len(test):
        raise ValueError("A previsão final não possui um valor por ID.")
    if forecast["sales"].isna().any() or submission["sales"].isna().any():
        raise ValueError("Existem previsões ausentes.")
    if not np.isfinite(forecast["sales"]).all():
        raise ValueError("Existem previsões não finitas.")
    if (forecast["sales"] < 0).any() or (submission["sales"] < 0).any():
        raise ValueError("Existem previsões negativas.")
    observations_per_date = forecast.groupby("date").size()
    if not observations_per_date.eq(500).all():
        raise ValueError("Cada data futura precisa conter as 500 séries.")


def save_final_model(
    model: object,
    training: pd.DataFrame,
    test: pd.DataFrame,
    training_seconds: float,
) -> None:
    """Persiste o modelo final e seus metadados de produção."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_key": MODEL_KEY,
            "model_name": MODEL_NAMES[MODEL_KEY],
            "feature_columns": FEATURE_COLUMNS,
            "training_start": training["date"].min().isoformat(),
            "training_end": training["date"].max().isoformat(),
            "training_rows": len(training),
            "forecast_start": test["date"].min().isoformat(),
            "forecast_end": test["date"].max().isoformat(),
            "forecast_days": test["date"].nunique(),
            "forecast_strategy": "recursive",
            "training_seconds": training_seconds,
        },
        FINAL_MODEL_OUTPUT_PATH,
        compress=3,
    )


def build_forecast_summary(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    training_seconds: float,
) -> str:
    """Cria um relatório executivo descritivo das previsões futuras."""
    daily = forecast.groupby("date", as_index=False)["sales"].sum()
    monthly = (
        forecast.assign(month=forecast["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["sales"]
        .sum()
    )
    by_store = forecast.groupby("store", as_index=False)["sales"].sum()
    by_item = forecast.groupby("item", as_index=False)["sales"].sum()

    peak_day = daily.loc[daily["sales"].idxmax()]
    top_store = by_store.loc[by_store["sales"].idxmax()]
    top_item = by_item.loc[by_item["sales"].idxmax()]
    monthly_rows = "\n".join(
        f"| {row.month} | {row.sales:,.0f} |"
        for row in monthly.itertuples(index=False)
    )
    return f"""# DemandWise — Previsão futura

## Configuração

- Modelo: Random Forest
- Treino efetivo: {training['date'].min():%d/%m/%Y} a {training['date'].max():%d/%m/%Y}
- Observações de treino: {len(training):,}
- Features: {len(FEATURE_COLUMNS)}
- Tempo de retreinamento: {training_seconds:.1f} segundos
- Horizonte previsto: {forecast['date'].min():%d/%m/%Y} a {forecast['date'].max():%d/%m/%Y}
- Estratégia: previsão recursiva para 500 combinações loja-produto

## Resumo das previsões

- Demanda total prevista: **{forecast['sales'].sum():,.0f} unidades**
- Média diária prevista: **{daily['sales'].mean():,.0f} unidades**
- Dia de maior demanda: **{peak_day['date']:%d/%m/%Y}**, com **{peak_day['sales']:,.0f} unidades**
- Loja com maior demanda prevista: **Loja {int(top_store['store'])}**
- Produto com maior demanda prevista: **Produto {int(top_item['item'])}**

## Demanda prevista por mês

| Mês | Unidades previstas |
| --- | ---: |
{monthly_rows}

As previsões são estimativas do modelo e devem apoiar decisões de estoque em
conjunto com restrições de capacidade, lead time, nível de serviço e contexto
comercial. O arquivo de entrega foi gerado em
`submissions/demandwise_submission.csv`.
"""


def save_forecast_artifacts(
    forecast: pd.DataFrame,
    submission: pd.DataFrame,
    training: pd.DataFrame,
    training_seconds: float,
) -> None:
    """Salva previsões detalhadas, submissão e agregações de negócio."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    forecast.to_csv(TEST_PREDICTIONS_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)

    monthly = (
        forecast.assign(month=forecast["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["sales"]
        .sum()
    )
    by_store = forecast.groupby("store", as_index=False)["sales"].sum().sort_values(
        "sales", ascending=False, ignore_index=True
    )
    by_item = forecast.groupby("item", as_index=False)["sales"].sum().sort_values(
        "sales", ascending=False, ignore_index=True
    )
    monthly.to_csv(MONTHLY_FORECAST_OUTPUT_PATH, index=False)
    by_store.to_csv(STORE_FORECAST_OUTPUT_PATH, index=False)
    by_item.to_csv(ITEM_FORECAST_OUTPUT_PATH, index=False)
    FORECAST_SUMMARY_OUTPUT_PATH.write_text(
        build_forecast_summary(training, forecast, training_seconds),
        encoding="utf-8",
    )


def run_final_forecast() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o retreinamento e gera a submissão final."""
    processed, features = load_modeling_data()
    test, sample_submission = load_future_data()
    forecast_start = test["date"].min()
    history = processed[processed["date"] < forecast_start].copy()

    print("\nPrevisão final do DemandWise")
    print("-" * 72)
    print(f"Retreinando Random Forest até {history['date'].max():%d/%m/%Y}...")
    model, training, training_seconds = train_final_model(features, forecast_start)
    print(f"Treinamento concluído em {training_seconds:.1f} segundos.")
    print(
        f"Gerando {test['date'].nunique()} dias de previsões recursivas para "
        f"{test[['store', 'item']].drop_duplicates().shape[0]} séries..."
    )
    forecast = create_future_forecast(model, history, test)
    submission = build_submission(forecast, sample_submission)
    validate_outputs(history, test, sample_submission, forecast, submission)

    save_final_model(model, training, test, training_seconds)
    save_forecast_artifacts(forecast, submission, training, training_seconds)

    daily = forecast.groupby("date")["sales"].sum()
    print(f"Demanda total prevista: {forecast['sales'].sum():,.0f} unidades")
    print(f"Média diária prevista: {daily.mean():,.0f} unidades")
    print(f"Submissão salva em: {SUBMISSION_OUTPUT_PATH}")
    print(f"Modelo final salvo em: {FINAL_MODEL_OUTPUT_PATH}")
    print(f"Resumo salvo em: {FORECAST_SUMMARY_OUTPUT_PATH}")
    print("-" * 72)
    return forecast, submission


def main() -> None:
    """Executa o pipeline final pela linha de comando."""
    run_final_forecast()


if __name__ == "__main__":
    main()
