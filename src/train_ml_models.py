"""Treina e avalia modelos supervisionados de previsão de demanda.

Os modelos são treinados com features históricas calculadas até 02/10/2017 e
avaliados recursivamente nos 90 dias seguintes. Durante a validação, lags,
janelas móveis e médias históricas são atualizados somente com previsões
anteriores; vendas reais futuras nunca entram nas features.
"""

from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

try:
    from src.evaluate_model import evaluate_by_group, evaluate_models
    from src.train_model import BASELINE_COLUMNS, temporal_split
except ModuleNotFoundError:
    from evaluate_model import evaluate_by_group, evaluate_models
    from train_model import BASELINE_COLUMNS, temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.csv"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.csv"

PREDICTIONS_OUTPUT_PATH = PROCESSED_DATA_DIR / "ml_validation_predictions.csv"
METRICS_OUTPUT_PATH = REPORTS_DIR / "model_metrics.csv"
TRAINING_TIMES_OUTPUT_PATH = REPORTS_DIR / "ml_training_times.csv"
STORE_METRICS_OUTPUT_PATH = REPORTS_DIR / "ml_metrics_by_store.csv"
ITEM_METRICS_OUTPUT_PATH = REPORTS_DIR / "ml_metrics_by_item.csv"
FEATURE_IMPORTANCE_OUTPUT_PATH = REPORTS_DIR / "ml_feature_importance.csv"
SUMMARY_OUTPUT_PATH = REPORTS_DIR / "ml_model_summary.md"
OPTIMIZED_PARAMS_PATH = REPORTS_DIR / "random_forest_best_params.json"

VALIDATION_DAYS = 90
MAX_TRAINING_ROWS = 300_000
RANDOM_STATE = 42
SERIES_KEYS = ["store", "item"]

DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 60,
    "max_depth": 14,
    "min_samples_leaf": 10,
    "max_features": 0.8,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

CALENDAR_FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",
]
HISTORICAL_FEATURES = [
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
FEATURE_COLUMNS = [*CALENDAR_FEATURES, *HISTORICAL_FEATURES]

MODEL_NAMES = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradient Boosting",
}


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega as bases processada e enriquecida com features."""
    for path in [TRAIN_PROCESSED_PATH, TRAIN_FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. Execute make_dataset.py e features.py."
            )

    processed = pd.read_csv(
        TRAIN_PROCESSED_PATH,
        usecols=["date", "store", "item", "sales"],
        parse_dates=["date"],
    ).sort_values(["date", *SERIES_KEYS], ignore_index=True)

    features = pd.read_csv(
        TRAIN_FEATURES_PATH,
        usecols=["date", "sales", *FEATURE_COLUMNS],
        parse_dates=["date"],
    ).sort_values(["date", *SERIES_KEYS], ignore_index=True)

    if len(processed) != len(features):
        raise ValueError("As bases processada e de features possuem tamanhos diferentes.")
    if not processed[["date", *SERIES_KEYS]].equals(
        features[["date", *SERIES_KEYS]]
    ):
        raise ValueError("As chaves date, store e item não estão alinhadas entre as bases.")

    return processed, features


def select_training_rows(
    features: pd.DataFrame,
    validation_start: pd.Timestamp,
    max_rows: int = MAX_TRAINING_ROWS,
) -> pd.DataFrame:
    """Seleciona as observações completas mais recentes antes da validação."""
    if max_rows <= 0:
        raise ValueError("max_rows precisa ser positivo.")

    training = features[features["date"] < validation_start].dropna(
        subset=[*FEATURE_COLUMNS, "sales"]
    )
    training = training.sort_values(["date", *SERIES_KEYS], ignore_index=True)
    if len(training) > max_rows:
        training = training.tail(max_rows).reset_index(drop=True)

    if training.empty:
        raise ValueError("Nenhuma observação completa disponível para treinamento.")
    return training


def build_random_forest(params: dict | None = None) -> RandomForestRegressor:
    """Cria o Random Forest padrão ou uma variante de hiperparâmetros."""
    model_params = {**DEFAULT_RANDOM_FOREST_PARAMS, **(params or {})}
    return RandomForestRegressor(**model_params)


def load_optimized_random_forest_params() -> dict:
    """Carrega a configuração temporalmente selecionada, quando disponível."""
    if not OPTIMIZED_PARAMS_PATH.exists():
        return {}
    payload = json.loads(OPTIMIZED_PARAMS_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("params", {}))


def load_optimized_training_rows() -> int:
    """Carrega o tamanho de janela selecionado, quando disponível."""
    if not OPTIMIZED_PARAMS_PATH.exists():
        return MAX_TRAINING_ROWS
    payload = json.loads(OPTIMIZED_PARAMS_PATH.read_text(encoding="utf-8"))
    return int(payload.get("max_training_rows", MAX_TRAINING_ROWS))


def build_models() -> dict[str, object]:
    """Define os três regressores e hiperparâmetros reproduzíveis."""
    return {
        "random_forest": build_random_forest(load_optimized_random_forest_params()),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.7,
            max_features=0.8,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def train_models(
    training: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Ajusta os modelos e registra o tempo de treinamento."""
    x_train = training[FEATURE_COLUMNS].astype("float32")
    y_train = training["sales"].astype("float32")
    fitted_models: dict[str, object] = {}
    timing_rows = []

    for model_key, model in build_models().items():
        print(f"Treinando {MODEL_NAMES[model_key]}...")
        started_at = perf_counter()
        model.fit(x_train, y_train)
        elapsed_seconds = perf_counter() - started_at
        fitted_models[model_key] = model
        timing_rows.append(
            {
                "model_key": model_key,
                "model": MODEL_NAMES[model_key],
                "training_rows": len(training),
                "training_seconds": elapsed_seconds,
            }
        )
        print(f"  concluído em {elapsed_seconds:.1f} segundos")

    return fitted_models, pd.DataFrame(timing_rows)


def _calendar_features(date: pd.Timestamp) -> dict[str, int]:
    """Retorna atributos de calendário para uma data."""
    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "week_of_year": int(date.isocalendar().week),
        "quarter": date.quarter,
        "is_weekend": int(date.dayofweek in (5, 6)),
    }


def recursive_forecast(
    model: object,
    history: pd.DataFrame,
    future: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Prevê cada dia futuro e realimenta o histórico com as previsões."""
    series = (
        future[SERIES_KEYS]
        .drop_duplicates()
        .sort_values(SERIES_KEYS, ignore_index=True)
    )
    series_index = pd.MultiIndex.from_frame(series)
    future_dates = pd.DatetimeIndex(sorted(future["date"].unique()))

    history_matrix = history.pivot(
        index="date", columns=SERIES_KEYS, values="sales"
    ).reindex(columns=series_index)
    if history_matrix.isna().any().any():
        raise ValueError("O histórico não possui todas as séries em todas as datas.")

    history_days = len(history_matrix)
    values = np.full(
        (len(series), history_days + len(future_dates)),
        np.nan,
        dtype="float32",
    )
    values[:, :history_days] = history_matrix.to_numpy(dtype="float32").T

    stores = series["store"].to_numpy(dtype="int16")
    items = series["item"].to_numpy(dtype="int16")

    store_stats = history.groupby("store")["sales"].agg(["sum", "count"])
    item_stats = history.groupby("item")["sales"].agg(["sum", "count"])
    month_history = history.assign(month=history["date"].dt.month)
    month_stats = month_history.groupby("month")["sales"].agg(["sum", "count"])
    store_stats["sum"] = store_stats["sum"].astype("float64")
    item_stats["sum"] = item_stats["sum"].astype("float64")
    month_stats["sum"] = month_stats["sum"].astype("float64")

    forecast_frames = []
    for step, date in enumerate(future_dates):
        position = history_days + step
        calendar = _calendar_features(date)
        model_frame = pd.DataFrame(
            {
                "store": stores,
                "item": items,
                **{name: value for name, value in calendar.items()},
                "lag_7": values[:, position - 7],
                "lag_14": values[:, position - 14],
                "lag_28": values[:, position - 28],
                "rolling_mean_7": values[:, position - 7 : position].mean(axis=1),
                "rolling_mean_14": values[:, position - 14 : position].mean(axis=1),
                "rolling_mean_28": values[:, position - 28 : position].mean(axis=1),
                "rolling_std_7": values[:, position - 7 : position].std(axis=1, ddof=1),
                "sales_by_store_mean": [
                    store_stats.at[store, "sum"] / store_stats.at[store, "count"]
                    for store in stores
                ],
                "sales_by_item_mean": [
                    item_stats.at[item, "sum"] / item_stats.at[item, "count"]
                    for item in items
                ],
                "sales_by_month_mean": (
                    month_stats.at[date.month, "sum"]
                    / month_stats.at[date.month, "count"]
                ),
            }
        )
        model_frame = model_frame[FEATURE_COLUMNS].astype("float32")
        if model_frame.isna().any().any():
            raise ValueError(f"Features ausentes durante a previsão de {date:%d/%m/%Y}.")

        day_predictions = np.asarray(model.predict(model_frame), dtype="float32")
        day_predictions = np.clip(day_predictions, 0, None)
        values[:, position] = day_predictions

        day_frame = series.copy()
        day_frame.insert(0, "date", date)
        day_frame[prediction_column] = day_predictions
        forecast_frames.append(day_frame)

        updates = day_frame.assign(sales=day_predictions)
        store_updates = updates.groupby("store")["sales"].agg(["sum", "count"])
        item_updates = updates.groupby("item")["sales"].agg(["sum", "count"])
        store_stats.loc[store_updates.index, "sum"] += store_updates["sum"]
        store_stats.loc[store_updates.index, "count"] += store_updates["count"]
        item_stats.loc[item_updates.index, "sum"] += item_updates["sum"]
        item_stats.loc[item_updates.index, "count"] += item_updates["count"]
        month_stats.at[date.month, "sum"] += float(day_predictions.sum())
        month_stats.at[date.month, "count"] += len(day_predictions)

    return pd.concat(forecast_frames, ignore_index=True)


def create_recursive_predictions(
    models: dict[str, object],
    history: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Gera e reúne as previsões recursivas de todos os modelos."""
    predictions = validation[["date", *SERIES_KEYS, "sales"]].copy()
    for model_key, model in models.items():
        prediction_column = f"prediction_{model_key}"
        print(f"Prevendo 90 dias com {MODEL_NAMES[model_key]}...")
        model_predictions = recursive_forecast(
            model,
            history,
            validation,
            prediction_column,
        )
        predictions = predictions.merge(
            model_predictions,
            on=["date", *SERIES_KEYS],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    return predictions.sort_values(["date", *SERIES_KEYS], ignore_index=True)


def validate_recursive_predictions(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Valida cobertura, finitude e não negatividade das previsões."""
    prediction_columns = [f"prediction_{key}" for key in MODEL_NAMES]
    if len(predictions) != len(validation):
        raise ValueError("A quantidade de previsões difere da validação.")
    if predictions[prediction_columns].isna().any().any():
        raise ValueError("Existem previsões ausentes.")
    if not np.isfinite(predictions[prediction_columns].to_numpy()).all():
        raise ValueError("Existem previsões não finitas.")
    if (predictions[prediction_columns] < 0).any().any():
        raise ValueError("Existem previsões negativas.")
    if predictions["date"].nunique() != VALIDATION_DAYS:
        raise ValueError("O horizonte recursivo não possui 90 dias.")


def save_models(
    models: dict[str, object],
    training: pd.DataFrame,
) -> None:
    """Persiste modelos e metadados necessários para reprodução."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "training_start": training["date"].min().isoformat(),
        "training_end": training["date"].max().isoformat(),
        "training_rows": len(training),
        "validation_days": VALIDATION_DAYS,
        "forecast_strategy": "recursive",
    }
    for model_key, model in models.items():
        joblib.dump(
            {"model": model, "model_name": MODEL_NAMES[model_key], **metadata},
            MODELS_DIR / f"{model_key}.joblib",
            compress=3,
        )


def extract_feature_importances(models: dict[str, object]) -> pd.DataFrame:
    """Organiza importâncias nativas dos modelos que disponibilizam o atributo."""
    rows = []
    for model_key, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            rows.append(
                {
                    "model": MODEL_NAMES[model_key],
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "importance"],
        ascending=[True, False],
        ignore_index=True,
    )


def combine_with_baselines(ml_metrics: pd.DataFrame) -> pd.DataFrame:
    """Reúne modelos supervisionados e baselines em um único ranking."""
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Métricas de baseline ausentes: {BASELINE_METRICS_PATH}. "
            "Execute 'python src/train_model.py'."
        )
    baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH).drop(columns="rank")
    baseline_metrics.insert(0, "model_type", "Baseline")

    supervised = ml_metrics.drop(columns="rank")
    supervised.insert(0, "model_type", "Supervisionado")
    ranking = pd.concat([baseline_metrics, supervised], ignore_index=True)
    ranking = ranking.sort_values(["MAE", "RMSE"], ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_summary(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    ranking: pd.DataFrame,
    timing: pd.DataFrame,
) -> str:
    """Cria o relatório Markdown da etapa supervisionada."""
    supervised = ranking[ranking["model_type"] == "Supervisionado"]
    baselines = ranking[ranking["model_type"] == "Baseline"]
    best_ml = supervised.iloc[0]
    best_baseline = baselines.iloc[0]
    improvement = (best_baseline["MAE"] - best_ml["MAE"]) / best_baseline["MAE"] * 100
    comparison = "\n".join(
        f"| {row.model_type} | {row.model} | {row.MAE:.3f} | {row.RMSE:.3f} | {row.MAPE:.2f}% | {row.SMAPE:.2f}% |"
        for row in ranking.itertuples(index=False)
    )
    times = "\n".join(
        f"| {row.model} | {row.training_rows:,} | {row.training_seconds:.1f} s |"
        for row in timing.itertuples(index=False)
    )
    gain_text = (
        f"superou o melhor baseline em **{improvement:.2f}%** no MAE"
        if improvement >= 0
        else f"ficou **{abs(improvement):.2f}%** acima do melhor baseline no MAE"
    )
    return f"""# DemandWise — Modelos supervisionados

## Configuração

- Features de treino: {len(FEATURE_COLUMNS)}
- Observações usadas: {len(training):,}
- Janela de treino efetiva: {training['date'].min():%d/%m/%Y} a {training['date'].max():%d/%m/%Y}
- Validação: {validation['date'].min():%d/%m/%Y} a {validation['date'].max():%d/%m/%Y}
- Horizonte: {validation['date'].nunique()} dias
- Estratégia: previsão recursiva por dia e por série loja-produto

Os lags, médias móveis e médias históricas da validação são atualizados apenas
com previsões anteriores. Nenhum valor real do horizonte futuro é usado.

## Ranking geral

| Tipo | Modelo | MAE | RMSE | MAPE | SMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
{comparison}

## Resultado

O melhor modelo supervisionado foi **{best_ml['model']}**, com MAE de
**{best_ml['MAE']:.3f}** e SMAPE de **{best_ml['SMAPE']:.2f}%**. Ele {gain_text}.

## Tempo de treinamento

| Modelo | Linhas | Tempo |
| --- | ---: | ---: |
{times}

O modelo escolhido para a previsão final deverá considerar desempenho,
estabilidade recursiva e custo computacional.
"""


def run_ml_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa treinamento, previsão recursiva, avaliação e persistência."""
    processed, features = load_modeling_data()
    history, validation, validation_start = temporal_split(
        processed,
        validation_days=VALIDATION_DAYS,
    )
    training = select_training_rows(
        features,
        validation_start,
        load_optimized_training_rows(),
    )

    print("\nTreinamento de modelos supervisionados")
    print("-" * 76)
    print(
        f"Amostra temporal: {training['date'].min():%d/%m/%Y} a "
        f"{training['date'].max():%d/%m/%Y} ({len(training):,} linhas)"
    )
    print(
        f"Validação recursiva: {validation['date'].min():%d/%m/%Y} a "
        f"{validation['date'].max():%d/%m/%Y} ({len(validation):,} linhas)\n"
    )

    models, timing = train_models(training)
    predictions = create_recursive_predictions(models, history, validation)
    validate_recursive_predictions(predictions, validation)

    prediction_columns = {
        model_name: f"prediction_{model_key}"
        for model_key, model_name in MODEL_NAMES.items()
    }
    ml_metrics = evaluate_models(predictions, prediction_columns)
    ranking = combine_with_baselines(ml_metrics)

    best_ml_name = str(ml_metrics.iloc[0]["model"])
    best_ml_key = next(key for key, name in MODEL_NAMES.items() if name == best_ml_name)
    best_prediction_column = f"prediction_{best_ml_key}"
    store_metrics = evaluate_by_group(
        predictions,
        best_prediction_column,
        "store",
        best_ml_name,
    )
    item_metrics = evaluate_by_group(
        predictions,
        best_prediction_column,
        "item",
        best_ml_name,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    ranking.to_csv(METRICS_OUTPUT_PATH, index=False)
    timing.to_csv(TRAINING_TIMES_OUTPUT_PATH, index=False)
    store_metrics.to_csv(STORE_METRICS_OUTPUT_PATH, index=False)
    item_metrics.to_csv(ITEM_METRICS_OUTPUT_PATH, index=False)
    extract_feature_importances(models).to_csv(
        FEATURE_IMPORTANCE_OUTPUT_PATH,
        index=False,
    )
    SUMMARY_OUTPUT_PATH.write_text(
        build_summary(training, validation, ranking, timing),
        encoding="utf-8",
    )
    save_models(models, training)

    print("\nRanking geral")
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMelhor modelo supervisionado: {best_ml_name}")
    print(f"Métricas salvas em: {METRICS_OUTPUT_PATH}")
    print(f"Resumo salvo em: {SUMMARY_OUTPUT_PATH}")
    print(f"Modelos salvos em: {MODELS_DIR}")
    print("-" * 76)
    return predictions, ranking


def main() -> None:
    """Executa a etapa supervisionada pela linha de comando."""
    run_ml_evaluation()


if __name__ == "__main__":
    main()
