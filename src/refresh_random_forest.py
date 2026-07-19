"""Atualiza somente o Random Forest após otimização de parâmetros e janela."""

from time import perf_counter

import joblib
import pandas as pd

from src.evaluate_model import evaluate_by_group, evaluate_models
from src.train_ml_models import (
    BASELINE_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_OUTPUT_PATH,
    ITEM_METRICS_OUTPUT_PATH,
    METRICS_OUTPUT_PATH,
    MODEL_NAMES,
    MODELS_DIR,
    PREDICTIONS_OUTPUT_PATH,
    STORE_METRICS_OUTPUT_PATH,
    SUMMARY_OUTPUT_PATH,
    TRAINING_TIMES_OUTPUT_PATH,
    build_random_forest,
    build_summary,
    combine_with_baselines,
    load_modeling_data,
    load_optimized_random_forest_params,
    load_optimized_training_rows,
    recursive_forecast,
    select_training_rows,
)
from src.train_model import temporal_split


def run_refresh() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retreina o vencedor e mantém os demais modelos já avaliados."""
    processed, features = load_modeling_data()
    history, validation, validation_start = temporal_split(processed, 90)
    training = select_training_rows(
        features, validation_start, load_optimized_training_rows()
    )
    model = build_random_forest(load_optimized_random_forest_params())
    started = perf_counter()
    model.fit(
        training[FEATURE_COLUMNS].astype("float32"),
        training["sales"].astype("float32"),
    )
    elapsed = perf_counter() - started
    forecast = recursive_forecast(
        model, history, validation, "prediction_random_forest"
    )

    predictions = pd.read_csv(PREDICTIONS_OUTPUT_PATH, parse_dates=["date"])
    predictions = predictions.drop(columns="prediction_random_forest").merge(
        forecast,
        on=["date", "store", "item"],
        how="left",
        validate="one_to_one",
    ).sort_values(["date", "store", "item"], ignore_index=True)
    prediction_columns = {
        "Random Forest": "prediction_random_forest",
        "Gradient Boosting": "prediction_gradient_boosting",
        "HistGradient Boosting": "prediction_hist_gradient_boosting",
    }
    ml_metrics = evaluate_models(predictions, prediction_columns)
    ranking = combine_with_baselines(ml_metrics)

    store_metrics = evaluate_by_group(
        predictions, "prediction_random_forest", "store", "Random Forest"
    )
    item_metrics = evaluate_by_group(
        predictions, "prediction_random_forest", "item", "Random Forest"
    )
    timing = pd.read_csv(TRAINING_TIMES_OUTPUT_PATH)
    timing.loc[timing["model_key"] == "random_forest", ["training_rows", "training_seconds"]] = [len(training), elapsed]

    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    ranking.to_csv(METRICS_OUTPUT_PATH, index=False)
    store_metrics.to_csv(STORE_METRICS_OUTPUT_PATH, index=False)
    item_metrics.to_csv(ITEM_METRICS_OUTPUT_PATH, index=False)
    timing.to_csv(TRAINING_TIMES_OUTPUT_PATH, index=False)
    pd.DataFrame(
        {
            "model": "Random Forest",
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        FEATURE_IMPORTANCE_OUTPUT_PATH, index=False
    )
    SUMMARY_OUTPUT_PATH.write_text(
        build_summary(training, validation, ranking, timing), encoding="utf-8"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": MODEL_NAMES["random_forest"],
            "feature_columns": FEATURE_COLUMNS,
            "training_start": training["date"].min().isoformat(),
            "training_end": training["date"].max().isoformat(),
            "training_rows": len(training),
            "validation_days": 90,
            "forecast_strategy": "recursive",
        },
        MODELS_DIR / "random_forest.joblib",
        compress=3,
    )
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return predictions, ranking


def main() -> None:
    run_refresh()


if __name__ == "__main__":
    main()
