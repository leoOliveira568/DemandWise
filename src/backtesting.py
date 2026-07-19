"""Executa backtesting recursivo em múltiplas janelas temporais."""

import json
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from src.evaluate_model import evaluate_predictions
    from src.train_ml_models import (
        FEATURE_COLUMNS,
        build_random_forest,
        load_modeling_data,
        load_optimized_training_rows,
        recursive_forecast,
        select_training_rows,
    )
except ModuleNotFoundError:
    from evaluate_model import evaluate_predictions
    from train_ml_models import (
        FEATURE_COLUMNS,
        build_random_forest,
        load_modeling_data,
        load_optimized_training_rows,
        recursive_forecast,
        select_training_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_PATH = REPORTS_DIR / "random_forest_best_params.json"
PREDICTIONS_PATH = PROCESSED_DIR / "backtest_predictions.csv"
METRICS_PATH = REPORTS_DIR / "backtest_metrics.csv"
SUMMARY_PATH = REPORTS_DIR / "backtest_summary.md"

FOLDS = 3
HORIZON_DAYS = 90
def generate_folds(
    data: pd.DataFrame,
    n_folds: int = FOLDS,
    horizon_days: int = HORIZON_DAYS,
) -> list[dict]:
    """Gera janelas consecutivas, da mais antiga para a mais recente."""
    if n_folds <= 0 or horizon_days <= 0:
        raise ValueError("n_folds e horizon_days precisam ser positivos.")
    max_date = data["date"].max()
    folds = []
    for offset in reversed(range(n_folds)):
        end = max_date - pd.Timedelta(days=offset * horizon_days)
        start = end - pd.Timedelta(days=horizon_days - 1)
        folds.append(
            {
                "fold": len(folds) + 1,
                "validation_start": start,
                "validation_end": end,
            }
        )
    return folds


def load_selected_params() -> dict:
    """Carrega os parâmetros escolhidos ou usa a configuração padrão."""
    if not PARAMS_PATH.exists():
        return {}
    payload = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("params", {}))


def run_backtesting() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina e avalia o Random Forest em três origens independentes."""
    processed, features = load_modeling_data()
    params = load_selected_params()
    max_training_rows = load_optimized_training_rows()
    fold_predictions = []
    metric_rows = []

    print("\nBacktesting temporal do Random Forest")
    print("-" * 76)
    for fold in generate_folds(processed):
        start = fold["validation_start"]
        end = fold["validation_end"]
        history = processed[processed["date"] < start].copy()
        validation = processed[
            (processed["date"] >= start) & (processed["date"] <= end)
        ].copy()
        if validation["date"].nunique() != HORIZON_DAYS:
            raise ValueError(f"Fold {fold['fold']} não possui 90 dias completos.")
        training = select_training_rows(features, start, max_training_rows)
        model = build_random_forest(params)

        print(
            f"Fold {fold['fold']}: {start:%d/%m/%Y} a {end:%d/%m/%Y} "
            f"({len(training):,} linhas de treino)"
        )
        started = perf_counter()
        model.fit(
            training[FEATURE_COLUMNS].astype("float32"),
            training["sales"].astype("float32"),
        )
        recursive = recursive_forecast(
            model,
            history,
            validation,
            "prediction",
        )
        predictions = validation.merge(
            recursive,
            on=["date", "store", "item"],
            how="left",
            validate="one_to_one",
        )
        predictions["fold"] = fold["fold"]
        predictions["horizon_day"] = (
            predictions["date"] - start
        ).dt.days + 1
        predictions["residual"] = predictions["sales"] - predictions["prediction"]
        predictions["absolute_error"] = predictions["residual"].abs()
        fold_predictions.append(predictions)

        metrics = evaluate_predictions(predictions["sales"], predictions["prediction"])
        metric_rows.append(
            {
                "fold": fold["fold"],
                "validation_start": start,
                "validation_end": end,
                "training_start": training["date"].min(),
                "training_end": training["date"].max(),
                "training_rows": len(training),
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    aggregate = {
        "fold": "mean",
        "MAE": metrics["MAE"].mean(),
        "RMSE": metrics["RMSE"].mean(),
        "MAPE": metrics["MAPE"].mean(),
        "SMAPE": metrics["SMAPE"].mean(),
        "elapsed_seconds": metrics["elapsed_seconds"].sum(),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(METRICS_PATH, index=False, date_format="%Y-%m-%d")
    SUMMARY_PATH.write_text(
        build_summary(metrics, aggregate, params),
        encoding="utf-8",
    )
    print("\n" + metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMAE médio: {aggregate['MAE']:.3f}")
    print("-" * 76)
    return all_predictions, metrics


def build_summary(metrics: pd.DataFrame, aggregate: dict, params: dict) -> str:
    """Cria o relatório de robustez entre janelas."""
    rows = "\n".join(
        f"| {int(row.fold)} | {row.validation_start:%d/%m/%Y} | "
        f"{row.validation_end:%d/%m/%Y} | {row.MAE:.3f} | {row.RMSE:.3f} | "
        f"{row.MAPE:.2f}% | {row.SMAPE:.2f}% |"
        for row in metrics.itertuples(index=False)
    )
    mae_cv = metrics["MAE"].std(ddof=1) / metrics["MAE"].mean() * 100
    return f"""# DemandWise — Backtesting temporal

Foram avaliadas três janelas consecutivas e não sobrepostas de 90 dias. Em
cada fold, o modelo foi treinado somente com dados anteriores à validação e
previu o horizonte recursivamente.

| Fold | Início | Fim | MAE | RMSE | MAPE | SMAPE |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
{rows}

- MAE médio: **{aggregate['MAE']:.3f}**
- RMSE médio: **{aggregate['RMSE']:.3f}**
- SMAPE médio: **{aggregate['SMAPE']:.2f}%**
- Variação relativa do MAE entre folds: **{mae_cv:.2f}%**
- Parâmetros aplicados: `{json.dumps(params, ensure_ascii=False)}`

O resultado médio oferece uma visão mais robusta que um único corte temporal e
serve de base para calibrar os intervalos de previsão.
"""


def main() -> None:
    run_backtesting()


if __name__ == "__main__":
    main()
