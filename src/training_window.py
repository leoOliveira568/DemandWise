"""Compara tamanhos de janela em um holdout anterior à validação oficial.

A janela de treinamento é uma decisão de modelagem e, por isso, não deve ser
escolhida nos mesmos 90 dias usados para reportar o resultado final. Este
módulo reutiliza o holdout de desenvolvimento encerrado em 02/10/2017 e mantém
o período de 03/10/2017 a 31/12/2017 intocado para a avaliação oficial.
"""

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
        load_optimized_random_forest_params,
        recursive_forecast,
        select_training_rows,
    )
    from src.optimize_model import (
        MAX_TRAINING_ROWS as TUNING_MAX_TRAINING_ROWS,
        RESULTS_PATH as TUNING_RESULTS_PATH,
        tuning_split,
    )
except ModuleNotFoundError:
    from evaluate_model import evaluate_predictions
    from train_ml_models import (
        FEATURE_COLUMNS,
        build_random_forest,
        load_modeling_data,
        load_optimized_random_forest_params,
        recursive_forecast,
        select_training_rows,
    )
    from optimize_model import (
        MAX_TRAINING_ROWS as TUNING_MAX_TRAINING_ROWS,
        RESULTS_PATH as TUNING_RESULTS_PATH,
        tuning_split,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_PATH = REPORTS_DIR / "random_forest_best_params.json"
RESULTS_PATH = REPORTS_DIR / "training_window_comparison.csv"
SUMMARY_PATH = REPORTS_DIR / "training_window_summary.md"

CANDIDATE_ROWS = [180_000, 220_000, 300_000]


def evaluate_frame(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return evaluate_predictions(actual, predicted)


def run_window_comparison() -> pd.DataFrame:
    """Seleciona a janela sem consultar a validação oficial de 90 dias."""
    processed, features = load_modeling_data()
    history, validation, validation_start = tuning_split(processed)
    optimized_params = load_optimized_random_forest_params()
    rows = []

    reusable_metrics = None
    if TUNING_RESULTS_PATH.exists():
        tuning_results = pd.read_csv(TUNING_RESULTS_PATH)
        payload = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        selected_candidate = tuning_results[
            tuning_results["candidate"] == payload.get("candidate")
        ]
        if len(selected_candidate) == 1:
            reusable_metrics = selected_candidate.iloc[0]

    for max_rows in CANDIDATE_ROWS:
        if max_rows == TUNING_MAX_TRAINING_ROWS and reusable_metrics is not None:
            metrics = {
                name: float(reusable_metrics[name])
                for name in ["MAE", "RMSE", "MAPE", "SMAPE"]
            }
            elapsed = 0.0
            source = "reused_tuning"
        else:
            training = select_training_rows(features, validation_start, max_rows)
            model = build_random_forest(optimized_params)
            started = perf_counter()
            model.fit(
                training[FEATURE_COLUMNS].astype("float32"),
                training["sales"].astype("float32"),
            )
            forecast = recursive_forecast(
                model, history, validation, "prediction"
            )
            evaluated = validation.merge(
                forecast,
                on=["date", "store", "item"],
                how="left",
                validate="one_to_one",
            )
            elapsed = perf_counter() - started
            metrics = evaluate_frame(evaluated["sales"], evaluated["prediction"])
            source = "trained_development_holdout"
        rows.append(
            {
                "max_training_rows": max_rows,
                "source": source,
                **metrics,
                "elapsed_seconds": elapsed,
            }
        )

    results = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ignore_index=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    selected_rows = int(results.iloc[0]["max_training_rows"])

    payload = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    payload["max_training_rows"] = selected_rows
    PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results.to_csv(RESULTS_PATH, index=False)
    SUMMARY_PATH.write_text(
        build_summary(results, validation),
        encoding="utf-8",
    )

    print("\nComparação de janela de treinamento")
    print("-" * 68)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nJanela selecionada: {selected_rows:,} linhas")
    print("-" * 68)
    return results


def build_summary(results: pd.DataFrame, validation: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {int(row.max_training_rows):,} | {row.MAE:.3f} | {row.RMSE:.3f} | "
        f"{row.MAPE:.2f}% | {row.SMAPE:.2f}% |"
        for row in results.itertuples(index=False)
    )
    best = results.iloc[0]
    return f"""# DemandWise — Janela de treinamento

As janelas foram comparadas no holdout de desenvolvimento de
**{validation['date'].min():%d/%m/%Y} a {validation['date'].max():%d/%m/%Y}**,
anterior à validação oficial. Os hiperparâmetros já selecionados foram mantidos.

| Linhas mais recentes | MAE | RMSE | MAPE | SMAPE |
| ---: | ---: | ---: | ---: | ---: |
{rows}

A janela selecionada foi **{int(best['max_training_rows']):,} linhas**, com MAE
de **{best['MAE']:.3f}**. O corte é temporal e mantém todas as 500 séries em cada
dia; nenhuma amostragem aleatória é usada. Os últimos 90 dias de 2017 não
participam desta escolha e permanecem independentes para o resultado final.
"""


def main() -> None:
    run_window_comparison()


if __name__ == "__main__":
    main()
