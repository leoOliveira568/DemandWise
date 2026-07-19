"""Seleciona hiperparâmetros do Random Forest com validação temporal.

A busca usa uma janela anterior à validação final do projeto. Cada candidato
prevê 60 dias recursivamente, preservando o comportamento esperado em produção.
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
        recursive_forecast,
        select_training_rows,
    )
except ModuleNotFoundError:
    from evaluate_model import evaluate_predictions
    from train_ml_models import (
        FEATURE_COLUMNS,
        build_random_forest,
        load_modeling_data,
        recursive_forecast,
        select_training_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "random_forest_tuning.csv"
PARAMS_PATH = REPORTS_DIR / "random_forest_best_params.json"
SUMMARY_PATH = REPORTS_DIR / "random_forest_tuning_summary.md"

TUNING_END = pd.Timestamp("2017-10-02")
TUNING_DAYS = 60
MAX_TRAINING_ROWS = 180_000

CANDIDATES = {
    "current": {
        "n_estimators": 60,
        "max_depth": 14,
        "min_samples_leaf": 10,
        "max_features": 0.8,
    },
    "regularized": {
        "n_estimators": 70,
        "max_depth": 12,
        "min_samples_leaf": 15,
        "max_features": 0.8,
    },
    "responsive": {
        "n_estimators": 70,
        "max_depth": 16,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
}


def tuning_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Cria o holdout que termina antes da validação oficial."""
    start = TUNING_END - pd.Timedelta(days=TUNING_DAYS - 1)
    history = data[data["date"] < start].copy()
    validation = data[(data["date"] >= start) & (data["date"] <= TUNING_END)].copy()
    if validation["date"].nunique() != TUNING_DAYS:
        raise ValueError("A janela de tuning não possui 60 dias completos.")
    return history, validation, start


def run_optimization() -> pd.DataFrame:
    """Treina os candidatos, avalia o horizonte e persiste o vencedor."""
    processed, features = load_modeling_data()
    history, validation, validation_start = tuning_split(processed)
    training = select_training_rows(features, validation_start, MAX_TRAINING_ROWS)
    rows = []

    print("\nOtimização temporal do Random Forest")
    print("-" * 72)
    print(
        f"Treino: {training['date'].min():%d/%m/%Y} a "
        f"{training['date'].max():%d/%m/%Y} ({len(training):,} linhas)"
    )
    print(
        f"Holdout: {validation['date'].min():%d/%m/%Y} a "
        f"{validation['date'].max():%d/%m/%Y} ({TUNING_DAYS} dias)"
    )

    for candidate, params in CANDIDATES.items():
        print(f"Avaliando configuração '{candidate}'...")
        model = build_random_forest(params)
        started = perf_counter()
        model.fit(
            training[FEATURE_COLUMNS].astype("float32"),
            training["sales"].astype("float32"),
        )
        forecast = recursive_forecast(
            model,
            history,
            validation,
            "prediction",
        )
        evaluated = validation.merge(
            forecast,
            on=["date", "store", "item"],
            how="left",
            validate="one_to_one",
        )
        metrics = evaluate_predictions(evaluated["sales"], evaluated["prediction"])
        rows.append(
            {
                "candidate": candidate,
                **params,
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    results = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ignore_index=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    best_name = str(results.iloc[0]["candidate"])
    best_params = CANDIDATES[best_name]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    PARAMS_PATH.write_text(
        json.dumps({"candidate": best_name, "params": best_params}, indent=2),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        build_summary(results, training, validation),
        encoding="utf-8",
    )
    print("\n" + results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nConfiguração selecionada: {best_name}")
    print("-" * 72)
    return results


def build_summary(
    results: pd.DataFrame,
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    """Cria o relatório da busca temporal."""
    rows = "\n".join(
        f"| {row.candidate} | {int(row.n_estimators)} | {int(row.max_depth)} | "
        f"{int(row.min_samples_leaf)} | {row.max_features:.1f} | {row.MAE:.3f} | "
        f"{row.RMSE:.3f} | {row.SMAPE:.2f}% |"
        for row in results.itertuples(index=False)
    )
    best = results.iloc[0]
    return f"""# DemandWise — Otimização temporal do Random Forest

- Treino: {training['date'].min():%d/%m/%Y} a {training['date'].max():%d/%m/%Y}
- Holdout: {validation['date'].min():%d/%m/%Y} a {validation['date'].max():%d/%m/%Y}
- Horizonte recursivo: {validation['date'].nunique()} dias

| Configuração | Árvores | Profundidade | Folha mínima | Features | MAE | RMSE | SMAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

A configuração selecionada foi **{best['candidate']}**, com MAE de
**{best['MAE']:.3f}** no holdout anterior à validação oficial.
"""


def main() -> None:
    run_optimization()


if __name__ == "__main__":
    main()
