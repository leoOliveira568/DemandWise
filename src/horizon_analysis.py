"""Compara estratégias de previsão em diferentes trechos do horizonte."""

from pathlib import Path

import pandas as pd

try:
    from src.evaluate_model import evaluate_predictions
    from src.uncertainty import HORIZON_BINS, HORIZON_LABELS
except ModuleNotFoundError:
    from evaluate_model import evaluate_predictions
    from uncertainty import HORIZON_BINS, HORIZON_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
BACKTEST_PATH = PROCESSED_DIR / "backtest_predictions.csv"
RESULTS_PATH = REPORTS_DIR / "horizon_strategy_comparison.csv"
SUMMARY_PATH = REPORTS_DIR / "horizon_strategy_summary.md"

STRATEGIES = {
    "Random Forest recursivo": "prediction",
    "Média histórica loja-produto": "prediction_store_item_mean",
    "Sazonal ingênuo — mesmo dia do ano anterior": "prediction_seasonal_naive",
}


def add_comparison_predictions(
    backtest: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona os baselines fixo e sazonal sem usar o horizonte futuro."""
    compared_frames = []
    historical_lookup = history[["date", "store", "item", "sales"]].rename(
        columns={"date": "reference_date", "sales": "prediction_seasonal_naive"}
    )
    for fold, validation in backtest.groupby("fold", sort=True):
        validation = validation.copy()
        start = validation["date"].min()
        fold_history = history[history["date"] < start]
        means = (
            fold_history.groupby(["store", "item"], as_index=False)["sales"]
            .mean()
            .rename(columns={"sales": "prediction_store_item_mean"})
        )
        validation = validation.merge(
            means,
            on=["store", "item"],
            how="left",
            validate="many_to_one",
        )
        validation["reference_date"] = validation["date"] - pd.DateOffset(years=1)
        validation = validation.merge(
            historical_lookup,
            on=["reference_date", "store", "item"],
            how="left",
            validate="many_to_one",
        )
        if validation[list(STRATEGIES.values())].isna().any().any():
            raise ValueError(f"Fold {fold} contém baseline ausente.")
        compared_frames.append(validation)
    return pd.concat(compared_frames, ignore_index=True)


def evaluate_by_horizon(compared: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas por estratégia, fold e faixa de horizonte."""
    evaluated = compared.copy()
    evaluated["horizon_bucket"] = pd.cut(
        evaluated["horizon_day"],
        bins=HORIZON_BINS,
        labels=HORIZON_LABELS,
        include_lowest=True,
    ).astype("string")
    rows = []
    for (fold, horizon), group in evaluated.groupby(
        ["fold", "horizon_bucket"], sort=True
    ):
        for strategy, column in STRATEGIES.items():
            rows.append(
                {
                    "fold": fold,
                    "horizon_bucket": horizon,
                    "strategy": strategy,
                    "observations": len(group),
                    **evaluate_predictions(group["sales"], group[column]),
                }
            )
    return pd.DataFrame(rows)


def run_horizon_analysis() -> pd.DataFrame:
    """Executa a comparação e persiste resultados detalhados."""
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError("Execute primeiro 'python src/backtesting.py'.")
    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])
    history = pd.read_csv(
        TRAIN_PATH,
        usecols=["date", "store", "item", "sales"],
        parse_dates=["date"],
    )
    compared = add_comparison_predictions(backtest, history)
    results = evaluate_by_horizon(compared)
    results.to_csv(RESULTS_PATH, index=False)
    SUMMARY_PATH.write_text(build_summary(results), encoding="utf-8")

    average = (
        results.groupby(["horizon_bucket", "strategy"], as_index=False)["MAE"]
        .mean()
        .sort_values(["horizon_bucket", "MAE"])
    )
    print("\nComparação de estratégias por horizonte — MAE médio")
    print(average.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return results


def build_summary(results: pd.DataFrame) -> str:
    average = (
        results.groupby(["horizon_bucket", "strategy"], as_index=False)[
            ["MAE", "RMSE", "SMAPE"]
        ]
        .mean()
        .sort_values(["horizon_bucket", "MAE"])
    )
    rows = "\n".join(
        f"| {row.horizon_bucket} | {row.strategy} | {row.MAE:.3f} | "
        f"{row.RMSE:.3f} | {row.SMAPE:.2f}% |"
        for row in average.itertuples(index=False)
    )
    winners = average.loc[average.groupby("horizon_bucket")["MAE"].idxmin()]
    winner_text = ", ".join(
        f"{row.horizon_bucket}: {row.strategy}" for row in winners.itertuples(index=False)
    )
    return f"""# DemandWise — Comparação por horizonte

As três estratégias foram avaliadas nos mesmos folds do backtesting:

- Random Forest com previsão recursiva;
- média histórica fixa por loja-produto;
- sazonal ingênuo usando o mesmo dia do ano anterior.

| Horizonte | Estratégia | MAE | RMSE | SMAPE |
| --- | --- | ---: | ---: | ---: |
{rows}

Melhor estratégia por faixa: **{winner_text}**.

A decomposição evidencia como o erro evolui quando previsões anteriores passam
a alimentar os lags. Ela também evita concluir sobre um horizonte inteiro a
partir de uma única métrica agregada.
"""


def main() -> None:
    run_horizon_analysis()


if __name__ == "__main__":
    main()
