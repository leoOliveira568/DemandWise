"""Calibra intervalos conformais e os aplica às previsões futuras."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

BACKTEST_PATH = PROCESSED_DIR / "backtest_predictions.csv"
FUTURE_PATH = PROCESSED_DIR / "test_predictions.csv"
FUTURE_INTERVALS_PATH = PROCESSED_DIR / "test_predictions_with_intervals.csv"
VALIDATION_INTERVALS_PATH = PROCESSED_DIR / "interval_validation_predictions.csv"
CALIBRATION_PATH = REPORTS_DIR / "interval_calibration.csv"
EVALUATION_PATH = REPORTS_DIR / "interval_evaluation.csv"
SUMMARY_PATH = REPORTS_DIR / "prediction_intervals_summary.md"

TARGET_COVERAGE = 0.90
TEMPORAL_BUFFER_COVERAGE = 0.94
HORIZON_BINS = [0, 7, 28, 60, 90]
HORIZON_LABELS = ["01-07", "08-28", "29-60", "61-90"]


def add_interval_groups(
    data: pd.DataFrame,
    volume_edges: tuple[float, float],
    prediction_column: str,
    horizon_column: str = "horizon_day",
) -> pd.DataFrame:
    """Classifica horizonte e volume para uma calibração condicional simples."""
    grouped = data.copy()
    grouped["horizon_bucket"] = pd.cut(
        grouped[horizon_column],
        bins=HORIZON_BINS,
        labels=HORIZON_LABELS,
        include_lowest=True,
    ).astype("string")
    low_edge, high_edge = volume_edges
    grouped["volume_bucket"] = pd.cut(
        grouped[prediction_column],
        bins=[-np.inf, low_edge, high_edge, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype("string")
    if grouped[["horizon_bucket", "volume_bucket"]].isna().any().any():
        raise ValueError("Não foi possível classificar todos os registros de intervalo.")
    return grouped


def conformal_quantile(values: pd.Series, coverage: float = TARGET_COVERAGE) -> float:
    """Calcula o quantil conformal finito com arredondamento conservador."""
    clean = values.dropna().to_numpy(dtype="float64")
    if clean.size == 0:
        raise ValueError("Não há resíduos para calibrar o intervalo.")
    level = min(np.ceil((clean.size + 1) * coverage) / clean.size, 1.0)
    return float(np.quantile(clean, level, method="higher"))


def build_calibration_table(
    calibration: pd.DataFrame,
    coverage: float = TARGET_COVERAGE,
) -> pd.DataFrame:
    """Calcula a margem absoluta por faixa de horizonte e volume."""
    rows = []
    for (horizon, volume), group in calibration.groupby(
        ["horizon_bucket", "volume_bucket"],
        observed=True,
    ):
        rows.append(
            {
                "horizon_bucket": horizon,
                "volume_bucket": volume,
                "observations": len(group),
                "target_coverage": coverage,
                "absolute_margin": conformal_quantile(group["absolute_error"], coverage),
            }
        )
    return pd.DataFrame(rows)


def apply_calibration(
    data: pd.DataFrame,
    calibration_table: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Adiciona limite inferior, superior e largura do intervalo."""
    calibrated = data.merge(
        calibration_table[
            ["horizon_bucket", "volume_bucket", "absolute_margin"]
        ],
        on=["horizon_bucket", "volume_bucket"],
        how="left",
        validate="many_to_one",
    )
    if calibrated["absolute_margin"].isna().any():
        raise ValueError("Há grupos futuros sem margem de calibração.")
    calibrated["prediction_lower"] = (
        calibrated[prediction_column] - calibrated["absolute_margin"]
    ).clip(lower=0)
    calibrated["prediction_upper"] = (
        calibrated[prediction_column] + calibrated["absolute_margin"]
    )
    calibrated["interval_width"] = (
        calibrated["prediction_upper"] - calibrated["prediction_lower"]
    )
    return calibrated


def evaluate_intervals(data: pd.DataFrame) -> dict[str, float]:
    """Avalia cobertura e largura quando o valor real está disponível."""
    covered = data["sales"].between(
        data["prediction_lower"], data["prediction_upper"], inclusive="both"
    )
    return {
        "observations": len(data),
        "coverage": float(covered.mean()),
        "average_width": float(data["interval_width"].mean()),
        "median_width": float(data["interval_width"].median()),
    }


def run_uncertainty_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibra em folds antigos, audita no mais recente e projeta o teste."""
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError("Execute primeiro 'python src/backtesting.py'.")
    if not FUTURE_PATH.exists():
        raise FileNotFoundError("Execute primeiro 'python src/predict.py'.")

    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])
    future = pd.read_csv(FUTURE_PATH, parse_dates=["date"])
    if "absolute_error" not in backtest.columns:
        backtest["absolute_error"] = (backtest["sales"] - backtest["prediction"]).abs()

    latest_fold = int(backtest["fold"].max())
    first_fold = backtest[backtest["fold"] == backtest["fold"].min()].copy()
    adjustment_fold = backtest[backtest["fold"] == backtest["fold"].min() + 1].copy()
    evaluation_fold = backtest[backtest["fold"] == latest_fold].copy()
    volume_edges = tuple(
        first_fold["prediction"].quantile([1 / 3, 2 / 3]).to_numpy()
    )
    first_fold = add_interval_groups(first_fold, volume_edges, "prediction")
    adjustment_fold = add_interval_groups(adjustment_fold, volume_edges, "prediction")
    evaluation_fold = add_interval_groups(
        evaluation_fold, volume_edges, "prediction"
    )
    historical_calibration = build_calibration_table(first_fold)
    adjustment = adjustment_fold.merge(
        historical_calibration[
            ["horizon_bucket", "volume_bucket", "absolute_margin"]
        ],
        on=["horizon_bucket", "volume_bucket"],
        how="left",
        validate="many_to_one",
    )
    adjustment["margin_ratio"] = (
        adjustment["absolute_error"] / adjustment["absolute_margin"]
    )
    inflation_factors = (
        adjustment.groupby(
            ["horizon_bucket", "volume_bucket"],
            observed=True,
        )["margin_ratio"]
        .quantile(TEMPORAL_BUFFER_COVERAGE, interpolation="higher")
        .clip(lower=1.0)
        .rename("inflation_factor")
        .reset_index()
    )
    historical_calibration = historical_calibration.merge(
        inflation_factors,
        on=["horizon_bucket", "volume_bucket"],
        how="left",
        validate="one_to_one",
    )
    historical_calibration["absolute_margin"] *= historical_calibration["inflation_factor"]
    evaluated = apply_calibration(
        evaluation_fold,
        historical_calibration,
        "prediction",
    )
    overall_evaluation = evaluate_intervals(evaluated)

    evaluation_rows = [
        {"group": "overall", **overall_evaluation},
    ]
    for horizon, group in evaluated.groupby("horizon_bucket"):
        evaluation_rows.append(
            {"group": f"horizon_{horizon}", **evaluate_intervals(group)}
        )
    evaluation = pd.DataFrame(evaluation_rows)

    all_volume_edges = tuple(backtest["prediction"].quantile([1 / 3, 2 / 3]).to_numpy())
    grouped_backtest = add_interval_groups(backtest, all_volume_edges, "prediction")
    final_calibration = build_calibration_table(grouped_backtest)
    final_calibration = final_calibration.merge(
        inflation_factors,
        on=["horizon_bucket", "volume_bucket"],
        how="left",
        validate="one_to_one",
    )
    final_calibration["absolute_margin"] *= final_calibration["inflation_factor"]

    future = future.sort_values(["date", "store", "item"], ignore_index=True)
    future["horizon_day"] = (
        future["date"] - future["date"].min()
    ).dt.days + 1
    grouped_future = add_interval_groups(future, all_volume_edges, "sales")
    future_intervals = apply_calibration(grouped_future, final_calibration, "sales")

    output_columns = [
        "id", "date", "store", "item", "sales", "prediction_lower",
        "prediction_upper", "interval_width", "horizon_day",
        "horizon_bucket", "volume_bucket",
    ]
    future_intervals = future_intervals[output_columns].sort_values("id", ignore_index=True)
    evaluated.to_csv(VALIDATION_INTERVALS_PATH, index=False, date_format="%Y-%m-%d")
    future_intervals.to_csv(FUTURE_INTERVALS_PATH, index=False, date_format="%Y-%m-%d")
    final_calibration.to_csv(CALIBRATION_PATH, index=False)
    evaluation.to_csv(EVALUATION_PATH, index=False)
    SUMMARY_PATH.write_text(
        build_summary(
            evaluation,
            final_calibration,
            future_intervals,
            inflation_factors,
        ),
        encoding="utf-8",
    )

    print("\nIntervalos de previsão")
    print("-" * 68)
    print(f"Cobertura-alvo: {TARGET_COVERAGE:.0%}")
    print(
        "Fator de ajuste temporal: "
        f"{inflation_factors['inflation_factor'].min():.3f} a "
        f"{inflation_factors['inflation_factor'].max():.3f}"
    )
    print(f"Cobertura no fold mais recente: {overall_evaluation['coverage']:.2%}")
    print(f"Largura média: {overall_evaluation['average_width']:.2f} unidades")
    print(f"Previsões futuras intervalares: {len(future_intervals):,}")
    print("-" * 68)
    return future_intervals, evaluation


def build_summary(
    evaluation: pd.DataFrame,
    calibration: pd.DataFrame,
    future: pd.DataFrame,
    inflation_factors: pd.DataFrame,
) -> str:
    """Cria o relatório dos intervalos."""
    overall = evaluation.iloc[0]
    groups = "\n".join(
        f"| {row.horizon_bucket} | {row.volume_bucket} | {int(row.observations):,} | "
        f"{row.absolute_margin:.2f} |"
        for row in calibration.itertuples(index=False)
    )
    return f"""# DemandWise — Intervalos de previsão

Os intervalos de 90% usam calibração conformal por faixa de horizonte e volume.
O primeiro fold define as margens, o segundo adiciona um buffer contra mudança
temporal e o terceiro funciona como teste fora da calibração.

- Cobertura-alvo: **{TARGET_COVERAGE:.0%}**
- Cobertura observada no fold mais recente: **{overall['coverage']:.2%}**
- Fator de ajuste temporal: **{inflation_factors['inflation_factor'].min():.3f} a {inflation_factors['inflation_factor'].max():.3f}**, condicionado por horizonte e volume
- Largura média observada: **{overall['average_width']:.2f} unidades**
- Largura média no horizonte futuro: **{future['interval_width'].mean():.2f} unidades**

| Horizonte | Volume | Observações | Margem absoluta |
| --- | --- | ---: | ---: |
{groups}

Os limites representam incerteza estatística estimada pelos resíduos históricos.
Eles não incorporam choques externos ausentes no dataset, como promoções,
rupturas, mudanças de preço ou eventos não observados.
"""


def main() -> None:
    run_uncertainty_pipeline()


if __name__ == "__main__":
    main()
