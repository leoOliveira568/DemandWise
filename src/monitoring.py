"""Simula o monitoramento de desempenho e qualidade do DemandWise."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.evaluate_model import evaluate_predictions
except ModuleNotFoundError:
    from evaluate_model import evaluate_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

BACKTEST_PATH = PROCESSED_DIR / "backtest_predictions.csv"
INTERVAL_VALIDATION_PATH = PROCESSED_DIR / "interval_validation_predictions.csv"
FUTURE_PATH = PROCESSED_DIR / "test_predictions_with_intervals.csv"

WEEKLY_PATH = REPORTS_DIR / "monitoring_weekly.csv"
ALERTS_PATH = REPORTS_DIR / "monitoring_alerts.csv"
QUALITY_PATH = REPORTS_DIR / "data_quality_checks.csv"
SUMMARY_PATH = REPORTS_DIR / "monitoring_summary.md"


def weekly_performance(validation: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por semana operacional."""
    monitored = validation.copy()
    monitored["week_start"] = monitored["date"].dt.to_period("W-MON").dt.start_time
    rows = []
    for week, group in monitored.groupby("week_start"):
        metrics = evaluate_predictions(group["sales"], group["prediction"])
        bias = float((group["prediction"] - group["sales"]).mean())
        bias_pct = float(
            (group["prediction"].sum() - group["sales"].sum())
            / group["sales"].sum()
            * 100
        )
        coverage = (
            group["sales"].between(
                group["prediction_lower"], group["prediction_upper"]
            ).mean()
            if {"prediction_lower", "prediction_upper"}.issubset(group.columns)
            else np.nan
        )
        rows.append(
            {
                "week_start": week,
                "observations": len(group),
                **metrics,
                "bias": bias,
                "bias_pct": bias_pct,
                "interval_coverage": coverage,
            }
        )
    return pd.DataFrame(rows).sort_values("week_start", ignore_index=True)


def create_alerts(weekly: pd.DataFrame) -> pd.DataFrame:
    """Gera alertas transparentes por limites operacionais."""
    alerts = []
    mae_threshold = weekly["MAE"].mean() + 1.5 * weekly["MAE"].std(ddof=1)
    for row in weekly.itertuples(index=False):
        if row.MAE > mae_threshold:
            alerts.append(
                {
                    "week_start": row.week_start,
                    "severity": "warning",
                    "indicator": "MAE",
                    "value": row.MAE,
                    "threshold": mae_threshold,
                    "message": "Erro semanal acima do limite histórico.",
                }
            )
        if abs(row.bias_pct) > 10:
            alerts.append(
                {
                    "week_start": row.week_start,
                    "severity": "warning",
                    "indicator": "bias_pct",
                    "value": row.bias_pct,
                    "threshold": 10,
                    "message": "Viés absoluto acima de 10%.",
                }
            )
        if pd.notna(row.interval_coverage) and row.interval_coverage < 0.85:
            alerts.append(
                {
                    "week_start": row.week_start,
                    "severity": "critical",
                    "indicator": "interval_coverage",
                    "value": row.interval_coverage,
                    "threshold": 0.85,
                    "message": "Cobertura semanal abaixo de 85%.",
                }
            )
    return pd.DataFrame(
        alerts,
        columns=["week_start", "severity", "indicator", "value", "threshold", "message"],
    )


def data_quality_checks(future: pd.DataFrame) -> pd.DataFrame:
    """Avalia integridade do lote futuro que seria enviado à produção."""
    checks = [
        ("rows", len(future) == 45_000, len(future), 45_000),
        ("unique_ids", future["id"].nunique() == len(future), future["id"].nunique(), len(future)),
        ("forecast_days", future["date"].nunique() == 90, future["date"].nunique(), 90),
        ("series", future[["store", "item"]].drop_duplicates().shape[0] == 500, future[["store", "item"]].drop_duplicates().shape[0], 500),
        ("missing_predictions", future[["sales", "prediction_lower", "prediction_upper"]].isna().sum().sum() == 0, int(future[["sales", "prediction_lower", "prediction_upper"]].isna().sum().sum()), 0),
        ("negative_predictions", (future[["sales", "prediction_lower", "prediction_upper"]] < 0).sum().sum() == 0, int((future[["sales", "prediction_lower", "prediction_upper"]] < 0).sum().sum()), 0),
        ("ordered_intervals", (future["prediction_lower"] <= future["sales"]).all() and (future["sales"] <= future["prediction_upper"]).all(), int(((future["prediction_lower"] <= future["sales"]) & (future["sales"] <= future["prediction_upper"])).sum()), len(future)),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "observed": observed, "expected": expected}
            for name, passed, observed, expected in checks
        ]
    )


def run_monitoring() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera métricas semanais, alertas e testes de qualidade."""
    for path in [BACKTEST_PATH, INTERVAL_VALIDATION_PATH, FUTURE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Artefato ausente: {path}")
    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])
    interval_validation = pd.read_csv(INTERVAL_VALIDATION_PATH, parse_dates=["date"])
    future = pd.read_csv(FUTURE_PATH, parse_dates=["date"])

    latest_fold = int(backtest["fold"].max())
    latest = backtest[backtest["fold"] == latest_fold].copy()
    interval_columns = interval_validation[
        ["date", "store", "item", "prediction_lower", "prediction_upper"]
    ]
    latest = latest.merge(
        interval_columns,
        on=["date", "store", "item"],
        how="left",
        validate="one_to_one",
    )
    weekly = weekly_performance(latest)
    alerts = create_alerts(weekly)
    quality = data_quality_checks(future)

    weekly.to_csv(WEEKLY_PATH, index=False, date_format="%Y-%m-%d")
    alerts.to_csv(ALERTS_PATH, index=False, date_format="%Y-%m-%d")
    quality.to_csv(QUALITY_PATH, index=False)
    SUMMARY_PATH.write_text(
        build_summary(weekly, alerts, quality),
        encoding="utf-8",
    )

    print("\nMonitoramento simulado")
    print("-" * 64)
    print(f"Semanas avaliadas: {len(weekly)}")
    print(f"Alertas gerados: {len(alerts)}")
    print(f"Testes de qualidade aprovados: {(quality['status'] == 'pass').sum()}/{len(quality)}")
    print("-" * 64)
    return weekly, alerts


def build_summary(
    weekly: pd.DataFrame,
    alerts: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    """Cria o relatório do monitoramento."""
    return f"""# DemandWise — Monitoramento simulado

## Desempenho

- Semanas avaliadas: **{len(weekly)}**
- MAE semanal médio: **{weekly['MAE'].mean():.3f}**
- SMAPE semanal médio: **{weekly['SMAPE'].mean():.2f}%**
- Viés percentual médio: **{weekly['bias_pct'].mean():.2f}%**
- Cobertura média dos intervalos: **{weekly['interval_coverage'].mean():.2%}**

## Alertas

- Alertas gerados: **{len(alerts)}**
- Alertas críticos: **{(alerts['severity'] == 'critical').sum() if not alerts.empty else 0}**

## Qualidade do lote futuro

- Testes aprovados: **{(quality['status'] == 'pass').sum()}/{len(quality)}**

Em produção, essas regras devem rodar após a chegada das vendas realizadas e
acionar investigação quando erro, viés, cobertura ou qualidade de dados
ultrapassarem os limites definidos.
"""


def main() -> None:
    run_monitoring()


if __name__ == "__main__":
    main()
