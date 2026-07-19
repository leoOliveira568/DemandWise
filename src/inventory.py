"""Segmenta séries ABC/XYZ e cria parâmetros indicativos de estoque."""

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
BACKTEST_PATH = PROCESSED_DIR / "backtest_predictions.csv"
FUTURE_INTERVALS_PATH = PROCESSED_DIR / "test_predictions_with_intervals.csv"

SEGMENTS_PATH = REPORTS_DIR / "abc_xyz_segments.csv"
SEGMENT_SUMMARY_PATH = REPORTS_DIR / "abc_xyz_summary.csv"
INVENTORY_POLICY_PATH = REPORTS_DIR / "inventory_policy.csv"
INVENTORY_SUMMARY_PATH = REPORTS_DIR / "inventory_policy_summary.md"

ANALYSIS_DAYS = 365
LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95


def classify_abc(series_sales: pd.Series) -> pd.Series:
    """Classifica por contribuição acumulada: A 80%, B 15%, C 5%."""
    ordered = series_sales.sort_values(ascending=False)
    cumulative = ordered.cumsum() / ordered.sum()
    previous = cumulative.shift(fill_value=0)
    labels = pd.Series(
        np.select(
            [previous < 0.80, previous < 0.95],
            ["A", "B"],
            default="C",
        ),
        index=ordered.index,
        dtype="string",
    )
    return labels.reindex(series_sales.index)


def classify_xyz(
    cv: pd.Series,
    thresholds: tuple[float, float] = (0.20, 0.40),
) -> pd.Series:
    """Classifica estabilidade semanal por coeficiente de variação."""
    low, high = thresholds
    if low >= high:
        raise ValueError("O limite X precisa ser menor que o limite Y.")
    return pd.Series(
        np.select(
            [cv <= low, cv <= high],
            ["X", "Y"],
            default="Z",
        ),
        index=cv.index,
        dtype="string",
    )


def create_segmentation(train: pd.DataFrame) -> pd.DataFrame:
    """Cria a matriz ABC/XYZ por combinação de loja e produto."""
    analysis_start = train["date"].max() - pd.Timedelta(days=ANALYSIS_DAYS - 1)
    recent = train[train["date"] >= analysis_start].copy()
    keys = ["store", "item"]

    totals = recent.groupby(keys)["sales"].sum().rename("annual_sales")
    weekly = (
        recent.set_index("date")
        .groupby(keys)["sales"]
        .resample("W-MON")
        .sum()
        .rename("weekly_sales")
        .reset_index()
    )
    weekly_stats = weekly.groupby(keys)["weekly_sales"].agg(
        weekly_mean="mean", weekly_std="std"
    )
    weekly_stats["coefficient_of_variation"] = (
        weekly_stats["weekly_std"] / weekly_stats["weekly_mean"].replace(0, np.nan)
    ).fillna(0)

    segments = totals.to_frame().join(weekly_stats).reset_index()
    indexed_totals = segments.set_index(keys)["annual_sales"]
    indexed_cv = segments.set_index(keys)["coefficient_of_variation"]
    xyz_thresholds = tuple(indexed_cv.quantile([1 / 3, 2 / 3]).to_numpy())
    segments = segments.set_index(keys)
    segments["abc_class"] = classify_abc(indexed_totals)
    segments["xyz_class"] = classify_xyz(indexed_cv, xyz_thresholds)
    segments["segment"] = segments["abc_class"] + segments["xyz_class"]
    segments["xyz_low_threshold"] = xyz_thresholds[0]
    segments["xyz_high_threshold"] = xyz_thresholds[1]
    segments["demand_share"] = segments["annual_sales"] / segments["annual_sales"].sum()
    return segments.reset_index().sort_values(
        ["abc_class", "xyz_class", "annual_sales"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def priority_from_segment(segment: str) -> str:
    """Traduz o segmento em uma prioridade operacional."""
    if segment in {"AY", "AZ"}:
        return "high"
    if segment in {"AX", "BY", "BZ", "CZ"}:
        return "medium"
    return "standard"


def create_inventory_policy(
    segments: pd.DataFrame,
    future: pd.DataFrame,
    backtest: pd.DataFrame,
    lead_time_days: int = LEAD_TIME_DAYS,
    review_period_days: int = REVIEW_PERIOD_DAYS,
    service_level: float = SERVICE_LEVEL,
) -> pd.DataFrame:
    """Calcula estoque de segurança, ponto de reposição e posição-alvo."""
    if lead_time_days <= 0 or review_period_days < 0:
        raise ValueError("Lead time precisa ser positivo e revisão não negativa.")
    if not 0.5 < service_level < 1:
        raise ValueError("O nível de serviço precisa estar entre 0,5 e 1.")

    keys = ["store", "item"]
    forecast_stats = future.groupby(keys)["sales"].agg(
        forecast_daily_mean="mean",
        forecast_daily_std="std",
        forecast_total="sum",
    )
    residual_stats = backtest.groupby(keys)["residual"].agg(
        residual_std="std",
        residual_bias="mean",
    )
    global_residual_std = float(backtest["residual"].std())
    policy = (
        segments.set_index(keys)
        .join(forecast_stats)
        .join(residual_stats)
        .reset_index()
    )
    policy["residual_std"] = policy["residual_std"].fillna(global_residual_std)
    z_score = NormalDist().inv_cdf(service_level)
    policy["service_level"] = service_level
    policy["lead_time_days"] = lead_time_days
    policy["review_period_days"] = review_period_days
    policy["safety_stock"] = np.ceil(
        z_score * policy["residual_std"] * np.sqrt(lead_time_days)
    ).clip(lower=0)
    policy["reorder_point"] = np.ceil(
        policy["forecast_daily_mean"] * lead_time_days + policy["safety_stock"]
    )
    policy["target_inventory_position"] = np.ceil(
        policy["forecast_daily_mean"] * (lead_time_days + review_period_days)
        + policy["safety_stock"]
    )
    policy["target_coverage_days"] = (
        policy["target_inventory_position"] / policy["forecast_daily_mean"]
    )
    policy["operational_priority"] = policy["segment"].map(priority_from_segment)
    return policy.sort_values(
        ["operational_priority", "forecast_total"],
        ascending=[True, False],
        ignore_index=True,
    )


def run_inventory_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa segmentação e política indicativa de estoque."""
    for path in [TRAIN_PATH, BACKTEST_PATH, FUTURE_INTERVALS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Artefato ausente: {path}")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])
    future = pd.read_csv(FUTURE_INTERVALS_PATH, parse_dates=["date"])

    segments = create_segmentation(train)
    policy = create_inventory_policy(segments, future, backtest)
    summary = (
        segments.groupby("segment", as_index=False)
        .agg(
            series=("segment", "size"),
            annual_sales=("annual_sales", "sum"),
            demand_share=("demand_share", "sum"),
        )
        .sort_values("annual_sales", ascending=False, ignore_index=True)
    )

    segments.to_csv(SEGMENTS_PATH, index=False)
    summary.to_csv(SEGMENT_SUMMARY_PATH, index=False)
    policy.to_csv(INVENTORY_POLICY_PATH, index=False)
    INVENTORY_SUMMARY_PATH.write_text(
        build_summary(segments, summary, policy),
        encoding="utf-8",
    )

    print("\nSegmentação e política indicativa de estoque")
    print("-" * 72)
    print(f"Séries classificadas: {len(segments):,}")
    print(f"Séries de prioridade alta: {(policy['operational_priority'] == 'high').sum():,}")
    print(f"Nível de serviço do cenário: {SERVICE_LEVEL:.0%}")
    print("-" * 72)
    return segments, policy


def build_summary(
    segments: pd.DataFrame,
    summary: pd.DataFrame,
    policy: pd.DataFrame,
) -> str:
    """Cria o relatório executivo de segmentação e parâmetros de estoque."""
    segment_rows = "\n".join(
        f"| {row.segment} | {int(row.series)} | {row.demand_share:.2%} |"
        for row in summary.itertuples(index=False)
    )
    high = policy[policy["operational_priority"] == "high"]
    return f"""# DemandWise — Segmentação e política de estoque

## Segmentação ABC/XYZ

- ABC: contribuição acumulada de vendas nos últimos {ANALYSIS_DAYS} dias.
- XYZ: tercis do coeficiente de variação das vendas semanais, adequados à
  dispersão observada no dataset.
- Séries classificadas: **{len(segments):,}**

| Segmento | Séries | Participação da demanda |
| --- | ---: | ---: |
{segment_rows}

## Cenário de estoque

- Lead time assumido: **{LEAD_TIME_DAYS} dias**
- Período de revisão: **{REVIEW_PERIOD_DAYS} dias**
- Nível de serviço: **{SERVICE_LEVEL:.0%}**
- Séries de prioridade alta: **{len(high):,}**

O arquivo `inventory_policy.csv` apresenta estoque de segurança, ponto de
reposição e posição-alvo por loja e produto. Esses valores são parâmetros de
cenário, não ordens de compra. A quantidade a comprar exige saldo disponível,
pedidos em trânsito, lote mínimo, lead time real e restrições comerciais, que
não existem no dataset.
"""


def main() -> None:
    run_inventory_pipeline()


if __name__ == "__main__":
    main()
