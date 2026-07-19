"""Exporta agregações compactas para o dashboard web do DemandWise."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DATA_PATH = PROJECT_ROOT / "dashboard" / "app" / "dashboard-data.json"
FORECAST_DATA_PATH = PROJECT_ROOT / "dashboard" / "public" / "forecast-data.json"


def records(frame: pd.DataFrame) -> list[dict]:
    """Converte um DataFrame em registros JSON serializáveis."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_dashboard_data() -> dict:
    """Lê os artefatos do projeto e prepara as séries do dashboard."""
    train = pd.read_csv(
        PROCESSED_DIR / "train_processed.csv",
        usecols=["date", "store", "item", "sales", "year", "month", "day_of_week"],
        parse_dates=["date"],
    )
    validation = pd.read_csv(
        PROCESSED_DIR / "ml_validation_predictions.csv",
        parse_dates=["date"],
    )
    future = pd.read_csv(
        PROCESSED_DIR / "test_predictions_with_intervals.csv",
        parse_dates=["date"],
    )
    inventory_policy = pd.read_csv(REPORTS_DIR / "inventory_policy.csv")
    model_metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    feature_importance = pd.read_csv(REPORTS_DIR / "ml_feature_importance.csv")

    monthly_history = (
        train.set_index("date")["sales"]
        .resample("MS")
        .sum()
        .rename("sales")
        .reset_index()
    )
    yearly = train.groupby("year", as_index=False)["sales"].sum()
    weekday = (
        train.groupby("date", as_index=False)["sales"]
        .sum()
        .assign(day_of_week=lambda frame: frame["date"].dt.dayofweek)
        .groupby("day_of_week", as_index=False)["sales"]
        .mean()
    )
    weekday["label"] = weekday["day_of_week"].map(
        {
            0: "Seg",
            1: "Ter",
            2: "Qua",
            3: "Qui",
            4: "Sex",
            5: "Sáb",
            6: "Dom",
        }
    )

    validation_daily = validation.groupby("date", as_index=False)[
        ["sales", "prediction_random_forest"]
    ].sum()
    future_daily = future.groupby("date", as_index=False)["sales"].sum()
    future_monthly = (
        future.assign(month=future["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["sales"]
        .sum()
    )
    future_by_store = (
        future.groupby("store", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False, ignore_index=True)
    )
    future_by_item = (
        future.groupby("item", as_index=False)["sales"]
        .sum()
        .nlargest(10, "sales")
        .sort_values("sales", ascending=False, ignore_index=True)
    )

    random_forest_importance = (
        feature_importance[feature_importance["model"] == "Random Forest"]
        .nlargest(10, "importance")
        .reset_index(drop=True)
    )
    best_model = model_metrics.iloc[0]
    best_baseline = model_metrics[model_metrics["model_type"] == "Baseline"].iloc[0]
    improvement = (best_baseline["MAE"] - best_model["MAE"]) / best_baseline["MAE"] * 100
    peak_future_day = future_daily.loc[future_daily["sales"].idxmax()]
    historical_growth = (yearly.iloc[-1]["sales"] / yearly.iloc[0]["sales"] - 1) * 100
    weekend_mean = weekday.loc[weekday["day_of_week"].isin([5, 6]), "sales"].mean()
    workday_mean = weekday.loc[~weekday["day_of_week"].isin([5, 6]), "sales"].mean()

    forecast_dates = sorted(future["date"].dt.strftime("%Y-%m-%d").unique())
    date_index = {date: index for index, date in enumerate(forecast_dates)}
    compact_future = future.assign(
        date_index=future["date"].dt.strftime("%Y-%m-%d").map(date_index)
    )[[
        "date_index", "store", "item", "sales", "prediction_lower",
        "prediction_upper",
    ]].copy()
    for column in ["sales", "prediction_lower", "prediction_upper"]:
        compact_future[column] = compact_future[column].round(3)

    compact_policies = inventory_policy[[
        "store", "item", "segment", "residual_std", "forecast_daily_mean",
    ]].copy()
    compact_policies["residual_std"] = compact_policies["residual_std"].round(4)
    compact_policies["forecast_daily_mean"] = compact_policies[
        "forecast_daily_mean"
    ].round(4)

    return {
        "overview": {
            "historicalSales": int(train["sales"].sum()),
            "historicalGrowthPct": round(float(historical_growth), 1),
            "forecastSales": round(float(future["sales"].sum())),
            "forecastDailyAverage": round(float(future_daily["sales"].mean())),
            "forecastStart": future["date"].min().strftime("%Y-%m-%d"),
            "forecastEnd": future["date"].max().strftime("%Y-%m-%d"),
            "bestModel": str(best_model["model"]),
            "bestMae": round(float(best_model["MAE"]), 3),
            "bestSmape": round(float(best_model["SMAPE"]), 2),
            "improvementPct": round(float(improvement), 1),
            "peakDate": peak_future_day["date"].strftime("%Y-%m-%d"),
            "peakSales": round(float(peak_future_day["sales"])),
            "weekendUpliftPct": round(float((weekend_mean / workday_mean - 1) * 100), 1),
        },
        "monthlyHistory": records(monthly_history),
        "yearlyHistory": records(yearly),
        "weekdayProfile": records(weekday[["label", "sales"]]),
        "modelMetrics": records(
            model_metrics[["rank", "model_type", "model", "MAE", "RMSE", "MAPE", "SMAPE"]]
        ),
        "validationDaily": records(validation_daily),
        "futureDaily": records(future_daily),
        "futureMonthly": records(future_monthly),
        "futureByStore": records(future_by_store),
        "futureTopItems": records(future_by_item),
        "featureImportance": records(random_forest_importance[["feature", "importance"]]),
        "forecastDates": forecast_dates,
        "forecastRows": compact_future.to_numpy().tolist(),
        "inventoryPolicies": compact_policies.to_numpy().tolist(),
    }


def main() -> None:
    """Gera o arquivo de dados estáticos consumido pelo dashboard."""
    DASHBOARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dashboard_payload = build_dashboard_data()
    forecast_payload = {
        "forecastDates": dashboard_payload.pop("forecastDates"),
        "forecastRows": dashboard_payload.pop("forecastRows"),
        "inventoryPolicies": dashboard_payload.pop("inventoryPolicies"),
    }
    DASHBOARD_DATA_PATH.write_text(
        json.dumps(dashboard_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    FORECAST_DATA_PATH.write_text(
        json.dumps(forecast_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Dados do dashboard salvos em: {DASHBOARD_DATA_PATH}")
    print(f"Dados granulares salvos em: {FORECAST_DATA_PATH}")


if __name__ == "__main__":
    main()
