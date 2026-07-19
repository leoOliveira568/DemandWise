"""Testes unitários rápidos para as regras críticas do DemandWise."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.evaluate_model import evaluate_predictions
from src.features import add_lag_features, add_rolling_features
from src.inventory import classify_abc, classify_xyz, create_inventory_policy
from src.train_ml_models import FEATURE_COLUMNS, recursive_forecast
from src.train_model import temporal_split
from src.optimize_model import tuning_split
from src.uncertainty import (
    add_interval_groups,
    apply_calibration,
    build_calibration_table,
    evaluate_intervals,
)


def test_metrics_known_values() -> None:
    metrics = evaluate_predictions([0, 10, 20], [0, 8, 24])
    assert np.isclose(metrics["MAE"], 2.0)
    assert np.isclose(metrics["RMSE"], np.sqrt(20 / 3))
    assert np.isclose(metrics["MAPE"], 20.0)
    assert np.isfinite(metrics["SMAPE"])


def test_temporal_split_has_no_overlap() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {"date": dates, "store": 1, "item": 1, "sales": np.arange(120)}
    )
    train, validation, start = temporal_split(data, validation_days=30)
    assert train["date"].max() < start == validation["date"].min()
    assert validation["date"].nunique() == 30


def test_development_holdout_precedes_official_validation() -> None:
    dates = pd.date_range("2013-01-01", "2017-12-31", freq="D")
    data = pd.DataFrame(
        {"date": dates, "store": 1, "item": 1, "sales": np.arange(len(dates))}
    )
    _, development, _ = tuning_split(data)
    _, official, _ = temporal_split(data, validation_days=90)
    assert development["date"].max() < official["date"].min()
    assert development["date"].nunique() == 60


def test_lags_and_rolling_never_use_current_sale() -> None:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "store": 1,
            "item": 1,
            "sales": np.arange(1, 36, dtype=float),
        }
    )
    featured = add_rolling_features(add_lag_features(data))
    row = featured.iloc[28]
    assert row["lag_7"] == data.iloc[21]["sales"]
    assert row["lag_28"] == data.iloc[0]["sales"]
    assert np.isclose(row["rolling_mean_7"], data.iloc[21:28]["sales"].mean())
    assert row["rolling_mean_7"] != data.iloc[22:29]["sales"].mean()


def test_recursive_forecast_does_not_require_future_sales() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    history = pd.DataFrame(
        {"date": dates[:28], "store": 1, "item": 1, "sales": np.arange(10, 38)}
    )
    future = pd.DataFrame(
        {"date": dates[28:], "store": 1, "item": 1, "sales": [-999, -999]}
    )
    model = DummyRegressor(strategy="constant", constant=25)
    model.fit(pd.DataFrame(np.zeros((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS), [25, 25])
    first = recursive_forecast(model, history, future, "prediction")
    future["sales"] = [999_999, 999_999]
    second = recursive_forecast(model, history, future, "prediction")
    assert first["prediction"].equals(second["prediction"])
    assert len(first) == 2


def test_conformal_interval_contains_expected_share() -> None:
    calibration = pd.DataFrame(
        {
            "horizon_day": np.tile(np.arange(1, 11), 30),
            "prediction": np.repeat([10, 30, 60], 100),
            "absolute_error": np.tile(np.arange(1, 11), 30),
        }
    )
    edges = (20.0, 45.0)
    grouped = add_interval_groups(calibration, edges, "prediction")
    table = build_calibration_table(grouped, coverage=0.9)
    evaluated = grouped.assign(sales=grouped["prediction"] + grouped["absolute_error"])
    evaluated = apply_calibration(evaluated, table, "prediction")
    result = evaluate_intervals(evaluated)
    assert result["coverage"] >= 0.9
    assert (evaluated["prediction_lower"] >= 0).all()


def test_abc_xyz_and_inventory_policy() -> None:
    index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 2), (1, 3)], names=["store", "item"]
    )
    sales = pd.Series([80.0, 15.0, 5.0], index=index)
    abc = classify_abc(sales)
    xyz = classify_xyz(pd.Series([0.1, 0.3, 0.6], index=index))
    assert abc.tolist() == ["A", "B", "C"]
    assert xyz.tolist() == ["X", "Y", "Z"]

    segments = pd.DataFrame(
        {
            "store": [1], "item": [1], "annual_sales": [3650],
            "weekly_mean": [70], "weekly_std": [7],
            "coefficient_of_variation": [0.1], "abc_class": ["A"],
            "xyz_class": ["X"], "segment": ["AX"], "demand_share": [1.0],
        }
    )
    future = pd.DataFrame({"store": [1] * 2, "item": [1] * 2, "sales": [10, 12]})
    backtest = pd.DataFrame({"store": [1] * 3, "item": [1] * 3, "residual": [1, -1, 2]})
    policy = create_inventory_policy(segments, future, backtest)
    assert policy.loc[0, "reorder_point"] >= 7 * policy.loc[0, "forecast_daily_mean"]
    assert policy.loc[0, "target_inventory_position"] >= policy.loc[0, "reorder_point"]
