"""Métricas reutilizáveis para avaliação de previsões de demanda."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["MAE", "RMSE", "MAPE", "SMAPE"]


def _as_valid_arrays(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Converte e valida os vetores usados nas métricas."""
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true e y_pred precisam ser vetores unidimensionais.")
    if actual.shape != predicted.shape:
        raise ValueError("y_true e y_pred precisam ter o mesmo tamanho.")
    if actual.size == 0:
        raise ValueError("Não é possível avaliar vetores vazios.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("As métricas não aceitam valores ausentes ou infinitos.")

    return actual, predicted


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calcula o erro absoluto médio em unidades vendidas."""
    actual, predicted = _as_valid_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Calcula a raiz do erro quadrático médio."""
    actual, predicted = _as_valid_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mean_absolute_percentage_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Calcula MAPE em percentual, ignorando observações reais iguais a zero."""
    actual, predicted = _as_valid_arrays(y_true, y_pred)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def symmetric_mean_absolute_percentage_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Calcula SMAPE em percentual; pares zero-zero contribuem com erro zero."""
    actual, predicted = _as_valid_arrays(y_true, y_pred)
    denominator = np.abs(actual) + np.abs(predicted)
    percentage_error = np.zeros_like(denominator)
    np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=percentage_error,
        where=denominator != 0,
    )
    return float(np.mean(percentage_error) * 100)


def evaluate_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Retorna todas as métricas padrão do projeto."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "SMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }


def evaluate_models(
    predictions: pd.DataFrame,
    prediction_columns: dict[str, str],
    actual_column: str = "sales",
) -> pd.DataFrame:
    """Compara várias colunas de previsão e ordena os modelos por MAE."""
    if actual_column not in predictions.columns:
        raise ValueError(f"Coluna real não encontrada: {actual_column}")

    results = []
    for model_name, prediction_column in prediction_columns.items():
        if prediction_column not in predictions.columns:
            raise ValueError(f"Coluna de previsão não encontrada: {prediction_column}")
        metrics = evaluate_predictions(
            predictions[actual_column], predictions[prediction_column]
        )
        results.append({"model": model_name, **metrics})

    metrics_frame = pd.DataFrame(results).sort_values(
        ["MAE", "RMSE"], ignore_index=True
    )
    metrics_frame.insert(0, "rank", np.arange(1, len(metrics_frame) + 1))
    return metrics_frame


def evaluate_by_group(
    predictions: pd.DataFrame,
    prediction_column: str,
    group_column: str,
    model_name: str,
    actual_column: str = "sales",
) -> pd.DataFrame:
    """Calcula as métricas de um modelo separadamente por grupo."""
    required = {actual_column, prediction_column, group_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para avaliação por grupo: {sorted(missing)}")

    rows = []
    for group_value, group in predictions.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: group_value,
                "model": model_name,
                "observations": len(group),
                **evaluate_predictions(group[actual_column], group[prediction_column]),
            }
        )
    return pd.DataFrame(rows)
