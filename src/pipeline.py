"""Orquestra os estágios reproduzíveis do projeto DemandWise."""

import argparse
from collections.abc import Callable

from src import export_dashboard_data, features, make_dataset
from src.backtesting import run_backtesting
from src.inventory import run_inventory_pipeline
from src.horizon_analysis import run_horizon_analysis
from src.monitoring import run_monitoring
from src.optimize_model import run_optimization
from src.predict import run_final_forecast
from src.refresh_random_forest import run_refresh
from src.train_ml_models import run_ml_evaluation
from src.train_model import run_baseline_evaluation
from src.training_window import run_window_comparison
from src.uncertainty import run_uncertainty_pipeline


STAGES: dict[str, Callable[[], object]] = {
    "data": make_dataset.main,
    "features": features.main,
    "optimize": run_optimization,
    "baselines": run_baseline_evaluation,
    "models": run_ml_evaluation,
    "forecast": run_final_forecast,
    "backtest": run_backtesting,
    "window": run_window_comparison,
    "horizon": run_horizon_analysis,
    "refresh": run_refresh,
    "uncertainty": run_uncertainty_pipeline,
    "inventory": run_inventory_pipeline,
    "monitor": run_monitoring,
    "dashboard": export_dashboard_data.main,
}

ALL_ORDER = [
    "data", "features", "optimize", "window", "baselines", "models", "backtest",
    "forecast", "horizon", "uncertainty", "inventory", "monitor",
    "dashboard",
]
IMPROVEMENT_ORDER = [
    "optimize", "window", "models", "backtest", "forecast",
    "horizon", "uncertainty", "inventory", "monitor", "dashboard",
]


def run_stages(stage_names: list[str]) -> None:
    """Executa uma sequência de estágios e interrompe em qualquer falha."""
    for index, stage_name in enumerate(stage_names, start=1):
        print("\n" + "=" * 80)
        print(f"DemandWise pipeline [{index}/{len(stage_names)}]: {stage_name}")
        print("=" * 80)
        STAGES[stage_name]()


def parse_args() -> argparse.Namespace:
    """Define a interface de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "improvements", "all"],
        default="all",
        help="Estágio isolado ou sequência completa.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "all":
        selected = ALL_ORDER
    elif args.stage == "improvements":
        selected = IMPROVEMENT_ORDER
    else:
        selected = [args.stage]
    run_stages(selected)


if __name__ == "__main__":
    main()
