"""DySANOS 1.1 model, training, and public-Dataset materialization.

Training symbols are loaded lazily so public-Dataset materialization remains
usable from the base installation without importing the optional Torch stack.
"""

from importlib import import_module
from typing import Any

from sanos.models.dysanos.artifacts import (
    ExperimentBuildResult,
    ExperimentPlan,
    ExperimentValidationSummary,
    build_experiment,
    open_experiment,
    plan_experiment,
    validate_experiment,
)
from sanos.models.dysanos.config import (
    DailyFrontGeometryConfig,
    DailyFrontMarketConfig,
    DirectModelConfig,
    DirectTrainingConfig,
    ExperimentBuildOptions,
    ExperimentConfig,
    Hdf5StorageConfig,
    ObjectiveConfig,
    SourceDatasetConfig,
    SplitConfig,
)
from sanos.models.dysanos.data import (
    DailyFrontDay,
    EligibilityAudit,
    daily_quote_objective,
    exact_weekly_pairs,
    project_daily_front,
    recompute_market_fingerprint,
)
from sanos.models.dysanos.hdf5 import (
    DailyFrontExperiment,
    ExperimentValidationError,
)

_LAZY_TRAINING_EXPORTS = {
    "DirectFitResult",
    "NativeFitResult",
    "TrainingEvaluation",
    "checkpoint_schema",
    "convergence_audit",
    "evaluate_direct_model",
    "fit_direct200_pipeline",
    "fit_joint_direct",
    "fit_native_daily_states",
    "load_checkpoint",
    "save_checkpoint",
}

def __getattr__(name: str) -> Any:
    if name in _LAZY_TRAINING_EXPORTS:
        module = import_module("sanos.models.dysanos.training")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "DailyFrontDay",
    "DailyFrontExperiment",
    "DailyFrontGeometryConfig",
    "DailyFrontMarketConfig",
    "DirectModelConfig",
    "DirectTrainingConfig",
    "DirectFitResult",
    "EligibilityAudit",
    "daily_quote_objective",
    "ExperimentBuildOptions",
    "ExperimentBuildResult",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentValidationError",
    "ExperimentValidationSummary",
    "Hdf5StorageConfig",
    "NativeFitResult",
    "ObjectiveConfig",
    "SourceDatasetConfig",
    "SplitConfig",
    "TrainingEvaluation",
    "build_experiment",
    "checkpoint_schema",
    "convergence_audit",
    "evaluate_direct_model",
    "exact_weekly_pairs",
    "fit_direct200_pipeline",
    "fit_joint_direct",
    "fit_native_daily_states",
    "load_checkpoint",
    "open_experiment",
    "plan_experiment",
    "project_daily_front",
    "recompute_market_fingerprint",
    "save_checkpoint",
    "validate_experiment",
]
