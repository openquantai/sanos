"""Strict configuration for disposable SANOS Experiment materializations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from cdxcore.version import version

from sanos.artifacts import ArtifactPointer, StorageClass

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REFERENCE_MODEL_DTES = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
)


def _require_exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(f"{label} fields changed: missing={missing}, unknown={unknown}")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@version("1.0", dependencies=[ArtifactPointer])
@dataclass(frozen=True, slots=True)
class SourceDatasetConfig:
    """Exact immutable Dataset selected as the Experiment source."""

    namespace: str
    artifact_id: str
    storage_class: Literal["dataset"] = "dataset"

    def __post_init__(self) -> None:
        pointer = ArtifactPointer(StorageClass.DATASET, self.namespace, self.artifact_id)
        object.__setattr__(self, "namespace", pointer.namespace)
        object.__setattr__(self, "artifact_id", pointer.artifact_id)
        if self.storage_class != "dataset":
            raise ValueError("An Experiment source must be a Dataset")

    @property
    def pointer(self) -> ArtifactPointer:
        return ArtifactPointer(StorageClass.DATASET, self.namespace, self.artifact_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceDatasetConfig:
        _require_exact(value, {"storage_class", "namespace", "artifact_id"}, "source_dataset")
        if not all(isinstance(value[name], str) for name in value):
            raise ValueError("source_dataset fields must be strings")
        if value["storage_class"] != "dataset":
            raise ValueError("source_dataset.storage_class must be 'dataset'")
        return cls(
            storage_class="dataset",
            namespace=cast(str, value["namespace"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class DailyFrontMarketConfig:
    """Exact recovered daily-front population contract."""

    smoothness: float = 0.25
    max_num_options: int = 100_000
    excluded_business_dtes: tuple[float, ...] = (0.2,)
    overnight_variance_fraction: float = 0.2

    def __post_init__(self) -> None:
        smoothness = _finite(self.smoothness, "smoothness")
        if smoothness <= 0.01:
            raise ValueError("smoothness must be finite and exceed 0.01")
        _positive_int(self.max_num_options, "max_num_options")
        excluded = np.asarray(self.excluded_business_dtes, dtype=np.float64)
        if (
            excluded.ndim != 1
            or np.any(~np.isfinite(excluded))
            or np.any(excluded < 0.0)
            or (len(excluded) > 1 and np.any(np.diff(excluded) <= 1.0e-4))
        ):
            raise ValueError(
                "excluded_business_dtes must be finite, non-negative, ordered, and distinct"
            )
        overnight = _finite(self.overnight_variance_fraction, "overnight_variance_fraction")
        if not 0.0 <= overnight <= 1.0:
            raise ValueError("overnight_variance_fraction must lie in [0,1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DailyFrontMarketConfig:
        fields = {
            "smoothness",
            "max_num_options",
            "excluded_business_dtes",
            "overnight_variance_fraction",
        }
        _require_exact(value, fields, "market")
        excluded = value["excluded_business_dtes"]
        if not isinstance(excluded, list):
            raise ValueError("market.excluded_business_dtes must be a list")
        return cls(
            smoothness=_finite(value["smoothness"], "market.smoothness"),
            max_num_options=_positive_int(value["max_num_options"], "market.max_num_options"),
            excluded_business_dtes=tuple(
                _finite(item, "market.excluded_business_dtes[]") for item in excluded
            ),
            overnight_variance_fraction=_finite(
                value["overnight_variance_fraction"],
                "market.overnight_variance_fraction",
            ),
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class DailyFrontGeometryConfig:
    """Model and population geometry recovered from the reference run."""

    model_business_dtes: tuple[float, ...] = _REFERENCE_MODEL_DTES
    nms_range: tuple[float, float] = (-3.0, 2.0)
    num_inner_strikes: int = 200
    min_dlvsq: float = 0.0001
    max_dlvsq: float = 9.0
    min_forward_vol: float = 0.01
    max_forward_vol: float = 1.0
    min_eta: float = 0.01
    max_eta: float = 0.5
    lower_boundary_scale: float = 0.5
    upper_boundary_scale: float = 1.5
    minimum_log_spacing_fraction: float = 0.05
    dlv_cap_barrier_log_width: float = 0.1
    dlv_cap_barrier_log_std: float = 0.35

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in self.model_business_dtes):
            raise ValueError("model_business_dtes cannot contain booleans")
        nodes = np.asarray(self.model_business_dtes, dtype=np.float64)
        if (
            nodes.ndim != 1
            or len(nodes) < 2
            or np.any(~np.isfinite(nodes))
            or np.any(nodes <= 0.0)
            or np.any(np.diff(nodes) <= 0.0)
        ):
            raise ValueError("model_business_dtes must be finite, positive, and increasing")
        if (
            len(self.nms_range) != 2
            or not all(
                not isinstance(item, bool) and math.isfinite(item) for item in self.nms_range
            )
            or not self.nms_range[0] < 0.0 < self.nms_range[1]
        ):
            raise ValueError("nms_range must be finite and straddle zero")
        if self.num_inner_strikes < 5:
            raise ValueError("num_inner_strikes must be at least five")
        pairs = (
            (self.min_dlvsq, self.max_dlvsq, "DLV square"),
            (self.min_forward_vol, self.max_forward_vol, "forward volatility"),
            (self.min_eta, self.max_eta, "eta"),
        )
        for lower, upper, label in pairs:
            lower_value = _finite(lower, f"minimum {label}")
            upper_value = _finite(upper, f"maximum {label}")
            if not 0.0 < lower_value < upper_value:
                raise ValueError(f"{label} bounds must satisfy 0 < lower < upper")
        lower_scale = _finite(self.lower_boundary_scale, "lower_boundary_scale")
        upper_scale = _finite(self.upper_boundary_scale, "upper_boundary_scale")
        if not 0.0 < lower_scale < 1.0 < upper_scale:
            raise ValueError("boundary scales must straddle one")
        for name in (
            "minimum_log_spacing_fraction",
            "dlv_cap_barrier_log_width",
            "dlv_cap_barrier_log_std",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DailyFrontGeometryConfig:
        fields = {
            "model_business_dtes",
            "nms_range",
            "num_inner_strikes",
            "min_dlvsq",
            "max_dlvsq",
            "min_forward_vol",
            "max_forward_vol",
            "min_eta",
            "max_eta",
            "lower_boundary_scale",
            "upper_boundary_scale",
            "minimum_log_spacing_fraction",
            "dlv_cap_barrier_log_width",
            "dlv_cap_barrier_log_std",
        }
        _require_exact(value, fields, "geometry")
        nodes = value["model_business_dtes"]
        nms = value["nms_range"]
        if not isinstance(nodes, list) or not isinstance(nms, list) or len(nms) != 2:
            raise ValueError("geometry model_business_dtes/nms_range must be lists")
        kwargs: dict[str, Any] = {
            name: _finite(value[name], f"geometry.{name}")
            for name in fields
            if name not in {"model_business_dtes", "nms_range", "num_inner_strikes"}
        }
        return cls(
            model_business_dtes=tuple(
                _finite(item, "geometry.model_business_dtes[]") for item in nodes
            ),
            nms_range=(
                _finite(nms[0], "geometry.nms_range[0]"),
                _finite(nms[1], "geometry.nms_range[1]"),
            ),
            num_inner_strikes=_positive_int(
                value["num_inner_strikes"], "geometry.num_inner_strikes"
            ),
            **kwargs,
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    fit_weight: float = 0.05
    date_weighting: Literal["mean_quotes_then_mean_dates"] = "mean_quotes_then_mean_dates"

    def __post_init__(self) -> None:
        if _finite(self.fit_weight, "fit_weight") < 0.0:
            raise ValueError("fit_weight must be finite and non-negative")
        if self.date_weighting != "mean_quotes_then_mean_dates":
            raise ValueError("Unsupported date_weighting")


@version("1.0")
@dataclass(frozen=True, slots=True)
class SplitConfig:
    mode: Literal["all_history_no_holdout"] = "all_history_no_holdout"
    diagnostic_boundary: str = "2025-01-01"

    def __post_init__(self) -> None:
        from datetime import date

        if self.mode != "all_history_no_holdout":
            raise ValueError("The reference Experiment uses all_history_no_holdout")
        if not isinstance(self.diagnostic_boundary, str):
            raise ValueError("diagnostic_boundary must be an ISO date")
        try:
            date.fromisoformat(self.diagnostic_boundary)
        except ValueError as exc:
            raise ValueError("diagnostic_boundary must be an ISO date") from exc


@version("1.0")
@dataclass(frozen=True, slots=True)
class Hdf5StorageConfig:
    float_dtype: Literal["float64"] = "float64"
    padding: Literal["global_max_with_masks_v1"] = "global_max_with_masks_v1"
    compression: Literal["gzip"] = "gzip"
    compression_level: int = 4
    shuffle: bool = True
    fletcher32: bool = True

    def __post_init__(self) -> None:
        if self.float_dtype != "float64" or self.padding != "global_max_with_masks_v1":
            raise ValueError("Unsupported Experiment dtype or padding contract")
        if (
            self.compression != "gzip"
            or isinstance(self.compression_level, bool)
            or not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("gzip compression_level must lie in [0,9]")
        if not isinstance(self.shuffle, bool) or not isinstance(self.fletcher32, bool):
            raise TypeError("shuffle and fletcher32 must be booleans")


@version("1.0")
@dataclass(frozen=True, slots=True)
class DirectModelConfig:
    state_size: int = 200
    decoded_state_size: int = 3636
    runtime_precision_bits: Literal[32] = 32
    minimum_mean_reversion_per_week: float = 0.001
    innovation_std_floor: float = 0.0001
    decoder_activation: Literal["Identity"] = "Identity"
    decoder_depth: int = 0
    decoder_width: int = 16
    decoder_use_bias: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.state_size, "model.state_size")
        _positive_int(self.decoded_state_size, "model.decoded_state_size")
        if isinstance(self.runtime_precision_bits, bool) or self.runtime_precision_bits != 32:
            raise ValueError("Reference runtime_precision_bits must be 32")
        if (
            self.decoder_activation != "Identity"
            or isinstance(self.decoder_depth, bool)
            or not isinstance(self.decoder_depth, int)
            or self.decoder_depth != 0
        ):
            raise ValueError("Reference Direct decoder must remain affine")
        _positive_int(self.decoder_width, "model.decoder_width")
        if not isinstance(self.decoder_use_bias, bool):
            raise TypeError("model.decoder_use_bias must be boolean")
        mean_reversion = _finite(
            self.minimum_mean_reversion_per_week,
            "model.minimum_mean_reversion_per_week",
        )
        innovation_floor = _finite(
            self.innovation_std_floor,
            "model.innovation_std_floor",
        )
        if mean_reversion <= 0.0 or innovation_floor <= 0.0:
            raise ValueError("Dynamics floors must be positive")


@version("1.0")
@dataclass(frozen=True, slots=True)
class DirectTrainingConfig:
    num_epochs: int = 300
    day_batch_size: int = 128
    state_learning_rate: float = 1.0e-4
    decoder_learning_rate: float = 3.0e-5
    dynamics_learning_rate: float = 3.0e-4
    dynamics_weight: float = 0.1
    gauge_weight: float = 0.1
    cap_penalty_weight: float = 1.0
    dynamics_steps_per_epoch: int = 1
    warmup_epochs: int = 20
    minimum_learning_rate_fraction: float = 0.1
    dynamics_ramp_epochs: int = 50
    gradient_clip: float = 10.0
    seed: int = 9281
    checkpoint_every: int = 5
    report_every: int = 5
    convergence_window: int = 30
    convergence_relative_tolerance: float = 0.002
    maximum_quote_containment_ratio_drop: float = 0.02
    compile_sanos_solver: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_epochs",
            "day_batch_size",
            "dynamics_steps_per_epoch",
            "checkpoint_every",
            "report_every",
            "convergence_window",
        ):
            _positive_int(getattr(self, name), f"training.{name}")
        for name in ("warmup_epochs", "dynamics_ramp_epochs", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"training.{name} must be a non-negative integer")
        for name in (
            "state_learning_rate",
            "decoder_learning_rate",
            "dynamics_learning_rate",
            "gradient_clip",
            "convergence_relative_tolerance",
        ):
            if _finite(getattr(self, name), f"training.{name}") <= 0.0:
                raise ValueError(f"training.{name} must be finite and positive")
        for name in (
            "dynamics_weight",
            "gauge_weight",
            "cap_penalty_weight",
            "minimum_learning_rate_fraction",
            "maximum_quote_containment_ratio_drop",
        ):
            if _finite(getattr(self, name), f"training.{name}") < 0.0:
                raise ValueError(f"training.{name} must be finite and non-negative")
        if not isinstance(self.compile_sanos_solver, bool):
            raise TypeError("training.compile_sanos_solver must be boolean")


@version(
    "1.0",
    dependencies=[
        SourceDatasetConfig,
        DailyFrontMarketConfig,
        DailyFrontGeometryConfig,
        ObjectiveConfig,
        SplitConfig,
        Hdf5StorageConfig,
        DirectModelConfig,
        DirectTrainingConfig,
    ],
)
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete content and consumer identity for one reference Experiment."""

    name: str
    source_dataset: SourceDatasetConfig
    profile: Literal["dysanos-daily-front-market-v1"] = "dysanos-daily-front-market-v1"
    consumer_contract: Literal["flex-market-day-v1"] = "flex-market-day-v1"
    market: DailyFrontMarketConfig = DailyFrontMarketConfig()
    geometry: DailyFrontGeometryConfig = DailyFrontGeometryConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    split: SplitConfig = SplitConfig()
    storage: Hdf5StorageConfig = Hdf5StorageConfig()
    model: DirectModelConfig = DirectModelConfig()
    training: DirectTrainingConfig = DirectTrainingConfig()

    def __post_init__(self) -> None:
        if _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("Experiment name must be a portable 1-128 character label")
        if self.profile != "dysanos-daily-front-market-v1":
            raise ValueError("Unsupported Experiment profile")
        if self.consumer_contract != "flex-market-day-v1":
            raise ValueError("Unsupported Experiment consumer contract")
        expected = len(self.geometry.model_business_dtes) * (self.geometry.num_inner_strikes + 2)
        if self.model.decoded_state_size != expected:
            raise ValueError(
                "model.decoded_state_size must equal num_model_expiries * (num_inner_strikes + 2)"
            )

    def to_document(self) -> dict[str, Any]:
        """Return the strict checked-in configuration document shape."""

        from dataclasses import asdict

        from sanos.config import canonical_value

        return cast(
            dict[str, Any],
            canonical_value(
                {
                    "experiment": {
                        "name": self.name,
                        "profile": self.profile,
                        "source_dataset": asdict(self.source_dataset),
                        "consumer_contract": self.consumer_contract,
                    },
                    "market": asdict(self.market),
                    "geometry": asdict(self.geometry),
                    "objective": asdict(self.objective),
                    "split": asdict(self.split),
                    "storage": asdict(self.storage),
                    "model": asdict(self.model),
                    "training": asdict(self.training),
                }
            ),
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ExperimentConfig:
        sections = {
            "experiment",
            "market",
            "geometry",
            "objective",
            "split",
            "storage",
            "model",
            "training",
        }
        _require_exact(document, sections, "Experiment configuration sections")
        if any(not isinstance(document[name], Mapping) for name in sections):
            raise ValueError("Every Experiment configuration section must be an object")
        experiment = cast(Mapping[str, Any], document["experiment"])
        _require_exact(
            experiment,
            {"name", "profile", "source_dataset", "consumer_contract"},
            "experiment",
        )
        source = experiment["source_dataset"]
        if not isinstance(source, Mapping):
            raise ValueError("experiment.source_dataset must be an object")
        simple_classes: tuple[tuple[str, type[Any]], ...] = (
            ("objective", ObjectiveConfig),
            ("split", SplitConfig),
            ("storage", Hdf5StorageConfig),
            ("model", DirectModelConfig),
            ("training", DirectTrainingConfig),
        )
        parsed: dict[str, Any] = {}
        for section, config_type in simple_classes:
            values = cast(Mapping[str, Any], document[section])
            fields = set(config_type.__dataclass_fields__)
            _require_exact(values, fields, section)
            try:
                parsed[section] = config_type(**values)
            except TypeError as exc:
                raise ValueError(f"Invalid {section} configuration: {exc}") from exc
        string_fields = ("name", "profile", "consumer_contract")
        if not all(isinstance(experiment[name], str) for name in string_fields):
            raise ValueError("experiment name/profile/consumer_contract must be strings")
        return cls(
            name=cast(str, experiment["name"]),
            profile=cast(Any, experiment["profile"]),
            source_dataset=SourceDatasetConfig.from_mapping(source),
            consumer_contract=cast(Any, experiment["consumer_contract"]),
            market=DailyFrontMarketConfig.from_mapping(cast(Mapping[str, Any], document["market"])),
            geometry=DailyFrontGeometryConfig.from_mapping(
                cast(Mapping[str, Any], document["geometry"])
            ),
            **parsed,
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class ExperimentBuildOptions:
    """Operational controls excluded from Experiment content identity."""

    max_workers: int = 1
    progress_interval_seconds: float = 30.0
    keep_work: bool = False
    resume: bool = True
    restart_work: bool = False
    version_override: Literal["*"] | None = None

    def __post_init__(self) -> None:
        if not 1 <= _positive_int(self.max_workers, "max_workers") <= 4:
            raise ValueError("max_workers must lie in [1,4]")
        if (
            isinstance(self.progress_interval_seconds, bool)
            or not isinstance(self.progress_interval_seconds, int | float)
            or not 0.0 < self.progress_interval_seconds <= 300.0
        ):
            raise ValueError("progress_interval_seconds must lie in (0,300]")
        for name in ("keep_work", "resume", "restart_work"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.version_override not in {None, "*"}:
            raise ValueError("version_override must be None or '*'")


__all__ = [
    "DailyFrontGeometryConfig",
    "DailyFrontMarketConfig",
    "DirectModelConfig",
    "DirectTrainingConfig",
    "ExperimentBuildOptions",
    "ExperimentConfig",
    "Hdf5StorageConfig",
    "ObjectiveConfig",
    "SourceDatasetConfig",
    "SplitConfig",
]
