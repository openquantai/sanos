"""Strict portable configuration for immutable internal SANOS datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from cdxcore.version import version

from sanos.widb.config import DailySurfaceConfig, DailyWidbConfig

_SAFE_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OPTIONMETRICS_TICKER = re.compile(r"^[A-Za-z0-9.^-]{1,32}$")


@version("1.0", dependencies=[DailyWidbConfig, DailySurfaceConfig])
@dataclass(frozen=True, slots=True)
class InternalDatasetConfig:
    """All content-affecting inputs for one internal WIDB/SANOS dataset."""

    name: str
    ticker: str
    start_date: date
    end_date: date
    include_half_days: bool
    source: Literal["optionmetrics"]
    source_generation: str
    daily: DailyWidbConfig
    surface: DailySurfaceConfig

    def __post_init__(self) -> None:
        if _SAFE_DATASET_NAME.fullmatch(self.name) is None:
            raise ValueError(
                "Dataset name must be 1-128 portable characters beginning with a letter "
                "or digit"
            )
        normalized_ticker = self.ticker.strip().upper()
        if _OPTIONMETRICS_TICKER.fullmatch(normalized_ticker) is None:
            raise ValueError("Dataset ticker is not a valid OptionMetrics ticker")
        object.__setattr__(self, "ticker", normalized_ticker)
        if self.end_date < self.start_date:
            raise ValueError("Dataset end_date cannot precede start_date")
        if not isinstance(self.include_half_days, bool):
            raise TypeError("include_half_days must be a boolean")
        if self.source != "optionmetrics":
            raise ValueError(f"Unsupported Dataset source: {self.source!r}")
        if not self.source_generation.strip():
            raise ValueError("source_generation cannot be empty")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> InternalDatasetConfig:
        """Parse a strict checked-in ``dataset``/``daily``/``surface`` document."""

        required = {"dataset", "daily", "surface"}
        if set(document) != required:
            missing = sorted(required - set(document))
            unknown = sorted(set(document) - required)
            raise ValueError(
                f"Dataset configuration sections changed: missing={missing}, unknown={unknown}"
            )
        dataset = document["dataset"]
        daily = document["daily"]
        surface = document["surface"]
        if not isinstance(dataset, Mapping):
            raise ValueError("Dataset configuration 'dataset' must be an object")
        if not isinstance(daily, Mapping):
            raise ValueError("Dataset configuration 'daily' must be an object")
        if not isinstance(surface, Mapping):
            raise ValueError("Dataset configuration 'surface' must be an object")
        expected = {
            "name",
            "ticker",
            "start_date",
            "end_date",
            "include_half_days",
            "source",
            "source_generation",
        }
        if set(dataset) != expected:
            missing = sorted(expected - set(dataset))
            unknown = sorted(set(dataset) - expected)
            raise ValueError(
                f"Internal Dataset fields changed: missing={missing}, unknown={unknown}"
            )
        string_fields = ("name", "ticker", "start_date", "end_date", "source", "source_generation")
        invalid_strings = [name for name in string_fields if not isinstance(dataset[name], str)]
        if invalid_strings:
            raise ValueError(
                "Internal Dataset fields must be strings: " + ", ".join(invalid_strings)
            )
        if not isinstance(dataset["include_half_days"], bool):
            raise ValueError("Internal Dataset include_half_days must be a boolean")
        try:
            start_date = date.fromisoformat(dataset["start_date"])
            end_date = date.fromisoformat(dataset["end_date"])
        except ValueError as exc:
            raise ValueError("Dataset start_date/end_date must be ISO calendar dates") from exc
        source = dataset["source"]
        if source != "optionmetrics":
            raise ValueError(f"Unsupported Dataset source: {source!r}")
        try:
            return cls(
                name=dataset["name"],
                ticker=dataset["ticker"],
                start_date=start_date,
                end_date=end_date,
                include_half_days=dataset["include_half_days"],
                source="optionmetrics",
                source_generation=dataset["source_generation"],
                daily=DailyWidbConfig.from_mapping(daily),
                surface=DailySurfaceConfig.from_mapping(surface),
            )
        except TypeError as exc:
            raise ValueError(f"Invalid internal Dataset configuration field: {exc}") from exc


@version("1.0")
@dataclass(frozen=True, slots=True)
class DatasetBuildOptions:
    """Operational build controls excluded from Dataset content identity."""

    max_workers: int = 1
    progress_interval_seconds: float = 30.0
    require_existing_intermediates: bool = False
    version_override: Literal["*"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int):
            raise TypeError("max_workers must be an integer")
        if not 1 <= self.max_workers <= 4:
            raise ValueError("Dataset surface fitting supports between one and four workers")
        if (
            isinstance(self.progress_interval_seconds, bool)
            or not isinstance(self.progress_interval_seconds, int | float)
            or not 0.0 < self.progress_interval_seconds <= 300.0
        ):
            raise ValueError("progress_interval_seconds must lie in (0, 300]")
        if not isinstance(self.require_existing_intermediates, bool):
            raise TypeError("require_existing_intermediates must be a boolean")
        if self.version_override not in {None, "*"}:
            raise ValueError("version_override must be None or '*'")


__all__ = ["DatasetBuildOptions", "InternalDatasetConfig"]
