"""Strict configuration for the lossy public SANOS/DLV projection."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from cdxcore.version import version

from sanos.artifacts import ArtifactPointer, StorageClass

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _date(value: Any, label: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _pointer(value: Any, label: str) -> ArtifactPointer:
    if not isinstance(value, Mapping) or set(value) != {
        "storage_class",
        "namespace",
        "artifact_id",
    }:
        raise ValueError(f"{label} must be an exact artifact pointer")
    if value["storage_class"] != StorageClass.DATASET.value:
        raise ValueError(f"{label}.storage_class must be 'dataset'")
    if any(
        not isinstance(value[name], str)
        for name in ("storage_class", "namespace", "artifact_id")
    ):
        raise ValueError(f"{label} pointer values must be strings")
    return ArtifactPointer(
        StorageClass.DATASET,
        str(value["namespace"]),
        str(value["artifact_id"]),
    )


@version("1.0")
@dataclass(frozen=True, slots=True, order=True)
class PublicExpectedOmission:
    """One policy-owned typed-empty source date allowed by public identity."""

    trading_date: date
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trading_date", _date(self.trading_date, "trading_date"))
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("An expected omission reason must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {"trading_date": self.trading_date.isoformat(), "reason": self.reason}

    @classmethod
    def from_value(cls, value: Any) -> PublicExpectedOmission:
        if not isinstance(value, Mapping) or set(value) != {"trading_date", "reason"}:
            raise ValueError("Each expected_typed_empty_omission needs date and reason")
        if not isinstance(value["reason"], str):
            raise ValueError("Expected omission reason must be a string")
        return cls(_date(value["trading_date"], "trading_date"), value["reason"])


@version("1.1", dependencies=[ArtifactPointer, PublicExpectedOmission])
@dataclass(frozen=True, slots=True)
class PublicProjectionConfig:
    """All content-affecting choices for one public-candidate projection."""

    name: str
    source_dataset: ArtifactPointer
    coverage_start_date: date
    coverage_end_date: date
    expected_typed_empty_omissions: tuple[PublicExpectedOmission, ...] = ()
    profile: Literal["daily-front-v1"] = "daily-front-v1"
    return_convention: Literal["previous-retained-log-carry-v1"] = (
        "previous-retained-log-carry-v1"
    )
    unit_convention: Literal["discounted-forward-over-spot-v1"] = (
        "discounted-forward-over-spot-v1"
    )
    nms_range: tuple[float, float] = (-3.0, 2.0)
    nms_epsilon_business_dte: float = 1.0
    maximum_business_dte: float = 70.0
    excluded_business_dtes: tuple[float, ...] = (0.2,)
    exclusion_tolerance: float = 1.0e-4
    smoothness: float = 0.25
    european_only: bool = True
    require_finite_smooth_price: bool = True
    separate_book_sides: bool = True
    state_scope: Literal["complete-common-surface-v1"] = "complete-common-surface-v1"
    retained_date_policy: Literal["populated-retained-dates-v1"] = (
        "populated-retained-dates-v1"
    )
    overnight_variance_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("Public Dataset name must be portable and filesystem-safe")
        if not isinstance(self.source_dataset, ArtifactPointer):
            raise TypeError("source_dataset must be an ArtifactPointer")
        if self.source_dataset.storage_class is not StorageClass.DATASET:
            raise ValueError("A public projection source must be a Dataset artifact")
        start = _date(self.coverage_start_date, "coverage_start_date")
        end = _date(self.coverage_end_date, "coverage_end_date")
        if end < start:
            raise ValueError("coverage_end_date cannot precede coverage_start_date")
        omissions = tuple(self.expected_typed_empty_omissions)
        if any(not isinstance(item, PublicExpectedOmission) for item in omissions):
            raise TypeError("expected_typed_empty_omissions must contain omission records")
        if omissions != tuple(sorted(set(omissions))):
            raise ValueError("Expected typed-empty omissions must be unique and date ordered")
        if any(not start <= item.trading_date <= end for item in omissions):
            raise ValueError("Expected typed-empty omission lies outside coverage")
        object.__setattr__(self, "coverage_start_date", start)
        object.__setattr__(self, "coverage_end_date", end)
        object.__setattr__(self, "expected_typed_empty_omissions", omissions)
        if self.profile != "daily-front-v1":
            raise ValueError("Unsupported public projection profile")
        if self.return_convention != "previous-retained-log-carry-v1":
            raise ValueError("Unsupported public return convention")
        if self.unit_convention != "discounted-forward-over-spot-v1":
            raise ValueError("Unsupported public unit convention")
        if self.state_scope != "complete-common-surface-v1":
            raise ValueError("Unsupported public state scope")
        if self.retained_date_policy != "populated-retained-dates-v1":
            raise ValueError("Unsupported public retained-date policy")
        if any(
            not isinstance(value, bool)
            for value in (
                self.european_only,
                self.require_finite_smooth_price,
                self.separate_book_sides,
            )
        ):
            raise TypeError("Public projection policy switches must be booleans")
        if not all(
            (self.european_only, self.require_finite_smooth_price, self.separate_book_sides)
        ):
            raise ValueError("daily-front-v1 requires European, finite-fit, separate-side books")
        if len(self.nms_range) != 2:
            raise ValueError("nms_range must contain exactly two values")
        lower, upper = (_finite(value, "nms_range[]") for value in self.nms_range)
        if not lower < 0.0 < upper:
            raise ValueError("nms_range must straddle zero")
        object.__setattr__(self, "nms_range", (lower, upper))
        epsilon_dte = _finite(self.nms_epsilon_business_dte, "nms_epsilon_business_dte")
        maximum_dte = _finite(self.maximum_business_dte, "maximum_business_dte")
        tolerance = _finite(self.exclusion_tolerance, "exclusion_tolerance")
        smoothness = _finite(self.smoothness, "smoothness")
        overnight = _finite(
            self.overnight_variance_fraction, "overnight_variance_fraction"
        )
        excluded = tuple(
            _finite(value, "excluded_business_dtes[]")
            for value in self.excluded_business_dtes
        )
        if (
            epsilon_dte <= 0.0
            or maximum_dte <= 0.0
            or tolerance <= 0.0
            or smoothness <= 0.0
        ):
            raise ValueError("Public DTE and tolerance values must be positive")
        if any(value < 0.0 for value in excluded):
            raise ValueError("excluded_business_dtes cannot contain negative values")
        if excluded != tuple(sorted(set(excluded))):
            raise ValueError("excluded_business_dtes must be unique and increasing")
        if not 0.0 <= overnight <= 1.0:
            raise ValueError("overnight_variance_fraction must lie in [0, 1]")
        object.__setattr__(self, "nms_epsilon_business_dte", epsilon_dte)
        object.__setattr__(self, "maximum_business_dte", maximum_dte)
        object.__setattr__(self, "exclusion_tolerance", tolerance)
        object.__setattr__(self, "smoothness", smoothness)
        object.__setattr__(self, "excluded_business_dtes", excluded)
        object.__setattr__(self, "overnight_variance_fraction", overnight)

    def as_dict(self) -> dict[str, Any]:
        """Return the exact portable content-identity document."""

        return {
            "name": self.name,
            "source_dataset": self.source_dataset.as_dict(),
            "coverage_start_date": self.coverage_start_date.isoformat(),
            "coverage_end_date": self.coverage_end_date.isoformat(),
            "expected_typed_empty_omissions": [
                item.as_dict() for item in self.expected_typed_empty_omissions
            ],
            "profile": self.profile,
            "return_convention": self.return_convention,
            "unit_convention": self.unit_convention,
            "nms_range": list(self.nms_range),
            "nms_epsilon_business_dte": self.nms_epsilon_business_dte,
            "maximum_business_dte": self.maximum_business_dte,
            "excluded_business_dtes": list(self.excluded_business_dtes),
            "exclusion_tolerance": self.exclusion_tolerance,
            "smoothness": self.smoothness,
            "european_only": self.european_only,
            "require_finite_smooth_price": self.require_finite_smooth_price,
            "separate_book_sides": self.separate_book_sides,
            "state_scope": self.state_scope,
            "retained_date_policy": self.retained_date_policy,
            "overnight_variance_fraction": self.overnight_variance_fraction,
        }

    def to_document(self) -> dict[str, Any]:
        """Return the checked-in one-section configuration document."""

        return {"public_dataset": self.as_dict()}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> PublicProjectionConfig:
        """Parse the strict public projection configuration contract."""

        if set(document) != {"public_dataset"}:
            raise ValueError("Public configuration must contain only 'public_dataset'")
        value = document["public_dataset"]
        if not isinstance(value, Mapping):
            raise ValueError("public_dataset must be an object")
        fields = {
            "name",
            "source_dataset",
            "coverage_start_date",
            "coverage_end_date",
            "expected_typed_empty_omissions",
            "profile",
            "return_convention",
            "unit_convention",
            "nms_range",
            "nms_epsilon_business_dte",
            "maximum_business_dte",
            "excluded_business_dtes",
            "exclusion_tolerance",
            "smoothness",
            "european_only",
            "require_finite_smooth_price",
            "separate_book_sides",
            "state_scope",
            "retained_date_policy",
            "overnight_variance_fraction",
        }
        if set(value) != fields:
            missing = sorted(fields - set(value))
            unknown = sorted(set(value) - fields)
            raise ValueError(
                f"Public Dataset fields changed: missing={missing}, unknown={unknown}"
            )
        source = _pointer(value["source_dataset"], "source_dataset")
        string_fields = (
            "name",
            "profile",
            "return_convention",
            "unit_convention",
            "state_scope",
            "retained_date_policy",
        )
        if any(not isinstance(value[name], str) for name in string_fields):
            raise ValueError("Public Dataset string fields must be strings")
        bool_fields = (
            "european_only",
            "require_finite_smooth_price",
            "separate_book_sides",
        )
        if any(not isinstance(value[name], bool) for name in bool_fields):
            raise ValueError("Public Dataset policy fields must be booleans")
        nms = value["nms_range"]
        excluded = value["excluded_business_dtes"]
        if not isinstance(nms, list) or len(nms) != 2:
            raise ValueError("nms_range must be a two-element list")
        if not isinstance(excluded, list):
            raise ValueError("excluded_business_dtes must be a list")
        omissions_value = value["expected_typed_empty_omissions"]
        if not isinstance(omissions_value, list):
            raise ValueError("expected_typed_empty_omissions must be a list")
        return cls(
            name=value["name"],
            source_dataset=source,
            coverage_start_date=_date(value["coverage_start_date"], "coverage_start_date"),
            coverage_end_date=_date(value["coverage_end_date"], "coverage_end_date"),
            expected_typed_empty_omissions=tuple(
                PublicExpectedOmission.from_value(item) for item in omissions_value
            ),
            profile=value["profile"],
            return_convention=value["return_convention"],
            unit_convention=value["unit_convention"],
            nms_range=tuple(_finite(item, "nms_range[]") for item in nms),
            nms_epsilon_business_dte=_finite(
                value["nms_epsilon_business_dte"], "nms_epsilon_business_dte"
            ),
            maximum_business_dte=_finite(
                value["maximum_business_dte"], "maximum_business_dte"
            ),
            excluded_business_dtes=tuple(
                _finite(item, "excluded_business_dtes[]") for item in excluded
            ),
            exclusion_tolerance=_finite(
                value["exclusion_tolerance"], "exclusion_tolerance"
            ),
            smoothness=_finite(value["smoothness"], "smoothness"),
            european_only=value["european_only"],
            require_finite_smooth_price=value["require_finite_smooth_price"],
            separate_book_sides=value["separate_book_sides"],
            state_scope=value["state_scope"],
            retained_date_policy=value["retained_date_policy"],
            overnight_variance_fraction=_finite(
                value["overnight_variance_fraction"], "overnight_variance_fraction"
            ),
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class PublicBuildOptions:
    """Operational controls which never participate in Dataset identity."""

    progress_interval_seconds: float = 30.0
    version_override: Literal["*"] | None = None

    def __post_init__(self) -> None:
        interval = _finite(self.progress_interval_seconds, "progress_interval_seconds")
        if not 0.0 < interval <= 300.0:
            raise ValueError("progress_interval_seconds must lie in (0, 300]")
        if self.version_override not in {None, "*"}:
            raise ValueError("version_override must be None or '*'")
        object.__setattr__(self, "progress_interval_seconds", interval)


@version("1.0", dependencies=[ArtifactPointer])
@dataclass(frozen=True, slots=True)
class PublicApproval:
    """Exact rights and candidate evidence required for a public release."""

    candidate_dataset: ArtifactPointer
    candidate_manifest_sha256: str
    candidate_logical_content_sha256: str
    release_license: str
    redistribution_basis: str
    approver: str
    approval_date: date
    approval_reference: str
    license_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_dataset, ArtifactPointer):
            raise TypeError("candidate_dataset must be an ArtifactPointer")
        if self.candidate_dataset.storage_class is not StorageClass.DATASET:
            raise ValueError("candidate_dataset must point to a Dataset")
        for name in (
            "candidate_manifest_sha256",
            "candidate_logical_content_sha256",
            "license_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "release_license",
            "redistribution_basis",
            "approver",
            "approval_reference",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "approval_date", _date(self.approval_date, "approval_date"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_dataset": self.candidate_dataset.as_dict(),
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "candidate_logical_content_sha256": self.candidate_logical_content_sha256,
            "release_license": self.release_license,
            "redistribution_basis": self.redistribution_basis,
            "approver": self.approver,
            "approval_date": self.approval_date.isoformat(),
            "approval_reference": self.approval_reference,
            "license_sha256": self.license_sha256,
        }

    @classmethod
    def from_value(cls, value: Any) -> PublicApproval:
        fields = {
            "candidate_dataset",
            "candidate_manifest_sha256",
            "candidate_logical_content_sha256",
            "release_license",
            "redistribution_basis",
            "approver",
            "approval_date",
            "approval_reference",
            "license_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("Public approval fields are missing or unexpected")
        strings = fields - {"candidate_dataset"}
        if any(not isinstance(value[name], str) for name in strings):
            raise ValueError("Public approval scalar fields must be strings")
        return cls(
            candidate_dataset=_pointer(value["candidate_dataset"], "candidate_dataset"),
            candidate_manifest_sha256=value["candidate_manifest_sha256"],
            candidate_logical_content_sha256=value["candidate_logical_content_sha256"],
            release_license=value["release_license"],
            redistribution_basis=value["redistribution_basis"],
            approver=value["approver"],
            approval_date=_date(value["approval_date"], "approval_date"),
            approval_reference=value["approval_reference"],
            license_sha256=value["license_sha256"],
        )


@version("1.0", dependencies=[PublicProjectionConfig, PublicApproval])
@dataclass(frozen=True, slots=True)
class PublicReleaseConfig:
    """Self-contained configuration for a distinct approved public release."""

    projection: PublicProjectionConfig
    approval: PublicApproval

    def __post_init__(self) -> None:
        if not isinstance(self.projection, PublicProjectionConfig):
            raise TypeError("projection must be a PublicProjectionConfig")
        if not isinstance(self.approval, PublicApproval):
            raise TypeError("approval must be a PublicApproval")
        if not self.approval.candidate_dataset.namespace.startswith(
            "sanos/public-candidate/"
        ):
            raise ValueError("Approval candidate must use the public-candidate namespace")

    def as_dict(self) -> dict[str, Any]:
        return {"projection": self.projection.as_dict(), "approval": self.approval.as_dict()}

    def to_document(self) -> dict[str, Any]:
        return {"public_release": self.as_dict()}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> PublicReleaseConfig:
        if set(document) != {"public_release"}:
            raise ValueError("Public release configuration must contain only 'public_release'")
        value = document["public_release"]
        if not isinstance(value, Mapping) or set(value) != {"projection", "approval"}:
            raise ValueError("public_release needs exact projection and approval objects")
        projection = PublicProjectionConfig.from_document(
            {"public_dataset": value["projection"]}
        )
        approval = PublicApproval.from_value(value["approval"])
        return cls(projection=projection, approval=approval)


__all__ = [
    "PublicApproval",
    "PublicBuildOptions",
    "PublicExpectedOmission",
    "PublicProjectionConfig",
    "PublicReleaseConfig",
]
