"""Explicit recovery of failed fitted days without replacing successful artifacts."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

from cdxcore.version import version

from sanos.artifacts import (
    ArtifactNotFoundError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    StorageClass,
    ValidatedArtifact,
    atomic_create_json,
)
from sanos.config import StorageLocations, canonical_sha256, canonical_value
from sanos.widb.build import publish_daily_frame, read_daily_frame
from sanos.widb.calendar import business_dates
from sanos.widb.config import DailySurfaceConfig
from sanos.widb.surface import (
    FILTERED_SCHEMA,
    FILTERED_SCHEMA_VERSION,
    FITTED_SCHEMA,
    FITTED_SCHEMA_VERSION,
    fit_smooth_sanos,
)

ORIGINAL_FITTED_PLAN_SCHEMA = "sanos-original-fitted-plan-v1"
FITTED_SOURCE_SELECTION_SCHEMA = "sanos-fitted-source-selection-v1"
FITTED_RECOVERY_SCHEMA_VERSION = "sanos-fitted-recovery-v1"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@version("1.0", dependencies=[ArtifactPointer])
def _recovery_pointer(
    value: Any,
    label: str,
    *,
    storage_class: StorageClass = StorageClass.INTERMEDIATE,
) -> ArtifactPointer:
    if not isinstance(value, Mapping) or set(value) != {
        "storage_class",
        "namespace",
        "artifact_id",
    }:
        raise ValueError(f"{label} must be an exact artifact pointer")
    pointer = ArtifactPointer(
        storage_class=StorageClass(str(value["storage_class"])),
        namespace=str(value["namespace"]),
        artifact_id=str(value["artifact_id"]),
    )
    if pointer.storage_class is not storage_class:
        raise ValueError(f"{label} must identify a {storage_class.value} artifact")
    return pointer


@version("2.0", dependencies=[ArtifactPointer, canonical_value])
@dataclass(frozen=True, slots=True)
class OriginalFittedSource:
    """One complete frozen pre-policy daily chain."""

    trading_date: date
    raw_security: ArtifactPointer
    raw_options: ArtifactPointer
    raw_spot: ArtifactPointer
    raw_previous_spot: ArtifactPointer
    raw_forwards: ArtifactPointer
    raw_zero_curve: ArtifactPointer
    paired: ArtifactPointer
    enriched: ArtifactPointer
    no_arb: ArtifactPointer
    filtered: ArtifactPointer
    expected_fitted: ArtifactPointer

    @property
    def raw_roles(self) -> dict[str, ArtifactPointer]:
        return {
            "security": self.raw_security,
            "options": self.raw_options,
            "spot": self.raw_spot,
            "previous_spot": self.raw_previous_spot,
            "forwards": self.raw_forwards,
            "zero_curve": self.raw_zero_curve,
        }

    def as_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            canonical_value(
                {
                    "trading_date": self.trading_date,
                    "raw_roles": self.raw_roles,
                    "paired": self.paired,
                    "enriched": self.enriched,
                    "no_arb": self.no_arb,
                    "filtered": self.filtered,
                    "expected_fitted": self.expected_fitted,
                }
            ),
        )


@version(
    "1.0",
    dependencies=[OriginalFittedSource, _recovery_pointer, canonical_sha256],
)
@dataclass(frozen=True, slots=True)
class OriginalFittedPlan:
    """Portable evidence exported from the frozen run before recovery code changes."""

    ticker: str
    source_revision: str
    configuration_sha256: str
    sources: tuple[OriginalFittedSource, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.ticker or self.ticker != self.ticker.strip().upper():
            raise ValueError("Original fitted plan ticker must be normalized")
        if _HEX_40.fullmatch(self.source_revision) is None:
            raise ValueError("Original fitted plan requires an exact source revision")
        if _HEX_64.fullmatch(self.configuration_sha256) is None:
            raise ValueError("Original fitted plan configuration digest is invalid")
        dates = tuple(item.trading_date for item in self.sources)
        if not dates or dates != tuple(sorted(set(dates))):
            raise ValueError("Original fitted plan dates must be unique and ordered")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
            "ticker": self.ticker,
            "source_revision": self.source_revision,
            "configuration_sha256": self.configuration_sha256,
            "sources": [item.as_dict() for item in self.sources],
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> OriginalFittedPlan:
        required = {
            "schema",
            "ticker",
            "source_revision",
            "configuration_sha256",
            "sources",
            "evidence_sha256",
        }
        if set(document) != required or document.get("schema") != ORIGINAL_FITTED_PLAN_SCHEMA:
            raise ValueError("Original fitted plan schema or fields changed")
        raw_sources = document["sources"]
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("Original fitted plan sources must be an array")
        sources: list[OriginalFittedSource] = []
        for index, value in enumerate(raw_sources):
            expected_source_fields = {
                "trading_date",
                "raw_roles",
                "paired",
                "enriched",
                "no_arb",
                "filtered",
                "expected_fitted",
            }
            if not isinstance(value, Mapping) or set(value) != expected_source_fields:
                raise ValueError(f"Original fitted plan source {index} fields changed")
            raw_roles = value["raw_roles"]
            expected_roles = {
                "security",
                "options",
                "spot",
                "previous_spot",
                "forwards",
                "zero_curve",
            }
            if not isinstance(raw_roles, Mapping) or set(raw_roles) != expected_roles:
                raise ValueError(f"Original fitted plan source {index} Raw roles changed")
            sources.append(
                OriginalFittedSource(
                    trading_date=date.fromisoformat(str(value["trading_date"])),
                    raw_security=_recovery_pointer(
                        raw_roles["security"],
                        f"source {index} Raw security",
                        storage_class=StorageClass.RAW,
                    ),
                    raw_options=_recovery_pointer(
                        raw_roles["options"],
                        f"source {index} Raw options",
                        storage_class=StorageClass.RAW,
                    ),
                    raw_spot=_recovery_pointer(
                        raw_roles["spot"],
                        f"source {index} Raw spot",
                        storage_class=StorageClass.RAW,
                    ),
                    raw_previous_spot=_recovery_pointer(
                        raw_roles["previous_spot"],
                        f"source {index} Raw previous_spot",
                        storage_class=StorageClass.RAW,
                    ),
                    raw_forwards=_recovery_pointer(
                        raw_roles["forwards"],
                        f"source {index} Raw forwards",
                        storage_class=StorageClass.RAW,
                    ),
                    raw_zero_curve=_recovery_pointer(
                        raw_roles["zero_curve"],
                        f"source {index} Raw zero_curve",
                        storage_class=StorageClass.RAW,
                    ),
                    paired=_recovery_pointer(value["paired"], f"source {index} paired"),
                    enriched=_recovery_pointer(value["enriched"], f"source {index} enriched"),
                    no_arb=_recovery_pointer(value["no_arb"], f"source {index} no_arb"),
                    filtered=_recovery_pointer(value["filtered"], f"source {index} filtered"),
                    expected_fitted=_recovery_pointer(
                        value["expected_fitted"], f"source {index} expected_fitted"
                    ),
                )
            )
        unsigned = {
            "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
            "ticker": document["ticker"],
            "source_revision": document["source_revision"],
            "configuration_sha256": document["configuration_sha256"],
            "sources": [item.as_dict() for item in sources],
        }
        evidence_sha256 = str(document["evidence_sha256"])
        if evidence_sha256 != canonical_sha256(unsigned):
            raise ValueError("Original fitted plan evidence_sha256 does not match its content")
        return cls(
            ticker=str(document["ticker"]),
            source_revision=str(document["source_revision"]),
            configuration_sha256=str(document["configuration_sha256"]),
            sources=tuple(sources),
            evidence_sha256=evidence_sha256,
        )


@version(
    "2.0",
    dependencies=[ArtifactPointer, OriginalFittedSource, canonical_sha256, canonical_value],
)
@dataclass(frozen=True, slots=True)
class FittedSourceBinding:
    """Exact original and selected pointer for one Dataset partition."""

    trading_date: date
    original_source: OriginalFittedSource
    selected: ArtifactPointer
    recovery_policy_id: str | None = None
    failure_evidence_sha256: str | None = None
    failure_evidence: tuple[Mapping[str, Any], ...] | None = None

    @property
    def original_expected(self) -> ArtifactPointer:
        return self.original_source.expected_fitted

    @property
    def filtered(self) -> ArtifactPointer:
        return self.original_source.filtered

    def __post_init__(self) -> None:
        if self.trading_date != self.original_source.trading_date:
            raise ValueError("Fitted binding date differs from its original source")
        recovered = self.selected != self.original_expected
        if recovered != (self.recovery_policy_id is not None):
            raise ValueError("Recovered fitted bindings require exactly one recovery policy ID")
        if recovered != (self.failure_evidence_sha256 is not None):
            raise ValueError("Recovered fitted bindings require failure evidence")
        if recovered != (self.failure_evidence is not None):
            raise ValueError("Recovered fitted bindings require embedded failure records")
        if recovered and canonical_sha256(self.failure_evidence) != self.failure_evidence_sha256:
            raise ValueError("Recovered fitted binding failure evidence digest differs")

    def as_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            canonical_value(
                {
                    "trading_date": self.trading_date,
                    "original_source": self.original_source.as_dict(),
                    "selected": self.selected,
                    "recovery_policy_id": self.recovery_policy_id,
                    "failure_evidence_sha256": self.failure_evidence_sha256,
                    "failure_evidence": self.failure_evidence,
                }
            ),
        )


@version(
    "1.0",
    dependencies=[
        FittedSourceBinding,
        OriginalFittedPlan,
        _recovery_pointer,
        canonical_sha256,
        canonical_value,
    ],
)
@dataclass(frozen=True, slots=True)
class FittedSourceSelection:
    """Complete exact fitted-source selection for one internal Dataset."""

    ticker: str
    original_plan_sha256: str
    original_source_revision: str
    original_configuration_sha256: str
    bindings: tuple[FittedSourceBinding, ...]

    def __post_init__(self) -> None:
        dates = tuple(item.trading_date for item in self.bindings)
        if not dates or dates != tuple(sorted(set(dates))):
            raise ValueError("Fitted source selection dates must be unique and ordered")
        if not self.ticker or self.ticker != self.ticker.strip().upper():
            raise ValueError("Fitted source selection ticker must be normalized")
        if _HEX_40.fullmatch(self.original_source_revision) is None:
            raise ValueError("Fitted source selection source revision is invalid")
        if _HEX_64.fullmatch(self.original_configuration_sha256) is None:
            raise ValueError("Fitted source selection configuration digest is invalid")
        if _HEX_64.fullmatch(self.original_plan_sha256) is None:
            raise ValueError("Fitted source selection original plan digest is invalid")
        unsigned_plan = {
            "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
            "ticker": self.ticker,
            "source_revision": self.original_source_revision,
            "configuration_sha256": self.original_configuration_sha256,
            "sources": [item.original_source.as_dict() for item in self.bindings],
        }
        if canonical_sha256(unsigned_plan) != self.original_plan_sha256:
            raise ValueError(
                "Fitted source selection original plan digest differs from embedded sources"
            )

    def as_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            canonical_value(
                {
                    "schema": FITTED_SOURCE_SELECTION_SCHEMA,
                    "ticker": self.ticker,
                    "original_plan_sha256": self.original_plan_sha256,
                    "original_source_revision": self.original_source_revision,
                    "original_configuration_sha256": self.original_configuration_sha256,
                    "bindings": [item.as_dict() for item in self.bindings],
                }
            ),
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> FittedSourceSelection:
        required = {
            "schema",
            "ticker",
            "original_plan_sha256",
            "original_source_revision",
            "original_configuration_sha256",
            "bindings",
        }
        if set(document) != required or document.get("schema") != FITTED_SOURCE_SELECTION_SCHEMA:
            raise ValueError("Fitted source selection schema or fields changed")
        raw_bindings = document["bindings"]
        if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
            raise ValueError("Fitted source selection bindings must be an array")
        bindings: list[FittedSourceBinding] = []
        expected_fields = {
            "trading_date",
            "original_source",
            "selected",
            "recovery_policy_id",
            "failure_evidence_sha256",
            "failure_evidence",
        }
        for index, value in enumerate(raw_bindings):
            if not isinstance(value, Mapping) or set(value) != expected_fields:
                raise ValueError(f"Fitted source binding {index} fields changed")
            policy = value["recovery_policy_id"]
            evidence = value["failure_evidence_sha256"]
            evidence_records = value["failure_evidence"]
            if evidence_records is not None and (
                not isinstance(evidence_records, Sequence)
                or isinstance(evidence_records, (str, bytes))
                or any(not isinstance(record, Mapping) for record in evidence_records)
            ):
                raise ValueError(f"Fitted source binding {index} failure evidence changed")
            bindings.append(
                FittedSourceBinding(
                    trading_date=date.fromisoformat(str(value["trading_date"])),
                    original_source=OriginalFittedPlan.from_document(
                        {
                            "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
                            "ticker": document["ticker"],
                            "source_revision": document["original_source_revision"],
                            "configuration_sha256": document["original_configuration_sha256"],
                            "sources": [value["original_source"]],
                            "evidence_sha256": canonical_sha256(
                                {
                                    "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
                                    "ticker": document["ticker"],
                                    "source_revision": document["original_source_revision"],
                                    "configuration_sha256": document[
                                        "original_configuration_sha256"
                                    ],
                                    "sources": [value["original_source"]],
                                }
                            ),
                        }
                    ).sources[0],
                    selected=_recovery_pointer(value["selected"], f"binding {index} selected"),
                    recovery_policy_id=None if policy is None else str(policy),
                    failure_evidence_sha256=None if evidence is None else str(evidence),
                    failure_evidence=(
                        None
                        if evidence_records is None
                        else tuple(cast(Mapping[str, Any], record) for record in evidence_records)
                    ),
                )
            )
        return cls(
            ticker=str(document["ticker"]),
            original_plan_sha256=str(document["original_plan_sha256"]),
            original_source_revision=str(document["original_source_revision"]),
            original_configuration_sha256=str(document["original_configuration_sha256"]),
            bindings=tuple(bindings),
        )


@version("1.0")
def feasible_candidate_recovery_policy_id() -> str:
    """Return the immutable identifier for the approved failed-day retry policy."""

    return str(feasible_candidate_recovery_policy_id.version)


@version("1.0", dependencies=[fit_smooth_sanos, publish_daily_frame, read_daily_frame])
def fitted_recovery_producer_version() -> str:
    return str(fitted_recovery_producer_version.version)


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        canonical_sha256,
        feasible_candidate_recovery_policy_id,
        fitted_recovery_producer_version,
    ],
)
def fitted_recovery_spec(
    *,
    ticker: str,
    trading_date: date,
    surface_config: DailySurfaceConfig,
    filtered: ArtifactPointer,
    original_expected: ArtifactPointer,
    original_plan_sha256: str,
    failure_evidence_sha256: str,
    failure_evidence: tuple[Mapping[str, Any], ...],
) -> ArtifactSpec:
    """Create a distinct recovery identity; never reuse the original namespace."""

    if canonical_sha256(failure_evidence) != failure_evidence_sha256:
        raise ValueError("Recovery failure evidence digest differs from embedded records")
    policy_id = feasible_candidate_recovery_policy_id()
    configuration = {
        "ticker": ticker,
        "trading_date": trading_date,
        "surface": surface_config,
        "original_expected": original_expected,
        "original_plan_sha256": original_plan_sha256,
        "failure_evidence_sha256": failure_evidence_sha256,
        "failure_evidence": failure_evidence,
        "recovery_policy_id": policy_id,
    }
    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=(
            f"widb/fitted-recovery/ticker-{ticker}/year-{trading_date.year}/"
            f"date-{trading_date.isoformat()}"
        ),
        configuration=configuration,
        schema_version=f"{FITTED_SCHEMA_VERSION}+{FITTED_RECOVERY_SCHEMA_VERSION}",
        producer_version=fitted_recovery_producer_version(),
        upstream=(filtered,),
    )


@version(
    "1.1",
    dependencies=[
        fitted_recovery_spec,
        fit_smooth_sanos,
        publish_daily_frame,
        read_daily_frame,
    ],
)
def build_fitted_recovery(
    store: ArtifactStore,
    *,
    ticker: str,
    trading_date: date,
    surface_config: DailySurfaceConfig,
    filtered: ArtifactPointer,
    original_expected: ArtifactPointer,
    original_plan_sha256: str,
    failure_evidence_sha256: str,
    failure_evidence: tuple[Mapping[str, Any], ...],
) -> ValidatedArtifact:
    """Retry exactly one absent failed target into a separate recovery artifact."""

    try:
        store.validate(original_expected)
    except ArtifactNotFoundError:
        pass
    else:
        raise ValueError(
            "Refusing recovery for successful original fitted target "
            f"{original_expected.artifact_id}"
        )
    spec = fitted_recovery_spec(
        ticker=ticker,
        trading_date=trading_date,
        surface_config=surface_config,
        filtered=filtered,
        original_expected=original_expected,
        original_plan_sha256=original_plan_sha256,
        failure_evidence_sha256=failure_evidence_sha256,
        failure_evidence=failure_evidence,
    )
    reuse = store.resolve(spec)
    if reuse is not None:
        return reuse.artifact
    filtered_artifact = store.validate(filtered, deep=True)
    frame = read_daily_frame(
        store,
        filtered_artifact,
        schema=FILTERED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )
    fitted = fit_smooth_sanos(frame, config=surface_config)
    return publish_daily_frame(
        store,
        spec,
        fitted,
        schema=FITTED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


@version("1.0")
def _read_regular_text(path: Path, *, label: str) -> str:
    """Read a bounded regular file without following a final-component symlink."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} path contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 256 * 1024 * 1024:
            raise ValueError(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return stream.read()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@version("1.0", dependencies=[_read_regular_text, canonical_value])
def _journal_failed_records(
    path: Path, *, ticker: str
) -> dict[date, tuple[Mapping[str, Any], ...]]:
    """Read exact failed unit keys from a durable original run journal."""

    failed: dict[date, list[Mapping[str, Any]]] = {}
    source = _read_regular_text(path, label="Failure evidence")
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failure evidence line {line_number} is not JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"Failure evidence line {line_number} is not an object")
        if value.get("event") != "unit_failed":
            continue
        key = value.get("key")
        if not isinstance(key, str) or not key.startswith(f"{ticker}/"):
            raise ValueError(f"Failure evidence line {line_number} has an invalid unit key")
        failed.setdefault(date.fromisoformat(key.rsplit("/", 1)[-1]), []).append(
            cast(Mapping[str, Any], canonical_value(value))
        )
    return {key: tuple(values) for key, values in failed.items()}


@version("1.0", dependencies=[ArtifactPointer, ValidatedArtifact, _recovery_pointer])
def _require_upstream(
    artifact: ValidatedArtifact,
    expected: tuple[ArtifactPointer, ...],
    *,
    label: str,
) -> None:
    values = artifact.manifest.get("upstream")
    if not isinstance(values, list) or len(values) != len(expected):
        raise ValueError(f"{label} has an incomplete upstream contract")
    observed = tuple(
        _recovery_pointer(value, f"{label} upstream", storage_class=pointer.storage_class)
        for value, pointer in zip(values, expected, strict=True)
    )
    if observed != tuple(sorted(expected)):
        raise ValueError(f"{label} differs from the frozen original chain")


@version("1.0", dependencies=[_require_upstream, canonical_value])
def _validate_original_filtered_chain(
    store: ArtifactStore,
    source: OriginalFittedSource,
    *,
    ticker: str,
    surface_config: DailySurfaceConfig,
) -> None:
    paired = store.validate(source.paired, deep=True)
    enriched = store.validate(source.enriched, deep=True)
    no_arb = store.validate(source.no_arb, deep=True)
    filtered = store.validate(source.filtered, deep=True)
    _require_upstream(
        paired,
        tuple(sorted(source.raw_roles.values())),
        label=f"{source.trading_date} paired",
    )
    _require_upstream(enriched, (source.paired,), label=f"{source.trading_date} enriched")
    _require_upstream(no_arb, (source.enriched,), label=f"{source.trading_date} no-arb")
    _require_upstream(filtered, (source.no_arb,), label=f"{source.trading_date} filtered")
    expected_configuration = canonical_value(
        {
            "stage": "filtered-day",
            "ticker": ticker,
            "trading_date": source.trading_date,
            "surface": {"options_filter": surface_config.options_filter},
        }
    )
    if (
        filtered.manifest.get("schema_version") != FILTERED_SCHEMA_VERSION
        or filtered.manifest.get("configuration") != expected_configuration
    ):
        raise ValueError(f"{source.trading_date}: frozen filtered source contract differs")


@version(
    "1.1",
    dependencies=[
        FittedSourceBinding,
        FittedSourceSelection,
        _journal_failed_records,
        _validate_original_filtered_chain,
        build_fitted_recovery,
        business_dates,
        canonical_sha256,
        feasible_candidate_recovery_policy_id,
        fitted_recovery_spec,
    ],
)
def recover_failed_fitted_sources(
    locations: StorageLocations,
    *,
    original_plan: OriginalFittedPlan,
    failure_evidence: Path,
    failed_dates: tuple[date, ...],
    surface_config: DailySurfaceConfig,
    include_half_days: bool,
    dry_run: bool = False,
) -> FittedSourceSelection:
    """Select all old successes and retry only explicitly failed absent targets."""

    requested_failures = frozenset(failed_dates)
    if not requested_failures or len(requested_failures) != len(failed_dates):
        raise ValueError("Recovery requires one or more unique --failed-date values")
    evidenced_failures = _journal_failed_records(failure_evidence, ticker=original_plan.ticker)
    if not requested_failures.issubset(evidenced_failures):
        missing = sorted(requested_failures - set(evidenced_failures))
        raise ValueError(
            "Requested recovery dates are not recorded failures: "
            + ", ".join(item.isoformat() for item in missing)
        )
    first = original_plan.sources[0].trading_date
    last = original_plan.sources[-1].trading_date
    expected_dates = business_dates(first, last, exclude_half_days=not include_half_days)
    if tuple(expected_dates) != tuple(item.trading_date for item in original_plan.sources):
        raise ValueError("Original fitted plan does not exactly cover the configured schedule")
    store = ArtifactStore(locations)
    bindings: list[FittedSourceBinding] = []
    for original in original_plan.sources:
        if original.trading_date not in requested_failures:
            bindings.append(
                FittedSourceBinding(
                    trading_date=original.trading_date,
                    original_source=original,
                    selected=original.expected_fitted,
                )
            )
            continue
        _validate_original_filtered_chain(
            store,
            original,
            ticker=original_plan.ticker,
            surface_config=surface_config,
        )
        evidence_records = evidenced_failures[original.trading_date]
        evidence_digest = canonical_sha256(evidence_records)
        spec = fitted_recovery_spec(
            ticker=original_plan.ticker,
            trading_date=original.trading_date,
            surface_config=surface_config,
            filtered=original.filtered,
            original_expected=original.expected_fitted,
            original_plan_sha256=original_plan.evidence_sha256,
            failure_evidence_sha256=evidence_digest,
            failure_evidence=evidence_records,
        )
        if dry_run:
            try:
                store.validate(original.expected_fitted)
            except ArtifactNotFoundError:
                pass
            else:
                raise ValueError(
                    f"{original.trading_date}: original fitted target succeeded; recovery refused"
                )
            selected = spec.pointer
        else:
            selected = build_fitted_recovery(
                store,
                ticker=original_plan.ticker,
                trading_date=original.trading_date,
                surface_config=surface_config,
                filtered=original.filtered,
                original_expected=original.expected_fitted,
                original_plan_sha256=original_plan.evidence_sha256,
                failure_evidence_sha256=evidence_digest,
                failure_evidence=evidence_records,
            ).pointer
        bindings.append(
            FittedSourceBinding(
                trading_date=original.trading_date,
                original_source=original,
                selected=selected,
                recovery_policy_id=feasible_candidate_recovery_policy_id(),
                failure_evidence_sha256=evidence_digest,
                failure_evidence=evidence_records,
            )
        )
    return FittedSourceSelection(
        ticker=original_plan.ticker,
        original_plan_sha256=original_plan.evidence_sha256,
        original_source_revision=original_plan.source_revision,
        original_configuration_sha256=original_plan.configuration_sha256,
        bindings=tuple(bindings),
    )


@version("1.0", dependencies=[_read_regular_text, OriginalFittedPlan])
def read_original_fitted_plan(path: Path) -> OriginalFittedPlan:
    value = json.loads(_read_regular_text(path, label="Original fitted plan"))
    if not isinstance(value, Mapping):
        raise ValueError("Original fitted plan must contain a JSON object")
    return OriginalFittedPlan.from_document(value)


def write_selection(path: Path, selection: FittedSourceSelection) -> None:
    """Write a portable exact selection without machine paths."""

    atomic_create_json(path, selection.as_dict())


__all__ = [
    "FITTED_SOURCE_SELECTION_SCHEMA",
    "FittedSourceBinding",
    "FittedSourceSelection",
    "OriginalFittedPlan",
    "OriginalFittedSource",
    "build_fitted_recovery",
    "feasible_candidate_recovery_policy_id",
    "fitted_recovery_spec",
    "read_original_fitted_plan",
    "recover_failed_fitted_sources",
    "write_selection",
]
