"""Read-only adapters for the pinned legacy parity references."""

from sanos.reference.compare import (
    APPROVAL_SCHEMA,
    ORACLE_GAP_DECLARATION_SCHEMA,
    REFERENCE_REPORT_SCHEMA,
    DailyComparisonRequest,
    DailyLegacyComparisonTolerances,
    ReferenceComparisonError,
    compare_daily_artifacts,
    compare_populated_day,
    load_comparison_approvals,
    load_oracle_gap_declaration,
    load_reference_catalog,
    repository_source_revision,
    write_comparison_report,
)
from sanos.reference.experiment_compare import (
    EXPERIMENT_REFERENCE_REPORT_SCHEMA,
    ExperimentComparisonRequest,
    ExperimentReferenceTolerances,
    compare_experiment_artifact,
    write_experiment_comparison_report,
)
from sanos.reference.legacy_market import (
    LegacyDayRecord,
    LegacyMarketCatalog,
    LegacyMarketReference,
    ReferenceFormatError,
    catalog_legacy_market,
    read_legacy_pytree_hdf5,
)

__all__ = [
    "APPROVAL_SCHEMA",
    "ORACLE_GAP_DECLARATION_SCHEMA",
    "REFERENCE_REPORT_SCHEMA",
    "DailyComparisonRequest",
    "DailyLegacyComparisonTolerances",
    "EXPERIMENT_REFERENCE_REPORT_SCHEMA",
    "ExperimentComparisonRequest",
    "ExperimentReferenceTolerances",
    "LegacyDayRecord",
    "LegacyMarketCatalog",
    "LegacyMarketReference",
    "ReferenceComparisonError",
    "ReferenceFormatError",
    "catalog_legacy_market",
    "compare_daily_artifacts",
    "compare_experiment_artifact",
    "compare_populated_day",
    "load_comparison_approvals",
    "load_oracle_gap_declaration",
    "load_reference_catalog",
    "read_legacy_pytree_hdf5",
    "repository_source_revision",
    "write_comparison_report",
    "write_experiment_comparison_report",
]
