"""OptionMetrics source schemas, table resolution, and safe SQL generation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from cdxcore.version import version

from sanos.wrds.plan import RawPartition, RawTable

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


@version("1.0", dependencies=[RawTable])
@dataclass(frozen=True, slots=True)
class RawField:
    """One vendor field retained without analytical transformation."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.name) is None:
            raise ValueError(f"Unsafe raw field name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"Raw field {self.name!r} requires a description")


@version("1.0", dependencies=[RawField])
@dataclass(frozen=True, slots=True)
class RawTableSpec:
    """Logical source-table contract independent of account table layout."""

    table: RawTable
    logical_name: str
    candidates: tuple[str, ...]
    fields: tuple[RawField, ...]
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", RawTable(self.table))
        if _IDENTIFIER.fullmatch(self.logical_name) is None:
            raise ValueError(f"Unsafe logical table name: {self.logical_name!r}")
        if not self.candidates or not self.fields:
            raise ValueError("A raw table requires candidates and fields")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError(f"Duplicate fields in {self.table.value}")

    @property
    def columns(self) -> tuple[str, ...]:
        """Return selected columns in stable vendor order."""

        return tuple(field.name for field in self.fields)


def _field(name: str, description: str) -> RawField:
    return RawField(name, description)


RAW_TABLE_SPECS = {
    RawTable.SECURITY: RawTableSpec(
        RawTable.SECURITY,
        "securd",
        ("securd", "securd1"),
        (
            _field("ticker", "OptionMetrics ticker used to resolve the requested security."),
            _field("secid", "OptionMetrics security identifier."),
        ),
        "OptionMetrics security identity for the requested ticker.",
    ),
    RawTable.OPTIONS: RawTableSpec(
        RawTable.OPTIONS,
        "opprcd",
        ("opprcd{year}", "opprcd"),
        tuple(
            _field(name, description)
            for name, description in (
                ("secid", "OptionMetrics underlying security identifier."),
                ("date", "Quote observation date."),
                ("symbol", "Vendor option symbol."),
                ("optionid", "Vendor option identifier."),
                ("am_settlement", "Vendor AM-settlement flag."),
                ("cp_flag", "Call/put flag."),
                (
                    "strike_price",
                    "Vendor integer strike in thousandths of one currency/index unit; "
                    "the compatibility stage later divides by 1000.",
                ),
                ("exdate", "Contract expiration date."),
                ("best_bid", "End-of-day best bid in currency/index units per option."),
                ("best_offer", "End-of-day best offer in currency/index units per option."),
                ("volume", "Reported daily contract volume."),
                ("open_interest", "Reported open interest."),
                ("impl_volatility", "Vendor implied volatility."),
                ("ss_flag", "Vendor special-settlement flag."),
                ("delta", "Vendor delta."),
                ("gamma", "Vendor gamma."),
                ("vega", "Vendor vega."),
                ("theta", "Vendor theta."),
                ("expiry_indicator", "Vendor expiry-cycle indicator."),
                ("contract_size", "Contract multiplier."),
            )
        ),
        "Daily OptionMetrics option price records, retained in vendor units.",
    ),
    RawTable.SPOT: RawTableSpec(
        RawTable.SPOT,
        "secprd",
        ("secprd", "secprd{year}"),
        (
            _field("secid", "OptionMetrics underlying security identifier."),
            _field("date", "Underlying close observation date."),
            _field("close", "Vendor end-of-day underlying close in index/currency units."),
        ),
        "Daily OptionMetrics underlying close.",
    ),
    RawTable.FORWARDS: RawTableSpec(
        RawTable.FORWARDS,
        "fwdprd",
        ("fwdprd{year}", "fwdprd"),
        (
            _field("secid", "OptionMetrics underlying security identifier."),
            _field("date", "Forward observation date."),
            _field("expiration", "Vendor forward expiration date."),
            _field("amsettlement", "Vendor forward AM-settlement flag."),
            _field("forwardprice", "Vendor forward level in index/currency units."),
        ),
        "Daily OptionMetrics implied forward curve.",
    ),
    RawTable.ZERO_CURVE: RawTableSpec(
        RawTable.ZERO_CURVE,
        "zerocd",
        ("zerocd",),
        (
            _field("date", "Zero-curve observation date."),
            _field("days", "Calendar-day curve tenor."),
            _field("rate", "Continuously compounded ACT/365 rate in percent."),
        ),
        "OptionMetrics USD zero-coupon curve.",
    ),
}


@version("1.0", dependencies=[RawTableSpec])
def raw_table_spec(table: RawTable) -> RawTableSpec:
    """Return the versioned source contract for a logical raw table."""

    return RAW_TABLE_SPECS[RawTable(table)]


@version("1.0")
@dataclass(frozen=True, slots=True)
class OptionMetricsCatalog:
    """Resolved account-visible OptionMetrics library and table names."""

    library: str
    tables: frozenset[str]

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.library) is None:
            raise ValueError(f"Unsafe WRDS library name: {self.library!r}")
        if not self.tables or any(_IDENTIFIER.fullmatch(name) is None for name in self.tables):
            raise ValueError("WRDS catalog contains no tables or an unsafe table name")


@version("1.0", dependencies=[OptionMetricsCatalog])
def choose_optionmetrics_catalog(
    libraries: tuple[str, ...] | list[str],
    tables_by_library: dict[str, tuple[str, ...] | list[str]],
) -> OptionMetricsCatalog:
    """Select the same preferred OptionMetrics library as the reference code."""

    available = set(libraries)
    for library in ("optionm_all", "optionm"):
        if library in available:
            tables = tables_by_library.get(library)
            if tables is None:
                raise RuntimeError(f"WRDS did not return the table catalog for {library}")
            return OptionMetricsCatalog(library, frozenset(tables))
    raise RuntimeError(
        "WRDS libraries do not include optionm_all or optionm; verify OptionMetrics access"
    )


@version("1.0", dependencies=[raw_table_spec, OptionMetricsCatalog, RawPartition])
def resolve_physical_table(partition: RawPartition, catalog: OptionMetricsCatalog) -> str:
    """Resolve one logical table using the reference candidate precedence."""

    spec = raw_table_spec(partition.table)
    for candidate in spec.candidates:
        if "{year}" in candidate:
            if partition.observation_date is None:
                continue
            candidate = candidate.format(year=partition.observation_date.year)
        if candidate in catalog.tables:
            return f"{catalog.library}.{candidate}"
    expanded = [
        candidate.format(year=partition.observation_date.year)
        if "{year}" in candidate and partition.observation_date is not None
        else candidate
        for candidate in spec.candidates
    ]
    raise RuntimeError(
        f"Cannot resolve OptionMetrics {partition.table.value}; tried {expanded} "
        f"in {catalog.library}"
    )


@version("1.0", dependencies=[RawPartition])
@dataclass(frozen=True, slots=True)
class RawQuery:
    """A safe, fully resolved query and its non-secret provenance."""

    partition: RawPartition
    physical_table: str
    sql: str

    @property
    def sha256(self) -> str:
        """Return a stable digest of the exact submitted SQL."""

        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


@version("1.0")
def _date_literal(value: date | None) -> str:
    if value is None:
        raise ValueError("A dated raw query requires an observation date")
    return value.isoformat()


@version(
    "1.0",
    dependencies=[RawQuery, _date_literal, raw_table_spec, resolve_physical_table],
)
def build_raw_query(
    partition: RawPartition,
    catalog: OptionMetricsCatalog,
    *,
    secid: int | None,
) -> RawQuery:
    """Build SQL only from validated identifiers and typed literals."""

    physical = resolve_physical_table(partition, catalog)
    spec = raw_table_spec(partition.table)
    columns = ", ".join(spec.columns)
    if partition.table is RawTable.SECURITY:
        where = f"ticker = '{partition.ticker}'"
    else:
        observation_date = _date_literal(partition.observation_date)
        if partition.table is RawTable.ZERO_CURVE:
            where = f"date = '{observation_date}'"
        else:
            if isinstance(secid, bool) or not isinstance(secid, int) or secid <= 0:
                raise ValueError(f"A positive secid is required for {partition.table.value}")
            where = f"secid = {secid} AND date = '{observation_date}'"
    order = " ORDER BY days" if partition.table is RawTable.ZERO_CURVE else ""
    sql = f"SELECT {columns} FROM {physical} WHERE {where}{order}"
    return RawQuery(partition=partition, physical_table=physical, sql=sql)


@version("1.0", dependencies=[RawQuery, raw_table_spec])
def render_source_readme(
    query: RawQuery,
    *,
    row_count: int,
    persisted_schema: Mapping[str, str],
) -> str:
    """Render a concise human- and agent-readable raw partition contract."""

    spec = raw_table_spec(query.partition.table)
    observed = (
        "not applicable"
        if query.partition.observation_date is None
        else query.partition.observation_date.isoformat()
    )
    expected = set(spec.columns)
    if set(persisted_schema) != expected:
        raise ValueError("Persisted schema must describe every selected raw field exactly")
    fields = "\n".join(
        f"- `{field.name}` (`{persisted_schema[field.name]}`): {field.description}"
        for field in spec.fields
    )
    return (
        f"# WRDS OptionMetrics {query.partition.table.value}\n\n"
        f"{spec.description}\n\n"
        f"- Logical source: `wrds/optionmetrics/{spec.logical_name}`\n"
        f"- Physical source table: `{query.physical_table}`\n"
        f"- Ticker: `{query.partition.ticker}`\n"
        f"- Observation date: `{observed}`\n"
        f"- Rows: `{row_count}`\n"
        f"- Empty vendor result: `{'yes' if row_count == 0 else 'no'}`\n"
        f"- SQL SHA-256: `{query.sha256}`\n"
        "- Transformation: SQL projection and Parquet encoding only; vendor numeric "
        "units are retained.\n\n"
        "## Fields\n\n"
        f"{fields}\n"
    )


__all__ = [
    "RAW_TABLE_SPECS",
    "OptionMetricsCatalog",
    "RawField",
    "RawQuery",
    "RawTableSpec",
    "build_raw_query",
    "choose_optionmetrics_catalog",
    "raw_table_spec",
    "render_source_readme",
    "resolve_physical_table",
]
