"""Read-only adapters for the pinned legacy parity references."""

from sanos.reference.legacy_market import (
    LegacyDayRecord,
    LegacyMarketCatalog,
    LegacyMarketReference,
    ReferenceFormatError,
    catalog_legacy_market,
    read_legacy_pytree_hdf5,
)

__all__ = [
    "LegacyDayRecord",
    "LegacyMarketCatalog",
    "LegacyMarketReference",
    "ReferenceFormatError",
    "catalog_legacy_market",
    "read_legacy_pytree_hdf5",
]
