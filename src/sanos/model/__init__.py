"""Standalone SANOS, DLV, and no-arbitrage numerical implementation."""

from __future__ import annotations

from cdxcore.version import version

IMPOSE_FLOOR_ON_DENSITY = False


@version("1.0")
def density_floor_enabled() -> bool:
    """Return the frozen reference density-floor policy."""

    return IMPOSE_FLOOR_ON_DENSITY


from sanos.model.arbfree import ArbFreePricer, MkArbFree  # noqa: E402
from sanos.model.fitter import SanosFitter  # noqa: E402
from sanos.model.tools import (  # noqa: E402
    SanosLPSolver,
    SanosPricer,
    make_variances,
    repair_density_nonnegative,
    warn_and_fix_density,
)
from sanos.model.tridiag import solve_sanos_tridiagonal  # noqa: E402

__all__ = [
    "ArbFreePricer",
    "IMPOSE_FLOOR_ON_DENSITY",
    "MkArbFree",
    "SanosFitter",
    "SanosLPSolver",
    "SanosPricer",
    "density_floor_enabled",
    "make_variances",
    "repair_density_nonnegative",
    "solve_sanos_tridiagonal",
    "warn_and_fix_density",
]
