"""Optional JAX fitter compatibility gate.

The pinned implementation contains the two output-changing defects recorded in
CD-015. Shared JAX transition and gradient kernels remain available through
``sanos.model.jax_backbone``; the full optimizer is deliberately unavailable
until a repaired semantic version is approved.
"""

from __future__ import annotations

from typing import Any, NoReturn

from cdxcore.version import version

from sanos.model.fitter import SanosFitter
from sanos.model.jax_backbone import BackboneJax
from sanos.model.tridiag import solve_sanos_tridiagonal


@version(
    "0.1-blocked-cd015",
    dependencies=[SanosFitter, BackboneJax, solve_sanos_tridiagonal],
)
def fit_dlvsqs(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject the defective reference optimizer until CD-015 is approved."""

    del args, kwargs
    raise NotImplementedError(
        "The optional JAX fitter is unavailable pending compatibility decision CD-015; "
        "use SanosFitter for the authoritative LP implementation."
    )


__all__ = ["fit_dlvsqs"]
