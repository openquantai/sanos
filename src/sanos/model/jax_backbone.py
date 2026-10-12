"""Minimal optional JAX numerical backbone for shared SANOS kernels."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.version import version

from sanos.model.backbone import Backbone


@version("1.0", dependencies=[Backbone])
class BackboneJax(Backbone):
    """JAX adapter used by the shared tridiagonal transition kernel."""

    def __init__(self, device: jax.Device | str = "cpu", *, debug: bool = False) -> None:
        if isinstance(device, str):
            matches = jax.devices(device)
            if not matches:
                raise RuntimeError(f"No JAX {device!r} device is available")
            device = matches[0]
        self.device = device
        self.debug = debug
        self.float32 = jnp.float32
        self.float64 = jnp.float64
        self.int32 = jnp.int32
        self.bool_ = jnp.bool_
        self.Array = cast(Any, jax.Array)

    def to_array(self, data: Any, dtype: type | None = None) -> jax.Array:
        """Place data on the configured device with an optional dtype."""

        value = np.asarray(data, dtype=dtype) if not isinstance(data, jax.Array) else data
        return jax.device_put(value, self.device)

    def to_numpy(self, array: jax.Array) -> np.ndarray:
        """Copy a JAX array to NumPy."""

        return np.asarray(jax.device_get(array))

    def stop_gradient(self, array: jax.Array) -> jax.Array:
        """Stop gradient propagation through an array."""

        return jax.lax.stop_gradient(array)

    def stack(
        self,
        arrays: list[jax.Array],
        axis: int = 0,
        dtype: type | None = None,
    ) -> jax.Array:
        """Stack arrays using JAX while preserving the requested dtype."""

        result = jnp.stack(arrays, axis=axis)
        return result if dtype is None else result.astype(dtype)

    def __getattr__(self, name: str) -> Any:
        if hasattr(jnp, name):
            return getattr(jnp, name)
        raise AttributeError(f"BackboneJax has no attribute {name}")
