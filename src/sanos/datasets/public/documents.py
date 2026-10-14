"""Deterministic human-facing documents for public Dataset artifacts."""

from __future__ import annotations

from cdxcore.version import version

from sanos.datasets.public.config import PublicProjectionConfig, PublicReleaseConfig


@version("1.1", dependencies=[PublicProjectionConfig])
def candidate_readme(config: PublicProjectionConfig, artifact_id: str) -> str:
    """Return the exact candidate README payload."""

    return f"""# SANOS public research candidate: {config.name}

This immutable Dataset is a deliberately lossy, technically validated
`public_candidate`. It is **not approved for redistribution**. A public release
requires a separate license, rights basis, approver, approval date, and durable
approval reference.

The single `data.parquet` file contains one row per populated retained trading
date. It supports normalized smooth-SANOS and discrete-DLV pricing, derived
model ATM volatility, forward-adjusted return dynamics, and normalized spread
experiments. See `schema.json` and `capabilities.json` for exact units and
limitations. The finite DLV state is the stored canonical projection of smooth
SANOS. Projection equivalence is validated only at retained expiries within
inclusive NMS [-3,2], not at arbitrary times.

It intentionally excludes absolute spot/forward/strike/price levels, raw
bids/asks/mids, identifiers, symbols, vendor Greeks/IVs, volume, open interest,
liquidity diagnostics, per-book model prices, and contract replay data.

Identity: `{artifact_id}`

Validate without WRDS or the internal source:

```console
sanos dataset public validate --name {config.name} --artifact-id {artifact_id}
```
"""


@version("1.1", dependencies=[PublicReleaseConfig])
def release_readme(config: PublicReleaseConfig, artifact_id: str) -> str:
    """Return the exact approved-release README payload."""

    approval = config.approval
    continuation = "\\"
    return f"""# SANOS public research dataset: {config.projection.name}

This is a distinct immutable `public` materialization approved under
`{approval.release_license}`. Redistribution basis: {approval.redistribution_basis}

Approver: {approval.approver}

Approval date: {approval.approval_date.isoformat()}

Approval reference: {approval.approval_reference}

The single `data.parquet` file contains the exact logical content approved in
candidate `{approval.candidate_dataset.artifact_id}`. It intentionally excludes
absolute market levels, vendor identifiers, raw quotes and unsupported replay
fields. See `schema.json`, `capabilities.json`, `approval.json`, and `LICENSE.txt`.

Identity: `{artifact_id}`

```console
sanos dataset public validate --distribution public {continuation}
  --name {config.projection.name} --artifact-id {artifact_id}
```
"""


__all__ = ["candidate_readme", "release_readme"]
