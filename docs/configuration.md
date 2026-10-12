# Configuration and data locations

SANOS has exactly four storage classes: Raw, Intermediate, Dataset, and
Experiment. Location resolution is deliberately separate from directory
creation, so `sanos doctor`, `sanos status`, and `sanos config show` are safe,
offline diagnostics.

## Precedence

Each location is resolved independently. The first applicable value wins:

1. a direct CLI option (`--raw-root`, `--intermediate-root`,
   `--datasets-root`, or `--experiments-root`), then `--root/<class>`;
2. its specific canonical environment variable;
3. `[locations]` in the repository's nearest `.sanos.toml`;
4. `[locations]` in `~/.config/quant_data/sanos.toml`;
5. the class directory beneath `QUANT_DATA_ROOT`.

An explicitly supplied `--repository-config` or `--user-config` must exist and
be valid. Relative TOML paths are relative to the TOML file. A repository
`.sanos.toml` is machine-local and ignored by Git; checked-in research
configuration must not contain machine paths or credentials.

Relative CLI and environment paths are resolved against the command's current
working directory. Diagnostic commands report the effective absolute path and
winning source.

The canonical variables are:

| Class | Canonical variable | Legacy fallback |
|---|---|---|
| Raw | `QUANT_DATA_RAW` | `QUANTDATA_RAW_DATA` |
| Intermediate | `QUANT_DATA_INTERMEDIATE` | `QUANTDATA_PROCESSING_DATA` |
| Dataset | `QUANT_DATA_DATASETS` | `QUANTDATA_TRAINING_DATA` |
| Experiment | `QUANT_DATA_EXPERIMENTS` | `QUANTDATA_EXPERIMENTS` |

Legacy variables are read-only compatibility aliases. A canonical specific
variable wins over its legacy counterpart, and a conflict is reported. The
general legacy `QUANTDATA_ROOT` maps to the historical child names `raw/`,
`processing/`, `training/`, and `experiments/`; canonical `QUANT_DATA_ROOT`
uses `raw/`, `intermediate/`, `datasets/`, and `experiments/`.

Example local file:

```toml
[locations]
root = "/data/quant_data"
# A class-specific entry can override the root default.
datasets = "/data/published/quant_data"
```

Inspect without writing:

```bash
sanos config show
sanos --log-format json config show --require-complete
sanos doctor --strict-locations
```

Configuration values used in artifact identity are serialized as compact,
sorted canonical JSON. Non-finite floating-point values, naive datetimes,
non-string mapping keys, bytes, sets, and unsupported objects are rejected.
Deterministic configuration and transformation APIs expose `@version` metadata
from `cdxcore.version`.
