# Fresh SPX 2022--2025 production runbook

This is the human-operable Phase 11 protocol for rebuilding and designating the
canonical SANOS reference Dataset. It uses a new storage root, downloads every
required WRDS partition, and never consumes a `quant_data` Raw, Intermediate,
Dataset, or Experiment artifact. The legacy 891-date HDF5 collection and
hash-pinned `quant_data` code are read-only comparison oracles.

Do not start until the guarded Phase 10 pilot has passed code and quantitative
review. WRDS access is permitted only while the repository owner is present and
has just confirmed immediate 2FA readiness and that no other client is using
the account.

## 1. Freeze paths and source state

Run from the repository root. Choose a durable writable parent on the target
host and keep every generated file below its SANOS-owned `phase-11/` directory
and outside Git. `SANOS_RUNS_ROOT` is a shell-only runbook variable, not a SANOS
configuration key; replace the placeholder before running anything.

```bash
set -o pipefail
export SANOS_RUNS_ROOT=/path/to/writable/sanos
mkdir -p "$SANOS_RUNS_ROOT"
test -d "$SANOS_RUNS_ROOT" && test -w "$SANOS_RUNS_ROOT"
export PRODUCTION_ROOT="$SANOS_RUNS_ROOT/phase-11/data"
export PRODUCTION_RUN_ID=<UTC-date-and-operator-label>
export PRODUCTION_EVIDENCE="$SANOS_RUNS_ROOT/phase-11/evidence/$PRODUCTION_RUN_ID"
export LEGACY_MARKET_ROOT='<legacy_QUANTDATA_TRAINING_DATA>/dysanos/market/bid=0.011, nat,eth=10,bth=4,mps=1e-06,m ca250824/sm_0.25 6b930b59/SPX'
export LEGACY_CODE_ROOT=/home/hans/python/quant_data
mkdir -p "$PRODUCTION_EVIDENCE"
```

The reference configuration deliberately uses source generation `default-v1`.
Freshness comes from the new, previously nonexistent `PRODUCTION_ROOT`, not by
copying or refreshing an old generation. Before any command creates that root:

```bash
test ! -e "$PRODUCTION_ROOT"
git --git-dir=.git-real --work-tree=. rev-parse HEAD \
  > "$PRODUCTION_EVIDENCE/sanos-revision.txt"
git --git-dir=.git-real --work-tree=. status --porcelain=v1 \
  > "$PRODUCTION_EVIDENCE/sanos-status.txt"
test ! -s "$PRODUCTION_EVIDENCE/sanos-status.txt"
```

If the root exists, do not delete or reuse it as a fresh attempt. Inspect it,
preserve its evidence, and choose a separately named attempt. An interrupted
accepted run resumes its same root and immutable identities.

## 2. Offline preflight

```bash
env PYTHONPATH=src WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  2>&1 | tee "$PRODUCTION_EVIDENCE/offline-tests.log"

env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  doctor --strict-locations | tee "$PRODUCTION_EVIDENCE/doctor.jsonl"

gpuctl resources --json | tee "$PRODUCTION_EVIDENCE/resources-before.json"

env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --dry-run --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29 \
  --source-generation default-v1 \
  --max-wrds-connections 2 --retry-attempts 1 \
  | tee "$PRODUCTION_EVIDENCE/raw-plan.jsonl"
```

Every `gpuctl resources` check must be `OK`. A command failure, local fallback,
`WARN`, `STOP`, `UNAVAILABLE`, or `ERROR` blocks material work. The Raw plan
must report exactly 3,676 planned and missing artifacts, zero reused artifacts,
and three previous-session boundary artifacts. These are one security identity,
four dated tables for each of 918 sessions, and three boundary dependencies.
Fitted daily lineage later contains 3,674 unique Raw pointers: it consumes the
2021-12-31 spot boundary, while the boundary forward and zero-curve partitions
are acquisition-only.

Prove the empty root cannot silently access WRDS:

```bash
set +e
env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29 \
  --source-generation default-v1 --cache-only \
  --max-wrds-connections 2 \
  2>&1 | tee "$PRODUCTION_EVIDENCE/raw-cache-miss.jsonl"
cache_status=${PIPESTATUS[0]}
set -e
test "$cache_status" -eq 4
```

## 3. Immediate 2FA gate and fresh Raw download

Stop here. The owner must now confirm immediate WRDS 2FA readiness and the
absence of any other process, notebook, host, or scheduled job using the same
account. An earlier confirmation does not satisfy this gate. Then repeat the
resource check and require every result to be `OK`:

The standard `.pgpass` entry must already have been configured and
permission-checked using [wrds-runbook.md](wrds-runbook.md). The live command is
fail-closed and must never ask for a username or password. If credentials or 2FA
are rejected, stop; update `.pgpass` only from a private terminal and obtain a
new immediate-readiness confirmation before another attempt.

```bash
gpuctl resources --json | tee "$PRODUCTION_EVIDENCE/resources-pre-live.json"
```

Run exactly one live command, with two connections and one query attempt:

```bash
env -u WRDS_OFF sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29 \
  --source-generation default-v1 \
  --max-wrds-connections 2 --retry-attempts 1 --confirm-2fa-ready \
  2>&1 | tee "$PRODUCTION_EVIDENCE/raw-live.jsonl"
```

Do not retry authentication, 2FA, permission, connection-limit, or unknown
failures. Preserve completed immutable partitions and wait for a new immediate
readiness confirmation. An identical command later reuses validated partitions
and queries only misses.

Disable WRDS and validate the completed Raw set:

```bash
env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29 \
  --source-generation default-v1 --cache-only \
  --max-wrds-connections 2 \
  | tee "$PRODUCTION_EVIDENCE/raw-cache-validation.jsonl"
```

Require 3,676 reused, zero missing, zero downloaded, and
`network_accessed=false`.

## 4. Build daily frames and the internal Dataset

Every remaining build and comparison command runs with `WRDS_OFF=1`.

```bash
env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --dry-run --log-format json \
  intermediate build --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation default-v1 --max-workers 2 \
  | tee "$PRODUCTION_EVIDENCE/intermediate-plan.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  intermediate build --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation default-v1 --max-workers 2 \
  2>&1 | tee "$PRODUCTION_EVIDENCE/intermediate-build.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  dataset build --config configs/widb-spx-reference.json \
  --require-existing-intermediates --max-workers 2 \
  2>&1 | tee "$PRODUCTION_EVIDENCE/dataset-build.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  dataset validate --config configs/widb-spx-reference.json --deep-lineage \
  2>&1 | tee "$PRODUCTION_EVIDENCE/dataset-validation.jsonl"
```

Require 918 completed daily partitions, including the exact-schema typed-empty
2025-01-08 policy day. Record the Dataset identity printed by the CLI. Never
edit either downstream template by hand: step 6 binds both templates to this
exact immutable identity and records their canonical digests.

## 5. Full daily migration comparison

```bash
env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  reference compare-daily --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --source optionmetrics --source-generation default-v1 \
  --legacy-root "$LEGACY_MARKET_ROOT" \
  --expected-oracle-gaps-file configs/spx-2022-2025-oracle-gaps.json \
  --phase11-production-contract \
  --output "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  2>&1 | tee "$PRODUCTION_EVIDENCE/daily-comparison.jsonl"
```

The frozen contract requires 918 ordered dates: 891 exact or explicitly
approved common-date comparisons, 26 declared populated oracle gaps, one
policy-empty day, and no unexplained discrepancy. It also verifies the legacy
catalog population and exact daily identities/checksums. A vendor revision is
blocking until an owner-approved evidence document is supplied with
`--approvals`; fresh values are never modified to force parity.

## 6. Finalize configs and build downstream artifacts

The checked templates are portable policy documents. Bind them mechanically to
the actual internal Dataset and write the canonical final documents once:

```bash
env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  production finalize-configs \
  --dataset-name widb-spx-2022-2025-reference \
  --dataset-artifact-id <dataset-id> \
  --public-template configs/sanos-public-candidate-reference.json \
  --experiment-template configs/dysanos-daily-front-reference.json \
  --output-directory "$PRODUCTION_EVIDENCE/final-configs" \
  | tee "$PRODUCTION_EVIDENCE/config-finalization.jsonl"
```

`final-configs/bundle.json` binds both canonical configuration digests to the
internal Dataset pointer. The command refuses to overwrite its output. Build
and deeply validate the separate technical public candidate; it is not legally
public until the rights-gated workflow in
[public-datasets.md](public-datasets.md) is complete:

```bash
env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  dataset public build \
  --config "$PRODUCTION_EVIDENCE/final-configs/public-candidate.json" \
  2>&1 | tee "$PRODUCTION_EVIDENCE/public-candidate-build.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  dataset public validate \
  --config "$PRODUCTION_EVIDENCE/final-configs/public-candidate.json" \
  --deep-lineage \
  2>&1 | tee "$PRODUCTION_EVIDENCE/public-candidate-validation.jsonl"
```

This verifies the allowlist, leakage boundary, normalized experiment
capabilities, and exact source projection. Do not run `approval-template` or
`approve` without separate redistribution authority.

Build, deeply validate, and compare the disposable Experiment:

```bash
env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  experiment build \
  --config "$PRODUCTION_EVIDENCE/final-configs/experiment.json" \
  --max-workers 2 \
  2>&1 | tee "$PRODUCTION_EVIDENCE/experiment-build.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  experiment validate \
  --config "$PRODUCTION_EVIDENCE/final-configs/experiment.json" \
  --deep-lineage \
  2>&1 | tee "$PRODUCTION_EVIDENCE/experiment-validation.jsonl"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  reference compare-experiment \
  --config "$PRODUCTION_EVIDENCE/final-configs/experiment.json" \
  --legacy-root "$LEGACY_MARKET_ROOT" \
  --legacy-code-root "$LEGACY_CODE_ROOT" \
  --phase11-production-contract --consumer-batch-size 32 \
  --output "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  2>&1 | tee "$PRODUCTION_EVIDENCE/experiment-comparison.jsonl"
```

The Experiment contract requires 917 populated fresh days, the ordered 891
common days, the exact 26 fresh-only dates, logical field/dtype/shape parity,
and exact float32 outputs from the transitively hash-pinned day and batch
consumer. The
legacy `market_uid` is expected to change because it identifies the fresh
Dataset; only that identity is rebased when checking the recovered training
fingerprint.

## 7. Machine-check the evidence boundary

Build the immutable evidence index before requesting either review:

```bash
env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  production evidence-index \
  --run-id "$PRODUCTION_RUN_ID" \
  --source-revision "$(tr -d '\n' < "$PRODUCTION_EVIDENCE/sanos-revision.txt")" \
  --raw-preflight "$PRODUCTION_EVIDENCE/raw-plan.jsonl" \
  --raw-validation "$PRODUCTION_EVIDENCE/raw-cache-validation.jsonl" \
  --dataset-validation "$PRODUCTION_EVIDENCE/dataset-validation.jsonl" \
  --daily-report "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  --downstream-bundle "$PRODUCTION_EVIDENCE/final-configs/bundle.json" \
  --public-validation "$PRODUCTION_EVIDENCE/public-candidate-validation.jsonl" \
  --experiment-validation "$PRODUCTION_EVIDENCE/experiment-validation.jsonl" \
  --experiment-report "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  --output "$PRODUCTION_EVIDENCE/evidence-index.json" \
  | tee "$PRODUCTION_EVIDENCE/evidence-index.jsonl"
```

This command requires the initial plan to contain 3,676 misses and no reuse,
the final Raw validation to contain 3,676 unique complete artifacts, the exact
three boundary dependencies, deep validation of all three materializations,
the exact 26-gap/one-typed-empty comparison contract, and consistent Dataset
pointers throughout. It hashes every input and prints `evidence_id`. Repeating
the command shallowly reuses only byte-identical evidence; any changed input is
blocking.

## 8. Independent reviews and canonical designation

Code and quantitative reviewers independently inspect the complete indexed
evidence, including `evidence-index.json` and its named files.
Only after both have approved it should the operator materialize attestations:

```bash
env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  dataset reference review-template \
  --role code --reviewer <code-reviewer> --review-date <YYYY-MM-DD> \
  --review-reference <durable-code-review-reference> \
  --dataset-name widb-spx-2022-2025-reference \
  --dataset-artifact-id <dataset-id> \
  --daily-report "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  --experiment-report "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  --evidence-index "$PRODUCTION_EVIDENCE/evidence-index.json" \
  --output "$PRODUCTION_EVIDENCE/code-review.json"

env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  dataset reference review-template \
  --role quantitative --reviewer <quant-reviewer> --review-date <YYYY-MM-DD> \
  --review-reference <durable-quant-review-reference> \
  --dataset-name widb-spx-2022-2025-reference \
  --dataset-artifact-id <dataset-id> \
  --daily-report "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  --experiment-report "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  --evidence-index "$PRODUCTION_EVIDENCE/evidence-index.json" \
  --output "$PRODUCTION_EVIDENCE/quantitative-review.json"

env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  dataset reference promotion-template \
  --name spx-2022-2025-reference \
  --dataset-name widb-spx-2022-2025-reference \
  --dataset-artifact-id <dataset-id> \
  --daily-report "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  --experiment-report "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  --evidence-index "$PRODUCTION_EVIDENCE/evidence-index.json" \
  --code-review "$PRODUCTION_EVIDENCE/code-review.json" \
  --quantitative-review "$PRODUCTION_EVIDENCE/quantitative-review.json" \
  --output "$PRODUCTION_EVIDENCE/promotion-config.json"

env WRDS_OFF=1 sanos -v --root "$PRODUCTION_ROOT" --log-format json \
  dataset reference promote \
  --config "$PRODUCTION_EVIDENCE/promotion-config.json" \
  --daily-report "$PRODUCTION_EVIDENCE/daily-comparison.json" \
  --experiment-report "$PRODUCTION_EVIDENCE/experiment-comparison.json" \
  --evidence-index "$PRODUCTION_EVIDENCE/evidence-index.json" \
  --code-review "$PRODUCTION_EVIDENCE/code-review.json" \
  --quantitative-review "$PRODUCTION_EVIDENCE/quantitative-review.json" \
  2>&1 | tee "$PRODUCTION_EVIDENCE/promotion.jsonl"
```

Promotion revalidates Dataset lineage and the Experiment, binds all 918 Dataset
partitions to the fitted artifacts in the daily report, and atomically publishes
a self-contained immutable designation. It does not alter the Dataset or grant
redistribution rights.

Validate by exact designation ID. Shallow validation needs no upstream
artifacts; `--deep-lineage` deliberately reopens them:

```bash
env WRDS_OFF=1 sanos --root "$PRODUCTION_ROOT" --log-format json \
  dataset reference validate --name spx-2022-2025-reference \
  --artifact-id <promotion-id>
```

`--version '*'` is reuse-only: it can reuse a complete designation while
ignoring producer-code version, but it cannot build, replace, or promote a
missing artifact.

## Restart rule

Raw partitions, daily stages, Datasets, public candidates, Experiments, and the
canonical designation are output-first. After an interruption, retain the same
`PRODUCTION_RUN_ID`, root, `default-v1` source generation, finalized configs,
and command arguments, verify the resource gate, and repeat the interrupted
command. Complete outputs are validated and reused. If a live Raw retry is
still required, stop for a new immediate owner 2FA confirmation; never infer it
from the earlier attempt. Do not delete a restart workspace, edit a config, or
choose a new generation merely to bypass a failure. Preserve every attempt log
in the evidence directory and give it a distinct suffix when needed.

## 9. Evidence retention and safe cleanup

Before cleanup, inspect evidence without printing credentials, record
checksums, and repeat the resource gate:

```bash
rg -n -i 'password|passwd|one[-_ ]time|otp|credential|wrds[_-]?username|dsn=' \
  "$PRODUCTION_EVIDENCE"
find "$PRODUCTION_EVIDENCE" -type f ! -name checksums.sha256 -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$PRODUCTION_EVIDENCE/checksums.sha256"
gpuctl resources --json | tee "$PRODUCTION_EVIDENCE/resources-after.json"
```

Keep Raw and the canonical Dataset. Intermediate artifacts are rebuildable and
may be removed only by an explicit artifact-scoped cleanup after promotion. The
HDF5 Experiment is reproducible and may be discarded after its comparison and
promotion evidence are complete. Never blanket-delete a shared root or remove
files used by another process.

For general installation, exit-code, troubleshooting, and quarantine guidance,
see the [human operator guide](operator-guide.md).
