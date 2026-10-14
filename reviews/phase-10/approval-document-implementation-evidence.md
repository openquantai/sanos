# Phase 10 exact approval-document implementation evidence

Date: 2026-10-14
Baseline: `f0ad8b0bfc50a1dd5327ee36db7fdc3f33f66b0a`
Network/WRDS access: none (`WRDS_OFF=1`)

## Scope

The repository now provides `sanos reference approval-document` for the narrow
case where the owner has already approved exact, independently reviewed daily
migration discrepancies. This command does not make or infer that decision and
does not grant redistribution rights.

The command requires:

- a canonical blocked comparison report produced offline without building;
- the physical legacy root and complete configured SANOS storage locations;
- one or more exact unresolved populated dates;
- bounded non-secret explanation, approver, approval date, and durable decision
  reference metadata;
- an explicit owner-attestation flag; and
- a new output path outside both the legacy root and Git working tree.

## Evidence binding

Before constructing a record, the implementation:

1. validates the closed report schema, report ID, request/configuration,
   coverage, classification counts, passing contracts/checks, sorted unique
   dates, and absence of a prior approval overlay;
2. reconstructs the exact daily plan from the report configuration;
3. resolves and integrity-checks the planned fresh fitted artifact, reads its
   Parquet payload, and runs the complete fitted-partition validator;
4. checks the physical legacy HDF5 path, byte size, SHA-256, catalog binding,
   population, and decodability;
5. reruns `compare_populated_day` under the report's exact tolerance document
   and requires canonical equality with the retained daily comparison; and
6. copies only the exact sorted mismatch fields and evidence digests into the
   established approval schema.

The output preflight rejects symlinked components, missing parents, existing
destinations, and destinations under the legacy or repository roots. Both a
normal invocation and `--dry-run` perform the complete evidence and destination
preflight. Publication is atomic create-if-absent; two concurrent writers cannot
replace one another.

Routine structured output reports only the record count, destination when
written, dry-run/mutation facts, technical authority class, and
`redistribution_approval=false`. It excludes the explanation, decision
reference, field names, and evidence digests.

## Adversarial coverage

`tests/test_reference_approvals.py` covers:

- resolved `@version` closures and strict output-document validation;
- a partial overlay remaining blocked and an exact complete overlay passing;
- re-signed inconsistent counts, duplicate dates, failed checks, widened
  mismatch fields, missing legacy evidence, stale legacy bytes, and stale
  fresh Parquet bytes;
- report, parent, and output symlinks;
- concurrent atomic create/no-replace behavior;
- mandatory owner attestation, zero-write dry-run, existing-output rejection,
  bounded routine logs, and no redistribution authority; and
- exact CLI integration through the ordinary configured SANOS command tree.

The focused offline verification command was:

```text
env WRDS_OFF=1 TMPDIR=/tmp/sanos/pytest-temp PYTHONDONTWRITEBYTECODE=1 \
  /home/hans/vqdata/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_reference_approvals.py tests/test_reference_compare.py tests/test_cli.py
```

Result: **83 passed** in 79.29 seconds.

The complete offline repository suite was then run under the same offline and
project-scoped temporary-file controls:

```text
env WRDS_OFF=1 TMPDIR=/tmp/sanos/pytest-temp PYTHONDONTWRITEBYTECODE=1 \
  /home/hans/vqdata/bin/python -m pytest -q -p no:cacheprovider
```

Result: **429 passed**, with two expected `version="*"` reuse-warning tests, in
654.53 seconds.

## Gate retained

No approval document was generated for the real pilot. The command cannot
authenticate the named owner or prove that external review occurred. The
retained report therefore remains blocked on explicit owner approval of exactly
2025-01-02 and 2025-01-03. After such a decision is recorded, the comparator
must be rerun and its passing report must receive both reviews before Phase 11.
