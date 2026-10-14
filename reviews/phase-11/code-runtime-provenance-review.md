# Phase 11 code runtime-provenance review

**Reviewer:** independent code agent (`/root/code`)

**Date:** 2026-10-14

**Scope:** Phase 11 runtime fingerprint, evidence-index binding, promotion
retention and adversarial tests; offline only

**Initial verdict:** **CHANGES REQUIRED**

**Final re-review verdict:** **PASS**

## Initial findings

1. The first writer used a lock leaf that could be replaced by a symlink and
   waited without a bounded failure mode. The fingerprint document itself was
   protected, but the auxiliary lock target and indefinite wait were not an
   acceptable production boundary.
2. Runtime validation interpreted the `sys.version_info` sequence with the
   wrong type positions. This admitted path-like release-level text and did not
   require the serial field to be an integer.

## Final re-review

Both findings are resolved.

- `write_runtime_fingerprint` no longer creates or acquires a lock path. It
  uses `atomic_create_json` hard-link create-if-absent publication and, on a
  collision, reloads the winner and requires canonical-identical content.
  Concurrent identical and conflicting writers remain covered. An adversarial
  legacy lock-symlink regression proves the redirected target is untouched.
- `version_info` now requires non-boolean integers at indices 0, 1, 2 and 4,
  and an allowlisted release level at index 3. Re-signed path-text and string
  serial documents are rejected.

The reviewer found no remaining issue in the narrow final re-review. The
coordinator subsequently ran the complete offline suite: 420 tests passed with
only the two expected explicit-wildcard-reuse warnings. No network, WRDS, GPU
or `quant_data` write was performed.

Signed: `/root/code` — independent Phase 11 code reviewer
