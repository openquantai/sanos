# Architecture

The stable interfaces are the `sanos` CLI and documented artifact contracts.
The implementation uses deterministic daily stages and a small local executor;
it does not use Redun initially. See `PROGRAM_PLAN.md` for the reviewed design,
phase gates, storage lifecycle, and output-first reuse semantics.

