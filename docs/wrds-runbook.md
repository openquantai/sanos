# WRDS runbook

WRDS access is not implemented in Phase 0. Ordinary tests and the current CLI
must not open a network connection.

Before live access is enabled, the implementation must provide an offline plan,
explicit user confirmation of 2FA readiness, an account-wide limit of at most
four connections, atomic resumable partitions, complete boundary dependencies,
and structured progress. See `AGENTS.md` and `PROGRAM_PLAN.md` for the binding
contract.

