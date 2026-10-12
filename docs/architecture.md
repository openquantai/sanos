# Architecture

The stable interfaces are the `sanos` CLI and documented artifact contracts.
The implementation uses deterministic daily stages and a small local executor;
it does not use Redun initially. See `PROGRAM_PLAN.md` for the reviewed design,
phase gates, storage lifecycle, and output-first reuse semantics.

## Foundation boundaries

Configuration resolution is read-only: it reports paths and their winning
sources but never creates them. Artifact-producing commands must call
`require_complete()` and then construct an `ArtifactStore`. Diagnostic commands
may operate with a partial resolution.

An artifact specification contains its storage class, namespace, canonical
configuration, schema version, concrete producer version, and sorted upstream
pointers. It also contains the Raw acquisition generation or Dataset
distribution contract when applicable. The full SHA-256 identity includes all
of these. A compatibility identity excludes only the producer version and
exists solely to implement the explicit `version="*"` reuse override.

Writers create a private staging directory under a per-identity file lock.
Payloads, the manifest, and the completion marker are fsynced; the directory is
then atomically renamed to its immutable final identity. Validation rejects
missing markers, changed checksums or file sets, symlinks, and identity drift.
New publication first resolves and shallow-validates every upstream pointer;
public projections additionally verify their direct Dataset source is
internal. This build-time requirement does not weaken output-first shallow
reuse after an upstream is later removed.
Shallow validation is the normal published-output boundary. Deep validation is
an explicit lineage audit.

The daily runner sorts work by `ticker/date`, hashes its full data-producing
plan, uses the multiprocessing `spawn` context with an explicit worker bound,
and permits workers to return only a validated artifact pointer or `None`. The
parent owns a recoverable append-only, fsynced JSONL journal and progress
callbacks. Its first directory entry is fsynced, and interruption cannot return
while an old pool worker remains able to publish. This keeps Polars frames at
daily artifact boundaries rather than moving them through the process-pool
result channel.

See [configuration.md](configuration.md) and [artifacts.md](artifacts.md) for
the operational contracts.
