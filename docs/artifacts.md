# Artifact identity and cache semantics

SANOS stores immutable artifacts below the configured root for their lifecycle
class. The on-disk location is:

```text
<class-root>/<namespace>/<artifact-id>/
```

The artifact ID is a SHA-256 hash of canonical configuration identity, storage
class, namespace, schema version, concrete producer version, sorted upstream
artifact pointers, and the class-specific identity fields below. User-facing
descriptions and filesystem timestamps never participate in identity.

- Raw requires a `source_generation`. An exact retry reuses the same generation
  identity; an explicit refresh creates a new acquisition generation before it
  contacts the source. This lets corrected vendor content produce a new
  immutable Raw identity while preserving query/table/date/security provenance
  in canonical configuration. Phase 4 must expose the generated identity in its
  dry run and result.
- Dataset requires a typed distribution contract. `internal`,
  `public_candidate`, and `public` are distinct identities. `public` also
  requires a license, redistribution basis, approver, ISO approval date, and
  durable approval reference. This infrastructure check does not perform the
  later public allowlist or leakage analysis and never upgrades a candidate.

Lifecycle edges are enforced when a specification is created and again when a
manifest is validated. Raw can aggregate only Raw; Intermediate can depend only
on Raw/Intermediate; Dataset can depend on Raw/Intermediate/Dataset (including
the internal-to-public projection); and Experiment can depend only on Dataset
and must have a Dataset source. Intermediate, Dataset, and Experiment cannot be
source-free. Immediately before a new artifact is published, every declared
upstream pointer must resolve to a complete, shallow-validated artifact. A
`public_candidate` or `public` Dataset requires its entire direct lineage to be
exactly one Dataset projection source, and that source manifest must declare
`internal`.

## Publication

An artifact is built under a per-identity lock in a private sibling staging
directory. Payload files are written atomically. `manifest.json` records the
full identity, canonical configuration, upstream pointers, metadata, byte
sizes, and SHA-256 checksums. `COMPLETE.json` records the manifest checksum and
is written last. After fsync, the staging directory is atomically renamed to the
final identity. Existing final artifacts are never overwritten.

An interrupted context removes its private staging directory. A later writer
holding the same identity lock may remove stale staging directories for that
same identity. Raw or published artifacts are never treated as general-purpose
cleanup targets.

Configured roots are resolved once, but every descendant component controlled
by SANOS must be a real directory rather than a symlink. Atomic destinations,
namespace and staging parents, payload parents, final paths, and lock paths are
checked without following a leaf symlink. A discovered symlink, special file,
or root escape fails before publication. Locks are released in a `finally`
path even when setup, commit, or cleanup fails.

Publication fsyncs payload files and every created/nested artifact directory
bottom-up, renames within one directory, and fsyncs the destination parent.
The guarantee assumes a local filesystem that implements POSIX-style atomic
same-filesystem rename, directory fsync, and advisory file locks. Multi-host
locking and filesystems without those guarantees are unsupported.

## Validation and reuse

Shallow validation checks the requested artifact itself: exact manifest and
completion schemas, recomputed identity, manifest checksum, payload file set,
sizes and checksums, and absence of symlinks. It does not require upstream
artifacts to remain. This is the default for Dataset and Experiment reuse.

Deep validation recursively checks recorded upstream pointers and is available
for explicit lineage audits or rebuild diagnosis. It is not the default cache
test at a self-contained published boundary.

With a concrete producer version, an exact identity miss returns a cache miss
and may let the caller enter an explicitly authorized build path.
`version="*"` is different: it may only select an already complete artifact
whose storage class, namespace, schema, configuration, and upstream identity
match. It emits a visible warning and a structured override record. It does not
modify the stored manifest and can never authorize construction, replacement,
or publication. A miss fails. Multiple old producer versions are ambiguous;
when the current exact artifact exists, it is preferred.

## Local execution

The deterministic runner plans one unit per trading date, sorted by
`ticker/date`. A plan ID hashes its ordered dates, ticker, stage, canonical
payload, schema version, and concrete producer version; executor tuning does
not alter data identity. A journal path cannot be reused for a different plan.
The runner uses a bounded spawn-based process pool. Workers exchange a
`DailyWorkItem` and return only a validated, size-bounded `ArtifactPointer` or
`None`; daily frames belong in artifacts, not the parent process result queue.

The parent writes canonical append-only JSONL with a process-safe contiguous
sequence, plan ID, timestamps, unit transitions, returned/reused artifact
pointers, failures, elapsed time, completed/total work, and ETA. Records are
flushed and fsynced. A partial final write is the only repairable corruption: a
new journal instance truncates that tail under the lock and records
`journal_recovered`; malformed earlier records remain fatal. Sequence numbers
advance only after durable append. Initial journal parent creation and the first
journal directory entry are directory-fsynced under the same local-POSIX
assumptions as artifact publication. The maximum progress interval is five
minutes. Restart logic supplies an output-completeness probe; complete units
are journaled with their immutable pointer and are not submitted again.

On interruption, queued futures are cancelled and active workers receive a
one-second grace period. The runner then terminates, and if necessary kills,
only the child processes belonging to that process pool and waits for their
exit before returning the interruption. Consequently an immediate restart
cannot race a worker retained from the preceding invocation.
