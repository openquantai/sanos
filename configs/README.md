# Portable configurations

Checked-in dataset and experiment configurations belong below this directory.

- `widb-spx-reference.json` defines the fresh internal 2022--2025 SANOS Dataset.
- `spx-2022-2025-oracle-gaps.json` predeclares the 26 planned NYSE sessions
  absent from the checksum-pinned legacy daily catalog. It is comparison policy,
  not permission to omit any fresh fitted day.
- `dysanos-daily-front-reference.json` and
  `sanos-public-candidate-reference.json` are reviewed portable templates. Their
  checked pointers document the expected reviewed identity. A fresh production
  run uses `sanos production finalize-configs` to bind both mechanically to the
  actual immutable internal Dataset and emit canonical digests; never perform
  undocumented JSON edits or substitute a mutable path/alias.

They must not contain machine paths or credentials. At build time they are
normalized to canonical JSON and included in artifact identity.

`widb-spx-reference.json` contains three independent sections. `dataset` owns
the immutable name, ticker, coverage, calendar policy and Raw generation.
`daily` owns the
paired/enriched WIDB semantics, including the 0.2 overnight variance fraction.
`surface` owns the Phase 6 initial linear no-arbitrage fit, the recovered
100,000-option/70-business-day filter profile, and the production smoothness
of 0.25. The solver timeout is intentionally not configurable while CD-006 is
open; the compatibility implementation fixes and records 300 seconds.
