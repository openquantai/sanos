# Portable configurations

Checked-in dataset and experiment configurations belong below this directory.

- `widb-spx-reference.json` defines the fresh internal 2022--2025 SANOS Dataset.
- `dysanos-daily-front-reference.json` pins that Dataset's deterministic pointer
  and the recovered all-history Direct-200 input materialization. Update the
  pinned pointer only when a reviewed, versioned internal Dataset identity
  changes; never substitute a mutable path or alias.

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
