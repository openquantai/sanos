"""CLI ownership marker for the DySANOS model package.

The installed command hierarchy is assembled in :mod:`sanos.cli.main` because
it owns the common location, logging, and error contracts.  Its ``dysanos``
group registers the model's build and validation command objects directly;
the legacy ``experiment`` group is only a compatibility alias.
"""

CANONICAL_COMMAND_GROUP = "dysanos"
LEGACY_COMMAND_GROUP = "experiment"

__all__ = ["CANONICAL_COMMAND_GROUP", "LEGACY_COMMAND_GROUP"]
