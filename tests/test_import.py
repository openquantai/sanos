from __future__ import annotations

import sanos


def test_package_exposes_version() -> None:
    assert isinstance(sanos.__version__, str)
    assert sanos.__version__

