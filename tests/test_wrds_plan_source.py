from __future__ import annotations

from datetime import date

import pytest

from sanos.wrds.plan import RawDownloadConfig, RawPartition, RawTable, plan_raw_partitions
from sanos.wrds.source import (
    OptionMetricsCatalog,
    build_raw_query,
    choose_optionmetrics_catalog,
    resolve_physical_table,
)


def test_plan_includes_fresh_previous_session_dependencies() -> None:
    config = RawDownloadConfig(
        ticker="spx",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )

    planned = plan_raw_partitions(config)

    assert planned[0].ticker == "SPX"
    assert sum(item.table is RawTable.SECURITY for item in planned) == 1
    boundary = [item for item in planned if item.is_boundary_dependency]
    assert {(item.table, item.observation_date) for item in boundary} == {
        (RawTable.SPOT, date(2023, 12, 29)),
        (RawTable.FORWARDS, date(2023, 12, 29)),
        (RawTable.ZERO_CURVE, date(2023, 12, 29)),
    }
    assert not any(
        item.table is RawTable.OPTIONS and item.is_boundary_dependency for item in planned
    )


def test_subset_plan_still_includes_security_identity() -> None:
    config = RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        tables=(RawTable.OPTIONS,),
    )

    planned = plan_raw_partitions(config)

    assert [(item.table, item.observation_date) for item in planned] == [
        (RawTable.OPTIONS, date(2024, 1, 3)),
        (RawTable.SECURITY, None),
    ]


def test_config_rejects_connections_above_wrds_limit_before_access() -> None:
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        RawDownloadConfig(
            ticker="SPX",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 3),
            max_connections=5,
        )


def test_catalog_and_queries_preserve_reference_table_precedence_and_vendor_units() -> None:
    catalog = choose_optionmetrics_catalog(
        ["optionm", "optionm_all"],
        {
            "optionm_all": [
                "securd1",
                "opprcd2024",
                "opprcd",
                "secprd",
                "fwdprd2024",
                "zerocd",
            ],
            "optionm": ["securd"],
        },
    )
    partition = RawPartition(RawTable.OPTIONS, "SPX", date(2024, 1, 3))

    assert catalog.library == "optionm_all"
    assert resolve_physical_table(partition, catalog) == "optionm_all.opprcd2024"
    query = build_raw_query(partition, catalog, secid=108105)
    assert "strike_price" in query.sql
    assert "strike_price / 1000" not in query.sql
    assert "secid = 108105 AND date = '2024-01-03'" in query.sql
    assert query.sql.startswith("SELECT secid, date, symbol")


def test_unavailable_physical_table_fails_with_visible_candidates() -> None:
    catalog = OptionMetricsCatalog("optionm", frozenset({"securd"}))
    partition = RawPartition(RawTable.FORWARDS, "SPX", date(2024, 1, 3))

    with pytest.raises(RuntimeError, match="fwdprd2024.*fwdprd"):
        resolve_physical_table(partition, catalog)
