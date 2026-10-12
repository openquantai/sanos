from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = REPOSITORY_ROOT / "reviews" / "phase-1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_reference_market_catalog_has_the_frozen_endpoint_and_population() -> None:
    catalog = _read_json(REVIEW_ROOT / "reference-market-catalog.json")

    assert catalog["schema"] == "sanos-legacy-market-catalog-v1"
    assert catalog["source_revision"] == (
        "de15be96d2f283b6c31e60d3ea47325a0c093ff0"
    )
    assert catalog["collection_sha256"] == (
        "f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280"
    )
    assert catalog["physical_file_count"] == 892
    assert catalog["populated_day_count"] == 891
    assert catalog["empty_marker_dates"] == ["2025-01-08"]
    assert catalog["first_populated_date"] == "2022-01-03"
    assert catalog["last_populated_date"] == "2025-08-29"
    assert catalog["payload_schema_versions"] == [4]
    assert catalog["tickers"] == ["SPX"]
    assert catalog["aggregate"] == {
        "active_books": 4_651_416,
        "active_call_books": 2_325_708,
        "active_point_two_day_books": 8_090,
        "active_put_books": 2_325_708,
        "expiries": 26_732,
        "strike_nodes": 2_414_341,
    }

    files = catalog["files"]
    assert isinstance(files, list)
    assert len(files) == 892
    names = [record["file"] for record in files]
    assert names == sorted(names)
    assert len(set(names)) == len(names)
    empty = [record for record in files if not record["populated"]]
    assert [record["trading_date"] for record in empty] == ["2025-01-08"]
    assert sum(int(record["byte_size"]) for record in files) == catalog["total_bytes"]


def test_reference_artifact_inventory_pins_catalog_and_distinct_universes() -> None:
    inventory = _read_json(REVIEW_ROOT / "reference-artifacts.json")
    catalog_path = REVIEW_ROOT / "reference-market-catalog.json"
    catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    smooth = inventory["smooth_market_collection"]
    experiment = inventory["dysanos_daily_front_experiment"]
    sdlv = inventory["separate_sdlv_daily_materialization"]

    assert inventory["role"] == "read_only_parity_oracles_not_build_inputs"
    assert smooth["catalog_sha256"] == catalog_digest
    assert smooth["populated_days"] == 891
    assert smooth["active_books_before_downstream_flex_filters"] == 4_651_416
    assert experiment["dates"] == 891
    assert experiment["option_books"] == 3_566_258
    assert experiment["train_range"] == ["2022-01-03", "2025-08-29"]
    assert experiment["producer"] == {
        "entry_point": "quantdata/dysanos/flex/run_all_history_joint_direct.py",
        "entry_point_blob": "6b1c408c3ab1d32e92115cb95c91b3472bcf6086",
        "direct_dependencies": {
            "quantdata/dysanos/flex/config.py": (
                "d9d8b18680f78fb6d23bf532784984a3a86f2518"
            ),
            "quantdata/dysanos/flex/failure_reporting.py": (
                "cc1bcd7ece1759bc8f544e28b62cbe6169dd4f0f"
            ),
            "quantdata/dysanos/flex/joint_direct.py": (
                "e2a1a35959cf506a2abef7b65bfa40964ab31b0b"
            ),
            "quantdata/dysanos/flex/market.py": (
                "67f30c151d1e5bfb33dbd31ba3fb8d4c76c9a4f3"
            ),
            "quantdata/dysanos/flex/market_encoder.py": (
                "fdc4457cf25c99646f5f9f1353592508e39842e8"
            ),
            "quantdata/dysanos/flex/model.py": (
                "d5f65c8df9d8365c304cd6a04feb63b33011d595"
            ),
            "quantdata/dysanos/flex/run_shared_strikes_full_encoder.py": (
                "cb2fe72f6307bac0ea07c275421370906e17b68f"
            ),
            "quantdata/dysanos/flex/training.py": (
                "b4d9c0039e1d07e324ddce2626ad8b8f9ede127d"
            ),
            "quantdata/dysanos/flex/weekly_validation.py": (
                "8756e5f9bc241dd9b42f06e01dcf680d6c4dead3"
            ),
            "quantdata/torch/__init__.py": (
                "aa4fe2a5f49c70b6ea540c2b151bb981a4404386"
            ),
        },
        "disposition": "defer_experiment",
        "role": "experiment_parity_oracle_not_internal_or_public_dataset_state",
    }
    assert experiment["checkpoint_payload"] == {
        "schema": "flex-joint-direct-dynamics-v1",
        "phase": "complete",
        "market_fingerprints": 891,
        "daily_state_embedding_shape": [891, 200],
        "daily_state_embedding_dtype": "torch.float32",
    }
    assert experiment["target_payload"] == {
        "schema": "flex-all-history-joint-direct-targets-v1",
        "dates": 891,
        "date_sequence_equals_populated_smooth_market_catalog": True,
        "states_shape": [891, 200],
        "states_dtype": "torch.float64",
        "spot_refs_shape": [891],
        "spot_refs_dtype": "torch.float64",
        "state_sha256": (
            "3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7"
        ),
    }
    assert sdlv["dates"] == 891
    assert sdlv["retained_option_books"] == 2_192_458
    assert len(
        {
            smooth["active_books_before_downstream_flex_filters"],
            experiment["option_books"],
            sdlv["retained_option_books"],
        }
    ) == 3


def test_runtime_baseline_freezes_versions_configs_solver_and_schema() -> None:
    baseline = _read_json(REVIEW_ROOT / "reference-runtime-baseline.json")

    assert baseline["schema"] == "sanos-phase-1-runtime-baseline-v1"
    assert baseline["quant_data_revision"] == (
        "de15be96d2f283b6c31e60d3ea47325a0c093ff0"
    )
    assert baseline["scope"]["original_artifact_generation_environment_known"] is False
    assert baseline["evidence_runtime"]["python"] == {
        "implementation": "CPython",
        "version": "3.12.3",
    }
    assert baseline["evidence_runtime"]["packages"]["cvxpy"] == "1.9.3"
    assert baseline["evidence_runtime"]["packages"]["highspy"] == "1.15.1"

    versions = baseline["transformation_versions"]
    assert versions["MARKET_SCHEMA_VERSION"] == 4
    assert versions["smooth_interpolated_market_cache"] == "0.15"
    assert versions["MkArbFree"] == "0.51"
    assert versions["SanosPricer"] == "0.8"

    production = baseline["production_configuration"]
    assert production["raw_day_filter"] == {
        "min_bid": 0.011,
        "drop_irregular_expiries": True,
    }
    assert production["expiry_filter"]["num_min_either_traded"] == 10
    assert production["expiry_filter"]["num_min_both_traded"] == 4
    assert production["smooth_market_options_filter"]["max_num_options"] == 100_000
    assert production["smooth_market_options_filter"]["max_nms"] == [-100.0, 100.0]
    assert production["smooth_sanos_fit"]["smoothness"] == 0.25
    assert production["daily_front_selection"]["canonical_nms_inclusive"] == [
        -3.0,
        2.0,
    ]
    neural = production["daily_front_neural_experiment"]
    assert neural["report_schema"] == "flex-all-history-joint-direct-v1"
    assert neural["runtime_precision_bits"] == 32
    assert neural["model"]["state_size"] == 200
    assert neural["model"]["geometry"]["first_model_business_dte"] == 4.8
    assert neural["model"]["geometry"]["explicit_model_business_dtes"] == (
        production["daily_front_selection"]["model_business_dtes"]
    )
    assert neural["training"]["seed"] == 9281
    assert neural["training"]["day_batch_size"] == 128

    solver = baseline["numerical_defaults"]["lp_solver"]
    assert solver["primary"] == {
        "cvxpy_solver": "HIGHS",
        "algorithm": "simplex",
        "simplex_strategy": 1,
        "presolve": "on",
        "threads": 0,
        "kkt_tolerance": 1e-07,
        "warm_start": True,
    }
    assert solver["fallback"]["cvxpy_solver"] == "CLARABEL"
    assert solver["fallback"]["tol_feas"] == 1e-06

    schema = baseline["daily_schema"]
    assert "spot" in schema["raw"]["float32"]
    assert "pure_strike" in schema["enriched_float32_additions"]
    assert schema["expiry_no_arbitrage_additions"]["bool"] == ["mask_arb_free"]
    assert schema["normalized_changes"] == {
        "drop": ["spot"],
        "add_float32": ["spot_ref", "spot_ret"],
    }
