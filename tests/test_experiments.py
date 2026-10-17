"""DySANOS artifact, HDF5, restart, and compatibility tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import sanos.datasets.internal as internal_dataset_module
from sanos.artifacts import ArtifactStore, WildcardReuseMissError
from sanos.datasets import build_internal_dataset
from sanos.models.dysanos import (
    ExperimentBuildOptions,
    ExperimentConfig,
    SourceDatasetConfig,
    build_experiment,
    exact_weekly_pairs,
    plan_experiment,
)
from sanos.models.dysanos.artifacts import experiment_producer_version
from tests.test_widb_build import _dataset_config, _locations, _publish_raw_inputs

_PINNED_TRAINING_SHA256 = "2b5d19f919932ed64eb6f7304a88551690771f72637517a769de6137f6c5a80a"
_PINNED_MARKET_SHA256 = "fc1a53efdf041c8326deca84226b47c96f89f7cae632d5934351d70165d6a303"
_PINNED_ARB_FREE_SHA256 = "d26dd8236160786dc880f30631217d1a8183503019b748e4a6f9bfa0c02f27ca"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_pinned_flex_consumer_if_available(hdf_path: Path) -> None:
    """Exercise the exact de15be96 consumer without making it a dependency."""

    reference_root = Path.home() / "python" / "quant_data"
    training_source = reference_root / "quantdata" / "dysanos" / "flex" / "training.py"
    market_source = reference_root / "quantdata" / "dysanos" / "flex" / "market.py"
    if not training_source.is_file():
        return
    assert _sha256_file(training_source) == _PINNED_TRAINING_SHA256
    assert _sha256_file(market_source) == _PINNED_MARKET_SHA256
    script = f"""
import json
import numpy as np
import h5py
from sanos.models.dysanos.hdf5 import read_hdf5_day
from quantdata.dysanos.flex.market import FlexMarketDay, MarketEligibilityAudit
from quantdata.dysanos.flex.training import materialize_batch, materialize_day
from quantdata.torch import BaseEnvironment

with h5py.File({str(hdf_path)!r}, 'r') as handle:
    day = read_hdf5_day(handle, 0)
reference_day = FlexMarketDay(
    ticker=day.ticker,
    trading_date=day.trading_date,
    expiries_y=day.expiries_y,
    expiry_dates=day.expiry_dates,
    expiry_ids=day.expiry_ids,
    expiry_is_am=day.expiry_is_am,
    expiry_index=day.expiry_index,
    strikes=day.strikes,
    is_call=day.is_call,
    bids=day.bids,
    asks=day.asks,
    fit_prices=day.fit_prices,
    log_volumes=day.log_volumes,
    log_open_interests=day.log_open_interests,
    quote_ids=day.quote_ids,
    atm_annual_vols=day.atm_annual_vols,
    model_atm_total_vols=day.model_atm_total_vols,
    strike_lower_bounds=day.strike_lower_bounds,
    strike_upper_bounds=day.strike_upper_bounds,
    market_uid=day.market_uid,
    audit=MarketEligibilityAudit(
        day.audit.active_european_books,
        day.audit.horizon_eligible_books,
        day.audit.nms_eligible_books,
        day.audit.input_expiries,
        day.audit.eligible_expiries,
    ),
    spot_ref=day.spot_ref,
)
assert day.training_fingerprint == reference_day.training_fingerprint
assert day.market_fingerprint != day.training_fingerprint
env = BaseEnvironment(precision=32, device='cpu', debug=True)
materialized = materialize_day(day, env=env)
reference_materialized = materialize_day(reference_day, env=env)
batch = materialize_batch((materialized,), np.asarray([0], dtype=np.int64), env=env)
reference_batch = materialize_batch(
    (reference_materialized,), np.asarray([0], dtype=np.int64), env=env
)
expected_float = np.float32
np.testing.assert_array_equal(
    materialized.expiries_y.numpy(), day.quote_expiry_y.astype(expected_float)
)
np.testing.assert_array_equal(materialized.strikes.numpy(), day.strikes.astype(expected_float))
np.testing.assert_array_equal(
    materialized.price_expiries_y.numpy(), day.price_expiry_y.astype(expected_float)
)
np.testing.assert_array_equal(
    materialized.price_strikes.numpy(), day.price_strike.astype(expected_float)
)
np.testing.assert_array_equal(materialized.quote_price_index.numpy(), day.quote_price_index)
np.testing.assert_array_equal(materialized.is_call.numpy(), day.is_call)
np.testing.assert_array_equal(materialized.bids.numpy(), day.bids.astype(expected_float))
np.testing.assert_array_equal(materialized.asks.numpy(), day.asks.astype(expected_float))
np.testing.assert_array_equal(
    materialized.fit_prices.numpy(), day.fit_prices.astype(expected_float)
)
assert str(materialized.strikes.dtype) == 'torch.float32'
assert str(materialized.quote_price_index.dtype) == 'torch.int64'
assert str(materialized.is_call.dtype) == 'torch.bool'
assert batch.strikes.shape == (1, day.num_quotes)
assert batch.mask.shape == (1, day.num_quotes)
assert bool(batch.mask.all())
for name in (
    'strikes',
    'price_expiries_y',
    'price_strikes',
    'quote_price_index',
    'is_call',
    'bids',
    'asks',
    'fit_prices',
    'mask',
):
    np.testing.assert_array_equal(
        getattr(batch, name).numpy(), getattr(reference_batch, name).numpy()
    )
print(json.dumps({{'quotes': day.num_quotes, 'prices': day.price_count}}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(Path(__file__).resolve().parents[1] / "src"),
                    str(reference_root),
                )
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["quotes"] > 0
    assert result["prices"] > 0


def _run_pinned_selector_oracle(
    day: object,
    *,
    boundary_day: object,
    boundary_fields: Mapping[str, np.ndarray],
    pricer_expiries: np.ndarray,
    pricer_variances: np.ndarray,
    model_business_dtes: tuple[float, ...],
    npz_path: Path,
) -> None:
    """Compare SANOS with the pinned, checked-in independent FLEX selector."""

    reference_root = Path(__file__).parents[1] / "reference" / "quant_data"
    market_source = reference_root / "quantdata" / "dysanos" / "flex" / "market.py"
    arb_free_source = reference_root / "quantdata" / "sanos" / "arbfree.py"
    assert _sha256_file(market_source) == _PINNED_MARKET_SHA256
    assert _sha256_file(arb_free_source) == _PINNED_ARB_FREE_SHA256

    strikes = np.unique(day.quote_strike)
    fields: dict[str, np.ndarray] = {}
    for side_name, is_call in (("call", True), ("put", False)):
        selected = np.flatnonzero(day.quote_is_call == is_call)
        locations = np.searchsorted(strikes, day.quote_strike[selected])
        active = np.zeros(len(strikes), dtype=np.bool_)
        active[locations] = True
        fields[f"{side_name}_active"] = active
        for output_name, source in (
            ("bid", day.quote_bid),
            ("ask", day.quote_ask),
            ("fit", day.quote_fit),
            ("log_volume", day.quote_log_volume),
            ("log_open_interest", day.quote_log_open_interest),
        ):
            values = np.zeros(len(strikes), dtype=np.float64)
            values[locations] = source[selected]
            fields[f"{side_name}_{output_name}"] = values
    np.savez_compressed(
        npz_path,
        ticker=np.asarray(day.ticker),
        trading_date=np.asarray(day.trading_date.isoformat()),
        spot_reference=np.asarray(day.spot_reference, dtype=np.float64),
        market_uid=np.asarray(day.market_uid),
        expiry_dates=np.asarray(day.expiry_dates),
        expiry_ids=day.expiry_ids,
        expiry_is_am=day.expiry_is_am,
        strikes=strikes,
        is_american=np.zeros(len(strikes), dtype=np.bool_),
        fit_available=np.ones(len(strikes), dtype=np.bool_),
        pricer_expiries=np.asarray(pricer_expiries, dtype=np.float64),
        pricer_variances=np.asarray(pricer_variances, dtype=np.float64),
        model_business_dtes=np.asarray(model_business_dtes, dtype=np.float64),
        expected_expiries_y=day.expiries_y,
        expected_expiry_ids=day.expiry_ids,
        expected_expiry_is_am=day.expiry_is_am,
        expected_expiry_index=day.expiry_index,
        expected_strikes=day.strikes,
        expected_is_call=day.is_call,
        expected_bids=day.bids,
        expected_asks=day.asks,
        expected_fit_prices=day.fit_prices,
        expected_log_volumes=day.log_volumes,
        expected_log_open_interests=day.log_open_interests,
        expected_atm_annual_vols=day.atm_annual_vols,
        expected_model_atm_total_vols=day.model_atm_total_vols,
        expected_strike_lower_bounds=day.strike_lower_bounds,
        expected_strike_upper_bounds=day.strike_upper_bounds,
        expected_training_fingerprint=np.asarray(day.training_fingerprint),
        boundary_expiry_y=np.asarray(boundary_fields["expiry_y"], dtype=np.float64),
        boundary_strikes=np.asarray(boundary_fields["strikes"], dtype=np.float64),
        boundary_is_american=np.asarray(boundary_fields["is_american"], dtype=np.bool_),
        boundary_fit_available=np.asarray(boundary_fields["fit_available"], dtype=np.bool_),
        boundary_call_active=np.asarray(boundary_fields["call_active"], dtype=np.bool_),
        boundary_put_active=np.asarray(boundary_fields["put_active"], dtype=np.bool_),
        boundary_call_fit=np.asarray(boundary_fields["call_fit"], dtype=np.float64),
        boundary_put_fit=np.asarray(boundary_fields["put_fit"], dtype=np.float64),
        boundary_expected_expiries_y=boundary_day.expiries_y,
        boundary_expected_expiry_index=boundary_day.expiry_index,
        boundary_expected_strikes=boundary_day.strikes,
        boundary_expected_is_call=boundary_day.is_call,
        boundary_expected_fit_prices=boundary_day.fit_prices,
        **fields,
    )
    script = f"""
import pathlib
import sys
import types
from datetime import date

import numpy as np

reference_root = pathlib.Path({str(reference_root)!r})
import quantdata.dysanos
flex_package = types.ModuleType('quantdata.dysanos.flex')
flex_package.__path__ = [str(reference_root / 'quantdata' / 'dysanos' / 'flex')]
sys.modules['quantdata.dysanos.flex'] = flex_package

from quantdata.dysanos.flex.config import FlexGeometryConfig, FlexMarketConfig
from quantdata.dysanos.flex.market import _atm_curves_and_bounds, market_day_from_payload
from quantdata.sanos.arbfree import ArbFreePricer

with np.load({str(npz_path)!r}, allow_pickle=False) as z:
    values = {{name: np.asarray(z[name]).copy() for name in z.files}}

pricer = ArbFreePricer(smoothness=0.25)
pricer.expiries = values['pricer_expiries']
pricer.variances = values['pricer_variances']
count = len(pricer.expiries)
pricer.input_index = np.arange(count, dtype=np.int32)
pricer.masks = [np.asarray([True])] * count
pricer.strikes = [np.asarray([1.0])] * count
pricer.asks = [np.asarray([1.0])] * count
pricer.bids = [np.asarray([0.0])] * count
pricer.fits1 = [np.asarray([0.5])] * count
pricer.fitsX = None
pricer.xstrikes = [np.asarray([0.5, 1.0, 1.5])] * count
pricer.xdensities = [np.asarray([0.25, 0.5, 0.25])] * count
pricer.xix_to_mix = None
pricer.mstrikes = None
pricer.mdensities = None
pricer._make_numpy()

geometry = FlexGeometryConfig(
    num_inner_strikes=200,
    nms_range=(-3.0, 2.0),
    num_model_expiries=len(values['model_business_dtes']),
    explicit_model_business_dtes=tuple(values['model_business_dtes']),
)
config = FlexMarketConfig(
    max_num_options=100_000,
    smoothness=0.25,
    excluded_business_dtes=(0.2,),
)

def payload(expiry_y, strikes, *, is_american=None, fit_available=None):
    n = len(strikes)
    zeros = np.zeros(n, dtype=np.float64)
    return {{
        'd': date.fromisoformat(str(values['trading_date'].item())),
        'spot_ref': float(values['spot_reference'].item()),
        'af_pricer': pricer.to_dict(),
        'market': {{
            'expiries': np.asarray([expiry_y], dtype=np.float64),
            'expiry_dates': [str(values['expiry_dates'][0])],
            'expiry_ids': values['expiry_ids'][:1],
            'is_am': values['expiry_is_am'][:1],
            'strikes': [np.asarray(strikes, dtype=np.float64)],
            'is_american': [
                np.zeros(n, dtype=bool) if is_american is None else is_american
            ],
            'af_mid_available': [
                np.ones(n, dtype=bool) if fit_available is None else fit_available
            ],
            'call_active': [np.ones(n, dtype=bool)],
            'put_active': [np.ones(n, dtype=bool)],
            'call_bids': [zeros],
            'call_asks': [np.ones(n, dtype=np.float64)],
            'af_call_mids': [np.full(n, 0.5, dtype=np.float64)],
            'call_log_volumes': [zeros],
            'call_log_open_interests': [zeros],
            'put_bids': [zeros],
            'put_asks': [np.ones(n, dtype=np.float64)],
            'af_put_mids': [np.full(n, 0.5, dtype=np.float64)],
            'put_log_volumes': [zeros],
            'put_log_open_interests': [zeros],
            'uid': str(values['market_uid'].item()),
        }},
    }}

market = payload(0.0, values['strikes'])['market']
market.update({{
    'is_american': [values['is_american']],
    'af_mid_available': [values['fit_available']],
    'call_active': [values['call_active']],
    'put_active': [values['put_active']],
    'call_bids': [values['call_bid']],
    'call_asks': [values['call_ask']],
    'af_call_mids': [values['call_fit']],
    'call_log_volumes': [values['call_log_volume']],
    'call_log_open_interests': [values['call_log_open_interest']],
    'put_bids': [values['put_bid']],
    'put_asks': [values['put_ask']],
    'af_put_mids': [values['put_fit']],
    'put_log_volumes': [values['put_log_volume']],
    'put_log_open_interests': [values['put_log_open_interest']],
}})
actual = market_day_from_payload(
    {{
        'd': date.fromisoformat(str(values['trading_date'].item())),
        'spot_ref': float(values['spot_reference'].item()),
        'af_pricer': pricer.to_dict(),
        'market': market,
    }},
    ticker=str(values['ticker'].item()),
    geometry=geometry,
    config=config,
)
assert actual is not None
for actual_name, expected_name in (
    ('expiries_y', 'expected_expiries_y'),
    ('expiry_ids', 'expected_expiry_ids'),
    ('expiry_is_am', 'expected_expiry_is_am'),
    ('expiry_index', 'expected_expiry_index'),
    ('strikes', 'expected_strikes'),
    ('is_call', 'expected_is_call'),
    ('bids', 'expected_bids'),
    ('asks', 'expected_asks'),
    ('fit_prices', 'expected_fit_prices'),
    ('log_volumes', 'expected_log_volumes'),
    ('log_open_interests', 'expected_log_open_interests'),
    ('atm_annual_vols', 'expected_atm_annual_vols'),
    ('model_atm_total_vols', 'expected_model_atm_total_vols'),
    ('strike_lower_bounds', 'expected_strike_lower_bounds'),
    ('strike_upper_bounds', 'expected_strike_upper_bounds'),
):
    np.testing.assert_array_equal(getattr(actual, actual_name), values[expected_name])
assert actual.training_fingerprint == str(values['expected_training_fingerprint'].item())

boundary_payload = payload(
    float(values['boundary_expiry_y'].item()), values['boundary_strikes'],
)
boundary_payload['market'].update({{
    'is_american': [values['boundary_is_american']],
    'af_mid_available': [values['boundary_fit_available']],
    'call_active': [values['boundary_call_active']],
    'put_active': [values['boundary_put_active']],
    'af_call_mids': [values['boundary_call_fit']],
    'af_put_mids': [values['boundary_put_fit']],
}})
boundary_actual = market_day_from_payload(
    boundary_payload, ticker=str(values['ticker'].item()),
    geometry=geometry, config=config,
)
assert boundary_actual is not None
for actual_name, expected_name in (
    ('expiries_y', 'boundary_expected_expiries_y'),
    ('expiry_index', 'boundary_expected_expiry_index'),
    ('strikes', 'boundary_expected_strikes'),
    ('is_call', 'boundary_expected_is_call'),
    ('fit_prices', 'boundary_expected_fit_prices'),
):
    np.testing.assert_array_equal(
        getattr(boundary_actual, actual_name), values[expected_name]
    )
assert (
    boundary_actual.audit.active_european_books,
    boundary_actual.audit.horizon_eligible_books,
    boundary_actual.audit.nms_eligible_books,
    boundary_actual.audit.input_expiries,
    boundary_actual.audit.eligible_expiries,
) == (8, 8, 4, 1, 1)

def bounds(expiry_y):
    sample = payload(expiry_y, np.asarray([1.0]))
    _, _, lower, upper = _atm_curves_and_bounds(
        sample, np.asarray([expiry_y], dtype=np.float64), geometry=geometry
    )
    return lower[0], upper[0]

def selected(expiry_y):
    lower, upper = bounds(expiry_y)
    strikes = np.asarray([
        np.nextafter(lower, -np.inf), lower, 1.0, upper,
        np.nextafter(upper, np.inf),
    ])
    result = market_day_from_payload(
        payload(expiry_y, strikes), ticker='SPX', geometry=geometry, config=config
    )
    return result, strikes

zero, zero_strikes = selected(0.0)
assert zero is not None
np.testing.assert_array_equal(
    zero.strikes,
    np.concatenate((zero_strikes[1:4], zero_strikes[1:4])),
)
assert market_day_from_payload(
    payload(0.2 / 252.0, np.asarray([1.0])),
    ticker='SPX', geometry=geometry, config=config,
) is None
for inside_dte in (0.2 - 0.000099999, 0.2 + 0.000099999):
    assert market_day_from_payload(
        payload(inside_dte / 252.0, np.asarray([1.0])),
        ticker='SPX', geometry=geometry, config=config,
    ) is None
for outside_dte in (0.2 - 0.000100001, 0.2 + 0.000100001):
    assert market_day_from_payload(
        payload(outside_dte / 252.0, np.asarray([1.0])),
        ticker='SPX', geometry=geometry, config=config,
    ) is not None
maximum = float(values['model_business_dtes'][-1]) / 252.0
at_horizon, _ = selected(maximum)
assert at_horizon is not None
assert market_day_from_payload(
    payload(maximum + 2.0e-12, np.asarray([1.0])),
    ticker='SPX', geometry=geometry, config=config,
) is None
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(reference_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _config(source_namespace: str, source_artifact_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-daily-front",
        source_dataset=SourceDatasetConfig(
            namespace=source_namespace,
            artifact_id=source_artifact_id,
        ),
    )


def _publish_source_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object]:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    _publish_raw_inputs(store, trading_date)
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    source = build_internal_dataset(locations, _dataset_config(trading_date))
    return locations, source


def test_reference_experiment_configuration_is_strict_and_deterministic() -> None:
    path = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_document(document)
    assert ExperimentConfig.from_document(config.to_document()) == config
    assert config.market.overnight_variance_fraction == 0.2
    assert config.market.excluded_business_dtes == (0.0, 0.2)
    assert config.geometry.model_business_dtes == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
    )
    assert config.model.decoded_state_size == 18 * 202
    assert plan_experiment(config).spec.upstream == (config.source_dataset.pointer,)

    malformed = json.loads(json.dumps(document))
    malformed["machine_path"] = "/private/data"
    with pytest.raises(ValueError, match="sections fields changed"):
        ExperimentConfig.from_document(malformed)

    malformed = json.loads(json.dumps(document))
    malformed["storage"]["shuffle"] = "true"
    with pytest.raises((TypeError, ValueError), match="booleans"):
        ExperimentConfig.from_document(malformed)

    invalid_bool = json.loads(json.dumps(document))
    invalid_bool["objective"]["fit_weight"] = True
    with pytest.raises(ValueError, match="finite number"):
        ExperimentConfig.from_document(invalid_bool)


def test_every_experiment_content_section_changes_identity() -> None:
    path = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    original_id = plan_experiment(ExperimentConfig.from_document(original)).spec.artifact_id
    mutations = (
        ("experiment", "name", "spx-daily-front-reference-v2"),
        ("market", "smoothness", 0.3),
        ("market", "excluded_business_dtes", []),
        ("market", "overnight_variance_fraction", 0.5),
        ("geometry", "nms_range", [-2.9, 2.0]),
        ("objective", "fit_weight", 0.06),
        ("split", "diagnostic_boundary", "2024-12-31"),
        ("storage", "compression_level", 5),
        ("model", "state_size", 201),
        ("training", "num_epochs", 301),
    )
    for section, field, value in mutations:
        changed = json.loads(json.dumps(original))
        changed[section][field] = value
        assert (
            plan_experiment(ExperimentConfig.from_document(changed)).spec.artifact_id != original_id
        ), (section, field)

    config = ExperimentConfig.from_document(original)
    assert ExperimentBuildOptions(max_workers=1) != ExperimentBuildOptions(max_workers=2)
    assert plan_experiment(config).spec.artifact_id == original_id
    overnight_ids = {
        plan_experiment(
            replace(
                config,
                market=replace(config.market, overnight_variance_fraction=value),
            )
        ).spec.artifact_id
        for value in (0.0, 0.2, 0.5, 1.0)
    }
    assert len(overnight_ids) == 4


def test_experiment_version_closure_and_identity_are_fresh_process_stable() -> None:
    path = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    config = ExperimentConfig.from_document(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        "producer_version": experiment_producer_version(),
        "artifact_id": plan_experiment(config).spec.artifact_id,
    }
    dependencies = dict(experiment_producer_version.version.dependencies[1])
    assert {
        "ExperimentConfig",
        "_parse_manifest_config",
        "_experiment_readme",
        "_experiment_schema_document",
        "_experiment_source_document",
        "_experiment_split_document",
        "_experiment_summary",
        "_experiment_validation_document",
        "_prepare_shards",
        "_project_partition",
        "_source_gate",
        "project_daily_front",
        "read_day_shard",
        "validate_experiment_hdf5",
        "write_day_shard",
        "write_experiment_hdf5",
    } <= set(dependencies)
    script = f"""
import json
from pathlib import Path
from sanos.models.dysanos import ExperimentConfig, plan_experiment
from sanos.models.dysanos.artifacts import experiment_producer_version
config = ExperimentConfig.from_document(json.loads(Path({str(path)!r}).read_text()))
print(json.dumps({{
    'producer_version': experiment_producer_version(),
    'artifact_id': plan_experiment(config).spec.artifact_id,
}}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == expected


@pytest.mark.parametrize(
    ("dates", "expected_origin", "expected_endpoint"),
    [
        (
            (date(2024, 1, 3), date(2024, 1, 8), date(2024, 1, 10), date(2024, 1, 17)),
            [0],
            [2],
        ),
        (
            (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 9), date(2024, 1, 10)),
            [0, 1],
            [2, 3],
        ),
    ],
)
def test_exact_weekly_pairs_require_five_nyse_sessions(
    dates: tuple[date, ...],
    expected_origin: list[int],
    expected_endpoint: list[int],
) -> None:
    origin, endpoint = exact_weekly_pairs(dates)
    np.testing.assert_array_equal(origin, np.asarray(expected_origin, dtype=np.int64))
    np.testing.assert_array_equal(endpoint, np.asarray(expected_endpoint, dtype=np.int64))


def test_experiment_dry_run_and_wildcard_miss_never_open_source(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    config = _config("widb/internal/name-absent", "a" * 64)
    result = build_experiment(locations, config, dry_run=True)
    assert result.dry_run
    assert not result.built
    assert not result.source_accessed
    assert not locations.experiments.exists()

    with pytest.raises(WildcardReuseMissError, match="reuse only"):
        build_experiment(
            locations,
            config,
            replace(ExperimentBuildOptions(), version_override="*"),
            dry_run=True,
        )
