from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_widb_build import _dataset_config, _locations, _publish_raw_inputs
from test_widb_surface import _two_expiry_raw_inputs

import sanos.datasets.internal as internal_dataset_module
import sanos.experiments.hdf5 as hdf5_module
from sanos.artifacts import (
    ArtifactNotFoundError,
    ArtifactPointer,
    ArtifactStore,
    StorageClass,
    WildcardReuseMissError,
)
from sanos.config import canonical_sha256
from sanos.datasets import build_internal_dataset, open_internal_dataset
from sanos.experiments import (
    DailyFrontDay,
    ExperimentBuildOptions,
    ExperimentConfig,
    ExperimentValidationError,
    SourceDatasetConfig,
    build_experiment,
    daily_quote_objective,
    exact_weekly_pairs,
    open_experiment,
    plan_experiment,
    project_daily_front,
    validate_experiment,
)
from sanos.experiments.build import _prepare_shards, experiment_producer_version
from sanos.experiments.daily_front import recompute_market_fingerprint
from sanos.experiments.hdf5 import (
    logical_content_digest,
    read_day_shard,
    validate_experiment_hdf5,
    write_day_shard,
    write_experiment_hdf5,
)
from sanos.pipeline import RunJournal
from sanos.widb import (
    ClockConfig,
    DailyDataError,
    DailyWidbConfig,
    apply_options_filter,
    enrich_daily_frame,
    fit_initial_no_arbitrage,
    fit_smooth_sanos,
    pair_daily_frame,
)
from sanos.widb.surface import smooth_sanos_pricer

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
from sanos.experiments.hdf5 import read_hdf5_day
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
    assert config.market.excluded_business_dtes == (0.2,)
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
        ("market", "max_num_options", 99_999),
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


def test_overnight_fraction_reaches_dlv_experiment_horizon_and_hdf5(
    tmp_path: Path,
) -> None:
    source = ArtifactPointer(
        StorageClass.DATASET,
        "widb/internal/name-overnight-sensitivity",
        "a" * 64,
    )
    base = _config(source.namespace, source.artifact_id)
    observed_dlv: list[np.ndarray] = []
    observed_days: dict[float, DailyFrontDay] = {}
    observed_frames: dict[float, pl.DataFrame] = {}
    for fraction in (0.1, 0.2, 0.5):
        daily = DailyWidbConfig(
            clock=ClockConfig(overnight_variance_fraction=fraction)
        )
        fitted = fit_smooth_sanos(
            apply_options_filter(
                fit_initial_no_arbitrage(
                    enrich_daily_frame(
                        pair_daily_frame(
                            ticker="SPX",
                            trading_date=date(2024, 1, 3),
                            raw=_two_expiry_raw_inputs(first_is_am=True),
                            config=daily,
                        ),
                        config=daily,
                    )
                )
            )
        )
        ordered = fitted.sort("time_to_expiry_years", "expiry_id")
        observed_frames[fraction] = fitted
        first = ordered.row(0, named=True)
        assert first["business_dte"] == pytest.approx(np.float32(10.0 + fraction))
        observed_dlv.append(
            np.asarray(first["initial_surface_dlv_squares"], dtype=np.float64)
        )
        config = replace(
            base,
            market=replace(
                base.market,
                overnight_variance_fraction=fraction,
            ),
        )
        day = project_daily_front(
            fitted,
            source_dataset=source,
            market=config.market,
            geometry=config.geometry,
        )
        observed_days[fraction] = day
        assert day.expiries_y[0] * 252.0 == pytest.approx(10.0 + fraction)
        assert day.expiries_y[1] * 252.0 == pytest.approx(31.0)
        hdf_path = tmp_path / f"overnight-{fraction}.h5"
        write_experiment_hdf5(
            hdf_path,
            (day,),
            config=config,
            producer_version="overnight-sensitivity-v1",
        )
        validation = validate_experiment_hdf5(
            hdf_path,
            config=config,
            producer_version="overnight-sensitivity-v1",
        )
        assert validation["dates"]["count"] == 1

    assert any(
        not np.array_equal(left, right)
        for left, right in zip(observed_dlv, observed_dlv[1:], strict=True)
    )
    short_geometry = replace(
        base.geometry,
        model_business_dtes=(1.0, 5.0, 10.25),
    )
    for fraction in (0.1, 0.2):
        day = observed_days[fraction]
        assert project_daily_front(
            observed_frames[fraction],
            source_dataset=source,
            market=replace(
                base.market,
                overnight_variance_fraction=fraction,
            ),
            geometry=short_geometry,
        ).expiry_count == 1
        assert day.expiry_count == 2
    with pytest.raises(DailyDataError, match="no daily-front eligible books"):
        fraction = 0.5
        project_daily_front(
            observed_frames[fraction],
            source_dataset=source,
            market=replace(
                base.market,
                overnight_variance_fraction=fraction,
            ),
            geometry=short_geometry,
        )


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
from sanos.experiments import ExperimentConfig, plan_experiment
from sanos.experiments.build import experiment_producer_version
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


def test_exact_weekly_pairs_require_five_nyse_sessions() -> None:
    dates = (
        date(2024, 1, 3),
        date(2024, 1, 8),
        date(2024, 1, 10),
        date(2024, 1, 17),
    )
    origin, endpoint = exact_weekly_pairs(dates)
    np.testing.assert_array_equal(origin, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(endpoint, np.asarray([2], dtype=np.int64))


def test_experiment_restart_reuses_valid_and_rebuilds_invalid_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source_dataset(tmp_path, monkeypatch)
    config = _config(source.artifact.namespace, source.artifact.artifact_id)
    plan = plan_experiment(config)
    store = ArtifactStore(locations)
    dataset = open_internal_dataset(store, source.artifact)
    workspace = tmp_path / "restart-workspace"
    journal = RunJournal(tmp_path / "restart-journal.jsonl")
    journal.require_plan(plan.plan_id)
    options = ExperimentBuildOptions(max_workers=1, progress_interval_seconds=0.1)

    def prepare() -> tuple[Path, list[dict[str, object]]]:
        events: list[dict[str, object]] = []
        shards, exclusions = _prepare_shards(
            dataset,
            plan=plan,
            workspace=workspace,
            config=config,
            options=options,
            progress=events.append,
            journal=journal,
        )
        assert exclusions == ()
        assert len(shards) == 1
        return shards[0], events

    shard, _ = prepare()
    original = read_day_shard(shard)
    reused_shard, events = prepare()
    assert reused_shard == shard
    completed = [event for event in events if event["event"] == "experiment_preparation_completed"]
    assert completed[-1]["reused"] == 1

    shard.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(ExperimentValidationError, match="Cannot read"):
        read_day_shard(shard)
    rebuilt, _ = prepare()
    assert read_day_shard(rebuilt).market_fingerprint == original.market_fingerprint

    wrong_date = replace(read_day_shard(shard), trading_date=date(2024, 1, 4))
    shard.unlink()
    write_day_shard(shard, wrong_date)
    prepare()
    assert read_day_shard(shard).trading_date == original.trading_date

    with np.load(shard, allow_pickle=False) as payload:
        malformed = {name: np.asarray(payload[name]).copy() for name in payload.files}
    malformed["schema"] = np.asarray(b"wrong-schema", dtype="S64")
    shard.unlink()
    with shard.open("xb") as stream:
        np.savez_compressed(stream, **malformed)
    prepare()
    assert read_day_shard(shard).market_fingerprint == original.market_fingerprint

    shard.unlink()
    write_day_shard(shard, replace(original, market_fingerprint="0" * 64))
    prepare()
    assert read_day_shard(shard).market_fingerprint == original.market_fingerprint


def test_experiment_build_is_self_contained_output_first_and_tamper_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source_dataset(tmp_path, monkeypatch)
    config = _config(source.artifact.namespace, source.artifact.artifact_id)
    events: list[dict[str, object]] = []
    result = build_experiment(
        locations,
        config,
        ExperimentBuildOptions(progress_interval_seconds=0.1),
        progress=events.append,
    )
    assert result.built
    assert result.source_accessed
    assert result.validation is not None
    assert result.validation.dates == 1
    assert result.validation.books > 0
    assert result.validation.calls == result.validation.puts
    assert result.validation.model_nodes == 18
    journal = tuple(
        json.loads(line) for line in result.journal_path.read_text(encoding="utf-8").splitlines()
    )
    assert [record["event"] for record in journal] == [
        "run_started",
        "unit_completed",
        "artifact_published",
        "run_completed",
    ]
    assert not result.workspace.exists()
    assert {event["event"] for event in events} >= {
        "experiment_plan",
        "experiment_preparation_started",
        "experiment_preparation_completed",
        "experiment_build_completed",
    }

    store = ArtifactStore(locations)
    opened = open_experiment(store, result.artifact)
    assert opened.dates() == (date(2024, 1, 3),)
    loaded = opened.load_day(0)
    assert loaded.market_uid == opened.load_day(date(2024, 1, 3)).market_uid
    assert loaded.quote_count == result.validation.books
    assert loaded.num_quotes == loaded.quote_count
    assert loaded.training_fingerprint != loaded.market_fingerprint
    assert loaded.expiries_y is loaded.expiry_time_y
    assert loaded.expiry_index is loaded.quote_expiry_index
    assert loaded.strikes is loaded.quote_strike
    assert loaded.bids is loaded.quote_bid
    assert loaded.asks is loaded.quote_ask
    assert loaded.fit_prices is loaded.quote_fit
    assert loaded.expiry_dates == tuple(
        str(np.datetime64("1970-01-01") + np.timedelta64(int(value), "D"))
        for value in loaded.expiry_date_days
    )
    assert loaded.audit.nms_eligible_books == loaded.quote_count
    np.testing.assert_array_equal(
        loaded.quote_expiry_y,
        loaded.expiry_time_y[loaded.quote_expiry_index],
    )
    assert (
        daily_quote_objective(
            loaded.quote_fit,
            loaded,
            fit_weight=config.objective.fit_weight,
        )
        == 0.0
    )
    outside = loaded.quote_ask + loaded.half_spreads
    expected = np.mean(
        (outside - loaded.quote_ask) / loaded.half_spreads
        + config.objective.fit_weight * np.abs(outside - loaded.quote_fit) / loaded.half_spreads
    )
    assert daily_quote_objective(
        outside,
        loaded,
        fit_weight=config.objective.fit_weight,
    ) == pytest.approx(float(expected), rel=0.0, abs=1.0e-14)
    assert validate_experiment(store, result.artifact) == result.validation
    assert validate_experiment(store, result.artifact, deep_lineage=True) == result.validation

    source_dataset = open_internal_dataset(store, source.artifact)
    source_frame = source_dataset.load_day(date(2024, 1, 3))
    first_expiry_id = (
        source_frame.filter(pl.col("options_filter_eligible"))
        .sort("time_to_expiry_years", "expiry_id")
        .get_column("expiry_id")[0]
    )
    zero_market_rows = source_frame.with_columns(
        pl.lit(0.0, dtype=pl.Float32).alias("time_to_expiry_years"),
        pl.col("options_filter_eligible").alias("options_filter_eligible"),
        pl.lit(None, dtype=pl.Int32).alias("smooth_input_expiry_index"),
    )
    model_rows = source_frame.with_columns(
        pl.lit(int(first_expiry_id) + 1_000_000, dtype=pl.Int32).alias("expiry_id"),
        pl.lit(False).alias("options_filter_eligible"),
    )
    zero_expiry_frame = pl.concat((zero_market_rows, model_rows), how="vertical")
    zero_day = project_daily_front(
        zero_expiry_frame,
        source_dataset=source.artifact,
        market=config.market,
        geometry=config.geometry,
    )
    assert zero_day.expiries_y.tolist() == [0.0]
    assert np.all(zero_day.quote_expiry_y == 0.0)
    assert np.all(zero_day.price_expiry_y == 0.0)

    lower = float(zero_day.strike_lower_bounds[0])
    upper = float(zero_day.strike_upper_bounds[0])
    boundary_strikes = np.asarray(
        (
            lower * 0.99,
            lower + (1.0 - lower) * 0.2,
            lower + (1.0 - lower) * 0.6,
            lower + (1.0 - lower) * 0.8,
            1.0,
            1.0 + (upper - 1.0) * 0.3,
            1.0 + (upper - 1.0) * 0.8,
            upper * 1.01,
        ),
        dtype=np.float32,
    )
    call_active = np.asarray((True, True, False, True, True, True, True, True), dtype=np.bool_)
    put_active = np.asarray((True, False, True, True, True, True, True, True), dtype=np.bool_)
    is_american = np.asarray(
        (False, False, False, True, False, False, False, False), dtype=np.bool_
    )
    smooth_available = np.asarray((True, True, True, True, False, True, True, True), dtype=np.bool_)
    pinned_available = smooth_available.copy()
    pinned_available[5] = False
    boundary_fit = np.full(len(boundary_strikes), 0.15, dtype=np.float64)
    boundary_fit[5] = np.nan
    base_row = source_frame.filter(pl.col("options_filter_eligible")).head(1)
    boundary_market_rows = pl.concat([base_row] * len(boundary_strikes)).with_columns(
        pl.Series("normalized_strike", boundary_strikes, dtype=pl.Float32),
        pl.Series("call_active", call_active, dtype=pl.Boolean),
        pl.Series("put_active", put_active, dtype=pl.Boolean),
        pl.Series("is_american", is_american, dtype=pl.Boolean),
        pl.Series("smooth_mid_available", smooth_available, dtype=pl.Boolean),
        pl.Series("smooth_call_mid", boundary_fit, dtype=pl.Float64),
        pl.Series("smooth_put_mid", boundary_fit, dtype=pl.Float64),
        pl.lit(0.10, dtype=pl.Float32).alias("pure_market_call_bid"),
        pl.lit(0.20, dtype=pl.Float32).alias("pure_market_call_ask"),
        pl.lit(0.10, dtype=pl.Float32).alias("pure_market_put_bid"),
        pl.lit(0.20, dtype=pl.Float32).alias("pure_market_put_ask"),
        pl.lit(0.0, dtype=pl.Float32).alias("time_to_expiry_years"),
        pl.lit(True).alias("options_filter_eligible"),
        pl.lit(None, dtype=pl.Int32).alias("smooth_input_expiry_index"),
    )
    boundary_frame = pl.concat((boundary_market_rows, model_rows), how="vertical")
    boundary_day = project_daily_front(
        boundary_frame,
        source_dataset=source.artifact,
        market=config.market,
        geometry=config.geometry,
    )
    assert boundary_day.audit == replace(
        boundary_day.audit,
        active_european_books=8,
        horizon_eligible_books=8,
        nms_eligible_books=4,
        input_expiries=1,
        eligible_expiries=1,
    )
    np.testing.assert_array_equal(
        boundary_day.quote_price_index,
        np.asarray((0, 1, 2, 1), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        boundary_day.quote_is_call,
        np.asarray((True, True, False, False), dtype=np.bool_),
    )
    zero_shard = tmp_path / "zero-boundary" / "day.npz"
    write_day_shard(zero_shard, zero_day)
    assert read_day_shard(zero_shard).expiries_y.tolist() == [0.0]
    zero_hdf = tmp_path / "zero-boundary.h5"
    write_experiment_hdf5(
        zero_hdf,
        (zero_day,),
        config=config,
        producer_version=result.plan.spec.producer_version,
    )
    zero_validation = validate_experiment_hdf5(
        zero_hdf,
        config=config,
        producer_version=result.plan.spec.producer_version,
    )
    assert zero_validation["consumer"]["adapter_status"] == "passed"

    def retimed_day(expiry_y: float) -> object:
        changed = replace(
            zero_day,
            expiry_time_y=np.full(zero_day.expiry_count, expiry_y, dtype=np.float64),
            price_expiry_y=np.full(zero_day.price_count, expiry_y, dtype=np.float64),
            market_fingerprint="0" * 64,
        )
        return replace(
            changed,
            market_fingerprint=recompute_market_fingerprint(changed, source.artifact),
        )

    semantic_mutations = (
        (
            "excluded-dte",
            retimed_day(0.2 / 252.0),
            "excluded business DTE",
        ),
        (
            "over-horizon",
            retimed_day(config.geometry.model_business_dtes[-1] / 252.0 + 2.0e-12),
            "horizon exceeds",
        ),
    )
    for label, changed_day, message in semantic_mutations:
        changed_hdf = tmp_path / f"{label}.h5"
        write_experiment_hdf5(
            changed_hdf,
            (changed_day,),
            config=config,
            producer_version=result.plan.spec.producer_version,
        )
        with pytest.raises(ExperimentValidationError, match=message):
            validate_experiment_hdf5(
                changed_hdf,
                config=config,
                producer_version=result.plan.spec.producer_version,
            )

    changed_bounds = replace(
        zero_day,
        expiry_strike_lower=zero_day.expiry_strike_lower * 0.99,
        market_fingerprint="0" * 64,
    )
    changed_bounds = replace(
        changed_bounds,
        market_fingerprint=recompute_market_fingerprint(changed_bounds, source.artifact),
    )
    changed_bounds_hdf = tmp_path / "changed-nms-bounds.h5"
    write_experiment_hdf5(
        changed_bounds_hdf,
        (changed_bounds,),
        config=config,
        producer_version=result.plan.spec.producer_version,
    )
    with pytest.raises(ExperimentValidationError, match="NMS bounds"):
        validate_experiment_hdf5(
            changed_bounds_hdf,
            config=config,
            producer_version=result.plan.spec.producer_version,
        )

    with pytest.raises(ValueError, match="outside their inclusive NMS bounds"):
        replace(
            zero_day,
            quote_strike=np.full(
                zero_day.quote_count,
                zero_day.strike_upper_bounds[0] * 1.01,
                dtype=np.float64,
            ),
        )

    class DayProxy:
        def __init__(self, base: object, **changes: object) -> None:
            self._base = base
            self._changes = changes

        def __getattr__(self, name: str) -> object:
            if name in self._changes:
                return self._changes[name]
            return getattr(self._base, name)

    def one_quote_day(original: object, trading_date: date) -> object:
        expiry_index = int(original.quote_expiry_index[0])
        changed = replace(
            original,
            trading_date=trading_date,
            market_fingerprint="0" * 64,
            quote_expiry_index=np.asarray((0,), dtype=np.int64),
            quote_price_index=np.asarray((0,), dtype=np.int64),
            quote_strike=original.quote_strike[:1].copy(),
            quote_is_call=original.quote_is_call[:1].copy(),
            quote_bid=original.quote_bid[:1].copy(),
            quote_ask=original.quote_ask[:1].copy(),
            quote_fit=original.quote_fit[:1].copy(),
            quote_log_volume=original.quote_log_volume[:1].copy(),
            quote_log_open_interest=original.quote_log_open_interest[:1].copy(),
            expiry_time_y=original.expiry_time_y[expiry_index : expiry_index + 1].copy(),
            expiry_date_days=original.expiry_date_days[expiry_index : expiry_index + 1].copy(),
            expiry_id=original.expiry_id[expiry_index : expiry_index + 1].copy(),
            expiry_is_am=original.expiry_is_am[expiry_index : expiry_index + 1].copy(),
            expiry_atm_annual_vol=original.expiry_atm_annual_vol[
                expiry_index : expiry_index + 1
            ].copy(),
            expiry_strike_lower=original.expiry_strike_lower[
                expiry_index : expiry_index + 1
            ].copy(),
            expiry_strike_upper=original.expiry_strike_upper[
                expiry_index : expiry_index + 1
            ].copy(),
            price_expiry_y=np.asarray((original.expiry_time_y[expiry_index],), dtype=np.float64),
            price_strike=original.quote_strike[:1].copy(),
            audit=replace(
                original.audit,
                nms_eligible_books=1,
                eligible_expiries=1,
            ),
        )
        return replace(
            changed,
            market_fingerprint=recompute_market_fingerprint(changed, source.artifact),
        )

    single_current = one_quote_day(zero_day, zero_day.trading_date)
    single_next = one_quote_day(zero_day, date(2024, 1, 4))
    outside_strike = np.asarray((single_current.expiry_strike_upper[0] * 1.01,), dtype=np.float64)
    invalid_unsigned = DayProxy(
        single_current,
        quote_strike=outside_strike,
        price_strike=outside_strike,
        market_fingerprint="0" * 64,
    )
    invalid_day = DayProxy(
        invalid_unsigned,
        market_fingerprint=recompute_market_fingerprint(invalid_unsigned, source.artifact),
    )
    containment_digest = logical_content_digest(
        (invalid_day,),
        config_id=canonical_sha256(config),
        source=source.artifact,
        model_business_dtes=np.asarray(config.geometry.model_business_dtes, dtype=np.float64),
        weekly_origin=np.asarray((), dtype=np.int64),
        weekly_endpoint=np.asarray((), dtype=np.int64),
    )
    containment_hdf = tmp_path / "outside-stored-bounds.h5"
    write_experiment_hdf5(
        containment_hdf,
        (single_current,),
        config=config,
        producer_version=result.plan.spec.producer_version,
    )
    h5py = pytest.importorskip("h5py")
    with h5py.File(containment_hdf, "r+") as handle:
        handle["quote/strike"][0, 0] = outside_strike[0]
        handle["price/strike"][0, 0] = outside_strike[0]
        handle["day/market_fingerprint"][0] = np.bytes_(invalid_day.market_fingerprint)
        handle.attrs["logical_content_sha256"] = containment_digest
    with pytest.raises(ExperimentValidationError, match="outside stored NMS bounds"):
        validate_experiment_hdf5(
            containment_hdf,
            config=config,
            producer_version=result.plan.spec.producer_version,
        )

    oversized_hdf = tmp_path / "non-maximal-dimensions.h5"
    hdf_progress: list[dict[str, object]] = []
    clock = [-301.0]

    def tick() -> float:
        clock[0] += 301.0
        return clock[0]

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(hdf5_module.time, "monotonic", tick)
        write_experiment_hdf5(
            oversized_hdf,
            (zero_day, single_next),
            config=config,
            producer_version=result.plan.spec.producer_version,
            progress=hdf_progress.append,
            progress_interval_seconds=300.0,
        )
    assembly_progress = [
        event for event in hdf_progress if event["event"] == "experiment_hdf5_assembly_progress"
    ]
    digest_progress = [
        event for event in hdf_progress if event["event"] == "experiment_hdf5_digest_progress"
    ]
    assert [event["completed"] for event in digest_progress] == [1, 2]
    assert all(event["total"] == 2 for event in digest_progress)
    assert digest_progress[0]["elapsed_seconds"] == 301.0
    assert digest_progress[0]["eta_seconds"] == 301.0
    assert [event["completed"] for event in assembly_progress] == [1, 2]
    assert all(event["total"] == 2 for event in assembly_progress)
    assert assembly_progress[0]["elapsed_seconds"] == 301.0
    assert assembly_progress[0]["eta_seconds"] == 301.0
    weekly_origin, weekly_endpoint = exact_weekly_pairs(
        (single_current.trading_date, single_next.trading_date)
    )
    resigned_digest = logical_content_digest(
        (single_current, single_next),
        config_id=canonical_sha256(config),
        source=source.artifact,
        model_business_dtes=np.asarray(config.geometry.model_business_dtes, dtype=np.float64),
        weekly_origin=weekly_origin,
        weekly_endpoint=weekly_endpoint,
    )
    with h5py.File(oversized_hdf, "r+") as handle:
        handle["day/quote_count"][0] = 1
        handle["day/price_count"][0] = 1
        handle["day/expiry_count"][0] = 1
        handle["day/market_fingerprint"][0] = np.bytes_(single_current.market_fingerprint)
        quote_fields = {
            "expiry_index": (single_current.quote_expiry_index, 0),
            "price_index": (single_current.quote_price_index, 0),
            "strike": (single_current.quote_strike, 1.0),
            "is_call": (single_current.quote_is_call, True),
            "bid": (single_current.quote_bid, 0.0),
            "ask": (single_current.quote_ask, 1.0),
            "fit": (single_current.quote_fit, 0.0),
            "log_volume": (single_current.quote_log_volume, 0.0),
            "log_open_interest": (single_current.quote_log_open_interest, 0.0),
        }
        handle["quote/mask"][0, :] = False
        handle["quote/mask"][0, 0] = True
        for name, (values, fill) in quote_fields.items():
            handle[f"quote/{name}"][0, :] = fill
            handle[f"quote/{name}"][0, 0] = values[0]
        handle["price/mask"][0, :] = False
        handle["price/mask"][0, 0] = True
        handle["price/expiry_y"][0, :] = 0.0
        handle["price/expiry_y"][0, 0] = single_current.price_expiry_y[0]
        handle["price/strike"][0, :] = 1.0
        handle["price/strike"][0, 0] = single_current.price_strike[0]
        for name in (
            "active_european_books",
            "horizon_eligible_books",
            "nms_eligible_books",
            "input_expiries",
            "eligible_expiries",
        ):
            handle[f"audit/{name}"][0] = getattr(single_current.audit, name)
        handle.attrs["logical_content_sha256"] = resigned_digest
    with pytest.raises(ExperimentValidationError, match="exact retained maxima"):
        validate_experiment_hdf5(
            oversized_hdf,
            config=config,
            producer_version=result.plan.spec.producer_version,
        )

    zero_pricer = smooth_sanos_pricer(zero_expiry_frame)
    _run_pinned_selector_oracle(
        zero_day,
        boundary_day=boundary_day,
        boundary_fields={
            "expiry_y": np.asarray(0.0),
            "strikes": boundary_strikes,
            "is_american": is_american,
            "fit_available": pinned_available,
            "call_active": call_active,
            "put_active": put_active,
            "call_fit": boundary_fit,
            "put_fit": boundary_fit,
        },
        pricer_expiries=np.asarray(zero_pricer.expiries, dtype=np.float64),
        pricer_variances=np.asarray(zero_pricer.variances, dtype=np.float64),
        model_business_dtes=config.geometry.model_business_dtes,
        npz_path=tmp_path / "pinned-selector-input.npz",
    )
    _run_pinned_flex_consumer_if_available(zero_hdf)

    hdf_path = result.path / "data.h5"
    _run_pinned_flex_consumer_if_available(hdf_path)
    link_mutations = {
        "dangling-soft-link": lambda handle: handle.__setitem__(
            "unexpected", h5py.SoftLink("/does-not-exist")
        ),
        "external-link": lambda handle: handle.__setitem__(
            "unexpected", h5py.ExternalLink("missing.h5", "/data")
        ),
        "hard-link-alias": lambda handle: handle.__setitem__("unexpected", handle["quote/bid"]),
        "dataset-attribute": lambda handle: handle["quote/bid"].attrs.__setitem__(
            "unexpected", "value"
        ),
    }
    for label, mutate in link_mutations.items():
        changed = tmp_path / f"{label}.h5"
        shutil.copyfile(hdf_path, changed)
        with h5py.File(changed, "r+") as handle:
            mutate(handle)
        with pytest.raises(ExperimentValidationError):
            validate_experiment_hdf5(
                changed,
                config=config,
                producer_version=result.plan.spec.producer_version,
            )

    shutil.rmtree(store.path_for(source.artifact))
    reused = build_experiment(locations, config)
    assert not reused.built
    assert reused.reuse is not None
    assert not reused.source_accessed
    assert reused.validation == result.validation
    assert validate_experiment(store, reused.artifact, deep_lineage=False) == result.validation
    with pytest.raises(ArtifactNotFoundError):
        validate_experiment(store, reused.artifact, deep_lineage=True)
    with pytest.warns(RuntimeWarning, match="version='\\*'"):
        wildcard = build_experiment(
            locations,
            config,
            ExperimentBuildOptions(version_override="*"),
        )
    assert wildcard.reuse is not None and wildcard.reuse.used_wildcard
    assert not wildcard.source_accessed

    hdf_path = reused.path / "data.h5"
    with h5py.File(hdf_path, "r+") as handle:
        handle["quote/bid"][0, 0] += 1.0e-6
    with pytest.raises(
        ExperimentValidationError,
        match="Market fingerprint|logical content digest",
    ):
        validate_experiment_hdf5(
            hdf_path,
            config=config,
            producer_version=reused.plan.spec.producer_version,
        )


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
