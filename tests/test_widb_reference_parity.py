"""Executable bitwise fixture from pinned quant_data revision de15be96."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import polars as pl
from cdxcore.bs import bs

from sanos.widb import (
    ENRICHED_SCHEMA,
    DailyRawInputs,
    enrich_daily_frame,
    pair_daily_frame,
)

_REFERENCE_FLOAT32_SHA256 = {
    "spot": "409e43d533cb2908b4674913e362169f116ae7c42b481b0c693c76cd52934eb2",
    "previous_spot": "52880f582900835e91195fe097abd93b06e34e1845d846376590e522a98590e8",
    "strike": "c811be8e5afab069597addcd179d3def2d235b106687b9e3dc5154e96549d54c",
    "vendor_forward": "05e142cd2d7e71dcb9263b031e3cd033cc7bcb05be6d7c3913b0b8bb470c22af",
    "discount_factor": "c185182f38f4462227a15c46b3e4899262ce175aa7781b5d5f6f238c92d3a32a",
    "business_dte": "4981b00a6fd6703e6a38dd62c630f290b403513f16165c3c3d777af1cce26b30",
    "time_to_expiry_years": "af3988a11ad76f3bd655386171f27fd429b333a37ec383c55edc01680e9e36a5",
    "market_call_ask": "a900d84634396a58292128865d3d562f58fab6cf27fab34186fd3c10d9020414",
    "market_call_bid": "e6b613ec0bc28e49dcd420a70dcd6a9c9449cbabddb620a9ffa294dff656c474",
    "market_call_log_volume": "6984acd3333dabac67d879ec1ed072e706c26e50deff92f26000124aab293d89",
    "market_call_log_open_interest": (
        "f589ab51d4e82a19261dd342594308eb49bc392781a9bd8fc626aecf4683338e"
    ),
    "market_put_ask": "19f4cf0d271101d53f1954e8ace471938ec5e689eed2d4029e6102064838cc9d",
    "market_put_bid": "68812e077eabcff55f58650fbbbbb2c9f08a9a5c94f3c3e4abdd9fb960974c02",
    "market_put_log_volume": "6984acd3333dabac67d879ec1ed072e706c26e50deff92f26000124aab293d89",
    "market_put_log_open_interest": (
        "f589ab51d4e82a19261dd342594308eb49bc392781a9bd8fc626aecf4683338e"
    ),
    "implied_forward": "8862c3db65ce64b47a376870cf0693470d17661fbe88ab5ce58520a7d238ca55",
    "normalized_moneyness": "9bb7df9ada92bdf45d9a1f10a0eb80e07b24067cc7a34bcaf944d272f59b7f04",
    "atm_normalized_moneyness": "09557e8cd53db41ade3acf983775b8f153aef9211bf4d1136e7afe7a213e2716",
    "synthetic_call_ask": "602b396406d2946a7df55f60883310f69647b18a1d22bef8413132ab16549d86",
    "synthetic_call_bid": "3b15cadd43010eeee4b5249ab2d573c0dceb3218f32baaaf6c06f08f7dba00fb",
    "ask_implied_volatility": "9b98b383472fa22253a493bc034ceb41765b70fcb65cae5319b28fbb533ebe95",
    "bid_implied_volatility": "63cecd8fbbabe0d37d4d759fdba840a231ac450708ac7ac8987ef9e08a3b2bee",
    "combined_log_volume": "d22c12fad6d999ac0840d610e5051b5699414ceec7d904099cc3bd13cbd2683e",
    "combined_log_open_interest": (
        "21a76f158ac0d4c5260e289a7aa21085ce5e49a89388e17940209376ad9aa082"
    ),
    "normalized_strike": "1d9d38c27da59394bf4e982c5a3787a9992531dca4cd2b2838ddbadd8ee8cb03",
    "pure_ask_price": "5fe3cfd145356236f70cd3cd1acd715e74d5579807aba84abc42a6f222ce7c8c",
    "pure_bid_price": "93bd5370ff8c4052c6dcf27239bbba116ee44eb9e790846248e0cb1a32a580ed",
    "pure_ask_delta": "ab7972d91c1e6936948137a4178c210386c74b3812a00fe23b530c0143bf3b83",
    "pure_bid_delta": "6baa8d08ac338583d87306d5b3864207e48c27afff9df2c8a1efad3701939b40",
    "pure_ask_gamma": "2b86d08768ae6e33f9811ca8670adee645ddfb21690a755a28a1bf0f7f42696b",
    "pure_bid_gamma": "eb953425143eb28819b6ad3fae17ecba17dd0c3f78e4b6f590c4abaaa080baee",
    "pure_ask_vega": "a88ae6d05d59ce0d95ff5538c0b93f26dc5c5ee1a0304bccd9c1c3fc9e00c4a9",
    "pure_bid_vega": "54401aaf6bd5936df6905f88e648f09f452b72dd24ae36c1ec4352945cf21cf1",
    "pure_ask_theta": "ae68d647a9489462f09dcd3a2c2b8e7faaadfc2a05d24644cdaf3d3e74a489b7",
    "pure_bid_theta": "68bb2ccf931ecf0068537ddfe193560340dfae60412e7094abc5ea8ab3d2eb8d",
    "pure_ask_dk": "3d41ccc4f5d2263a9c6e64df0b3327f0947a3ca6dbddec5a7a3a3865b12e0341",
    "pure_bid_dk": "bd9a249a055c6c4e92683931b64ce3169c4a7316b12744aea2198207ee6f5407",
}


def _reference_probe_inputs() -> DailyRawInputs:
    trading_date = date(2024, 1, 3)
    expiry = date(2024, 1, 19)
    spot = 4321.37
    previous_spot = 4312.19
    forward = 4330.25
    strikes = np.arange(4000.0, 4660.0, 50.0)
    discount = np.exp(-0.0475 * (expiry - trading_date).days / 365.0)
    calls = (
        np.asarray(bs(k=strikes / forward, vol=0.235, sqrtT=np.sqrt(11.0 / 252.0)))
        * discount
        * forward
    )
    puts = calls - discount * (forward - strikes)
    rows: list[dict[str, object]] = []
    for index, (strike, call, put) in enumerate(zip(strikes, calls, puts, strict=True)):
        for side, price in (("C", call), ("P", put)):
            rows.append(
                {
                    "secid": 108105,
                    "date": trading_date.isoformat(),
                    "symbol": f"SPX-{side}-{index}",
                    "optionid": index * 2 + int(side == "P"),
                    "am_settlement": 0,
                    "cp_flag": side,
                    "strike_price": int(strike * 1000),
                    "exdate": expiry.isoformat(),
                    "best_bid": float(
                        0.02
                        if side == "P" and index == len(strikes) - 1
                        else price - (0.12 if side == "C" else 0.08)
                    ),
                    "best_offer": float(price + (0.15 if side == "C" else 0.11)),
                    "volume": index + 1,
                    "open_interest": 100 + index,
                    "impl_volatility": 0.235,
                    "ss_flag": "0",
                    "delta": 0.0,
                    "gamma": 0.0,
                    "vega": 0.0,
                    "theta": 0.0,
                    "expiry_indicator": "w",
                    "contract_size": 100,
                }
            )
    return DailyRawInputs(
        security=pl.DataFrame({"ticker": ["SPX"], "secid": [108105]}),
        options=pl.DataFrame(rows),
        spot=pl.DataFrame({"secid": [108105], "date": [trading_date.isoformat()], "close": [spot]}),
        previous_spot=pl.DataFrame(
            {"secid": [108105], "date": ["2024-01-02"], "close": [previous_spot]}
        ),
        forwards=pl.DataFrame(
            {
                "secid": [108105],
                "date": [trading_date.isoformat()],
                "expiration": [expiry.isoformat()],
                "amsettlement": [0],
                "forwardprice": [forward],
            }
        ),
        zero_curve=pl.DataFrame(
            {
                "date": [trading_date.isoformat()] * 3,
                "days": [1, 30, 90],
                "rate": [5.0, 4.0, 3.0],
            }
        ),
    )


def test_pinned_reference_enrichment_float32_vectors_are_bitwise_exact() -> None:
    """Freeze the reference subprocess output without requiring quant_data at runtime."""

    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_reference_probe_inputs(),
    )
    assert paired.get_column("spot_return").item(0) == np.float32(0.0021288395)
    enriched = enrich_daily_frame(paired)
    assert enriched.schema == pl.Schema(ENRICHED_SCHEMA)
    assert enriched.height == 13
    assert enriched.get_column("call_option_id").to_list() == list(range(0, 26, 2))
    assert enriched.get_column("put_option_id").to_list() == list(range(1, 27, 2))
    normalization = json.loads(enriched.get_column("normalization_diagnostics_json").item(0))
    assert normalization["filter_counts"]["bad_settlement"] == {
        "after": 28,
        "before": 28,
        "removed": 0,
    }
    assert normalization["unpaired_call_rows"] == 0
    assert normalization["unpaired_put_rows"] == 0
    assert normalization["unmatched_forward_rows"] == 0
    enrichment = json.loads(enriched.get_column("enrichment_diagnostics_json").item(0))
    assert enrichment["dropped_rows"] == 1
    assert enrichment["row_rejections"]["synthetic_price"] == 1
    assert enrichment["output_rows_by_expiry"] == {"202401191": 13}
    assert set(enrichment["ranges"]) == {
        "ask_implied_volatility",
        "atm_normalized_moneyness",
        "bid_implied_volatility",
        "implied_forward",
        "normalized_moneyness",
        "normalized_strike",
        "pure_ask_price",
        "pure_bid_price",
    }

    observed = {
        name: hashlib.sha256(
            enriched.get_column(name).to_numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest()
        for name in _REFERENCE_FLOAT32_SHA256
    }
    assert observed == _REFERENCE_FLOAT32_SHA256
