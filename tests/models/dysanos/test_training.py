from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sanos.models.dysanos.config import (
    DailyFrontGeometryConfig,
    DirectModelConfig,
    DirectTrainingConfig,
    ObjectiveConfig,
)
from sanos.models.dysanos.dynamics import DirectDynamicsModel, affine_svd_initialization
from sanos.models.dysanos.objective import ObjectiveEvaluation, TensorDay, make_batch
from sanos.models.dysanos.surface import bound_native_state, native_state_width
from sanos.models.dysanos.training import (
    _clip_sparse_gradient_,
    _date_weighted_epoch_loss,
    _require_no_quote_objective_regression,
    convergence_audit,
    fine_tune_affine_quotes,
    fit_joint_direct,
    fit_native_daily_states,
    native_raw_from_day,
)


def _setup() -> tuple[object, ...]:
    geometry = replace(
        DailyFrontGeometryConfig(),
        model_business_dtes=(1.0, 5.0, 10.0),
        num_inner_strikes=10,
    )
    config = replace(
        DirectModelConfig(),
        state_size=2,
        decoded_state_size=native_state_width(geometry),
    )
    generator = torch.Generator().manual_seed(18)
    native = torch.randn(
        (6, native_state_width(geometry)), generator=generator, dtype=torch.float64
    )
    initialization = affine_svd_initialization(native, config.state_size)
    model = DirectDynamicsModel(6, geometry, config, dtype=torch.float64)
    model.install_affine_initialization(initialization)
    days = tuple(
        TensorDay(
            source_index=index,
            price_expiries_y=torch.tensor([1.0 / 252.0], dtype=torch.float64),
            price_strikes=torch.tensor([1.0], dtype=torch.float64),
            quote_price_index=torch.tensor([0]),
            quote_strikes=torch.tensor([1.0], dtype=torch.float64),
            quote_is_call=torch.tensor([True]),
            quote_fit=torch.tensor([0.02], dtype=torch.float64),
            quote_half_spread=torch.tensor([0.02], dtype=torch.float64),
        )
        for index in range(6)
    )
    return model, initialization, make_batch(days)


def test_joint_checkpoint_resume_is_deterministic(tmp_path) -> None:
    model, initialization, batch = _setup()
    origin = torch.tensor([0, 1, 2, 3])
    endpoint = torch.tensor([1, 2, 3, 4])
    objective = ObjectiveConfig()
    first_config = replace(
        DirectTrainingConfig(),
        num_epochs=1,
        checkpoint_every=1,
        warmup_epochs=0,
        dynamics_ramp_epochs=1,
        day_batch_size=2,
        dynamics_steps_per_epoch=2,
        compile_sanos_solver=False,
    )
    checkpoint = tmp_path / "direct.pt"
    fit_joint_direct(
        model,
        batch,
        origin,
        endpoint,
        objective,
        first_config,
        initialization,
        configuration_id="tiny-v1",
        source_id="source-v1",
        checkpoint_path=checkpoint,
        require_convergence=False,
    )
    resumed_config = replace(first_config, num_epochs=2)
    resumed = fit_joint_direct(
        model,
        batch,
        origin,
        endpoint,
        objective,
        resumed_config,
        initialization,
        configuration_id="tiny-v1",
        source_id="source-v1",
        checkpoint_path=checkpoint,
        require_convergence=False,
    )
    assert resumed.completed_epochs == 2
    assert [record["epoch"] for record in resumed.history] == [1.0, 2.0]

    replay_model, replay_initialization, replay_batch = _setup()
    replay = fit_joint_direct(
        replay_model,
        replay_batch,
        origin,
        endpoint,
        objective,
        resumed_config,
        replay_initialization,
        configuration_id="fresh-v1",
        source_id="source-v1",
        require_convergence=False,
    )
    for expected, actual in zip(
        replay.model.state_dict().values(), resumed.model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_checkpoint_identity_and_symlink_fail_closed(tmp_path) -> None:
    model, initialization, batch = _setup()
    origin = torch.tensor([0, 1, 2, 3])
    endpoint = torch.tensor([1, 2, 3, 4])
    training = replace(
        DirectTrainingConfig(),
        num_epochs=1,
        checkpoint_every=1,
        warmup_epochs=0,
        compile_sanos_solver=False,
    )
    checkpoint = tmp_path / "direct.pt"
    fit_joint_direct(
        model,
        batch,
        origin,
        endpoint,
        ObjectiveConfig(),
        training,
        initialization,
        configuration_id="one",
        source_id="source-v1",
        checkpoint_path=checkpoint,
        require_convergence=False,
    )
    with pytest.raises(ValueError, match="identity"):
        fit_joint_direct(
            model,
            batch,
            origin,
            endpoint,
            ObjectiveConfig(),
            replace(training, num_epochs=2),
            initialization,
            configuration_id="two",
            source_id="source-v1",
            checkpoint_path=checkpoint,
            require_convergence=False,
        )
    link = tmp_path / "linked.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="Unsafe"):
        fit_joint_direct(
            model,
            batch,
            origin,
            endpoint,
            ObjectiveConfig(),
            training,
            initialization,
            configuration_id="one",
            source_id="source-v1",
            checkpoint_path=link,
            require_convergence=False,
        )


def test_native_initialization_uses_public_day_fields_only() -> None:
    geometry = replace(
        DailyFrontGeometryConfig(),
        model_business_dtes=(1.0, 5.0, 10.0),
        num_inner_strikes=10,
    )

    class PublicDay:
        model_atm_total_vol = torch.tensor([0.01, 0.025, 0.04]).numpy()
        price_expiry_y = torch.tensor([1.0 / 252.0]).numpy()
        price_strike = torch.tensor([1.0]).numpy()
        quote_price_index = torch.tensor([0]).numpy()
        quote_strike = torch.tensor([1.0]).numpy()
        quote_is_call = torch.tensor([True]).numpy()
        quote_fit = torch.tensor([0.02]).numpy()
        quote_half_spread = torch.tensor([0.02]).numpy()

    result = fit_native_daily_states(
        (PublicDay(),),  # type: ignore[arg-type]
        geometry,
        ObjectiveConfig(),
        num_epochs=1,
        learning_rate=1.0e-4,
        dtype=torch.float64,
        require_convergence=False,
    )
    assert result.raw_states.shape == (1, native_state_width(geometry))
    assert len(result.history) == 1
    assert result.evaluation.book_count == 1


def test_native_initialization_reconstructs_public_atm_curve() -> None:
    geometry = replace(
        DailyFrontGeometryConfig(),
        model_business_dtes=(1.0, 5.0, 10.0),
        num_inner_strikes=10,
    )

    class PublicDay:
        model_atm_total_vol = torch.tensor([0.01, 0.025, 0.04]).numpy()

    raw = native_raw_from_day(
        PublicDay(),  # type: ignore[arg-type]
        geometry,
        dtype=torch.float64,
        device="cpu",
    )
    parameters = bound_native_state(raw.unsqueeze(0), geometry)
    expiries = torch.tensor(geometry.model_business_dtes, dtype=torch.float64) / 252.0
    expected_annual = torch.tensor(
        PublicDay.model_atm_total_vol, dtype=torch.float64
    ) / torch.sqrt(expiries)
    torch.testing.assert_close(
        parameters.dlv_squares[0], expected_annual.square().unsqueeze(-1).expand(-1, 10)
    )
    torch.testing.assert_close(parameters.eta[0], torch.full((3,), 0.20, dtype=torch.float64))


def test_native_sparse_checkpoint_resume_is_bitwise_deterministic(tmp_path) -> None:
    geometry = replace(
        DailyFrontGeometryConfig(),
        model_business_dtes=(1.0, 5.0, 10.0),
        num_inner_strikes=10,
    )

    class PublicDay:
        model_atm_total_vol = torch.tensor([0.01, 0.025, 0.04]).numpy()
        price_expiry_y = torch.tensor([1.0 / 252.0]).numpy()
        price_strike = torch.tensor([1.0]).numpy()
        quote_price_index = torch.tensor([0]).numpy()
        quote_strike = torch.tensor([1.0]).numpy()
        quote_is_call = torch.tensor([True]).numpy()
        quote_fit = torch.tensor([0.02]).numpy()
        quote_half_spread = torch.tensor([0.02]).numpy()

    days = (PublicDay(), PublicDay())  # type: ignore[assignment]
    checkpoint = tmp_path / "direct.pt"
    fit_native_daily_states(
        days,  # type: ignore[arg-type]
        geometry,
        ObjectiveConfig(),
        num_epochs=1,
        learning_rate=1.0e-4,
        dtype=torch.float64,
        day_batch_size=1,
        require_convergence=False,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        configuration_id="config-v1",
        source_id="source-v1",
    )
    resumed = fit_native_daily_states(
        days,  # type: ignore[arg-type]
        geometry,
        ObjectiveConfig(),
        num_epochs=2,
        learning_rate=1.0e-4,
        dtype=torch.float64,
        day_batch_size=1,
        require_convergence=False,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        configuration_id="config-v1",
        source_id="source-v1",
    )
    replay = fit_native_daily_states(
        days,  # type: ignore[arg-type]
        geometry,
        ObjectiveConfig(),
        num_epochs=2,
        learning_rate=1.0e-4,
        dtype=torch.float64,
        day_batch_size=1,
        require_convergence=False,
    )
    torch.testing.assert_close(resumed.raw_states, replay.raw_states, rtol=0.0, atol=0.0)
    assert resumed.history == replay.history


def test_affine_sparse_checkpoint_resume_is_bitwise_deterministic(tmp_path) -> None:
    model, _, batch = _setup()
    training = replace(DirectTrainingConfig(), day_batch_size=1)
    checkpoint = tmp_path / "direct.pt"
    fine_tune_affine_quotes(
        model,
        batch,
        ObjectiveConfig(),
        training,
        num_epochs=1,
        require_convergence=False,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        configuration_id="config-v1",
        source_id="source-v1",
    )
    fine_tune_affine_quotes(
        model,
        batch,
        ObjectiveConfig(),
        training,
        num_epochs=2,
        require_convergence=False,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        configuration_id="config-v1",
        source_id="source-v1",
    )
    replay_model, _, replay_batch = _setup()
    fine_tune_affine_quotes(
        replay_model,
        replay_batch,
        ObjectiveConfig(),
        training,
        num_epochs=2,
        require_convergence=False,
    )
    for expected, actual in zip(
        replay_model.state_dict().values(), model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_convergence_audit_fails_closed_without_trailing_window() -> None:
    result = convergence_audit(
        ({"epoch": 1.0, "total": 1.0, "dynamics_ramp": 1.0},),
        window=1,
        relative_tolerance=0.01,
    )
    assert result == {
        "converged": False,
        "reason": "insufficient_or_nonfinite_history",
        "num_epochs": 1,
        "num_comparable_epochs": 1,
    }


def test_sparse_and_dense_gradient_norms_are_clipped_independently() -> None:
    embedding = torch.nn.Embedding(3, 2, sparse=True, dtype=torch.float64)
    decoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    selected = embedding(torch.tensor([0, 2]))
    loss = 100.0 * selected.sum() + 100.0 * decoder(selected).sum()
    loss.backward()

    sparse_before = _clip_sparse_gradient_(embedding.weight, 0.5)
    dense_before = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.25))
    sparse_values = embedding.weight.grad.coalesce().values()  # type: ignore[union-attr]
    dense_norm = torch.linalg.vector_norm(
        torch.cat(tuple(parameter.grad.reshape(-1) for parameter in decoder.parameters()))
    )
    assert sparse_before > 0.5
    assert dense_before > 0.25
    torch.testing.assert_close(
        torch.linalg.vector_norm(sparse_values), torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(dense_norm, torch.tensor(0.25, dtype=torch.float64))


def test_date_weighting_and_quote_regression_policy_are_exact() -> None:
    # A final two-date minibatch must not receive the weight of a four-date batch.
    assert _date_weighted_epoch_loss(4.0 * 2.0 + 2.0 * 8.0, 6) == 4.0
    initial = 0.5
    accepted = ObjectiveEvaluation(0.5 + 1.0e-10, 0.0, 0.0, 0.0, 1)
    _require_no_quote_objective_regression(initial, accepted, phase="test")
    rejected = ObjectiveEvaluation(0.5 + 1.1e-10, 0.0, 0.0, 0.0, 1)
    with pytest.raises(RuntimeError, match="objective regressed"):
        _require_no_quote_objective_regression(initial, rejected, phase="test")
