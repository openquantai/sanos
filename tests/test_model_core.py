from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cvxpy as cp
import numpy as np
import pytest
from cdxcore.bs import bs
from cdxcore.verbose import Context
from cvxpy.error import SolverError

import sanos.model.fitter as fitter_module
from sanos.model import (
    ArbFreePricer,
    MkArbFree,
    SanosFitter,
    SanosLPSolver,
    SanosPricer,
    make_variances,
    repair_density_nonnegative,
    solve_sanos_tridiagonal,
)
from sanos.model.backbone import BackboneNumpy
from sanos.model.tools import _highs_failure_details

FIXTURE = Path(__file__).parent / "fixtures" / "reference_numerics_v1.json"


@pytest.mark.parametrize("cache", [None, [], {"HIGHS": ()}, {"HIGHS": (None, None, None)}])
def test_solver_failure_diagnostics_tolerate_unsupported_cache(cache: Any) -> None:
    problem = cp.Problem(cp.Minimize(0))
    problem._solver_cache = cache
    assert _highs_failure_details(problem) == ""


def _reference_inputs() -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    expiries = np.array([5.0 / 252.0, 10.0 / 252.0], dtype=np.float64)
    strikes = [
        np.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
        np.array([0.4, 0.7, 1.0, 1.3, 1.6], dtype=np.float64),
    ]
    dummy_densities = [
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    ]
    dlvsqs = [
        np.array([0.04, 0.09, 0.16], dtype=np.float64),
        np.array([0.05, 0.10, 0.15], dtype=np.float64),
    ]
    return expiries, strikes, dummy_densities, dlvsqs


def test_standalone_core_matches_pinned_reference_fixture() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expiries, strikes, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=dummy_densities,
    )
    densities = template.densities_from_dlvsqs(input_dlvsqs)
    pricer = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=densities,
    )
    prices = pricer(
        expiries[:, None],
        np.array([[0.6, 0.8, 1.0, 1.2, 1.4]], dtype=np.float64),
    )
    repaired = repair_density_nonnegative(
        p=np.array([-0.08, 0.52, 0.41, 0.16], dtype=np.float64),
        x=np.array([0.45, 0.8, 1.1, 1.65], dtype=np.float64),
    )

    for observed, field in (
        (densities, "densities"),
        (pricer.dlvsqs(), "dlvsqs"),
        (prices, "prices"),
        (repaired, "repair"),
    ):
        np.testing.assert_allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected[field], dtype=np.float64),
            rtol=1.0e-13,
            atol=1.0e-14,
        )


def test_density_dlv_round_trip_and_call_invariants_on_unequal_grids() -> None:
    expiries, strikes, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=dummy_densities,
    )
    densities = template.densities_from_dlvsqs(input_dlvsqs)
    pricer = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=densities,
    )

    recovered = pricer.dlvsqs()
    reconstructed = pricer.densities_from_dlvsqs(recovered)
    for strike_grid, density, original, expected_dlv, observed_dlv in zip(
        strikes,
        reconstructed,
        densities,
        input_dlvsqs,
        recovered,
        strict=True,
    ):
        np.testing.assert_allclose(density, original, rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(observed_dlv, expected_dlv, rtol=0.0, atol=1.0e-12)
        assert np.min(density) >= 0.0
        assert np.sum(density) == pytest.approx(1.0, abs=1.0e-12)
        assert strike_grid @ density == pytest.approx(1.0, abs=1.0e-12)

    query = np.linspace(0.0, 2.0, 401, dtype=np.float64)
    prices = np.asarray(pricer(float(expiries[-1]), query))
    zero_smooth = SanosPricer(
        smoothness=0.0,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=densities,
    )
    np.testing.assert_allclose(
        zero_smooth(float(expiries[-1]), query), prices, rtol=0.0, atol=2.0e-16
    )
    assert prices[0] == pytest.approx(1.0, abs=1.0e-12)
    assert prices[-1] == pytest.approx(0.0, abs=1.0e-12)
    assert (prices[1] - prices[0]) / (query[1] - query[0]) == pytest.approx(-1.0, abs=1.0e-12)
    assert np.all(prices >= np.maximum(1.0 - query, 0.0) - 1.0e-12)
    assert np.all(prices <= 1.0 + 1.0e-12)
    assert np.all(np.diff(prices) <= 1.0e-12)
    assert np.all(np.diff(prices, n=2) >= -1.0e-12)

    full_dlvsqs = pricer.dlvsqs(on_all_strikes=True)
    for strike_grid, dlvsq in zip(strikes, full_dlvsqs, strict=True):
        assert dlvsq.shape == strike_grid.shape
        np.testing.assert_array_equal(dlvsq[[0, -1]], np.zeros(2))
    reconstructed_full = pricer.densities_from_dlvsqs(full_dlvsqs)
    for observed, expected in zip(reconstructed_full, densities, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-14)


def test_dlv_extraction_tolerates_a_nonrepresentable_sparse_audit_state() -> None:
    """Sparse audit densities need not define a finite standalone DLV state."""

    strikes = [np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)]
    densities = [np.asarray([0.2, 0.0, 0.6, 0.0, 0.2], dtype=np.float64)]
    pricer = SanosPricer(
        smoothness=None,
        expiries=np.asarray([5.0 / 252.0], dtype=np.float64),
        variances=np.asarray([5.0 / 252.0 * 0.04], dtype=np.float64),
        xstrikes=strikes,
        xdensities=densities,
    )

    dlv_squares = pricer.dlvsqs(on_all_strikes=True)

    assert len(dlv_squares) == 1
    assert np.all(np.isfinite(dlv_squares[0]))
    assert dlv_squares[0][1] == 0.0
    reconstructed = pricer.densities_from_dlvsqs(dlv_squares)
    assert not np.allclose(reconstructed[0], densities[0], rtol=0.0, atol=1.0e-12)


def test_pre_first_expiry_jump_is_preserved_for_parity() -> None:
    expiries, strikes, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=dummy_densities,
    )
    pricer = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=template.densities_from_dlvsqs(input_dlvsqs),
    )
    query = np.array([0.8, 1.0, 1.2], dtype=np.float64)
    just_before = np.asarray(pricer(float(expiries[0] / 2.0), query))
    at_first = np.asarray(pricer(float(expiries[0]), query))
    np.testing.assert_allclose(just_before, at_first, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        pricer(0.0, query),
        np.maximum(1.0 - query, 0.0),
        rtol=0.0,
        atol=0.0,
    )


def test_tridiagonal_kernel_matches_dense_numpy_and_jax_gradient() -> None:
    w_minus = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    w_plus = np.array([0.1, 0.25, 0.35], dtype=np.float64)
    rhs = np.array([0.01, 0.12, 0.68, 0.17, 0.02], dtype=np.float64)
    matrix = np.eye(5, dtype=np.float64)
    matrix[0, 1] = -w_minus[0]
    for index in range(1, 4):
        matrix[index, index] = 1.0 + w_minus[index - 1] + w_plus[index - 1]
        if index > 1:
            matrix[index, index - 1] = -w_plus[index - 2]
        if index < 3:
            matrix[index, index + 1] = -w_minus[index]
    matrix[-1, -2] = -w_plus[-1]
    expected = np.linalg.solve(matrix, rhs)
    observed = solve_sanos_tridiagonal(
        w_minus=w_minus,
        w_plus=w_plus,
        rhs=rhs,
        backbone=BackboneNumpy(),
    )
    np.testing.assert_allclose(observed, expected, rtol=1.0e-14, atol=1.0e-14)

    import jax
    import jax.numpy as jnp

    from sanos.model.jax_backbone import BackboneJax

    jax.config.update("jax_enable_x64", True)
    backbone = BackboneJax()

    def total(candidate: object) -> object:
        return jnp.sum(
            solve_sanos_tridiagonal(
                w_minus=jnp.asarray(candidate),
                w_plus=jnp.asarray(w_plus),
                rhs=jnp.asarray(rhs),
                backbone=backbone,
            )
            ** 2
        )

    value, gradient = jax.value_and_grad(total)(jnp.asarray(w_minus))
    assert np.isfinite(float(cast(Any, value)))
    assert np.all(np.isfinite(np.asarray(gradient)))
    finite_difference = np.empty_like(w_minus)
    step = 1.0e-6
    for index in range(len(w_minus)):
        upper = w_minus.copy()
        lower = w_minus.copy()
        upper[index] += step
        lower[index] -= step
        upper_value = float(cast(Any, total(upper)))
        lower_value = float(cast(Any, total(lower)))
        finite_difference[index] = (upper_value - lower_value) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(gradient), finite_difference, rtol=2.0e-7, atol=1.0e-9)


def test_density_projection_respects_constraints() -> None:
    x = np.array([0.7, 0.95, 1.25, 1.8], dtype=np.float64)
    repaired = repair_density_nonnegative(
        p=np.array([0.01, 0.72, 0.18, 0.09], dtype=np.float64),
        x=x,
        target_sum=1.2,
        target_xmoment=1.32,
        floor=0.03,
    )
    assert np.min(repaired) >= 0.03 - 1.0e-10
    assert np.sum(repaired) == pytest.approx(1.2, abs=1.0e-10)
    assert repaired @ x == pytest.approx(1.32, abs=1.0e-10)

    projected = cp.Variable(len(x))
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(projected - np.array([0.01, 0.72, 0.18, 0.09]))),
        [projected >= 0.03, cp.sum(projected) == 1.2, x @ projected == 1.32],
    )
    problem.solve(solver="CLARABEL")
    assert problem.status in {"optimal", "optimal_inaccurate"}
    assert projected.value is not None
    np.testing.assert_allclose(repaired, projected.value, rtol=0.0, atol=1.0e-7)


def test_variance_geometry_explicit_and_inferred() -> None:
    expiries = np.array([5.0, 10.0, 20.0], dtype=np.float64) / 252.0
    strikes = [np.array([0.9, 1.0, 1.1], dtype=np.float64) for _ in expiries]
    explicit_vols = np.array([0.20, 0.21, 0.22], dtype=np.float64)
    prices = [
        np.asarray(bs.price(k=grid, vol=vol, sqrtT=np.sqrt(expiry)), dtype=np.float64)
        for expiry, grid, vol in zip(expiries, strikes, explicit_vols, strict=True)
    ]
    explicit = make_variances(expiries, strikes, prices, vols=explicit_vols)
    inferred = make_variances(expiries, strikes, prices)
    np.testing.assert_allclose(explicit, expiries * explicit_vols**2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(inferred, explicit, rtol=1.0e-4, atol=1.0e-12)
    assert np.all(np.diff(inferred) > 0.0)


@pytest.mark.parametrize("smoothness", [None, 0.0, 0.25])
def test_lp_fitter_reprices_deterministic_sanos_market(smoothness: float | None) -> None:
    expiries, xstrikes, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(
        smoothness=smoothness,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=xstrikes,
        xdensities=dummy_densities,
    )
    source = SanosPricer(
        smoothness=smoothness,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=xstrikes,
        xdensities=template.densities_from_dlvsqs(input_dlvsqs),
    )
    market_strikes = [
        np.array([0.75, 1.0, 1.25], dtype=np.float64),
        np.array([0.7, 1.0, 1.3], dtype=np.float64),
    ]
    mids = [
        np.asarray(source(float(expiry), strikes), dtype=np.float64)
        for expiry, strikes in zip(expiries, market_strikes, strict=True)
    ]
    fitted = SanosFitter(smoothness=smoothness, timeout_seconds=20.0)(
        expiries,
        market_strikes,
        mids,
        xstrikes,
        vols=0.2,
        verbose=cast(Any, Context).quiet,
    )
    for expiry, strikes, expected in zip(expiries, market_strikes, mids, strict=True):
        observed = np.asarray(fitted(float(expiry), strikes))
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-7)
    for strikes, density in zip(fitted.xstrikes, fitted.xdensities, strict=True):
        assert np.min(density) >= 0.0
        assert np.sum(density) == pytest.approx(1.0, abs=2.0e-7)
        assert strikes @ density == pytest.approx(1.0, abs=2.0e-7)


def test_strict_fitter_rejects_solver_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def forced_fallback(*args: Any, **kwargs: Any) -> tuple[bool, str, bool]:
        del args, kwargs
        return True, "optimal", True

    monkeypatch.setattr(
        fitter_module._StrictSanosLPSolver,
        "__call__",
        forced_fallback,
    )
    expiry = np.asarray([5.0 / 252.0], dtype=np.float64)
    strikes = [np.asarray([0.9, 1.0, 1.1], dtype=np.float64)]
    mids = [
        np.asarray(
            bs.price(k=strikes[0], vol=0.2, sqrtT=np.sqrt(expiry[0])),
            dtype=np.float64,
        )
    ]
    supports = [np.asarray([0.5, 0.9, 1.0, 1.1, 1.5], dtype=np.float64)]
    with pytest.raises(RuntimeError, match="forbids fallback"):
        SanosFitter(
            smoothness=None,
            scale_dlv_constraints=True,
            solver_kkt_tolerance=1.0e-10,
            max_abs_fit_error=9.0e-5,
            allow_solver_fallback=False,
        )(
            expiry,
            strikes,
            mids,
            supports,
            vols=0.2,
            verbose=cast(Any, Context).quiet,
        )


def test_strict_fitter_configures_all_highs_feasibility_tolerances() -> None:
    solver, options = fitter_module._StrictSanosLPSolver(kkt_tolerance=1.0e-10)._lp_solver_settings(
        for_expiry=False, is_fallback=False
    )
    assert solver == "HIGHS"
    highs_options = options["highs_options"]
    for name in (
        "kkt_tolerance",
        "primal_feasibility_tolerance",
        "dual_feasibility_tolerance",
    ):
        assert highs_options[name] == 1.0e-10


def test_fitter_applies_optional_per_node_price_bounds() -> None:
    expiry = np.asarray([5.0 / 252.0], dtype=np.float64)
    strikes = [np.asarray([0.9, 1.0, 1.1], dtype=np.float64)]
    mids = [
        np.asarray(
            bs.price(k=strikes[0], vol=0.2, sqrtT=np.sqrt(expiry[0])),
            dtype=np.float64,
        )
    ]
    supports = [np.asarray([0.5, 0.9, 1.0, 1.1, 1.5], dtype=np.float64)]
    bounds = [(mids[0] - 1.0e-6, mids[0] + 1.0e-6)]
    fitted = SanosFitter(smoothness=None, timeout_seconds=20.0)(
        expiry,
        strikes,
        mids,
        supports,
        fit_price_bounds=bounds,
        vols=0.2,
        verbose=cast(Any, Context).quiet,
    )
    observed = np.asarray(fitted(float(expiry[0]), strikes[0]))
    assert np.all(observed >= bounds[0][0] - 1.0e-7)
    assert np.all(observed <= bounds[0][1] + 1.0e-7)

    with pytest.raises(ValueError, match="same shape"):
        SanosFitter(smoothness=None, timeout_seconds=20.0)(
            expiry,
            strikes,
            mids,
            supports,
            fit_price_bounds=[
                (
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray([1.0], dtype=np.float64),
                )
            ],
            vols=0.2,
            verbose=cast(Any, Context).quiet,
        )

    impossible = [
        (
            np.full(3, 2.0, dtype=np.float64),
            np.full(3, 2.0, dtype=np.float64),
        )
    ]
    with pytest.raises(RuntimeError, match="infeasible"):
        SanosFitter(smoothness=None, timeout_seconds=20.0)(
            expiry,
            strikes,
            mids,
            supports,
            fit_price_bounds=impossible,
            vols=0.2,
            verbose=cast(Any, Context).quiet,
        )


def test_arbfree_surface_contains_quotes_and_preserves_geometry() -> None:
    expiries, xstrikes, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=xstrikes,
        xdensities=dummy_densities,
    )
    source = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=xstrikes,
        xdensities=template.densities_from_dlvsqs(input_dlvsqs),
    )
    strikes = [grid[1:-1].copy() for grid in xstrikes]
    mids = [
        np.asarray(source(float(expiry), strike), dtype=np.float64)
        for expiry, strike in zip(expiries, strikes, strict=True)
    ]
    bids = [mid - 1.0e-4 for mid in mids]
    asks = [mid + 1.0e-4 for mid in mids]
    result = MkArbFree("surface", timeout_seconds=20.0, min_strikes_per_expiry=3)(
        expiries,
        strikes,
        bids,
        asks,
        vols=0.2,
        verbose=cast(Any, Context).quiet,
    )
    assert np.array_equal(result.expiries, expiries)
    assert result.fitsX is not None
    assert result.mstrikes is not None
    assert result.mdensities is not None
    for index, (mask, fit1, fitx) in enumerate(
        zip(result.masks, result.fits1, result.fitsX, strict=True)
    ):
        assert np.all(fit1[mask] >= bids[index][mask] - 1.0e-7)
        assert np.all(fit1[mask] <= asks[index][mask] + 1.0e-7)
        assert np.all(np.isfinite(fitx))
    for strikes_t, density_t in zip(result.mstrikes, result.mdensities, strict=True):
        assert np.min(density_t) >= -1.0e-7
        assert np.sum(density_t) == pytest.approx(1.0, abs=1.0e-7)
        assert strikes_t @ density_t == pytest.approx(1.0, abs=1.0e-7)

    query = np.linspace(0.5, 1.5, 101, dtype=np.float64)
    calls = np.asarray(result.price(expiries[:, None], query[None, :], mode="surface"))
    assert np.all(calls[1] >= calls[0] - 1.0e-7)

    restored = ArbFreePricer.from_dict(result.to_dict())
    restored_calls = np.asarray(restored.price(expiries[:, None], query[None, :], mode="surface"))
    np.testing.assert_allclose(restored_calls, calls, rtol=0.0, atol=0.0)
    assert restored.to_dict().keys() == result.to_dict().keys()


def test_lp_solver_reports_infeasible_problem() -> None:
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    success, status, fell_back = SanosLPSolver(timeout_seconds=10.0)(
        objective=variable,
        constraints=[variable >= 1.0, variable <= 0.0],
        variables=[],
        diagnostic_sink=lambda event: events.append(dict(event)),
        verbose=cast(Any, Context).quiet,
    )
    assert not success
    assert status in {"infeasible", "infeasible_inaccurate"}
    assert not fell_back
    assert events == [
        {
            "solver": "HIGHS",
            "algorithm": "simplex",
            "status": status,
            "is_fallback": False,
            "for_expiry": False,
            "objective": None,
        }
    ]


def test_lp_solver_diagnostics_are_numerically_neutral() -> None:
    def solve(events: list[dict[str, object]] | None) -> tuple[str, float]:
        variable = cp.Variable()
        success, status, _ = SanosLPSolver(timeout_seconds=10.0)(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=(None if events is None else lambda event: events.append(dict(event))),
            verbose=cast(Any, Context).quiet,
        )
        assert success
        return status, float(variable.value)

    without_sink = solve(None)
    events: list[dict[str, object]] = []
    with_sink = solve(events)
    assert with_sink == without_sink
    assert len(events) == 1
    assert events[0]["status"] == with_sink[0]


def test_lp_solver_forced_primary_failure_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve = cp.Problem.solve

    def force_primary_failure(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("solver") == "HIGHS":
            raise SolverError("forced primary failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", force_primary_failure)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.warns(UserWarning, match="fallback solver"):
        success, status, fell_back = SanosLPSolver(timeout_seconds=10.0)(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert success
    assert status in {"optimal", "optimal_inaccurate"}
    assert fell_back
    assert [(event["solver"], event["status"], event["is_fallback"]) for event in events] == [
        ("HIGHS", "solver_error", False),
        ("CLARABEL", status, True),
    ]


def test_lp_solver_forced_primary_user_limit_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve = cp.Problem.solve

    def force_primary_limit(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("solver") == "HIGHS":
            problem._status = "user_limit"
            return 0.0
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", force_primary_limit)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.warns(UserWarning, match="reached its limit"):
        success, status, fell_back = SanosLPSolver(timeout_seconds=10.0)(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert success
    assert status in {"optimal", "optimal_inaccurate"}
    assert fell_back
    assert [(event["solver"], event["status"], event["is_fallback"]) for event in events] == [
        ("HIGHS", "user_limit", False),
        ("CLARABEL", status, True),
    ]


def test_lp_solver_smooth_policy_uses_ipm_before_clarabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve = cp.Problem.solve

    def reject_simplex(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        highs = kwargs.get("highs_options", {})
        if kwargs.get("solver") == "HIGHS" and highs.get("solver") == "simplex":
            problem._solver_cache["HIGHS"] = (
                None,
                None,
                {
                    "model_status": "kUnknown",
                    "run_time": 1.5,
                    "info": SimpleNamespace(max_primal_infeasibility=0.25),
                },
            )
            raise ValueError("Cannot unpack invalid solution")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", reject_simplex)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.warns(UserWarning, match="ipm fallback solver") as caught:
        success, status, fell_back = SanosLPSolver(
            timeout_seconds=10.0,
            fallback_algorithms=("ipm", "clarabel"),
        )(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert success and fell_back
    assert "kUnknown" in str(caught[0].message)
    assert "max_primal_infeasibility" in str(caught[0].message)
    assert status in {"optimal", "optimal_inaccurate"}
    assert [(event["solver"], event["algorithm"], event["status"]) for event in events] == [
        ("HIGHS", "simplex", "invalid_solution"),
        ("HIGHS", "ipm", status),
    ]


@pytest.mark.parametrize("ipm_failure", ["solver_error", "invalid_solution", "user_limit"])
def test_lp_solver_ipm_retry_trigger_reaches_clarabel(
    monkeypatch: pytest.MonkeyPatch,
    ipm_failure: str,
) -> None:
    original_solve = cp.Problem.solve

    def force_retries(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        highs = kwargs.get("highs_options", {})
        if kwargs.get("solver") == "HIGHS" and highs.get("solver") == "simplex":
            raise SolverError("forced simplex failure")
        if kwargs.get("solver") == "HIGHS" and highs.get("solver") == "ipm":
            if ipm_failure == "solver_error":
                raise SolverError("forced IPM failure")
            if ipm_failure == "invalid_solution":
                raise ValueError("Cannot unpack invalid solution")
            problem._status = "user_limit"
            return 0.0
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", force_retries)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.warns(UserWarning) as caught:
        success, status, fell_back = SanosLPSolver(
            timeout_seconds=10.0,
            fallback_algorithms=("ipm", "clarabel"),
        )(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert len(caught) == 2
    assert success and fell_back
    assert status in {"optimal", "optimal_inaccurate"}
    assert [(event["solver"], event["algorithm"], event["status"]) for event in events] == [
        ("HIGHS", "simplex", "solver_error"),
        ("HIGHS", "ipm", ipm_failure),
        ("CLARABEL", "clarabel", status),
    ]


def test_lp_solver_terminal_user_limit_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def force_limit(problem: cp.Problem, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        problem._status = "user_limit"
        return 0.0

    monkeypatch.setattr(cp.Problem, "solve", force_limit)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.warns(UserWarning, match="reached its limit"):
        success, status, fell_back = SanosLPSolver(timeout_seconds=10.0)(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert not success
    assert status == "user_limit"
    assert fell_back
    assert [(event["solver"], event["status"], event["is_fallback"]) for event in events] == [
        ("HIGHS", "user_limit", False),
        ("CLARABEL", "user_limit", True),
    ]


def test_lp_solver_direct_clarabel_user_limit_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def force_limit(problem: cp.Problem, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        problem._status = "user_limit"
        return 0.0

    monkeypatch.setattr(cp.Problem, "solve", force_limit)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    success, status, fell_back = SanosLPSolver(
        timeout_seconds=10.0,
        algorithm="clarabel",
    )(
        objective=variable,
        constraints=[variable >= 1.0],
        variables=[(variable, np.asarray(0.0))],
        diagnostic_sink=lambda event: events.append(dict(event)),
        verbose=cast(Any, Context).quiet,
    )
    assert (success, status, fell_back) == (False, "user_limit", False)
    assert [(event["solver"], event["is_fallback"]) for event in events] == [("CLARABEL", False)]


def test_lp_solver_direct_clarabel_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def force_error(problem: cp.Problem, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        del problem, args, kwargs
        calls += 1
        raise SolverError("forced direct CLARABEL failure")

    monkeypatch.setattr(cp.Problem, "solve", force_error)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with pytest.raises(SolverError, match="forced direct CLARABEL failure"):
        SanosLPSolver(timeout_seconds=10.0, algorithm="clarabel")(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert calls == 1
    assert [(event["solver"], event["is_fallback"]) for event in events] == [("CLARABEL", False)]


def test_lp_solver_terminal_unexpected_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def force_terminal_error(
        problem: cp.Problem,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del problem, args
        if kwargs.get("solver") == "HIGHS":
            raise SolverError("forced primary failure")
        raise ValueError("forced terminal failure")

    monkeypatch.setattr(cp.Problem, "solve", force_terminal_error)
    variable = cp.Variable()
    events: list[dict[str, object]] = []
    with (
        pytest.warns(UserWarning, match="fallback solver"),
        pytest.raises(ValueError, match="Unexpected error: forced terminal failure"),
    ):
        SanosLPSolver(timeout_seconds=10.0)(
            objective=variable,
            constraints=[variable >= 1.0],
            variables=[(variable, np.asarray(0.0))],
            diagnostic_sink=lambda event: events.append(dict(event)),
            verbose=cast(Any, Context).quiet,
        )
    assert [(event["solver"], event["status"], event["is_fallback"]) for event in events] == [
        ("HIGHS", "solver_error", False),
        ("CLARABEL", "unexpected_error", True),
    ]


def test_extended_strikes_and_blocked_vector_diagnostic_contract() -> None:
    arbfree = MkArbFree(no_arb_mode="expiry")
    strikes = np.array([0.60, 0.75, 0.90, 1.00, 1.10, 1.20, 1.30, 1.45])
    mids = np.array([0.50, 0.32, 0.20, 0.13, 0.08, 0.04, 0.03995, 0.03990])
    xstrikes, mask, notice = arbfree.make_extended_strikes(
        strikes,
        mids - 0.01,
        mids + 0.01,
    )
    np.testing.assert_array_equal(
        mask,
        np.array([False, True, True, True, True, False, False, False]),
    )
    np.testing.assert_allclose(xstrikes[[0, -1]], np.array([0.2, 1.5]), atol=1.0e-12)
    assert notice is not None
    assert "removed 4 out of 8 strikes" in cast(Callable[[], str], notice)()

    expiries, grids, dummy_densities, input_dlvsqs = _reference_inputs()
    template = SanosPricer(None, expiries, expiries * 0.04, grids, dummy_densities)
    pricer = SanosPricer(
        None,
        expiries,
        expiries * 0.04,
        grids,
        template.densities_from_dlvsqs(input_dlvsqs),
    )
    with pytest.raises(NotImplementedError, match="CD-013"):
        pricer.compute_market_fit(
            expiries=np.array([expiries[0], expiries[0]]),
            strikes=np.array([0.9, 1.1]),
            bids=np.array([0.1, 0.01]),
            asks=np.array([0.2, 0.02]),
        )


def test_numerical_components_have_resolved_versions() -> None:
    def direct_dependency_names(component: type[object]) -> set[str]:
        resolved = cast(Any, getattr(component, "version")).dependencies
        assert isinstance(resolved, tuple)
        return set(cast(dict[str, Any], resolved[1]))

    assert direct_dependency_names(SanosPricer) == {
        "_compute_market_fit",
        "one_minute_year_fraction",
        "solve_sanos_tridiagonal",
    }
    assert direct_dependency_names(SanosFitter) == {
        "SanosLPSolver",
        "SanosPricer",
        "_StrictSanosLPSolver",
        "density_floor_enabled",
        "one_minute_year_fraction",
        "warn_and_fix_density",
    }
    assert direct_dependency_names(ArbFreePricer) == {
        "_compute_market_fit",
        "one_minute_year_fraction",
    }
    assert direct_dependency_names(MkArbFree) == {
        "ArbFreePricer",
        "SanosLPSolver",
        "density_floor_enabled",
        "one_minute_year_fraction",
        "warn_and_fix_density",
    }
    assert "1.0-blocked-cd013" in str(getattr(SanosPricer, "version"))
    assert str(getattr(SanosLPSolver, "version")).startswith("1.2")


def test_defective_reference_jax_optimizer_is_explicitly_blocked() -> None:
    from sanos.model.jax_fit import fit_dlvsqs

    with pytest.raises(NotImplementedError, match="CD-015"):
        fit_dlvsqs()
