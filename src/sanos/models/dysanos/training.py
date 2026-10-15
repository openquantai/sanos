"""Deterministic native initialization and joint Direct-200 training."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.version import version

from sanos.models.dysanos.config import (
    DailyFrontGeometryConfig,
    DirectModelConfig,
    DirectTrainingConfig,
    ObjectiveConfig,
)
from sanos.models.dysanos.data import DailyFrontDay, exact_weekly_pairs
from sanos.models.dysanos.dynamics import (
    AffineInitialization,
    DirectDynamicsModel,
    NativeDailyModel,
    affine_svd_initialization,
    dynamics_negative_log_likelihood,
    factor_gauge_penalty,
    initialize_stable_dynamics,
)
from sanos.models.dysanos.objective import (
    ObjectiveEvaluation,
    TensorBatch,
    evaluate_quote_objective,
    make_batch,
    quote_objective,
    restore_quote_prices,
    tensorize_day,
)
from sanos.models.dysanos.surface import (
    NativeParameters,
    SanosSurfaceDecoder,
    bound_native_state,
    dlv_cap_penalty,
    raw_from_parameters,
)

CHECKPOINT_SCHEMA = "sanos-dysanos-direct200-checkpoint-v3"


@version("1.1")
def checkpoint_schema() -> str:
    return "sanos-dysanos-direct200-checkpoint-v3"


@version("1.0")
@dataclass(frozen=True, slots=True)
class TrainingEvaluation:
    quote: ObjectiveEvaluation
    dynamics_negative_log_likelihood: float
    gauge_penalty: float


@version("1.0")
@dataclass(frozen=True, slots=True)
class NativeFitResult:
    raw_states: torch.Tensor
    evaluation: ObjectiveEvaluation
    history: tuple[float, ...]


@version("1.0")
@dataclass(frozen=True, slots=True)
class DirectFitResult:
    model: DirectDynamicsModel
    initialization: AffineInitialization
    evaluation: TrainingEvaluation
    history: tuple[Mapping[str, float], ...]
    completed_epochs: int


@version("1.0")
def native_fit_epochs() -> int:
    """Version-owned native optimization budget, separate from PCA warmup."""

    return 300


@version("1.0")
def affine_fine_tune_epochs() -> int:
    """Version-owned PCA market fine-tune budget, separate from LR warmup."""

    return 300


@version("1.0")
def convergence_audit(
    history: tuple[Mapping[str, float], ...],
    *,
    window: int,
    relative_tolerance: float,
) -> Mapping[str, float | int | bool | str]:
    """Pinned trailing-window/final-gap completion audit."""

    eligible = tuple(
        item for item in history if item.get("dynamics_ramp", 1.0) >= 1.0 - 1.0e-12
    )
    values = np.asarray([item["total"] for item in eligible], dtype=np.float64)
    if len(values) <= window or not np.all(np.isfinite(values)):
        return {
            "converged": False,
            "reason": "insufficient_or_nonfinite_history",
            "num_epochs": len(history),
            "num_comparable_epochs": len(values),
        }
    start = float(values[-window - 1])
    final = float(values[-1])
    best_index = int(np.argmin(values))
    best = float(values[best_index])
    denominator = max(abs(start), abs(best), 1.0e-8)
    window_change = (start - final) / denominator
    final_gap = (final - best) / max(abs(best), 1.0e-8)
    converged = (
        abs(window_change) <= relative_tolerance
        and final_gap <= 2.0 * relative_tolerance
    )
    return {
        "converged": converged,
        "reason": "converged" if converged else "trailing_window_not_converged",
        "num_epochs": len(history),
        "num_comparable_epochs": len(values),
        "window_relative_change": window_change,
        "final_relative_gap_to_best": final_gap,
        "best_epoch": int(eligible[best_index]["epoch"]),
    }


@version("1.0")
def _clip_sparse_gradient_(parameter: torch.Tensor, maximum_norm: float) -> float:
    """Clip populated sparse-gradient values without coupling dense parameters."""

    if not np.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError("Sparse gradient norm limit must be finite and positive")
    gradient = parameter.grad
    if gradient is None:
        return 0.0
    if not gradient.is_sparse:
        raise ValueError("Expected a sparse embedding gradient")
    gradient = gradient.coalesce()
    norm = torch.linalg.vector_norm(gradient.values())
    scale = torch.clamp(maximum_norm / (norm + 1.0e-12), max=1.0)
    gradient.values().mul_(scale)
    parameter.grad = gradient
    return float(norm.detach().cpu())


@version("1.0")
def _date_weighted_epoch_loss(weighted_total: float, date_count: int) -> float:
    """Return an epoch mean in which every selected date has equal weight."""

    if date_count <= 0 or not np.isfinite(weighted_total):
        raise ValueError("Date-weighted epoch loss inputs are invalid")
    return weighted_total / date_count


@version("1.0", dependencies=[ObjectiveEvaluation])
def _require_no_quote_objective_regression(
    initial_objective: float,
    final: ObjectiveEvaluation,
    *,
    phase: str,
) -> None:
    """Reject a fitted candidate whose eager market objective materially regressed."""

    if not np.isfinite(initial_objective) or not np.isfinite(final.objective):
        raise RuntimeError(f"{phase} DySANOS objective is non-finite")
    allowed = 1.0e-10 * max(1.0, abs(initial_objective))
    if final.objective > initial_objective + allowed:
        raise RuntimeError(
            f"{phase} DySANOS objective regressed from "
            f"{initial_objective:.10g} to {final.objective:.10g}"
        )


@version("1.0", dependencies=[NativeParameters, raw_from_parameters])
def native_raw_from_day(
    day: DailyFrontDay,
    geometry: DailyFrontGeometryConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Initialize native state from the public fitted ATM total-vol curve."""

    total = torch.as_tensor(day.model_atm_total_vol, dtype=torch.float64)
    expiries = torch.as_tensor(geometry.model_business_dtes, dtype=torch.float64) / 252.0
    if total.shape != expiries.shape or not bool(torch.all(torch.isfinite(total))):
        raise ValueError("Day ATM total-vol curve does not match canonical geometry")
    increments = torch.diff(total, prepend=torch.zeros_like(total[:1]))
    time_steps = torch.diff(expiries, prepend=torch.zeros_like(expiries[:1]))
    forward = increments / torch.sqrt(time_steps)
    annual = total / torch.sqrt(expiries)
    forward = torch.clamp(
        forward, geometry.min_forward_vol, geometry.max_forward_vol
    )
    dlv = torch.clamp(
        annual.square(), geometry.min_dlvsq, geometry.max_dlvsq
    ).unsqueeze(-1).expand(-1, geometry.num_inner_strikes)
    eta = torch.full_like(forward, 0.20)
    return raw_from_parameters(
        NativeParameters(dlv, forward, eta), geometry, probability_floor=1.0e-6
    ).to(dtype=dtype, device=device)


@version("1.0")
def _model_quote_prices(
    model: DirectDynamicsModel,
    batch: TensorBatch,
) -> torch.Tensor:
    calls = model.call_prices(
        batch.day_indices,
        batch.price_expiries_y,
        batch.price_strikes,
    )
    return restore_quote_prices(calls, batch)


@version("1.0", dependencies=[bound_native_state, restore_quote_prices])
def _affine_quote_prices(
    model: DirectDynamicsModel,
    states: torch.nn.Embedding,
    batch: TensorBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = model.decoder(states(batch.day_indices))
    parameters = bound_native_state(raw, model.geometry)
    calls = model.surface_decoder.price_calls(
        model.surface_decoder(parameters), batch.price_expiries_y, batch.price_strikes
    )
    return restore_quote_prices(calls, batch), raw


@version("1.0", dependencies=[SanosSurfaceDecoder, bound_native_state, restore_quote_prices])
def _native_quote_prices(
    model: NativeDailyModel,
    decoder: SanosSurfaceDecoder,
    geometry: DailyFrontGeometryConfig,
    batch: TensorBatch,
) -> torch.Tensor:
    parameters = bound_native_state(model(batch.day_indices), geometry)
    calls = decoder.price_calls(
        decoder(parameters), batch.price_expiries_y, batch.price_strikes
    )
    return restore_quote_prices(calls, batch)


@version("1.0")
def _safe_checkpoint_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Unsafe DySANOS checkpoint path: {path}")
    parent = path.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError(f"Unsafe DySANOS checkpoint directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError(f"Unsafe DySANOS checkpoint directory: {parent}")


@version("1.0")
def _stage_checkpoint_path(path: Path, phase: str) -> Path:
    if phase not in {"native", "affine_quote"}:
        raise ValueError("Unsupported DySANOS pretraining checkpoint phase")
    return path.with_name(f"{path.name}.{phase}.pt")


@version("1.0", dependencies=[_safe_checkpoint_path])
def _atomic_checkpoint_payload(path: Path, payload: Mapping[str, object]) -> None:
    _safe_checkpoint_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@version("1.0", dependencies=[_atomic_checkpoint_payload, checkpoint_schema])
def _save_pretrain_checkpoint(
    path: Path,
    *,
    phase: str,
    epoch: int,
    configuration_id: str,
    source_id: str,
    state: Mapping[str, object],
    history: tuple[float, ...],
    permutation_rng: np.random.Generator,
) -> None:
    _atomic_checkpoint_payload(
        path,
        {
            "schema": checkpoint_schema(),
            "phase": phase,
            "epoch": epoch,
            "configuration_id": configuration_id,
            "source_id": source_id,
            "state": dict(state),
            "history": list(history),
            "torch_rng_state": torch.get_rng_state(),
            "permutation_rng_state": permutation_rng.bit_generator.state,
        },
    )


@version("1.0", dependencies=[_safe_checkpoint_path, checkpoint_schema])
def _load_pretrain_checkpoint(
    path: Path,
    *,
    phase: str,
    configuration_id: str,
    source_id: str,
    permutation_rng: np.random.Generator,
) -> tuple[int, tuple[float, ...], Mapping[str, object]]:
    _safe_checkpoint_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "phase",
        "epoch",
        "configuration_id",
        "source_id",
        "state",
        "history",
        "torch_rng_state",
        "permutation_rng_state",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("DySANOS pretraining checkpoint fields changed")
    if (
        payload["schema"] != checkpoint_schema()
        or payload["phase"] != phase
        or payload["configuration_id"] != configuration_id
        or payload["source_id"] != source_id
    ):
        raise ValueError("DySANOS pretraining checkpoint identity changed")
    epoch, history, state = payload["epoch"], payload["history"], payload["state"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or not isinstance(history, list)
        or len(history) != epoch
        or not isinstance(state, Mapping)
    ):
        raise ValueError("DySANOS pretraining checkpoint state is malformed")
    rng_state = payload["torch_rng_state"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.dtype is not torch.uint8:
        raise ValueError("DySANOS pretraining checkpoint RNG state is invalid")
    torch.set_rng_state(rng_state.cpu())
    permutation_rng.bit_generator.state = payload["permutation_rng_state"]
    return epoch, tuple(float(value) for value in history), state


@version(
    "1.0",
    dependencies=[
        NativeDailyModel,
        NativeFitResult,
        DailyFrontGeometryConfig,
        ObjectiveConfig,
        native_raw_from_day,
        SanosSurfaceDecoder,
        _native_quote_prices,
        dlv_cap_penalty,
        evaluate_quote_objective,
        make_batch,
        quote_objective,
        tensorize_day,
        convergence_audit,
        _clip_sparse_gradient_,
        _date_weighted_epoch_loss,
        _require_no_quote_objective_regression,
        _load_pretrain_checkpoint,
        _save_pretrain_checkpoint,
        _stage_checkpoint_path,
    ],
)
def fit_native_daily_states(
    days: tuple[DailyFrontDay, ...],
    geometry: DailyFrontGeometryConfig,
    objective: ObjectiveConfig,
    *,
    num_epochs: int,
    learning_rate: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    day_batch_size: int | None = None,
    seed: int = 9281,
    report_every: int = 5,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    require_convergence: bool = True,
    convergence_window: int = 30,
    convergence_relative_tolerance: float = 0.002,
    gradient_clip: float = 10.0,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
    configuration_id: str | None = None,
    source_id: str | None = None,
    resume: bool = True,
) -> NativeFitResult:
    """Fit independent daily native states without a legacy checkpoint."""

    if not days or num_epochs <= 0 or learning_rate <= 0.0 or report_every <= 0:
        raise ValueError("Native initialization requires days, epochs, and learning rate")
    batch_size = len(days) if day_batch_size is None else day_batch_size
    if (
        batch_size <= 0
        or checkpoint_every <= 0
        or convergence_window <= 0
        or not 0.0 < convergence_relative_tolerance < 1.0
        or not np.isfinite(gradient_clip)
        or gradient_clip <= 0.0
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("Native batching, convergence, or seed policy is invalid")
    if checkpoint_path is not None and (not configuration_id or not source_id):
        raise ValueError("Native checkpointing requires source/configuration identities")
    tensor_days = tuple(
        tensorize_day(day, index, dtype=dtype, device=device)
        for index, day in enumerate(days)
    )
    batch = make_batch(tensor_days)
    initial = torch.stack(
        [native_raw_from_day(day, geometry, dtype=dtype, device=device) for day in days]
    )
    model = NativeDailyModel(
        len(days), geometry, initial, dtype=dtype, device=device
    )
    decoder = SanosSurfaceDecoder(geometry, dtype=dtype, device=device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[float] = []
    start_epoch = 0
    stage_path = (
        None if checkpoint_path is None else _stage_checkpoint_path(checkpoint_path, "native")
    )
    if stage_path is not None and stage_path.exists():
        if not resume:
            raise FileExistsError("Native checkpoint exists but resume is disabled")
        start_epoch, restored, state = _load_pretrain_checkpoint(
            stage_path,
            phase="native",
            configuration_id=configuration_id or "",
            source_id=source_id or "",
            permutation_rng=rng,
        )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history.extend(restored)
        initial_objective = float(state["initial_objective"])
    else:
        with torch.no_grad():
            initial_prices = _native_quote_prices(model, decoder, geometry, batch)
            initial_evaluation = evaluate_quote_objective(
                initial_prices, batch, fit_weight=objective.fit_weight
            )
        initial_objective = initial_evaluation.objective
    for epoch in range(start_epoch, num_epochs):
        loss_total = 0.0
        date_count = 0
        permutation = rng.permutation(len(tensor_days))
        for start in range(0, len(permutation), batch_size):
            selected_days = tuple(
                tensor_days[int(index)] for index in permutation[start : start + batch_size]
            )
            selected = make_batch(selected_days)
            optimizer.zero_grad(set_to_none=True)
            prices = _native_quote_prices(model, decoder, geometry, selected)
            loss = quote_objective(prices, selected, fit_weight=objective.fit_weight)
            loss = loss + dlv_cap_penalty(model(selected.day_indices), geometry)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Native DySANOS fit produced a non-finite loss")
            loss.backward()
            _clip_sparse_gradient_(model.raw_states, gradient_clip)
            optimizer.step()
            count = len(selected_days)
            loss_total += float(loss.detach().cpu()) * count
            date_count += count
        history.append(_date_weighted_epoch_loss(loss_total, date_count))
        if stage_path is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch + 1 == num_epochs
        ):
            _save_pretrain_checkpoint(
                stage_path,
                phase="native",
                epoch=epoch + 1,
                configuration_id=configuration_id or "",
                source_id=source_id or "",
                state={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "initial_objective": initial_objective,
                },
                history=tuple(history),
                permutation_rng=rng,
            )
        if progress is not None and (
            (epoch + 1) % report_every == 0 or epoch == 0 or epoch + 1 == num_epochs
        ):
            progress(
                {"phase": "native", "epoch": epoch + 1, "total": num_epochs, "loss": history[-1]}
            )
    audit = convergence_audit(
        tuple(
            {"epoch": float(index + 1), "total": value, "dynamics_ramp": 1.0}
            for index, value in enumerate(history)
        ),
        window=convergence_window,
        relative_tolerance=convergence_relative_tolerance,
    )
    if require_convergence and not audit["converged"]:
        raise RuntimeError(f"Native DySANOS convergence failed: {audit['reason']}")
    with torch.no_grad():
        prices = _native_quote_prices(model, decoder, geometry, batch)
        evaluation = evaluate_quote_objective(
            prices, batch, fit_weight=objective.fit_weight
        )
    _require_no_quote_objective_regression(
        initial_objective,
        evaluation,
        phase="native",
    )
    return NativeFitResult(
        model.raw_states.detach().cpu().to(torch.float64),
        evaluation,
        tuple(history),
    )


@version("1.0")
def _optimizer(
    model: DirectDynamicsModel,
    config: DirectTrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        (
            {"params": [model.daily_states], "lr": config.state_learning_rate},
            {"params": model.decoder.parameters(), "lr": config.decoder_learning_rate},
            {
                "params": (
                    model.damping_factor,
                    model.rotation_raw,
                    model.equilibrium_mean,
                    model.innovation_cholesky_raw,
                ),
                "lr": config.dynamics_learning_rate,
            },
        )
    )


@version("1.0")
def _learning_rate_multiplier(epoch: int, config: DirectTrainingConfig) -> float:
    if epoch < config.warmup_epochs:
        return (epoch + 1) / max(config.warmup_epochs, 1)
    progress = (epoch - config.warmup_epochs) / max(
        config.num_epochs - config.warmup_epochs - 1, 1
    )
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return config.minimum_learning_rate_fraction + (
        1.0 - config.minimum_learning_rate_fraction
    ) * float(cosine)


@version("1.0", dependencies=[TensorBatch])
def _select_batch(batch: TensorBatch, rows: torch.Tensor) -> TensorBatch:
    return TensorBatch(
        batch.day_indices[rows],
        batch.price_expiries_y[rows],
        batch.price_strikes[rows],
        batch.quote_price_index[rows],
        batch.quote_strikes[rows],
        batch.quote_is_call[rows],
        batch.quote_fit[rows],
        batch.quote_half_spread[rows],
        batch.quote_mask[rows],
    )


@version("1.0", dependencies=[_safe_checkpoint_path, checkpoint_schema])
def save_checkpoint(
    path: Path,
    *,
    model: DirectDynamicsModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    configuration_id: str,
    source_id: str,
    phase: str,
    history: tuple[Mapping[str, float], ...],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    permutation_rng: np.random.Generator,
    baseline_percent_books_outside: float,
    baseline_quote_objective: float,
) -> None:
    """Atomically replace one deterministic, restartable training checkpoint."""

    _safe_checkpoint_path(path)
    if epoch < 0 or not configuration_id or not source_id or phase != "joint":
        raise ValueError("Checkpoint epoch/configuration identity is invalid")
    payload = {
        "schema": checkpoint_schema(),
        "configuration_id": configuration_id,
        "source_id": source_id,
        "phase": phase,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": [dict(item) for item in history],
        "torch_rng_state": torch.get_rng_state(),
        "permutation_rng_state": permutation_rng.bit_generator.state,
        "baseline_percent_books_outside": baseline_percent_books_outside,
        "baseline_quote_objective": baseline_quote_objective,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@version("1.0", dependencies=[_safe_checkpoint_path, checkpoint_schema])
def load_checkpoint(
    path: Path,
    *,
    model: DirectDynamicsModel,
    optimizer: torch.optim.Optimizer,
    configuration_id: str,
    source_id: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    permutation_rng: np.random.Generator,
) -> tuple[int, tuple[Mapping[str, float], ...], float, float]:
    """Restore an exact checkpoint while rejecting identity/schema drift."""

    _safe_checkpoint_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=model.daily_states.device, weights_only=True)
    required = {
        "schema",
        "configuration_id",
        "source_id",
        "phase",
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "torch_rng_state",
        "permutation_rng_state",
        "baseline_percent_books_outside",
        "baseline_quote_objective",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("DySANOS checkpoint fields changed")
    if payload["schema"] != checkpoint_schema():
        raise ValueError("Unsupported DySANOS checkpoint schema")
    if payload["configuration_id"] != configuration_id:
        raise ValueError("DySANOS checkpoint configuration identity changed")
    if payload["source_id"] != source_id or payload["phase"] != "joint":
        raise ValueError("DySANOS checkpoint source identity or phase changed")
    epoch = payload["epoch"]
    history = payload["history"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("DySANOS checkpoint epoch is invalid")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("DySANOS checkpoint history is inconsistent")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    rng_state = payload["torch_rng_state"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.dtype is not torch.uint8:
        raise ValueError("DySANOS checkpoint RNG state is invalid")
    torch.set_rng_state(rng_state.cpu())
    permutation_rng.bit_generator.state = payload["permutation_rng_state"]
    baseline = payload["baseline_percent_books_outside"]
    baseline_objective = payload["baseline_quote_objective"]
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, int | float)
        or isinstance(baseline_objective, bool)
        or not isinstance(baseline_objective, int | float)
        or not np.isfinite(baseline)
        or not np.isfinite(baseline_objective)
    ):
        raise ValueError("DySANOS checkpoint evaluation baseline is invalid")
    return (
        epoch,
        tuple(dict(item) for item in history),
        float(baseline),
        float(baseline_objective),
    )


@version(
    "1.0",
    dependencies=[
        DirectDynamicsModel,
        ObjectiveConfig,
        TrainingEvaluation,
        _model_quote_prices,
        dynamics_negative_log_likelihood,
        evaluate_quote_objective,
        factor_gauge_penalty,
    ],
)
def evaluate_direct_model(
    model: DirectDynamicsModel,
    batch: TensorBatch,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
    objective: ObjectiveConfig,
) -> TrainingEvaluation:
    """Evaluate public-book fit and weekly dynamics without mutation."""

    model.eval()
    with torch.no_grad():
        prices = _model_quote_prices(model, batch)
        quote = evaluate_quote_objective(
            prices, batch, fit_weight=objective.fit_weight
        )
        dynamics = dynamics_negative_log_likelihood(
            model, origin_indices, endpoint_indices
        )
        gauge = factor_gauge_penalty(model.daily_states)
    return TrainingEvaluation(quote, float(dynamics.cpu()), float(gauge.cpu()))


@version(
    "1.0",
    dependencies=[
        DirectDynamicsModel,
        DirectTrainingConfig,
        ObjectiveConfig,
        _affine_quote_prices,
        _model_quote_prices,
        _select_batch,
        dlv_cap_penalty,
        evaluate_quote_objective,
        quote_objective,
        convergence_audit,
        _clip_sparse_gradient_,
        _date_weighted_epoch_loss,
        _require_no_quote_objective_regression,
        affine_fine_tune_epochs,
        _load_pretrain_checkpoint,
        _save_pretrain_checkpoint,
        _stage_checkpoint_path,
    ],
)
def fine_tune_affine_quotes(
    model: DirectDynamicsModel,
    batch: TensorBatch,
    objective: ObjectiveConfig,
    training: DirectTrainingConfig,
    *,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
    configuration_id: str | None = None,
    source_id: str | None = None,
    resume: bool = True,
    num_epochs: int | None = None,
    require_convergence: bool = True,
) -> ObjectiveEvaluation:
    """Fine-tune PCA states/decoder against books before dynamics coupling."""

    states = torch.nn.Embedding.from_pretrained(
        model.daily_states.detach().clone(), freeze=False, sparse=True
    )
    state_optimizer = torch.optim.SparseAdam(
        states.parameters(), lr=training.state_learning_rate
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=training.decoder_learning_rate
    )
    rng = np.random.default_rng(training.seed)
    losses: list[float] = []
    fine_tune_epochs = affine_fine_tune_epochs() if num_epochs is None else num_epochs
    if fine_tune_epochs <= 0 or checkpoint_every <= 0:
        raise ValueError("Affine fine-tune epochs/checkpoint interval must be positive")
    if checkpoint_path is not None and (not configuration_id or not source_id):
        raise ValueError("Affine checkpointing requires source/configuration identities")
    start_epoch = 0
    stage_path = (
        None
        if checkpoint_path is None
        else _stage_checkpoint_path(checkpoint_path, "affine_quote")
    )
    if stage_path is not None and stage_path.exists():
        if not resume:
            raise FileExistsError("Affine checkpoint exists but resume is disabled")
        start_epoch, restored, state = _load_pretrain_checkpoint(
            stage_path,
            phase="affine_quote",
            configuration_id=configuration_id or "",
            source_id=source_id or "",
            permutation_rng=rng,
        )
        states.load_state_dict(state["states"], strict=True)
        model.decoder.load_state_dict(state["decoder"], strict=True)
        state_optimizer.load_state_dict(state["state_optimizer"])
        decoder_optimizer.load_state_dict(state["decoder_optimizer"])
        losses.extend(restored)
        initial_objective = float(state["initial_objective"])
    else:
        with torch.no_grad():
            initial_prices = _model_quote_prices(model, batch)
            initial_evaluation = evaluate_quote_objective(
                initial_prices, batch, fit_weight=objective.fit_weight
            )
        initial_objective = initial_evaluation.objective
    for epoch in range(start_epoch, fine_tune_epochs):
        permutation = rng.permutation(len(batch.day_indices))
        loss_total = 0.0
        date_count = 0
        for start in range(0, len(permutation), training.day_batch_size):
            rows = torch.as_tensor(
                permutation[start : start + training.day_batch_size],
                dtype=torch.long,
                device=batch.day_indices.device,
            )
            selected = _select_batch(batch, rows)
            state_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)
            prices, raw = _affine_quote_prices(model, states, selected)
            quote = quote_objective(prices, selected, fit_weight=objective.fit_weight)
            cap = dlv_cap_penalty(raw, model.geometry)
            loss = quote + training.cap_penalty_weight * cap
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Affine DySANOS quote fine-tune produced non-finite loss")
            loss.backward()
            _clip_sparse_gradient_(states.weight, training.gradient_clip)
            torch.nn.utils.clip_grad_norm_(
                model.decoder.parameters(), training.gradient_clip
            )
            state_optimizer.step()
            decoder_optimizer.step()
            count = len(rows)
            loss_total += float(loss.detach().cpu()) * count
            date_count += count
        losses.append(_date_weighted_epoch_loss(loss_total, date_count))
        if stage_path is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch + 1 == fine_tune_epochs
        ):
            _save_pretrain_checkpoint(
                stage_path,
                phase="affine_quote",
                epoch=epoch + 1,
                configuration_id=configuration_id or "",
                source_id=source_id or "",
                state={
                    "states": states.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "state_optimizer": state_optimizer.state_dict(),
                    "decoder_optimizer": decoder_optimizer.state_dict(),
                    "initial_objective": initial_objective,
                },
                history=tuple(losses),
                permutation_rng=rng,
            )
        if progress is not None and (
            (epoch + 1) % training.report_every == 0
            or epoch == 0
            or epoch + 1 == fine_tune_epochs
        ):
            progress(
                {
                    "phase": "affine_quote",
                    "epoch": epoch + 1,
                    "total": fine_tune_epochs,
                    "loss": losses[-1],
                }
            )
    audit = convergence_audit(
        tuple(
            {"epoch": float(index + 1), "total": value, "dynamics_ramp": 1.0}
            for index, value in enumerate(losses)
        ),
        window=training.convergence_window,
        relative_tolerance=training.convergence_relative_tolerance,
    )
    if require_convergence and not audit["converged"]:
        raise RuntimeError(f"Affine DySANOS convergence failed: {audit['reason']}")
    with torch.no_grad():
        model.daily_states.copy_(states.weight)
        prices = _model_quote_prices(model, batch)
        evaluation = evaluate_quote_objective(
            prices, batch, fit_weight=objective.fit_weight
        )
    _require_no_quote_objective_regression(
        initial_objective,
        evaluation,
        phase="affine",
    )
    return evaluation


@version(
    "1.0",
    dependencies=[
        DirectDynamicsModel,
        DirectTrainingConfig,
        DirectFitResult,
        ObjectiveConfig,
        _model_quote_prices,
        _learning_rate_multiplier,
        _optimizer,
        _select_batch,
        _date_weighted_epoch_loss,
        dlv_cap_penalty,
        dynamics_negative_log_likelihood,
        evaluate_direct_model,
        factor_gauge_penalty,
        load_checkpoint,
        quote_objective,
        save_checkpoint,
        convergence_audit,
        _require_no_quote_objective_regression,
    ],
)
def fit_joint_direct(
    model: DirectDynamicsModel,
    batch: TensorBatch,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
    objective: ObjectiveConfig,
    training: DirectTrainingConfig,
    initialization: AffineInitialization,
    *,
    configuration_id: str,
    source_id: str,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    require_convergence: bool = True,
) -> DirectFitResult:
    """Fine-tune affine states/surface and stable weekly dynamics jointly."""

    if not configuration_id or not source_id:
        raise ValueError("DySANOS training requires configuration and source identities")
    optimizer = _optimizer(model, training)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, training)
    )
    permutation_rng = np.random.default_rng(training.seed)
    if training.compile_sanos_solver:
        model.surface_decoder.enable_solver_compile()
    start_epoch = 0
    history: tuple[Mapping[str, float], ...] = ()
    baseline_evaluation = evaluate_direct_model(
        model, batch, origin_indices, endpoint_indices, objective
    )
    baseline_outside = baseline_evaluation.quote.percent_books_outside
    baseline_objective = baseline_evaluation.quote.objective
    if checkpoint_path is not None and checkpoint_path.exists():
        if not resume:
            raise FileExistsError("DySANOS checkpoint exists but resume is disabled")
        start_epoch, history, baseline_outside, baseline_objective = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            configuration_id=configuration_id,
            source_id=source_id,
            scheduler=scheduler,
            permutation_rng=permutation_rng,
        )
    records = list(history)
    model.train()
    for epoch in range(start_epoch, training.num_epochs):
        quote_sum = 0.0
        cap_sum = 0.0
        quote_date_count = 0
        permutation = permutation_rng.permutation(len(batch.day_indices))
        for start in range(0, len(permutation), training.day_batch_size):
            rows = torch.as_tensor(
                permutation[start : start + training.day_batch_size],
                dtype=torch.long,
                device=batch.day_indices.device,
            )
            selected = _select_batch(batch, rows)
            optimizer.zero_grad(set_to_none=True)
            prices = _model_quote_prices(model, selected)
            quote_loss = quote_objective(prices, selected, fit_weight=objective.fit_weight)
            cap_loss = dlv_cap_penalty(
                model.raw_states(selected.day_indices), model.geometry
            )
            batch_total = quote_loss + training.cap_penalty_weight * cap_loss
            if not bool(torch.isfinite(batch_total)):
                raise RuntimeError("Joint DySANOS quote fit produced a non-finite loss")
            batch_total.backward()
            torch.nn.utils.clip_grad_norm_(
                (model.daily_states, *model.decoder.parameters()), training.gradient_clip
            )
            optimizer.step()
            count = len(rows)
            quote_sum += float(quote_loss.detach().cpu()) * count
            cap_sum += float(cap_loss.detach().cpu()) * count
            quote_date_count += count
        ramp = min(1.0, (epoch + 1) / max(1, training.dynamics_ramp_epochs))
        for _ in range(training.dynamics_steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            dynamics_loss = dynamics_negative_log_likelihood(
                model, origin_indices, endpoint_indices
            )
            gauge_loss = factor_gauge_penalty(model.daily_states)
            dynamics_total = (
                ramp * training.dynamics_weight * dynamics_loss
                + training.gauge_weight * gauge_loss
            )
            if not bool(torch.isfinite(dynamics_total)):
                raise RuntimeError("Joint DySANOS dynamics produced a non-finite loss")
            dynamics_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            optimizer.step()
        scheduler.step()
        quote_value = _date_weighted_epoch_loss(quote_sum, quote_date_count)
        cap_value = _date_weighted_epoch_loss(cap_sum, quote_date_count)
        total_value = (
            quote_value
            + training.cap_penalty_weight * cap_value
            + ramp * training.dynamics_weight * float(dynamics_loss.detach().cpu())
            + training.gauge_weight * float(gauge_loss.detach().cpu())
        )
        records.append(
            {
                "epoch": float(epoch + 1),
                "total": total_value,
                "quote": quote_value,
                "dlv_cap": cap_value,
                "dynamics": float(dynamics_loss.detach().cpu()),
                "gauge": float(gauge_loss.detach().cpu()),
                "dynamics_ramp": ramp,
                "state_learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if progress is not None and (
            (epoch + 1) % training.report_every == 0 or epoch == start_epoch
        ):
            progress(records[-1])
        if checkpoint_path is not None and (
            (epoch + 1) % training.checkpoint_every == 0
            or epoch + 1 == training.num_epochs
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                configuration_id=configuration_id,
                source_id=source_id,
                phase="joint",
                history=tuple(records),
                scheduler=scheduler,
                permutation_rng=permutation_rng,
                baseline_percent_books_outside=baseline_outside,
                baseline_quote_objective=baseline_objective,
            )
    model.surface_decoder.disable_solver_compile()
    evaluation = evaluate_direct_model(
        model, batch, origin_indices, endpoint_indices, objective
    )
    allowed_outside = (
        baseline_outside
        + 100.0 * training.maximum_quote_containment_ratio_drop
        + 1.0e-12
    )
    if evaluation.quote.percent_books_outside > allowed_outside:
        raise RuntimeError("Joint DySANOS fit degraded quote containment beyond policy")
    _require_no_quote_objective_regression(
        baseline_objective,
        evaluation.quote,
        phase="joint",
    )
    audit = convergence_audit(
        tuple(records),
        window=training.convergence_window,
        relative_tolerance=training.convergence_relative_tolerance,
    )
    if require_convergence and not audit["converged"]:
        raise RuntimeError(f"Joint DySANOS convergence failed: {audit['reason']}")
    return DirectFitResult(
        model,
        initialization,
        evaluation,
        tuple(records),
        training.num_epochs,
    )


@version(
    "1.0",
    dependencies=[
        DirectDynamicsModel,
        DirectFitResult,
        DailyFrontGeometryConfig,
        DirectModelConfig,
        DirectTrainingConfig,
        ObjectiveConfig,
        affine_svd_initialization,
        exact_weekly_pairs,
        fit_joint_direct,
        fit_native_daily_states,
        fine_tune_affine_quotes,
        affine_fine_tune_epochs,
        initialize_stable_dynamics,
        make_batch,
        tensorize_day,
        native_fit_epochs,
    ],
)
def fit_direct200_pipeline(
    days: tuple[DailyFrontDay, ...],
    geometry: DailyFrontGeometryConfig,
    model_config: DirectModelConfig,
    objective: ObjectiveConfig,
    training: DirectTrainingConfig,
    *,
    configuration_id: str,
    source_id: str,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> DirectFitResult:
    """Run native fitting, float64 affine SVD, then joint Direct-200 training."""

    if (
        len(geometry.model_business_dtes) != 18
        or geometry.num_inner_strikes != 200
        or model_config.state_size != 200
        or model_config.decoded_state_size != 18 * 202
    ):
        raise ValueError("The production pipeline implements canonical Direct-200 only")
    if len(days) <= model_config.state_size:
        raise ValueError("Direct-200 requires more dates than latent factors")
    torch.manual_seed(training.seed)
    native = fit_native_daily_states(
        days,
        geometry,
        objective,
        num_epochs=native_fit_epochs(),
        learning_rate=training.state_learning_rate,
        dtype=dtype,
        device=device,
        day_batch_size=training.day_batch_size,
        seed=training.seed,
        report_every=training.report_every,
        progress=progress,
        convergence_window=training.convergence_window,
        convergence_relative_tolerance=training.convergence_relative_tolerance,
        gradient_clip=training.gradient_clip,
        checkpoint_path=checkpoint_path,
        checkpoint_every=training.checkpoint_every,
        configuration_id=configuration_id,
        source_id=source_id,
        resume=resume,
    )
    initialization = affine_svd_initialization(native.raw_states, model_config.state_size)
    direct = DirectDynamicsModel(
        len(days), geometry, model_config, dtype=dtype, device=device
    )
    direct.install_affine_initialization(initialization)
    tensor_days = tuple(
        tensorize_day(day, index, dtype=dtype, device=device)
        for index, day in enumerate(days)
    )
    batch = make_batch(tensor_days)
    fine_tune_affine_quotes(
        direct,
        batch,
        objective,
        training,
        progress=progress,
        checkpoint_path=checkpoint_path,
        checkpoint_every=training.checkpoint_every,
        configuration_id=configuration_id,
        source_id=source_id,
        resume=resume,
    )
    origins, endpoints = exact_weekly_pairs(tuple(day.trading_date for day in days))
    origin_indices = torch.as_tensor(origins, dtype=torch.long, device=device)
    endpoint_indices = torch.as_tensor(endpoints, dtype=torch.long, device=device)
    initialize_stable_dynamics(direct, origin_indices, endpoint_indices)
    return fit_joint_direct(
        direct,
        batch,
        origin_indices,
        endpoint_indices,
        objective,
        training,
        initialization,
        configuration_id=configuration_id,
        source_id=source_id,
        checkpoint_path=checkpoint_path,
        resume=resume,
        progress=progress,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "checkpoint_schema",
    "convergence_audit",
    "affine_fine_tune_epochs",
    "DirectFitResult",
    "NativeFitResult",
    "TrainingEvaluation",
    "evaluate_direct_model",
    "fit_direct200_pipeline",
    "fine_tune_affine_quotes",
    "fit_joint_direct",
    "fit_native_daily_states",
    "load_checkpoint",
    "native_raw_from_day",
    "native_fit_epochs",
    "save_checkpoint",
]
