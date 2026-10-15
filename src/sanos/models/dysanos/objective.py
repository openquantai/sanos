"""Canonical model-centred DySANOS 1.1 quote objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from cdxcore.version import version

if TYPE_CHECKING:
    from sanos.models.dysanos.data import DailyFrontDay


@version("1.0")
@dataclass(frozen=True, slots=True)
class TensorDay:
    """One immutable public market day on a Torch device."""

    source_index: int
    price_expiries_y: torch.Tensor
    price_strikes: torch.Tensor
    quote_price_index: torch.Tensor
    quote_strikes: torch.Tensor
    quote_is_call: torch.Tensor
    quote_fit: torch.Tensor
    quote_half_spread: torch.Tensor


@version("1.0")
@dataclass(frozen=True, slots=True)
class TensorBatch:
    """Padded market days with an exact prefix mask."""

    day_indices: torch.Tensor
    price_expiries_y: torch.Tensor
    price_strikes: torch.Tensor
    quote_price_index: torch.Tensor
    quote_strikes: torch.Tensor
    quote_is_call: torch.Tensor
    quote_fit: torch.Tensor
    quote_half_spread: torch.Tensor
    quote_mask: torch.Tensor


@version("1.0", dependencies=[TensorDay])
def tensorize_day(
    day: DailyFrontDay,
    source_index: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> TensorDay:
    """Copy only the public objective fields to the training device."""

    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    floating = lambda value: torch.as_tensor(value, dtype=dtype, device=device)  # noqa: E731
    price_expiries = floating(day.price_expiry_y)
    price_strikes = floating(day.price_strike)
    quote_indices = torch.as_tensor(day.quote_price_index, dtype=torch.long, device=device)
    quote_strikes = floating(day.quote_strike)
    quote_sides = torch.as_tensor(day.quote_is_call, dtype=torch.bool, device=device)
    quote_fit = floating(day.quote_fit)
    half_spread = floating(day.quote_half_spread)
    quote_count = len(quote_strikes)
    if quote_count == 0 or any(
        len(value) != quote_count
        for value in (quote_indices, quote_sides, quote_fit, half_spread)
    ):
        raise ValueError("DySANOS day quote fields have inconsistent lengths")
    if len(price_strikes) == 0 or len(price_expiries) != len(price_strikes):
        raise ValueError("DySANOS day price locations are empty or inconsistent")
    if bool(torch.any(quote_indices < 0)) or bool(torch.any(quote_indices >= len(price_strikes))):
        raise ValueError("DySANOS quote price index is out of range")
    if bool(torch.any(~torch.isfinite(half_spread))) or bool(torch.any(half_spread <= 0.0)):
        raise ValueError("Every DySANOS half-spread must be positive and finite")
    return TensorDay(
        source_index,
        price_expiries,
        price_strikes,
        quote_indices,
        quote_strikes,
        quote_sides,
        quote_fit,
        half_spread,
    )


@version("1.0", dependencies=[TensorBatch, TensorDay])
def make_batch(days: tuple[TensorDay, ...]) -> TensorBatch:
    """Pad days without changing any real-book weight."""

    if not days:
        raise ValueError("Cannot create an empty DySANOS batch")
    device = days[0].quote_fit.device
    dtype = days[0].quote_fit.dtype
    if any(day.quote_fit.device != device or day.quote_fit.dtype != dtype for day in days):
        raise ValueError("All DySANOS batch days must share dtype and device")
    max_quotes = max(len(day.quote_fit) for day in days)
    max_prices = max(len(day.price_strikes) for day in days)

    def pad(value: torch.Tensor, length: int, fill: float | int | bool) -> torch.Tensor:
        return torch.nn.functional.pad(value, (0, length - len(value)), value=fill)

    mask = torch.zeros((len(days), max_quotes), dtype=torch.bool, device=device)
    for row, day in enumerate(days):
        mask[row, : len(day.quote_fit)] = True
    return TensorBatch(
        day_indices=torch.as_tensor(
            [day.source_index for day in days], dtype=torch.long, device=device
        ),
        price_expiries_y=torch.stack(
            [pad(day.price_expiries_y, max_prices, 0.0) for day in days]
        ),
        price_strikes=torch.stack(
            [pad(day.price_strikes, max_prices, 1.0) for day in days]
        ),
        quote_price_index=torch.stack(
            [pad(day.quote_price_index, max_quotes, 0) for day in days]
        ),
        quote_strikes=torch.stack(
            [pad(day.quote_strikes, max_quotes, 1.0) for day in days]
        ),
        quote_is_call=torch.stack(
            [pad(day.quote_is_call, max_quotes, True) for day in days]
        ),
        quote_fit=torch.stack([pad(day.quote_fit, max_quotes, 0.0) for day in days]),
        quote_half_spread=torch.stack(
            [pad(day.quote_half_spread, max_quotes, 1.0) for day in days]
        ),
        quote_mask=mask,
    )


@version("1.0")
def restore_quote_prices(call_prices: torch.Tensor, batch: TensorBatch) -> torch.Tensor:
    """Map unique call prices back to separate public call/put books."""

    if call_prices.shape != batch.price_strikes.shape:
        raise ValueError("Call-price matrix does not match unique price locations")
    selected_calls = torch.gather(call_prices, 1, batch.quote_price_index)
    return torch.where(
        batch.quote_is_call,
        selected_calls,
        selected_calls - 1.0 + batch.quote_strikes,
    )


@version("1.0")
def per_day_quote_loss(
    prices: torch.Tensor,
    batch: TensorBatch,
    *,
    fit_weight: float,
) -> torch.Tensor:
    """Return the exact mean model-centred spread loss for each day."""

    if prices.shape != batch.quote_fit.shape or fit_weight < 0.0:
        raise ValueError("Invalid DySANOS quote objective inputs")
    lower = batch.quote_fit - batch.quote_half_spread
    upper = batch.quote_fit + batch.quote_half_spread
    excess = torch.relu(prices - upper) + torch.relu(lower - prices)
    central = torch.abs(prices - batch.quote_fit)
    loss = (excess + fit_weight * central) / batch.quote_half_spread
    valid = batch.quote_mask.to(loss.dtype)
    counts = valid.sum(dim=1)
    if bool(torch.any(counts <= 0.0)):
        raise ValueError("Every DySANOS batch day must contain a real book")
    return (loss * valid).sum(dim=1) / counts


@version("1.0", dependencies=[per_day_quote_loss])
def quote_objective(
    prices: torch.Tensor,
    batch: TensorBatch,
    *,
    fit_weight: float,
) -> torch.Tensor:
    """Weight dates equally after averaging books within each date."""

    return per_day_quote_loss(prices, batch, fit_weight=fit_weight).mean()


@version("1.0")
@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    objective: float
    mean_absolute_half_spreads: float
    percent_books_outside: float
    maximum_excess_half_spreads: float
    book_count: int


@version("1.0", dependencies=[ObjectiveEvaluation, per_day_quote_loss])
def evaluate_quote_objective(
    prices: torch.Tensor,
    batch: TensorBatch,
    *,
    fit_weight: float,
) -> ObjectiveEvaluation:
    """Return deterministic public-book fit diagnostics."""

    day_loss = per_day_quote_loss(prices, batch, fit_weight=fit_weight)
    lower = batch.quote_fit - batch.quote_half_spread
    upper = batch.quote_fit + batch.quote_half_spread
    excess = (
        torch.relu(prices - upper) + torch.relu(lower - prices)
    ) / batch.quote_half_spread
    selected = excess[batch.quote_mask]
    central = torch.abs(prices - batch.quote_fit)[batch.quote_mask] / batch.quote_half_spread[
        batch.quote_mask
    ]
    return ObjectiveEvaluation(
        objective=float(day_loss.mean().detach().cpu()),
        mean_absolute_half_spreads=float(central.mean().detach().cpu()),
        percent_books_outside=float((selected > 0.0).to(torch.float64).mean().cpu() * 100.0),
        maximum_excess_half_spreads=float(selected.max().detach().cpu()),
        book_count=int(selected.numel()),
    )


__all__ = [
    "ObjectiveEvaluation",
    "TensorBatch",
    "TensorDay",
    "evaluate_quote_objective",
    "make_batch",
    "per_day_quote_loss",
    "quote_objective",
    "restore_quote_prices",
    "tensorize_day",
]
