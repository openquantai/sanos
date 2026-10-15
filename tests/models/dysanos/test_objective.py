from __future__ import annotations

import pytest
import torch

from sanos.models.dysanos.objective import (
    TensorDay,
    evaluate_quote_objective,
    make_batch,
    per_day_quote_loss,
    restore_quote_prices,
)


def _day(source_index: int, fit: tuple[float, ...]) -> TensorDay:
    count = len(fit)
    return TensorDay(
        source_index=source_index,
        price_expiries_y=torch.full((count,), 1.0 / 252.0),
        price_strikes=torch.tensor([0.9, 1.1][:count]),
        quote_price_index=torch.arange(count),
        quote_strikes=torch.tensor([0.9, 1.1][:count]),
        quote_is_call=torch.tensor([True, False][:count]),
        quote_fit=torch.tensor(fit),
        quote_half_spread=torch.full((count,), 0.01),
    )


def test_model_centred_spread_loss_weights_days_equally() -> None:
    batch = make_batch((_day(0, (0.12, 0.12)), _day(1, (0.20,))))
    calls = torch.tensor([[0.12, 0.02], [0.25, 0.0]])
    prices = restore_quote_prices(calls, batch)
    losses = per_day_quote_loss(prices, batch, fit_weight=0.0)
    torch.testing.assert_close(losses, torch.tensor([0.0, 4.0]))
    diagnostics = evaluate_quote_objective(prices, batch, fit_weight=0.0)
    assert diagnostics.objective == pytest.approx(2.0)
    assert diagnostics.book_count == 3
    assert diagnostics.percent_books_outside == pytest.approx(100.0 / 3.0)


def test_puts_are_reconstructed_from_call_equivalence() -> None:
    batch = make_batch((_day(0, (0.12, 0.12)),))
    calls = torch.tensor([[0.12, 0.02]])
    restored = restore_quote_prices(calls, batch)
    torch.testing.assert_close(restored, torch.tensor([[0.12, 0.12]]))
