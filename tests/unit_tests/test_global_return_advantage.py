"""One mean across opens and misses, so a rare open stays above the batch."""

import pytest
import torch

from rlinf.algorithms.advantages import compute_global_return_advantages


def test_an_open_stays_above_a_miss():
    """The sparse bonus is not cancelled by centering inside the open group."""
    rewards = torch.tensor([[0.01, 0.01], [0.90, 0.01]])
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[-1] = True
    advantages, returns = compute_global_return_advantages(
        rewards=torch.zeros(2),
        step_rewards=rewards,
        dones=dones,
    )
    assert returns is None
    assert advantages[0, 0].item() == pytest.approx(0.445)
    assert advantages[0, 1].item() == pytest.approx(-0.445)


def test_short_tail_after_a_success_is_left_out():
    """An open stays. The unfinished attempt after it does not train.

    Column 0 opens on step 1 and then has two steps with no done. Column 1
    is a miss that ends on a done. The tail's advantage is 0 and its loss
    mask is cleared. A second open and a timeout after an open stay in.
    """
    rewards = torch.tensor(
        [
            [0.01, 0.01],
            [1.00, 0.01],
            [-0.01, 0.01],
            [-0.01, 0.01],
        ]
    )
    dones = torch.zeros(5, 2, dtype=torch.bool)
    dones[2, 0] = True
    dones[-1, 1] = True
    step_loss = torch.ones(4, 2, 2, dtype=torch.bool)
    advantages, _ = compute_global_return_advantages(
        rewards=torch.zeros(2),
        step_rewards=rewards,
        dones=dones,
        step_loss_mask=step_loss,
    )
    assert advantages[2, 0].item() == 0.0
    assert advantages[3, 0].item() == 0.0
    assert not bool(step_loss[2, 0].any())
    assert not bool(step_loss[3, 0].any())
    assert bool(step_loss[1, 0].all())
    assert bool(step_loss[:, 1].all())

    again = torch.tensor([[0.01, 0.01], [1.00, 0.01], [0.01, -0.01], [1.00, -0.01]])
    again_dones = torch.zeros(5, 2, dtype=torch.bool)
    again_dones[2, 0] = True
    again_dones[4, 0] = True
    again_dones[-1, 1] = True
    kept, _ = compute_global_return_advantages(
        rewards=torch.zeros(2),
        step_rewards=again,
        dones=again_dones,
    )
    assert kept[3, 0].item() != 0.0
    assert kept[2, 1].item() != 0.0
    assert kept[3, 1].item() != 0.0


def test_misses_still_rank_when_nothing_opens():
    """A higher miss stays positive, so the batch can climb without an open."""
    rewards = torch.tensor([[0.20, 0.01], [0.01, 0.01]])
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[-1] = True
    advantages, _ = compute_global_return_advantages(
        rewards=torch.zeros(2),
        step_rewards=rewards,
        dones=dones,
    )
    assert advantages[0, 0].item() > 0.0
    assert advantages[0, 1].item() < 0.0
