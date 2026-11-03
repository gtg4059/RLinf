"""Actor advantage on a miss drops the chunks before the door moves."""

import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns


def _run(rewards: torch.Tensor, dones: torch.Tensor, **kwargs):
    values = torch.zeros(rewards.shape[0] + 1, rewards.shape[1])
    return compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
        zero_failure_prefix=True,
        success_chunk_reward_min=0.5,
        reach_chunk_reward_min=0.1,
        **kwargs,
    )


def test_miss_prefix_is_zero_and_returns_stay_raw():
    """A success keeps its prefix. A reach miss keeps the tail. A plain miss is 0."""
    rewards = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.01, 0.20, 0.01],
            [0.90, 0.01, 0.01],
        ]
    )
    dones = torch.zeros(4, 3, dtype=torch.bool)
    dones[-1] = True
    advantages, returns = _run(rewards, dones)

    assert torch.allclose(returns[:, 0], torch.tensor([0.92, 0.91, 0.90]))
    assert torch.allclose(returns[:, 1], torch.tensor([0.22, 0.21, 0.01]))
    assert torch.allclose(returns[:, 2], torch.tensor([0.03, 0.02, 0.01]))

    mean = torch.tensor([0.92, 0.22, 0.03]).mean()
    assert advantages[0, 0].item() == pytest.approx(0.92 - mean)
    assert advantages[1, 0].item() == pytest.approx(0.91 - mean)
    assert advantages[0, 1].item() == pytest.approx(0.0)
    assert advantages[1, 1].item() == pytest.approx(0.21 - mean)
    assert advantages[2, 1].item() == pytest.approx(0.01 - mean)
    assert torch.all(advantages[:, 2] == 0)


def test_open_attempt_then_miss_only_zeros_the_second_prefix():
    """The open keeps both chunks. The later miss drops only its approach chunk."""
    rewards = torch.tensor([[0.01], [0.90], [0.01], [0.20]])
    dones = torch.tensor([[False], [False], [True], [False], [True]])
    advantages, returns = _run(rewards, dones)

    assert returns[0, 0].item() == pytest.approx(0.91)
    assert returns[2, 0].item() == pytest.approx(0.21)
    mean = torch.tensor([0.91, 0.21]).mean()
    assert advantages[0, 0].item() == pytest.approx(0.91 - mean)
    assert advantages[1, 0].item() == pytest.approx(0.90 - mean)
    assert advantages[2, 0].item() == pytest.approx(0.0)
    assert advantages[3, 0].item() == pytest.approx(0.20 - mean)


def test_prefix_stays_when_the_flag_is_off():
    """Without the flag the reach miss still carries a negative approach."""
    rewards = torch.tensor([[0.01, 0.01], [0.90, 0.20]])
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[-1] = True
    values = torch.zeros(3, 2)
    advantages, _ = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
        zero_failure_prefix=False,
    )
    assert advantages[0, 1].item() != 0.0
