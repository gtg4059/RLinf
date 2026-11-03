"""Policy loss ignores below-mean chunks, and success replay starts empty."""

import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import (
    compute_ppo_actor_critic_loss,
    compute_ppo_actor_loss,
)
from rlinf.algorithms.success_replay import SuccessReplay, hold_policy_update


def test_negative_advantage_does_not_enter_the_policy_loss():
    """A negative advantage has zero policy gradient. The value loss stays."""
    logprobs = torch.zeros(2, 1, requires_grad=True)
    old = torch.zeros(2, 1)
    advantages = torch.tensor([[-1.0], [1.0]])
    values = torch.zeros(2, 1, requires_grad=True)
    returns = torch.tensor([[0.2], [0.9]])
    prev_values = torch.zeros(2, 1)
    loss_mask = torch.ones(2, 1, dtype=torch.bool)
    actor_loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
        positive_advantage_only=True,
    )
    actor_loss.backward()
    assert logprobs.grad[0, 0].item() == 0.0
    assert logprobs.grad[1, 0].item() != 0.0

    logprobs2 = torch.zeros(2, 1, requires_grad=True)
    values2 = torch.zeros(2, 1, requires_grad=True)
    joint, _ = compute_ppo_actor_critic_loss(
        logprobs=logprobs2,
        old_logprobs=old,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
        values=values2,
        returns=returns,
        prev_values=prev_values,
        value_clip=10.0,
        huber_delta=10.0,
        positive_advantage_only=True,
    )
    joint.backward()
    assert values2.grad[0, 0].item() != 0.0


def test_first_mix_keeps_the_current_success_and_later_replaces_a_miss():
    """The empty buffer does not touch the first batch. The next mix does."""
    replay = SuccessReplay(fraction=0.5, capacity=4, success_reward_min=0.5)
    image = torch.zeros(2, 2, 1)
    image[:, 1] = 7
    first = {
        "rewards": torch.tensor([[0.01, 0.90], [0.01, 0.01]]),
        "obs": {"image": image},
    }
    assert replay.mix(first) == 0
    assert first["rewards"][0, 1].item() == pytest.approx(0.90)
    assert len(replay) == 1

    second = {
        "rewards": torch.tensor(
            [[0.01, 0.20, 0.05, 0.90], [0.01, 0.01, 0.01, 0.01]]
        ),
        "obs": {"image": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)},
    }
    assert replay.mix(second) == 1
    # Lowest miss is column 0 (return 0.02). The current open at column 3 stays.
    assert second["rewards"][0, 0].item() == pytest.approx(0.90)
    assert second["rewards"][0, 3].item() == pytest.approx(0.90)
    assert second["obs"]["image"][0, 3, 0].item() == 3.0
    assert second["obs"]["image"][0, 0, 0].item() == 7.0


def test_advantage_starts_at_the_door_move_and_a_dead_batch_holds():
    """Chunks before the door moves stay at 0, including inside a success."""
    rewards = torch.tensor(
        [
            [0.01, 0.01],
            [0.20, 0.01],
            [0.90, 0.01],
        ]
    )
    dones = torch.zeros(4, 2, dtype=torch.bool)
    dones[-1] = True
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=torch.zeros(4, 2),
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
        advantage_after_reach=True,
        reach_chunk_reward_min=0.1,
    )
    assert advantages[0, 0].item() == 0.0
    assert advantages[1, 0].item() != 0.0
    assert torch.all(advantages[:, 1] == 0)
    assert returns[0, 0].item() == pytest.approx(1.11)

    assert hold_policy_update(0.0, 32, 0.05)
    assert not hold_policy_update(0.0, 0, 0.05)
    assert not hold_policy_update(0.4, 32, 0.05)


def test_floor_refill_restores_opens_and_a_restored_batch_trains():
    """A batch under the floor is filled back. One restored open does not hold."""
    replay = SuccessReplay(fraction=0.0, capacity=4, success_reward_min=0.5)
    kept = {
        "rewards": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "marker": torch.tensor([[8.0, 1.0], [8.0, 1.0]]),
    }
    assert replay.mix(kept, floor_fraction=0.4) == 0
    assert len(replay) == 1

    dropped = {
        "rewards": torch.zeros(2, 4),
        "marker": torch.zeros(2, 4),
    }
    assert replay.mix(dropped, floor_fraction=0.4) == 1
    assert dropped["rewards"][0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert dropped["marker"][0, 0].item() == pytest.approx(8.0)
    assert not hold_policy_update(
        0.0, len(replay), 0.4, open_fraction_after_mix=0.25
    )
    assert hold_policy_update(0.0, len(replay), 0.4, open_fraction_after_mix=0.0)


def test_a_miss_keeps_one_fifth_of_its_advantage():
    """An open stays full after the door moves. A miss is scaled to 0.2."""
    rewards = torch.tensor(
        [
            [0.01, 0.01],
            [0.20, 0.01],
            [0.90, 0.01],
        ]
    )
    dones = torch.zeros(4, 2, dtype=torch.bool)
    dones[-1] = True
    common = dict(
        rewards=rewards,
        values=torch.zeros(4, 2),
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
        success_chunk_reward_min=0.5,
        reach_chunk_reward_min=0.1,
    )
    raw, _ = compute_gae_advantages_and_returns(**common)
    scaled, returns = compute_gae_advantages_and_returns(
        **common,
        advantage_after_reach=True,
        failure_advantage_scale=0.2,
    )
    assert scaled[0, 0].item() == 0.0
    assert scaled[1, 0].item() == pytest.approx(raw[1, 0].item())
    assert scaled[:, 1] == pytest.approx(raw[:, 1] * 0.2)
    assert returns[0, 0].item() == pytest.approx(1.11)

    misses = torch.tensor([[0.02, 0.01], [0.02, 0.01], [0.02, 0.01]])
    miss_dones = torch.zeros(4, 2, dtype=torch.bool)
    miss_dones[-1] = True
    better, _ = compute_gae_advantages_and_returns(
        rewards=misses,
        values=torch.zeros(4, 2),
        dones=miss_dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
        advantage_after_reach=True,
        failure_advantage_scale=0.2,
        success_chunk_reward_min=0.5,
        reach_chunk_reward_min=0.1,
    )
    assert better[0, 0].item() > 0.0
    assert better[0, 1].item() < 0.0
