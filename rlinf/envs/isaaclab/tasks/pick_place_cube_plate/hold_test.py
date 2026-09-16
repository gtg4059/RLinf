"""Tests for done-env action/obs hold (no Isaac)."""

from __future__ import annotations

import torch

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.hold import (
    apply_hold_actions,
    clone_obs,
    overwrite_frozen_obs,
    update_hold_actions,
)


def test_apply_hold_actions_replaces_frozen_rows():
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hold = torch.tensor([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])
    frozen = torch.tensor([False, True, False])
    out = apply_hold_actions(actions, hold, frozen)
    assert torch.equal(out[0], actions[0])
    assert torch.equal(out[1], hold[1])
    assert torch.equal(out[2], actions[2])


def test_overwrite_frozen_obs_nested():
    obs = {"states": torch.tensor([[1.0], [2.0]]), "flag": torch.tensor([0, 1])}
    hold = {"states": torch.tensor([[9.0], [8.0]]), "flag": torch.tensor([3, 4])}
    frozen = torch.tensor([True, False])
    out = overwrite_frozen_obs(obs, hold, frozen)
    assert torch.equal(out["states"][0], hold["states"][0])
    assert torch.equal(out["states"][1], obs["states"][1])
    assert torch.equal(out["flag"][0], hold["flag"][0])
    assert torch.equal(out["flag"][1], obs["flag"][1])


def test_update_hold_actions_keeps_frozen():
    hold = torch.tensor([[1.0], [2.0]])
    actions = torch.tensor([[5.0], [6.0]])
    frozen = torch.tensor([True, False])
    out = update_hold_actions(hold, actions, frozen)
    assert torch.equal(out, torch.tensor([[1.0], [6.0]]))


def test_update_hold_actions_cpu_actions_cuda_mask():
    if not torch.cuda.is_available():
        return
    hold = torch.tensor([[1.0], [2.0]])
    actions = torch.tensor([[5.0], [6.0]])
    frozen = torch.tensor([True, False], device="cuda")
    out = update_hold_actions(hold, actions, frozen)
    assert out.device.type == "cpu"
    assert torch.equal(out, torch.tensor([[1.0], [6.0]]))


def test_apply_hold_actions_cpu_actions_cuda_mask():
    if not torch.cuda.is_available():
        return
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hold = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    frozen = torch.tensor([False, True], device="cuda")
    out = apply_hold_actions(actions, hold, frozen)
    assert torch.equal(out[1], hold[1])


def test_apply_hold_actions_mixed_float_double():
    """Isaac states are often float32; leftover chunk actions can be float64."""
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    hold = torch.tensor([[9.0, 9.0], [8.0, 8.0]], dtype=torch.float32)
    frozen = torch.tensor([False, True])
    out = apply_hold_actions(actions, hold, frozen)
    assert out.dtype == torch.float64
    assert torch.equal(out[0], actions[0])
    assert torch.allclose(out[1], hold[1].to(dtype=torch.float64))


def test_update_hold_actions_mixed_float_double():
    hold = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    actions = torch.tensor([[5.0], [6.0]], dtype=torch.float64)
    frozen = torch.tensor([True, False])
    out = update_hold_actions(hold, actions, frozen)
    assert out.dtype == torch.float64
    assert torch.allclose(out, torch.tensor([[1.0], [6.0]], dtype=torch.float64))


def test_clone_obs_is_detached():
    src = {"x": torch.tensor([1.0, 2.0])}
    cloned = clone_obs(src)
    cloned["x"][0] = 9.0
    assert float(src["x"][0]) == 1.0
