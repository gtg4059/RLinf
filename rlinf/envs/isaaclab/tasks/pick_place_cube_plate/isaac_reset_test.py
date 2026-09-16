"""Tests for Isaac in-step reset helpers (no Isaac)."""

from __future__ import annotations

import torch

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.isaac_reset import (
    isaac_reset_row_ids,
    zero_rows,
)


def test_isaac_reset_row_ids_union():
    term = torch.tensor([True, False, False])
    trunc = torch.tensor([False, False, True])
    ids = isaac_reset_row_ids(term, trunc)
    assert torch.equal(ids, torch.tensor([0, 2]))


def test_isaac_reset_row_ids_empty():
    term = torch.zeros(3, dtype=torch.bool)
    trunc = torch.zeros(3, dtype=torch.bool)
    ids = isaac_reset_row_ids(term, trunc)
    assert int(ids.numel()) == 0


def test_zero_rows_leaves_other_rows():
    cache = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    zero_rows(cache, torch.tensor([1]))
    assert torch.equal(cache[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(cache[1], torch.tensor([0.0, 0.0]))
    assert torch.equal(cache[2], torch.tensor([5.0, 6.0]))


def test_zero_rows_noop_on_empty_ids():
    cache = torch.tensor([[1.0], [2.0]])
    zero_rows(cache, torch.tensor([], dtype=torch.long))
    assert torch.equal(cache, torch.tensor([[1.0], [2.0]]))


def test_set_chunk_hold_writes_reset_rows():
    from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.isaac_reset import (
        set_chunk_hold,
    )

    mask = torch.zeros(3, dtype=torch.bool)
    hold = torch.zeros(3, 2)
    states = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mask, hold = set_chunk_hold(mask, hold, states, torch.tensor([1]))
    assert bool(mask[1])
    assert not bool(mask[0])
    assert torch.equal(hold[1], states[1])
    assert torch.equal(hold[0], torch.tensor([0.0, 0.0]))


def test_set_chunk_hold_casts_hold_to_states_dtype():
    from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.isaac_reset import (
        set_chunk_hold,
    )

    mask = torch.zeros(2, dtype=torch.bool)
    hold = torch.zeros(2, 2, dtype=torch.float32)
    states = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    mask, hold = set_chunk_hold(mask, hold, states, torch.tensor([1]))
    assert hold.dtype == torch.float64
    assert torch.equal(hold[1], states[1])
