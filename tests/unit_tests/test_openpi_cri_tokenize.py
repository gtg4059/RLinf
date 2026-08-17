# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
    NUM_CRI_POINTS,
)
from rlinf.models.embodiment.openpi.dataconfig.cri_tokenize import (
    TokenizePromptWithCri,
    tokenize_paligemma_with_cri,
)
from rlinf.models.embodiment.openpi.dataconfig.polaris_dataconfig import (
    DroidJointPosInputs,
)


class _FakePiece:
    def __init__(self):
        self.last_text = None
        self.calls = []

    def encode(self, text, add_bos=True):
        self.last_text = text
        self.calls.append(text)
        return [1, 2, 3, len(text)]


class _FakeTokenizer:
    def __init__(self, max_len=64):
        self._max_len = max_len
        self._tokenizer = _FakePiece()
        self.last_call = None

    def tokenize(self, prompt, state=None, *, cri=None):
        self.last_call = {"prompt": prompt, "state": state, "cri": cri}
        return tokenize_paligemma_with_cri(self, prompt, state, cri=cri)


def test_paligemma_cri_span_matches_droid_cri_string():
    tok = _FakeTokenizer()
    state = np.zeros(8, dtype=np.float32)
    cri = np.linspace(0.0, 2.0, NUM_CRI_POINTS, dtype=np.float32)
    tokens, mask = tokenize_paligemma_with_cri(tok, "pick up the cup", state, cri=cri)
    assert tokens.shape == (64,)
    assert mask.dtype == bool
    text = tok._tokenizer.last_text
    assert text.startswith("Task: pick up the cup, State:")
    assert ", CRI:" in text
    assert text.endswith("Action: ")
    # Without cri the official Pi05 string has no CRI span.
    tokenize_paligemma_with_cri(tok, "pick up the cup", state, cri=None)
    bins = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    expected = (
        "Task: pick up the cup, State: "
        + " ".join(map(str, bins))
        + ";\nAction: "
    )
    assert tok._tokenizer.last_text == expected


def test_tokenize_prompt_with_cri_forwards_same_arrays():
    tok = _FakeTokenizer()
    state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    cri = np.linspace(0.0, 2.0, NUM_CRI_POINTS, dtype=np.float32)
    out = TokenizePromptWithCri(tok, discrete_state_input=True, discrete_cri_input=True)(
        {"prompt": "pick", "state": state, "cri": cri, "keep": 1}
    )
    assert "cri" not in out
    assert out["keep"] == 1
    np.testing.assert_array_equal(tok.last_call["state"], state)
    np.testing.assert_allclose(tok.last_call["cri"], cri)
    assert tok.last_call["cri"].dtype == np.float32
    assert tok.last_call["prompt"] == "pick"


def test_tokenize_prompt_with_cri_requires_cri():
    transform = TokenizePromptWithCri(
        _FakeTokenizer(), discrete_state_input=True, discrete_cri_input=True
    )
    with pytest.raises(ValueError, match="CRI is required"):
        transform({"prompt": "pick", "state": np.zeros(8, dtype=np.float32)})


def test_droid_joint_pos_inputs_forwards_cri():
    cri = np.linspace(0.0, 1.0, NUM_CRI_POINTS, dtype=np.float32)
    data = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/cri": cri,
        "prompt": "pick",
    }
    out = DroidJointPosInputs(action_dim=32, use_cri=True)(data)
    assert out["cri"].shape == (NUM_CRI_POINTS,)
    np.testing.assert_allclose(out["cri"], cri)


def test_droid_joint_pos_inputs_cri_missing_raises():
    data = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "pick",
    }
    with pytest.raises(KeyError, match="observation/cri"):
        DroidJointPosInputs(action_dim=32, use_cri=True)(data)
