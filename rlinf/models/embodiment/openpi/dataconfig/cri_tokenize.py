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

"""droid-cri Paligemma CRI span on the installed OpenPI tokenizer.

Installed ``openpi`` has no ``tokenize(..., cri=)`` and no
``TokenizePrompt(discrete_cri_input=True)``. This module does not invent a
second prompt format. It:

1. Patches ``PaligemmaTokenizer.tokenize`` to the droid-cri body (same
   digitize bins, same ``Task: ..., State: ..., CRI: ...;\\nAction: `` string).
2. Mirrors droid-cri ``TokenizePrompt`` so the arrays passed in are
   ``prompt``, normalized ``state``, and ``cri`` float32 ``(9,)``.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging

import numpy as np
from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer


def tokenize_paligemma_with_cri(
    self: _tokenizer.PaligemmaTokenizer,
    prompt: str,
    state: np.ndarray | None = None,
    *,
    cri: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """droid-cri ``PaligemmaTokenizer.tokenize`` (byte-for-byte control flow)."""
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        if cri is not None:
            cri_arr = np.asarray(cri, dtype=np.float64).reshape(-1)
            discretized_cri = np.digitize(cri_arr, bins=np.linspace(0, 2, 256 + 1)[:-1]) - 1
            cri_str = " ".join(map(str, discretized_cri))
            full_prompt = f"Task: {cleaned_text}, State: {state_str}, CRI: {cri_str};\nAction: "
        else:
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
    else:
        if cri is not None:
            raise ValueError(
                "cri tokens require discrete state input (Pi05); state must not be None"
            )
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode(
            "\n"
        )
    tokens_len = len(tokens)
    if tokens_len < self._max_len:
        padding = [False] * (self._max_len - tokens_len)
        mask = [True] * tokens_len + padding
        tokens = tokens + padding
    else:
        if len(tokens) > self._max_len:
            logging.warning(
                "Token length (%s) exceeds max length (%s), truncating. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently.",
                tokens_len,
                self._max_len,
            )
        tokens = tokens[: self._max_len]
        mask = [True] * self._max_len
    return np.asarray(tokens), np.asarray(mask)


def ensure_paligemma_cri_tokenize() -> None:
    """Install droid-cri ``cri=`` on site-packages Paligemma if missing."""
    tokenize = _tokenizer.PaligemmaTokenizer.tokenize
    if "cri" in inspect.signature(tokenize).parameters:
        return
    _tokenizer.PaligemmaTokenizer.tokenize = tokenize_paligemma_with_cri


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithCri(_transforms.DataTransformFn):
    """droid-cri ``TokenizePrompt`` (``discrete_cri_input=True``).

    Does not build the CRI string itself. Forwards the same ``prompt``,
    ``state``, and ``cri`` arrays to ``tokenizer.tokenize(..., cri=cri)``.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    discrete_cri_input: bool = True

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        cri = None
        if self.discrete_cri_input:
            if (cri := data.get("cri", None)) is None:
                raise ValueError("CRI is required when discrete_cri_input=True")
            cri = np.asarray(cri, dtype=np.float32).reshape(-1)

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state, cri=cri)
        out = {k: v for k, v in data.items() if k != "cri"}
        return {**out, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
