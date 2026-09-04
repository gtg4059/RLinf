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

"""Initial-policy eval (``runner.eval_at_start``) and step-0 video flush."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from rlinf.utils.runner_utils import should_enable_eval, should_eval_at_start
from rlinf.workers.env.env_worker import EnvWorker


def test_should_eval_at_start_only_at_step_zero():
    cfg = OmegaConf.create({"runner": {"eval_at_start": True}})
    assert should_eval_at_start(cfg, 0)
    assert not should_eval_at_start(cfg, 1)


def test_should_eval_at_start_respects_flag():
    cfg = OmegaConf.create({"runner": {"eval_at_start": False}})
    assert not should_eval_at_start(cfg, 0)


def test_should_enable_eval_for_eval_at_start():
    cfg = OmegaConf.create(
        {
            "runner": {
                "val_check_interval": -1,
                "only_eval": False,
                "eval_at_start": True,
            }
        }
    )
    assert should_enable_eval(cfg)


def test_should_enable_eval_stays_off_without_flags():
    cfg = OmegaConf.create(
        {
            "runner": {
                "val_check_interval": -1,
                "only_eval": False,
                "eval_at_start": False,
            }
        }
    )
    assert not should_enable_eval(cfg)


def test_finish_rollout_writes_eval_video_under_step_0():
    worker = object.__new__(EnvWorker)
    worker.global_step = 0
    worker.stage_num = 1
    worker.cfg = SimpleNamespace(
        env=SimpleNamespace(
            eval=SimpleNamespace(
                video_cfg=SimpleNamespace(save_video=True),
                auto_reset=True,
            )
        )
    )
    flush_video = MagicMock()
    eval_env = SimpleNamespace(flush_video=flush_video)
    worker.eval_env_list = [eval_env]

    worker.finish_rollout(mode="eval")

    flush_video.assert_called_once_with(video_sub_dir="step_0")
