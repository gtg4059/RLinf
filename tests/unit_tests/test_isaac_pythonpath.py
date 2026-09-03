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

"""Isaac Sim ml_archive must not shadow OpenPI torch on actor/rollout."""

from __future__ import annotations

from rlinf.scheduler.cluster.cluster import Cluster
from rlinf.utils.isaac_pythonpath import (
    scrub_isaac_ml_archive_env,
    scrub_isaac_ml_archive_path,
    should_scrub_isaac_ml_archive,
    torch_is_isaac_build,
    worker_keeps_isaac_ml_archive,
)


class _FakeEnvWorker:
    __module__ = "rlinf.workers.env.env_worker"


class _FakeAsyncEnvWorker:
    __module__ = "rlinf.workers.env.async_env_worker"


class _FakeActor:
    __module__ = "rlinf.workers.actor.fsdp_actor_worker"


class _FakeRollout:
    __module__ = "rlinf.workers.rollout.hf.huggingface_worker"


def test_scrub_keeps_kit_paths_and_drops_ml_archive() -> None:
    raw = (
        "/workspace/RLinf:"
        "/workspace/isaac_sim/kit/python/lib/python3.11:"
        "/workspace/isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle:"
        "/workspace/isaac_sim/exts/omni.isaac.core_archive/pip_prebundle:"
        "/opt/venv/openpi/lib/python3.11/site-packages"
    )
    cleaned = scrub_isaac_ml_archive_path(raw)
    assert "omni.isaac.ml_archive" not in cleaned
    assert "/workspace/isaac_sim/kit/python/lib/python3.11" in cleaned
    assert "omni.isaac.core_archive" in cleaned
    assert "/opt/venv/openpi/lib/python3.11/site-packages" in cleaned


def test_scrub_env_drops_empty_pythonpath() -> None:
    env = scrub_isaac_ml_archive_env(
        {
            "PYTHONPATH": "/workspace/isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle",
            "FOO": "bar",
        }
    )
    assert "PYTHONPATH" not in env
    assert env["FOO"] == "bar"


def test_env_workers_keep_ml_archive() -> None:
    assert worker_keeps_isaac_ml_archive(_FakeEnvWorker)
    assert worker_keeps_isaac_ml_archive(_FakeAsyncEnvWorker)
    assert not worker_keeps_isaac_ml_archive(_FakeActor)
    assert not worker_keeps_isaac_ml_archive(_FakeRollout)
    assert not should_scrub_isaac_ml_archive(_FakeEnvWorker)
    assert should_scrub_isaac_ml_archive(_FakeActor)
    assert should_scrub_isaac_ml_archive(_FakeRollout)


def test_keep_flag_disables_scrub(monkeypatch) -> None:
    monkeypatch.setenv("RLINF_KEEP_ISAAC_ML_ARCHIVE", "1")
    assert not should_scrub_isaac_ml_archive(_FakeActor)


def test_cluster_helpers_match_utils() -> None:
    env = {
        "PYTHONPATH": (
            "/workspace/RLinf:"
            "/workspace/isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle"
        )
    }
    assert Cluster._should_scrub_isaac_ml_archive(_FakeActor)
    assert not Cluster._should_scrub_isaac_ml_archive(_FakeEnvWorker)
    cleaned = Cluster._scrub_isaac_ml_archive_env_vars(env)
    assert cleaned["PYTHONPATH"] == "/workspace/RLinf"


def test_torch_is_isaac_build_detects_archive_path() -> None:
    assert torch_is_isaac_build(
        "/workspace/isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/__init__.py"
    )
    assert not torch_is_isaac_build(
        "/opt/venv/openpi/lib/python3.11/site-packages/torch/__init__.py"
    )
