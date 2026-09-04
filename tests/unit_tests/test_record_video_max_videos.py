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

"""RecordVideo ``max_videos`` stops after the first N flushed clips."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rlinf.envs.wrappers.record_video import RecordVideo


class _VideoCfg:
    def __init__(self, video_base_dir: str, max_videos: int) -> None:
        self.video_base_dir = video_base_dir
        self.save_video = True
        self.info_on_video = False
        self.save_camera_stills = False
        self.fps = 5
        self.max_videos = max_videos
        self.max_envs_in_video = 1
        self.video_image_keys = ["main_images"]

    def get(self, key, default=None):
        return getattr(self, key, default)


class _DummyEnv:
    def __init__(self) -> None:
        self.seed = 0
        self.num_envs = 1

    def reset(self, *args, **kwargs):
        return {"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)}, {}

    def step(self, action):
        obs = {"main_images": np.full((1, 8, 8, 3), 40, dtype=np.uint8)}
        return obs, 0.0, False, False, {}


def test_max_videos_stops_after_first_flush(tmp_path, monkeypatch):
    written: list[str] = []

    def _fake_save(self, frames, mp4_path):
        written.append(mp4_path)

    monkeypatch.setattr(RecordVideo, "_save_video", _fake_save)
    env = RecordVideo(_DummyEnv(), _VideoCfg(str(tmp_path), max_videos=1), fps=5)
    env.add_new_frames({"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)})
    env.flush_video()
    assert env.video_cnt == 1
    assert len(written) == 1

    env.add_new_frames({"main_images": np.full((1, 8, 8, 3), 9, dtype=np.uint8)})
    assert env.render_images == []
    env.flush_video()
    assert env.video_cnt == 1
    assert len(written) == 1


def test_max_videos_none_keeps_recording(tmp_path, monkeypatch):
    written: list[str] = []

    def _fake_save(self, frames, mp4_path):
        written.append(mp4_path)

    monkeypatch.setattr(RecordVideo, "_save_video", _fake_save)
    cfg = _VideoCfg(str(tmp_path), max_videos=1)
    cfg.max_videos = None
    env = RecordVideo(_DummyEnv(), cfg, fps=5)
    env.add_new_frames({"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)})
    env.flush_video()
    env.add_new_frames({"main_images": np.ones((1, 8, 8, 3), dtype=np.uint8)})
    env.flush_video()
    assert env.video_cnt == 2
    assert len(written) == 2


def test_flush_video_writes_under_step_subdir(tmp_path, monkeypatch):
    written: list[str] = []

    def _fake_save(self, frames, mp4_path):
        written.append(mp4_path)

    monkeypatch.setattr(RecordVideo, "_save_video", _fake_save)
    env = RecordVideo(_DummyEnv(), _VideoCfg(str(tmp_path), max_videos=2), fps=5)
    env.add_new_frames({"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)})
    env.flush_video(video_sub_dir="step_0")
    assert len(written) == 1
    assert written[0].endswith("step_0/0.mp4")
    assert "seed_0" in written[0]


def test_video_cfg_namespace_without_get(tmp_path):
    cfg = SimpleNamespace(
        video_base_dir=str(tmp_path),
        info_on_video=False,
        save_camera_stills=False,
        fps=5,
        max_videos=2,
        max_envs_in_video=1,
        video_image_keys=["main_images"],
    )
    env = RecordVideo(_DummyEnv(), cfg, fps=5)
    assert env._max_videos == 2
    assert env._recording_active()
