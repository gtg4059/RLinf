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

"""Length-prefixed pickle IPC for the CRI subprocess worker.

The worker writes replies on a dedicated pipe (``CRI_IPC_WRITE_FD``). Stdout
and stderr go to the CRI worker log so TensorRT / ``sfd_setup`` prints cannot
be parsed as a payload length.
"""

from __future__ import annotations

import pickle
import select
import struct
from typing import BinaryIO

CRI_IPC_WRITE_FD_ENV = "CRI_IPC_WRITE_FD"
# CRI replies are small arrays. A larger length is almost always a log line
# that leaked onto the pickle stream (e.g. ``[spike] ...``).
CRI_IPC_MAX_BYTES = 64 * 1024 * 1024


def decode_payload_size(header: bytes) -> int:
    """Parse an 8-byte little-endian length and reject implausible values."""
    if len(header) < 8:
        raise EOFError("CRI IPC header truncated")
    (n,) = struct.unpack("<Q", header)
    if n == 0 or n > CRI_IPC_MAX_BYTES:
        preview = header.decode("ascii", errors="replace")
        raise ValueError(
            f"CRI IPC length {n} is invalid (max {CRI_IPC_MAX_BYTES} bytes; "
            f"header hex={header.hex()} ascii={preview!r}). "
            "Worker stdout was likely polluted by logs."
        )
    return int(n)


def pickle_send(buf: BinaryIO, obj: object) -> None:
    """Write a length-prefixed pickle payload and flush."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    buf.write(struct.pack("<Q", len(payload)))
    buf.write(payload)
    buf.flush()


def pickle_recv(buf: BinaryIO, *, timeout_s: float | None = None) -> object:
    """Read one length-prefixed pickle payload."""
    if timeout_s is not None:
        ready, _, _ = select.select([buf], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"CRI worker did not respond within {timeout_s}s")
    header = buf.read(8)
    if len(header) < 8:
        raise EOFError("CRI worker exited before sending a reply")
    n = decode_payload_size(header)
    data = buf.read(n)
    if len(data) < n:
        raise EOFError("CRI worker reply truncated")
    return pickle.loads(data)
