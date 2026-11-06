# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Each chat's capture: made when its chat starts, else read from disk.

A chat's capture lives in `captures/<record id>/chat-<n>/` of the workspace
(workspace.chat_directory()); its CaptureIndex is made, or read and its turns
derived, once and kept, so that the turns a Model Server adds are the turns
the analyses read.
"""

import pathlib
import threading
from typing import Any

from model_debugger.server.capture import capture_index
from model_debugger.server.session import session_store
from model_debugger.server.session import workspace


class CaptureStore:
  """The CaptureIndex of every chat of every Session, made or read once."""

  def __init__(
      self,
      workspace_: workspace.Workspace,
      sessions: session_store.SessionStore,
  ) -> None:
    """Keeps captures in `workspace_`, for the records of `sessions`."""
    self._workspace = workspace_
    self._sessions = sessions
    self._lock = threading.Lock()
    self._indexes: dict[tuple[str, int], capture_index.CaptureIndex] = {}

  def create(
      self, record_id: str, n: int, sides: dict[str, dict[str, Any]]
  ) -> capture_index.CaptureIndex:
    """Starts a chat's capture, with the fixed facts of each side.

    Args:
        record_id: The Session record.
        n: The chat's number.
        sides: Each side's capture_index.side_entry.

    Returns:
        The chat's new, empty index; get() answers it from now on.
    """
    index = capture_index.CaptureIndex.create(self._root(record_id, n), sides)
    with self._lock:
      self._indexes[record_id, n] = index
    return index

  def get(self, record_id: str, n: int) -> capture_index.CaptureIndex:
    """The stored capture of a Session's chat.

    A turn that cannot be derived is rejected on read, and the capture
    kept as read like any other.

    Args:
      record_id: The Session record ID.
      n: The chat's number.

    Returns:
      The stored capture index for the session's chat.

    Raises:
      NotFound: No such record or chat, or the chat has no capture.
      ValueError: The index is of another format version, or its turns
        are out of order.
    """
    session_store.chat(self._sessions.get(record_id), n)
    with self._lock:
      index = self._indexes.get((record_id, n))
    if index is None:
      index = capture_index.CaptureIndex.load(self._root(record_id, n))
      with self._lock:
        index = self._indexes.setdefault((record_id, n), index)
    return index

  @property
  def workspace(self) -> pathlib.Path:
    """The workspace directory the captures are in."""
    return self._workspace.root

  def forget(self, record_id: str) -> None:
    """Drops the indexes kept of a Session deleted; its files go apart."""
    with self._lock:
      for key in [key for key in self._indexes if key[0] == record_id]:
        del self._indexes[key]

  def _root(self, record_id: str, n: int) -> pathlib.Path:
    return self._workspace.captures / record_id / workspace.chat_directory(n)
