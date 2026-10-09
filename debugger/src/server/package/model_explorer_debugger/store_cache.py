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
# ==============================================================================

"""Bounded cache of capture readers.

Readers are invalidated by metadata and evidence file identity.
"""

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from .store import SessionStore


def revision(root):
  rows = []
  for path in sorted(Path(root).rglob('*')):
    if path.is_file():
      stat = path.stat()
      rows.append((
          str(path.relative_to(root)),
          stat.st_dev,
          stat.st_ino,
          stat.st_size,
          stat.st_mtime_ns,
          stat.st_ctime_ns,
      ))
  return tuple(rows)


class StoreCache:

  def __init__(self, max_entries=4, metadata_bytes=32 * 1024**2):
    self.max_entries = max_entries
    self.metadata_bytes = metadata_bytes
    self.values = OrderedDict()
    self.lock = RLock()
    self.hits = 0

  def get(self, root):
    root = Path(root).resolve()
    with self.lock:
      signature = revision(root)
      old = self.values.get(root)
      if old and old[0] == signature:
        self.hits += 1
        self.values.move_to_end(root)
        return old[1]
      self.values.pop(root, None)
      store = SessionStore(root)
      if revision(root) != signature:
        raise ValueError('Capture changed while loading; retry the query')
      size = sum(row[3] for row in signature if row[0].endswith('.json'))
      if size <= self.metadata_bytes:
        while self.values and (
            len(self.values) >= self.max_entries
            or sum(value[2] for value in self.values.values()) + size
            > self.metadata_bytes
        ):
          self.values.popitem(last=False)
        self.values[root] = (signature, store, size)
      return store
