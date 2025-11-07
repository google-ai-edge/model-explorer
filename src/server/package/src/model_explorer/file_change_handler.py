# Copyright 2025 The AI Edge Model Explorer Authors.
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

import os
from typing import Callable

from watchdog.events import DirModifiedEvent, FileModifiedEvent, FileSystemEventHandler


class FileChangeHandler(FileSystemEventHandler):
  """Handles file system events to detect changes in specified files."""

  def __init__(self,
               host: str, port: int,
               callback: Callable[[str, int], None]):
    super().__init__()
    self.host = host
    self.port = port
    self.target_file_paths = []
    self.callback = callback

  def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
    if not event.is_directory:
      if os.path.abspath(event.src_path) in self.target_file_paths:
        self.callback(self.host, self.port)

  def add_target_file_path(self, file_path: str):
    self.target_file_paths.append(file_path)
