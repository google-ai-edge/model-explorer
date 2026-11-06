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

"""JSON files replaced whole, so that a reader sees the old file or the new."""

import json
import os
import pathlib
from typing import Any
import uuid


def write_json(path: pathlib.Path, value: Any) -> None:
  """Writes `value` as JSON so that a reader sees the old file or the new one.

  Args:
      path: The file to replace.
      value: What to write; must be JSON serializable.
  """
  temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
  try:
    with open(temporary, 'w', encoding='utf-8') as file:
      json.dump(value, file, ensure_ascii=False, indent=1)
      file.flush()
      os.fsync(file.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  directory = os.open(path.parent, os.O_RDONLY)
  try:
    os.fsync(directory)
  finally:
    os.close(directory)


def read_json(path: pathlib.Path) -> Any:
  """Returns the JSON value in `path`."""
  with open(path, encoding='utf-8') as file:
    return json.load(file)
