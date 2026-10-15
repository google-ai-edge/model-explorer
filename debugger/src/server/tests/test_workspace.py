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

"""Tests of the workspace."""

import pathlib
import sys

from model_debugger_server import workspace
import pytest


def test_creates_directories_and_holds_lock(tmp_path):
  """Creates directories and holds lock."""
  opened = workspace.Workspace(tmp_path / 'ws')
  try:
    for directory in (
        opened.models,
        opened.records,
        opened.captures,
        opened.uploads,
    ):
      assert directory.is_dir()
      assert directory.parent == opened.root
    with pytest.raises(RuntimeError, match='Another Model Debugger server'):
      workspace.Workspace(tmp_path / 'ws')
  finally:
    opened.close()
  workspace.Workspace(tmp_path / 'ws').close()


def test_close_is_idempotent(tmp_path):
  """Close is idempotent."""
  opened = workspace.Workspace(tmp_path)
  opened.close()
  opened.close()


def test_removes_unfinished_uploads(tmp_path):
  """Removes unfinished uploads."""
  opened = workspace.Workspace(tmp_path)
  (opened.uploads / 'partial.litertlm').write_bytes(b'xx')
  opened.close()
  opened = workspace.Workspace(tmp_path)
  assert list(opened.uploads.iterdir()) == []
  opened.close()


def test_json_round_trip_leaves_no_temporary(tmp_path):
  """A JSON round trip leaves no temporary."""
  path = tmp_path / 'value.json'
  workspace.write_json(path, {'a': [1, 'ü']})
  workspace.write_json(path, {'a': [2]})
  assert workspace.read_json(path) == {'a': [2]}
  assert [p.name for p in tmp_path.iterdir()] == ['value.json']


def test_failed_write_keeps_the_old_file(tmp_path):
  """Failed write keeps the old file."""
  path = tmp_path / 'value.json'
  workspace.write_json(path, {'a': 1})
  with pytest.raises(TypeError):
    workspace.write_json(path, {'a': object()})
  assert workspace.read_json(path) == {'a': 1}
  assert [p.name for p in tmp_path.iterdir()] == ['value.json']


def test_default_root(monkeypatch):
  """The default root is the platform's place for application data."""
  root = workspace.default_root()
  assert root.parts[-2:] == ('model_debugger', 'server')
  if sys.platform == 'darwin':
    assert 'Application Support' in root.parts
  else:
    monkeypatch.setenv('XDG_DATA_HOME', '/data')
    assert workspace.default_root() == pathlib.Path(
        '/data/model_debugger/server'
    )
