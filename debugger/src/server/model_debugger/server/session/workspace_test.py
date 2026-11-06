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

import os
import pathlib
import sys
from unittest import mock

from absl.testing import absltest
from model_debugger.server.session import workspace
from model_debugger.server.testing import test_main


class WorkspaceTest(absltest.TestCase):
  """Tests of the workspace."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_creates_directories_and_holds_lock(self):
    """Creates directories and holds lock."""
    opened = workspace.Workspace(self.tmp_path / 'ws')
    try:
      for directory in (
          opened.models,
          opened.records,
          opened.captures,
          opened.uploads,
          opened.prepared,
          opened.graphs,
          opened.mappings,
          opened.facts,
      ):
        self.assertTrue(directory.is_dir())
        self.assertEqual(directory.parent, opened.root)
      with self.assertRaisesRegex(
          RuntimeError, 'Another Model Debugger server'
      ):
        workspace.Workspace(self.tmp_path / 'ws')
    finally:
      opened.close()
    workspace.Workspace(self.tmp_path / 'ws').close()

  def test_close_is_idempotent(self):
    """Close is idempotent."""
    opened = workspace.Workspace(self.tmp_path)
    opened.close()
    opened.close()

  def test_removes_unfinished_uploads(self):
    """Removes unfinished uploads."""
    opened = workspace.Workspace(self.tmp_path)
    (opened.uploads / 'partial.litertlm').write_bytes(b'xx')
    # A Session being imported is unpacked into a directory.
    (opened.uploads / 'unpacking' / 'captures').mkdir(parents=True)
    opened.close()
    opened = workspace.Workspace(self.tmp_path)
    self.assertEqual(list(opened.uploads.iterdir()), [])
    opened.close()

  def test_removes_unfinished_copies_and_keeps_finished_ones(self):
    """A copy the last server did not finish goes; a kept one stays."""
    opened = workspace.Workspace(self.tmp_path)
    pending = opened.prepared / (workspace.PENDING + 'abc')
    pending.mkdir()
    (pending / 'model.litertlm').write_bytes(b'half')
    kept = opened.prepared / ('0' * 64)
    kept.mkdir()
    (kept / 'model.litertlm').write_bytes(b'whole')
    opened.close()
    opened = workspace.Workspace(self.tmp_path)
    self.assertEqual(list(opened.prepared.iterdir()), [kept])
    self.assertEqual((kept / 'model.litertlm').read_bytes(), b'whole')
    opened.close()

  def test_an_older_servers_bookmarks_go(self):
    """The bookmarks/ an older server kept goes as the workspace opens."""
    (self.tmp_path / 'bookmarks').mkdir()
    (self.tmp_path / 'bookmarks' / 'a.json').write_text('{}')
    opened = workspace.Workspace(self.tmp_path)
    self.addCleanup(opened.close)
    self.assertFalse((self.tmp_path / 'bookmarks').exists())

  def test_a_sessions_files_go_and_leave_the_others(self):
    """A Session's capture goes; another's stays."""
    opened = workspace.Workspace(self.tmp_path)
    self.addCleanup(opened.close)
    for record_id in ('gone', 'kept'):
      (opened.captures / record_id / 'turns').mkdir(parents=True)
    opened.remove_session_files('gone')
    self.assertEqual(
        list(opened.captures.iterdir()), [opened.captures / 'kept']
    )
    # One with no capture, as a Session that never ran.
    opened.remove_session_files('draft')

  def test_files_no_record_names_go(self):
    """Captures of no record go, whatever their names."""
    opened = workspace.Workspace(self.tmp_path)
    self.addCleanup(opened.close)
    for name in ('kept', 'orphan', workspace.PENDING + 'abc'):
      (opened.captures / name / 'turns').mkdir(parents=True)
    (opened.captures / 'stray').write_text('x')
    opened.remove_orphan_session_files({'kept': {}})
    self.assertEqual(
        list(opened.captures.iterdir()), [opened.captures / 'kept']
    )
    self.assertTrue((opened.captures / 'kept' / 'turns').is_dir())

  def test_default_root(self):
    """The default root is the platform's place for application data."""
    root = workspace.default_root()
    self.assertEqual(root.parts[-2:], ('model_debugger', 'server'))
    if sys.platform == 'darwin':
      self.assertIn('Application Support', root.parts)
    else:
      self.enter_context(
          mock.patch.dict(os.environ, {'XDG_DATA_HOME': '/data'})
      )
      self.assertEqual(
          workspace.default_root(),
          pathlib.Path('/data/model_debugger/server'),
      )


if __name__ == '__main__':
  test_main.main()
