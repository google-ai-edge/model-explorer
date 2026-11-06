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

"""Tests of the Server facade: starting, its state, closing."""

import pathlib
import shutil
import socket
from unittest import mock

from absl.testing import absltest
from model_debugger.server import server
from model_debugger.server.runner import android_launcher
from model_debugger.server.testing import test_main


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind((server.HOST, 0))
    return probe.getsockname()[1]


class ServerTest(absltest.TestCase):
  """Tests of the Server facade: starting, its state, closing."""

  def setUp(self):
    """An empty directory; no adb and no xcrun, so no device."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.enter_context(
        mock.patch.object(android_launcher, 'find_adb', lambda: None)
    )
    self.enter_context(mock.patch.object(shutil, 'which', lambda tool: None))

  def test_start_reports_the_port_and_the_workspace(self):
    """Port 0 picks a free port; the workspace is reported resolved."""
    started = server.Server.start(
        port=0,
        workspace=self.tmp_path / 'a/../ws',
        runner=None,
        ui=None,
    )
    try:
      state = started.state()
      self.assertNotEqual(state.port, 0)
      self.assertEqual(state.workspace, (self.tmp_path / 'ws').resolve())
    finally:
      started.close()
    started.close()

  def test_a_held_workspace_refuses_a_second_server_and_frees_its_port(self):
    """A second Server on the workspace fails and releases its port."""
    first = server.Server.start(
        port=0, workspace=self.tmp_path, runner=None, ui=None
    )
    port = _free_port()
    try:
      with self.assertRaisesRegex(
          RuntimeError, 'Another Model Debugger server'
      ):
        server.Server.start(
            port=port,
            workspace=self.tmp_path,
            runner=None,
            ui=None,
        )
    finally:
      first.close()
    # Closed, the workspace and that port may be used again.
    again = server.Server.start(
        port=port, workspace=self.tmp_path, runner=None, ui=None
    )
    try:
      self.assertEqual(again.state().port, port)
    finally:
      again.close()

  def test_a_port_in_use_is_unavailable(self):
    """A port another socket listens on is refused, naming the port.

    On every address, not HOST alone: SO_REUSEADDR would let HOST be
    bound under such a listener.
    """
    for address in (server.HOST, '0.0.0.0'):
      with self.subTest(address=address), socket.socket() as taken:
        taken.bind((address, 0))
        taken.listen()
        port = taken.getsockname()[1]
        with self.assertRaises(server.PortInUseError) as refused:
          server.Server.start(
              port=port,
              workspace=self.tmp_path,
              runner=None,
              ui=None,
          )
        self.assertEqual(refused.exception.port, port)
        self.assertEqual(str(refused.exception), f'Port {port} is in use')

  def test_without_a_port_the_next_free_default_is_taken(self):
    """The first default port held, the next free one is listened on."""
    with socket.socket() as taken:
      taken.bind((server.HOST, 0))
      taken.listen()
      first = taken.getsockname()[1]
      # Default ports of the test's, from the held one: never 8080.
      self.enter_context(mock.patch.object(server, 'DEFAULT_PORT', first))
      self.enter_context(
          mock.patch.object(server, 'LAST_DEFAULT_PORT', first + 9)
      )
      started = server.Server.start(
          port=None,
          workspace=self.tmp_path,
          runner=None,
          ui=None,
      )
      try:
        self.assertBetween(started.state().port, first + 1, first + 9)
      finally:
        started.close()

  def test_every_default_port_in_use_is_unavailable(self):
    """With every default port held, none is listened on."""
    with socket.socket() as taken:
      taken.bind((server.HOST, 0))
      taken.listen()
      port = taken.getsockname()[1]
      self.enter_context(mock.patch.object(server, 'DEFAULT_PORT', port))
      self.enter_context(mock.patch.object(server, 'LAST_DEFAULT_PORT', port))
      with self.assertRaises(server.PortInUseError) as refused:
        server.Server.start(
            port=None,
            workspace=self.tmp_path,
            runner=None,
            ui=None,
        )
    self.assertIsNone(refused.exception.port)


if __name__ == '__main__':
  test_main.main()
