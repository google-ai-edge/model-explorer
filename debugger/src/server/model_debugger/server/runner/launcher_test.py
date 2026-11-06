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

"""Tests of what every launched runner shares."""

import socket
import threading

from absl.testing import absltest
from model_debugger.server.runner import launcher
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main


class LauncherTest(absltest.TestCase):
  """Tests of what every launched runner shares."""

  def test_serving_needs_a_grpc_server(self):
    """Only a gRPC server counts as serving, and probing makes no RPC."""
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      self.assertTrue(launcher._serving(runner.address))
      # The probe made no RPC.
      self.assertEqual(runner.peers, [])
    # A listener that accepts and closes, as an adb forward does with no
    # App.
    with socket.socket() as listener:
      listener.bind(('127.0.0.1', 0))
      listener.listen()
      address = f'127.0.0.1:{listener.getsockname()[1]}'

      def accept_and_close():
        connection, _ = listener.accept()
        connection.close()

      threading.Thread(target=accept_and_close, daemon=True).start()
      self.assertFalse(launcher._serving(address))
    self.assertFalse(launcher._serving(address))


if __name__ == '__main__':
  test_main.main()
