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

"""The command with a --port that is in use, run as the user runs it."""

import socket
import subprocess
import sys

from absl.testing import absltest
from model_debugger.server.testing import test_main


class ServerMainPortTest(absltest.TestCase):
  """The command with a --port that is in use."""

  def test_a_port_in_use_exits_with_the_reason_alone(self):
    """It names the port on stderr, exits 1 and shows no traceback."""
    tmp = self.create_tempdir()
    with socket.socket() as taken:
      taken.bind(('127.0.0.1', 0))
      taken.listen()
      port = taken.getsockname()[1]
      ended = subprocess.run(
          [
              sys.executable,
              '-m',
              'model_debugger.server.cli.server_main',
              '--port',
              str(port),
              '--workspace',
              tmp.mkdir('workspace').full_path,
              '--kept_builds',
              tmp.mkdir('kept').full_path,
              '--pytorch_runner',
              '',
          ],
          capture_output=True,
          text=True,
          timeout=120,
          check=False,
      )
    self.assertEqual(ended.returncode, 1, ended.stderr)
    self.assertEqual(ended.stdout, '')
    self.assertStartsWith(
        ended.stderr,
        f'model-debugger-server: Port {port} is in use: ',
    )
    self.assertNotIn('Traceback', ended.stderr)


if __name__ == '__main__':
  test_main.main()
