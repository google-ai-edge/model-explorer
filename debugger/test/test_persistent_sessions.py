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

"""Runner close behavior with isolated requests and synthetic transports."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from model_debugger_runner.python_workers import PythonWorkers
from model_explorer_debugger.runtime.ios_device import IOSDevices


class PersistentSessionTests(unittest.TestCase):

  def test_closed_python_runner_does_not_auto_load_on_generate(self):
    with tempfile.TemporaryDirectory() as folder:
      manager = PythonWorkers()
      self.addCleanup(manager.close)
      request = dict(
          operation='generate',
          session_id='session-a',
          run={'id': 'ref', 'runtime': 'PyTorch'},
          output=str(Path(folder) / 'output'),
      )
      with self.assertRaisesRegex(ValueError, 'closed'):
        manager.execute(request, lambda *a, **k: None, lambda: False)
      self.assertFalse(manager.workers)

  def test_phone_close_is_scoped_and_acknowledged(self):
    with tempfile.TemporaryDirectory() as folder:
      manager = IOSDevices(Path(folder))
      sent = []
      peer = SimpleNamespace(
          send=sent.append,
          receive=lambda _: dict(
              type='closed', requestId=sent[-1]['requestId']
          ),
      )
      manager.connections['phone'] = peer
      manager.sessions = {
          ('a', 'ref', 'phone'): {},
          ('a', 'target', 'phone'): {},
          ('b', 'ref', 'phone'): {},
      }
      manager.close_session('a')
      self.assertEqual(len(sent), 1)
      self.assertEqual(sent[0]['sessionId'], 'a')
      self.assertEqual(list(manager.sessions), [('b', 'ref', 'phone')])
