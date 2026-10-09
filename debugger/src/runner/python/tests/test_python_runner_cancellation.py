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

"""Cancellation distinguishes no dispatch from loss of the model worker."""

import pathlib
import tempfile
import unittest
from unittest import mock

from model_debugger_runner import python_workers


class PythonCancellationTests(unittest.TestCase):

  def test_stop_before_dispatch_retains_the_resident_process(self):
    with tempfile.TemporaryDirectory() as folder:
      workers = python_workers.PythonWorkers()
      process, log = mock.MagicMock(), mock.MagicMock()
      workers.workers[('session', 'ref')] = process, log
      request = dict(
          operation='generate',
          session_id='session',
          run=dict(id='ref', runtime='PyTorch'),
          output=str(pathlib.Path(folder) / 'turn'),
      )
      with self.assertRaises(InterruptedError):
        workers.execute(request, lambda *a, **k: None, lambda: True)
      process.stdin.write.assert_not_called()
      process.kill.assert_not_called()
      self.assertIs(workers.workers[('session', 'ref')][0], process)

  def test_unacknowledged_stop_is_worker_loss_not_a_clean_chat_stop(self):
    with tempfile.TemporaryDirectory() as folder:
      workers = python_workers.PythonWorkers()
      process, log = mock.MagicMock(), mock.MagicMock()
      process.poll.return_value = None
      workers.workers[('session', 'ref')] = process, log
      request = dict(
          operation='generate',
          session_id='session',
          run=dict(id='ref', runtime='PyTorch'),
          output=str(pathlib.Path(folder) / 'turn'),
      )
      # First check precedes dispatch; subsequent checks see Stop.
      stopped = iter((False, True, True))
      with (
          mock.patch.object(
              python_workers.time,
              'monotonic',
              autospec=True,
              spec_set=True,
              side_effect=(1, 2, 8),
          ),
          mock.patch.object(
              workers, '_close', autospec=True, spec_set=True
          ) as close,
      ):
        with self.assertRaisesRegex(ConnectionError, 'did not acknowledge'):
          workers.execute(request, lambda *a, **k: None, lambda: next(stopped))
        close.assert_called_once_with(('session', 'ref'))


if __name__ == '__main__':
  unittest.main()
