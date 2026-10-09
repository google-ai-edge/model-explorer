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

"""Runner-owned resident Python workers.

Never instantiated by the HTTP server.
"""

from collections.abc import Callable, Mapping
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Any, BinaryIO
import model_debugger_contracts
from model_debugger_contracts import errors as contract_errors


def atomic_json(path: pathlib.Path, value: Any) -> None:
  """Atomically writes a JSON-serializable value to path via a temp file."""
  temporary = path.with_suffix('.tmp')
  temporary.write_text(json.dumps(value, allow_nan=False))
  os.replace(temporary, path)


class PythonWorkers:
  """Manages resident PyTorch worker subprocesses keyed by session and role."""

  def __init__(self) -> None:
    """Initializes an empty worker process registry."""
    self.workers: dict[
        tuple[str, str], tuple[subprocess.Popen[str], BinaryIO]
    ] = {}

  def reset_session(self, identity: str | None = None) -> None:
    """Resets conversation state on matching resident session workers."""
    for session, role in list(self.workers):
      if identity is not None and session != identity:
        continue
      with tempfile.TemporaryDirectory(prefix='runner-chat-reset-') as folder:
        self.execute(
            dict(
                operation='reset',
                session_id=session,
                run=dict(id=role, runtime='PyTorch'),
                output=str(pathlib.Path(folder) / 'reset'),
            ),
            lambda *args, **kwargs: None,
            lambda: False,
        )

  def _close(self, key: tuple[str, str]) -> None:
    """Terminates and cleans up a single resident worker subprocess."""
    worker = self.workers.pop(key, None)
    if worker is None:
      return
    process, log = worker
    try:
      if process.poll() is None:
        try:
          if process.stdin is not None:
            process.stdin.write(json.dumps({'operation': 'close'}) + '\n')
            process.stdin.flush()
            process.stdin.close()
          process.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
          process.kill()
          process.wait(timeout=3)
    finally:
      log.close()

  def close(self) -> None:
    """Closes all active resident worker subprocesses."""
    for key in list(self.workers):
      self._close(key)

  def execute(
      self,
      request: Mapping[str, Any],
      emit: Callable[..., None],
      cancelled: Callable[[], bool],
  ) -> dict[str, Any]:
    """Dispatches a request to a resident worker and streams its events."""
    key = request['session_id'], request['run']['id']
    directory = pathlib.Path(request['output'])
    request_path = directory.parent / (directory.name + '-request.json')
    atomic_json(request_path, dict(request))
    if request['operation'] == 'initialize' and key not in self.workers:
      log = (directory.parent / (directory.name + '.log')).open('ab')
      try:
        if request['run'].get('runtime') != 'PyTorch':
          raise ValueError('Python Runner workers support PyTorch only')
        root = pathlib.Path(request['pytorch_root'])
        package_roots = (
            str(pathlib.Path(__file__).resolve().parents[1]),
            str(
                pathlib.Path(model_debugger_contracts.__file__)
                .resolve()
                .parents[1]
            ),
            str(root),
        )
        process = subprocess.Popen(
            [
                str(root / '.venv/bin/python'),
                '-m',
                'model_debugger_runner.pytorch_worker',
                '--serve',
            ],
            stdin=subprocess.PIPE,
            text=True,
            stdout=log,
            stderr=log,
            env={
                **os.environ,
                'PYTHONUNBUFFERED': '1',
                'PYTHONPATH': os.pathsep.join(dict.fromkeys(package_roots)),
            },
        )
      except BaseException:
        log.close()
        raise
      self.workers[key] = process, log
    if key not in self.workers:
      raise ValueError('Session worker is closed. Initialize again.')
    process, _ = self.workers[key]
    retain_worker = False
    try:
      if cancelled():
        # Nothing was dispatched. Normal Chat reset can safely follow;
        # do not kill a resident model just because Stop won admission.
        retain_worker = True
        raise InterruptedError('Cancelled')
      if process.stdin is None:
        raise RuntimeError('Session worker stdin is closed')
      process.stdin.write(json.dumps({'request': str(request_path)}) + '\n')
      process.stdin.flush()
      cursor = 0
      began = time.monotonic()
      cancel_at = None
      while True:
        event_path = directory / 'events.jsonl'
        if event_path.exists():
          complete = [
              line
              for line in event_path.read_text().splitlines(keepends=True)
              if line.endswith('\n')
          ]
          for line in complete[cursor:]:
            event = json.loads(line)
            kind = event.pop('type')
            emit(kind, **event)
            if kind == 'error':
              retain_worker = True
              if event.get('errorCode') == contract_errors.InputRejected.code:
                raise contract_errors.InputRejected(event['error'])
              if event.get('errorCode') == 'stopped':
                raise InterruptedError('Cancelled; Chat ended')
              raise RuntimeError(event['error'])
            if kind == 'finished':
              retain_worker = True
              return json.loads((directory / 'result.json').read_text())
          cursor = len(complete)
        if process.poll() is not None:
          raise RuntimeError(
              'Session worker exited. Initialize again; see the initialization'
              ' log.'
          )
        if cancelled():
          (directory.parent / 'cancel').touch()
          cancel_at = cancel_at or time.monotonic()
          if time.monotonic() - cancel_at > 5:
            raise ConnectionError(
                'Python worker did not acknowledge cancellation; execution is'
                ' unavailable'
            )
        if time.monotonic() - began > 600:
          raise TimeoutError('Runtime exceeded the 10 minute task limit')
        if shutil.disk_usage(directory.parent).free < 256 * 1024**2:
          raise ValueError('Capture stopped because disk space is low')
        time.sleep(0.08)
    except BaseException:
      if not retain_worker:
        self._close(key)
      raise
