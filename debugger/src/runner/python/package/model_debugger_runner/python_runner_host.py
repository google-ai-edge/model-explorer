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

"""Runner-owned PyTorch subprocess supervisor; JSON lines are private App IPC.

No HTTP listener, no server-owned model process. Executable and package roots
come from the Runner's local configuration, never from a network request.
"""

import argparse
from collections.abc import Callable, Mapping
import json
import pathlib
import sys
import threading
from typing import Any
import uuid
from model_debugger_contracts import errors as contract_errors
from model_debugger_runner import python_workers


class PythonRunnerHost:
  """Supervises PyTorch runner requests over private JSON-line IPC."""

  def __init__(
      self,
      root: str | pathlib.Path,
      emit: Callable[[dict[str, Any]], None],
      workers: python_workers.PythonWorkers | None = None,
  ) -> None:
    """Initializes the host supervisor for a configured PyTorch runtime root."""
    self.root = pathlib.Path(root).resolve()
    self.emit = emit
    self.workers = workers or python_workers.PythonWorkers()
    self.lock = threading.RLock()
    self.active: tuple[str, threading.Event, dict[str, Any]] | None = None
    self.thread: threading.Thread | None = None
    self.seen: set[str] = set()
    self.closed = False

  def accept(self, command: Mapping[str, Any]) -> None:
    """Validates and dispatches a runner IPC command."""
    identity = str(uuid.UUID(command['requestId']))
    kind = command['type']
    with self.lock:
      if kind == 'cancel':
        if self.active is None or self.active[0] != identity:
          raise ValueError('No matching active Python request')
        self.active[1].set()
        (pathlib.Path(self.active[2]['output']).parent / 'cancel').touch()
        return
      if self.closed or self.active is not None:
        raise ValueError('Python Runner is busy or closed')
      if identity in self.seen:
        raise ValueError('Request already executed; do not replay it')
      self.seen.add(identity)
      if kind == 'reset':
        request: dict[str, Any] = {'operation': 'reset'}
      elif kind in ('initialize', 'generate', 'preflight'):
        request = dict(command['request'])
        if (
            request.get('operation') != kind
            or request['run'].get('runtime') != 'PyTorch'
        ):
          raise ValueError('This adapter accepts PyTorch requests only')
        request['session_id'] = str(uuid.UUID(request['session_id']))
        if request['run'].get('id') not in ('ref', 'target'):
          raise ValueError('Invalid Session role')
        # Interpreter choice cannot be overridden by the remote request.
        request['pytorch_root'] = str(self.root)
        if kind in ('generate', 'preflight'):
          request.pop('messages', None)
        output = pathlib.Path(request['output'])
        if (
            not output.is_absolute()
            or output.exists()
            or not output.parent.is_dir()
        ):
          raise ValueError('A fresh local capture directory is required')
      else:
        raise ValueError('Unsupported Python Runner command')
      cancel = threading.Event()
      self.active = identity, cancel, request
      self.thread = threading.Thread(
          target=self._execute, args=(identity, request, cancel)
      )
      self.thread.start()

  def _execute(
      self,
      identity: str,
      request: dict[str, Any],
      cancel: threading.Event,
  ) -> None:
    """Executes an accepted request on the worker pool and emits completion."""
    result = None
    error = None
    status = None
    try:
      if request['operation'] == 'reset':
        self.workers.reset_session()
      else:
        result = self.workers.execute(
            request,
            lambda kind, **data: self.emit(
                dict(
                    type='event',
                    requestId=identity,
                    event=dict(type=kind, **data),
                )
            ),
            cancel.is_set,
        )
    except BaseException as exc:
      error = str(exc)
      status = (
          'rejected'
          if isinstance(exc, contract_errors.InputRejected)
          else 'stopped'
          if isinstance(exc, InterruptedError)
          else 'failed'
      )
    with self.lock:
      self.active = None
      self.emit(
          dict(
              type='error',
              requestId=identity,
              error=error,
              generationStatus=status,
              errorCode=(
                  contract_errors.InputRejected.code
                  if status == 'rejected'
                  else status
              ),
          )
          if error
          else dict(
              type={'reset': 'reset', 'preflight': 'preflight'}.get(
                  request['operation'], 'completed'
              ),
              requestId=identity,
              generationStatus='succeeded',
              result=result,
              output=(result or {}).get('output', ''),
              summary={
                  key: result.get(key)
                  for key in (
                      'processed_token_count',
                      'captured_tensors',
                      'effective',
                      'runtime_instance',
                      'worker_pid',
                      'model_load_count',
                  )
              }
              if result
              else None,
          )
      )

  def close(self) -> None:
    """Cancels any active request and shuts down all managed workers."""
    with self.lock:
      self.closed = True
      if self.active and self.active[2]['operation'] != 'reset':
        self.active[1].set()
        (pathlib.Path(self.active[2]['output']).parent / 'cancel').touch()
      thread = self.thread
    if thread:
      thread.join(timeout=8)
    self.workers.close()


def main() -> None:
  """Runs the stdin/stdout JSON-line IPC loop for the Python Runner host."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=pathlib.Path, required=True)
  args = parser.parse_args()
  output_lock = threading.Lock()

  def emit(value: dict[str, Any]) -> None:
    with output_lock:
      print(json.dumps(value, allow_nan=False), flush=True)

  host = PythonRunnerHost(args.root, emit)
  try:
    for line in sys.stdin:
      command: dict[str, Any] = {}
      try:
        command = json.loads(line)
        host.accept(command)
      except Exception as error:  # pylint: disable=broad-exception-caught
        emit(
            dict(
                type='error',
                requestId=command.get('requestId', ''),
                error=str(error),
            )
        )
  finally:
    host.close()


if __name__ == '__main__':
  main()
