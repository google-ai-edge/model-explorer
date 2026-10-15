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

"""Jobs: the long operations the UI starts and follows.

A job is an initialize, a new_chat or a generate. It lives in memory only: a
job cannot outlive the runners it drives, and those end with the server.
Every change to a job is an event with the next sequence number; the UI reads
the job once and follows its events over SSE from that sequence on.

Events are emitted from worker threads and read by `events()` on the event
loop: an emit wakes each reader through its loop's call_soon_threadsafe.
"""

import asyncio
from collections.abc import AsyncIterator
import copy
import datetime
import threading
import types
from typing import Any
import uuid

from model_debugger_server import errors

TERMINAL = ('completed', 'failed', 'cancelled')
_OPERATIONS = types.MappingProxyType({
    'initialize': 'initializing',
    'new_chat': 'initializing',
    'generate': 'running',
})


class Jobs:
  """Every job of this server's life, with their events."""

  def __init__(self) -> None:
    """Starts with no jobs."""
    self._lock = threading.Lock()
    self._jobs: dict[str, dict] = {}
    self._events: dict[str, list[dict]] = {}
    # (session_id, request_id) -> (job id, digest of the request).
    self._requests: dict[tuple[str, str], tuple[str, str]] = {}
    # job id -> the (loop, event) of each reader waiting for its events.
    self._waiters: dict[
        str, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]
    ] = {}

  def create(
      self,
      session_id: str,
      operation: str,
      request_id: str | None,
      digest: str,
      turn: int | None = None,
  ) -> tuple[dict, bool]:
    """Creates a job, once per request.

    Args:
      session_id: The record the job works on.
      operation: 'initialize', 'new_chat' or 'generate'.
      request_id: The UI's id of the request; a request sent again with it gets
        the job it created. None: every call is a new job.
      digest: What identifies the request's content.
      turn: The turn a generate makes.

    Returns:
      The job, and whether this call created it.

    Raises:
      ValueError: `request_id` was used for a different request.
    """
    status = _OPERATIONS[operation]
    with self._lock:
      if request_id is not None:
        known = self._requests.get((session_id, request_id))
        if known is not None:
          job_id, known_digest = known
          if known_digest != digest:
            raise ValueError(
                'This request id was already used for a different request.'
            )
          return copy.deepcopy(self._jobs[job_id]), False
      job = {
          'id': str(uuid.uuid4()),
          'session_id': session_id,
          'operation': operation,
          'status': status,
          'error': '',
          'sequence': 0,
          'output': {'ref': '', 'target': ''},
          'turn': turn,
          'progress': '',
          'created_at': (
              datetime.datetime.now(datetime.timezone.utc).isoformat(
                  timespec='seconds'
              )
          ),
      }
      self._jobs[job['id']] = job
      self._events[job['id']] = []
      self._waiters[job['id']] = set()
      if request_id is not None:
        self._requests[(session_id, request_id)] = (job['id'], digest)
      return copy.deepcopy(job), True

  def get(self, job_id: str) -> dict:
    """Returns a copy of a job.

    Raises:
      NotFound: No such job.
    """
    with self._lock:
      return copy.deepcopy(self._job(job_id))

  def emit(self, job_id: str, event_type: str, **fields: Any) -> None:
    """Records an event of a job and wakes its readers.

    The event is `{sequence, jobId, sessionId, turnId, type, **fields}`. A
    'delta' appends `text` to the output of `runId`, a 'progress' sets the
    job's progress to `message`, a 'status' sets its `status` and `error`.

    Args:
      job_id: The job.
      event_type: The event's type.
      **fields: The event's fields.
    """
    with self._lock:
      job = self._job(job_id)
      job['sequence'] += 1
      event = {
          'sequence': job['sequence'],
          'jobId': job['id'],
          'sessionId': job['session_id'],
          'turnId': job['turn'],
          'type': event_type,
          **fields,
      }
      if event_type == 'delta':
        job['output'][fields['runId']] += fields['text']
      elif event_type == 'progress':
        job['progress'] = fields['message']
      elif event_type == 'status':
        job['status'] = fields['status']
        job['error'] = fields.get('error', '')
      self._events[job_id].append(event)
      waiters = list(self._waiters[job_id])
    for loop, wake in waiters:
      try:
        loop.call_soon_threadsafe(wake.set)
      except RuntimeError:
        pass  # The reader's loop is closed: no one is left to wake.

  def finish(self, job_id: str, status: str, error: str = '') -> None:
    """Ends a job with a terminal status.

    Args:
      job_id: The job.
      status: One of TERMINAL.
      error: Why it failed or was cancelled; '' when it completed.
    """
    if status not in TERMINAL:
      raise ValueError(f'{status} is not a terminal status.')
    self.emit(job_id, 'status', status=status, error=error)

  async def events(self, job_id: str, after: int) -> AsyncIterator[dict]:
    """Yields a job's events after a sequence number, as they come.

    Ends once the job is terminal and every event was yielded.

    Args:
      job_id: The job.
      after: The sequence of the last event the reader has.

    Yields:
      Each event, in order.

    Raises:
      NotFound: No such job (on the first iteration).
    """
    wake = asyncio.Event()
    waiter = (asyncio.get_running_loop(), wake)
    with self._lock:
      self._job(job_id)
      self._waiters[job_id].add(waiter)
    try:
      while True:
        with self._lock:
          # Cleared while no emit can run: an emit after this read sets it.
          wake.clear()
          pending = [
              copy.deepcopy(event)
              for event in self._events[job_id]
              if event['sequence'] > after
          ]
          ended = self._jobs[job_id]['status'] in TERMINAL
        if pending:
          for event in pending:
            after = event['sequence']
            yield event
        elif ended:
          return
        else:
          await wake.wait()
    finally:
      with self._lock:
        self._waiters[job_id].discard(waiter)

  def _job(self, job_id: str) -> dict:
    job = self._jobs.get(job_id)
    if job is None:
      raise errors.NotFound('This task does not exist.')
    return job
