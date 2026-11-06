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

A job is an initialize, a chat (a chat's start), a generate, or a download;
an initialize of a Session that observes tensors prepares its copies first
(prepared_models). A chat and a generate name their chat (`chat`). A
download (hf_models) fetches a model from Hugging Face for the editor, of no
Session (its session_id ''): its 'model' event says the model it registered
(the job's `model`). It lives in memory only: a job cannot outlive the runners
it drives, and those end with the server; a start's copy of a model is kept
once done. Every change to a job is an event with the next sequence
number; the UI reads the job once and follows its events over SSE from that
sequence on. A generate's reply comes as 'delta' events, each a piece of one
side's reply on one channel, as the runner streams it: 'content', the reply, or
'reasoning', the model's thinking; the job holds each channel's pieces so far,
joined (`output` and `reasoning`, by side). A delta keeps no text of its own,
only where its piece is in what the job joined, so a job of a long reply holds
its text once, however many events it took. What a job's status means to the
reader, its `note` and `status_text`, is derived from the status as the job or
its status event is sent, not kept.

Events are emitted from worker threads and read by `events()` on the event
loop: an emit wakes each reader through its loop's call_soon_threadsafe.
"""

import array
import asyncio
from collections.abc import AsyncIterator
import copy
import threading
import types
from typing import Any
import uuid

from model_debugger.server.util import clock
from model_debugger.server.util import errors

TERMINAL = ('completed', 'failed', 'cancelled')
# The note of a cancelled job, shown with the prompt and partial output of its
# turn: only a generate is ever cancelled, and Stop drops its turn.
STOPPED_TURN = 'This turn has no debug data: a stopped turn is not captured.'
# What a 'delta' event's channel adds to, in the job.
_CHANNELS = types.MappingProxyType(
    {'content': 'output', 'reasoning': 'reasoning'}
)
# The status text of a generate, by status; '' for any other.
_GENERATE_STATUS_TEXT = types.MappingProxyType({
    'cancelled': 'Generation stopped',
    'failed': 'Generation failed',
})
# What brief() answers of a job.
_BRIEF = ('id', 'operation', 'status', 'error', 'progress', 'created_at')
_OPERATIONS = types.MappingProxyType({
    'initialize': 'initializing',
    'chat': 'starting',
    'generate': 'running',
    'download': 'downloading',
})


class _Events:
  """A job's events, the event of sequence n at index n - 1.

  A 'delta' is kept as where its text is: its stream (side and channel)
  and the span of its text in what the job joined for that stream.

  Attributes:
      count: How many events there are.
  """

  def __init__(self) -> None:
    """Holds no events."""
    self.count = 0
    self._others: dict[int, dict[str, Any]] = (
        {}
    )  # Any but a delta, by sequence.
    self._streams: list[tuple[str, str]] = []  # (runId, channel).
    # By index: the stream of a delta, or -1; and its text's span.
    self._stream = array.array('b')
    self._starts = array.array('q')
    self._ends = array.array('q')

  def add(self, event: dict[str, Any], said: str) -> None:
    """Adds the next event; for a delta, `said` is its stream's text."""
    self.count += 1
    if event['type'] != 'delta':
      self._others[self.count] = event
      self._stream.append(-1)
      self._starts.append(0)
      self._ends.append(0)
      return
    stream = (event['runId'], event['channel'])
    if stream not in self._streams:
      self._streams.append(stream)
    self._stream.append(self._streams.index(stream))
    self._starts.append(len(said) - len(event['text']))
    self._ends.append(len(said))

  def after(self, sequence: int, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns copies of the events after `sequence`, of the job `job`."""
    events = []
    for index in range(max(sequence, 0), self.count):
      stream = self._stream[index]
      if stream < 0:
        event = copy.deepcopy(self._others[index + 1])
        if event['type'] == 'status':
          event = _sent(event, job['operation'])
        events.append(event)
        continue
      run, channel = self._streams[stream]
      said = job[_CHANNELS[channel]][run]
      events.append({
          'sequence': index + 1,
          'jobId': job['id'],
          'sessionId': job['session_id'],
          'turnId': job['turn'],
          'type': 'delta',
          'runId': run,
          'channel': channel,
          'text': said[self._starts[index] : self._ends[index]],
      })
    return events


class Jobs:
  """Every job of this server's life, with their events."""

  def __init__(self) -> None:
    """Starts with no jobs."""
    self._lock = threading.Lock()
    self._jobs: dict[str, dict[str, Any]] = {}
    self._events: dict[str, _Events] = {}
    # job id -> run id -> the message of the run's latest progress event.
    self._progress: dict[str, dict[str, str]] = {}
    # job id -> the (loop, event) of each reader waiting for its events.
    self._waiters: dict[
        str, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]
    ] = {}

  def create(
      self,
      session_id: str,
      operation: str,
      turn: int | None = None,
      prompt: str = '',
      chat: int | None = None,
  ) -> dict[str, Any]:
    """Creates a job.

    Args:
        session_id: The record the job works on.
        operation: 'initialize', 'chat', 'generate' or 'download'.
        turn: The turn a generate makes.
        prompt: What the turn of a generate asks; a reader of the job, such as a
          page loaded again, shows it with the turn's output.
        chat: The chat a chat job starts, by the number it gets, or a generate's
          turn is of.

    Returns:
        The job.
    """
    status = _OPERATIONS[operation]
    with self._lock:
      job = {
          'id': str(uuid.uuid4()),
          'session_id': session_id,
          'operation': operation,
          'status': status,
          'error': '',
          'sequence': 0,
          'output': {'ref': '', 'target': ''},
          'reasoning': {'ref': '', 'target': ''},
          'chat': chat,
          'turn': turn,
          'prompt': prompt,
          'progress': '',
          'created_at': clock.now(),
      }
      self._jobs[job['id']] = job
      self._events[job['id']] = _Events()
      self._progress[job['id']] = {}
      self._waiters[job['id']] = set()
      return _sent(copy.deepcopy(job), operation)

  def get(self, job_id: str) -> dict[str, Any]:
    """Returns a copy of a job.

    Args:
      job_id: The job identifier.

    Raises:
      NotFound: No such job.
    """
    with self._lock:
      job = self._job(job_id)
      return _sent(copy.deepcopy(job), job['operation'])

  def brief(self, job_id: str) -> dict[str, Any]:
    """What a listing reads of a job, without copying its outputs.

    Args:
      job_id: The job identifier.

    Returns:
      Its id, operation, status, error, progress and created_at.

    Raises:
      NotFound: No such job.
    """
    with self._lock:
      job = self._job(job_id)
      return {name: job[name] for name in _BRIEF}

  def progress(self, job_id: str) -> dict[str, str]:
    """Returns the latest progress message of each run of a job.

    Args:
      job_id: The job identifier.

    Raises:
      NotFound: No such job.
    """
    with self._lock:
      self._job(job_id)
      return dict(self._progress[job_id])

  def emit(  # pylint: disable=redefined-builtin
      self, job_id: str, type: str, **fields: Any
  ) -> None:
    """Records an event of a job and wakes its readers.

    The event is `{sequence, jobId, sessionId, turnId, type, **fields}`. A
    'delta', whose fields are `runId`, `channel` and `text` alone,
    appends `text` to what `runId` said on its `channel` (the job's
    `output` for 'content', its `reasoning` for 'reasoning'), a
    'progress' sets the job's progress, and that of its `runId` if any,
    to `message`, a 'model' sets its `model`, a 'status' sets its
    `status` and `error`.

    Args:
      job_id: The job.
      type: The event's type.
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
          'type': type,
          **fields,
      }
      said = ''
      if type == 'delta':
        by_side = job[_CHANNELS[fields['channel']]]
        by_side[fields['runId']] += fields['text']
        said = by_side[fields['runId']]
      elif type == 'progress':
        job['progress'] = fields['message']
        if 'runId' in fields:
          self._progress[job_id][fields['runId']] = fields['message']
      elif type == 'model':
        job['model'] = fields['model']
      elif type == 'status':
        job['status'] = fields['status']
        job['error'] = fields.get('error', '')
        for key in ('does_not_fit', 'failed_sides'):
          if key in fields:
            job[key] = fields[key]
      self._events[job_id].add(event, said)
      waiters = list(self._waiters[job_id])
    for loop, wake in waiters:
      try:
        loop.call_soon_threadsafe(wake.set)
      except RuntimeError:
        pass  # The reader's loop is closed: no one is left to wake.

  def finish(
      self,
      job_id: str,
      status: str,
      error: str = '',
      does_not_fit: dict[str, dict[str, int]] | None = None,
      failed_sides: list[str] | None = None,
  ) -> None:
    """Ends a job with a terminal status.

    Args:
        job_id: The job.
        status: One of TERMINAL.
        error: Why it failed or was cancelled; '' when it completed.
        does_not_fit: Of a generate whose turn both sides refused for want of
          room in their contexts, each side's `{tokens, left}`: the tokens the
          turn needs and those its context has left. The job and its status
          event then carry it as `does_not_fit`.
        failed_sides: Of a failed generate, the sides that failed, when known;
          the job and its status event then carry them as `failed_sides`, and
          only those sides show its status text.
    """
    if status not in TERMINAL:
      raise ValueError(f'{status} is not a terminal status.')
    fields = {} if does_not_fit is None else {'does_not_fit': does_not_fit}
    if failed_sides:
      fields['failed_sides'] = list(failed_sides)
    self.emit(job_id, 'status', status=status, error=error, **fields)

  async def events(
      self, job_id: str, after: int
  ) -> AsyncIterator[dict[str, Any]]:
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
          # Cleared while no emit can run: an emit after this read
          # sets it.
          wake.clear()
          job = self._jobs[job_id]
          pending = self._events[job_id].after(after, job)
          ended = job['status'] in TERMINAL
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

  def _job(self, job_id: str) -> dict[str, Any]:
    job = self._jobs.get(job_id)
    if job is None:
      raise errors.NotFound('This task does not exist.')
    return job


def _sent(item: dict[str, Any], operation: str) -> dict[str, Any]:
  """Adds what a status means to a copy of a job or status event sent out.

  Args:
      item: The copy; it has a `status`.
      operation: The job's operation.

  Returns:
      The copy, with `note`, what the end means for the job's turn, and
      `status_text`, the label of its status; each '' for none.
  """
  status = item['status']
  item['note'] = (
      STOPPED_TURN if status == 'cancelled' and operation == 'generate' else ''
  )
  item['status_text'] = (
      _GENERATE_STATUS_TEXT.get(status, '') if operation == 'generate' else ''
  )
  return item
