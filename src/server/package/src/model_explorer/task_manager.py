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
"""In-memory manager for asynchronous model conversion tasks.

A conversion task runs on a background worker thread. While it runs, it
publishes progress dictionaries to per-subscriber queues (consumed by the SSE
stream) and to a latest-progress snapshot (consumed by Colab polling). Callers
cancel a task by setting its cancellation flag: the task's progress callback
then returns `False`, which the C++ engine observes as a cancellation request.
Finished tasks are evicted after a TTL.
"""

import atexit
from collections.abc import Callable, Iterator, Mapping
from concurrent import futures
import dataclasses
import enum
import json
import logging
import queue
import threading
import time
import types
from typing import Any
import uuid

# Not the standard-library `types` imported above; pylint conflates the two.
from . import types as server_types  # pylint: disable=reimported

# Finished tasks (and their results) are kept for 10 minutes.
DEFAULT_TTL_SECONDS = 600.0
# At least the web app's `MAX_MODELS_COUNT`, which converts all models of its
# table at once, so no model waits behind another.
DEFAULT_MAX_WORKERS = 10
# How long an event stream may be idle before it sends a heartbeat. Strictly
# shorter than the web app's 1 s SSE watchdog, so a live stream never trips it.
DEFAULT_HEARTBEAT_SECONDS = 0.5

# A unit of conversion work. It receives the task's progress callback, which
# returns `False` once cancellation was requested, and returns the result.
TaskFn = Callable[[server_types.ProgressCallback], Any]

PROGRESS_EVENT = 'progress'
HEARTBEAT_EVENT = 'heartbeat'

# The diagnostic counts always present in published progress. Other counts
# reported by the C++ engine (e.g. `incompleteEdges`) go into `details`.
_DIAGNOSTIC_SUMMARY_KEYS = ('warningCount', 'quantMismatches', 'missingOpDefs')

# The progress a stream reports for a task that hasn't reported any yet.
_QUEUED_PROGRESS = types.MappingProxyType({
    'stage': 'queued',
    'stageId': -1,
    'currentStep': 0,
    'totalSteps': -1,
    'progressFraction': -1.0,
    'message': 'Queued',
    'elapsedTimeMs': 0,
})


class TaskStatus(str, enum.Enum):
  """The lifecycle status of a conversion task."""

  QUEUED = 'QUEUED'
  RUNNING = 'RUNNING'
  CANCELLING = 'CANCELLING'
  COMPLETED = 'COMPLETED'
  FAILED = 'FAILED'
  CANCELLED = 'CANCELLED'


TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

# SSE event name emitted when a task reaches each terminal status.
_TERMINAL_EVENT_NAMES: Mapping[TaskStatus, str] = types.MappingProxyType({
    TaskStatus.COMPLETED: 'complete',
    TaskStatus.FAILED: 'failed',
    TaskStatus.CANCELLED: 'cancelled',
})
_TERMINAL_EVENTS = frozenset(_TERMINAL_EVENT_NAMES.values())


@dataclasses.dataclass(frozen=True)
class TaskEvent:
  """A task event delivered to stream subscribers.

  Attributes:
    name: `progress`, `heartbeat`, or the terminal event `complete`, `failed` or
      `cancelled`.
    data: The JSON-serializable event payload. It contains `taskId`, except for
      heartbeats, whose payload is empty.
  """

  name: str
  data: dict[str, Any]

  @property
  def is_terminal(self) -> bool:
    """Whether this event ends the stream: `complete`, `failed` or `cancelled`."""
    return self.name in _TERMINAL_EVENTS


@dataclasses.dataclass(frozen=True)
class TaskSnapshot:
  """A point-in-time copy of a task's state.

  Attributes:
    task_id: Unique task identifier.
    status: Current lifecycle status of the task.
    progress: Latest progress dictionary, or None before the first report.
    error: Error message if the task failed, or None.
    result: Conversion result once the task completes, or None.
  """

  task_id: str
  status: TaskStatus
  progress: server_types.ProgressReport | None = None
  error: str | None = None
  result: Any = None

  def to_status_json(self) -> dict[str, Any]:
    """Returns the polling (`get_task_status`) response payload."""
    resp: dict[str, Any] = {
        'taskId': self.task_id,
        'status': self.status.value,
        'progress': self.progress,
    }
    if self.error is not None:
      resp['error'] = self.error
    return resp


def _parse_diagnostics(payload_json: Any) -> dict[str, int] | None:
  """Returns the diagnostic counts of a C++ report's `payloadJson`, if any."""
  if not isinstance(payload_json, str) or not payload_json:
    return None
  try:
    payload = json.loads(payload_json)
  except ValueError:
    logging.warning('Ignoring malformed progress payload: %r', payload_json)
    return None
  diagnostics = (
      payload.get('diagnostics') if isinstance(payload, dict) else None
  )
  if not isinstance(diagnostics, dict):
    return None
  return {
      key: count
      for key, count in diagnostics.items()
      if isinstance(count, int) and not isinstance(count, bool)
  }


def _sum_diagnostics(
    diagnostics_by_subgraph: Mapping[str, Mapping[str, int]],
) -> server_types.DiagnosticsSummary:
  """Sums per-subgraph diagnostic counts into the published summary.

  Args:
    diagnostics_by_subgraph: The latest diagnostic counts of each subgraph.

  Returns:
    `{warningCount, quantMismatches, missingOpDefs}`, plus `details` with the
    other non-zero counts if there are any.
  """
  totals: dict[str, int] = {}
  for diagnostics in diagnostics_by_subgraph.values():
    for key, count in diagnostics.items():
      totals[key] = totals.get(key, 0) + count
  summary: server_types.DiagnosticsSummary = {
      key: totals.pop(key, 0) for key in _DIAGNOSTIC_SUMMARY_KEYS
  }
  details = {key: count for key, count in totals.items() if count > 0}
  if details:
    summary['details'] = details
  return summary


class _ProgressTracker:
  """Normalizes C++ progress reports and aggregates per-subgraph diagnostics.

  Adds `currentOpLabel` (the wire name of `currentOp`) and `diagnostics`:
  the latest diagnostic counts of each subgraph, summed over subgraphs. The
  engine reports a subgraph's diagnostics only once, so carrying the totals on
  every update lets late subscribers and polling clients see them.

  Attributes:
    latest: The latest normalized progress, or None before the first report.
  """

  def __init__(self) -> None:
    self.latest: server_types.ProgressReport | None = None
    self._diagnostics_by_subgraph: dict[str, dict[str, int]] = {}

  def record(self, report: Mapping[str, Any]) -> server_types.ProgressReport:
    """Normalizes `report`, updates subgraph diagnostics, and stores snapshot."""
    progress: server_types.ProgressReport = dict(report)
    if 'currentOp' in progress:
      progress['currentOpLabel'] = progress['currentOp']
    diagnostics = _parse_diagnostics(progress.get('payloadJson'))
    if diagnostics is not None:
      self._diagnostics_by_subgraph[progress.get('subgraphName') or ''] = (
          diagnostics
      )
    progress['diagnostics'] = _sum_diagnostics(self._diagnostics_by_subgraph)
    self.latest = progress
    return progress

  def current_or_queued(self) -> server_types.ProgressReport:
    """Returns the latest normalized progress, or a queued placeholder."""
    if self.latest is not None:
      return self.latest
    return {
        **_QUEUED_PROGRESS,
        'diagnostics': _sum_diagnostics(self._diagnostics_by_subgraph),
    }


class _Task:
  """Mutable task state. Guarded by `TaskManager._lock`.

  Attributes:
    task_id: Unique task identifier.
    status: Current lifecycle status.
    progress_tracker: Normalizes and stores the task's progress reports.
    error: Error message once the task failed, else None.
    result: The work's result once the task completed, else None.
    finished_at: Clock time at which the task reached a terminal status.
    future: The executor future running the task.
    subscribers: Queues of the open event streams.
    on_finish: Cleanup to run once; cleared when taken.
    cancel_requested: Set once cancellation was requested. Read lock-free by the
      progress callback on the worker thread.
  """

  def __init__(self, task_id: str, on_finish: Callable[[], None] | None):
    self.task_id = task_id
    self.status = TaskStatus.QUEUED
    self.progress_tracker = _ProgressTracker()
    self.error: str | None = None
    self.result: Any = None
    self.finished_at: float | None = None
    self.future: futures.Future[None] | None = None
    self.subscribers: list[queue.Queue[TaskEvent]] = []
    self.on_finish = on_finish
    self.cancel_requested = threading.Event()

  def snapshot(self) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=self.task_id,
        status=self.status,
        progress=self.progress_tracker.latest,
        error=self.error,
        result=self.result,
    )

  def take_on_finish(self) -> Callable[[], None] | None:
    on_finish, self.on_finish = self.on_finish, None
    return on_finish

  def progress_event(self) -> TaskEvent:
    """Returns the latest progress, or a queued placeholder, as an event."""
    return TaskEvent(
        PROGRESS_EVENT,
        {'taskId': self.task_id, **self.progress_tracker.current_or_queued()},
    )

  def terminal_event(self) -> TaskEvent:
    data: dict[str, Any] = {'taskId': self.task_id, 'status': self.status.value}
    if self.error is not None:
      data['error'] = self.error
    return TaskEvent(_TERMINAL_EVENT_NAMES[self.status], data)


def _run_on_finish(on_finish: Callable[[], None] | None) -> None:
  if on_finish is None:
    return
  try:
    on_finish()
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception('Conversion task cleanup failed')


class TaskManager:
  """Runs conversion tasks on a background thread pool.

  All methods are thread-safe.
  """

  def __init__(
      self,
      max_workers: int = DEFAULT_MAX_WORKERS,
      ttl_seconds: float = DEFAULT_TTL_SECONDS,
      clock: Callable[[], float] = time.monotonic,
  ):
    """Initializes the task manager.

    Args:
      max_workers: Maximum number of conversion tasks to execute concurrently.
      ttl_seconds: Seconds to retain finished tasks before eviction.
      clock: Monotonic clock function returning current time in seconds.
    """
    self._executor = futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix='model_explorer_task'
    )
    self._ttl_seconds = ttl_seconds
    self._clock = clock
    self._lock = threading.Lock()
    self._tasks: dict[str, _Task] = {}

  def submit(
      self, fn: TaskFn, on_finish: Callable[[], None] | None = None
  ) -> dict[str, str]:
    """Queues `fn` for background execution.

    Args:
      fn: The conversion work. See `TaskFn`.
      on_finish: Optional cleanup, called exactly once when the task finishes:
        after `fn` returns or raises (before the task reports its terminal
        status), or when the task is cancelled before it starts. Exceptions it
        raises are logged and don't change the task's outcome.

    Returns:
      The `start_convert` response payload: `{"taskId": ..., "status":
      "QUEUED"}`.

    Raises:
      RuntimeError: If the manager was shut down. `on_finish` has already run.
    """
    task = _Task(task_id=str(uuid.uuid4()), on_finish=on_finish)
    try:
      with self._lock:
        self._evict_expired_locked()
        # Submitted under the lock so `cancel` always sees `task.future`, and
        # before the task is registered so a rejected task is never visible.
        task.future = self._executor.submit(self._run, task, fn)
        self._tasks[task.task_id] = task
    except RuntimeError:
      _run_on_finish(task.take_on_finish())
      raise
    return {'taskId': task.task_id, 'status': TaskStatus.QUEUED.value}

  def get(self, task_id: str) -> TaskSnapshot | None:
    """Returns a snapshot of the task, or None if it is unknown or evicted."""
    with self._lock:
      self._evict_expired_locked()
      task = self._tasks.get(task_id)
      return task.snapshot() if task is not None else None

  def cancel(self, task_id: str) -> dict[str, str] | None:
    """Requests cooperative cancellation of a task.

    A queued task is cancelled immediately. A running task moves to
    `CANCELLING`; its progress callback returns `False` from then on, and the
    task becomes `CANCELLED` when its work returns. Cancellation wins even if
    the work ignores the callback and returns a result: the result is dropped,
    because the caller has already stopped waiting for it. Finished tasks are
    left unchanged.

    Args:
      task_id: The task to cancel.

    Returns:
      The `cancel_task` response payload with the task's resulting status, or
      None if the task is unknown or evicted.
    """
    on_finish = None
    with self._lock:
      self._evict_expired_locked()
      task = self._tasks.get(task_id)
      if task is None:
        return None
      if task.status not in TERMINAL_STATUSES:
        task.cancel_requested.set()
        if task.future is not None and task.future.cancel():
          # The task never runs, so its cleanup runs here.
          on_finish = task.take_on_finish()
          self._finish_locked(task, TaskStatus.CANCELLED)
        else:
          task.status = TaskStatus.CANCELLING
      resp = {'taskId': task.task_id, 'status': task.status.value}
    _run_on_finish(on_finish)
    return resp

  def open_event_stream(
      self, task_id: str, heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
  ) -> Iterator[TaskEvent] | None:
    """Subscribes to a task's events.

    The subscription is registered before this method returns, so no event is
    missed. The stream first yields a progress snapshot (the latest progress,
    or a `queued` placeholder) so that clients observe a named event as soon as
    they connect, then yields live events, and ends after the terminal event.

    Args:
      task_id: The task to subscribe to.
      heartbeat_seconds: The stream yields a `heartbeat` event after this long
        without an event, so the caller can keep its connection (and its
        client's liveness watchdog) alive.

    Returns:
      The event iterator, or None if the task is unknown or evicted.
    """
    events: queue.Queue[TaskEvent] = queue.Queue()
    with self._lock:
      self._evict_expired_locked()
      task = self._tasks.get(task_id)
      if task is None:
        return None
      events.put(task.progress_event())
      if task.status in TERMINAL_STATUSES:
        events.put(task.terminal_event())
      else:
        task.subscribers.append(events)
    return self._iter_events(task, events, heartbeat_seconds)

  def shutdown(self) -> None:
    """Cancels all unfinished tasks and stops the worker threads."""
    with self._lock:
      task_ids = list(self._tasks)
    for task_id in task_ids:
      self.cancel(task_id)
    self._executor.shutdown(wait=False)

  def shutdown_at_exit(self) -> None:
    """Calls `shutdown` when the interpreter exits.

    The hook is registered with `threading`'s exit hooks when available: those
    run before `concurrent.futures` joins its worker threads, whereas `atexit`
    callbacks run only after running conversions have finished.
    """
    register = getattr(threading, '_register_atexit', None)
    if register is None:
      atexit.register(self.shutdown)
      return
    register(self.shutdown)

  def _iter_events(
      self,
      task: _Task,
      events: queue.Queue[TaskEvent],
      heartbeat_seconds: float,
  ) -> Iterator[TaskEvent]:
    """Yields `events`, plus heartbeats while idle; unsubscribes on exit."""
    try:
      while True:
        try:
          event = events.get(timeout=heartbeat_seconds)
        except queue.Empty:
          yield TaskEvent(HEARTBEAT_EVENT, {})
          continue
        yield event
        if event.is_terminal:
          return
    finally:
      # Runs on normal completion and when the client disconnects.
      with self._lock:
        if events in task.subscribers:
          task.subscribers.remove(events)

  def _run(self, task: _Task, fn: TaskFn) -> None:
    """Executes a task on a worker thread."""
    with self._lock:
      cancelled_before_start = task.cancel_requested.is_set()
      if not cancelled_before_start:
        task.status = TaskStatus.RUNNING
    if cancelled_before_start:
      self._finish_after_cleanup(task)
      return

    def on_progress(progress: dict[str, Any]) -> bool:
      self._publish_progress(task, progress)
      return not task.cancel_requested.is_set()

    try:
      result = fn(on_progress)
    except BaseException as err:  # pylint: disable=broad-exception-caught
      # Catches non-`Exception`s too (e.g. `KeyboardInterrupt`): otherwise the
      # task would stay `RUNNING`, skip its cleanup, and never end its streams.
      self._finish_after_cleanup(task, error=f'{type(err).__name__}: {err}')
      return
    self._finish_after_cleanup(task, result=result)

  def _finish_after_cleanup(
      self, task: _Task, result: Any = None, error: str | None = None
  ) -> None:
    """Runs the task's cleanup, then finishes the task on the worker thread.

    The task ends `CANCELLED` if cancellation was requested, else `FAILED` if
    `error` is set, else `COMPLETED` with `result`.

    Args:
      task: The task whose work has returned (or never started).
      result: The work's result.
      error: The work's error message, if it raised.
    """
    with self._lock:
      on_finish = task.take_on_finish()
    _run_on_finish(on_finish)
    with self._lock:
      if task.cancel_requested.is_set():
        self._finish_locked(task, TaskStatus.CANCELLED)
      elif error is not None:
        self._finish_locked(task, TaskStatus.FAILED, error=error)
      else:
        self._finish_locked(task, TaskStatus.COMPLETED, result=result)

  def _publish_progress(self, task: _Task, report: Mapping[str, Any]) -> None:
    """Normalizes and publishes a C++ progress report to subscribers."""
    with self._lock:
      progress = task.progress_tracker.record(report)
      event = TaskEvent(PROGRESS_EVENT, {'taskId': task.task_id, **progress})
      for subscriber in task.subscribers:
        subscriber.put(event)

  def _finish_locked(
      self,
      task: _Task,
      status: TaskStatus,
      result: Any = None,
      error: str | None = None,
  ) -> None:
    """Records `task`'s terminal state and notifies its subscribers."""
    task.status = status
    task.result = result
    task.error = error
    task.finished_at = self._clock()
    event = task.terminal_event()
    for subscriber in task.subscribers:
      subscriber.put(event)
    task.subscribers.clear()

  def _evict_expired_locked(self) -> None:
    now = self._clock()
    expired = [
        task_id
        for task_id, task in self._tasks.items()
        if task.finished_at is not None
        and now - task.finished_at >= self._ttl_seconds
    ]
    for task_id in expired:
      del self._tasks[task_id]
