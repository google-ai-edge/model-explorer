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

"""Supervised, bounded analysis processes with capture-local caches."""

from collections.abc import Callable, Mapping
from concurrent import futures
import copy
import json
import multiprocessing
import queue
import sys
import threading
import time
import traceback
import types
from typing import Any

from model_explorer_debugger import errors
from model_explorer_debugger import kv_analysis
from model_explorer_debugger import kv_head
from model_explorer_debugger import kv_range
from model_explorer_debugger import resource_views
from model_explorer_debugger import store_cache
from model_explorer_debugger import token_analysis
import psutil

try:
  import resource  # pylint: disable=g-import-not-at-top
except ImportError:  # pragma: no cover - non-POSIX fallback
  resource = None

# Re-exported: callers name the busy error through this module.
AnalysisBusy = errors.AnalysisBusy


class AnalysisFailedError(RuntimeError):
  """An analysis query hit a server fault; carries the worker's traceback."""


AnalysisFailed = AnalysisFailedError


AnalysisHandler = Callable[[Any, Mapping[str, list[str]], Any], Any]


def _handle_kv_analysis(
    store: Any, query: Mapping[str, list[str]], _payload: Any
) -> Any:
  turn = int(query['turn'][0])
  if query.get('mode', [''])[0] == 'metadata':
    return kv_range.metadata(store, turn)
  return kv_analysis.analyze_kv(store, turn)


def _handle_comparisons(
    store: Any, query: Mapping[str, list[str]], _payload: Any
) -> Any:
  return store.compare_batch(int(query['batch'][0]))


def _handle_node(
    store: Any, query: Mapping[str, list[str]], _payload: Any
) -> Any:
  return store.node_details.details(
      int(query['layer'][0]), int(query['batch'][0]), query['semantic'][0]
  )


def _handle_resource(
    store: Any, query: Mapping[str, list[str]], _payload: Any
) -> Any:
  return resource_views.preview_resource(
      store,
      query['resource_id'][0],
      int(query.get('offset', ['0'])[0]),
      int(query.get('limit', ['64'])[0]),
  )


def _handle_pair_tensor(
    store: Any, query: Mapping[str, list[str]], _payload: Any
) -> Any:
  return resource_views.preview_pair_tensor(
      store,
      query['pair_id'][0],
      query['role'][0],
      query.get('mode', ['original'])[0],
      int(query.get('offset', ['0'])[0]),
      int(query.get('limit', ['64'])[0]),
  )


def _handle_resources_compare(
    store: Any, _query: Mapping[str, list[str]], payload: Any
) -> Any:
  return store.compare_resources(payload['reference'], payload['target'])


def _handle_selection_compare(
    store: Any, _query: Mapping[str, list[str]], payload: Any
) -> Any:
  return store.node_details.compare(payload)


GET_ANALYSIS_HANDLERS: Mapping[str, AnalysisHandler] = types.MappingProxyType({
    '/api/kv-analysis': _handle_kv_analysis,
    '/api/kv-range': lambda store, query, _: kv_range.range_query(store, query),
    '/api/kv-selection': lambda store, query, _: kv_range.selection_query(
        store, query
    ),
    '/api/kv-find': lambda store, query, _: kv_range.find_query(store, query),
    '/api/kv-head': lambda store, query, _: kv_head.kv_head_query(store, query),
    '/api/explicit-pairs': lambda store, _query, _: store.explicit_pairs(),
    '/api/overview': lambda store, _query, _: store.overview(),
    '/api/comparisons': _handle_comparisons,
    '/api/node': _handle_node,
    '/api/resource': _handle_resource,
    '/api/pair-tensor': _handle_pair_tensor,
})

POST_ANALYSIS_HANDLERS: Mapping[str, AnalysisHandler] = types.MappingProxyType({
    '/api/token-analysis': (
        lambda store, _query, payload: token_analysis.analyze_tokens(
            store, payload
        )
    ),
    '/api/resources/compare': _handle_resources_compare,
    '/api/selection/compare': _handle_selection_compare,
})

GET_ANALYSIS = frozenset(GET_ANALYSIS_HANDLERS)
POST_ANALYSIS = frozenset(POST_ANALYSIS_HANDLERS)


def evaluate(
    store: Any,
    path: str,
    query: Mapping[str, list[str]],
    payload: Any = None,
) -> Any:
  """Dispatches an analysis route to its registered domain handler."""
  handler = GET_ANALYSIS_HANDLERS.get(path) or POST_ANALYSIS_HANDLERS.get(path)
  if handler is None:
    raise ValueError('Unknown analysis operation')
  return handler(store, query, payload)


def _worker(connection, memory_bytes, evaluator):
  """Executes isolated analysis queries inside a bounded child process."""
  if sys.platform != 'darwin' and resource is not None:
    # Cap RLIMIT_AS at baseline_vms + memory_bytes so pre-existing virtual
    # address space reserved by Python/NumPy/PyTorch libraries at spawn time
    # does not immediately trip the per-worker analysis allocation budget.
    baseline_vms = psutil.Process().memory_info().vms
    target_limit = baseline_vms + memory_bytes
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = (
        target_limit
        if hard == resource.RLIM_INFINITY
        else min(target_limit, hard)
    )
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))

  cache = store_cache.StoreCache()
  try:
    while True:
      request = connection.recv()
      if request is None:
        break
      root, path, query, payload = request
      try:
        before = store_cache.revision(root)
        value = evaluator(cache.get(root), path, query, payload)
        if store_cache.revision(root) != before:
          raise ValueError('Capture changed during analysis; retry the query')
        if len(json.dumps(value, allow_nan=False).encode()) > 32 * 1024**2:
          raise ValueError('Analysis response exceeds 32 MiB')
        connection.send((True, value))
      except Exception as error:  # pylint: disable=broad-exception-caught
        connection.send((False, _describe_failure(error)))
  except (EOFError, BrokenPipeError):
    pass
  finally:
    connection.close()


def _describe_failure(error: Exception) -> tuple[str, str]:
  """Classifies an evaluator error as a picklable (category, message) pair."""
  if isinstance(error, AnalysisBusy):
    return 'busy', str(error)
  if isinstance(error, MemoryError):
    return 'busy', 'Analysis memory limit exceeded'
  if isinstance(error, errors.NotFound):
    return 'not_found', str(error)
  if isinstance(error, ValueError):
    return 'bad_request', str(error)
  return 'internal', ''.join(traceback.format_exception(error))


def _raise_failure(category: str, message: str) -> None:
  """Re-raises a worker failure in the parent with its original category."""
  if category == 'busy':
    raise AnalysisBusy(message)
  if category == 'not_found':
    raise errors.NotFound(message)
  if category == 'bad_request':
    raise errors.BadRequest(message)
  raise AnalysisFailedError(message)


class AnalysisPool:
  """Manages supervised worker processes for read-only tensor analysis."""

  def __init__(
      self,
      workers: int = 2,
      waiting: int = 8,
      timeout: float = 60,
      memory_bytes: int = 8 * 1024**3,
      evaluator: Callable[..., Any] = evaluate,
  ) -> None:
    if workers < 1 or waiting < 0 or timeout <= 0 or memory_bytes <= 0:
      raise ValueError('Invalid analysis limits')
    self.context = multiprocessing.get_context('spawn')
    self.timeout = timeout
    self.memory_bytes = memory_bytes
    self.evaluator = evaluator
    self.capacity = threading.BoundedSemaphore(workers + waiting)
    self.requests = threading.BoundedSemaphore(workers + waiting)
    self.pending: dict[Any, futures.Future[Any]] = {}
    self.available: queue.Queue[int] = queue.Queue()
    self.lock = threading.RLock()
    self.slots: dict[int, tuple[Any, Any]] = {}
    self.closed = False
    self.active = 0
    self.restarts = 0
    for slot in range(workers):
      self.available.put(slot)

  def _start(self, slot: int) -> tuple[Any, Any]:
    parent, child = self.context.Pipe()
    process = self.context.Process(
        target=_worker,
        args=(child, self.memory_bytes, self.evaluator),
        daemon=True,
    )
    try:
      process.start()
    except BaseException:
      parent.close()
      child.close()
      raise
    child.close()
    self.slots[slot] = (process, parent)
    return process, parent

  def _retire(self, slot: int) -> None:
    item = self.slots.pop(slot, None)
    if item:
      process, pipe = item
      pipe.close()
      if process.is_alive():
        process.terminate()
      process.join(2)
      if process.is_alive():
        process.kill()
        process.join(2)
      process.close()
      self.restarts += 1

  def query(
      self,
      root: Any,
      path: str,
      query: Mapping[str, list[str]],
      payload: Any = None,
      cancelled: Callable[[], bool] | None = None,
  ) -> Any:
    """Deduplicates and executes an analysis query against the worker pool."""
    if not self.requests.acquire(blocking=False):
      raise AnalysisBusy('Analysis queue is full; retry shortly')
    leader = False
    key = None
    try:
      key = (
          str(root),
          store_cache.revision(root),
          path,
          json.dumps([query, payload], sort_keys=True),
      )
      with self.lock:
        future = self.pending.get(key)
        if future is None:
          future = self.pending[key] = futures.Future()
          leader = True
      if leader:
        try:
          value = self._perform(root, path, query, payload, cancelled)
          future.set_result(value)
        except BaseException as error:
          future.set_exception(error)
          raise
      deadline = time.monotonic() + self.timeout
      while True:
        if cancelled and cancelled():
          raise AnalysisBusy('Analysis request was cancelled')
        try:
          return copy.deepcopy(future.result(timeout=0.05))
        except futures.TimeoutError:
          if time.monotonic() >= deadline:
            raise AnalysisBusy('Analysis deadline exceeded') from None
    finally:
      if leader:
        with self.lock:
          self.pending.pop(key, None)
      self.requests.release()

  def _perform(
      self,
      root: Any,
      path: str,
      query: Mapping[str, list[str]],
      payload: Any = None,
      cancelled: Callable[[], bool] | None = None,
  ) -> Any:
    if not self.capacity.acquire(blocking=False):
      raise AnalysisBusy('Analysis queue is full; retry shortly')
    slot = None
    try:
      deadline = time.monotonic() + self.timeout
      while slot is None:
        if self.closed or (cancelled and cancelled()):
          raise AnalysisBusy('Analysis request was cancelled')
        try:
          slot = self.available.get(
              timeout=min(0.1, max(0.001, deadline - time.monotonic()))
          )
        except queue.Empty as error:
          if time.monotonic() >= deadline:
            raise AnalysisBusy('Analysis queue deadline exceeded') from error
      with self.lock:
        if self.closed:
          raise AnalysisBusy('Analysis pool is closed')
        self.active += 1
        process, pipe = self.slots.get(slot) or self._start(slot)
      try:
        pipe.send((str(root), path, query, payload))
        monitor = psutil.Process(process.pid)
        while not pipe.poll(0.05):
          if self.closed:
            raise AnalysisBusy('Analysis pool is closed')
          try:
            if monitor.memory_info().rss > self.memory_bytes:
              raise AnalysisBusy('Analysis memory limit exceeded')
          except psutil.NoSuchProcess:
            raise AnalysisBusy('Analysis worker exited') from None
          if cancelled and cancelled():
            raise AnalysisBusy('Analysis request was cancelled')
          if time.monotonic() >= deadline:
            raise AnalysisBusy('Analysis deadline exceeded')
          if not process.is_alive():
            raise AnalysisBusy('Analysis worker exited')
        ok, result = pipe.recv()
      except (AnalysisBusy, EOFError, BrokenPipeError, OSError) as error:
        with self.lock:
          self._retire(slot)
        raise AnalysisBusy(
            f'Analysis interrupted: {error}; worker replaced'
        ) from None
      if not ok:
        if self.closed:
          raise AnalysisBusy('Analysis pool is closed')
        _raise_failure(*result)
      return result
    finally:
      if slot is not None:
        with self.lock:
          self.active = max(0, self.active - 1)
        self.available.put(slot)
      self.capacity.release()

  def diagnostics(self) -> dict[str, Any]:
    """Returns active worker count, restarts, and configured resource bounds."""
    with self.lock:
      return dict(
          active=self.active,
          processes=len(self.slots),
          restarts=self.restarts,
          closed=self.closed,
          timeout_seconds=self.timeout,
          memory_limit_bytes=self.memory_bytes,
      )

  def close(self) -> None:
    """Terminates all spawned worker processes and closes the pool."""
    with self.lock:
      self.closed = True
      for slot in list(self.slots):
        self._retire(slot)
