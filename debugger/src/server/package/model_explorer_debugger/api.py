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

"""Transport-independent application interface for ASGI and QA adapters."""

from collections.abc import Callable, Mapping
import re
from typing import Any

from model_explorer_debugger import analysis as analysis_module
from model_explorer_debugger import errors
from model_explorer_debugger import jobs as jobs_module
from model_explorer_debugger import session_metadata
from model_explorer_debugger import session_rules
from model_explorer_debugger import sessions_backend
from model_explorer_debugger import tap_scans as tap_scans_module
from model_explorer_debugger.runtime import runner_builds
from model_explorer_debugger.runtime import runner_discovery

PACKAGE_NAME = __package__

_SESSION_EXECUTION_RE = re.compile(r'/api/sessions/([^/]+)/execution\Z')
_ARTIFACT_SCAN_RE = re.compile(r'/api/artifacts/([^/]+)/scan\Z')
_JOB_STATUS_RE = re.compile(r'/api/jobs/([^/]+)\Z')
_SESSION_CAPTURE_RE = re.compile(r'/api/sessions/([^/]+)/capture\Z')
_SESSION_ACTION_RE = re.compile(
    r'/api/sessions/([^/]+)/(prepare|initialize|turns|close)\Z'
)
_JOB_CANCEL_RE = re.compile(r'/api/jobs/([^/]+)/cancel\Z')

_SESSION_MANAGE_PATHS = frozenset(
    '/api/sessions/' + op
    for op in (
        'create',
        'update',
        'duplicate',
        'chat',
        'chat-config',
        'delete',
        'restore',
    )
)


class Application:
  """Transport-independent controller routing requests to domain services."""

  def __init__(
      self,
      store: Any = None,
      registry: Any = None,
      analysis: Any = None,
  ) -> None:
    self.store = store
    self.registry = registry
    # Static contract; test doubles may implement only the methods their case
    # exercises.
    self.metadata: sessions_backend.SessionsBackend = (
        registry
        if registry is not None
        else session_metadata.SessionMetadata(store)
    )
    self.analysis = analysis
    self.jobs = self.scans = None
    self.closed = False
    if registry is not None:
      self.jobs = jobs_module.JobManager(registry)
      self.scans = tap_scans_module.TapScans(registry)
    self._get_exact_routes: Mapping[
        str, Callable[[Mapping[str, list[str]]], Any]
    ] = {
        '/api/sessions': lambda _query: self.metadata.listing(),
        '/api/devices': self._get_devices,
        '/api/devices/builds': self._get_device_builds,
        '/api/runtime/capabilities': self._get_runtime_capabilities,
        '/api/telemetry': self._get_telemetry,
        '/api/session': lambda query: self.selected_store(query).session,
        '/api/semantic': lambda query: self.selected_store(query).semantic,
        '/api/execution': lambda query: self.selected_store(query).execution,
    }
    self._get_pattern_routes: tuple[
        tuple[re.Pattern[str], Callable[[re.Match[str]], Any]], ...
    ] = (
        (_SESSION_EXECUTION_RE, self._get_session_execution),
        (_ARTIFACT_SCAN_RE, self._get_artifact_scan),
        (_JOB_STATUS_RE, self._get_job_status),
        (_SESSION_CAPTURE_RE, self._get_session_capture),
    )
    self._post_exact_routes: Mapping[
        str, Callable[[Mapping[str, list[str]], Any], Any]
    ] = {
        '/api/devices/inspect': self._post_devices_inspect,
        '/api/sessions/rename': lambda _query, payload: self.metadata.rename(
            payload
        ),
        '/api/mappings': (
            lambda query, payload: self.selected_store(query).node_details.save(
                payload
            )
        ),
        '/api/mappings/remove': (
            lambda query, payload: self.selected_store(
                query
            ).node_details.remove(payload)
        ),
        **{
            route: self._make_session_manage_handler(route.rsplit('/', 1)[-1])
            for route in _SESSION_MANAGE_PATHS
        },
    }
    self._post_pattern_routes: tuple[
        tuple[re.Pattern[str], Callable[[re.Match[str], Any], Any]], ...
    ] = (
        (_ARTIFACT_SCAN_RE, self._post_artifact_scan),
        (_SESSION_ACTION_RE, self._post_session_action),
        (_JOB_CANCEL_RE, self._post_job_cancel),
    )

  def _get_devices(self, _query: Mapping[str, list[str]]) -> Any:
    if self.registry is None:
      raise errors.NotFound('not_found')
    return runner_discovery.candidates(self.registry.root)

  def _get_device_builds(self, query: Mapping[str, list[str]]) -> Any:
    if self.registry is None:
      raise errors.NotFound('not_found')
    return runner_builds.catalog(self.registry.root, query.get('id', [''])[0])

  def _get_runtime_capabilities(
      self, _query: Mapping[str, list[str]]
  ) -> dict[str, Any]:
    if self.registry is not None:
      return self.registry.capabilities()
    return {
        'available': False,
        'models': [],
        'runtimes': [],
        'reason': 'Start the server with --workspace and --runtime-root.',
        'upload_limit_bytes': session_rules.UPLOAD_LIMIT_BYTES,
        'upload_idle_timeout_seconds': (
            session_rules.UPLOAD_IDLE_TIMEOUT_SECONDS
        ),
    }

  def _get_telemetry(self, query: Mapping[str, list[str]]) -> Any:
    return self.selected_store(query).telemetry(
        turn=int(query['turn'][0]) if 'turn' in query else None,
        run=query.get('run', [None])[0],
    )

  def _get_session_execution(self, match: re.Match[str]) -> Any:
    if not self.jobs:
      raise errors.NotFound('not_found')
    return self.jobs.execution(self.jobs.owner_id(match[1]))

  def _get_artifact_scan(self, match: re.Match[str]) -> Any:
    if not self.scans:
      raise errors.NotFound('not_found')
    return self.scans.get(match[1])

  def _get_job_status(self, match: re.Match[str]) -> Any:
    if not self.jobs:
      raise errors.NotFound('not_found')
    return self.jobs.get(match[1])

  def _get_session_capture(self, match: re.Match[str]) -> Any:
    if self.registry is None:
      raise errors.NotFound('not_found')
    return self.registry.capture_store(match[1]).session

  def _post_devices_inspect(
      self, _query: Mapping[str, list[str]], payload: Mapping[str, Any]
  ) -> Any:
    if self.registry is None:
      raise errors.NotFound('not_found')
    return runner_discovery.inspect(self.registry.root, payload.get('id'))

  def _make_session_manage_handler(
      self, operation: str
  ) -> Callable[[Mapping[str, list[str]], Any], Any]:
    def _handle(_query: Mapping[str, list[str]], payload: Any) -> Any:
      manager = self.jobs.manage if self.jobs else self.metadata.manage
      return manager(operation, payload)

    return _handle

  def _post_artifact_scan(self, match: re.Match[str], _payload: Any) -> Any:
    if not self.scans:
      raise errors.NotFound('not_found')
    return self.scans.start(match[1])

  def _post_session_action(self, match: re.Match[str], payload: Any) -> Any:
    if not self.jobs:
      raise errors.NotFound('not_found')
    if match[2] == 'close':
      return self.jobs.close_session(match[1])
    operation = 'generate' if match[2] == 'turns' else match[2]
    return self.jobs.start(match[1], operation, payload)

  def _post_job_cancel(self, match: re.Match[str], _payload: Any) -> Any:
    if not self.jobs:
      raise errors.NotFound('not_found')
    return self.jobs.cancel(match[1])

  def selected_store(self, query: Mapping[str, list[str]]) -> Any:
    """Returns the CaptureStore for the active session_id query parameter."""
    if self.registry is not None:
      return self.registry.capture_store(query.get('session_id', [''])[0])
    if self.store is None:
      raise ValueError('No saved capture selected')
    return self.store

  def analyze(
      self,
      path: str,
      query: Mapping[str, list[str]],
      payload: Any = None,
      cancelled: Callable[[], bool] | None = None,
  ) -> Any:
    """Executes an analysis route via the worker pool or inline evaluator."""
    store = self.selected_store(query)
    if self.analysis is not None:
      return self.analysis.query(
          store.root, path, query, payload, cancelled=cancelled
      )
    return analysis_module.evaluate(store, path, query, payload)

  def get(
      self,
      path: str,
      query: Mapping[str, list[str]],
      cancelled: Callable[[], bool] | None = None,
  ) -> Any:
    """Dispatches a GET route to its exact, regex, or analysis handler."""
    if path in analysis_module.GET_ANALYSIS:
      return self.analyze(path, query, cancelled=cancelled)
    exact = self._get_exact_routes.get(path)
    if exact is not None:
      return exact(query)
    for pattern, handler in self._get_pattern_routes:
      if match := pattern.fullmatch(path):
        return handler(match)
    raise errors.NotFound('not_found')

  def post(
      self,
      path: str,
      query: Mapping[str, list[str]],
      payload: Any,
      cancelled: Callable[[], bool] | None = None,
  ) -> Any:
    """Dispatches a POST route to its exact, regex, or analysis handler."""
    if path in analysis_module.POST_ANALYSIS:
      return self.analyze(path, query, payload, cancelled)
    exact = self._post_exact_routes.get(path)
    if exact is not None:
      return exact(query, payload)
    for pattern, handler in self._post_pattern_routes:
      if match := pattern.fullmatch(path):
        return handler(match, payload)
    raise errors.NotFound('not_found')

  def diagnostics(self) -> dict[str, Any]:
    """Returns server readiness, storage backend, and worker diagnostics."""
    return {
        'status': 'stopping' if self.closed else 'ok',
        'package': PACKAGE_NAME,
        'storage': 'sqlite' if self.registry is not None else 'saved_capture',
        'active_jobs': len(self.jobs.active_jobs) if self.jobs else 0,
        'analysis': self.analysis.diagnostics() if self.analysis else None,
    }

  def close(self) -> None:
    """Stops active analysis pools, job managers, tap scans, and registries."""
    if self.closed:
      return
    self.closed = True
    if self.analysis:
      self.analysis.close()
    try:
      if self.jobs:
        self.jobs.close()
    finally:
      try:
        if self.scans:
          self.scans.close()
      finally:
        if hasattr(self.registry, 'close'):
          self.registry.close()
