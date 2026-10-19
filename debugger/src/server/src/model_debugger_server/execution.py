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

"""The Model Server of each Session: its two runners, their chat and turns.

A Session (the owner) runs one execution at a time: a runner per side, ref
and target, each over one connection that holds one model and one chat. A
runner serves one connection for its whole life and one chat per connection,
so a new chat relaunches both runners, and an execution that ended can never
be resumed, not even by this server after a restart.

The UI polls the listing for the execution's phase and watches jobs for
progress: initialize, generate and new_chat each run as a job.
"""

import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import logging
import pathlib
import queue
import shutil
import threading
import types

from google.protobuf import json_format
import grpc
from model_debugger.v1 import runner_pb2
from model_debugger_server import capture_index, errors, options, runner_client
from model_debugger_server import jobs as jobs_lib
from model_debugger_server import launchers as launchers_lib
from model_debugger_server import models as models_lib
from model_debugger_server import sessions as sessions_lib
from model_debugger_server import workspace as workspace_lib

SIDES = ('ref', 'target')
SIDE_NAMES = types.MappingProxyType({'ref': 'Reference', 'target': 'Target'})
# The longest prompt, in UTF-8 bytes. The runner's gRPC server takes
# messages of at most 4 MiB, and a Send request wraps the prompt in fields of
# its own; a larger one would be refused before the runner saw it.
PROMPT_LIMIT_BYTES = 4 * 1024 * 1024 - 1024
# Phases in which an execution holds its devices.
OWNING_PHASES = ('starting', 'active', 'ending', 'unavailable')
CHAT_OVER = 'This chat is over; start a new chat to continue.'

_logger = logging.getLogger(__name__)


class Failed(Exception):
  """A runner failed what the request needed; the message says which side."""


class _SideError(Exception):
  """One side's failure, named for the user."""

  def __init__(self, side: str, cause: Exception) -> None:
    super().__init__(f'{SIDE_NAMES[side]}: {cause}')
    self.side = side
    self.cause = cause


class _Aborted(Exception):
  """The execution was turned off, or the other side failed, meanwhile."""


@dataclasses.dataclass
class _Side:
  """One side's runner as this execution holds it."""

  role: str
  device: launchers_lib.Device
  phase: str = 'starting'
  runner: launchers_lib.LaunchedRunner | None = None
  client: runner_client.RunnerClient | None = None
  info: runner_pb2.RunnerInfo | None = None


@dataclasses.dataclass
class _Turn:
  """A generate job in flight."""

  job_id: str
  n: int
  prompt: str
  generations: dict[str, runner_client.Generation] = dataclasses.field(
      default_factory=dict
  )
  cancel_requested: bool = False
  # Both completions arrived: the turn is the runners' now, cancel or not.
  completed: bool = False


@dataclasses.dataclass
class _Execution:
  """The Model Server of one owner Session."""

  owner_id: str
  sides: dict[str, _Side]
  phase: str = 'starting'
  error: str = ''
  active_chat_id: str = ''
  new_chat_allowed: bool = False
  index: capture_index.CaptureIndex | None = None
  # The running job, its operation and the record it runs for.
  job_id: str | None = None
  operation: str | None = None
  job_record: str | None = None
  turn: _Turn | None = None
  # The record whose start failed, shown as failed while interrupted.
  failed_record: str | None = None
  became_active: bool = False
  # Set by close and by a failed start: runners launched afterwards stop.
  aborted: bool = False
  closing: bool = False
  # A start (initialize or new_chat) may still launch a runner: a closed
  # execution keeps its devices until it no longer can.
  starting: bool = True


def _now() -> str:
  return datetime.datetime.now(datetime.UTC).isoformat()


def _digest(*parts: object) -> str:
  return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _code(error: object) -> grpc.StatusCode | None:
  """The gRPC status of a runner's error; None for anything else."""
  if isinstance(error, runner_client.RunnerError):
    return error.code
  return None


def _owner_id(record: dict) -> str:
  return record.get('parent_session_id') or record['id']


def _run(record: dict, side: str) -> dict:
  for run in record.get('runs', []):
    if run.get('id') == side:
      return run
  raise ValueError(f'The Session has no {SIDE_NAMES[side]} run.')


def _backends(
    device: launchers_lib.Device, info: runner_pb2.RunnerInfo
) -> list[str]:
  gpu = info.host.HasField('gpu') and device.gpu
  return ['CPU', 'GPU'] if gpu else ['CPU']


def _check_info(info: runner_pb2.RunnerInfo) -> None:
  if info.protocol_version != runner_client.PROTOCOL_VERSION:
    raise RuntimeError(
        f'The runner speaks protocol {info.protocol_version or "(none)"}; '
        f'this server speaks {runner_client.PROTOCOL_VERSION}.'
    )
  if info.runtime.name != runner_pb2.RuntimeInfo.LITERT_LM:
    name = runner_pb2.RuntimeInfo.Name.Name(info.runtime.name)
    raise RuntimeError(f'The runner hosts {name}; this server runs LiteRT-LM.')


def _side_entry(
    device: launchers_lib.Device,
    model: models_lib.Model,
    info: runner_pb2.RunnerInfo,
    load: runner_pb2.LoadRequest,
    chat: runner_pb2.CreateChatRequest,
    loaded: runner_pb2.LoadResponse,
) -> dict:
  """The fixed facts of one side of a chat, as index.json keeps them."""
  stop_tokens = []
  for token in loaded.stop_tokens:
    if token.WhichOneof('token') == 'ids':
      stop_tokens.append(list(token.ids.ids))
    else:
      stop_tokens.append({'text': token.text})
  return {
      'model': {'sha256': model.sha256, 'name': model.name, 'size': model.size},
      'runtime': {'name': 'LiteRT-LM', 'version': info.runtime.version},
      'host': json_format.MessageToDict(
          info.host, preserving_proto_field_name=True
      ),
      'device_id': device.id,
      'device_name': device.name,
      'load': {
          'device': runner_pb2.Device.Name(load.device),
          'context_length': load.context_length,
          'runtime_options': json_format.MessageToDict(load.runtime_options),
      },
      'chat': {
          'messages': [
              {
                  'role': (
                      runner_pb2.ChatMessage.Role.Name(message.role).lower()
                  ),
                  'content': message.content,
              }
              for message in chat.messages
          ],
          'options': json_format.MessageToDict(
              chat.options, preserving_proto_field_name=True
          ),
      },
      'start_token': (
          loaded.start_token if loaded.HasField('start_token') else None
      ),
      'stop_tokens': stop_tokens,
  }


class Executions:
  """Every Session's Model Server, and the devices they run on."""

  def __init__(
      self,
      sessions: sessions_lib.SessionStore,
      models: models_lib.ModelStore,
      jobs: jobs_lib.Jobs,
      launchers: launchers_lib.Launchers,
      workspace: workspace_lib.Workspace,
  ) -> None:
    """Holds the stores; nothing runs until a Session is turned on."""
    self._sessions = sessions
    self._models = models
    self._jobs = jobs
    self._launchers = launchers
    self._workspace = workspace
    # Guards every execution's state; never held across a runner call.
    self._lock = threading.RLock()
    self._executions: dict[str, _Execution] = {}
    # The last job this server ran for each record, for the UI to reattach.
    self._last_jobs: dict[str, str] = {}
    # Non-shared devices being inspected right now.
    self._probing: set[str] = set()
    # The last RunnerInfo each device answered with.
    self._infos: dict[str, runner_pb2.RunnerInfo] = {}
    self._indexes: dict[str, capture_index.CaptureIndex] = {}
    self._upload_locks: dict[str, threading.Lock] = {}

  # ---- Projections for the listing.

  def view(self, owner_id: str) -> dict | None:
    """The UI's SessionExecution of an owner, or None if it never ran here."""
    with self._lock:
      execution = self._executions.get(owner_id)
      if execution is None:
        return None
      runners = []
      launched = []
      for side in execution.sides.values():
        runner = {
            'role': side.role,
            'deviceId': side.device.id,
            'deviceName': side.device.name,
            'phase': side.phase,
            'connected': False,
        }
        if side.info is not None:
          runner['buildId'] = side.info.runtime.version
        runners.append(runner)
        launched.append(side.runner)
      view = {
          'phase': execution.phase,
          'runners': runners,
          'newChatAllowed': execution.new_chat_allowed,
          'activeChatId': execution.active_chat_id,
      }
      if execution.error:
        view['error'] = execution.error
      if execution.job_id:
        view['activeJobId'] = execution.job_id
        view['activeOperation'] = execution.operation
    # Asking a device (adb) takes time: never while holding the lock.
    for runner, process in zip(runners, launched):
      runner['connected'] = process is not None and process.alive()
    return view

  def summary(self, record: dict) -> dict:
    """A record as the listing shows it: with its execution and status."""
    record_id = record['id']
    owner_id = _owner_id(record)
    view = self.view(owner_id)
    with self._lock:
      execution = self._executions.get(owner_id)
      owning = execution is not None and execution.phase in OWNING_PHASES
      initialized = (
          execution is not None
          and execution.phase == 'active'
          and execution.active_chat_id == record_id
          and not record.get('chat_over')
      )
      if execution and execution.job_record == record_id:
        status = (
            'running' if execution.operation == 'generate' else 'initializing'
        )
      elif record.get('has_capture'):
        status = 'saved'
      elif (
          execution
          and execution.phase == 'interrupted'
          and execution.failed_record == record_id
      ):
        status = 'failed'
      elif initialized:
        status = 'ready'
      else:
        status = 'draft'
      summary = dict(record)
      if view is not None:
        summary['execution'] = view
      summary.update(
          status=status,
          initialized=initialized,
          has_capture=bool(record.get('has_capture')),
          # Recorded once a Session first runs; shown once that run is over.
          execution_ended=bool(record.get('execution_ended')) and not owning,
          tap_profile='',
          tap_points={},
          notice=record.get('notice', ''),
      )
      if record_id in self._last_jobs:
        summary['job_id'] = self._last_jobs[record_id]
      return summary

  def require_idle(self, record_id: str) -> None:
    """Refuses changing a record whose Session's Model Server is on.

    Raises:
      Conflict: the owner holds an execution.
    """
    owner_id = _owner_id(self._sessions.get(record_id))
    with self._lock:
      execution = self._executions.get(owner_id)
      if execution is not None and execution.phase in OWNING_PHASES:
        raise errors.Conflict('Turn the Model Server off first.')

  # ---- Lifecycle.

  def initialize(self, session_id: str, request_id: str) -> dict:
    """Turns a Session's Model Server on: both runners, model and chat.

    Returns:
      The initialize job; it runs on a thread of its own.
    """
    owner = self._sessions.get(_owner_id(self._sessions.get(session_id)))
    owner_id = owner['id']
    available = {device.id: device for device in self._launchers.devices()}
    with self._lock:
      job, created = self._jobs.create(
          owner_id, 'initialize', request_id, _digest('initialize', owner_id)
      )
      if not created:
        return job
      try:
        execution = self._admit_start(owner, available)
      except Exception as error:
        self._jobs.finish(job['id'], 'failed', str(error))
        raise
      execution.job_id = job['id']
      execution.operation = 'initialize'
      execution.job_record = owner_id
      self._executions[owner_id] = execution
      self._last_jobs[owner_id] = job['id']
    threading.Thread(
        target=self._run_start,
        args=(execution, owner, owner, job['id'], False),
        name=f'initialize-{owner_id}',
        daemon=True,
    ).start()
    return self._jobs.get(job['id'])

  def _admit_start(
      self, owner: dict, available: dict[str, launchers_lib.Device]
  ) -> _Execution:
    """Checks a start against the record and every device's holder."""
    current = self._executions.get(owner['id'])
    if current is not None and current.phase in OWNING_PHASES:
      raise errors.Conflict("This Session's Model Server is already on.")
    if owner.get('execution_ended'):
      raise errors.Conflict(
          'This Session has ended; duplicate it to run again.'
      )
    sides = {}
    for role in SIDES:
      run = _run(owner, role)
      device_id = run['device']
      device = available.get(device_id)
      if device is None:
        raise errors.Conflict(
            f'{SIDE_NAMES[role]}: the device {device_id} is not connected.'
        )
      if not device.shared:
        if any(side.device.id == device.id for side in sides.values()):
          raise errors.Conflict(
              f'{device.name} runs one runner at a time; choose different '
              'devices for Reference and Target.'
          )
        holder = self._holder(device.id)
        if holder is not None or device.id in self._probing:
          raise errors.Conflict(
              f'{SIDE_NAMES[role]}: {device.name} is in use by another Session.'
          )
      self._models.get(run.get('artifact', ''))
      sides[role] = _Side(role, device)
    return _Execution(owner['id'], sides)

  def _holder(self, device_id: str) -> tuple[_Execution, _Side] | None:
    """The execution and side holding a device, if any."""
    for execution in self._executions.values():
      if execution.phase not in OWNING_PHASES:
        continue
      for side in execution.sides.values():
        if side.device.id == device_id:
          return execution, side
    return None

  def new_chat(
      self, parent_id: str, name: str, generation: dict | None
  ) -> dict:
    """Starts a new chat on fresh runners; blocks until it is ready.

    Returns:
      The new chat's record, as the listing shows it.

    Raises:
      Conflict: the Model Server is not on, or a job runs.
      Failed: a runner failed; the Model Server is then interrupted.
    """
    owner = self._sessions.get(_owner_id(self._sessions.get(parent_id)))
    owner_id = owner['id']
    with self._lock:
      execution = self._executions.get(owner_id)
      if execution is None or execution.phase != 'active':
        raise errors.Conflict('Turn the Model Server on first.')
      if execution.job_id:
        raise errors.Conflict('Wait for the current task to finish.')
      chat = self._sessions.create_chat(owner_id, name, generation)
      job, _ = self._jobs.create(chat['id'], 'new_chat', None, '')
      previous = execution.active_chat_id
      execution.job_id = job['id']
      execution.operation = 'new_chat'
      execution.job_record = chat['id']
      execution.new_chat_allowed = False
      execution.starting = True
      self._last_jobs[chat['id']] = job['id']
    try:
      self._mark_chat_over(previous, '')
      self._set_phases(execution, 'stopping')
      self._stop_runners(execution)
      with self._lock:
        for side in execution.sides.values():
          side.runner = side.client = side.info = None
    except Exception as error:
      self._fail_start(execution, chat['id'], job['id'], error, True)
    else:
      self._run_start(execution, owner, chat, job['id'], True)
    job = self._jobs.get(job['id'])
    if job['status'] != 'completed':
      raise Failed(job['error'])
    return self.summary(self._sessions.get(chat['id']))

  def _run_start(
      self,
      execution: _Execution,
      owner: dict,
      chat: dict,
      job_id: str,
      new_chat: bool,
  ) -> None:
    """Brings both sides up for a chat and makes it the active one."""
    try:
      entries = self._start_sides(execution, owner, chat, job_id)
      with self._lock:
        if execution.aborted:
          raise _Aborted()
        index = capture_index.CaptureIndex.create(
            self._capture_root(chat['id']), entries
        )
        self._indexes[chat['id']] = index
        execution.index = index
        execution.phase = 'active'
        execution.error = ''
        execution.active_chat_id = chat['id']
        execution.new_chat_allowed = True
        execution.became_active = True
        execution.starting = False
        execution.job_id = execution.operation = execution.job_record = None
        for side in execution.sides.values():
          side.phase = 'active'
    except Exception as error:
      self._fail_start(execution, chat['id'], job_id, error, new_chat)
      return
    try:
      # A runner cannot be reattached, so this record can never run again.
      self._sessions.set_flags(chat['id'], execution_ended=True)
    except Exception:
      _logger.exception('Failed to mark execution_ended on %s', chat['id'])
    self._jobs.finish(job_id, 'completed')

  def _start_sides(
      self, execution: _Execution, owner: dict, chat: dict, job_id: str
  ) -> dict[str, dict]:
    """Starts both sides in parallel; the first failure stops the other."""
    with concurrent.futures.ThreadPoolExecutor(len(SIDES)) as pool:
      futures = {
          pool.submit(
              self._start_side, execution, role, owner, chat, job_id
          ): role
          for role in SIDES
      }
      entries = {}
      failure = None
      for future in concurrent.futures.as_completed(futures):
        try:
          entries[futures[future]] = future.result()
        except Exception as error:
          if failure is None or isinstance(failure, _Aborted):
            failure = error
          with self._lock:
            execution.aborted = True
          # Stopping the runners makes the other side's call fail at once.
          self._stop_runners(execution)
      if failure is not None:
        raise failure
      return entries

  def _start_side(
      self,
      execution: _Execution,
      role: str,
      owner: dict,
      chat: dict,
      job_id: str,
  ) -> dict:
    side = execution.sides[role]
    try:
      run = _run(owner, role)
      model = self._models.get(run.get('artifact', ''))
      self._progress(execution, job_id, role, 'starting', 'Starting runner')
      launched = self._launchers.launch(side.device.id)
      with self._lock:
        side.runner = launched
        aborted = execution.aborted
      if aborted:
        launched.stop()
        raise _Aborted()
      client = runner_client.RunnerClient(launched.address)
      with self._lock:
        side.client = client
      self._set_phase(execution, role, 'connecting')
      info = client.get_info()
      _check_info(info)
      with self._lock:
        side.info = info
        self._infos[side.device.id] = info
      load = options.load_request(run, model, info)
      self._progress(execution, job_id, role, 'loading', 'Loading model')
      loaded = self._load(execution, job_id, role, client, load, model)
      generation = (chat.get('generation') or {}).get(role) or {}
      chat_request = options.create_chat_request(generation, info)
      self._progress(execution, job_id, role, 'initializing', 'Starting chat')
      client.create_chat(chat_request)
      return _side_entry(side.device, model, info, load, chat_request, loaded)
    except _Aborted:
      raise
    except Exception as error:
      # Refusals and failures the user can act on carry their own message.
      expected = (
          runner_client.RunnerError,
          ValueError,
          RuntimeError,
          errors.NotFound,
      )
      if not isinstance(error, expected):
        _logger.exception('%s failed to start', SIDE_NAMES[role])
      raise _SideError(role, error) from error

  def _load(
      self,
      execution: _Execution,
      job_id: str,
      role: str,
      client: runner_client.RunnerClient,
      request: runner_pb2.LoadRequest,
      model: models_lib.Model,
  ) -> runner_pb2.LoadResponse:
    """Loads the model, uploading it first if the runner lacks it."""
    try:
      return client.load(request)
    except runner_client.RunnerError as error:
      if error.code != grpc.StatusCode.NOT_FOUND:
        raise
    # Runners on one device may share a store: one upload of a model per
    # device at a time, and a runner that waited may find it there already.
    device_id = execution.sides[role].device.id
    with self._lock:
      upload_lock = self._upload_locks.setdefault(
          (device_id, model.sha256), threading.Lock()
      )
    with upload_lock:
      try:
        return client.load(request)
      except runner_client.RunnerError as error:
        if error.code != grpc.StatusCode.NOT_FOUND:
          raise
      shown = -1

      def on_progress(sent: int, total: int) -> None:
        nonlocal shown
        percent = sent * 100 // total if total else 100
        if percent != shown:
          shown = percent
          self._progress(
              execution,
              job_id,
              role,
              'loading',
              f'Uploading model · {percent}%',
          )

      client.upload_model(model.path, model.sha256, on_progress)
    self._progress(execution, job_id, role, 'loading', 'Loading model')
    return client.load(request)

  def _fail_start(
      self,
      execution: _Execution,
      record_id: str,
      job_id: str,
      error: Exception,
      new_chat: bool,
  ) -> None:
    """Stops every runner of a start that failed or was turned off."""
    with self._lock:
      execution.aborted = True
      closing = execution.closing
    self._stop_runners(execution)
    message = (
        'The Model Server was turned off while starting.'
        if closing or isinstance(error, _Aborted)
        else str(error)
    )
    with self._lock:
      execution.job_id = execution.operation = execution.job_record = None
      execution.starting = False
      if execution.closing:
        # Every runner of this start is stopped: close may end it now.
        self._end_closed(execution)
      else:
        execution.phase = 'interrupted'
        execution.error = message
        execution.failed_record = record_id
        execution.new_chat_allowed = False
        for side in execution.sides.values():
          side.phase = 'failed'
    try:
      if new_chat:
        self._mark_chat_over(record_id, message)
    finally:
      self._jobs.finish(job_id, 'failed', message)

  def generate(self, chat_id: str, request_id: str, prompt: str) -> dict:
    """Sends a turn to both sides and stores its capture.

    Returns:
      The generate job; it runs on a thread of its own.
    """
    if not isinstance(prompt, str) or not prompt:
      raise ValueError('Type a prompt.')
    if len(prompt.encode()) > PROMPT_LIMIT_BYTES:
      raise ValueError(
          f'The prompt is longer than {PROMPT_LIMIT_BYTES} bytes, the most a '
          'runner takes.'
      )
    record = self._sessions.get(chat_id)
    owner_id = _owner_id(record)
    with self._lock:
      execution = self._executions.get(owner_id)
      index = execution.index if execution is not None else None
      n = len(index.turns) + 1 if index is not None else None
      job, created = self._jobs.create(
          chat_id,
          'generate',
          request_id,
          _digest('generate', chat_id, prompt),
          n,
      )
      if not created:
        return job
      try:
        if execution is None or execution.phase != 'active':
          raise errors.Conflict('Turn the Model Server on first.')
        if execution.active_chat_id != chat_id or record.get('chat_over'):
          raise errors.Conflict(CHAT_OVER)
        if execution.job_id:
          raise errors.Conflict('Wait for the current task to finish.')
      except errors.Conflict as error:
        self._jobs.finish(job['id'], 'failed', str(error))
        raise
      turn = _Turn(job['id'], n, prompt)
      execution.job_id = job['id']
      execution.operation = 'generate'
      execution.job_record = chat_id
      execution.turn = turn
      self._last_jobs[chat_id] = job['id']
      for side in execution.sides.values():
        side.phase = 'generating'
    threading.Thread(
        target=self._run_generate,
        args=(execution, chat_id, turn),
        name=f'generate-{chat_id}-{n}',
        daemon=True,
    ).start()
    return self._jobs.get(job['id'])

  def _run_generate(
      self, execution: _Execution, chat_id: str, turn: _Turn
  ) -> None:
    try:
      results = self._send(execution, turn)
      failures = {
          role: result
          for role, result in results.items()
          if isinstance(result, Exception)
      }
      if not failures:
        self._finish_turn(execution, chat_id, turn, results)
      elif len(failures) == len(SIDES) and all(
          _code(failure) == grpc.StatusCode.INVALID_ARGUMENT
          for failure in failures.values()
      ):
        # Both refused the turn and nothing changed: the chat goes on.
        # The sides finish in either order; name them in a fixed one.
        message = ' '.join(
            f'{SIDE_NAMES[role]}: {failures[role]}' for role in SIDES
        )
        self._end_job(execution, turn, 'active')
        self._jobs.finish(turn.job_id, 'failed', message)
      else:
        self._end_chat(execution, chat_id, turn, failures)
    except _SideError as error:
      self._end_chat(
          execution, chat_id, turn, {error.side: error.cause}, str(error)
      )
    except Exception as error:
      _logger.exception('Turn %d failed', turn.n)
      self._end_chat(execution, chat_id, turn, {}, str(error))

  def _send(
      self, execution: _Execution, turn: _Turn
  ) -> dict[str, runner_pb2.Completion | Exception]:
    """Runs one Send per side; returns each side's Completion or error.

    A side that fails cancels the other, since the sides are then out of
    step, except that a refusal (INVALID_ARGUMENT) waits to see whether the
    other side refuses too; it is cancelled once it starts replying.
    """
    events = queue.Queue()
    results: dict[str, runner_pb2.Completion | Exception] = {}
    replying: set[str] = set()

    def other(role: str) -> str:
      return SIDES[1 - SIDES.index(role)]

    def refused(role: str) -> bool:
      return _code(results.get(role)) == grpc.StatusCode.INVALID_ARGUMENT

    for role in SIDES:
      side = execution.sides[role]
      started = threading.Event()

      def on_chunk(chunk, role=role, started=started):
        if chunk.WhichOneof('delta') == 'content':
          self._jobs.emit(turn.job_id, 'delta', runId=role, text=chunk.content)
        if not started.is_set():
          started.set()
          events.put((role, 'chunk', None))

      try:
        generation = side.client.send(turn.prompt, on_chunk)
      except Exception as error:
        events.put((role, 'result', error))
        continue
      with self._lock:
        turn.generations[role] = generation
        if turn.cancel_requested:
          generation.cancel()

      def wait(role=role, generation=generation):
        try:
          events.put((role, 'result', generation.wait()))
        except Exception as error:
          if not isinstance(error, runner_client.RunnerError):
            _logger.exception('%s Send failed', SIDE_NAMES[role])
          events.put((role, 'result', error))

      threading.Thread(target=wait, daemon=True).start()

    while len(results) < len(SIDES):
      role, kind, value = events.get()
      if kind == 'chunk':
        replying.add(role)
        if refused(other(role)):
          self._cancel_side(turn, role)
        continue
      results[role] = value
      if isinstance(value, Exception) and (
          not refused(role) or other(role) in replying or other(role) in results
      ):
        self._cancel_side(turn, other(role))
    with self._lock:
      turn.completed = all(
          isinstance(result, runner_pb2.Completion)
          for result in results.values()
      )
    return results

  def _cancel_side(self, turn: _Turn, role: str) -> None:
    with self._lock:
      generation = turn.generations.get(role)
    if generation is not None:
      generation.cancel()

  def _finish_turn(
      self,
      execution: _Execution,
      chat_id: str,
      turn: _Turn,
      completions: dict[str, runner_pb2.Completion],
  ) -> None:
    """Fetches both sides' captures, then records the turn."""
    for role, completion in completions.items():
      if completion.model_debugger.turn != turn.n:
        raise RuntimeError(
            f'{SIDE_NAMES[role]}: the runner ended turn'
            f' {completion.model_debugger.turn}; this chat expected turn'
            f' {turn.n}.'
        )
    index = execution.index
    for role in SIDES:
      self._progress(
          execution, turn.job_id, role, 'transferring', 'Retrieving capture'
      )
    try:
      with concurrent.futures.ThreadPoolExecutor(len(SIDES)) as pool:
        futures = [
            pool.submit(self._fetch_capture, execution, role, index, turn.n)
            for role in SIDES
        ]
        for future in futures:
          future.result()
      index.add_turn(
          turn.n,
          turn.prompt,
          {
              role: capture_index.completion_dict(completion)
              for role, completion in completions.items()
          },
      )
    except Exception:
      # The turn is not recorded, so neither are its files.
      for role in SIDES:
        shutil.rmtree(index.turn_directory(role, turn.n), ignore_errors=True)
      raise
    self._sessions.set_flags(chat_id, has_capture=True)
    self._end_job(execution, turn, 'active')
    self._jobs.finish(turn.job_id, 'completed')

  def _fetch_capture(
      self,
      execution: _Execution,
      role: str,
      index: capture_index.CaptureIndex,
      n: int,
  ) -> None:
    client = execution.sides[role].client
    try:
      directory = index.turn_directory(role, n)
      for file in client.list_captures(n):
        path = pathlib.PurePosixPath(file.path)
        if path.is_absolute() or '..' in path.parts or not path.parts:
          raise ValueError(f'The runner listed a capture file {file.path!r}.')
        destination = directory.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.read_capture(n, file, destination)
      client.release_captures(n)
    except Exception as error:
      raise _SideError(role, error) from error

  def _end_job(self, execution: _Execution, turn: _Turn, phase: str) -> None:
    with self._lock:
      if execution.turn is turn:
        execution.turn = None
        execution.job_id = execution.operation = execution.job_record = None
      if not execution.closing:
        for side in execution.sides.values():
          side.phase = phase

  def _end_chat(
      self,
      execution: _Execution,
      chat_id: str,
      turn: _Turn,
      failures: dict[str, Exception],
      message: str = '',
  ) -> None:
    """Ends a chat whose sides are out of step or whose runner failed."""
    with self._lock:
      cancelled = turn.cancel_requested
      closing = execution.closing
    # A side this server cancelled because the other failed says nothing.
    cause = [
        f'{SIDE_NAMES[role]}: {failures[role]}'
        for role in SIDES
        if role in failures
        and _code(failures[role]) != grpc.StatusCode.CANCELLED
    ]
    if cancelled or closing:
      message = 'Generation stopped.'
    elif not message:
      message = ' '.join(cause) or 'Generation stopped.'
    message = f'{message.rstrip(".")}. {CHAT_OVER}'
    lost = any(
        _code(failure) == grpc.StatusCode.UNAVAILABLE
        for failure in failures.values()
    )
    self._end_job(execution, turn, 'ended')
    with self._lock:
      if not execution.closing:
        if lost and not cancelled:
          execution.phase = 'unavailable'
          execution.error = message
          execution.new_chat_allowed = False
          for role, failure in failures.items():
            if _code(failure) == grpc.StatusCode.UNAVAILABLE:
              execution.sides[role].phase = 'disconnected'
        else:
          execution.new_chat_allowed = True
    try:
      if not closing:
        self._mark_chat_over(chat_id, message)
    finally:
      self._jobs.finish(
          turn.job_id,
          'cancelled' if cancelled or closing else 'failed',
          message,
      )

  def cancel(self, job_id: str) -> dict:
    """Stops a turn whose replies have not both arrived; ends its chat.

    Raises:
      Conflict: the job starts a Model Server, which is turned off instead.
    """
    job = self._jobs.get(job_id)
    if job['status'] in jobs_lib.TERMINAL:
      return job
    if job['operation'] != 'generate':
      raise errors.Conflict(
          'Starting a Model Server cannot be cancelled; turn it off instead.'
      )
    with self._lock:
      generations = []
      for execution in self._executions.values():
        turn = execution.turn
        if turn is not None and turn.job_id == job_id and not turn.completed:
          turn.cancel_requested = True
          generations = list(turn.generations.values())
    for generation in generations:
      generation.cancel()
    return self._jobs.get(job_id)

  def close(self, session_id: str) -> dict:
    """Turns a Session's Model Server off: stops both runners for good.

    The devices stay held until no start of the execution can launch a
    runner any more: one still launching stops its runner once it returns.

    Returns:
      The record, as the listing shows it.
    """
    record = self._sessions.get(session_id)
    owner_id = _owner_id(record)
    generations = []
    with self._lock:
      execution = self._executions.get(owner_id)
      if execution is None or execution.phase in ('ended', 'ending'):
        execution = None
      else:
        execution.closing = execution.aborted = True
        execution.phase = 'ending'
        execution.new_chat_allowed = False
        for side in execution.sides.values():
          side.phase = 'ending'
        if execution.turn is not None and not execution.turn.completed:
          execution.turn.cancel_requested = True
          generations = list(execution.turn.generations.values())
    if execution is not None:
      for generation in generations:
        generation.cancel()
      self._stop_runners(execution)
      with self._lock:
        if not execution.starting:
          self._end_closed(execution)
    return self.summary(self._sessions.get(session_id))

  def _end_closed(self, execution: _Execution) -> None:
    """Ends an execution being turned off; it frees its devices.

    Called with the lock held, once every runner of it is stopped and no
    start of it can launch another.
    """
    if execution.became_active:
      execution.phase = 'ended'
      for side in execution.sides.values():
        side.phase = 'ended'
    elif self._executions.get(execution.owner_id) is execution:
      # Never on: nothing ended, the Session can start again.
      del self._executions[execution.owner_id]

  def shutdown(self) -> None:
    """Stops every runner: the server is exiting."""
    with self._lock:
      executions = list(self._executions.values())
      generations = []
      for execution in executions:
        execution.closing = execution.aborted = True
        if execution.turn is not None:
          execution.turn.cancel_requested = True
          generations.extend(execution.turn.generations.values())
    for generation in generations:
      generation.cancel()
    for execution in executions:
      self._stop_runners(execution)

  def _stop_runners(self, execution: _Execution) -> None:
    """Closes both channels and stops both runners; idempotent."""
    with self._lock:
      sides = list(execution.sides.values())
    for side in sides:
      if side.client is not None:
        side.client.close()
    with concurrent.futures.ThreadPoolExecutor(len(sides)) as pool:
      for side in sides:
        if side.runner is not None:
          pool.submit(side.runner.stop)

  def _mark_chat_over(self, record_id: str, notice: str) -> None:
    if not record_id:
      return
    flags = {'chat_over': True}
    if notice:
      flags['notice'] = notice
    try:
      self._sessions.set_flags(record_id, **flags)
    except Exception:
      _logger.exception('Failed to mark chat_over on %s', record_id)

  def _set_phases(self, execution: _Execution, phase: str) -> None:
    with self._lock:
      for side in execution.sides.values():
        side.phase = phase

  def _set_phase(self, execution: _Execution, role: str, phase: str) -> None:
    with self._lock:
      if not execution.closing:
        execution.sides[role].phase = phase

  def _progress(
      self,
      execution: _Execution,
      job_id: str,
      role: str,
      phase: str,
      message: str,
  ) -> None:
    self._set_phase(execution, role, phase)
    self._jobs.emit(job_id, 'progress', runId=role, message=message)

  # ---- Devices and capabilities.

  def devices(self) -> dict:
    """The UI's DeviceCandidates: every device a runner can run on now."""
    return {
        'devices': [
            {
                'id': device.id,
                'name': device.name,
                'platform': device.platform,
                'transport': device.transport,
                'status': 'unknown',
            }
            for device in self._launchers.devices()
        ]
    }

  def inspect(self, device_id: str) -> dict:
    """Checks a device by running a runner on it and asking who it is.

    A device held by a Model Server is answered from that Model Server, so
    the check never disturbs it.

    Returns:
      The UI's Partial<RunnerDevice>.
    """
    available = {device.id: device for device in self._launchers.devices()}
    device = available.get(device_id)
    if device is None:
      raise errors.NotFound(f'No device {device_id} is connected.')
    with self._lock:
      if not device.shared:
        holder = self._holder(device.id)
        if holder is not None:
          execution, side = holder
          return self._busy(device, execution, side)
        if device.id in self._probing:
          return {
              'id': device.id,
              'status': 'busy',
              'runnerState': 'unknown',
              'checkedAt': _now(),
              'reason': 'Another check of this device is running.',
          }
        self._probing.add(device.id)
    try:
      info = self._probe(device)
    except Exception as error:
      return {
          'id': device.id,
          'status': 'unreachable',
          'runnerState': 'unknown',
          'checkedAt': _now(),
          'reason': str(error),
      }
    finally:
      with self._lock:
        self._probing.discard(device.id)
    try:
      _check_info(info)
    except RuntimeError as error:
      return {
          'id': device.id,
          'status': 'unsupported',
          'runnerState': 'unknown',
          'checkedAt': _now(),
          'reason': str(error),
      }
    with self._lock:
      self._infos[device.id] = info
    return {
        'id': device.id,
        'status': 'ready',
        'runnerState': 'idle',
        'checkedAt': _now(),
        'runner': self._runner_view(device, info),
    }

  def _probe(self, device: launchers_lib.Device) -> runner_pb2.RunnerInfo:
    launched = self._launchers.launch(device.id)
    try:
      client = runner_client.RunnerClient(launched.address)
      try:
        return client.get_info()
      finally:
        client.close()
    finally:
      launched.stop()

  def _busy(
      self, device: launchers_lib.Device, execution: _Execution, side: _Side
  ) -> dict:
    try:
      owner = self._sessions.get(execution.owner_id)
      session_name = owner.get('name', '')
    except errors.NotFound:
      session_name = ''
    result = {
        'id': device.id,
        'status': 'busy',
        'runnerState': 'active',
        'checkedAt': _now(),
        'runner': {
            'owner': {
                'sessionId': execution.owner_id,
                'sessionName': session_name,
                'runId': side.role,
                'isCurrentServer': True,
            }
        },
    }
    info = side.info or self._infos.get(device.id)
    if info is not None:
      result['runner'].update(self._runner_view(device, info))
    return result

  def _runner_view(
      self, device: launchers_lib.Device, info: runner_pb2.RunnerInfo
  ) -> dict:
    backends = _backends(device, info)
    version = info.runtime.version
    host = info.host
    environment = {
        'modelIdentifier': host.model,
        'architecture': host.architecture,
        'physicalMemoryBytes': host.memory_bytes,
    }
    if host.HasField('chip'):
      environment['chip'] = host.chip
    return {
        'buildId': version,
        'build': {
            'id': version,
            'label': f'LiteRT-LM {version[:12]}',
            'available': True,
        },
        'protocolVersion': info.protocol_version,
        'runtimes': [{
            'id': 'LiteRT-LM',
            'backends': backends,
            'transport': device.transport,
        }],
        'capabilities': {
            'runtime': 'LiteRT-LM',
            'backends': backends,
            'engineOptions': [
                key
                for key in options.supported_options(info)
                if key not in ('backend', 'contextLength')
            ],
            'conversationOptions': [
                key
                for key in options.chat_option_keys(info)
                if key != 'maxOutputTokens'
            ],
            'samplerTypes': list(options.SAMPLER_TYPES),
        },
        'environment': environment,
    }

  def capabilities(self) -> dict:
    """The UI's RuntimeCapabilities."""
    devices = self._launchers.devices()
    with self._lock:
      # Each connected device that answered, judged by its own answer;
      # this computer comes first in the launchers' list.
      answered = [
          (d, self._infos[d.id]) for d in devices if d.id in self._infos
      ]
    local = next((d for d in devices if d.id == 'local'), None)
    available = bool(devices)
    reason = (
        ''
        if available
        else (
            'No device can run a runner: build the command-line runner, '
            'or connect a device with the Runner App.'
        )
    )
    # Every runner offers the CPU; a GPU when a device that answered has one.
    offered = {
        backend
        for device, info in answered
        for backend in _backends(device, info)
    }
    backends = ['CPU', 'GPU'] if 'GPU' in offered else ['CPU']
    if answered:
      info = answered[0][1]
      supported = options.supported_options(info)
      chat = options.chat_option_keys(info)
    else:
      # Nothing connected has answered yet: only the fixed options are known.
      supported, chat = ['backend', 'contextLength'], []
    runtime = {
        'id': 'LiteRT-LM',
        'available': available,
        'reason': reason,
        'backends': backends,
        'supported_options': supported,
        'chat_options': chat,
        'sampler_types': list(options.SAMPLER_TYPES),
        'max_output_tokens': options.MAX_OUTPUT_TOKENS,
        'prompt_limit_bytes': PROMPT_LIMIT_BYTES,
    }
    result = {
        'available': available,
        'reason': reason,
        'models': [
            {
                'artifact': model.artifact,
                'name': model.name,
                'runtime': 'LiteRT-LM',
            }
            for model in self._models.list()
        ],
        'backends': backends,
        'supported_options': supported,
        'runtimes': [runtime],
        'tap_profiles': [],
        'upload_limit_bytes': models_lib.UPLOAD_LIMIT_BYTES,
        'upload_idle_timeout_seconds': models_lib.UPLOAD_IDLE_TIMEOUT_SECONDS,
    }
    if local is not None:
      result['platform'] = local.platform
    return result

  # ---- Captures.

  def capture(self, record_id: str) -> capture_index.CaptureIndex:
    """The stored capture of a chat record.

    Raises:
      NotFound: no such record, or it has no capture.
    """
    self._sessions.get(record_id)
    with self._lock:
      index = self._indexes.get(record_id)
    if index is None:
      index = capture_index.CaptureIndex.load(self._capture_root(record_id))
      with self._lock:
        index = self._indexes.setdefault(record_id, index)
    return index

  def _capture_root(self, record_id: str) -> pathlib.Path:
    return self._workspace.captures / record_id
