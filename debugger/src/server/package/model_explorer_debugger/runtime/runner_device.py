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

"""Shared native Runner Session lifecycle, independent of device transport."""

from collections.abc import Callable, Mapping
import json
import pathlib
import subprocess
import sys
import threading
from typing import Any, Protocol, runtime_checkable
import uuid

from model_debugger_contracts import errors
from model_debugger_contracts import schema
from model_explorer_debugger import fsutil
from model_explorer_debugger.runtime import litert_lm_adapter
from model_explorer_debugger.runtime import protocol
from model_explorer_debugger.runtime import runner_channel
from model_explorer_debugger.runtime import turn_outcome


def validate_native_configuration(run):
  """Validates backend, thread count, and context length for native Runners."""
  if run.get('backend') not in ('', None, *protocol.NATIVE_BACKENDS) or run.get(
      'cpuThreads'
  ):
    raise ValueError(
        'The native Runner supports CPU or GPU with default thread settings'
    )
  if (
      str(run.get('device', '')).startswith('ios:')
      and run.get('backend') == 'GPU'
  ):
    raise ValueError('This iOS Runner build does not support GPU execution')
  if str(run.get('contextLength') or 1024) not in tuple(
      map(str, protocol.NATIVE_CONTEXT_LENGTHS)
  ):
    raise ValueError('The native Runner supports context lengths 1024 and 4096')


def validate_native_request(request):
  """Validates generation options, prompt size, and capture manifest."""
  validate_native_configuration(request['run'])
  generation = request.get('generation') or {}
  defaults = {
      'temperature': 0,
      'topK': 1,
      'topP': 1,
      'seed': 0,
      'thinking': 'off',
      'systemPrompt': '',
      'thinkingBudget': -1,
  }
  for key, value in generation.items():
    if key != 'maxOutputTokens' and (
        key not in defaults or value != defaults[key]
    ):
      raise ValueError(
          f'native Runner generation option {key} is not supported at this'
          ' value'
      )
  limit = generation.get(
      'maxOutputTokens', request.get('max_output_tokens', 32)
  )
  if (
      isinstance(limit, bool)
      or not isinstance(limit, int)
      or not 1 <= limit <= protocol.NATIVE_MAX_OUTPUT_TOKENS
  ):
    raise ValueError(
        f'The native Runner supports 1–{protocol.NATIVE_MAX_OUTPUT_TOKENS}'
        ' output tokens'
    )
  if (
      request['operation'] in ('generate', 'preflight')
      and not request.get('prompt', '').strip()
  ):
    raise ValueError('Enter a non-empty prompt')
  if (
      len(request.get('prompt', '').encode('utf-8'))
      > protocol.NATIVE_MAX_PROMPT_BYTES
  ):
    raise ValueError(
        'Prompt exceeds the native Runner'
        f' {protocol.NATIVE_MAX_PROMPT_BYTES}-byte transport limit'
    )
  if not request.get('manifest'):
    raise ValueError(
        'Prepare verified capture points before running on native Runner'
    )


class RunnerDevices:
  """Base transport adapter managing per-slot native Runner session channels."""

  DEVICE_PREFIXES: tuple[str, ...] = ()
  SUPPORTED_RUNTIMES: frozenset[str] = frozenset({'LiteRT-LM'})
  SUPPORTS_MULTI_SLOT: bool = False

  def __init__(self, root):
    self.root = pathlib.Path(root)
    self.connections: dict[str, Any] = {}
    self.stopping = False
    self.sessions: dict[tuple[str, str, str], dict[str, Any]] = {}
    self.owners: dict[str, dict[str, Any]] = {}
    self.lock = threading.RLock()
    self.on_lifecycle: Callable[..., None] | None = None
    self.lost_executions: set[tuple[str, str]] = set()

  def matches_device(self, device: str) -> bool:
    """Returns True if the device ID belongs to this adapter's prefix set."""
    return str(device or '').startswith(self.DEVICE_PREFIXES)

  def supports_run(self, run: Mapping[str, Any]) -> bool:
    """Returns True if both the device prefix and runtime are supported."""
    return (
        self.matches_device(str(run.get('device', '')))
        and str(run.get('runtime', '')) in self.SUPPORTED_RUNTIMES
    )

  def physical_device_key(self, resolved_device: str) -> str:
    """Returns a canonical physical-host ID for single-occupancy checks."""
    return resolved_device

  def close(self):
    """Closes all open transport connections and clears active sessions."""
    self.stopping = True
    for connection in list(self.connections.values()):
      connection.close()
    self.connections.clear()
    self.sessions.clear()

  def _session_command(self, identity, kind):
    devices = {key[2] for key in self.sessions if key[0] == identity}
    devices.update(
        device
        for device, owner in self.owners.items()
        if owner['sessionId'] == identity
    )
    failures = []
    for device in devices:
      connection = self.connections.get(device)
      try:
        if connection:
          if kind == 'close':
            connection._expected_close = True
          runner_channel.call(
              connection,
              kind,
              timeout=30,
              reply={'close': 'closed', 'reset': 'reset'}[kind],
              failure='Runner did not acknowledge Session ' + kind,
              sessionId=identity,
          )
      except Exception as error:
        failures.append(str(error))
      finally:
        if kind == 'close':
          if connection and hasattr(connection, 'close'):
            connection.close()
          self.connections.pop(device, None)
          self.owners.pop(device, None)
        for key in list(self.sessions):
          if key[0] == identity and key[2] == device:
            self.sessions.pop(key)
    if failures:
      raise ConnectionError('; '.join(failures))

  def abandon_session(self, identity):
    """Fence an unresponsive request.

    No second response consumer is started.
    """
    devices = {key[2] for key in self.sessions if key[0] == identity}
    devices.update(
        device
        for device, owner in self.owners.items()
        if owner['sessionId'] == identity
    )
    for device in devices:
      connection = self.connections.pop(device, None)
      if connection:
        connection.close()
      self.owners.pop(device, None)
    for key in list(self.sessions):
      if key[0] == identity:
        self.sessions.pop(key, None)

  def close_session(self, identity):
    self._session_command(identity, 'close')

  def reset_session(self, identity):
    self._session_command(identity, 'reset')

  def check_owner(self, identity, request):
    owner = request.get('owner')
    existing = self.owners.get(identity)
    if (
        existing
        and owner
        and any(
            existing.get(key) != owner.get(key)
            for key in ('serverId', 'sessionId', 'runId')
        )
    ):
      raise ValueError(
          'Device is occupied by another Session or execution side'
      )

  @staticmethod
  def device_id(value: str) -> str:
    return str(value)

  def connect(self, identity: str) -> Any:
    raise NotImplementedError

  def connection_key(self, request: Mapping[str, Any]) -> str:
    return self.device_id(request['run']['device'])

  @staticmethod
  def validate_connection_capabilities(connection, run):
    if run.get('runtime', 'LiteRT-LM') != 'LiteRT-LM':
      return
    hello = getattr(connection, 'hello', {})
    capabilities = (
        hello.get('capabilities', {}) if isinstance(hello, dict) else {}
    )
    if not isinstance(capabilities, dict):
      raise ValueError('Runner advertised invalid native capabilities')
    backend = run.get('backend') or 'CPU'
    context = int(run.get('contextLength') or 1024)
    if backend not in capabilities.get(
        'backends', ['CPU']
    ) or context not in capabilities.get('contextLengths', [1024]):
      raise ValueError(
          f'This Runner build does not support {backend} with context length'
          f' {context}; select the matching rebuilt Runner'
      )

  def acquire(self, identity, request):
    self.check_owner(identity, request)
    owner = request.get('owner')
    execution = (identity, request.get('execution_id', request['session_id']))
    if execution in self.lost_executions:
      raise ConnectionError(
          'Runner execution was lost; reconnection is disabled'
      )
    connection = self.reusable_connection(
        identity, initializing=request['operation'] == 'initialize'
    )
    if connection is None:
      if request['operation'] != 'initialize':
        raise ValueError('Runner disconnected. Start Session again.')
      connection = self.connect(identity)
      self.connections[identity] = connection
    try:
      self.validate_connection_capabilities(connection, request['run'])
    except Exception:
      connection.close()
      self.connections.pop(identity, None)
      raise
    if owner and identity not in self.owners:
      try:
        connection.on_lifecycle = (
            lambda kind, fields: self.on_lifecycle
            and self.on_lifecycle(
                request['session_id'],
                request['run']['id'],
                kind,
                dict(fields, executionId=request.get('execution_id')),
            )
        )
        activation = dict(owner)
        if request['run'].get('runnerBuild'):
          activation['buildId'] = request['run']['runnerBuild']
        acknowledgement = connection.activate_owner(activation)
        acknowledgement['deviceId'] = request['run']['device']
        acknowledgement['executionId'] = request.get('execution_id')
        self.owners[identity] = dict(
            owner,
            executionId=request.get('execution_id', request['session_id']),
        )
        if self.on_lifecycle:
          self.on_lifecycle(
              request['session_id'],
              request['run']['id'],
              'activated',
              acknowledgement,
          )
      except BaseException:
        connection.close()
        self.connections.pop(identity, None)
        raise
    return connection

  def _forget_connection(self, identity, request):
    self.lost_executions.add(
        (identity, request.get('execution_id', request['session_id']))
    )
    connection = self.connections.pop(identity, None)
    if connection:
      connection.close()
    self.owners.pop(identity, None)
    for key in list(self.sessions):
      if key[2] == identity:
        self.sessions.pop(key, None)

  def execute(self, request, emit, cancelled):
    """One operation on one side, from job to Conversation.

    Order: job → connection → terminal event → text → dump → Conversation.
    """
    self.validate(request)
    device = self.connection_key(request)
    self.check_owner(device, request)
    key = request['session_id'], request['run']['id'], device
    initializing = request['operation'] == 'initialize'
    connection = self.reusable_connection(device, initializing=initializing)
    resident = self.sessions.get(key)
    config, job = self._native_job(request, resident)
    directory = pathlib.Path(request['output'])
    result = None
    try:
      if connection is None and not initializing:
        raise ConnectionError('Runner disconnected; reconnection is disabled')
      if initializing:
        connection = self.acquire(device, request)
        connection.ensure_model(
            pathlib.Path(request['model']),
            job['modelSHA256'],
            emit,
            cancelled=lambda: self.stopping,
        )
        resident = self.sessions[key] = {
            'config': config,
            'manifest': job['manifest'],
            'messages': list(job['messages']),
        }
      elif job['messages']:
        raise ValueError('Generate accepts only a new message')
      terminal = self._native_operation(
          connection, request, job, emit, cancelled
      )
      if terminal['type'] == 'preflight':
        return terminal
      status, output, phone, terminal_error = self._terminal_text(
          terminal, job, None if initializing else resident
      )
      result = dict(
          turn_outcome.text(status, output),
          input=request.get('prompt', ''),
          model_sha256=job['modelSHA256'],
          device=request['run']['device'],
          transport=self.transport,
          runner_result=phone,
          trace_log_path=None,
          dump_complete=False,
          connection_lost=False,
          debug_data={'status': 'unavailable', 'error': ''},
      )
      if result['output_confirmed'] and not initializing:
        emit(
            'generation_completed',
            output=output,
            output_confirmed=True,
            generation_status='completed',
        )
      result.update(
          self._collect_dump(
              connection, terminal, job['id'], directory, phone, emit
          )
      )
      if status != 'completed':
        result['error'] = terminal_error or terminal.get(
            'error', 'Generation ' + status
        )
        self.sessions.pop(key, None)
        if initializing:
          raise turn_outcome.RunnerRefused(result['error'])
      else:
        try:
          result.update(
              self._record_turn(
                  resident, phone, job, request, output, initializing
              )
          )
        except Exception as error:
          if initializing:
            raise
          result.update(turn_outcome.debug(error), import_error=str(error))
      if result['connection_lost']:
        self._forget_connection(device, request)
      if hasattr(connection, 'transport_info'):
        result['transport_details'] = connection.transport_info
      (directory / 'result.json').write_text(json.dumps(result))
      if initializing:
        emit('initialized', debug_enabled=True, verified_taps=True)
      return result
    except errors.InputRejected:
      raise
    except BaseException as error:
      link_open = connection is not None and getattr(
          connection, 'is_open', True
      )
      if isinstance(error, turn_outcome.RunnerRefused) and link_open:
        self.sessions.pop(key, None)
        raise
      if result is not None and result['output_confirmed']:
        result.update(turn_outcome.dump_failed(error, lost=not link_open))
        if not link_open:
          self._forget_connection(device, request)
        return result
      self._forget_connection(device, request)
      if request['operation'] == 'generate':
        return turn_outcome.unconfirmed(error, input=request.get('prompt', ''))
      raise

  def _native_job(self, request, resident):
    """The capture-job for this request.

    It is checked against the model and the resident Chat.
    """
    initializing = request['operation'] == 'initialize'
    config = {
        k: request.get(k)
        for k in ('model', 'model_sha256', 'manifest', 'generation', 'run')
    }
    if not initializing and (resident is None or resident['config'] != config):
      raise ValueError('Runner Chat is closed or configuration changed')
    pathlib.Path(request['output']).mkdir(parents=True, exist_ok=True)
    model = pathlib.Path(request['model'])
    if initializing and fsutil.file_digest(model) != request['model_sha256']:
      raise ValueError('Registered model changed')
    runtime_src = str(pathlib.Path(request['runtime_root']) / 'src')
    if runtime_src not in sys.path:
      sys.path.insert(0, runtime_src)
    manifest = (
        litert_lm_adapter.verify_taps(model, request['manifest'])
        if initializing
        else resident['manifest']
    )
    if not 1 <= len(manifest['taps']) <= protocol.NATIVE_MAX_CAPTURE_POINTS:
      raise ValueError(
          f'The Runner supports 1–{protocol.NATIVE_MAX_CAPTURE_POINTS} capture'
          ' points'
      )
    generation = request.get('generation') or {}
    job = dict(
        formatVersion=1,
        id=str(uuid.uuid4()),
        modelSHA256=request['model_sha256'],
        prompt=request.get('prompt') or 'Initialize.',
        backend=request['run'].get('backend') or 'CPU',
        contextLength=int(request['run'].get('contextLength') or 1024),
        maxOutputTokens=generation.get(
            'maxOutputTokens', request.get('max_output_tokens', 32)
        ),
        manifest=manifest,
        messages=request.get('messages') or [],
    )
    if request['operation'] == 'generate' and 'turn_sequence' in resident:
      # The Runner checks this before the message touches its KV cache;
      # _terminal_text checks the result again.
      job['expect'] = dict(
          runtimeInstance=str(resident['runtime_instance']),
          turnSequence=resident['turn_sequence'],
          tokenCount=resident['token_count'],
      )
    schema.validate('capture-job', job)
    return config, job

  def _native_operation(self, connection, request, job, emit, cancelled):
    """Run the operation to its terminal event, forwarding stream events.

    The job ID is the request ID.
    """
    operation = request['operation']
    message = {
        'type': operation,
        'requestId': job['id'],
        'sessionId': request['session_id'],
        'runId': request['run']['id'],
        'chatId': request.get('chat_id'),
        'job': job,
    }
    stop = (
        (lambda: self.stopping or cancelled())
        if operation == 'generate'
        else None
    )
    for event in runner_channel.request(connection, message, cancelled=stop):
      kind = event.get('type')
      if operation == 'preflight' and kind in ('preflight', 'error'):
        if (
            event.get('errorCode') == errors.InputRejected.code
            or kind == 'preflight'
            and event.get('accepted') is False
        ):
          raise errors.InputRejected(
              event.get(
                  'error',
                  event.get('reason', 'Input exceeds available context'),
              )
          )
        if kind == 'error':
          raise turn_outcome.RunnerRefused(
              event.get('error', 'Runner capacity check failed')
          )
        return event
      if kind in ('completed', 'error'):
        if operation == 'initialize' and kind == 'error':
          raise turn_outcome.RunnerRefused(
              event.get('error', 'Runner initialization failed')
          )
        return event
      if kind == 'delta':
        emit('delta', text=event['text'])
      elif kind == 'progress':
        emit('progress', message=event['message'])

  def _terminal_text(self, terminal, job, resident):
    """Status, text, Runner result and error of a terminal event.

    Text counts as completed only with verified provenance and, within a Chat,
    an unbroken Conversation: the same runtime, model load, token count and next
    turn.
    """
    status = {
        'succeeded': 'completed',
        'stopped': 'stopped',
        'failed': 'failed',
    }.get(
        terminal.get('generationStatus'),
        'completed' if terminal['type'] == 'completed' else 'failed',
    )
    phone = terminal.get('result') or {}
    output = terminal.get('output', phone.get('output'))
    if status != 'completed':
      return status, output, phone, ''
    try:
      self._validate_result(phone, job['id'], job)
      if not isinstance(output, str) or phone.get('output') != output:
        raise ValueError('Runner terminal output is inconsistent')
      if resident is not None and (
          phone['runtimeInstance'] != resident['runtime_instance']
          or phone['modelLoadCount'] != resident.get('model_load_count', 1)
          or phone['tokenCountBefore'] != resident['token_count']
          or phone['turnSequence'] != resident['turn_sequence'] + 1
      ):
        raise ValueError(
            'Runner did not preserve the initialized Conversation/KV cache'
        )
    except Exception as error:
      return 'failed', output, phone, str(error)
    return status, output, phone, ''

  def _collect_dump(
      self, connection, terminal, identity, directory, phone, emit
  ):
    """Result fields for this turn's dump.

    A failure here never touches settled text.
    """
    expected = 'Documents/Runs/' + str(uuid.UUID(identity)).upper()
    raw = directory / self.capture_folder
    fields = {}
    try:
      if (
          terminal.get('runDirectory') != expected
          or terminal.get('dumpStatus') == 'unavailable'
      ):
        raise ValueError(
            terminal.get('dumpError', 'Runner dump is unavailable')
        )
      emit('progress', message='Retrieving and validating Runner dump…')
      fields.update(
          connection.copy_from(expected, raw) or {},
          dump_complete=True,
          raw_capture=str(raw),
      )
      for filename, field in (
          ('generated_tokens.jsonl', 'trace_log_path'),
          ('runtime_trace.jsonl', 'native_trace_path'),
      ):
        trace = raw / 'raw' / filename
        if trace.is_file():
          fields[field] = str(trace.relative_to(directory))
      if not phone and (raw / 'result.json').is_file():
        phone = fields['runner_result'] = json.loads(
            (raw / 'result.json').read_text()
        )
      if phone and phone.get('runtimeTracePath') is not None:
        if (
            phone['runtimeTracePath'] != 'raw/runtime_trace.jsonl'
            or not (raw / phone['runtimeTracePath']).is_file()
        ):
          raise ValueError('Runner runtime trace path is missing or invalid')
      fields.update(turn_outcome.debug())
    except Exception as error:
      lost = not getattr(connection, 'is_open', True) or isinstance(
          error, (ConnectionError, TimeoutError)
      )
      fields.update(turn_outcome.dump_failed(error, lost=lost))
    return fields

  def _record_turn(self, resident, phone, job, request, output, initializing):
    """Advance the resident Conversation past a verified turn.

    Returns the turn's result fields.
    """
    fields = dict(
        runtime_instance=phone['runtimeInstance'],
        model_load_count=phone['modelLoadCount'],
        token_count_before=phone['tokenCountBefore'],
        token_count=phone['tokenCount'],
        turn_sequence=phone['turnSequence'],
        backend_requested=job['backend'],
        backend_effective=phone['backend'],
        backend_evidence=phone.get('backendEvidence'),
        debug_enabled=True,
        verified_taps=True,
        captured_tensors=len(phone['tensors']),
        elapsed_seconds_with_capture=phone['elapsedSecondsWithCapture'],
    )
    resident.update(
        runtime_instance=phone['runtimeInstance'],
        model_load_count=phone['modelLoadCount'],
        token_count=phone['tokenCount'],
        turn_sequence=phone['turnSequence'],
    )
    fields['messages'] = list(resident['messages'])
    if not initializing:
      resident['messages'].extend([
          {
              'role': 'user',
              'content': [{'type': 'text', 'text': request['prompt']}],
          },
          {'role': 'assistant', 'content': [{'type': 'text', 'text': output}]},
      ])
    if phone.get('environment') is not None:
      fields['environment'] = phone['environment']
    if self.platform == 'iOS':
      fields['phone_result'] = phone
    return fields

  def _validate_result(self, phone, identity, job):
    if (
        str(phone['jobID']).lower() != identity
        or phone['modelSHA256'] != job['modelSHA256']
        or phone['platform'] != self.platform
        or phone['backend'] != job.get('backend', 'CPU')
        or not phone['debuggerEnabled']
        or phone['input'] != job['prompt']
        or phone['contextLength'] != job['contextLength']
        or phone['maxOutputTokens'] != job['maxOutputTokens']
        or phone['thinkingEnabled']
        or phone['speculativeDecodingEnabled']
        or phone['sampler']
        != {'seed': 0, 'temperature': 0, 'topK': 1, 'topP': 1}
    ):
      raise ValueError('Runner result identity/provenance mismatch')
    evidence = phone.get('backendEvidence')
    if job.get('backend') == 'GPU' or evidence is not None:
      if (
          not isinstance(evidence, dict)
          or evidence.get('requestedBackend') != job.get('backend', 'CPU')
          or evidence.get('effectiveBackend') != phone['backend']
          or evidence.get('engineInitialized') is not True
      ):
        raise ValueError(
            'Runner backend initialization evidence is missing or inconsistent'
        )

  def import_result(self, request, result):
    if (
        result.get('generation_status') != 'completed'
        or not result.get('dump_complete')
        or result.get('import_error')
    ):
      return result
    importer = pathlib.Path(__file__).with_name('import_capture.py')
    directory = pathlib.Path(request['output'])
    try:
      process = subprocess.run(
          [
              str(pathlib.Path(request['runtime_root']) / '.venv/bin/python'),
              str(importer),
              '--runtime-root',
              request['runtime_root'],
              '--capture',
              result['raw_capture'],
              '--model',
              request['model'],
              '--output',
              str(directory / 'export'),
          ],
          capture_output=True,
          text=True,
          timeout=120,
      )
      if process.returncode:
        raise ValueError(
            'Runner capture validation failed: ' + process.stderr[-2000:]
        )
    except Exception as error:
      result['import_error'] = str(error)
      result['debug_data'] = {'status': 'unavailable', 'error': str(error)}
    return result

  def reusable_connection(self, identity, *, initializing):
    connection = self.connections.get(identity)
    if connection is not None and not getattr(connection, 'is_open', True):
      owner = self.owners.get(identity, {})
      self.lost_executions.add(
          (identity, owner.get('executionId', owner.get('sessionId')))
      )
      connection.close()
      self.connections.pop(identity, None)
      self.owners.pop(identity, None)
      for key in list(self.sessions):
        if key[2] == identity:
          self.sessions.pop(key, None)
      raise ConnectionError('Runner disconnected; reconnection is disabled')
    return connection


@runtime_checkable
class RunnerDeviceAdapter(Protocol):
  """Structural protocol implemented by platform Runner device adapters."""

  DEVICE_PREFIXES: tuple[str, ...]
  SUPPORTED_RUNTIMES: frozenset[str]
  SUPPORTS_MULTI_SLOT: bool
  connections: dict[str, Any]
  lost_executions: set[tuple[str, str]]
  on_lifecycle: Callable[..., None] | None

  def matches_device(self, device: str) -> bool:
    ...

  def supports_run(self, run: Mapping[str, Any]) -> bool:
    ...

  def physical_device_key(self, resolved_device: str) -> str:
    ...

  def connection_key(self, request: Mapping[str, Any]) -> str:
    ...

  def connect(self, identity: str) -> Any:
    ...

  def execute(
      self,
      request: Mapping[str, Any],
      emit: Callable[..., Any],
      cancelled: Callable[[], bool] | None,
  ) -> dict[str, Any]:
    ...

  def import_result(
      self, request: Mapping[str, Any], result: dict[str, Any]
  ) -> dict[str, Any]:
    ...

  def close_session(self, identity: str) -> None:
    ...

  def abandon_session(self, identity: str) -> None:
    ...

  def reset_session(self, identity: str) -> None:
    ...

  def close(self) -> None:
    ...
