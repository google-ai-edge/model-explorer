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

"""Persistent local workspace, separate from immutable captured session data."""

import contextlib
import copy
import json
import pathlib
import platform
import threading
import uuid

from model_debugger_contracts import model_identity
from model_explorer_debugger import fsutil
from model_explorer_debugger import generation_config
from model_explorer_debugger import persistence
from model_explorer_debugger import session_rules
from model_explorer_debugger import store_cache
from model_explorer_debugger.runtime import ios_device
from model_explorer_debugger.runtime import macos_device
from model_explorer_debugger.runtime import protocol
from model_explorer_debugger.runtime import runner_device
from model_explorer_debugger.runtime import runners
from model_explorer_debugger.runtime import tap_profiles

__all__ = ['SessionRegistry']


class SessionRegistry:
  """Manages persistent workspace state, active sessions, and capabilities."""

  def __init__(
      self,
      root: pathlib.Path | str,
      runtime_root: pathlib.Path | str | None = None,
      pytorch_root: pathlib.Path | str | None = None,
      home_dir: pathlib.Path | str | None = None,
  ) -> None:
    self.root = pathlib.Path(root).resolve()
    self.root.mkdir(parents=True, exist_ok=True)
    self.runtime_root = None
    if runtime_root:
      self.runtime_root = pathlib.Path(runtime_root).resolve()
    self.pytorch_root = None
    if pytorch_root:
      self.pytorch_root = pathlib.Path(pytorch_root).resolve()
    self.home_dir = None
    if home_dir and str(home_dir).strip():
      self.home_dir = pathlib.Path(home_dir).resolve()
    self.lock = threading.RLock()
    self.execution_provider = None
    self.server_identity = None
    self.capture_cache = store_cache.StoreCache()
    self.path = self.root / 'workspace.json'
    self.database = persistence.WorkspaceDatabase(self.root)
    self._state = self.database.workspace()
    active_owners = {
        record.get('parent_session_id', record['id'])
        for record in self.state['sessions']
        if record.get('initialized')
        or record['status'] in ('initializing', 'running', 'finalizing')
    }
    for record in self.state['sessions']:
      was_active = (
          record.get('parent_session_id', record['id']) in active_owners
      )
      if record['status'] in (
          'preparing',
          'initializing',
          'running',
          'finalizing',
      ):
        record.update(
            status='failed',
            notice=(
                'Server stopped during the previous task; partial artifacts'
                ' were retained.'
            ),
        )
      record['execution_interrupted'] = was_active
      record['initialized'] = False
      if was_active:
        record.update(read_only=True, chat_state='unavailable')
    self._save()
    self.export_workspace()

  @property
  def state(self):
    """The in-memory workspace.

    Mutate records in place; replace it only through restore().
    """
    return self._state

  @state.setter
  def state(self, value):
    raise AttributeError(
        'Use SessionRegistry.restore(snapshot); registry.state is not'
        ' assignable'
    )

  def snapshot(self):
    """Returns a thread-safe deep copy of the in-memory workspace state."""
    with self.lock:
      return copy.deepcopy(self._state)

  def restore(self, snapshot):
    """Replaces the workspace in place so existing references stay current."""
    with self.lock:
      self._state.clear()
      self._state.update(snapshot)

  def _save(self):
    previous = self.database.workspace()
    try:
      with self.database.transaction():
        self.database.save_workspace(self._state)
    except BaseException:
      self.restore(previous)
      raise

  def export_workspace(self):
    """Writes workspace.json for local tooling while SQLite stays primary."""
    with self.lock:
      fsutil.atomic_json(self.path, self._state)

  def close(self):
    """Exports the workspace JSON and closes the SQLite database connection."""
    self.export_workspace()
    self.database.close()

  @contextlib.contextmanager
  def transaction(self):
    """Commit coupled workspace/job changes or restore the memory snapshot."""
    with self.lock:
      previous = self.snapshot()
      try:
        with self.database.transaction():
          yield
      except BaseException:
        self.restore(previous)
        raise

  def capabilities(self):
    """Returns supported runtimes, backends, and upload limits for the host."""
    root = self.runtime_root
    available = bool(
        platform.system() == 'Darwin'
        and platform.machine() == 'arm64'
        and root
        and (root / '.venv/bin/python').is_file()
        and (root / 'artifacts/python/litert_lm/liblitert-lm.dylib').is_file()
    )
    pytorch = dict(
        runners.local_python_capability(home_dir=self.home_dir),
        max_output_tokens=generation_config.MAX_OUTPUT_TOKENS,
    )
    litert_reason = ''
    if not available:
      litert_reason = (
          'Configure LiteRT model preparation with --runtime-root and open a'
          ' Runner App.'
      )
    litert = dict(
        id='LiteRT-LM',
        available=available,
        backends=list(protocol.NATIVE_BACKENDS),
        precisions=[''],
        supported_options=['backend', 'contextLength'],
        max_output_tokens=protocol.NATIVE_MAX_OUTPUT_TOKENS,
        prompt_limit_bytes=protocol.NATIVE_MAX_PROMPT_BYTES,
        reason=litert_reason,
    )
    runtimes = [litert, pytorch]
    any_available = any(
        runtime_entry['available'] for runtime_entry in runtimes
    )
    overall_reason = ''
    if not any_available:
      overall_reason = 'Configure a Runner and model preparation tools.'
    profiles = tap_profiles.available_profiles(root) if available else []
    return dict(
        available=any_available,
        platform=platform.system(),
        backends=['CPU'],
        upload_limit_bytes=session_rules.UPLOAD_LIMIT_BYTES,
        upload_idle_timeout_seconds=session_rules.UPLOAD_IDLE_TIMEOUT_SECONDS,
        supported_options=['backend', 'cpuThreads', 'contextLength'],
        reason=overall_reason,
        runtimes=copy.deepcopy(runtimes),
        capture=(
            'LiteRT-LM verified taps; PyTorch layer 0 modules across Prefill'
            ' and Decode, model boundaries, KV snapshots and token events'
        ),
        tap_profiles=profiles,
        models=[
            {
                'artifact': k,
                'name': v['name'],
                'runtime': v.get('runtime', 'LiteRT-LM'),
            }
            for k, v in self.state['artifacts'].items()
        ],
    )

  def require_runtime(self, run):
    """Validates that the requested runtime and backend are available."""
    capability = next(
        (
            runtime_entry
            for runtime_entry in self.capabilities()['runtimes']
            if runtime_entry['id'] == run['runtime']
        ),
        None,
    )
    if not capability or not capability['available']:
      raise ValueError(
          capability['reason'] if capability else 'Unsupported runtime'
      )
    if run.get('backend') and run['backend'] not in capability['backends']:
      raise ValueError(
          f"{run['backend']} is unavailable in the configured {run['runtime']}"
          ' runtime'
      )

  def worker_command(self, run, operation, request_path):
    """Builds the command line for the LiteRT-LM tap preparation worker."""
    if run['runtime'] != 'LiteRT-LM' or operation != 'prepare':
      raise ValueError(
          'Model execution is owned by Runner; server workers only prepare taps'
      )
    return [
        str(self.runtime_root / '.venv/bin/python'),
        '-m',
        'model_explorer_debugger.runtime.prepare_tap',
        str(request_path),
    ]

  def listing(self):
    """Returns the session listing, configuration template, and capabilities."""
    with self.lock:
      caps = self.capabilities()
      if (
          caps['runtimes'][0]['available']
          or not caps['runtimes'][1]['available']
      ):
        runtime = 'LiteRT-LM'
      else:
        runtime = 'PyTorch'
      backends = ['CPU', 'CPU']
      if runtime == 'PyTorch':
        model = next(
            (
                model_entry['name']
                for model_entry in caps['models']
                if model_entry['runtime'] == runtime
            ),
            'Local model',
        )
      else:
        model = 'Gemma 4 E2B'
      template = dict(
          id='',
          name='New session',
          created_at=None,
          model=model,
          status='draft',
          has_capture=False,
          notice='',
          runs=[
              dict(
                  id=k,
                  device='local',
                  runtime=runtime,
                  backend=b,
                  artifact='',
                  source='registered' if runtime == 'PyTorch' else 'upload',
              )
              for k, b in zip(('ref', 'target'), backends)
          ],
      )
      return copy.deepcopy(
          dict(
              server=self.server_identity,
              sessions=[
                  self._with_execution(s)
                  for s in self.state['sessions']
                  if not s.get('deleted') and not s.get('pending_chat')
              ],
              configuration_template=template,
              capabilities=dict(
                  create=True, duplicate=True, delete=True, rename=True
              ),
              unavailable_reason='',
              generation_available=self.capabilities()['available'],
          )
      )

  def _with_execution(self, record):
    value = copy.deepcopy(record)
    if self.execution_provider:
      value['execution'] = self.execution_provider(
          record.get('parent_session_id', record['id'])
      )
      state = value['execution']
      value['session_initialized'] = state['phase'] == 'active'
      value['initialized'] = (
          state['phase'] == 'active'
          and state.get('activeChatId') == record['id']
          and not record.get('read_only')
          and not record.get('pending_chat')
      )
      value['read_only'] = bool(
          record.get('read_only') or not value['initialized']
      )
    else:
      value['execution'] = {'phase': 'inactive', 'runners': []}
    return value

  def get(self, identity):
    """Returns the active session record matching identity with execution."""
    with self.lock:
      record = next(
          (
              s
              for s in self.state['sessions']
              if s['id'] == identity and not s.get('deleted')
          ),
          None,
      )
      if not record:
        raise ValueError('Unknown session')
      return self._with_execution(record)

  def update(self, identity, **fields):
    """Updates fields on a session record and persists the workspace state."""
    with self.lock:
      record = next(s for s in self.state['sessions'] if s['id'] == identity)
      previous = copy.deepcopy(record)
      record.update(fields)
      try:
        self._save()
      except BaseException:
        # A failed publication must not advance the in-memory capture
        # pointer beyond the last durable session snapshot.
        record.clear()
        record.update(previous)
        raise
      return self._with_execution(record)

  def update_family(self, identity, **fields):
    """Persist a Session-wide Chat transition in one workspace snapshot."""
    with self.lock:
      previous = copy.deepcopy(self.state['sessions'])
      for record in self.state['sessions']:
        if record.get('parent_session_id', record['id']) == identity:
          record.update(fields)
      try:
        self._save()
      except BaseException:
        self.state['sessions'] = previous
        raise

  def stage_chat(self, payload):
    """Keep initialization candidates addressable by job, outside Chat list."""
    with self.lock:
      source = self.get(payload['id'])
      generation = generation_config.generation_config(
          payload.get(
              'generation', source.get('generation', {'ref': {}, 'target': {}})
          )
      )
      name = session_rules.validate_name(payload.get('name', 'New Chat'))
      record = {
          **copy.deepcopy(source),
          'id': str(uuid.uuid4()),
          'parent_session_id': source.get('parent_session_id', source['id']),
          'name': name,
          'generation': generation,
          'created_at': fsutil.now(),
          'status': 'draft',
          'has_capture': False,
          'initialized': False,
          'pending_chat': True,
          'read_only': True,
          'chat_state': 'candidate',
          'successful_turns': [],
      }
      for key in (
          'capture',
          'job_id',
          'initialization',
          'execution',
          'session_initialized',
      ):
        record.pop(key, None)
      self.state['sessions'].append(record)
      try:
        self._save()
      except BaseException:
        self.state['sessions'].remove(record)
        raise
      return self._with_execution(record)

  def accept_chat(self, identity, previous_chat, job_id):
    """Transitions the previous chat to ended and marks staged chat active."""
    with self.lock:
      previous = copy.deepcopy(self.state['sessions'])
      for record in self.state['sessions']:
        if record['id'] == previous_chat:
          record.update(initialized=False, read_only=True, chat_state='ended')
        if record['id'] == identity:
          record.update(
              status='initializing', chat_state='initializing', job_id=job_id
          )
      try:
        self._save()
      except BaseException:
        self.state['sessions'] = previous
        raise

  def register_model(self, path, manifest=None, semantic=None):
    """Registers a local .litertlm model and returns its opaque artifact ID."""
    path = pathlib.Path(path).resolve()
    if path.suffix != '.litertlm' or not path.is_file():
      raise ValueError('Choose an existing .litertlm model')
    checksum = fsutil.file_digest(path)
    identity = 'model-' + checksum[:24]
    with self.lock:
      existing = self.state['artifacts'].get(identity, {})
      resolved_manifest = existing.get('manifest')
      if manifest:
        resolved_manifest = str(pathlib.Path(manifest).resolve())
      resolved_semantic = existing.get('semantic')
      if semantic:
        resolved_semantic = str(pathlib.Path(semantic).resolve())
      self.state['artifacts'][identity] = dict(
          path=str(path),
          name=path.name,
          sha256=checksum,
          runtime='LiteRT-LM',
          manifest=resolved_manifest,
          semantic=resolved_semantic,
      )
      self._save()
    return identity

  def register_pytorch_model(self, path):
    """Register a trusted local HF directory; no browser paths or downloads."""
    path = pathlib.Path(path).resolve()
    files, checksum = model_identity.describe_model(path)
    if path.parent.name == 'snapshots':
      name = path.parent.parent.name.removeprefix('models--').replace('--', '/')
    else:
      name = path.name
    identity = 'pytorch-' + checksum[:24]
    with self.lock:
      self.state['artifacts'][identity] = dict(
          path=str(path),
          name=name,
          sha256=checksum,
          model_files=files,
          runtime='PyTorch',
          manifest=None,
          semantic=None,
      )
      self._save()
    return identity

  def upload(self, stream, size, name):
    """Saves an uploaded .litertlm stream and registers the artifact."""
    if not name.endswith('.litertlm'):
      raise ValueError('Choose a .litertlm model')
    result = session_rules.upload_artifact(self.root, stream, size, name)
    identity = self.register_model(self.root / result['artifact'])
    result['artifact'] = identity
    return result

  def resolve_run(self, run):
    """Validates a run configuration and returns its registered artifact."""
    runtime = run.get('runtime')
    if runtime not in ('LiteRT-LM', 'PyTorch'):
      raise ValueError('Choose LiteRT-LM or PyTorch')
    if run.get('source') == 'huggingface':
      raise ValueError(
          'Download the model locally first; Hugging Face downloads are not'
          ' connected'
      )
    for key in (
        'audioBackend',
        'audioCpuThreads',
        'visionBackend',
        'forceF32',
        'prefillBatchSizes',
    ):
      if run.get(key):
        raise ValueError(
            f'{key} is not supported by this adapter; reset it to Default'
        )
    if run.get('cpuThreads') and run.get('backend') in ('GPU', 'MPS', 'CUDA'):
      raise ValueError('CPU threads cannot be applied to the GPU backend')
    if runtime == 'PyTorch':
      if run.get('device', '').startswith('ios:'):
        raise ValueError('This iPhone Runner does not support PyTorch')
      if run.get('backend') not in (None, '', 'CPU', 'MPS', 'CUDA'):
        raise ValueError('PyTorch backend must be CPU, MPS or CUDA')
      if run.get('precision') not in (
          None,
          '',
          'default',
          'float32',
          'float16',
          'bfloat16',
      ):
        raise ValueError(
            'PyTorch precision must be Default, float32, float16 or bfloat16'
        )
    device = run.get('device') or 'local'
    if device.startswith(('ios:', 'macos:', 'ssh:')):
      if device.startswith(('macos:', 'ssh:')):
        macos_device.device_id(device)
      else:
        ios_device.device_id(device)
    elif device not in ('local', 'Server host'):
      raise ValueError('Select a Runner device')
    if runtime == 'LiteRT-LM':
      runner_device.validate_native_configuration(run)
    with self.lock:
      artifact = copy.deepcopy(self.state['artifacts'].get(run.get('artifact')))
    if not artifact:
      raise ValueError('Select a registered model')
    if artifact.get('runtime', 'LiteRT-LM') != runtime:
      raise ValueError('Selected model artifact does not match the runtime')
    artifact_path = pathlib.Path(artifact['path'])
    if runtime == 'PyTorch':
      artifact_exists = artifact_path.is_dir()
    else:
      artifact_exists = artifact_path.is_file()
    if not artifact_exists:
      raise ValueError('Registered model is no longer available')
    return artifact

  def resolve_taps(self, run, config):
    """Validates and resolves custom tap point selections for a run."""
    if run.get('runtime') == 'PyTorch':
      if config.get('tap_profile') and all(
          r['runtime'] == 'PyTorch' for r in config['runs']
      ):
        raise ValueError(
            'PyTorch captures eager execution automatically; clear the LiteRT'
            ' tap profile'
        )
      return None
    if config.get('tap_profile') != tap_profiles.CUSTOM_PROFILE:
      return None
    artifact = self.resolve_run(run)
    identity = run['artifact']
    ids = config.get('tap_points', {}).get(identity, [])
    path = self.root / 'tap-scans' / identity / 'result.json'
    if not ids or not path.is_file():
      raise ValueError(
          'Wait for scanning and select capture points for each model'
      )
    scan = json.loads(path.read_text())
    if (
        scan.get('status') != 'completed'
        or scan.get('model_sha256') != artifact['sha256']
    ):
      raise ValueError('A completed scan of this exact model is required')
    candidates = {p['id']: p for p in scan['points']}
    if any(i not in candidates or not candidates[i]['selectable'] for i in ids):
      raise ValueError('Selection includes an unavailable capture point')
    return {
        'section_sha256': scan['section_sha256'],
        'points': [candidates[i] for i in sorted(ids)],
        'reviewed_e2b': bool(scan.get('recommended')),
    }

  def manage(self, operation, payload):
    """Applies a session lifecycle mutation and persists the updated record."""
    with self.lock:
      if operation == 'create':
        record = {
            **session_rules.validate_config(payload),
            **tap_profiles.profile_config(payload),
            'id': str(uuid.uuid4()),
            'created_at': fsutil.now(),
            'status': 'draft',
            'has_capture': False,
            'initialized': False,
            'notice': 'Initialize the local runtime to start.',
        }
        for run in record['runs']:
          self.resolve_run(run)
          self.resolve_taps(run, record)
        self.state['sessions'].append(record)
      else:
        record = next(
            (s for s in self.state['sessions'] if s['id'] == payload.get('id')),
            None,
        )
        if not record or (record.get('deleted') and operation != 'restore'):
          raise ValueError('Unknown session')
        if record['status'] in (
            'preparing',
            'initializing',
            'running',
            'finalizing',
        ) and operation not in ('rename',):
          raise ValueError('Stop the active task before changing this session')
        if operation == 'restore':
          record['deleted'] = False
        elif operation == 'delete':
          record['deleted'] = True
        elif operation == 'rename':
          record['name'] = session_rules.validate_name(payload['name'])
        elif operation == 'chat-config':
          if session_rules.chat_config_locked(record):
            raise ValueError(
                'Captured Chat configuration is immutable; create a new Chat'
            )
          record.update(
              generation=generation_config.generation_config(
                  payload.get('generation')
              ),
              initialized=False,
              status='draft',
          )
        elif operation == 'update':
          if record['has_capture'] or record.get('successful_turns'):
            raise ValueError(
                'Duplicate a captured session to change its configuration'
            )
          config = {
              **session_rules.validate_config(payload),
              **tap_profiles.profile_config(payload),
          }
          for run in config['runs']:
            self.resolve_run(run)
            self.resolve_taps(run, config)
          record.update(**config, initialized=False, status='draft')
        elif operation in ('duplicate', 'chat'):
          parent_id = record.get('parent_session_id', record['id'])
          record = {
              **copy.deepcopy(record),
              'id': str(uuid.uuid4()),
              'name': session_rules.validate_name(
                  payload.get('name', session_rules.copy_name(record['name']))
              ),
              'created_at': fsutil.now(),
              'has_capture': False,
              'initialized': False,
              'status': 'draft',
              'notice': 'Configuration copied.',
          }
          for key in (
              'capture',
              'job_id',
              'initialization',
              'execution',
              'successful_turns',
              'pending_chat',
              'read_only',
              'chat_state',
              'session_initialized',
          ):
            record.pop(key, None)
          if operation == 'chat':
            record['parent_session_id'] = parent_id
            if 'generation' in payload:
              record['generation'] = generation_config.generation_config(
                  payload['generation']
              )
          else:
            record.pop('parent_session_id', None)
          self.state['sessions'].append(record)
        else:
          raise ValueError('Unknown session operation')
      self._save()
      return self._with_execution(record)

  def rename(self, payload):
    """Renames a session record using the validated payload name."""
    return self.manage('rename', payload)

  def capture_store(self, identity):
    """Returns the cached capture store for a completed session."""
    record = self.get(identity)
    if not record.get('capture'):
      raise ValueError('No completed capture for this session yet')
    path = (self.root / record['capture']).resolve()
    if not path.is_relative_to(self.root):
      raise ValueError('Invalid capture path')
    return self.capture_cache.get(path)
