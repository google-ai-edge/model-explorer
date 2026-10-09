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

"""Session-scoped operations, paired generation, and durable outcomes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import shutil
import threading
from uuid import uuid4
from model_debugger_contracts.errors import InputRejected
from model_debugger_contracts.jobs import TERMINAL_STATUSES
from .event_journal import EventJournal
from .execution_state import ACTIVE_PHASES, ExecutionState, OWNING_PHASES
from .fsutil import digest, now
from .job_admission import Admission
from .prepare_worker import PrepareWorker
from .publication import Publication

TERMINAL = set(TERMINAL_STATUSES)


class JobManager:
  """Session-scoped job table and dispatch.

  Collaborators: ExecutionState (phases, device ownership, lifecycle monitor),
  Admission (prepare/commit/dispatch with rollback), Publication (durable
  Turns and captures) and PrepareWorker (tap preparation subprocesses). The
  dictionaries `executions` and `device_owners` alias the ExecutionState ones.
  """

  def __init__(self, registry):
    self.registry = registry
    from .runtime.runners import Runners

    self.runners = Runners(registry.root)
    self.lock = threading.RLock()
    self.state = ExecutionState(lambda: self.runners.identity, self._device_key)
    self.state.mark_interrupted(registry.state['sessions'])
    self.lifecycle_lock = self.state.lock
    self.executions = self.state.executions
    self.device_owners = self.state.device_owners
    self.lifecycle_events = self.state.events
    self.runners.on_lifecycle = lambda *event: self.lifecycle_events.put(event)
    registry.server_identity = self.runners.identity
    registry.execution_provider = self.execution
    self.admission = Admission(self)
    self.publication = Publication(self)
    self.preparation = PrepareWorker(self)
    self.journal = EventJournal(registry.database)
    self.cancels = {}
    self.active_jobs = {}
    self.threads = {}
    self.processes = {}
    self.thread = None  # Last submitted thread, retained for local tooling.
    self.stopping = False
    self.jobs = {}
    for job in registry.database.jobs():
      if job['status'] not in TERMINAL:
        job.update(
            status='failed',
            error='Server restarted during this task',
            finished_at=now(),
        )
        job['sequence'] += 1
        registry.database.save_job(
            job,
            dict(
                sequence=job['sequence'],
                jobId=job['id'],
                sessionId=job['session_id'],
                turnId=job['turn'],
                type='status',
                status='failed',
                error=job['error'],
            ),
        )
      self.jobs[job['id']] = job
    self.publication.recover()
    self.state.start(self)

  @property
  def active(self):
    """Compatibility view; scheduling uses active_jobs keyed by Session."""
    return next(iter(self.active_jobs.values()), None)

  def owner_id(self, identity):
    record = self.registry.get(identity)
    return record.get('parent_session_id', identity)

  def execution(self, identity):
    return self.state.view(identity)

  def _phase(self, identity, phase, **fields):
    return self.state.set_phase(identity, phase, **fields)

  def _runner_phase(
      self, identity, role, phase, *, active_only=False, **fields
  ):
    self.state.set_runner_phase(
        identity, role, phase, active_only=active_only, **fields
    )

  def _device_key(self, value):
    method = getattr(self.runners, 'device_key', None)
    return method(value) if method else value

  def reserve_runs(self, identity, record):
    """Resolves a Session's Runner runs and reserves them for its owner."""
    runs = self.runners.resolve_runs(record['runs'])
    self.state.reserve(identity, record['name'], runs)
    return runs

  def _owner(self, identity, name, role):
    return self.state.owner(identity, name, role)

  # ExecutionState monitor callbacks.
  def stop_requested(self, identity):
    with self.lock:
      job = self.jobs.get(self.active_jobs.get(identity))
      if job and job['operation'] == 'generate':
        self.cancel(job['id'])

  def unavailable(self, identity, error):
    self._unavailable(identity, error)

  def _unavailable(self, identity, error):
    with self.lock:
      state = self.execution(identity)
      if state['phase'] in ('ended', 'inactive', 'ending'):
        return
      self._phase(
          identity,
          'unavailable',
          newChatAllowed=False,
          activeChatId=None,
          error=str(error),
      )
      self.registry.update_family(
          identity,
          initialized=False,
          read_only=True,
          chat_state='unavailable',
          notice=str(error),
      )
      job = self.jobs.get(self.active_jobs.get(identity))
      # Complete texts already confirmed remain publishable despite late
      # cleanup loss.
      if (
          job
          and job['operation'] == 'generate'
          and not self._both_confirmed(job)
      ):
        self._request_cancel(job)

  def _finish_end(self, identity, worker=None, interrupted=None):
    if (
        worker
        and worker is not threading.current_thread()
        and worker.ident is not None
    ):
      worker.join(timeout=35)
    if worker and worker.is_alive():
      interrupted = (
          interrupted
          or 'Task cancellation did not finish; Runner state is unknown'
      )
      self.runners.abandon_session(identity)
      worker.join(timeout=2)
    try:
      self.runners.close_session(identity)
    except Exception as error:
      interrupted = interrupted or str(error)
    self.registry.update_family(
        identity,
        initialized=False,
        read_only=True,
        chat_state='ended',
        notice=interrupted or 'Session ended; saved history is retained.',
    )
    self.state.release(identity, interrupted)

  def _request(
      self,
      record,
      run,
      artifact,
      directory,
      operation,
      previous=None,
      prompt='',
      max_output_tokens=32,
      turn=1,
      request_id='',
  ):
    owner = record.get('parent_session_id', record['id'])
    parent = self.registry.get(owner)
    return dict(
        session_id=owner,
        chat_id=record['id'],
        execution_id=self.execution(owner).get('instanceId'),
        owner=self._owner(owner, parent['name'], run['id']),
        runtime_root=str(self.registry.runtime_root),
        pytorch_root=str(self.registry.pytorch_root),
        output=str(directory / run['id']),
        run=run,
        turn=turn,
        request_id=request_id,
        model=artifact['path'],
        model_sha256=artifact['sha256'],
        manifest=artifact.get('manifest'),
        model_files=artifact.get('model_files'),
        operation=operation,
        messages=[],
        generation=record.get('generation', {}).get(run['id'], {}),
        tap_selection=self.registry.resolve_taps(run, record)
        if operation == 'prepare'
        else None,
        tap_profile=record.get('tap_profile'),
        tap_cache=str(self.registry.root / 'prepared-models'),
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )

  def directory(self, job):
    return (
        self.registry.root / 'sessions' / job['session_id'] / 'jobs' / job['id']
    )

  def get(self, identity):
    with self.lock:
      if identity not in self.jobs:
        raise ValueError('Unknown task')
      return deepcopy(self.jobs[identity])

  def _set(self, job, **fields):
    """Updates a job under the jobs lock.

    Every write to a job dict happens under the jobs lock; get() copies under
    it.
    """
    with self.lock:
      job.update(fields)

  def record_event(self, job, kind, **data):
    """Appends a sequenced event to a job and persists it."""
    with self.lock:
      job['sequence'] += 1
      event = dict(
          sequence=job['sequence'],
          jobId=job['id'],
          sessionId=job['session_id'],
          turnId=job['turn'],
          type=kind,
          **data,
      )
      if kind in ('delta', 'progress'):
        # Streaming rows are journaled and committed in batches off the token
        # path.
        self.journal.append(job['id'], job['sequence'], event)
        return
      try:
        # Earlier streaming rows must be durable before a later status or
        # result row.
        self.journal.flush()
        self.registry.database.save_job(job, event)
      except BaseException:
        job['sequence'] -= 1
        raise

  def events(self, identity, after=0):
    self.get(identity)
    self.journal.flush()
    return self.registry.database.events(identity, after)

  def _cancel_flag(self, job):
    with self.lock:
      return self.cancels.setdefault(job['id'], threading.Event())

  def _request_cancel(self, job):
    """Signals cancellation to the job's threads and preparation subprocess.

    Threads observe the Event; the preparation subprocess observes the
    sentinel file.
    """
    self._cancel_flag(job).set()
    (self.directory(job) / 'cancel').touch()

  def _cancelled(self, job):
    return self._cancel_flag(job).is_set()

  def start(self, identity, operation, payload):
    if operation not in ('prepare', 'initialize', 'generate', 'new_chat'):
      raise ValueError('Unknown runtime operation')
    if operation == 'initialize':
      identity = self.owner_id(identity)
    key = payload.get('request_id')
    if not isinstance(key, str) or not 1 <= len(key) <= 120:
      raise ValueError('request_id is required')
    request_digest = digest([identity, operation, payload])
    # Registry projections take the lifecycle lock briefly on their own;
    # admission holds only the jobs and registry locks so Runner heartbeats
    # keep flowing.
    with self.lock, self.registry.lock:
      for job in self.jobs.values():
        if job['session_id'] == identity and job['request_id'] == key:
          if job['request_digest'] != request_digest:
            raise ValueError(
                'request_id was already used for a different request'
            )
          return self.get(job['id'])
      if self.stopping:
        raise ValueError('Server is shutting down')
      record = self.registry.get(identity)
      owner = record.get('parent_session_id', identity)
      if owner in self.active_jobs:
        raise ValueError('Another operation is active in this Session')
      artifacts = {
          run['id']: self.registry.resolve_run(run) for run in record['runs']
      }
      for run in record['runs']:
        self.registry.require_runtime(run)
      state = self.execution(owner)
      if operation == 'prepare':
        if not any(run['runtime'] == 'LiteRT-LM' for run in record['runs']):
          raise ValueError('PyTorch preparation is not required')
        for run in record['runs']:
          self.registry.resolve_taps(run, record)
        if not record.get('tap_profile') or record.get('has_capture'):
          raise ValueError(
              'Select a tap profile on an uncaptured Session first'
          )
      elif (
          operation != 'initialize'
          and record.get('tap_profile')
          and not record.get('tap_prepared')
      ):
        raise ValueError('Prepare the selected tensor capture profile first')
      if operation == 'generate':
        if (
            state['phase'] != 'active'
            or state.get('activeChatId') != identity
            or record.get('read_only')
            or not record.get('initialized')
        ):
          raise ValueError(
              'This Chat is read-only; create a new Chat when the Session'
              ' permits it'
          )
        if (
            not isinstance(payload.get('prompt'), str)
            or not payload['prompt'].strip()
            or len(payload['prompt'].encode('utf-8')) > 4 * 1024 * 1024
        ):
          raise ValueError('Enter a non-empty prompt up to 4 MiB of UTF-8 text')
        if (
            type(payload.get('max_output_tokens')) is not int
            or not 1 <= payload['max_output_tokens'] <= 256
        ):
          raise ValueError('max_output_tokens must be between 1 and 256')
      if operation == 'new_chat':
        if (
            not record.get('pending_chat')
            or state['phase'] != 'active'
            or not state.get('newChatAllowed')
        ):
          raise ValueError('This Session cannot initialize a new Chat')
      previous = (
          self.registry.capture_store(identity).session
          if record.get('capture')
          else None
      )
      if operation in ('generate', 'new_chat'):
        devices = {runner['role']: runner for runner in state['runners']}
        record['runs'] = [
            {
                **run,
                'device': devices[run['id']]['deviceId'],
                **(
                    {'_runner_slot': devices[run['id']]['runnerSlot']}
                    if devices[run['id']].get('runnerSlot') is not None
                    else {}
                ),
            }
            for run in record['runs']
        ]
      saved = record.get('successful_turns', [])
      turn = (
          max(
              [item['n'] for item in saved]
              + [
                  item.get('n', index + 1)
                  for index, item in enumerate(
                      (previous or {}).get('turns', [])
                  )
              ],
              default=0,
          )
          + 1
      )
      job = dict(
          id=str(uuid4()),
          session_id=identity,
          owner_session_id=owner,
          operation=operation,
          status={
              'prepare': 'preparing',
              'initialize': 'initializing',
              'new_chat': 'initializing',
              'generate': 'running',
          }[operation],
          request_id=key,
          request_digest=request_digest,
          created_at=now(),
          sequence=0,
          error='',
          turn=turn,
          output={'ref': '', 'target': ''},
          sides={},
          stage='accepted',
          previous_chat_id=state.get('activeChatId'),
      )
      if operation == 'new_chat' and payload.get('creation_request_id'):
        job.update(
            creation_request_id=payload['creation_request_id'],
            creation_request_digest=payload['creation_request_digest'],
        )
      return self._admit(job, record, payload, artifacts, previous)

  def _admit(self, job, record, payload, artifacts, previous):
    """Prepare -> commit -> dispatch, or abort without touching a Runner.

    See `job_admission.Admission`.
    """
    return self.admission.admit(job, record, payload, artifacts, previous)

  def cancel(self, identity):
    with self.lock:
      job = self.jobs.get(identity)
      if not job:
        raise ValueError('Unknown task')
      if job['status'] not in TERMINAL:
        if job['operation'] in ('initialize', 'new_chat'):
          raise ValueError(
              'Session and Chat initialization cannot be cancelled'
          )
        job['stop_requested'] = True
        job['discard_turn'] = job.get(
            'discard_turn', False
        ) or not self._both_confirmed(job)
        if job['discard_turn']:
          self._request_cancel(job)
        for role in ('ref', 'target'):
          self._runner_phase(
              job['owner_session_id'], role, 'stopping', active_only=True
          )
        self.record_event(job, 'cancelling')
      return self.get(identity)

  def close(self):
    with self.lock:
      if self.stopping:
        return
      self.stopping = True
      active = list(self.active_jobs.values())
      for identity in active:
        self._request_cancel(self.jobs[identity])
    for identity in active:
      thread = self.threads[identity]
      if thread.ident is not None:
        thread.join(timeout=6)
    with self.lock:
      processes = list(self.processes.values())
    for process in processes:
      if process.poll() is None:
        process.kill()
        process.wait(timeout=2)
    for identity in list(self.executions):
      if self.execution(identity)['phase'] in OWNING_PHASES:
        self._finish_end(
            identity, self.threads.get(self.active_jobs.get(identity))
        )
    self.runners.close()
    self.state.stop()
    self.journal.close()

  def close_session(self, identity, interrupted=None):
    identity = self.owner_id(identity)
    with self.lock:
      record = self.registry.get(identity)
      if self.execution(identity)['phase'] in (
          'inactive',
          'ended',
          'interrupted',
          'ending',
      ):
        return record
      active = self.jobs.get(self.active_jobs.get(identity))
      if (
          active
          and active['operation'] in ('initialize', 'new_chat')
          and not self.stopping
      ):
        raise ValueError('Session and Chat initialization cannot be cancelled')
      self._phase(identity, 'ending', newChatAllowed=False, activeChatId=None)
      self.registry.update_family(identity, initialized=False, read_only=True)
      job = self.jobs.get(self.active_jobs.get(identity))
      if job:
        self._request_cancel(job)
      threading.Thread(
          target=self._finish_end,
          args=(
              identity,
              self.threads.get(job['id']) if job else None,
              interrupted,
          ),
          daemon=True,
          name='end-session',
      ).start()
      return self.registry.get(identity)

  def manage(self, operation, payload):
    pending = None
    with self.lock:
      identity = payload.get('id')
      owner = self.owner_id(identity) if identity else None
      request_key = payload.get('request_id') if operation == 'chat' else None
      if request_key is not None and (
          not isinstance(request_key, str) or not 1 <= len(request_key) <= 120
      ):
        raise ValueError('Invalid Chat creation request_id')
      creation_digest = digest([owner, payload]) if request_key else None
      repeated = (
          next(
              (
                  job
                  for job in self.jobs.values()
                  if job.get('owner_session_id') == owner
                  and job.get('creation_request_id') == request_key
              ),
              None,
          )
          if request_key
          else None
      )
      if repeated:
        if repeated['creation_request_digest'] != creation_digest:
          raise ValueError(
              'request_id was already used for a different Chat request'
          )
        pending = (
            repeated['session_id'],
            repeated['id'],
            self.threads.get(repeated['id']),
        )
      if (
          pending is None
          and owner in self.active_jobs
          and operation in ('update', 'chat-config', 'delete', 'chat')
      ):
        raise ValueError('Wait for the active operation in this Session')
      state = self.execution(owner) if owner else {'phase': 'inactive'}
      if operation in ('update', 'delete') and state['phase'] in OWNING_PHASES:
        raise ValueError(
            'End this Session before changing its devices or deleting it'
        )
      if operation == 'chat' and pending is None:
        if state['phase'] != 'active' or not state.get('newChatAllowed'):
          raise ValueError('This Session cannot create a new Chat')
        candidate = self.registry.stage_chat(payload)
        try:
          job = self.start(
              candidate['id'],
              'new_chat',
              {
                  'request_id': request_key or str(uuid4()),
                  'creation_request_id': request_key,
                  'creation_request_digest': creation_digest,
              },
          )
        except Exception as error:
          self.registry.update(
              candidate['id'], chat_state='failed', notice=str(error)
          )
          raise
        pending = candidate['id'], job['id'], self.threads[job['id']]
      elif operation == 'chat-config' and state.get('activeChatId') == identity:
        # Existing Conversation configuration is immutable, even before a first
        # Turn.
        raise ValueError(
            'Create a new Chat to change initialized Conversation settings'
        )
      elif pending is None:
        return self.registry.manage(operation, payload)
    # Existing callers navigate to the returned Chat immediately. Only return
    # after publication; the initialization job remains independently pollable.
    candidate_id, job_id, thread = pending
    if thread is not None:
      thread.join()
    job = self.get(job_id)
    if job['status'] != 'completed':
      raise ValueError(
          f"New Chat initialization failed ({job_id}): {job['error']}"
      )
    return self.registry.get(candidate_id)

  @staticmethod
  def _both_confirmed(job):
    return all(
        job.get('sides', {}).get(role, {}).get('generation_status')
        == 'completed'
        and job['sides'][role].get('output_confirmed') is True
        for role in ('ref', 'target')
    )

  def _emit(self, job, role, kind, data):
    with self.lock:
      if kind == 'delta':
        job['output'][role] += data['text']
      if kind == 'progress':
        job['progress'] = data.get('message', '')
      if kind == 'generation_completed' and data.get('output_confirmed', True):
        job['sides'][role] = dict(
            generation_status='completed',
            output_confirmed=True,
            output=data['output'],
        )
        job['output'][role] = data['output']
      if (
          not self.stopping
          and not job.get('stop_requested')
          and not self._cancelled(job)
      ):
        phase = {'loading': 'loading', 'initialized': 'ready'}.get(kind)
        if kind == 'generation_completed' and data.get(
            'output_confirmed', True
        ):
          phase = 'transferring'
        if phase is not None:
          self._runner_phase(
              job['owner_session_id'], role, phase, active_only=True
          )
      self.record_event(job, kind, runId=role, **data)

  def _paired(self, job, record, artifacts, payload, operation, directory):
    directory.mkdir(parents=True, exist_ok=True)
    results, errors = {}, {}

    def execute(run):
      role = run['id']
      request = self._request(
          record,
          run,
          artifacts[role],
          directory,
          operation,
          prompt=payload.get('prompt', ''),
          max_output_tokens=payload.get('max_output_tokens', 32),
          turn=job['turn'],
          request_id=job['id'],
      )
      self._runner_phase(
          job['owner_session_id'],
          role,
          'checking_input'
          if operation == 'preflight'
          else 'loading'
          if operation == 'initialize'
          else 'generating',
          active_only=True,
      )
      result = self.runners.execute(
          request,
          lambda kind, **data: self._emit(job, role, kind, data),
          lambda: self.stopping or self._cancelled(job),
      )
      if operation == 'generate':
        status = result.get('generation_status', 'completed')
        result['generation_status'] = status
        result['output_confirmed'] = result.get(
            'output_confirmed',
            status == 'completed' and isinstance(result.get('output'), str),
        )
        with self.lock:
          job['sides'][role] = deepcopy(result)
          if result['output_confirmed']:
            job['output'][role] = result['output']
          self.record_event(
              job,
              'generation_result',
              runId=role,
              generation_status=status,
              output_confirmed=result['output_confirmed'],
          )
      if result.get('connection_lost'):
        self._runner_phase(
            job['owner_session_id'], role, 'disconnected', connected=False
        )
        self._unavailable(
            job['owner_session_id'],
            result.get('connection_error', 'Runner disconnected'),
        )
      else:
        status = result.get('generation_status', 'completed')
        interrupted = (
            self.stopping
            or job.get('stop_requested')
            or self._cancelled(job)
            or status in ('stopped', 'cancelled')
        )
        phase = (
            'interrupted'
            if interrupted
            else 'ready'
            if status == 'completed'
            else 'failed'
        )
        self._runner_phase(
            job['owner_session_id'],
            role,
            phase,
            active_only=True,
            connected=True,
        )
        if operation == 'initialize' and status != 'completed':
          error = (
              InterruptedError
              if status in ('stopped', 'cancelled')
              else RuntimeError
          )
          raise error(result.get('error', 'Runner initialization failed'))
      return result

    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix='runner-side'
    ) as pool:
      pending = {pool.submit(execute, run): run['id'] for run in record['runs']}
      for future in as_completed(pending):
        role = pending[future]
        try:
          results[role] = future.result()
        except BaseException as error:
          errors[role] = error
          if not isinstance(error, InputRejected):
            phase = (
                'disconnected'
                if isinstance(error, ConnectionError)
                else 'interrupted'
                if isinstance(error, InterruptedError)
                else 'failed'
            )
            self._runner_phase(
                job['owner_session_id'],
                role,
                phase,
                active_only=True,
                **(
                    {'connected': False}
                    if isinstance(error, ConnectionError)
                    else {}
                ),
            )
          with self.lock:
            job.setdefault('failures', {})[role] = dict(
                stage=operation,
                error=str(error),
                code=getattr(error, 'code', None),
            )
            self.record_event(
                job, 'side_error', runId=role, stage=operation, error=str(error)
            )
          if isinstance(error, ConnectionError):
            self._unavailable(job['owner_session_id'], str(error))
          elif operation != 'preflight':
            self._request_cancel(job)
    return results, errors

  def _close_chat(self, job, *, allow_new, error=''):
    owner = job['owner_session_id']
    phase = self.execution(owner)['phase']
    if phase not in ('unavailable', 'ending', 'ended'):
      try:
        self.runners.reset_session(owner)
      except Exception as cleanup_error:
        self._unavailable(
            owner, 'Conversation cleanup failed: ' + str(cleanup_error)
        )
        phase = 'unavailable'
    with self.lock:
      self.registry.update(
          job['session_id'],
          initialized=False,
          read_only=True,
          chat_state='ended' if allow_new else 'failed',
          notice=error or 'Chat ended; models remain loaded.',
      )
      if self.execution(owner)['phase'] == 'active':
        self._phase(
            owner,
            'active',
            activeChatId=None,
            newChatAllowed=allow_new,
            **({'error': error} if error else {}),
        )

  def _publish_turn(self, job, record, results, artifacts):
    self.publication.publish_turn(job, record, results, artifacts)

  def execute(self, job, record, payload, artifacts, previous):
    """Runs an admitted job to a terminal status on its worker thread."""
    directory = self.directory(job)
    owner = job['owner_session_id']
    results = {}
    self._set(job, prompt=payload.get('prompt', ''))
    try:
      if shutil.disk_usage(directory).free < 1024**3:
        raise ValueError(
            'At least 1 GiB free disk space is required for capture'
        )
      operation = job['operation']
      if operation == 'prepare':
        results = self._prepare(job, record, artifacts)
      else:
        if (
            operation == 'initialize'
            and record.get('tap_profile')
            and not record.get('tap_prepared')
        ):
          self._set(job, stage='preparing_capture')
          self.record_event(
              job,
              'progress',
              message='Preparing tensor capture before Runner initialization…',
          )
          self._prepare(job, record, artifacts)
          record = self.registry.get(record['id'])
          record['generation'] = {'ref': {}, 'target': {}}
          artifacts = {
              run['id']: self.registry.resolve_run(run)
              for run in record['runs']
          }
        if operation == 'new_chat':
          self._set(job, stage='closing_previous_chat')
          self.runners.reset_session(owner)
        if operation == 'generate':
          self._set(job, stage='preflight')
          checks, errors = self._paired(
              job,
              record,
              artifacts,
              payload,
              'preflight',
              directory / 'preflight',
          )
          self._set(job, input_checks=checks)
          self.record_event(job, 'input_checks', checks=checks)
          if errors:
            raise next(iter(errors.values()))
          if self._cancelled(job):
            raise InterruptedError('Generation stopped before execution')
        self._set(
            job,
            stage='initializing_chat'
            if operation in ('initialize', 'new_chat')
            else 'generating',
        )
        results, errors = self._paired(
            job,
            record,
            artifacts,
            payload,
            'initialize' if operation == 'new_chat' else operation,
            directory,
        )
        if operation == 'generate':
          # Early terminal metadata survives transfer/receipt exceptions.
          for role in ('ref', 'target'):
            if role not in results and job.get('sides', {}).get(role, {}).get(
                'output_confirmed'
            ):
              results[role] = {
                  **job['sides'][role],
                  'debug_data': {
                      'status': 'unavailable',
                      'error': str(errors.get(role, 'Dump unavailable')),
                  },
              }
          if self._both_confirmed(job) and not job.get('discard_turn'):
            self._set(job, stage='publishing')
            self._publish_turn(job, record, results, artifacts)
          else:
            if errors:
              raise next(iter(errors.values()))
            failed = [
                r
                for r in results.values()
                if r.get('generation_status')
                not in ('completed', 'stopped', 'cancelled')
            ]
            if failed:
              raise RuntimeError(
                  failed[0].get('error', 'Runner generation failed')
              )
            raise InterruptedError('Generation stopped')
        else:
          if errors:
            raise next(iter(errors.values()))
          with self.lock:
            if self.execution(owner)['phase'] not in ACTIVE_PHASES:
              raise ConnectionError(
                  'Runner became unavailable before Chat initialization'
                  ' completed'
              )
            self.registry.update(
                record['id'],
                initialized=True,
                read_only=False,
                pending_chat=False,
                chat_state='active',
                status='ready',
                initialization=results,
                generation=record.get('generation', {}),
                notice='Both Runners and the Chat are ready.',
            )
            self._phase(
                owner, 'active', activeChatId=record['id'], newChatAllowed=True
            )
      while True:
        with self.lock:
          close_stopped_chat = (
              job.get('generation_status') == 'completed'
              and job.get('stop_requested')
              and not job.get('stop_handled')
          )
          if close_stopped_chat:
            job['stop_handled'] = True
          else:
            # Serializes the final Stop-acceptance boundary with cancel().
            job.update(status='completed', finished_at=now(), results=results)
            self.record_event(
                job,
                'status',
                status='completed',
                debug_data=job.get('debug_data'),
            )
            break
        self._close_chat(job, allow_new=True)
    except BaseException as error:
      operation = job['operation']
      if isinstance(error, InputRejected) and job.get('stop_requested'):
        error = InterruptedError('Generation stopped before execution')
      state = self.execution(owner)
      if (
          isinstance(error, InputRejected)
          and operation == 'generate'
          and job['stage'] == 'preflight'
          and state['phase'] == 'active'
      ):
        status = 'failed'
        self._set(job, input_rejected=True)
        with self.lock:
          if self.execution(owner)['phase'] == 'active':
            self.registry.update(
                record['id'],
                status='saved'
                if record.get('successful_turns') or record.get('has_capture')
                else 'ready',
                notice=str(error),
                initialized=True,
                read_only=False,
            )
            for run in record['runs']:
              self._runner_phase(owner, run['id'], 'ready')
      else:
        status = (
            'cancelled' if isinstance(error, InterruptedError) else 'failed'
        )
        if operation == 'initialize':
          self._finish_end(owner, interrupted=str(error))
          self.registry.update(
              record['id'],
              status='failed',
              notice=str(error),
              initialization_error={
                  'stage': job['stage'],
                  'failures': job.get('failures', {}),
                  'error': str(error),
              },
          )
        elif operation == 'new_chat':
          if state['phase'] == 'active':
            try:
              self.runners.reset_session(owner)
              with self.lock:
                if self.execution(owner)['phase'] == 'active':
                  self._phase(
                      owner,
                      'active',
                      activeChatId=None,
                      newChatAllowed=True,
                      error=str(error),
                  )
            except Exception as cleanup:
              self._unavailable(
                  owner, 'Candidate cleanup failed: ' + str(cleanup)
              )
          self.registry.update(
              record['id'],
              pending_chat=True,
              initialized=False,
              read_only=True,
              chat_state='failed',
              status='failed',
              notice=str(error),
          )
          self.registry.update(
              owner, notice='New Chat initialization failed: ' + str(error)
          )
        elif operation == 'generate':
          self._close_chat(
              job,
              allow_new=isinstance(error, InterruptedError),
              error=str(error),
          )
        else:
          self.registry.update(record['id'], status=status, notice=str(error))
      with self.lock:
        job.update(
            status=status,
            error=str(error),
            error_code=getattr(error, 'code', None),
            finished_at=now(),
            results=results,
        )
        self.record_event(
            job,
            'status',
            status=status,
            error=str(error),
            error_code=job['error_code'],
        )
    finally:
      with self.lock:
        if self.active_jobs.get(owner) == job['id']:
          self.active_jobs.pop(owner, None)
          if owner in self.executions:
            self._phase(
                owner,
                self.execution(owner)['phase'],
                activeJobId=None,
                activeOperation=None,
            )
        self.processes.pop(job['id'], None)
        self.cancels.pop(job['id'], None)

  def _prepare(self, job, record, artifacts):
    return self.preparation.run(job, record, artifacts)
