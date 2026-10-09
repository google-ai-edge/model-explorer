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

"""Prepare -> commit -> dispatch of a job, or abort before touching a Runner."""

from copy import deepcopy
import shutil
import threading

from .fsutil import atomic_json


class Admission:
  """Called with the manager's jobs and registry locks held in that order.

  All fallible preparation, including starting the gated thread, precedes the
  workspace commit. Registry update/accept_chat restore their raw records if
  that atomic save fails; no compensating workspace write is needed. The
  lifecycle lock is taken only inside ExecutionState calls, so Runner
  heartbeats and execution reads keep flowing while a job is admitted;
  rollback therefore undoes only the fields admission itself set.
  """

  def __init__(self, manager):
    self.manager = manager

  def admit(self, job, record, payload, artifacts, previous):
    manager, registry, state = (
        self.manager,
        self.manager.registry,
        self.manager.state,
    )
    identity, owner = job['session_id'], job['owner_session_id']
    prior_state = state.snapshot(owner)
    prior_thread = manager.thread
    prior_registry = registry.snapshot()
    released, accepted = threading.Event(), threading.Event()
    directory = manager.directory(job)
    directory_existed = directory.exists()
    if directory_existed or job['id'] in manager.jobs:
      raise ValueError('Job identity already exists')
    job['admission'] = 'pending'

    def run_if_accepted():
      released.wait()
      # Do not enter execution until start() has released admission locks.
      with manager.lock:
        should_run = accepted.is_set()
      if should_run:
        manager.execute(job, record, payload, artifacts, previous)

    try:
      with registry.database.transaction():
        if job['operation'] == 'initialize':
          record['runs'] = manager.reserve_runs(owner, record)
          record['generation'] = {'ref': {}, 'target': {}}
        directory.mkdir(parents=True)
        atomic_json(directory / 'job.json', job)
        manager.record_event(job, 'status', status=job['status'])
        manager.jobs[job['id']] = job
        manager.active_jobs[owner] = job['id']
        if job['operation'] == 'new_chat':
          state.set_phase(
              owner, 'active', activeChatId=None, newChatAllowed=False
          )
        if job['operation'] != 'prepare':
          state.set_phase(
              owner,
              state.view(owner)['phase'],
              activeJobId=job['id'],
              activeOperation=job['operation'],
          )
        thread = threading.Thread(
            target=run_if_accepted, daemon=True, name='job-' + job['id']
        )
        manager.threads[job['id']] = manager.thread = thread
        thread.start()
        result = deepcopy(job)
        result.pop('admission')

        # This is the acceptance commit. Nothing that performs I/O or starts
        # a thread may be added after it and before opening the dispatch gate.
        if job['operation'] == 'new_chat':
          registry.accept_chat(identity, job['previous_chat_id'], job['id'])
        else:
          registry.update(
              identity,
              status=job['status'],
              job_id=job['id'],
              notice='Operation accepted.',
          )
    except BaseException as error:
      registry.restore(prior_registry)
      manager.jobs.pop(job['id'], None)
      manager.active_jobs.pop(owner, None)
      manager.threads.pop(job['id'], None)
      manager.thread = prior_thread
      state.rollback(owner, prior_state, job)
      try:
        if not directory_existed and directory.exists():
          shutil.rmtree(directory)
      except OSError as cleanup_error:
        # SQLite admission is authoritative; leftover staging files
        # cannot resurrect a rejected job. Preserve both errors.
        raise OSError(
            f'{error}; rejected job cleanup failed: {cleanup_error}'
        ) from error
      finally:
        released.set()
      raise

    job.pop('admission')
    accepted.set()
    released.set()
    return result
