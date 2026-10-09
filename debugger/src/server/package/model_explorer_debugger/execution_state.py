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

"""Per-Session execution phases and Runner ownership of devices.

Also contains the lifecycle monitor.
"""

from copy import deepcopy
import queue
import threading
from uuid import uuid4

from .fsutil import now

ACTIVE_PHASES = ('starting', 'active')
OWNING_PHASES = ('starting', 'active', 'ending', 'unavailable')
INACTIVE = {'phase': 'inactive', 'runners': [], 'newChatAllowed': False}


class ExecutionState:
  """Owns `executions` and `device_owners` under one lock.

  JobManager keeps aliases to the two dictionaries for tests and local tooling;
  every mutation here happens in place so those aliases stay valid.
  """

  def __init__(self, server_identity, device_key):
    self.lock = threading.RLock()
    self.executions = {}
    self.device_owners = {}
    self.events = queue.Queue()
    self.server_identity = server_identity
    self.device_key = device_key
    self.thread = None

  def mark_interrupted(self, sessions):
    for record in sessions:
      if record.get('execution_interrupted') and not record.get(
          'parent_session_id'
      ):
        self.executions[record['id']] = {
            'phase': 'unavailable',
            'runners': [],
            'newChatAllowed': False,
            'error': 'Server restarted; this execution cannot reconnect.',
        }

  def view(self, identity):
    with self.lock:
      result = deepcopy(self.executions.get(identity, INACTIVE))
    for runner in result['runners']:
      owner = runner.get('owner')
      if isinstance(owner, dict):
        owner['isCurrentServer'] = (
            owner.get('serverId') == self.server_identity()['id']
        )
    return result

  def set_phase(self, identity, phase, **fields):
    with self.lock:
      state = self.executions.setdefault(
          identity, {'phase': 'inactive', 'runners': []}
      )
      state.update(phase=phase, **fields)
      return deepcopy(state)

  def set_runner_phase(
      self, identity, role, phase, *, active_only=False, **fields
  ):
    with self.lock:
      state = self.executions.get(identity, {})
      if active_only and state.get('phase') not in ACTIVE_PHASES:
        return
      for runner in state.get('runners', []):
        if runner['role'] == role:
          runner.update(fields)
          if phase is not None:
            runner['phase'] = phase

  def owner(self, identity, name, role):
    identity_record = self.server_identity()
    return dict(
        serverId=identity_record['id'],
        serverName=identity_record['name'],
        sessionId=identity,
        sessionName=name,
        runId=role,
    )

  def reserve(self, identity, name, runs):
    """Claim every device of `runs` for `identity`; enter the starting phase."""
    devices = {self.device_key(run['device']) for run in runs}
    with self.lock:
      if self.executions.get(identity, {}).get('phase') in OWNING_PHASES:
        raise ValueError(
            'This Session already owns an execution; end it before starting'
            ' again'
        )
      if any(device in self.device_owners for device in devices):
        raise ValueError('Device is occupied by another Session')
      for device in devices:
        self.device_owners[device] = identity
      self.executions[identity] = dict(
          phase='starting',
          instanceId=str(uuid4()),
          activeChatId=None,
          newChatAllowed=False,
          runners=[
              dict(
                  role=run['id'],
                  deviceId=run['device'],
                  runnerSlot=run.get('_runner_slot'),
                  buildId=run.get('runnerBuild'),
                  phase='starting',
                  connected=False,
                  owner=self.owner(identity, name, run['id']),
              )
              for run in runs
          ],
      )

  def activate(self, identity, role, fields):
    """A Runner confirmed the activation; record its actual device and build."""
    with self.lock:
      if self.executions[identity]['phase'] not in ACTIVE_PHASES:
        return
      for runner in self.executions[identity]['runners']:
        if runner['role'] == role:
          actual = fields.get('deviceId', runner['deviceId'])
          self.device_owners[self.device_key(actual)] = identity
          runner.update(
              deviceId=actual,
              connected=True,
              owner=fields.get('owner'),
              buildId=fields.get('buildId'),
              lastSeen=now(),
          )
          # Lifecycle delivery can lag the completed initialize call.
          if runner['phase'] in ('starting', 'connecting'):
            runner['phase'] = 'loading'

  def release(self, identity, interrupted=None):
    """Free the Session's devices and enter ended (or interrupted) phase."""
    with self.lock:
      state = self.executions.get(identity, {'runners': []})
      for runner in state['runners']:
        runner.update(
            phase='unknown' if interrupted else 'ended', connected=False
        )
      for device, owner in list(self.device_owners.items()):
        if owner == identity:
          self.device_owners.pop(device)
      state.update(
          phase='interrupted' if interrupted else 'ended',
          newChatAllowed=False,
          activeChatId=None,
      )
      if interrupted:
        state['error'] = interrupted
      self.executions[identity] = state

  def snapshot(self, identity):
    with self.lock:
      return deepcopy(self.executions.get(identity)), dict(self.device_owners)

  def rollback(self, identity, snapshot, job):
    """Undo what admitting `job` changed.

    Concurrent Runner heartbeats are left intact.
    """
    prior_execution, prior_devices = snapshot
    with self.lock:
      if job['operation'] == 'initialize':
        # reserve() created this execution; no Runner was contacted for it yet.
        if prior_execution is None:
          self.executions.pop(identity, None)
        else:
          self.executions[identity] = prior_execution
        for device, owner in list(self.device_owners.items()):
          if owner == identity and device not in prior_devices:
            self.device_owners.pop(device)
        return
      current = self.executions.get(identity)
      if current is None:
        return
      if prior_execution is None:
        self.executions.pop(identity, None)
        return
      current['phase'] = prior_execution['phase']
      for key in (
          'activeJobId',
          'activeOperation',
          'activeChatId',
          'newChatAllowed',
      ):
        if key in prior_execution:
          current[key] = prior_execution[key]
        else:
          current.pop(key, None)

  def start(self, handler):
    """Consume Runner lifecycle events on a daemon thread until `stop()`."""
    self.thread = threading.Thread(
        target=self._monitor,
        args=(handler,),
        daemon=True,
        name='session-lifecycle',
    )
    self.thread.start()

  def stop(self):
    self.events.put(None)

  def _monitor(self, handler):
    while True:
      event = self.events.get()
      if event is None:
        return
      if handler.stopping:
        continue
      identity, role, kind, fields = event
      state = self.view(identity)
      if 'executionId' in fields and fields['executionId'] != state.get(
          'instanceId'
      ):
        continue
      if state['phase'] not in ACTIVE_PHASES:
        continue
      try:
        if kind == 'heartbeat':
          self.set_runner_phase(
              identity,
              role,
              None,
              active_only=True,
              lastSeen=fields['lastSeen'],
              connected=True,
          )
        elif kind == 'activated':
          self.activate(identity, role, fields)
        elif kind == 'stop_requested':
          handler.stop_requested(identity)
        elif kind == 'session_end_requested':
          handler.close_session(identity)
        elif kind == 'disconnected':
          self.set_runner_phase(identity, role, 'disconnected', connected=False)
          handler.unavailable(
              identity, fields.get('error', 'Runner disconnected')
          )
      except Exception as error:
        # A persistence error must not silently kill the lifecycle monitor.
        self.set_phase(
            identity,
            'unavailable',
            newChatAllowed=False,
            activeChatId=None,
            error=str(error),
        )
