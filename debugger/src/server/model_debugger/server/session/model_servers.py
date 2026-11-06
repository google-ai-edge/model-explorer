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

"""Every Session's Model Server, and the devices they hold.

A Session (the owner) runs one Model Server at a time; what it does with its
two sides is a model_server.ModelServer's. This module keeps which Session
runs which Model Server, which device each holds, one upload of a model at a
time, and the last job run for each record.

The UI polls the listing for the Model Server's phase (see state()) and
watches jobs for progress: initialize, chat (a chat's start) and generate
each run as a job. A
device runs a runner only for the Model Server that holds it; a device check
asks holding() and launches nothing, and what the last runner of each build
there reported, reported(), which is kept from the runner's start on.
"""

import dataclasses
import threading
from typing import Any

from absl import logging
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import launcher
from model_debugger.server.session import capture_store
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import model_graphs as model_graphs_lib
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import prepared_models as prepared_lib
from model_debugger.server.session import session_store
from model_debugger.server.session import side as side_lib
from model_debugger.server.util import errors

# Phases in which a Model Server holds its devices.
OWNING_PHASES = ('starting', 'active', 'ending')
OFF_FIRST = 'Turn the Model Server off first.'
ALREADY_ON = "This Session's Model Server is already on."
RELEASING = 'Releasing both devices.'


def check_idle(state: model_server_lib.ModelServerState | None) -> None:
  """Refuses changing a Session whose Model Server is on.

  The rule of every change a running Model Server forbids (deleting),
  for require_idle() to apply and the listing to show.

  Args:
      state: The Session's Model Server now, or None if it never ran here.

  Raises:
      Conflict: The Model Server holds its devices.
  """
  if state is not None and state.phase in OWNING_PHASES:
    raise errors.Conflict(OFF_FIRST)


def check_close(state: model_server_lib.ModelServerState | None) -> None:
  """Refuses turning off a Model Server that is being turned off.

  The rule of close(), for it to apply and the listing to show. Any other
  close is accepted: one of a Model Server that is not on does nothing.

  Args:
      state: The Session's Model Server now, or None if it never ran here.

  Raises:
      Conflict: It is releasing its devices already.
  """
  if state is not None and state.phase == 'ending':
    raise errors.Conflict(RELEASING)


@dataclasses.dataclass(frozen=True)
class Holding:
  """The side of a Model Server that holds a device.

  Attributes:
      owner_id: The Session whose Model Server holds it.
      role: The side that runs there.
  """

  owner_id: str
  role: str


class ModelServers:
  """Every Session's Model Server, and the devices they hold."""

  def __init__(
      self,
      sessions: session_store.SessionStore,
      models: model_store.ModelStore,
      jobs: jobs_lib.Jobs,
      launchers: launcher.Devices,
      captures: capture_store.CaptureStore,
      prepared: prepared_lib.PreparedModels,
      graphs: model_graphs_lib.ModelGraphs | None = None,
  ) -> None:
    """Holds the stores; nothing runs until a Session is turned on.

    Args:
        sessions: The Session records.
        models: The registered models.
        jobs: The jobs the UI watches.
        launchers: What launches runners.
        captures: Where each chat's captures go.
        prepared: The copies of models that observe tensors, which the sides of
          a Session that observes some load.
        graphs: The models' graphs, which a start converts first if they are not
          kept, and which keep the execution graph a PyTorch model's turn
          exports; None converts and keeps none.
    """
    self._sessions = sessions
    self._jobs = jobs
    self._launchers = launchers
    # Guards every Model Server's state and this registry's; never held
    # across a runner call.
    self._lock = threading.RLock()
    self._model_servers: dict[str, model_server_lib.ModelServer] = {}
    # The last job this server ran for each record, for the UI to reattach.
    self._last_jobs: dict[str, str] = {}
    self._upload_locks: dict[str, threading.Lock] = {}
    # What the last runner on each device answered GetInfo with, by
    # device id and runner build ('' for the one installed there); kept
    # after it stopped.
    self._reported: dict[tuple[str, str], runner_pb2.RunnerInfo] = {}
    self._context = model_server_lib.Context(
        sessions=sessions,
        models=models,
        jobs=jobs,
        launchers=launchers,
        captures=captures,
        lock=self._lock,
        upload_lock=self._upload_lock,
        forget=self._forget,
        reported=self._keep_reported,
        capture_model=prepared.capture_model,
        prepare=prepared.prepare_start,
        loaded=prepared.loaded,
        observe_modules=prepared.observe_modules,
        **(
            {'convert': graphs.prepare_start, 'exported': graphs.exported}
            if graphs
            else {}
        ),
    )

  # ---- What the listing and the device checks read.

  def state(self, owner_id: str) -> model_server_lib.ModelServerState | None:
    """An owner's Model Server now, or None if it never ran here."""
    with self._lock:
      model_server = self._model_servers.get(owner_id)
      if model_server is None:
        return None
      return model_server.state()

  def last_job(self, record_id: str) -> str | None:
    """The last job this server ran for a record, for the UI to reattach."""
    with self._lock:
      return self._last_jobs.get(record_id)

  def holding(self, device: launcher.Device) -> Holding | None:
    """The side of a Model Server that holds a device, if any.

    A shared device is never held: any number of runners run there.

    Args:
      device: The device to check.

    Returns:
      The Holding object describing the holding owner and role, or None.
    """
    if device.shared:
      return None
    with self._lock:
      holder = self._holder(device.id)
      if holder is None:
        return None
      model_server, side = holder
      return Holding(model_server.owner_id, side.role)

  def runners(self, device_id: str) -> int:
    """How many runners the Model Servers that are on run on a device."""
    with self._lock:
      count = 0
      for model_server in self._model_servers.values():
        if model_server.phase in OWNING_PHASES:
          for side in model_server.sides.values():
            if side.device.id == device_id and side.client is not None:
              count += 1
      return count

  def reported(
      self, device_id: str, build: str | None = None
  ) -> runner_pb2.RunnerInfo | None:
    """What the last runner of a build on a device reported, if one ran.

    Kept for as long as this server runs, the runner long stopped or not.

    Args:
        device_id: The device.
        build: The runner build a Session chose there (runner_builds), or None
          for the runner installed on the device.

    Returns:
        Its RunnerInfo, or None if no runner of that build ran there.
    """
    with self._lock:
      info = self._reported.get((device_id, build or ''))
    if info is None:
      return None
    kept = runner_pb2.RunnerInfo()
    kept.CopyFrom(info)
    return kept

  def require_idle(self, session_id: str) -> None:
    """Refuses changing a Session whose Model Server is on.

    Args:
      session_id: The session ID to check.

    Raises:
      NotFound: No such Session.
      Conflict: As check_idle().
    """
    self._sessions.get(session_id)
    check_idle(self.state(session_id))

  def delete(self, session_id: str) -> dict[str, Any]:
    """Deletes the record of a Session whose Model Server is off.

    For good (session_store.SessionStore.delete()); what this registry
    kept of its Model Server is dropped. The check and the removal hold
    the lock initialize() admits a start under: a Session is never
    deleted as it starts. Its files are the caller's to remove next
    (workspace.Workspace.remove_session_files()), outside the lock.

    Args:
      session_id: The session ID to delete.

    Returns:
      The record as it was.

    Raises:
      NotFound: No such Session.
      Conflict: As check_idle().
    """
    with self._lock:
      self.require_idle(session_id)
      self._model_servers.pop(session_id, None)
      self._last_jobs.pop(session_id, None)
      return self._sessions.delete(session_id)

  # ---- Lifecycle.

  def create(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Creates a Session and turns its Model Server on, or does neither.

    The start is checked against the devices connected now before the
    record is written (session_store.SessionStore.create()), and both
    are made under the lock a start is admitted under: no other start
    takes a device meanwhile, and no Session is saved that did not
    start. Its start then runs as initialize()'s.

    Args:
        payload: What the UI's editor posts.

    Returns:
        The record, started; its initialize job is last_job().

    Raises:
        Conflict, NotFound, ValueError: The configuration, or its start,
          is refused, as initialize() refuses one: nothing is written.
    """
    devices = self._launchers.devices()
    available = {device.id: device for device in devices}
    with self._lock:
      record = self._sessions.create(
          payload, lambda owner: self._check_initialize(owner, available)
      )
      try:
        self.initialize(record['id'], devices)
      except BaseException:
        self._sessions.delete(record['id'])
        raise
      return self._sessions.get(record['id'])

  def initialize(
      self, session_id: str, devices: list[launcher.Device] | None = None
  ) -> dict[str, Any]:
    """Turns a Session's Model Server on: both runners and models, no chat.

    The devices are listed anew, unless given: a start goes by what is
    connected now.

    A Session starts once: its record says so (`started`) as the start
    is admitted, and a start that does not come on ends the Session
    (model_server.ended()). The graphs of its models are converted
    first, unless kept, before its runners launch (another Session's
    runners may be running): one that cannot be converted fails the
    start. A Session that observes
    tensors has its copies prepared then, in the same job
    (model_graphs.ModelGraphs.prepare_start(),
    prepared_models.PreparedModels.prepare_start()).

    Args:
        session_id: The Session.
        devices: The devices connected now, as create() listed them; None lists
          them.

    Returns:
        The initialize job; it runs on a thread of its own.
    """
    owner = self._sessions.get(session_id)
    owner_id = owner['id']
    if devices is None:
      devices = self._launchers.devices()
    available = {device.id: device for device in devices}
    with self._lock:
      # Deleted meanwhile (delete()): it does not start.
      self._sessions.get(owner_id)
      job = self._jobs.create(owner_id, 'initialize')
      try:
        devices = self._check_initialize(owner, available)
        model_server = model_server_lib.ModelServer.admit(
            owner, devices, job['id'], self._context
        )
      except Exception as error:
        self._jobs.finish(job['id'], 'failed', str(error))
        # A start the rules refuse is no failure of the server's.
        refused = isinstance(
            error, (errors.Conflict, errors.NotFound, ValueError)
        )
        logging.log(
            logging.INFO if refused else logging.ERROR,
            '%s: start refused: %s',
            session_store.label(owner),
            errors.message(error),
        )
        raise
      self._model_servers[owner_id] = model_server
      self._last_jobs[owner_id] = job['id']
      self._sessions.set_flags(owner_id, started=True)
    model_server.initialize(owner)
    return self._jobs.get(job['id'])

  def _check_initialize(
      self, owner: dict[str, Any], available: dict[str, launcher.Device]
  ) -> dict[str, launcher.Device]:
    """Refuses a start of an owner's Model Server; returns its devices.

    Called with the lock held.

    Args:
      owner: The owner dictionary.
      available: Mapping of available device IDs to devices.

    Returns:
      Mapping of role to chosen devices.

    Raises:
      Conflict: The Model Server is on, or as model_server.check_start().
      NotFound, ValueError: As model_server.check_start().
    """
    current = self._model_servers.get(owner['id'])
    if current is not None and current.phase in OWNING_PHASES:
      raise errors.Conflict(ALREADY_ON)
    return model_server_lib.check_start(
        owner,
        available,
        lambda device_id: self._holder(device_id) is not None,
        self._context.models,
    )

  def _holder(
      self, device_id: str
  ) -> tuple[model_server_lib.ModelServer, side_lib.Side] | None:
    """The Model Server and side holding a device, if any."""
    for model_server in self._model_servers.values():
      if model_server.phase not in OWNING_PHASES:
        continue
      for side in model_server.sides.values():
        if side.device.id == device_id:
          return model_server, side
    return None

  def start_chat(self, session_id: str, generation: object) -> dict[str, Any]:
    """Starts a chat of a Session whose Model Server is on.

    Its settings are checked first (session_store's check_generation(),
    then each side's runner's, as its CreateChat is made): settings
    refused change nothing. Once admitted, the live chat, if any, is
    over (model_server.NEW_CHAT), and the new one is live once both
    sides created it (model_server.ModelServer.start_chat()).

    Args:
        session_id: The Session.
        generation: Each side's settings, `{ref, target}`, as the UI sends them.

    Returns:
        The chat job, of the number the chat gets (`chat`); it runs on a
        thread of its own.

    Raises:
        NotFound: No such Session.
        ValueError: The settings are refused.
        Conflict: As model_server.check_new_chat().
    """
    generation = self._sessions.check_generation(session_id, generation)
    record = self._sessions.get(session_id)
    with self._lock:
      model_server = self._model_servers.get(session_id)
      try:
        model_server_lib.check_new_chat(self.state(session_id), record)
        requests = model_server.chat_requests(generation)
      except (errors.Conflict, ValueError) as error:
        logging.info(
            '%s: chat refused: %s',
            session_store.label(record),
            errors.message(error),
        )
        raise
      # The record as it is now, under the lock no chat starts without.
      now = self._sessions.get(session_id)
      chats = now['chats']
      job = self._jobs.create(
          session_id, 'chat', chat=chats[-1]['n'] + 1 if chats else 1
      )
      live = model_server.admit_chat(job['id'])
      if live is not None:
        has_capture = session_store.chat(now, live)['has_capture']
        self._sessions.set_chat_flags(
            session_id,
            live,
            over=True,
            notice=model_server_lib.ended_chat(
                model_server_lib.NEW_CHAT, has_capture
            ),
        )
      self._last_jobs[session_id] = job['id']
    model_server.start_chat(job['id'], generation, requests)
    return self._jobs.get(job['id'])

  def generate(
      self,
      session_id: str,
      prompt: object,
      chat: object = None,
  ) -> dict[str, Any]:
    """Sends a turn of the live chat to both sides and stores its capture.

    Args:
        session_id: The Session.
        prompt: What the user says, to both sides.
        chat: The number of the chat the turn is of: refused unless it is the
          live one (model_server.check_turn()). None for whichever is live.

    Returns:
        The generate job, of the live chat (`chat`); it runs on a thread
        of its own.

    Raises:
        ValueError: The prompt is empty or not text, or `chat` is no
          number.
        NotFound, Conflict: As model_server.check_turn().
    """
    model_server_lib.check_prompt(prompt)
    if chat is not None and (
        isinstance(chat, bool) or not isinstance(chat, int)
    ):
      raise ValueError('Name the chat by its number.')
    record = self._sessions.get(session_id)
    with self._lock:
      model_server = self._model_servers.get(session_id)
      n = model_server.next_turn() if model_server is not None else None
      job = self._jobs.create(
          session_id,
          'generate',
          n,
          prompt,
          chat=model_server.chat if model_server is not None else None,
      )
      try:
        model_server_lib.check_turn(self.state(session_id), record, chat)
        self._sessions.turn_sent(session_id)
        turn = model_server.admit_turn(prompt, job['id'])
      except (errors.Conflict, errors.NotFound) as error:
        self._jobs.finish(job['id'], 'failed', str(error))
        logging.info(
            '%s: turn refused: %s',
            session_store.label(record),
            error,
        )
        raise
      self._last_jobs[session_id] = job['id']
    model_server.generate(turn)
    return self._jobs.get(job['id'])

  def cancel(self, job_id: str) -> dict[str, Any]:
    """Stops a turn whose captures are not both held; ends its chat.

    Args:
      job_id: The ID of the turn job to cancel.

    Returns:
      The updated job dictionary.

    Raises:
      Conflict: The job starts a Model Server, which is turned off
        instead, or a chat, or prepares a Session.
    """
    job = self._jobs.get(job_id)
    if not model_server_lib.cancellable(job):
      return job
    stop = None
    with self._lock:
      for model_server in self._model_servers.values():
        found = model_server.begin_cancel(job_id)
        if found is not None:
          stop = found
    if stop is not None:
      stop()
    return self._jobs.get(job_id)

  def close(self, session_id: str) -> None:
    """Turns a Session's Model Server off: stops both runners for good.

    Its live chat, if any, ends with it (model_server.TURNED_OFF). The
    devices stay held until no start of the Model Server can launch a
    runner any more: one still launching stops its runner once it returns.
    A close of a Model Server that is not on does nothing.

    Args:
      session_id: The session ID to close.

    Raises:
      Conflict: As check_close(): it is being turned off already.
    """
    self._sessions.get(session_id)
    with self._lock:
      check_close(self.state(session_id))
      model_server = self._model_servers.get(session_id)
      if model_server is not None and not model_server.begin_close():
        model_server = None
    if model_server is not None:
      try:
        model_server.end_live_chat(model_server_lib.TURNED_OFF)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception(
            'The live chat of %s could not be marked over', session_id
        )
      model_server.close()

  def shutdown(self) -> None:
    """Stops every runner: the server is exiting."""
    with self._lock:
      model_servers = list(self._model_servers.values())
      for model_server in model_servers:
        model_server.begin_shutdown()
    for model_server in model_servers:
      model_server.stop_runners()

  # ---- What the Model Servers ask of the registry.

  def _keep_reported(
      self,
      device: launcher.Device,
      build: str | None,
      info: runner_pb2.RunnerInfo,
  ) -> None:
    kept = runner_pb2.RunnerInfo()
    kept.CopyFrom(info)
    with self._lock:
      self._reported[(device.id, build or '')] = kept

  def _upload_lock(self, sha256: str) -> threading.Lock:
    """The lock held while uploading the model of a sha256."""
    with self._lock:
      return self._upload_locks.setdefault(sha256, threading.Lock())

  def _forget(self, model_server: model_server_lib.ModelServer) -> None:
    """Drops a Model Server that never came on.

    Called with the lock held.

    Args:
      model_server: The Model Server to forget.
    """
    if self._model_servers.get(model_server.owner_id) is model_server:
      del self._model_servers[model_server.owner_id]
