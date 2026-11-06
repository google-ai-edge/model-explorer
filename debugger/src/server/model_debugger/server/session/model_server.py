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

"""The Model Server of one Session: both sides, their chats and its turns.

A Model Server runs a runner per side, ref and target, each over one
connection that holds one model for the runner's whole life. Turned on, it
launches both runners and loads both models, and holds no chat: a chat is
started apart (start_chat()), with each side's settings, as CreateChat on
both sides, and a Session holds several over its life, one live at a time.
Starting a chat ends the live one; a chat whose start failed on a side is
never live, and another may be started. A chat is over once it takes no more
turns: a reply filled a side's context, a turn failed, was stopped, or did not
come back whole, or another chat was started; what it captured stays for
Debug, each chat in a capture of its own. The Model Server stays on through
all of that; it turns itself off, as Turn off does, only once a side's runner
is gone (runner_gone()), which ends its live chat too. A Session starts
once: a Model Server that ended can never be resumed, not even by this
server after a restart, and a start that did not come on (it failed, was
turned off, or the server stopped meanwhile) ends its Session too, with no
chat; another try is a new Session. Each side's own work is a side.Side's;
this module keeps the rules of both: how they start, how a chat starts, what
a turn does when one side fails, how a turn is stopped, and how the Model
Server is turned off. Which Session runs which Model Server, and which
device each holds, is model_servers'.

Every Model Server shares one lock with its registry (see Context); methods
documented as called with the lock held are the registry's to call inside a
critical section of its own.

The log tells a Model Server's life at INFO (on, off, each chat, each turn),
and each start, chat start or turn that fails at ERROR, in one line that
names the Session (session_store.label()) and says why as the UI does.
"""

from collections.abc import Callable
import concurrent.futures
import copy
import dataclasses
import pathlib
import shutil
import threading
from typing import Any

from absl import logging
from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_client
from model_debugger.server.session import capture_store
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.session import session_store
from model_debugger.server.session import side as side_lib
from model_debugger.server.tap import model_facts
from model_debugger.server.util import errors

SIDE_NAMES = capture_index.SIDE_NAMES

# Why a chat takes no more turns (chat_over()): Debug is named only for a chat
# with a turn to inspect.
CHAT_OVER = 'This chat is over.'
CHAT_OVER_CAPTURED = (
    'This chat is over; its turns can still be inspected in Debug.'
)
NOT_ON = 'Wait for the Model Server to come on.'
# Why the same is refused while it starts, and while it is being turned off.
STARTING = 'The Model Server is starting.'
STOPPING = 'The Model Server is being turned off.'
BUSY = 'Wait for the current task to finish.'
ENDED = 'This Session has ended.'
# Why a turn is refused while no chat is live, and an ended chat's turns.
NO_CHAT = 'Start a chat first.'
CHAT_ENDED = 'This chat has ended.'
# Why the live chat ended, as its notice begins.
NEW_CHAT = 'A new chat was started.'
TURNED_OFF = 'The Model Server was turned off.'
# Why a chat is over, as its record keeps it (`over_cause`): a reply filled a
# side's context. The text shown is derived from it (notice()).
CONTEXT_FULL = 'context_full'
# The job channel of each delta of a reply's Chunk.
_CHANNELS = {'content': 'content', 'reasoning_content': 'reasoning'}
# How often a Model Server with no job running asks its connections whether
# both runners are still there.
_WATCH_SECONDS = 0.5
# What a generate's progress says once its reply ended, while both sides'
# captures arrive: the only progress a generate has.
RETRIEVING = 'Retrieving the capture'
# What a chat start's progress says of each side.
STARTING_CHAT = 'Starting chat'


def chat_over(has_capture: bool) -> str:
  """Why a chat takes no more turns: every text of it is this one's.

  Args:
      has_capture: Whether the chat kept a turn, as its record's `has_capture`
        says: only then has Debug a turn to show.

  Returns:
      The reason string why the chat takes no more turns.
  """
  return CHAT_OVER_CAPTURED if has_capture else CHAT_OVER


def retrieving(received: int) -> str:
  """What a turn's progress says with `received` bytes of its capture held.

  The protocol announces each file's size as it starts, never the
  capture's, so only what arrived so far is said: in steps of 100 MB, then
  of 0.1 GB, so that a capture of gigabytes is told a few times a GB.

  Args:
      received: The number of bytes received so far.

  Returns:
      The progress description string.
  """
  if received >= 10**9:
    return f'{RETRIEVING} · {received / 10**9:.1f} GB so far'
  if received >= 10**8:
    return f'{RETRIEVING} · {received // 10**8 * 100} MB so far'
  return RETRIEVING


def does_not_fit(refusals: dict[str, runner_client.NoRoomError]) -> str:
  """What the user is told of a turn both sides refused for want of room.

  Args:
      refusals: Each side's refusal. Where the sides' counts differ, the side
        with the fewer tokens left is named.

  Returns:
      The message telling the user why the turn does not fit.
  """
  role = min(refusals, key=lambda r: (refusals[r].left, -refusals[r].tokens))
  tokens, left = refusals[role].tokens, refusals[role].left
  same = len({(r.tokens, r.left) for r in refusals.values()}) == 1
  where = '' if same else f'on the {SIDE_NAMES[role]} side '
  return (
      f"This input doesn't fit: {where}this turn needs {tokens} tokens, "
      f'the chat has {left} left. Shorten it and send again.'
  )


def runner_gone(side: str) -> str:
  """Why a Session ended once a side's runner is not connected.

  The live chat ends with it, saying so before chat_over().

  Args:
      side: The side whose runner is not connected.

  Returns:
      The explanation string why the session ended.
  """
  return (
      f'Runner closed: the connection to the {SIDE_NAMES[side]} runner is lost.'
  )


def ended_chat(reason: str, has_capture: bool) -> str:
  """What an ended chat says: why, then that it is over (chat_over()).

  Args:
      reason: Why it ended, a sentence or more.
      has_capture: Whether the chat kept a turn.

  Returns:
      The ended chat explanation string.
  """
  return f'{reason.rstrip(".")}. {chat_over(has_capture)}'


class _SideError(Exception):
  """One side's failure, named for the user."""

  def __init__(self, side: str, cause: Exception) -> None:
    super().__init__(f'{SIDE_NAMES[side]}: {cause}')
    self.side = side
    self.cause = cause


class _AbortedError(Exception):
  """The Model Server was turned off, or the other side failed, meanwhile."""


class _StoppedError(Exception):
  """The user stopped a turn before both its captures were held."""


@dataclasses.dataclass(frozen=True)
class SideState:
  """One side of a Model Server, as state() shows it.

  Attributes:
      role: 'ref' or 'target'.
      device: Where its runner runs.
      phase: The side's phase.
      build_id: The runtime version its runner answered with, once it did.
      online: Whether the connection to its runner is up, as the connection
        itself says: no device is asked.
  """

  role: str
  device: launcher.Device
  phase: str
  build_id: str | None
  online: bool


@dataclasses.dataclass(frozen=True)
class ModelServerState:
  """A Model Server at one moment, as state() shows it.

  Attributes:
      phase: 'starting', 'active', 'ending' or 'ended'.
      job_id: The running job, if any.
      operation: The running job's operation.
      sides: Both sides, ref first.
      chat: The number of its live chat, if any: the chat its runners hold,
        which takes turns.
  """

  phase: str
  job_id: str | None
  operation: str | None
  sides: tuple[SideState, ...]
  chat: int | None = None


@dataclasses.dataclass(frozen=True)
class Context:
  """What every Model Server shares with its registry.

  Attributes:
      sessions: The Session records.
      models: The registered models.
      jobs: The jobs the UI watches.
      launchers: What launches runners.
      captures: Where each chat's captures go.
      lock: Guards every Model Server's state, its sides' and the registry's;
        never held across a runner call.
      upload_lock: Returns the lock to hold while uploading the model of a
        sha256, one per sha256 for every runner; called without `lock` held.
      forget: Drops a Model Server that never came on from the registry, if it
        is still there, once its start ended; called with the lock held.
      reported: Keeps what a runner of a build (None for the one installed) on a
        device answered GetInfo with; called without `lock` held.
      capture_model: Returns the model a run of a Session loads: the run's own,
        or its copy that observes the tensors the Session chose
        (prepared_models); called without `lock` held.
      observe_modules: Returns the modules a PyTorch run of a Session observes
        beside the anchors (prepared_models' observe_modules(): those paired
        with the LiteRT-LM points chosen), given its record and the run; called
        without `lock` held, after `prepare`.
      prepare: Writes the copies a Session's runs load, given its record, the
        start's job and whether the start was turned off meanwhile; returns the
        record as prepared, which the sides start from (the anchors it captures
        are found as it is prepared), or None if the start was turned off.
        Called on the start's thread, before any runner launches, without `lock`
        held.
      loaded: Lets go of the copies a start prepared, given its job, once both
        sides loaded them or the start ended without; none is deleted before.
        Called on the start's thread, without `lock` held.
      convert: Converts the graphs of a Session's models that are not kept
        (model_graphs), given its record, the start's job and whether the start
        was turned off meanwhile; False if it was, and it raises if a graph
        cannot be converted, which fails the start. Called on the start's thread
        first, before `prepare` and before this Session's runners launch
        (another Session's may be running), without `lock` held.
      exported: Keeps the execution graph a PyTorch side's turn exported
        (model_graphs.ModelGraphs.exported()), given the side's model, the file
        and where it is from (`{session, chat, turn, side}`). Called once the
        turn is recorded, without `lock` held; it never raises.
  """

  sessions: session_store.SessionStore
  models: model_store.ModelStore
  jobs: jobs_lib.Jobs
  launchers: launcher.Devices
  captures: capture_store.CaptureStore
  lock: threading.RLock
  upload_lock: Callable[[str], threading.Lock]
  forget: Callable[['ModelServer'], None]
  reported: Callable[[launcher.Device, str | None, runner_pb2.RunnerInfo], None]
  capture_model: Callable[[dict[str, Any], dict[str, Any]], model_store.Model]
  prepare: Callable[
      [dict[str, Any], str, Callable[[], bool]], dict[str, Any] | None
  ]
  loaded: Callable[[str], None]
  convert: Callable[[dict[str, Any], str, Callable[[], bool]], bool] = (
      lambda record, job_id, aborted: True
  )
  exported: Callable[[str, pathlib.Path, dict[str, Any]], object] = (
      lambda artifact, path, source: None
  )
  observe_modules: Callable[
      [dict[str, Any], dict[str, Any]], list[dict[str, Any]]
  ] = lambda record, run: []


@dataclasses.dataclass
class Turn:
  """A generate job in flight, admitted by admit_turn() for generate().

  Guarded by the Context's lock.

  Attributes:
      job_id: The generate job.
      chat: The chat it is a turn of, the live one.
      index: The chat's capture.
      n: The turn's number in its chat.
      prompt: What the user says.
      cancel_requested: Whether the user, or turning off, stopped it.
      completed: Both completions arrived: the turn is the runners' now, cancel
        or not.
      held: Both captures are held: a Stop no longer drops the turn.
      stopped: Sides this server cancelled because the other failed: their
        errors are no cause.
      replying: Sides whose reply began: a chunk of it arrived.
      no_room: Sides that refused the turn as having no room for it in the
        context: they changed nothing.
      refused: Both sides refused it so: the turn did not happen, and a Stop has
        nothing to stop.
  """

  job_id: str
  chat: int
  index: capture_index.CaptureIndex
  n: int
  prompt: str
  cancel_requested: bool = False
  completed: bool = False
  held: bool = False
  stopped: set[str] = dataclasses.field(default_factory=set)
  replying: set[str] = dataclasses.field(default_factory=set)
  no_room: set[str] = dataclasses.field(default_factory=set)
  refused: bool = False


def check_prompt(prompt: object) -> None:
  """Refuses a turn without a prompt, before anything else is checked.

  Args:
      prompt: The prompt the turn was asked with, as the request holds it.

  Raises:
      ValueError: The prompt is not text, or is empty.
  """
  if not isinstance(prompt, str) or not prompt:
    raise ValueError('Type a prompt.')


def in_use(device: launcher.Device) -> str:
  """Why a start refuses a device another Model Server holds."""
  return f'{device.name} is in use by another Session.'


def no_app(device: launcher.Device) -> str:
  """Why a start refuses a phone without the Runner App."""
  return f'No Runner App is installed on {device.name}.'


def ended(owner: dict[str, Any]) -> bool:
  """Whether a Session was turned on once: it never starts again.

  A Session starts once (`started`, set as its start is admitted): a runner
  cannot be reattached, so a Session that ran never runs again (ENDED), and
  a start that did not come on ends it too. A record of an older server,
  or of a Session imported, says so by `execution_ended`.

  Args:
      owner: The Session's record.

  Returns:
      True if the Session was started or ended; False otherwise.
  """
  return bool(owner.get('started') or owner.get('execution_ended'))


def notice(
    chat: dict[str, Any], index: capture_index.CaptureIndex | None
) -> str:
  """What a chat's record tells the user now.

  A chat that a full context ended keeps that fact (`over_cause`), and its
  text is derived when the chat is shown: ending() of each side whose
  context the turn filled, then chat_over(). Any other notice is kept as
  its text, and so is a full context's text that an older record kept.

  Args:
      chat: The chat, as its Session's record holds it.
      index: The chat's capture; None when it cannot be read.

  Returns:
      The notice string for the chat.
  """
  cause = chat.get('over_cause')
  if not isinstance(cause, dict) or cause.get('code') != CONTEXT_FULL:
    return chat.get('notice', '')
  full = []
  if index is not None:
    try:
      full = _context_full(index, cause['turn'])
    except (errors.NotFound, errors.Conflict):
      pass  # The turn cannot be read; reading the chat says why.
  over = chat_over(bool(chat.get('has_capture')))
  return f'{" ".join(full or ["A reply filled its context."])} {over}'


def _context_full(index: capture_index.CaptureIndex, n: int) -> list[str]:
  """What ending() says of each side whose context turn `n` filled.

  Args:
      index: The capture index for the chat.
      n: The turn number.

  Returns:
      A list of strings describing context-full status for each affected side.

  Raises:
      NotFound, Conflict: As index.turn().
  """
  return [
      f'{SIDE_NAMES[role]}: {capture_index.ending(index.sides[role], entry)}.'
      for role, entry in index.turn(n)['sides'].items()
      if entry['stop_reason'] == 'context_full'
  ]


def check_not_ended(
    state: ModelServerState | None, record: dict[str, Any]
) -> None:
  """Refuses what needs a Session not ended: it never goes on again.

  A Session has ended once its Model Server, turned on once, is neither
  starting nor on any more (ENDED, as a start then refuses, check_start):
  its chats went with its runners. The rule of a chat's start and of a
  turn, and the listing's of whether a Session is read as ended.

  Args:
      state: The Session's Model Server now, or None if it never ran here.
      record: The Session's record.

  Raises:
      Conflict: The Session has ended (ENDED).
  """
  if ended(record) and (
      state is None or state.phase not in ('starting', 'active')
  ):
    raise errors.Conflict(ENDED)


def check_start(
    owner: dict[str, Any],
    available: dict[str, launcher.Device],
    held: Callable[[str], bool],
    models: model_store.ModelStore,
) -> dict[str, launcher.Device]:
  """Refuses turning on a Session's Model Server that cannot start now.

  The rule of a start, for a Session's create to apply.

  Args:
      owner: The Session's record.
      available: The devices connected now, by id.
      held: Whether a Model Server holds the device of an id.
      models: The registered models.

  Returns:
      Each side's device.

  Whether the runner build a side chose can still be launched is not
  checked here: its launcher refuses it, on every start (side.Side).

  Raises:
      Conflict: The Session ended, a device is missing, held or has no
        Runner App, a side on a device that chooses its runner build
        chose none, or a run's model is not on the server (an older
        server's, kept outside its workspace: model_store.upgrade()).
      ValueError: The Session lacks a run.
  """
  if ended(owner):
    raise errors.Conflict(ENDED)
  devices = {}
  for role in capture_index.SIDES:
    run = _run(owner, role)
    device_id = run['device']
    device = available.get(device_id)
    if device is None:
      raise errors.Conflict(
          f'{SIDE_NAMES[role]}: the device {device_id} is not connected.'
      )
    if device.without_app:
      raise errors.Conflict(f'{SIDE_NAMES[role]}: {no_app(device)}')
    if run.get('runtime') == model_store.PYTORCH and (
        device.id != launcher.LOCAL_DEVICE_ID
    ):
      raise errors.Conflict(
          f'{SIDE_NAMES[role]}: a PyTorch model runs on this computer only.'
      )
    if device.chooses_build and not run.get('runnerBuild'):
      raise errors.Conflict(f'{SIDE_NAMES[role]}: choose a runner version.')
    if not device.shared:
      if any(other.id == device.id for other in devices.values()):
        raise errors.Conflict(
            f'{device.name} runs one runner at a time; choose '
            'different devices for Reference and Target.'
        )
      if held(device.id):
        raise errors.Conflict(f'{SIDE_NAMES[role]}: {in_use(device)}')
    try:
      model = models.get(run.get('artifact', ''))
    except errors.NotFound:
      raise errors.Conflict(
          f'{SIDE_NAMES[role]}: its model is not on the Server; choose one.'
      ) from None
    if not model.present():
      raise errors.Conflict(
          f'{SIDE_NAMES[role]}: {model.name} is not on the Server; '
          'upload or download it.'
      )
    devices[role] = device
  return devices


def check_on(state: ModelServerState | None) -> None:
  """Refuses what needs a Model Server on, as it starts or ends, or not.

  Args:
      state: The current model server state, or None.

  Raises:
      Conflict: It starts (STARTING), is being turned off (STOPPING), or
        is not on otherwise (NOT_ON).
  """
  phase = state.phase if state is not None else None
  if phase == 'starting':
    raise errors.Conflict(STARTING)
  if phase == 'ending':
    raise errors.Conflict(STOPPING)
  if phase != 'active':
    raise errors.Conflict(NOT_ON)


def _check_on(state: ModelServerState | None, record: dict[str, Any]) -> None:
  """Refuses what needs the Model Server on, idle, both runners there.

  Args:
      state: The current model server state, or None.
      record: The Session's record dictionary.

  Raises:
      Conflict: The Session has ended (check_not_ended); its Model Server
        is not on (check_on()); a job runs (BUSY); or a side's runner is not
        connected, as its connection says now (runner_gone()).
  """
  check_not_ended(state, record)
  check_on(state)
  if state.job_id:
    raise errors.Conflict(BUSY)
  for side in state.sides:
    if not side.online:
      raise errors.Conflict(runner_gone(side.role))


def check_new_chat(
    state: ModelServerState | None, record: dict[str, Any]
) -> None:
  """Refuses starting a chat the Model Server cannot start now.

  The rule of start_chat, for it to apply and the listing to show; the
  chat's settings are checked before (session_store's check_generation).
  A live chat refuses none: it ends as the new one starts.

  Args:
      state: The Session's Model Server now, or None if it never ran here.
      record: The Session's record.

  Raises:
      Conflict: As _check_on(): the Session has ended, its Model Server is
        not on, a job runs (a turn, or another chat's start), or a side's
        runner is not connected.
  """
  _check_on(state, record)


def check_turn(
    state: ModelServerState | None,
    record: dict[str, Any],
    chat: int | None = None,
) -> None:
  """Refuses a turn the Model Server cannot run now.

  The rule of generate, for it to apply; the prompt is checked before
  (check_prompt).

  Args:
      state: The Session's Model Server now, or None if it never ran here.
      record: The Session's record.
      chat: The chat the turn is of, as the page that sends it shows it; None
        for whichever is live.

  Raises:
      NotFound: The Session has no chat `chat`.
      Conflict: As _check_on() (a turn running among the jobs); the chat
        named is not the live one (CHAT_ENDED), as when another page
        started a chat meanwhile; or no chat is live (NO_CHAT).
  """
  if chat is not None and all(one['n'] != chat for one in record['chats']):
    raise errors.NotFound(f'Chat {chat} does not exist.')
  _check_on(state, record)
  if chat is not None and state.chat != chat:
    raise errors.Conflict(CHAT_ENDED)
  if state.chat is None:
    raise errors.Conflict(NO_CHAT)


def check_chat_turn(
    state: ModelServerState | None, record: dict[str, Any], chat: dict[str, Any]
) -> None:
  """Refuses a turn of a Session's chat: of the live one, as check_turn.

  The listing's rule of each chat's Send.

  Args:
      state: The Session's Model Server now, or None if it never ran here.
      record: The Session's record.
      chat: One of its chats.

  Raises:
      Conflict: The chat is not the live one (CHAT_ENDED), or as
        check_turn().
  """
  if state is None or state.phase != 'active' or state.chat != chat['n']:
    raise errors.Conflict(CHAT_ENDED)
  check_turn(state, record, chat['n'])


def cancellable(job: dict[str, Any]) -> bool:
  """Whether a cancel of a job may stop a turn; checked before the lock.

  Args:
      job: The job, as jobs.Jobs answers it.

  Returns:
      False if the job already finished: the cancel answers it unchanged.

  Raises:
      Conflict: The job starts a Model Server, which is turned off instead,
        or starts a chat.
  """
  if job['status'] in jobs_lib.TERMINAL:
    return False
  if job['operation'] == 'chat':
    raise errors.Conflict('Starting a chat cannot be stopped.')
  if job['operation'] != 'generate':
    raise errors.Conflict(
        'Starting a Model Server cannot be cancelled; turn it off instead.'
    )
  return True


def _run(record: dict[str, Any], side: str) -> dict[str, Any]:
  for run in record.get('runs', []):
    if run.get('id') == side:
      return run
  raise ValueError(f'The Session has no {SIDE_NAMES[side]} run.')


class ModelServer:
  """The Model Server of one owner Session.

  Its attributes are guarded by the Context's lock.

  Attributes:
      owner_id: The Session that runs it.
      sides: Both sides, ref first.
      phase: 'starting', 'active', 'ending' or 'ended'.
      chat: The number of its live chat, if any.
      index: The live chat's capture index.
      job_id: The running job, if any.
      operation: The running job's operation.
      turn: The generate job in flight, if any.
      became_active: Whether it ever came on.
      aborted: Set by close and by a failed start: runners launched afterwards
        stop.
      closing: Whether it is being turned off.
      starting: Whether its start may still launch a runner: a closed Model
        Server keeps its devices until it no longer can.
      turning_off: Set once it is being turned off, or the server exits: its
        watch of the runners (_watch) then ends.
  """

  def __init__(
      self, owner_id: str, sides: dict[str, side_lib.Side], context: Context
  ) -> None:
    """Holds a Model Server that is starting.

    Args:
        owner_id: The Session that runs it.
        sides: Both sides, ref first.
        context: What it shares with its registry.
    """
    self.owner_id = owner_id
    self.sides = sides
    self.phase = 'starting'
    self.chat: int | None = None
    self.index: capture_index.CaptureIndex | None = None
    self.job_id: str | None = None
    self.operation: str | None = None
    self.turn: Turn | None = None
    self.became_active = False
    self.aborted = False
    self.closing = False
    self.starting = True
    self.turning_off = threading.Event()
    self._context = context
    self._sessions = context.sessions
    self._models = context.models
    self._jobs = context.jobs
    self._launchers = context.launchers
    self._captures = context.captures
    self._lock = context.lock
    # Each side's model, once it started; a PyTorch side's turns may
    # export its graph.
    self._side_models: dict[str, model_store.Model] = {}
    # What each side's capture entry holds of its start, once loaded:
    # capture_index.side_entry()'s arguments but the chat's, and what
    # the model's files say. Each chat's capture starts from them.
    self._loaded: dict[str, dict[str, Any]] = {}

  # ---- What the registry reads.

  def state(self) -> ModelServerState:
    """The Model Server now; called with the lock held."""
    return ModelServerState(
        phase=self.phase,
        job_id=self.job_id,
        operation=self.operation,
        sides=tuple(
            SideState(
                role=side.role,
                device=side.device,
                phase=side.phase,
                build_id=(
                    side.info.runtime.version if side.info is not None else None
                ),
                online=side.client is not None and side.client.online(),
            )
            for side in self.sides.values()
        ),
        chat=self.chat,
    )

  # ---- Starting.

  @classmethod
  def admit(
      cls,
      owner: dict[str, Any],
      devices: dict[str, launcher.Device],
      job_id: str,
      context: Context,
  ) -> 'ModelServer':
    """Makes a starting Model Server, once check_start() passed.

    Called with the lock held.

    Args:
        owner: The Session's record.
        devices: Each side's device, as check_start() answered.
        job_id: The initialize job.
        context: What the Model Server shares with its registry.

    Returns:
        The Model Server, starting, with the job as its running one.
    """
    sides = {}
    for role in capture_index.SIDES:
      run = _run(owner, role)
      model = context.models.get(run.get('artifact', ''))
      sides[role] = side_lib.Side(
          role,
          devices[role],
          run.get('runnerBuild'),
          context.lock,
          model.path if model.runtime == model_store.PYTORCH else None,
      )
    model_server = cls(owner['id'], sides, context)
    model_server.job_id = job_id
    model_server.operation = 'initialize'
    return model_server

  def initialize(self, owner: dict[str, Any]) -> None:
    """Starts the Model Server admitted, on a thread of its own.

    Args:
        owner: The Session's record, as admit() was given it.
    """
    with self._lock:
      job_id = self.job_id  # admit()'s; only this start clears it.
    threading.Thread(
        target=self._run_start,
        args=(owner, job_id),
        name=f'initialize-{self.owner_id}',
        daemon=True,
    ).start()

  def _run_start(self, owner: dict[str, Any], job_id: str) -> None:
    """Converts graphs, prepares copies, starts both sides: no chat."""
    try:
      try:
        self._convert(owner, job_id)
        prepared = self._context.prepare(owner, job_id, self._aborted)
        if prepared is None:
          raise _AbortedError()
        self._start_sides(prepared, job_id)
      finally:
        self._context.loaded(job_id)
      with self._lock:
        if self.aborted:
          raise _AbortedError()
        self.phase = 'active'
        self.became_active = True
        self.starting = False
        self.job_id = self.operation = None
        for side in self.sides.values():
          side.phase = 'active'
      # A runner cannot be reattached, so this record can never run again.
      self._sessions.set_flags(owner['id'], execution_ended=True)
      self._jobs.finish(job_id, 'completed')
      logging.info('%s: the Model Server is on', self._label())
    except Exception as error:  # pylint: disable=broad-exception-caught
      self._fail_start(job_id, error)
      return
    threading.Thread(
        target=self._watch,
        name=f'watch-{self.owner_id}',
        daemon=True,
    ).start()

  def _start_sides(self, owner: dict[str, Any], job_id: str) -> None:
    """Starts both sides in parallel; the first failure stops the other."""
    # Neither side loads before both sides' requests are checked: settings a
    # runner refuses fail the start before any upload.
    checked = threading.Barrier(len(capture_index.SIDES))
    with concurrent.futures.ThreadPoolExecutor(
        len(capture_index.SIDES)
    ) as pool:
      futures = {
          pool.submit(
              self._start_side,
              role,
              owner,
              job_id,
              checked,
          ): role
          for role in capture_index.SIDES
      }
      failure = None
      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as error:  # pylint: disable=broad-exception-caught
          if failure is None or isinstance(failure, _AbortedError):
            failure = error
          checked.abort()
          with self._lock:
            self.aborted = True
          # Stopping the runners makes the other side's call fail at
          # once.
          self.stop_runners()
      if failure is not None:
        raise failure

  def _start_side(
      self,
      role: str,
      owner: dict[str, Any],
      job_id: str,
      checked: threading.Barrier,
  ) -> None:
    """Starts one side's runner and loads its model.

    Args:
        role: The side role ('baseline' or 'target').
        owner: The Session's record.
        job_id: The start job identifier.
        checked: Barrier synchronizing the load check across sides.
    """
    side = self.sides[role]
    try:
      run = _run(owner, role)
      model = self._models.get(run.get('artifact', ''))
      with self._lock:
        self._side_models[role] = model
      # What the side loads: the model, or its copy that observes the
      # tensors chosen; a chat's sampling is the model's either way.
      loaded_model = self._context.capture_model(owner, run)
      self._progress(job_id, role, 'starting', 'Starting runner')
      if not side.launch(self._launchers, lambda: self.aborted):
        raise _AbortedError()
      self._set_phase(role, 'connecting')
      info = side.connect()
      self._context.reported(side.device, side.build, info)
      # Both requests are checked before anything is uploaded.
      load = options.load_request(
          run,
          loaded_model,
          info,
          self._context.observe_modules(owner, run),
      )
      try:
        checked.wait()
      except threading.BrokenBarrierError:
        raise _AbortedError() from None
      self._progress(job_id, role, 'loading', 'Loading model')
      loaded = self._load(job_id, side, load, loaded_model)
      self._models.report_loaded(model.artifact, loaded)
      facts = {
          'entry': {
              'model_sha256': loaded_model.sha256,
              'model_name': loaded_model.name,
              'model_size': loaded_model.size,
              'device_id': side.device.id,
              'device_name': side.device.name,
              'info': info,
              'load': load,
              'loaded': loaded,
          },
          # What the model's files say, by which two models are
          # compared.
          'described': copy.deepcopy(
              model_facts.described(loaded_model.path, loaded_model.sha256)
          ),
          'context_length': options.context_length(run),
      }
      with self._lock:
        self._loaded[role] = facts
    except _AbortedError:
      raise
    except Exception as error:
      # Refusals and failures the user can act on carry their own message.
      expected = (
          runner_client.Error,
          ValueError,
          RuntimeError,
          errors.NotFound,
      )
      if not isinstance(error, expected):
        logging.exception('%s failed to start', SIDE_NAMES[role])
      raise _SideError(role, error) from error

  def _load(
      self,
      job_id: str,
      side: side_lib.Side,
      request: runner_pb2.LoadRequest,
      model: model_store.Model,
  ) -> runner_pb2.LoadResponse:
    """Loads the model, uploading it first if the runner lacks it."""
    # Runners on one machine may share a store: one upload of a model at a
    # time, to any runner.
    upload_lock = self._context.upload_lock(model.sha256)
    shown = -1

    def on_progress(sent: int, total: int) -> None:
      nonlocal shown
      percent = sent * 100 // total if total else 100
      if percent != shown:
        shown = percent
        # The side's phase follows: uploading until it is all sent.
        self._progress(
            job_id,
            side.role,
            'uploading',
            f'Uploading model · {percent}%',
        )
      if sent == total:
        self._progress(job_id, side.role, 'loading', 'Loading model')

    return side.load(request, model.path, upload_lock, on_progress)

  def _convert(self, owner: dict[str, Any], job_id: str) -> None:
    """Converts the graphs of the Session's models (Context.convert)."""
    if not self._context.convert(owner, job_id, self._aborted):
      raise _AbortedError()

  def _fail_start(self, job_id: str, error: Exception) -> None:
    """Stops every runner of a start that failed or was turned off.

    The start ends its Session (ended()): its record keeps why, as its
    `notice`, and the Model Server leaves the registry once its runners
    are stopped. The job ends failed whatever fails meanwhile: no start
    is left running.

    Args:
        job_id: The identifier of the failed start job.
        error: The exception that caused the failure.
    """
    with self._lock:
      self.aborted = True
      closing = self.closing
    stopped = closing or isinstance(error, _AbortedError)
    message = (
        'The Model Server was turned off while starting.'
        if stopped
        else str(error)
    )
    if stopped:
      logging.info('%s: start stopped: %s', self._label(), message)
    else:
      logging.error('%s: start failed: %s', self._label(), message)
    try:
      try:
        self.stop_runners()
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception('The runners of a failed start did not stop')
      try:
        self._sessions.set_flags(self.owner_id, notice=message)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception(
            'The record of %s did not keep why its start failed',
            self.owner_id,
        )
      with self._lock:
        self.job_id = self.operation = None
        self.starting = False
        if self.closing:
          # Every runner of this start is stopped: close may end it.
          self._end_closed()
        else:
          self._context.forget(self)
    finally:
      self._jobs.finish(job_id, 'failed', message)

  # ---- Chats.

  def chat_requests(
      self, generation: dict[str, Any]
  ) -> dict[str, runner_pb2.CreateChatRequest]:
    """Each side's CreateChat of a chat's settings, as its runner takes it.

    Called with the lock held, once the Model Server is on.

    Args:
        generation: Each side's settings, as session_store's check_generation()
          returns them.

    Returns:
        A mapping from side role to its CreateChatRequest.

    Raises:
        ValueError: A side's runner does not take a setting; the error
          names the side.
    """
    requests = {}
    for role, side in self.sides.items():
      model = self._side_models[role]
      try:
        # The model's own sampling, as its runner reported it loading.
        model = self._models.get(model.artifact)
      except errors.NotFound:
        pass
      try:
        requests[role] = options.create_chat_request(
            generation[role],
            side.info,
            self._loaded[role]['context_length'],
            model.sampler_defaults,
        )
      except ValueError as error:
        raise ValueError(f'{SIDE_NAMES[role]}: {error}') from None
    return requests

  def admit_chat(self, job_id: str) -> int | None:
    """Makes a chat's start the running job, once check_new_chat() passed.

    Called with the lock held. The live chat, if any, is no longer: its
    runners' chats are replaced as the new one starts, and the caller
    marks it over (NEW_CHAT).

    Args:
        job_id: The chat job.

    Returns:
        The number of the chat that was live, if one was.
    """
    live = self.chat
    self.chat = None
    self.index = None
    self.job_id = job_id
    self.operation = 'chat'
    for side in self.sides.values():
      side.phase = 'initializing'
    return live

  def start_chat(
      self,
      job_id: str,
      generation: dict[str, Any],
      requests: dict[str, runner_pb2.CreateChatRequest],
  ) -> None:
    """Creates a chat admitted on both sides, on a thread of its own.

    Args:
        job_id: The chat job, as admit_chat() was given it.
        generation: Each side's settings, as the chat's record keeps them.
        requests: Each side's CreateChat (chat_requests()).
    """
    threading.Thread(
        target=self._run_chat,
        args=(job_id, generation, requests),
        name=f'chat-{self.owner_id}',
        daemon=True,
    ).start()

  def _run_chat(
      self,
      job_id: str,
      generation: dict[str, Any],
      requests: dict[str, runner_pb2.CreateChatRequest],
  ) -> None:
    """Sends both CreateChats in parallel; the chat is live once both did.

    A side that fails fails the start, its error naming the side: the
    chat is not kept, and the Model Server stays on with no live chat,
    unless the side's runner is gone, which ends the Session as _watch
    would. Each side's chat is the runner's whatever the other's did: the
    next start replaces both. A chat its record holds already, that does
    not come live (turned off meanwhile, its capture not made), is over.

    Args:
        job_id: The chat job identifier.
        generation: Each side's settings dictionary.
        requests: Each side's CreateChatRequest.
    """
    added = None
    try:
      with concurrent.futures.ThreadPoolExecutor(
          len(capture_index.SIDES)
      ) as pool:
        futures = {
            role: pool.submit(self._create_chat, job_id, role, request)
            for role, request in requests.items()
        }
      for role in capture_index.SIDES:  # Ref first, whichever failed.
        futures[role].result()
      with self._lock:
        if self.closing:
          raise _AbortedError()
      added = self._sessions.add_chat(self.owner_id, generation)
      entries = {}
      for role in capture_index.SIDES:
        with self._lock:
          facts = self._loaded[role]
        entry = capture_index.side_entry(**facts['entry'], chat=requests[role])
        entry['model'].update(copy.deepcopy(facts['described']))
        entries[role] = entry
      index = self._captures.create(self.owner_id, added['n'], entries)
      with self._lock:
        if self.closing:
          raise _AbortedError()
        self.chat = added['n']
        self.index = index
        self.job_id = self.operation = None
        for side in self.sides.values():
          side.phase = 'active'
      logging.info('%s: Chat %d started', self._label(), added['n'])
      self._jobs.finish(job_id, 'completed')
    except Exception as error:  # pylint: disable=broad-exception-caught
      self._fail_chat(job_id, error, added['n'] if added else None)

  def _create_chat(
      self, job_id: str, role: str, request: runner_pb2.CreateChatRequest
  ) -> None:
    """Creates one side's chat; its failure is a _SideError."""
    self._progress(job_id, role, 'initializing', STARTING_CHAT)
    try:
      self.sides[role].create_chat(request)
    except Exception as error:
      if not isinstance(error, runner_client.Error):
        logging.exception('%s CreateChat failed', SIDE_NAMES[role])
      raise _SideError(role, error) from error

  def _fail_chat(
      self, job_id: str, error: Exception, added: int | None = None
  ) -> None:
    """Ends the job of a chat that did not start; no chat is live.

    A side whose runner is gone ends the Session (_end_session()) before
    the job ends; else the Model Server stays on for another start.

    Args:
        job_id: The chat job.
        error: Why the chat did not start.
        added: The chat's number, if its record holds it already: it is marked
          over, saying why.
    """
    with self._lock:
      closing = self.closing
      if not closing:
        self.job_id = self.operation = None
        for side in self.sides.values():
          side.phase = 'active'
      gone = self._gone() if not closing else []
    message = (
        TURNED_OFF
        if closing or isinstance(error, _AbortedError)
        else str(error)
    )
    # A start the rules or a runner refused, or that was turned off, is
    # no failure of the server's.
    cause = error.cause if isinstance(error, _SideError) else error
    refused = closing or isinstance(
        cause,
        (
            _AbortedError,
            runner_client.NoRoomToStartError,
            errors.Conflict,
            ValueError,
        ),
    )
    logging.log(
        logging.INFO if refused else logging.ERROR,
        '%s: a chat did not start: %s',
        self._label(),
        message,
    )
    try:
      if added is not None:
        try:
          self._sessions.set_chat_flags(
              self.owner_id,
              added,
              over=True,
              notice=ended_chat(message, False),
          )
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception(
              'Chat %d of %s could not be marked over',
              added,
              self.owner_id,
          )
      if gone:
        self._end_session(gone[0])
    finally:
      self._jobs.finish(job_id, 'failed', message)

  # ---- Turns.

  def next_turn(self) -> int | None:
    """The live chat's next turn number; called with the lock held."""
    index = self.index
    return index.next_turn() if index is not None else None

  def admit_turn(self, prompt: str, job_id: str) -> Turn:
    """Makes a generate job the running one, once check_turn() passed.

    Called with the lock held.

    Args:
        prompt: What the user says.
        job_id: The generate job, created for next_turn()'s number.

    Returns:
        The turn of the live chat, numbered as next_turn() answers, for
        generate() to run.
    """
    # A live chat has its index.
    turn = Turn(job_id, self.chat, self.index, self.index.next_turn(), prompt)
    self.job_id = job_id
    self.operation = 'generate'
    self.turn = turn
    for side in self.sides.values():
      side.phase = 'generating'
    return turn

  def generate(self, turn: Turn) -> None:
    """Sends a turn admitted to both sides and stores its capture.

    It runs on a thread of its own.

    Args:
        turn: The admitted turn to generate.
    """
    threading.Thread(
        target=self._run_generate,
        args=(turn,),
        name=f'generate-{self.owner_id}-{turn.chat}-{turn.n}',
        daemon=True,
    ).start()

  def _run_generate(self, turn: Turn) -> None:
    """Executes turn generation, handles completions and failures.

    Args:
        turn: The turn to run.
    """
    try:
      results = self._send(turn)
      # Ref first, whichever side ended first.
      failures = {
          role: results[role]
          for role in capture_index.SIDES
          if isinstance(results[role], Exception) and role not in turn.stopped
      }
      if turn.refused:
        self._refuse_turn(turn, failures)
      elif not failures:
        self._finish_turn(turn, results)
      else:
        self._end_chat(turn, failures)
    except _StoppedError:
      self._end_chat(turn, {})
    except _SideError as error:
      message = str(error)
      if isinstance(error.cause, capture_index.Rejected):
        # The turn is kept, and shown with why it was rejected: the
        # chat's notice says no more than which turn it was.
        logging.error('%s: %s', self._label(), error)
        message = f'Turn {turn.n} was rejected'
      self._end_chat(turn, {error.side: error.cause}, message)
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.exception('Turn %d failed', turn.n)
      self._end_chat(turn, {}, str(error))

  def _send(self, turn: Turn) -> dict[str, runner_pb2.Completion | Exception]:
    """Runs one Send per side; returns each side's Completion or error.

    A side that fails cancels the other, since the sides are then out of
    step. A side that refuses the turn for want of room in its context
    changed nothing, and the other may refuse it too: that one is
    cancelled only once its reply begins, and then the turn has happened
    on one side only. Both refusing, the turn is `refused`.

    Args:
        turn: The turn to send to each side.

    Returns:
        Mapping from side role to runner Completion or Exception.
    """

    def other_of(role: str) -> str:
      return capture_index.SIDES[1 - capture_index.SIDES.index(role)]

    def send(role: str) -> runner_pb2.Completion | Exception:
      def on_chunk(chunk: runner_pb2.Chunk) -> None:
        with self._lock:
          turn.replying.add(role)
          # The other side refused the turn: this reply is stopped.
          unwanted = other_of(role) in turn.no_room
          stop = unwanted and role not in turn.stopped
          if stop:
            turn.stopped.add(role)
        if stop:
          self.sides[role].cancel()
        if unwanted:
          return
        delta = chunk.WhichOneof('delta')
        if delta in _CHANNELS:
          self._jobs.emit(
              turn.job_id,
              'delta',
              runId=role,
              channel=_CHANNELS[delta],
              text=getattr(chunk, delta),
          )

      other = other_of(role)
      try:
        return self.sides[role].send(turn.prompt, on_chunk)
      except runner_client.NoRoomError as error:
        with self._lock:
          turn.no_room.add(role)
          stop = other in turn.replying and other not in turn.stopped
          if stop:
            turn.stopped.add(other)
        if stop:
          self.sides[other].cancel()
        return error
      except Exception as error:  # pylint: disable=broad-exception-caught
        if not isinstance(error, runner_client.Error):
          logging.exception('%s Send failed', SIDE_NAMES[role])
        with self._lock:
          if role not in turn.stopped:  # Not itself a consequence.
            turn.stopped.add(other)
        self.sides[other].cancel()
        return error

    with concurrent.futures.ThreadPoolExecutor(
        len(capture_index.SIDES)
    ) as pool:
      results = dict(
          zip(capture_index.SIDES, pool.map(send, capture_index.SIDES))
      )
    with self._lock:
      turn.completed = all(
          isinstance(result, runner_pb2.Completion)
          for result in results.values()
      )
      # A Stop that came meanwhile cancelled both sides: the chat ends.
      turn.refused = not turn.cancel_requested and all(
          isinstance(result, runner_client.NoRoomError)
          for result in results.values()
      )
    return results

  def _refuse_turn(
      self, turn: Turn, refusals: dict[str, runner_client.NoRoomError]
  ) -> None:
    """Ends the job of a turn both sides refused for want of room.

    The turn did not happen: nothing is captured, the chat goes on as
    before it, and the job fails saying how many tokens the turn needs
    and the chat has left (does_not_fit()), each side's counts its
    `does_not_fit`.

    Args:
        turn: The refused turn.
        refusals: Each side's NoRoomError refusal.
    """
    message = does_not_fit(refusals)
    logging.info(
        '%s: Chat %d turn %d refused: %s',
        self._label(),
        turn.chat,
        turn.n,
        message,
    )
    self._end_turn(
        turn,
        'failed',
        message,
        does_not_fit={
            role: {'tokens': refusal.tokens, 'left': refusal.left}
            for role, refusal in refusals.items()
        },
    )

  def _finish_turn(
      self,
      turn: Turn,
      completions: dict[str, runner_pb2.Completion],
  ) -> None:
    """Fetches both sides' captures, then records the turn.

    While they arrive, the job's progress says how much of both has
    arrived (retrieving()). A turn whose contents the index rejects is
    recorded as rejected, its files kept as the evidence, and ends the
    chat. A turn that filled a side's context is recorded, and ends the
    chat too; the Model Server stays on for another.

    Args:
        turn: The turn being completed.
        completions: Each side's Completion.
    """
    index = turn.index
    for role in capture_index.SIDES:
      self._progress(turn.job_id, role, 'transferring', RETRIEVING)
    received = dict.fromkeys(capture_index.SIDES, 0)
    shown = [RETRIEVING]
    counting = threading.Lock()

    def report(role: str, count: int) -> None:
      with counting:
        received[role] = count
        text = retrieving(sum(received.values()))
        if text == shown[0]:
          return
        shown[0] = text
        # In order: a later count is never said before an earlier.
        self._jobs.emit(turn.job_id, 'progress', message=text)

    try:
      with concurrent.futures.ThreadPoolExecutor(
          len(capture_index.SIDES)
      ) as pool:
        futures = [
            pool.submit(self._fetch_capture, role, index, turn.n, report)
            for role in capture_index.SIDES
        ]
        for future in futures:
          future.result()
      with self._lock:
        turn.held = not turn.cancel_requested
      if not turn.held:
        raise _StoppedError()
      index.add_turn(
          turn.n,
          turn.prompt,
          {
              role: capture_index.completion_dict(completion)
              for role, completion in completions.items()
          },
      )
    except capture_index.Rejected as error:
      # Recorded as rejected: its files are the evidence, kept.
      self._sessions.set_chat_flags(self.owner_id, turn.chat, has_capture=True)
      raise _SideError(error.side, error) from error
    except Exception:
      # The turn is not recorded, so neither are its files.
      for role in capture_index.SIDES:
        shutil.rmtree(index.turn_directory(role, turn.n), ignore_errors=True)
      raise
    self._sessions.set_chat_flags(self.owner_id, turn.chat, has_capture=True)
    self._keep_exported(turn)
    full = bool(_context_full(index, turn.n))
    if full:
      # The runner refuses every Send after a reply that filled its
      # context: the chat is over, its turn kept. The record keeps the
      # fact; notice() derives its text.
      self._end_live_chat(
          turn.chat,
          over_cause={'code': CONTEXT_FULL, 'turn': turn.n},
      )
    logging.info(
        '%s: Chat %d turn %d completed%s',
        self._label(),
        turn.chat,
        turn.n,
        '; a reply filled its context, the chat is over' if full else '',
    )
    self._end_turn(turn, 'completed', '')

  def _keep_exported(self, turn: Turn) -> None:
    """Hands each PyTorch side's execution graph of a turn to be kept.

    In role order, ref first: of two sides of one model that both
    exported its graph, the ref side's is kept, the target's then has
    nothing new (model_graphs keeps a signature once).

    Args:
        turn: The turn whose graph is kept.
    """
    for role in capture_index.SIDES:
      with self._lock:
        model = self._side_models.get(role)
      path = (
          turn.index.turn_directory(role, turn.n)
          / capture_index.EXECUTION_GRAPH
      )
      if (
          model is not None
          and model.runtime == model_store.PYTORCH
          and path.is_file()
      ):
        self._context.exported(
            model.artifact,
            path,
            {
                'session': self.owner_id,
                'chat': turn.chat,
                'turn': turn.n,
                'side': role,
            },
        )

  def _fetch_capture(
      self,
      role: str,
      index: capture_index.CaptureIndex,
      n: int,
      report: Callable[[str, int], None],
  ) -> None:
    """Fetches a side's capture files for a turn.

    Args:
        role: The side role ('baseline' or 'target').
        index: The capture index.
        n: The turn number.
        report: Callback invoked with bytes retrieved so far.
    """
    side = self.sides[role]
    try:
      side.take_capture(
          n,
          index.turn_directory(role, n),
          self._context.captures.workspace,
          lambda count: report(role, count),
      )
    except Exception as error:
      raise _SideError(role, error) from error

  def _end_turn(  # pylint: disable=redefined-outer-name
      self,
      turn: Turn,
      status: str,
      message: str,
      does_not_fit: dict[str, dict[str, int]] | None = None,
      failed_sides: list[str] | None = None,
  ) -> None:
    """Ends a turn's job; the Model Server stays as it is.

    Args:
        turn: The turn.
        status: The job's end: 'completed', 'failed' or 'cancelled'.
        message: Why it did not complete, or ''.
        does_not_fit: Of a turn both sides refused for want of room, each side's
          counts (jobs.Jobs.finish()).
        failed_sides: Of a failed turn, the sides that failed, when known
          (jobs.Jobs.finish()).
    """
    try:
      with self._lock:
        if self.turn is turn:
          self.turn = None
          self.job_id = self.operation = None
        if not self.closing:
          for side in self.sides.values():
            side.phase = 'active'
    finally:
      self._jobs.finish(
          turn.job_id, status, message, does_not_fit, failed_sides
      )

  def _end_chat(
      self,
      turn: Turn,
      failures: dict[str, Exception],
      message: str = '',
  ) -> None:
    """Ends a chat whose sides are out of step or whose runner failed.

    A Stop ends the chat too. Its text names Debug only if the chat kept
    a turn, rejected or not (chat_over()). The Model Server stays on for
    another chat, unless a side's runner is gone, as its connection says
    now: then the Session ends (_end_session()) before the turn's job
    does. Being turned off meanwhile, the chat ends with the Model Server
    (TURNED_OFF).

    Args:
        turn: The turn.
        failures: Each failing side's Exception.
        message: An optional explanation why the chat ended.
    """
    with self._lock:
      cancelled = turn.cancel_requested
      closing = self.closing
    cause = [
        f'{SIDE_NAMES[role]}: {failure}' for role, failure in failures.items()
    ]
    if cancelled or closing:
      message = 'Generation stopped.'
    elif not message:
      message = ' '.join(cause) or 'Generation stopped.'
    # Every turn kept, rejected or not, set the chat's has_capture.
    has_capture = turn.index.next_turn() > 1
    message = ended_chat(message, has_capture)
    if cancelled or closing:
      logging.info(
          '%s: Chat %d turn %d stopped', self._label(), turn.chat, turn.n
      )
    else:
      logging.error(
          '%s: Chat %d turn %d failed: %s',
          self._label(),
          turn.chat,
          turn.n,
          message,
      )
    try:
      if not closing:
        self._end_live_chat(turn.chat, message)
        with self._lock:
          gone = self._gone()
        # The job ends once the Model Server is off: a page that sees
        # it end sees the Session ended.
        if gone:
          self._end_session(gone[0])
    finally:
      if cancelled or closing:
        self._end_turn(turn, 'cancelled', message)
      else:
        self._end_turn(turn, 'failed', message, None, list(failures))

  def _watch(self) -> None:
    """Ends the Session once a side's runner is gone while no job runs.

    It runs on a thread of its own from the moment the Model Server came
    on until it is being turned off. A runner gone during a job fails
    that job's call to it, which ends the chat and the Session
    (_end_chat, _fail_chat); one gone meanwhile only its connection
    tells (SideState.online).
    """
    while not self.turning_off.wait(_WATCH_SECONDS):
      with self._lock:
        if self.phase != 'active':
          return
        if self.job_id is not None:
          continue
        gone = self._gone()
      if gone:
        self._end_session(gone[0])
        return

  def _gone(self) -> list[str]:
    """The sides whose runner is not connected; with the lock held."""
    return [
        side.role
        for side in self.sides.values()
        if side.client is None or not side.client.online()
    ]

  def _end_session(self, side: str) -> None:
    """Turns the Model Server off for good: a side's runner is gone.

    The Session keeps why (runner_gone()) as its notice, and its live
    chat, if any, ends saying so. Its other runner is stopped, as Turn
    off does. Nothing happens if it is being turned off already.

    Args:
        side: The side whose runner is not connected.
    """
    with self._lock:
      if not self.begin_close():
        return
    notice_text = runner_gone(side)
    logging.error('%s: %s', self._label(), notice_text)
    try:
      self._sessions.set_flags(self.owner_id, notice=notice_text)
      self.end_live_chat(notice_text)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          'The record of %s did not keep why it ended', self.owner_id
      )
    self.close()

  def end_live_chat(self, reason: str) -> None:
    """Ends the live chat, if any, saying why (ended_chat()).

    Called without the lock held, as the Model Server is turned off.

    Args:
        reason: Why the chat ended.
    """
    with self._lock:
      n, index = self.chat, self.index
    if n is not None:
      has_capture = index.next_turn() > 1
      self._end_live_chat(n, ended_chat(reason, has_capture))

  def _end_live_chat(  # pylint: disable=redefined-outer-name
      self, n: int, notice: str = '', over_cause: dict[str, Any] | None = None
  ) -> None:
    """Marks chat `n` over in its record; it is no longer live.

    Args:
        n: The chat.
        notice: Why it is over, as its record keeps it.
        over_cause: Of a chat a full context ended, the fact notice() derives
          its text from; `notice` is then ''.
    """
    with self._lock:
      if self.chat == n:
        self.chat = None
        self.index = None
    flags = {'over': True, 'notice': notice}
    if over_cause is not None:
      flags['over_cause'] = over_cause
    self._sessions.set_chat_flags(self.owner_id, n, **flags)

  def begin_cancel(self, job_id: str) -> Callable[[], None] | None:
    """Marks the turn of a job stopped, if its captures are not both held.

    Called with the lock held.

    Args:
        job_id: The identifier of the job being cancelled.

    Returns:
        What stops the turn's sides, to call once the lock is released;
        None if the job is not this Model Server's turn, the turn's
        captures are both held, or both sides refused it.
    """
    turn = self.turn
    if turn is None or turn.job_id != job_id or turn.held or turn.refused:
      return None
    turn.cancel_requested = True
    sides = list(self.sides.values())

    def stop() -> None:
      # Its Send, or its capture if it is coming: the turn is dropped.
      for side in sides:
        side.cancel()

    return stop

  # ---- Turning off.

  def begin_close(self) -> bool:
    """Marks the Model Server being turned off; called with the lock held.

    Returns:
        False if it already is being turned off, or ended.
    """
    if self.phase in ('ended', 'ending'):
      return False
    self.closing = self.aborted = True
    self.turning_off.set()
    self.phase = 'ending'
    for side in self.sides.values():
      side.phase = 'ending'
    if self.turn is not None and not self.turn.completed:
      self.turn.cancel_requested = True
    return True

  def close(self) -> None:
    """Stops both runners for good, once begin_close() said so.

    The devices stay held until no start of the Model Server can launch a
    runner any more: one still launching stops its runner once it returns.
    A runner that does not stop is logged, and the Model Server ends all
    the same.
    """
    # Closing the channels stops a turn in flight.
    try:
      self.stop_runners()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('The runners of %s did not stop', self.owner_id)
    with self._lock:
      if not self.starting:
        self._end_closed()

  def _end_closed(self) -> None:
    """Ends a Model Server being turned off; it frees its devices.

    Called with the lock held, once every runner of it is stopped and no
    start of it can launch another.
    """
    if self.became_active:
      self.phase = 'ended'
      for side in self.sides.values():
        side.phase = 'ended'
      logging.info('%s: the Model Server is off', self._label())
    else:
      # Never on: its start ended its Session (_fail_start).
      self._context.forget(self)

  def begin_shutdown(self) -> None:
    """Marks the Model Server turned off as the server exits.

    Called with the lock held; stop_runners() follows.
    """
    self.closing = self.aborted = True
    self.turning_off.set()
    if self.turn is not None:
      self.turn.cancel_requested = True

  def stop_runners(self) -> None:
    """Closes both channels and stops both runners; idempotent.

    A side that does not stop is logged; the other is stopped all the
    same.
    """
    with self._lock:
      sides = list(self.sides.values())

    with concurrent.futures.ThreadPoolExecutor(len(sides)) as pool:
      futures = {pool.submit(side.stop): side.role for side in sides}
    for future, role in futures.items():
      if future.exception() is not None:
        logging.error(
            'The %s runner did not stop: %r',
            SIDE_NAMES[role],
            future.exception(),
        )

  # ---- Shared steps.

  def _aborted(self) -> bool:
    with self._lock:
      return self.aborted

  def _label(self) -> str:
    """How the log names the Session, by its name now."""
    try:
      return session_store.label(self._sessions.get(self.owner_id))
    except errors.NotFound:
      return session_store.label({'id': self.owner_id})

  def _set_phase(self, role: str, phase: str) -> None:
    with self._lock:
      if not self.closing:
        self.sides[role].phase = phase

  def _progress(
      self,
      job_id: str,
      role: str,
      phase: str,
      message: str,
  ) -> None:
    self._set_phase(role, phase)
    self._jobs.emit(job_id, 'progress', runId=role, message=message)
