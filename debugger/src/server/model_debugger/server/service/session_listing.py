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

"""The listing of Sessions: GET /api/sessions, and each record.

A record as the UI lists it carries its Session's Model Server (the UI's
SessionExecution), its status, its chats, and what the user may do with it
and each chat now. Each action is allowed or refused by the very rule its
request is refused by (session_store, model_servers, model_server), asked
here without making the request, so its reason is the request's refusal
word for word. A chat's `debug` is not a request but whether it is read in
Debug (see _chat_view). Nothing
here lists the devices: whether a runner is online its connection says.
"""

from collections.abc import Callable
from typing import Any

from absl import logging
from model_debugger.server.capture import capture_index
from model_debugger.server.runner import launcher
from model_debugger.server.session import capture_store
from model_debugger.server.session import hf_models
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.session import model_servers as model_servers_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.session import prepared_models as prepared_lib
from model_debugger.server.session import session_archive
from model_debugger.server.session import session_store
from model_debugger.server.util import errors

# A record's status text, by its Model Server's phase while it runs here;
# once it has ended, ENDED or ENDED_NO_TURN.
_PHASE_TEXT = {
    'starting': 'Starting',
    'active': 'Active',
    'ending': 'Ending',
}
# A Session that has ended: with a turn to inspect in some chat, or with
# none, which is not opened.
ENDED = 'Ended'
ENDED_NO_TURN = 'Ended · no turn'
# What a Model Server is, to the user, while its button turns it off; an
# ended Session's reads ENDED or ENDED_NO_TURN.
_MODEL_SERVER_TEXT = {
    'on': 'On',
    'starting': 'Starting',
}
# What every side is doing while its Model Server is in one of these phases.
_PHASE_TEXT_BY_SERVER = {
    'ending': 'Ending',
    'ended': 'Off',
}
# What each side is doing otherwise, by the side's phase.
_SIDE_PHASE_TEXT = {
    'starting': 'Starting',
    'connecting': 'Connecting',
    'uploading': 'Uploading model',
    'loading': 'Loading model',
    'initializing': 'Starting chat',
    'active': 'Ready',
    'generating': 'Generating',
    'transferring': 'Transferring capture',
    'ending': 'Ending',
    'ended': 'Off',
    'failed': 'Failure',
}
# The refusals a rule raises; anything else is a fault of this server.
_REFUSALS = (errors.Conflict, errors.NotFound, ValueError, KeyError, TypeError)
# Why a chat is not read in Debug while it has no turn.
_NO_TURN = 'This chat has no turn to inspect.'
# Why a Session that ended with no turn is not opened.
_ENDED_NO_TURN_REASON = (
    'This Session ended with no turn; to try again, create a new Session.'
)
# How often a capture that cannot be read is logged while the UI polls.
_LOG_SECONDS = 60


def _delete_note(models: list[model_store.Model]) -> str:
  """What deleting a Session deletes beside it: the models only it names.

  Args:
    models: Those models (session_store.SessionStore.sole_models()).

  Returns:
    A message describing deleted models and freed space, or empty string.
  """
  if not models:
    return ''
  names = ' and '.join(model.name for model in models)
  text = f'It also deletes {"the models" if len(models) > 1 else "the model"}'
  text += f' {names} from the Server'
  freed = sum(model.size for model in models if model.present())
  if freed:
    text += f', freeing {hf_models.size_text(freed)}'
  return text + '.'


def _action(check: Callable[[], object]) -> dict[str, Any]:
  """An action as the UI shows it: whether `check` passes, and if not why."""
  try:
    check()
  except _REFUSALS as error:
    return {'allowed': False, 'reason': errors.message(error)}
  return {'allowed': True, 'reason': ''}


class SessionListing:
  """Every record as the UI lists it."""

  def __init__(
      self,
      sessions: session_store.SessionStore,
      model_servers: model_servers_lib.ModelServers,
      jobs: jobs_lib.Jobs,
      captures: capture_store.CaptureStore,
      prepared: prepared_lib.PreparedModels,
      exports: session_archive.Exports | None = None,
  ) -> None:
    """Lists the records of `sessions` with their Model Servers.

    Args:
        sessions: The records.
        model_servers: Their Model Servers.
        jobs: The jobs, for each side's latest progress.
        captures: Each chat's capture, which reading it in Debug needs.
        prepared: The copies that observe tensors, and their preparation.
        exports: The Sessions being exported; none if None.
    """
    self._prepared = prepared
    self._exports = (
        exports if exports is not None else session_archive.Exports()
    )
    self._sessions = sessions
    self._model_servers = model_servers
    self._jobs = jobs
    self._captures = captures

  def listing(self) -> dict[str, Any]:
    """The response of GET /api/sessions."""
    return {
        'sessions': [self.summary(record) for record in self._sessions.list()],
        'configuration_template': {
            'id': '',
            'name': '',
            'model': '',
            'created_at': None,
            'tap_anchors': dict.fromkeys(
                capture_index.SIDES, session_store.DEFAULT_TAP_ANCHORS
            ),
            'tap_ops': [],
            'tap_profile': '',
            'tap_points': {},
            'runs': [
                {
                    'id': side,
                    'runtime': 'LiteRT-LM',
                    'artifact': '',
                    'source': 'upload',
                    'device': launcher.LOCAL_DEVICE_ID,
                    'backend': 'CPU',
                    'contextLength': '4096',
                }
                for side in capture_index.SIDES
            ],
        },
        'unavailable_reason': '',
    }

  def summary(self, record: dict[str, Any]) -> dict[str, Any]:
    """A record as the listing shows it: its execution, status, actions.

    Fields added to the record: `execution` (once its Session's Model
    Server ran here), `status_text`, `model_server`, `actions`,
    `live_chat` (the number of its live chat, or None), `has_capture`
    (whether any chat kept a turn), `tap_preparation` (its preparation
    as prepared_models.view() says), and `job_id` (the last job its Model
    Server ran). Its `chats` are each as _chat_view() shows
    it. The record's own `started` and `execution_ended` flags are left
    out: the Model Server's switch says them (ENDED, ENDED_NO_TURN);
    `notice` says why the Session ended, if it did for a reason of its
    own (its start failed, a runner closed). `delete_note` says which
    models deleting it deletes too, those no other Session names, and
    the room that frees; '' if none. `exporting` says whether its
    export is being sent (session_archive.Exports). `tap_points` are
    the points its editor chose (session_store.chosen_points()),
    `tap_anchors` whether each LiteRT-LM side captures its anchors too, by
    side (session_store.anchor_sides()); the anchors found are left out
    (`tap_anchor_points`): the editor shows each model's as GET
    /api/artifacts/{id}/anchors describes them; so are the points of the
    operators chosen inside the layers (`tap_op_points`), which the editor
    shows by `tap_ops`, as GET /api/models/ops lists them.

    Args:
      record: The raw session record dictionary.

    Returns:
      A summarized dictionary representing the session and its current state.
    """
    record_id = record['id']
    state = self._model_servers.state(record_id)
    summary = dict(record)
    for key in (
        'started',
        'execution_ended',
        'tap_anchor_points',
        'tap_op_points',
    ):
      summary.pop(key, None)
    if state is not None:
      summary['execution'] = self._execution_view(state)
    summary.update(
        status_text=_status_text(record, state),
        exporting=self._exports.exporting(record_id),
        model_server=self._model_server_view(record, state),
        actions=self._actions(record, state),
        has_capture=session_store.has_turn(record),
        live_chat=_live(state),
        chats=[
            self._chat_view(record, state, chat) for chat in record['chats']
        ],
        tap_anchors=session_store.anchor_sides(record),
        tap_profile=record.get('tap_profile', ''),
        tap_points=session_store.chosen_points(record),
        tap_ops=session_store.tap_ops(record),
        notice=record.get('notice', ''),
    )
    summary['tap_preparation'] = self._prepared.view(record)
    summary['delete_note'] = _delete_note(self._sessions.sole_models(record))
    job_id = self._model_servers.last_job(record_id)
    if job_id is not None:
      summary['job_id'] = job_id
    return summary

  def _actions(
      self,
      record: dict[str, Any],
      state: model_server_lib.ModelServerState | None,
  ) -> dict[str, Any]:
    """What the user may do with a record now, each with its refusal.

    Each is the rule of its request: enter (GET /api/session), delete,
    new_chat
    (POST /api/sessions/{id}/chats) and export
    (GET /api/sessions/{id}/export). What the request carries
    (a configuration, a chat's settings) is checked when it comes. A
    turn's rule is each chat's (_chat_view()).

    While its export is being sent, a Session is neither entered,
    deleted nor exported again (session_archive.Exports).

    Args:
      record: The session record dictionary.
      state: Current model server state, or None.

    Returns:
      A dictionary of action names mapped to action dictionaries.
    """

    def delete() -> None:
      self._exports.check_not_exporting(record['id'])
      model_servers_lib.check_idle(state)

    def export() -> None:
      session_archive.check_exportable(state, record)
      self._exports.check_not_exporting(record['id'])

    return {
        'enter': self._enter(record, state),
        'delete': _action(delete),
        'new_chat': _action(
            lambda: model_server_lib.check_new_chat(state, record)
        ),
        'export': _action(export),
    }

  def _chat_view(
      self,
      record: dict[str, Any],
      state: model_server_lib.ModelServerState | None,
      chat: dict[str, Any],
  ) -> dict[str, Any]:
    """A chat as the listing shows it.

    `{n, title, summary, generation, state, notice, has_capture,
    created_at, actions}`: `summary` its settings in a few words
    (options.summary()), once if both sides have the same, else each
    side's; `state` 'live' while its Model Server's runners hold it,
    else 'ended', with why as `notice` (model_server.notice(); ENDED
    when it kept none, as of a Session that ended as this server
    stopped). Its actions: `send`, a turn's rule
    (model_server.check_chat_turn()), and `debug`, whether it is read in
    Debug: once it kept a turn, rejected or not, and its capture reads.

    Args:
      record: The session record dictionary.
      state: Current model server state, or None.
      chat: The chat dictionary.

    Returns:
      A dictionary representing the view of the chat.
    """
    live = _live(state) == chat['n']
    index = None
    debug = {'allowed': False, 'reason': _NO_TURN}
    if chat['has_capture']:
      index, debug = self._capture(record, chat)
    notice = ''
    if not live:
      notice = model_server_lib.notice(chat, index) or model_server_lib.ENDED
    return {
        'n': chat['n'],
        'title': f'Chat {chat["n"]}',
        'summary': _summary(chat['generation']),
        'generation': chat['generation'],
        'state': 'live' if live else 'ended',
        'notice': notice,
        'has_capture': chat['has_capture'],
        'created_at': chat.get('created_at'),
        'actions': {
            'send': _action(
                lambda: model_server_lib.check_chat_turn(state, record, chat)
            ),
            'debug': debug,
        },
    }

  def _capture(
      self, record: dict[str, Any], chat: dict[str, Any]
  ) -> tuple[object, dict[str, Any]]:
    """A chat's capture, read as GET does, and whether it reads.

    Args:
      record: The session record dictionary.
      chat: The chat dictionary.

    Returns:
      The index, or None, and the `debug` action it makes: one that
      cannot be read for another reason than a refusal refuses this
      chat alone; the others list as ever.
    """
    try:
      return self._captures.get(record['id'], chat['n']), {
          'allowed': True,
          'reason': '',
      }
    except _REFUSALS as error:
      return None, {'allowed': False, 'reason': errors.message(error)}
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.log_every_n_seconds(
          logging.WARNING,
          'The capture of Chat %d of %s cannot be read: %r',
          _LOG_SECONDS,
          chat['n'],
          record['id'],
          error,
      )
      return None, {
          'allowed': False,
          'reason': f"This chat's capture cannot be read: {error}",
      }

  def _enter(
      self,
      record: dict[str, Any],
      state: model_server_lib.ModelServerState | None,
  ) -> dict[str, Any]:
    """Whether a record may be entered.

    While its Model Server is on, with or without a chat, and once any
    chat kept a turn. A Session that ended with no turn has nothing to
    show and is not entered, nor is one whose export is being sent.

    Args:
      record: The session record dictionary.
      state: Current model server state, or None.

    Returns:
      An action dictionary indicating whether entering is allowed.
    """

    def enter() -> None:
      self._exports.check_not_exporting(record['id'])
      if _live_server(state) or session_store.has_turn(record):
        return
      if _ended(record, state):
        raise errors.Conflict(_ENDED_NO_TURN_REASON)
      model_server_lib.check_on(state)

    return _action(enter)

  def _model_server_view(
      self,
      record: dict[str, Any],
      state: model_server_lib.ModelServerState | None,
  ) -> dict[str, Any]:
    """The Session's Model Server switch: its state, text, error, button.

    The button turns it off while it is on and cancels its start while it
    starts (both a close, refused while one is under way); a Session
    starts once, as it is created, so one that has ended has none
    (None). An ended Session's switch reads ENDED, or ENDED_NO_TURN with
    why it ended (its `notice`, such as why its start failed) as its
    error.

    Args:
      record: The session record dictionary.
      state: Current model server state, or None.

    Returns:
      A dictionary describing the model server switch state.
    """
    phase = state.phase if state is not None else None
    if phase in ('active', 'ending'):
      switch = 'on'
    elif phase == 'starting':
      switch = 'starting'
    else:
      switch = 'ended'
    button = None
    error = ''
    if switch == 'ended':
      text = _status_text(record, state)
      if not session_store.has_turn(record):
        error = record.get('notice', '')
    else:
      text = _MODEL_SERVER_TEXT[switch]
      button = {
          'label': 'Turn off' if switch == 'on' else 'Cancel',
          **_action(lambda: model_servers_lib.check_close(state)),
      }
    return {
        'state': switch,
        'text': text,
        'error': error,
        'button': button,
    }

  def _execution_view(
      self, state: model_server_lib.ModelServerState
  ) -> dict[str, Any]:
    """The UI's SessionExecution of a Model Server.

    Each runner shows its phase and its latest progress line in the
    running job, if any, such as 'Uploading model · 45%'.

    Args:
      state: Current model server state.

    Returns:
      A dictionary describing the session execution view.
    """
    progress = self._jobs.progress(state.job_id) if state.job_id else {}
    runners = []
    for side in state.sides:
      if state.phase in ('ended', 'ending'):
        phase_text = _PHASE_TEXT_BY_SERVER[state.phase]
      else:
        phase_text = _SIDE_PHASE_TEXT.get(side.phase, 'Unknown')
      runner = {
          'role': side.role,
          'deviceId': side.device.id,
          'deviceName': side.device.name,
          'phase': side.phase,
          'phaseText': phase_text,
          'progress': progress.get(side.role, ''),
          'connected': side.online,
      }
      if side.build_id is not None:
        runner['buildId'] = side.build_id
      runners.append(runner)
    view = {'phase': state.phase, 'runners': runners}
    if state.job_id:
      view['activeJobId'] = state.job_id
      view['activeOperation'] = state.operation
    return view


def _live(state: model_server_lib.ModelServerState | None) -> int | None:
  """The number of a Model Server's live chat, if it is on and holds one."""
  return state.chat if _live_server(state) else None


def _live_server(state: model_server_lib.ModelServerState | None) -> bool:
  """Whether a Model Server is on."""
  return state is not None and state.phase == 'active'


def _summary(generation: dict[str, Any]) -> str:
  """A chat's settings in a few words: once, or each side's if they differ."""
  said = {
      side: options.summary(generation.get(side) or {})
      for side in capture_index.SIDES
  }
  if len(set(said.values())) == 1:
    return said['ref']
  return ' / '.join(
      f'{capture_index.SIDE_NAMES[side]}: {text}' for side, text in said.items()
  )


def _ended(
    record: dict[str, Any], state: model_server_lib.ModelServerState | None
) -> bool:
  """Whether a Session has ended: it was turned on once, and is not on now."""
  if state is not None and state.phase in ('starting', 'active', 'ending'):
    return False
  return model_server_lib.ended(record)


def _status_text(
    record: dict[str, Any], state: model_server_lib.ModelServerState | None
) -> str:
  """A record's status in words: its phase while it runs, or its end.

  A Session starts as it is created (model_servers.ModelServers.create()):
  one that is not running here has ended.

  Args:
    record: The session record dictionary.
    state: Current model server state, or None.

  Returns:
    Status text describing the current phase or termination status.
  """
  if state is not None and state.phase in _PHASE_TEXT:
    return _PHASE_TEXT[state.phase]
  return ENDED if session_store.has_turn(record) else ENDED_NO_TURN
