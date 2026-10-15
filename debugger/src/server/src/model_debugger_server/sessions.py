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

"""Session and Chat records.

A Session is what the user configures: a name, a model to show, and two runs,
`ref` and `target`, each a runtime, a model, a device and its options. A Chat
is one conversation of a Session's execution, with its own generation
settings. Both are the same kind of record, one JSON file each in
`records/`; a Chat names its Session in `parent_session_id`, and a Session is
itself its first Chat.

Stored fields: id, name, model, created_at, deleted, runs, generation,
parent_session_id (Chats only), has_capture, execution_ended, chat_over,
notice.
"""

import copy
import datetime
import threading
import types
from typing import Any
import uuid

from model_debugger_server import errors, models, options, workspace

_NAME_LIMIT = 160
_MODEL_LIMIT = 200
_VALUE_LIMIT = 2048
_RUNTIME = 'LiteRT-LM'
_SIDES = ('ref', 'target')
_SIDE_NAMES = types.MappingProxyType({'ref': 'Reference', 'target': 'Target'})
# The run fields every run has; the rest are options.RUNTIME_OPTION_KEYS.
_RUN_FIELDS = (
    'runtime',
    'artifact',
    'source',
    'device',
    'backend',
    'contextLength',
)
# Keys the UI sends that this server has no use for: what the old runner
# took, kept on the UI's side. A duplicated Session carries them, so they are
# dropped, not refused.
_DROPPED_RUN_KEYS = frozenset((
    'runnerBuild',
    'repository',
    'revision',
    'sourceUrl',
    'precision',
))
_DROPPED_CONFIG_KEYS = frozenset(('tap_profile', 'tap_points'))
_CONFIG_KEYS = frozenset(('id', 'name', 'model', 'runs'))


def _now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(
      timespec='microseconds'
  )


def _validate_name(value: Any) -> str:
  if (
      not isinstance(value, str)
      or not value.strip()
      or len(value.strip()) > _NAME_LIMIT
  ):
    raise ValueError(f'A name must have 1 to {_NAME_LIMIT} characters.')
  if any(ord(char) < 32 for char in value):
    raise ValueError('A name cannot contain control characters.')
  return value.strip()


def _text(value: Any, what: str) -> str:
  if (
      not isinstance(value, str)
      or len(value) > _VALUE_LIMIT
      or any(ord(char) < 32 for char in value)
  ):
    raise ValueError(f'{what} is not valid text.')
  return value.strip()


class SessionStore:
  """Every Session and Chat record of a workspace.

  Records are read once at start and kept in memory; each change is written
  to the record's file before it is answered. Every method returns copies.
  """

  def __init__(
      self, workspace_: workspace.Workspace, models_: models.ModelStore
  ) -> None:
    """Loads the workspace's records.

    Args:
      workspace_: Where the records are kept.
      models_: The models a run may name.
    """
    self._directory = workspace_.records
    self._models = models_
    self._lock = threading.RLock()
    self._records: dict[str, dict] = {}
    for path in self._directory.glob('*.json'):
      record = workspace.read_json(path)
      self._records[record['id']] = record

  def list(self) -> list[dict]:
    """Returns every record not deleted, Sessions and Chats, oldest first."""
    with self._lock:
      records = [r for r in self._records.values() if not r['deleted']]
      records.sort(key=lambda r: (r['created_at'], r['id']))
      return copy.deepcopy(records)

  def get(self, record_id: str) -> dict[str, Any]:
    """Returns a record.

    Raises:
      NotFound: No such record, or it is deleted.
    """
    with self._lock:
      return copy.deepcopy(self._live(record_id))

  def create(self, payload: dict) -> dict:
    """Creates a Session from what the UI's editor posts.

    Args:
      payload: `{name, model, runs}`; also what a duplicate posts, which may
        carry keys of the old runner (dropped).

    Returns:
      The new record.

    Raises:
      ValueError: The configuration is refused.
    """
    config = self._config(payload)
    record = {
        'id': str(uuid.uuid4()),
        **config,
        'created_at': _now(),
        'deleted': False,
        'generation': {'ref': {}, 'target': {}},
        'has_capture': False,
        'execution_ended': False,
        'chat_over': False,
        'notice': '',
    }
    with self._lock:
      self._write(record)
      return copy.deepcopy(record)

  def update(self, payload: dict) -> dict:
    """Changes a Session's configuration.

    Args:
      payload: As for `create`, plus `id`.

    Returns:
      The record.

    Raises:
      NotFound: No such record.
      Conflict: The Session has a capture or has run; duplicate it instead.
      ValueError: The configuration is refused.
    """
    config = self._config(payload)
    with self._lock:
      record = self._live(payload.get('id'))
      if record['has_capture'] or record['execution_ended']:
        raise errors.Conflict(
            'This session has run; duplicate it to change its configuration.'
        )
      self._write({**record, **config})
      return self.get(record['id'])

  def rename(self, record_id: str, name: str) -> dict[str, Any]:
    """Renames a record; allowed at any time."""
    name = _validate_name(name)
    with self._lock:
      self._write({**self._live(record_id), 'name': name})
      return self.get(record_id)

  def delete(self, record_id: str) -> dict[str, Any]:
    """Marks a record deleted; its files stay, and `restore` undoes it."""
    with self._lock:
      record = {**self._live(record_id), 'deleted': True}
      self._write(record)
      return copy.deepcopy(record)

  def restore(self, record_id: str) -> dict[str, Any]:
    """Undoes `delete`.

    Raises:
      NotFound: No such record, deleted or not.
    """
    with self._lock:
      if record_id not in self._records:
        raise errors.NotFound('This session does not exist.')
      self._write({**self._records[record_id], 'deleted': False})
      return self.get(record_id)

  def create_chat(
      self, parent_id: str, name: str | None, generation: dict | None
  ) -> dict:
    """Creates a Chat of the Session `parent_id` belongs to.

    Args:
      parent_id: The Session, or one of its Chats; the new Chat copies its runs
        and, when `generation` is None, its generation settings.
      name: The Chat's name; None is 'New chat'.
      generation: The Chat's generation settings, as the UI sends them.

    Returns:
      The new record.

    Raises:
      NotFound: No such record.
      ValueError: The name or the generation settings are refused.
    """
    name = _validate_name('New chat' if name is None else name)
    with self._lock:
      source = self._live(parent_id)
      owner = self._records[source.get('parent_session_id', source['id'])]
      record = {
          'id': str(uuid.uuid4()),
          'name': name,
          'model': source['model'],
          'runs': copy.deepcopy(source['runs']),
          'created_at': _now(),
          'deleted': False,
          'generation': (
              copy.deepcopy(source['generation'])
              if generation is None
              else options.validate_generation(generation)
          ),
          'parent_session_id': owner['id'],
          'has_capture': False,
          'execution_ended': owner['execution_ended'],
          'chat_over': False,
          'notice': '',
      }
      self._write(record)
      return copy.deepcopy(record)

  def set_generation(
      self, record_id: str, generation: dict[str, Any]
  ) -> dict[str, Any]:
    """Replaces a record's generation settings.

    Raises:
      NotFound: No such record.
      Conflict: The record has a capture: its settings are what made it.
      ValueError: The settings are refused.
    """
    generation = options.validate_generation(generation)
    with self._lock:
      record = self._live(record_id)
      if record['has_capture']:
        raise errors.Conflict(
            "A captured chat's settings cannot change; start a new chat."
        )
      self._write({**record, 'generation': generation})
      return self.get(record_id)

  def set_flags(self, record_id: str, **flags: Any) -> None:
    """Sets the execution's flags of a record, deleted or not, in one step.

    The flags are `has_capture`, `execution_ended`, `chat_over` and
    `notice`. Read and written under the store's lock, so a rename or a
    delete meanwhile is kept.

    Raises:
      NotFound: No such record.
    """
    with self._lock:
      if record_id not in self._records:
        raise errors.NotFound('This session does not exist.')
      self._write({**self._records[record_id], **flags})

  def _live(self, record_id: Any) -> dict[str, Any]:
    record = (
        self._records.get(record_id) if isinstance(record_id, str) else None
    )
    if record is None or record['deleted']:
      raise errors.NotFound('This session does not exist.')
    return record

  def _write(self, record: dict) -> None:
    workspace.write_json(self._directory / f'{record["id"]}.json', record)
    self._records[record['id']] = record

  def _config(self, payload: Any) -> dict:
    """Checks what the editor posts and returns `{name, model, runs}`."""
    if not isinstance(payload, dict):
      raise ValueError('The session configuration is not valid.')
    unknown = set(payload) - _CONFIG_KEYS - _DROPPED_CONFIG_KEYS
    if unknown:
      raise ValueError(
          'Unknown session settings: ' + ', '.join(sorted(unknown)) + '.'
      )
    name = _validate_name(payload.get('name'))
    model = _text(payload.get('model', ''), 'The model')
    if len(model) > _MODEL_LIMIT:
      raise ValueError(
          f'The model must have at most {_MODEL_LIMIT} characters.'
      )
    runs = payload.get('runs')
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or not all(isinstance(run, dict) for run in runs)
        or {run.get('id') for run in runs} != set(_SIDES)
    ):
      raise ValueError('Reference and Target runs are required.')
    by_side = {run['id']: run for run in runs}
    return {
        'name': name,
        'model': model,
        'runs': [self._run(side, by_side[side]) for side in _SIDES],
    }

  def _run(self, side: str, run: dict) -> dict:
    """Checks one run and returns what is kept of it."""
    where = _SIDE_NAMES[side]
    unknown = (
        set(run)
        - {'id'}
        - set(_RUN_FIELDS)
        - set(options.RUNTIME_OPTION_KEYS)
        - _DROPPED_RUN_KEYS
    )
    if unknown:
      raise ValueError(
          f'Unknown {where} settings: ' + ', '.join(sorted(unknown)) + '.'
      )
    clean = {'id': side}
    for key in _RUN_FIELDS:
      clean[key] = _text(run.get(key, ''), f'{where} {key}')
    if clean['runtime'] != _RUNTIME:
      raise ValueError(f'{where}: the runtime must be {_RUNTIME}.')
    try:
      model = self._models.get(clean['artifact'])
    except errors.NotFound:
      raise ValueError(f'{where}: select a model.') from None
    if not model.path.is_file():
      raise ValueError(f'{where}: the model {model.name} is no longer there.')
    if not clean['device']:
      raise ValueError(f'{where}: select a device.')
    if clean['backend'] not in ('CPU', 'GPU'):
      raise ValueError(f'{where}: the backend must be CPU or GPU.')
    try:
      options.context_length(clean)
    except ValueError as error:
      raise ValueError(f'{where}: {error}') from None
    for key in options.RUNTIME_OPTION_KEYS:
      value = _text(run.get(key, ''), f'{where} {key}')
      if value:
        try:
          options.check_run_option(key, value)
        except ValueError as error:
          raise ValueError(f'{where}: {error}') from None
        clean[key] = value
    return clean
