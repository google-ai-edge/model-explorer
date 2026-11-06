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

"""Session records.

A Session is what the user configures: a name and two runs, `ref` and
`target`, each a runtime, a model, a device and its options; its `model` is
what the listing shows of the two models, which the server names them by
(models_label()) as it saves the record. Its
chats, started one after another while its Model Server is on, are in its
record too, `chats`, oldest first: one JSON file in `records/` holds the
Session and its chats, and its id names their captures.

Deleting a Session is for good: its record goes, then its captures
(workspace.Workspace.remove_session_files()), and the models no other
Session names (sole_models(), model_store.ModelStore.collect()). A record
an older server only marked `deleted` is deleted for good as the store
loads, and so are the captures of a Session no record names. A record an
older server wrote, of a Session of one chat, is read as one of `chats`
(upgrade()).

Stored fields: id, name, model, created_at, runs,
tap_anchors, tap_anchor_points, tap_ops, tap_op_points, tap_profile,
tap_points,
started (once its Model Server was turned on: a Session starts once,
model_server.ended()), execution_ended (once it came on), notice (why the
Session ended, its start failed say), chats, imported (of a Session imported
from another computer, adopt(): `{session_id, exported_at, imported_at}`,
the id it had there and when it was exported and imported; such a Session
has ended), turn_sent_at (when its last turn was sent, turn_sent(); absent
before its first).

A chat holds n (its number, from 1: 'Chat <n>'), created_at, generation (the
settings each side runs it with, options.validate_generation()),
has_capture (it kept a turn, rejected or not), over (it takes no more
turns), notice (why it is over) and, once a full context ended it,
over_cause (its notice is derived then, model_server.notice).
A run's `runnerBuild` is the runner build it chose, on a device that
chooses one (runner_builds): a LiteRT-LM commit, or the
development build. A run whose `source` is 'huggingface' holds where its
model came from (`huggingface`: repo, file, revision, commit), which the
server takes from the model store, whatever the editor posts.

A PyTorch run observes by its own run options instead: `observe` 'anchors'
captures every decoder layer's four anchors each turn, within
`observeMemoryGiB` (its runner's default when ''); it chooses no points.
Every run created now has it 'anchors', whatever is posted; a
record written before, with '', keeps it.
Beside such a run, the points chosen of the LiteRT-LM model are each
paired with the PyTorch module of the same semantic operator where that is
certain, which the PyTorch run then observes too (prepared_models).

A Session may observe tensors of its LiteRT-LM models (tap). Each
LiteRT-LM side captures the anchors, always, whatever the editor posts:
`tap_anchors`, by side, true for each such side, which observes, of its
model, every decoder layer's anchors that its graph binds (anchor_points),
which the server finds as the Session starts, once the graph is converted,
and keeps in `tap_anchor_points`, by artifact: those it ran with. A record
written before the sides chose apart holds one `tap_anchors` for both,
true or false (anchor_sides()). Beside them, or instead, a side may observe
points chosen (the editor's advanced picker): `tap_profile` is TAP_PROFILE
then, and `tap_points` names the points chosen (tap_scan ids) of the models
of the sides that chose some, by artifact, each checked against the model's
scan when saved (tap_scans). Both sides of one
model observe the same points, the anchors and the chosen alike
(tap_points()): one model file's two sides capture its anchors both or
neither. Each side then loads a copy of its model that observes them
(prepared_models). A record without these fields, as every one written
before the anchors were, or with no side's `tap_anchors` true and
`tap_profile` '', observes none.

Beside the anchors, a Session may capture semantic operators inside the
decoder layers (the editor's list, graph/layer_ops): `tap_ops`, one list
for both sides, each an operator as the bindings name it (`attn/q_proj`,
`norm.ffn.in`), captured in every layer that has it on both sides. The
server derives what that is as the Session starts: of each LiteRT-LM
model, the tap points of those operators, which it keeps in
`tap_op_points`, by artifact, beside the anchors found; of a PyTorch run,
the modules (prepared_models). A record without `tap_ops` captures none.
"""

from collections.abc import Callable, Collection, Sequence
import copy
import pathlib
import re
import threading
from typing import Any
import uuid

from model_debugger.server.capture import capture_index
from model_debugger.server.graph import layer_ops
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.tap import tap_model
from model_debugger.server.util import clock
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_NAME_LIMIT = 160
_MODEL_LIMIT = 200
_VALUE_LIMIT = 2048
_RUNTIMES = (model_store.LITERT_LM, model_store.PYTORCH)
_SIDE_NAMES = {'ref': 'Reference', 'target': 'Target'}
# The run fields every run has; the rest are options.RUNTIME_OPTION_KEYS.
_RUN_FIELDS = (
    'runtime',
    'artifact',
    'source',
    'device',
    'backend',
    'contextLength',
)
# The runner build a run chose, kept when given: whether a start needs one is
# its device's to say (model_server.check_start).
_BUILD = 'runnerBuild'
_COMMIT = re.compile(r'[0-9a-f]{40}')
# Keys the UI sends that this server has no use for: what the old runner
# took, kept on the UI's side. The editor carries them over from the record it
# starts from, so they are dropped, not refused.
_DROPPED_RUN_KEYS = frozenset((
    'repository',
    'revision',
    'sourceUrl',
    'precision',
    # The server's own: where a Hugging Face model came from, which it
    # takes from the model store (HUGGING_FACE).
    'huggingface',
))
# A run's source when its model came from Hugging Face (hf_models); the run
# then holds, under this key too, where from: `{repo, file, revision,
# commit}`, as the model store has it.
HUGGING_FACE = 'huggingface'
# `model` is what an older editor chose, dropped: the server names the
# models itself (models_label()).
_CONFIG_KEYS = frozenset((
    'id',
    'name',
    'model',
    'runs',
    'tap_anchors',
    'tap_ops',
    'tap_profile',
    'tap_points',
))
# What every Session observes: every decoder layer's anchors (no option).
DEFAULT_TAP_ANCHORS = True
# The one way a Session observes tensors: the outputs the user chose.
TAP_PROFILE = 'custom-outputs-v1'
# The fields of a chat set_chat_flags() sets.
_CHAT_FLAGS = frozenset(('has_capture', 'over', 'notice', 'over_cause'))
# A chat's settings an older server kept for tool calling, which this one
# does not have: the tools the chat declared and constrained decoding of its
# tool calls. upgrade() drops them.
_DROPPED_GENERATION_KEYS = ('tools', 'constrainedDecoding')
# Why a Session an older server saved without starting it has ended
# (upgrade()): a Session now starts as it is created.
NEVER_STARTED = (
    'This Session was saved without starting; to run it, create a new Session.'
)


def label(record: dict[str, Any]) -> str:
  """How the log names a Session: its name and id."""
  return f"Session '{record.get('name', '')}' ({record.get('id', '')})"


def has_turn(record: dict[str, Any]) -> bool:
  """Whether any chat of a Session kept a turn, rejected or not."""
  return any(one['has_capture'] for one in record['chats'])


def last_active(record: dict[str, Any]) -> str:
  """When a Session was last active, as list() orders them.

  The latest of when it was created and when its last turn was sent; of a
  Session imported from another computer, when it was imported, whenever it
  was created or ran there. '' if a record an older server wrote recorded
  none.

  Args:
    record: The session record dictionary.

  Returns:
    The timestamp string when the session was last active.
  """
  imported = record.get('imported')
  if isinstance(imported, dict):
    return imported.get('imported_at') or ''
  return max(record.get('created_at') or '', record.get('turn_sent_at') or '')


def chat(record: dict[str, Any], n: int) -> dict[str, Any]:
  """A Session's chat `n`.

  Args:
    record: The session record dictionary.
    n: The chat number.

  Returns:
    The chat dictionary corresponding to `n`.

  Raises:
    NotFound: The Session has no such chat.
  """
  for found in record['chats']:
    if found['n'] == n:
      return found
  raise errors.NotFound(f'This Session has no Chat {n}.')


def _check_chat(chat: Any) -> None:  # pylint: disable=redefined-outer-name
  """Refuses a chat of a record imported that is not one this server keeps.

  Args:
    chat: The chat dictionary to check.

  Raises:
    ValueError: It lacks a field the listing and Debug read, or holds
      one of another type.
  """
  n = chat.get('n') if isinstance(chat, dict) else None
  if isinstance(n, bool) or not isinstance(n, int) or n < 1:
    raise ValueError('A chat of the Session has no number.')
  generation = chat.get('generation')
  cause = chat.get('over_cause')
  if (
      not isinstance(generation, dict)
      or set(generation) != set(capture_index.SIDES)
      or not all(isinstance(side, dict) for side in generation.values())
      or not isinstance(chat.get('has_capture'), bool)
      or not isinstance(chat.get('over'), bool)
      or not isinstance(chat.get('notice'), str)
      or not isinstance(chat.get('created_at'), str | None)
      or not (cause is None or isinstance(cause, dict))
  ):
    raise ValueError(f'Chat {n} of the Session is not valid.')


def upgrade(record: dict[str, Any], captures: pathlib.Path) -> dict[str, Any]:
  """Returns the record of an older server as this one keeps it.

  An older server's Session held one chat in its record: its settings
  (`generation`), whether it kept a turn (`has_capture`), whether it was
  over (`chat_over`, with `chat_over_cause` and `notice`), and its capture
  in `captures` itself. Such a Session that came on is one of an ended
  chat, Chat 1, its capture moved under chat_directory(1) of `captures`;
  one that never came on has no chat, and keeps its notice (why its start
  failed, say) as the Session's. A chat's settings lose those of tool
  calling (_DROPPED_GENERATION_KEYS); its capture keeps what it ran with.
  A Session an older server saved and never started has ended, as one
  whose start failed: `started`, with NEVER_STARTED as its notice.
  Of a record of this server, nothing changes. Both a workspace's records
  (SessionStore) and a Session imported (session_archive) are read through
  this.

  Args:
      record: The record, as it was written.
      captures: The directory of its captures; it may not exist. A move that a
        crash cut short is finished on the next read.

  Returns:
      The record, with `chats`; `record` itself if nothing changes.
  """
  if 'chats' in record:
    return _started(_without_dropped_generation(record))
  upgraded = dict(record)
  generation = upgraded.pop('generation', None)
  has_capture = bool(upgraded.pop('has_capture', False))
  cause = upgraded.pop('chat_over_cause', None)
  upgraded.pop('chat_over', None)
  notice = upgraded.pop('notice', '')
  upgraded['chats'] = []
  upgraded['notice'] = notice
  if has_capture or upgraded.get('execution_ended'):
    one = {
        'n': 1,
        'created_at': upgraded.get('created_at'),
        'generation': generation or {side: {} for side in capture_index.SIDES},
        'has_capture': has_capture,
        'over': True,
        'notice': notice,
    }
    if cause is not None:
      one['over_cause'] = cause
    upgraded['chats'] = [one]
    upgraded['notice'] = ''
    if captures.is_dir():
      first = captures / workspace.chat_directory(1)
      first.mkdir(exist_ok=True)
      for entry in captures.iterdir():
        if entry != first:
          entry.rename(first / entry.name)
  return _started(_without_dropped_generation(upgraded))


def _started(record: dict[str, Any]) -> dict[str, Any]:
  """Returns `record` ended if it never started (NEVER_STARTED), or itself."""
  if record.get('started') or record.get('execution_ended'):
    return record
  if record.get('chats'):
    return record
  return {**record, 'started': True, 'notice': NEVER_STARTED}


def _without_dropped_generation(record: dict[str, Any]) -> dict[str, Any]:
  """Returns `record` without _DROPPED_GENERATION_KEYS, or itself."""
  has_dropped = False
  for side in _generation_sides(record):
    for key in _DROPPED_GENERATION_KEYS:
      if key in side:
        has_dropped = True
        break
    if has_dropped:
      break
  if not has_dropped:
    return record
  upgraded = copy.deepcopy(record)
  for side in _generation_sides(upgraded):
    for key in _DROPPED_GENERATION_KEYS:
      side.pop(key, None)
  return upgraded


def _generation_sides(record: dict[str, Any]) -> list[dict[str, Any]]:
  """Each side's settings of each chat of `record`.

  A chat or settings not as this server keeps them are left out, for
  _check_chat() to refuse.

  Args:
    record: The session record dictionary.

  Returns:
    A list of generation side settings dictionaries.
  """
  chats = record['chats'] if isinstance(record['chats'], list) else []
  sides = []
  for one_chat in chats:
    if isinstance(one_chat, dict) and isinstance(
        one_chat.get('generation'), dict
    ):
      for side in one_chat['generation'].values():
        if isinstance(side, dict):
          sides.append(side)
  return sides


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


def tap_points(record: dict[str, Any]) -> dict[str, list[str]]:
  """The points a record observes, by artifact; {} if it observes none.

  The anchors found (anchor_points()), then the points of the operators
  chosen (op_points()), then the points chosen (chosen_points()), each
  once.

  Args:
    record: A Session's record, of any age.

  Returns:
    A mapping from artifact to observed point IDs.
  """
  found = {
      artifact: list(ids) for artifact, ids in anchor_points(record).items()
  }
  for artifact, ids in [
      *op_points(record).items(),
      *chosen_points(record).items(),
  ]:
    mine = found.setdefault(artifact, [])
    seen = set(mine)
    mine.extend(i for i in ids if i not in seen)
  return {artifact: ids for artifact, ids in found.items() if ids}


def models_label(ref: model_store.Model, target: model_store.Model) -> str:
  """What the listing shows of a Session's two models: their names.

  One model's name when both sides load it; else both, Reference first.

  Args:
    ref: The reference model.
    target: The target model.

  Returns:
    The display label showing the model names.
  """
  name = (
      ref.name
      if ref.artifact == target.artifact
      else f'{ref.name} vs {target.name}'
  )
  if len(name) > _MODEL_LIMIT:
    name = name[: _MODEL_LIMIT - 1] + '…'
  return name


def tap_ops(record: dict[str, Any]) -> list[str]:
  """The operators inside the layers a record captures; [] if none."""
  return list(record.get('tap_ops') or [])


def op_points(record: dict[str, Any]) -> dict[str, list[str]]:
  """The points of the operators chosen, by artifact, once found.

  Of the LiteRT-LM models the record's runs load; {} until the record is
  prepared (prepared_models), and when it chooses no operator.

  Args:
    record: A Session's record dictionary.

  Returns:
    A dictionary mapping artifact to list of operator point IDs.
  """
  if not tap_ops(record):
    return {}
  wanted = {
      run['artifact']
      for run in record.get('runs', [])
      if run.get('runtime', model_store.LITERT_LM) == model_store.LITERT_LM
  }
  return {
      artifact: list(ids)
      for artifact, ids in (record.get('tap_op_points') or {}).items()
      if artifact in wanted
  }


def _ops(value: Any) -> list[str]:
  """Checks the operators chosen inside the layers; returns them.

  Args:
    value: The operators value to check.

  Returns:
    The list of chosen operator names.

  Raises:
    ValueError: Not a list of different operators some binding maps.
  """
  if value is None:
    return []
  if not isinstance(value, list) or not all(
      isinstance(op, str) for op in value
  ):
    raise ValueError('Choose operators inside the layers by their ids.')
  if len(set(value)) != len(value):
    raise ValueError('An operator inside the layers is chosen twice.')
  unknown = sorted(set(value) - layer_ops.known())
  if unknown:
    raise ValueError(
        'Unknown operators inside the layers: ' + ', '.join(unknown) + '.'
    )
  return list(value)


def chosen_points(record: dict[str, Any]) -> dict[str, list[str]]:
  """The points a record's editor chose, by artifact; {} if none."""
  if record.get('tap_profile') != TAP_PROFILE:
    return {}
  return copy.deepcopy(record.get('tap_points') or {})


def anchor_sides(record: dict[str, Any]) -> dict[str, bool]:
  """Whether each LiteRT-LM side of a record captures its anchors.

  Args:
      record: A Session's record, of any age: one written before the sides chose
        apart holds one bool for both.

  Returns:
      Per LiteRT-LM side, in SIDES' order, whether it captures them.
  """
  value = record.get('tap_anchors')
  found = {}
  for run in record.get('runs', []):
    if run.get('runtime', model_store.LITERT_LM) != model_store.LITERT_LM:
      continue
    target_val = value.get(run['id']) if isinstance(value, dict) else value
    found[run['id']] = isinstance(target_val, bool) and target_val
  return {side: found[side] for side in capture_index.SIDES if side in found}


def anchor_artifacts(record: dict[str, Any]) -> set[str]:
  """The LiteRT-LM models whose anchors a record captures."""
  sides = anchor_sides(record)
  return {run['artifact'] for run in record['runs'] if sides.get(run['id'])}


def captures_anchors(record: dict[str, Any]) -> bool:
  """Whether a record captures the anchors of any LiteRT-LM model."""
  return any(anchor_sides(record).values())


def anchors_found(record: dict[str, Any]) -> bool:
  """Whether a record captures anchors and they are found.

  They are not until the Session starts (prepared_models).

  Args:
    record: A Session's record dictionary.

  Returns:
    True if anchors are captured and found in the record.
  """
  return captures_anchors(record) and 'tap_anchor_points' in record


def anchor_points(record: dict[str, Any]) -> dict[str, list[str]]:
  """The anchors found that a record captures, by artifact; {} if none."""
  wanted = anchor_artifacts(record)
  return {
      artifact: list(ids)
      for artifact, ids in (record.get('tap_anchor_points') or {}).items()
      if artifact in wanted
  }


def _anchors(runs: list[dict[str, Any]]) -> dict[str, bool]:
  """Which LiteRT-LM sides capture their anchors: every one, always.

  What the editor posts as `tap_anchors` is ignored: capturing the anchors
  is no option. A record written before keeps what it stored
  (anchor_sides()).

  Args:
      runs: The runs, checked.

  Returns:
      Per LiteRT-LM side, True.
  """
  return {
      run['id']: True for run in runs if run['runtime'] == model_store.LITERT_LM
  }


def _taps(
    profile: Any,
    points: Any,
    artifacts: set[str],
    models: model_store.ModelStore,
    scans: tap_scans.TapScans,
    anchored: frozenset[str] = frozenset(),
) -> tuple[str, dict[str, list[str]]]:
  """Checks what a Session observes; returns (tap_profile, tap_points).

  Each model's points are checked against its scan, as a copy that
  observes them would be (tap_model.check_points()); the model is scanned
  now if no scan is kept.

  Args:
      profile: '' or TAP_PROFILE.
      points: The points chosen, by artifact: of models the runs load, those of
        the sides that chose some, with 1 to tap_model.MAX_POINTS points each,
        as its scan lists them, each one that can be observed. Ignored when
        `profile` is ''.
      artifacts: The LiteRT-LM models the runs load, each registered.
      models: The registered models.
      scans: Their scans.
      anchored: The models whose anchors the Session captures too: one may have
        no point chosen; none chosen of any is profile ''.

  Returns:
      A tuple of (tap_profile, tap_points).
  """
  if not profile:
    return '', {}
  if profile != TAP_PROFILE:
    raise ValueError('Unknown tensor capture.')
  if not artifacts:
    raise ValueError(
        'No LiteRT-LM model to choose tensor outputs of: a PyTorch run '
        'captures its anchors by its own option.'
    )
  if not isinstance(points, dict) or not set(points) <= artifacts:
    raise ValueError(
        'Choose tensor outputs of the LiteRT-LM models the Session loads.'
    )
  clean = {}
  for artifact, ids in points.items():
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
      raise ValueError(
          f'Choose 1 to {tap_model.MAX_POINTS} different tensor '
          'outputs of each model.'
      )
    if artifact in anchored and not ids:
      continue
    model = models.get(artifact)
    try:
      tap_model.check_points(scans.scan(model), ids)
    except (ValueError, OSError) as error:
      raise ValueError(f'{model.name}: {error}') from error
    clean[artifact] = list(ids)
  if not clean:
    return '', {}
  return profile, clean


class SessionStore:
  """Every Session record of a workspace.

  Records are read once at start and kept in memory; each change is written
  to the record's file before it is answered. Every method returns copies.
  """

  def __init__(
      self,
      workspace_: workspace.Workspace,
      models: model_store.ModelStore,
      scans: tap_scans.TapScans,
  ) -> None:
    """Loads the workspace's records.

    Args:
        workspace_: Where the records are kept.
        models: The models a run may name.
        scans: The scans of the models, which the points a Session observes are
          checked against.
    """
    self._directory = workspace_.records
    self._models = models
    self._scans = scans
    self._lock = threading.RLock()
    self._records: dict[str, dict[str, Any]] = {}
    for path in self._directory.glob('*.json'):
      record = json_file.read_json(path)
      # What an older server only marked deleted is deleted for good.
      if record.pop('deleted', False):
        path.unlink()
        continue
      upgraded = upgrade(record, workspace_.captures / record['id'])
      if upgraded is not record:
        json_file.write_json(path, upgraded)
      self._records[record['id']] = upgraded
    workspace_.remove_orphan_session_files(self._records)

  @property
  def lock(self) -> threading.RLock:
    """Held while which models the records name changes.

    collect_models() holds it, so that no model is deleted between a
    record's models being checked and the record written; an import
    holds it from adopting its models until its record is written.
    """
    return self._lock

  def collect_models(
      self, among: Collection[str] | None = None
  ) -> list[model_store.Model]:
    """Deletes the models no record names (model_store's collect()).

    Args:
        among: As model_store.ModelStore.collect(): the models of a record just
          deleted, or None for every model.

    Returns:
        The models deleted.
    """
    with self._lock:
      used = set()
      for rec in self._records.values():
        for run in rec['runs']:
          used.add(run['artifact'])
      return self._models.collect(used, among)

  def _check_models_kept(self, config: dict[str, Any]) -> None:
    """Refuses a configuration whose model was deleted once checked.

    Called with the lock held, before the record is written.

    Args:
        config: The session configuration dictionary.

    Raises:
        ValueError: A run's model is no longer there.
    """
    for run in config['runs']:
      try:
        model = self._models.get(run['artifact'])
      except errors.NotFound:
        model = None
      if model is None or not model.present():
        raise ValueError(
            f'{_SIDE_NAMES[run["id"]]}: the model was deleted from '
            'the Server meanwhile; choose it again.'
        )

  def sole_models(self, record: dict[str, Any]) -> list[model_store.Model]:
    """The models a record's runs name and no other record's do.

    Deleting the record deletes them (model_store.ModelStore.collect()).

    Args:
        record: The record, kept or just deleted.

    Returns:
        The models referenced only by this record.
    """
    with self._lock:
      others = set()
      for other in self._records.values():
        if other['id'] != record['id']:
          for run in other['runs']:
            others.add(run['artifact'])
      artifacts = dict.fromkeys(run['artifact'] for run in record['runs'])
    found = []
    for artifact in artifacts:
      if artifact in others:
        continue
      try:
        found.append(self._models.get(artifact))
      except errors.NotFound:
        continue
    return found

  def list(self) -> list[dict[str, Any]]:
    """Returns every record, the last active first (last_active()).

    Records as active are in the order of their ids.
    """
    with self._lock:
      records = sorted(self._records.values(), key=lambda r: r['id'])
      records.sort(key=last_active, reverse=True)
      return copy.deepcopy(records)

  def get(self, id: str) -> dict[str, Any]:  # pylint: disable=redefined-builtin
    """Returns a record.

    Args:
        id: The session ID.

    Returns:
        The session record.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      return copy.deepcopy(self._live(id))

  def create(
      self,
      payload: dict[str, Any],
      check: Callable[[dict[str, Any]], object] | None = None,
  ) -> dict[str, Any]:
    """Creates a Session from what the UI's editor posts.

    Args:
        payload: `{name, model, runs}`; it may carry keys of the old runner
          (dropped). It has no chat until one is started (add_chat()).
        check: Called with the new record before it is written, under the
          store's lock; what it raises, nothing is written
          (model_servers.ModelServers.create(): whether it starts).

    Returns:
        The new record.

    Raises:
        ValueError: The configuration is refused.
    """
    config = self._config(payload)
    record = {
        'id': str(uuid.uuid4()),
        **config,
        'created_at': clock.now(),
        'execution_ended': False,
        'notice': '',
        'chats': [],
    }
    with self._lock:
      self._check_models_kept(config)
      if check is not None:
        check(copy.deepcopy(record))
      self._write(record)
      return copy.deepcopy(record)

  def rename(  # pylint: disable=redefined-builtin
      self, id: str, name: str
  ) -> dict[str, Any]:
    """Renames a record; allowed at any time."""
    name = _validate_name(name)
    with self._lock:
      self._write({**self._live(id), 'name': name})
      return self.get(id)

  def delete(self, id: str) -> dict[str, Any]:  # pylint: disable=redefined-builtin
    """Deletes a record for good; the Session's other files go next.

    The record goes first, so that a crash before its capture is
    removed (workspace.Workspace.remove_session_files(),
    the caller's next step) leaves no Session listed: the next server
    removes them as it loads the records. Whether its Model Server is off
    is the caller's to check (model_servers.ModelServers.delete()).

    Args:
        id: The session ID.

    Returns:
        The record as it was.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      record = self._live(id)
      (self._directory / f'{record["id"]}.json').unlink()
      del self._records[record['id']]
    return copy.deepcopy(record)

  def check_generation(  # pylint: disable=redefined-builtin
      self, id: str, generation: Any
  ) -> dict[str, Any]:
    """Checks the settings of a chat to start in a Session.

    Each side's are checked as options.validate_generation() checks
    them, against the Session's runs: no reply longer than its side's
    context.

    Args:
        id: The session ID.
        generation: The chat generation settings to validate.

    Returns:
        The settings each side runs with.

    Raises:
        NotFound: No such record.
        ValueError: The settings are refused.
    """
    with self._lock:
      runs = copy.deepcopy(self._live(id)['runs'])
    return options.validate_generation(
        generation, runs, self._model_sampling(runs)
    )

  def add_chat(  # pylint: disable=redefined-builtin
      self, id: str, generation: dict[str, Any]
  ) -> dict[str, Any]:
    """Adds a Session's next chat, which has kept no turn yet.

    Args:
        id: The Session.
        generation: Its settings, as check_generation() returns them.

    Returns:
        The chat, numbered after the Session's last.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      record = self._live(id)
      chats = record['chats']
      added = {
          'n': chats[-1]['n'] + 1 if chats else 1,
          'created_at': clock.now(),
          'generation': copy.deepcopy(generation),
          'has_capture': False,
          'over': False,
          'notice': '',
      }
      self._write({**record, 'chats': [*chats, added]})
      return copy.deepcopy(added)

  def turn_sent(self, id: str) -> None:  # pylint: disable=redefined-builtin
    """Records that a turn of a Session was sent now (`turn_sent_at`).

    Args:
        id: The session ID.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      self._write({**self._live(id), 'turn_sent_at': clock.now()})

  def set_chat_flags(  # pylint: disable=redefined-builtin
      self, id: str, n: int, **flags: Any
  ) -> None:
    """Sets the flags of a Session's chat in one step.

    The flags are `has_capture`, `over`, `notice` and `over_cause`. Read
    and written under the store's lock, as set_flags().

    Args:
        id: The session ID.
        n: The chat index.
        **flags: The chat flags to set.

    Raises:
        NotFound: No such record, or no such chat.
    """
    unknown = set(flags) - _CHAT_FLAGS
    if unknown:
      raise ValueError(f'Unknown chat flags: {sorted(unknown)}.')
    with self._lock:
      record = self._live(id)
      chat(record, n)
      self._write({
          **record,
          'chats': [
              {**found, **flags} if found['n'] == n else found
              for found in record['chats']
          ],
      })

  def set_anchor_points(  # pylint: disable=redefined-builtin
      self,
      id: str,
      points: dict[str, Sequence[str]],
      ops: dict[str, Sequence[str]] | None = None,
  ) -> dict[str, Any]:
    """Keeps the anchors found that a record captures, by artifact.

    And the points of the operators it chose inside the layers, `ops`,
    by artifact (`tap_op_points`), if it chose any. A record that ran
    keeps those it ran with: this changes nothing.

    Args:
        id: The session ID.
        points: The anchor points by artifact.
        ops: The operator points by artifact.

    Returns:
        The record.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      record = self._live(id)
      if has_turn(record) or record['execution_ended']:
        return copy.deepcopy(record)
      clean = {artifact: list(ids) for artifact, ids in points.items()}
      changed = {**record, 'tap_anchor_points': clean}
      changed.pop('tap_op_points', None)
      if tap_ops(record):
        changed['tap_op_points'] = {
            artifact: list(ids) for artifact, ids in (ops or {}).items()
        }
      if changed != record:
        self._write(changed)
      return self.get(id)

  def free_id(self, wanted: Any) -> str:
    """An id for a Session imported: the one it had, unless taken.

    Args:
        wanted: The id it had where it was exported.

    Returns:
        `wanted` if it is a UUID no record here has, else a new one.
    """
    with self._lock:
      if isinstance(wanted, str) and wanted not in self._records:
        try:
          if str(uuid.UUID(wanted)) == wanted:
            return wanted
        except ValueError:
          pass
      return str(uuid.uuid4())

  def adopt(  # pylint: disable=redefined-builtin
      self, record: Any, id: str, imported: dict[str, Any]
  ) -> dict[str, Any]:
    """Keeps the record of a Session imported from another computer.

    Its captures are in place under `id` already (session_archive). It
    has ended: its chats are read, and it never starts, as any Session
    that ran. It keeps its name unless a Session here has it: then it is
    named '<name> (2)', or '(3)', and so on, the first no Session has.

    Args:
        record: The record as it was exported, read as this server keeps records
          (upgrade()).
        id: Its id here (free_id()).
        imported: Where it came from: `{session_id, exported_at, imported_at}`.

    Returns:
        The record.

    Raises:
        ValueError: The record is not one of a Session of two runs, each
          of a registered model, whose chats, each as this server keeps
          them, kept a turn.
        Conflict: A record has the id.
    """
    if not isinstance(record, dict):
      raise ValueError('The Session record is not valid.')
    runs = record.get('runs')
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or not all(isinstance(run, dict) for run in runs)
        or {run.get('id') for run in runs} != set(capture_index.SIDES)
    ):
      raise ValueError('The Session has no Reference and Target runs.')
    for run in runs:
      if not isinstance(run.get('artifact'), str):
        raise ValueError('A run of the Session names no model.')
      self._models.get(run['artifact'])
    chats = record.get('chats')
    if not isinstance(chats, list):
      raise ValueError('The Session record holds no chats.')
    for one in chats:
      _check_chat(one)
    if len({one['n'] for one in chats}) != len(chats):
      raise ValueError('The Session record holds a chat twice.')
    if not has_turn(record):
      raise ValueError('The Session kept no turn.')
    if not isinstance(record.get('created_at'), str | None):
      raise ValueError('The Session record is not valid.')
    name = _validate_name(record.get('name'))
    adopted = {
        **copy.deepcopy(record),
        'id': id,
        'execution_ended': True,
        'imported': dict(imported),
    }
    # Exported by an older server, which marked deleted records so.
    adopted.pop('deleted', None)
    with self._lock:
      if id in self._records:
        raise errors.Conflict(f'A Session has the id {id}.')
      adopted['name'] = self._free_name(name)
      self._write(adopted)
      return copy.deepcopy(adopted)

  def _free_name(self, name: str) -> str:
    """`name`, or else the first '<name> (n)' from 2 no record has.

    Called with the lock held.

    Args:
        name: The base name to find a free variant of.

    Returns:
        The free name.
    """
    taken = {record['name'] for record in self._records.values()}
    number = 1
    free = name
    while free in taken:
      number += 1
      suffix = f' ({number})'
      free = name[: _NAME_LIMIT - len(suffix)].rstrip() + suffix
    return free

  def set_flags(self, id: str, **flags: Any) -> None:  # pylint: disable=redefined-builtin
    """Sets the execution's flags of a record in one step.

    The flags are `started`, `execution_ended` and `notice`. Read and
    written under the store's lock, so a rename meanwhile is kept.

    Args:
        id: The session ID.
        **flags: The execution flags to set.

    Raises:
        NotFound: No such record.
    """
    with self._lock:
      if id not in self._records:
        raise errors.NotFound('This session does not exist.')
      self._write({**self._records[id], **flags})

  def _model_sampling(self, runs: list[Any]) -> dict[str, Any]:
    """Each side's model's own sampling, for options.sampler_default().

    Args:
        runs: A record's runs.

    Returns:
        By side: what the last runner to load the run's model reported of
        its sampling, or None if none did or the model is gone.
    """
    sampling = {}
    for run in runs:
      if not isinstance(run, dict):
        continue
      try:
        model = self._models.get(run.get('artifact', ''))
      except errors.NotFound:
        model = None
      sampling[run.get('id')] = model.sampler_defaults if model else None
    return sampling

  def _live(self, id: Any) -> dict[str, Any]:  # pylint: disable=redefined-builtin
    record = self._records.get(id) if isinstance(id, str) else None
    if record is None:
      raise errors.NotFound('This session does not exist.')
    return record

  def _write(self, record: dict[str, Any]) -> None:
    json_file.write_json(self._directory / f'{record["id"]}.json', record)
    self._records[record['id']] = record

  def _config(self, payload: Any) -> dict[str, Any]:
    """Checks what the editor posts; returns its kept fields.

    They are name, model (models_label() of the runs' models), runs,
    tap_anchors, tap_ops, tap_profile and tap_points.

    Args:
        payload: What the editor posts; a `model` it names is dropped.

    Returns:
        The validated configuration dictionary.
    """
    if not isinstance(payload, dict):
      raise ValueError('The session configuration is not valid.')
    unknown = set(payload) - _CONFIG_KEYS
    if unknown:
      raise ValueError(
          'Unknown session settings: ' + ', '.join(sorted(unknown)) + '.'
      )
    name = _validate_name(payload.get('name'))
    runs = payload.get('runs')
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or not all(isinstance(run, dict) for run in runs)
        or {run.get('id') for run in runs} != set(capture_index.SIDES)
    ):
      raise ValueError('Reference and Target runs are required.')
    by_side = {run['id']: run for run in runs}
    clean = [self._run(side, by_side[side]) for side in capture_index.SIDES]
    # A PyTorch run observes by its own option (observe), not points.
    artifacts = {
        run['artifact']
        for run in clean
        if run['runtime'] == model_store.LITERT_LM
    }
    anchors = _anchors(clean)
    profile, points = _taps(
        payload.get('tap_profile', ''),
        payload.get('tap_points', {}),
        artifacts,
        self._models,
        self._scans,
        frozenset(run['artifact'] for run in clean if anchors.get(run['id'])),
    )
    return {
        'name': name,
        'model': models_label(
            *(self._models.get(run['artifact']) for run in clean)
        ),
        'runs': clean,
        'tap_anchors': anchors,
        'tap_ops': _ops(payload.get('tap_ops')),
        'tap_profile': profile,
        'tap_points': points,
    }

  def _run(self, side: str, run: dict[str, Any]) -> dict[str, Any]:
    """Checks one run and returns what is kept of it."""
    where = _SIDE_NAMES[side]
    unknown = (
        set(run)
        - {'id', _BUILD}
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
    if clean['runtime'] not in _RUNTIMES:
      raise ValueError(
          f'{where}: the runtime must be ' + ' or '.join(_RUNTIMES) + '.'
      )
    try:
      model = self._models.get(clean['artifact'])
    except errors.NotFound:
      raise ValueError(f'{where}: select a model.') from None
    if model.runtime != clean['runtime']:
      raise ValueError(
          f'{where}: {model.name} is a {model.runtime} model; choose a '
          f'{clean["runtime"]} one.'
      )
    if not model.present():
      raise ValueError(
          f'{where}: the model {model.name} is not on the Server; '
          'upload it or download it again.'
      )
    if clean['source'] == HUGGING_FACE:
      if not model.source:
        raise ValueError(
            f'{where}: {model.name} did not come from Hugging Face.'
        )
      clean[HUGGING_FACE] = dict(model.source)
    if not clean['device']:
      raise ValueError(f'{where}: select a device.')
    build = _text(run.get(_BUILD) or '', f'{where} runner version')
    pytorch = clean['runtime'] == model_store.PYTORCH
    if pytorch and build != runner_builds.PYTORCH:
      raise ValueError(f'{where}: choose the PyTorch runner.')
    if build:
      if (
          not pytorch
          and build != runner_builds.DEVELOPMENT
          and not _COMMIT.fullmatch(build)
      ):
        raise ValueError(
            f'{where}: the runner version must be a whole LiteRT-LM '
            'commit or the development build.'
        )
      clean[_BUILD] = build
    if pytorch and clean['device'] != launcher.LOCAL_DEVICE_ID:
      raise ValueError(f'{where}: a PyTorch model runs on this computer only.')
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
    if pytorch:
      # A PyTorch run always captures its anchors; an `observe` posted
      # off is ignored.
      clean['observe'] = options.OBSERVE_ANCHORS
    elif clean.get('observe') or clean.get('observeMemoryGiB'):
      raise ValueError(
          f'{where}: only a PyTorch run captures anchors by its own '
          'option; choose tensor outputs for a LiteRT-LM model.'
      )
    return clean
