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

"""A Session moved to another computer: exported as one file, imported there.

A Session that has ended (check_exportable()) is exported as one file,
`<name>.mdsession`: an uncompressed tar, streamed as it is read and never
held whole, of

    manifest.json               {format, version, session_id, exported_at}
    record.json                 its record (session_store), its chats in it
    captures/chat-<n>/...       each chat's capture, as captures/<id>/ holds
                                them
    graphs/<sha256>/...         each of its models' graph, if one is kept
    mappings/<a>_<b>.json       the pairs made by hand of its two models, if any
    models/<sha256>/model.json  each of its models: {sha256, name, size,
                                runtime, source, sampler_defaults}
    models/<sha256>/facts/...   what the model's file says that reading the
                                Session reads (model_store.Model.facts):
                                described.json, a PyTorch model's
                                config.json, a LiteRT-LM model's scans

The model files themselves are not in it: the computer it is imported on
reads the Session in Debug from these alone. An import is streamed into
`uploads/` first, then unpacked there member by member: a member of a name
not above, or that is not a regular file, refuses the whole file, as does a
manifest of another format or version; the `bookmarks.json` an older
server's export holds is taken and ignored. An export of VERSION 1, of an
older server, holds a Session of one chat, its record's and its capture
directly under captures/: it is read as a Session of one ended chat, Chat 1
(session_store.upgrade()). Its models are registered with their
facts and without a file (model_store.ModelStore.adopt()), unless known
already; its graphs and pairs made by hand are kept unless the workspace has
its own; its captures move under its id, the one it had unless
a record here has it (session_store.SessionStore.free_id()); its record,
written last, says it was imported and has ended
(session_store.SessionStore.adopt()). One import runs at a time.

While its file is being sent, a Session is exporting (Exports): from the
export's start until its stream ends, sent whole or given up by the client,
it is neither exported again, nor deleted, nor opened (EXPORTING), and the
listing says so.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
import dataclasses
import json
import os
import pathlib
import re
import shutil
import tarfile
import threading
import time
from typing import Any
import urllib.parse
import uuid

from absl import logging
from model_debugger.server.capture import capture_index
from model_debugger.server.session import capture_store
from model_debugger.server.session import mappings
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.session import model_servers as model_servers_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.util import clock
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

FORMAT = 'model-debugger-session'
# The version this server writes: a Session of several chats. It reads
# _VERSIONS, those of older servers among them.
VERSION = 2
_VERSIONS = (1, VERSION)
SUFFIX = '.mdsession'
IMPORT_LIMIT_BYTES = 256 * 1024**3
# Why a Session is not exported, as the listing shows it.
NO_TURN = 'This Session has no turn to export.'
NOT_ENDED = 'A Session is exported once it has ended.'
# Why a Session being exported is not exported again, deleted or opened.
EXPORTING = 'This Session is being exported; wait until its file is saved.'
# The members of an export, by name; captures/ and graphs/<sha256>/ hold any
# files below them.
_MANIFEST = 'manifest.json'
_RECORD = 'record.json'
# What an older server's export held of its bookmarks, ignored: there are none
# any more.
_BOOKMARKS = 'bookmarks.json'
_CAPTURES = 'captures'
_GRAPHS = 'graphs'
_MAPPINGS = 'mappings'
_MODELS = 'models'
_MODEL = 'model.json'
_FACTS = 'facts'
_INDEX = 'index.json'
_FACT_FILES = (
    model_store.DESCRIBED,
    model_store.CONFIG,
    tap_scans.SCAN,
    tap_scans.LAYERED_SCAN,
)
_SHA = '[0-9a-f]{64}'
_MEMBER = re.compile(
    '|'.join((
        re.escape(_MANIFEST),
        re.escape(_RECORD),
        re.escape(_BOOKMARKS),
        f'{_CAPTURES}/.+',
        f'{_GRAPHS}/{_SHA}/.+',
        f'{_MAPPINGS}/{_SHA}_{_SHA}\\.json',
        f'{_MODELS}/{_SHA}/{re.escape(_MODEL)}',
        (
            f'{_MODELS}/{_SHA}/{_FACTS}/'
            f'(?:{"|".join(re.escape(name) for name in _FACT_FILES)})'
        ),
    ))
)
_NAME_LIMIT = 1024
# Bytes read or written at once.
_CHUNK = 1024 * 1024
# Disk an import leaves free, beyond twice its size: the file and what it
# unpacks to.
_FREE_MARGIN_BYTES = 256 * 1024**2
# The two zero blocks that end a tar.
_END = 2 * tarfile.BLOCKSIZE
# Characters a file name of the Session's name cannot hold.
_UNSAFE = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')


def check_exportable(
    state: model_server_lib.ModelServerState | None, record: dict[str, Any]
) -> None:
  """Refuses exporting a Session that has not ended.

  The rule of an export, for it to apply and the listing to show: a Session
  a chat of which kept a turn, whose Model Server is off, and that has
  ended (model_server.check_not_ended()), with all its chats.

  Args:
      state: The Session's Model Server now, or None if it never ran here.
      record: The Session's record.

  Raises:
      Conflict: It kept no turn, its Model Server is on, or it has not
        ended.
  """
  if not session_store.has_turn(record):
    raise errors.Conflict(NO_TURN)
  model_servers_lib.check_idle(state)
  try:
    model_server_lib.check_not_ended(state, record)
  except errors.Conflict:
    return
  raise errors.Conflict(NOT_ENDED)


def file_name(name: str) -> str:
  """The name of the file a Session of `name` is exported as."""
  safe = _UNSAFE.sub('_', name).strip(' .') or 'session'
  return safe + SUFFIX


def content_disposition(name: str) -> str:
  """A Content-Disposition that saves an export as file_name(name)."""
  saved = file_name(name)
  fallback = saved.encode('ascii', 'replace').decode().replace('?', '_')
  quoted = urllib.parse.quote(saved, safe='')
  return f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quoted}'


class Exports:
  """The Sessions being exported now, each until its stream ends.

  Thread-safe.
  """

  def __init__(self) -> None:
    """Holds no export."""
    self._lock = threading.Lock()
    self._exporting: set[str] = set()

  def exporting(self, record_id: str) -> bool:
    """Whether a Session's export is being sent now."""
    with self._lock:
      return record_id in self._exporting

  def check_not_exporting(self, record_id: str) -> None:
    """Refuses what waits for a Session's export to end.

    The rule of opening, deleting and exporting it again, for each to
    apply and the listing to show.

    Args:
      record_id: The Session identifier.

    Raises:
      Conflict: Its export is being sent (EXPORTING).
    """
    if self.exporting(record_id):
      raise errors.Conflict(EXPORTING)

  def begin(self, record_id: str) -> Callable[[], None]:
    """Marks a Session exporting until the function returned is called.

    Args:
      record_id: The Session identifier.

    Returns:
      Cleanup callback to invoke when export completes.

    Raises:
      Conflict: It is being exported already (EXPORTING).
    """
    with self._lock:
      if record_id in self._exporting:
        raise errors.Conflict(EXPORTING)
      self._exporting.add(record_id)
    ended = threading.Event()

    def end() -> None:
      if not ended.is_set():
        ended.set()
        with self._lock:
          self._exporting.discard(record_id)

    return end


@dataclasses.dataclass(frozen=True)
class Export:
  """A Session's export, read as it is sent.

  Attributes:
      name: The Session's name.
      size: The bytes `chunks` yields.
      chunks: The tar, in pieces; the files are read as it is iterated.
      end: Ends the export: called once its stream ended, sent whole or not
        (Exports); idempotent.
  """

  name: str
  size: int
  chunks: Iterator[bytes]
  end: Callable[[], None]


@dataclasses.dataclass(frozen=True)
class _Member:
  """One file of an export: its bytes, or the file holding them."""

  name: str
  size: int
  data: bytes | None = None
  path: pathlib.Path | None = None


class SessionArchives:
  """Exports and imports the Sessions of a workspace."""

  def __init__(
      self,
      workspace_: workspace.Workspace,
      sessions: session_store.SessionStore,
      models: model_store.ModelStore,
      scans: tap_scans.TapScans,
      graphs: model_graphs.ModelGraphs | None,
      captures: capture_store.CaptureStore,
      model_servers: model_servers_lib.ModelServers,
      exports: Exports | None = None,
  ) -> None:
    """Reads and writes the stores given.

    Args:
        workspace_: The workspace.
        sessions: The Session records.
        models: The registered models.
        scans: Their scans.
        graphs: Their graphs; None if the server keeps none.
        captures: Each chat's capture.
        model_servers: Every Session's Model Server, which an export of it must
          be off.
        exports: The Sessions being exported; a new set if None.
    """
    self._workspace = workspace_
    self._sessions = sessions
    self._models = models
    self._scans = scans
    self._graphs = graphs
    self._captures = captures
    self._model_servers = model_servers
    self._exports = exports if exports is not None else Exports()
    self._lock = threading.Lock()

  # ---- Export.

  def export(self, record_id: str) -> Export:
    """A Session's export: what it holds is listed now, read as sent.

    The Session is exporting from now until the export's `end` is
    called (Exports).

    Args:
      record_id: The Session identifier.

    Returns:
      The prepared Export instance.

    Raises:
      NotFound: No such Session.
      Conflict: As check_exportable(), or it is being exported already.
    """
    record = self._sessions.get(record_id)
    check_exportable(self._model_servers.state(record_id), record)
    self._exports.check_not_exporting(record_id)
    for chat in record['chats']:
      if chat['has_capture']:
        self._captures.get(record_id, chat['n'])
    members = [
        _json_member(
            _MANIFEST,
            {
                'format': FORMAT,
                'version': VERSION,
                'session_id': record_id,
                'exported_at': clock.now(),
            },
        ),
        _json_member(_RECORD, record),
    ]
    models = {}
    for run in record['runs']:
      model = self._models.get(run['artifact'])
      models[run['id']] = model
    for model in {m.sha256: m for m in models.values()}.values():
      members += self._model_members(model)
    pairs = self._workspace.mappings / mappings.file_name(
        *(models[side].sha256 for side in capture_index.SIDES)
    )
    if pairs.is_file():
      members.append(_file_member(f'{_MAPPINGS}/{pairs.name}', pairs))
    members += _tree(
        self._workspace.captures / record_id, _CAPTURES, sort_index=True
    )
    mtime = int(time.time())
    size = sum(_member_size(m, mtime) for m in members) + _END
    end = self._exports.begin(record_id)
    logging.info('Exporting %s: %d bytes', session_store.label(record), size)
    return Export(record['name'], size, _stream(members, mtime), end)

  def _model_members(self, model: model_store.Model) -> list[_Member]:
    """A model's metadata, facts and kept graph, as an export holds them.

    Facts its file cannot give, as of a model that is gone or cannot be
    scanned, are left out: what reads them says why, here and there.

    Args:
      model: The model to extract members for.

    Returns:
      List of archive members.
    """
    base = f'{_MODELS}/{model.sha256}'
    members = [
        _json_member(
            f'{base}/{_MODEL}',
            {
                'sha256': model.sha256,
                'name': model.name,
                'size': model.size,
                'runtime': model.runtime,
                'source': model.source,
                'sampler_defaults': model.sampler_defaults,
            },
        ),
        _json_member(
            f'{base}/{_FACTS}/{model_store.DESCRIBED}', model.described()
        ),
    ]
    try:
      if model.runtime == model_store.PYTORCH:
        members.append(
            _json_member(
                f'{base}/{_FACTS}/{model_store.CONFIG}', model.config()
            )
        )
      else:
        layers = (
            self._graphs.layers(model)
            if self._graphs is not None
            else model_graphs.Layers(None)
        )
        files = self._scans.stored_files(model, layers.stems, layers.groups)
        members += [
            _Member(f'{base}/{_FACTS}/{name}', len(data), data=data)
            for name, data in files.items()
        ]
    except (OSError, ValueError) as error:
      # What reads them here says why too; there it will say so alike.
      logging.warning(
          'The export of %s holds no scan or config.json: %s',
          model.name,
          error,
      )
    graph = self._workspace.graphs / model.sha256
    if graph.is_dir():
      members += _tree(graph, f'{_GRAPHS}/{model.sha256}')
    return members

  # ---- Import.

  async def import_file(
      self, size: int, chunks: AsyncIterator[bytes]
  ) -> dict[str, Any]:
    """Imports a Session from an export uploaded.

    Args:
        size: The bytes announced; exactly these many must arrive.
        chunks: The bytes, in order.

    Returns:
        The record imported.

    Raises:
        ValueError: The size is refused, the disk is too full, the bytes
          that arrived are not `size` bytes, or they are not an export
          this server reads.
    """
    if not 0 < size <= IMPORT_LIMIT_BYTES:
      raise ValueError(
          'Choose a non-empty file of at most '
          f'{IMPORT_LIMIT_BYTES // 1024**3} GiB.'
      )
    if (
        shutil.disk_usage(self._workspace.root).free
        < 2 * size + _FREE_MARGIN_BYTES
    ):
      raise ValueError(
          'There is not enough disk space to import this Session: it '
          'takes twice its size while it is unpacked.'
      )
    staging = self._workspace.uploads / f'{uuid.uuid4().hex}{SUFFIX}'
    received = 0
    try:
      with open(staging, 'xb') as file:
        pending = bytearray()
        async for chunk in chunks:
          received += len(chunk)
          if received > size:
            raise ValueError(f'The upload sent more than {size} bytes.')
          pending += chunk
          if len(pending) >= _CHUNK:
            await asyncio.to_thread(file.write, bytes(pending))
            pending.clear()
        if received != size:
          raise ValueError(
              f'The upload ended after {received} of {size} bytes.'
          )
        await asyncio.to_thread(file.write, bytes(pending))
      return await asyncio.to_thread(self.import_path, staging)
    finally:
      staging.unlink(missing_ok=True)

  def import_path(self, path: pathlib.Path) -> dict[str, Any]:
    """Imports a Session from an export on this computer.

    Args:
      path: Path to the export file.

    Returns:
      The record imported.

    Raises:
      ValueError: It is not an export this server reads.
    """
    unpacked = self._workspace.uploads / f'{uuid.uuid4().hex}.unpacked'
    try:
      unpacked.mkdir()
      _unpack(path, unpacked)
      record = self._adopt(unpacked)
      logging.info('Imported %s', session_store.label(record))
      return record
    finally:
      shutil.rmtree(unpacked, ignore_errors=True)

  def _adopt(self, unpacked: pathlib.Path) -> dict[str, Any]:
    """Moves an unpacked export into the workspace; returns its record.

    Args:
      unpacked: Directory where archive was unpacked.

    Returns:
      The adopted record.

    Raises:
      ValueError: What it holds is not a Session this server reads.
    """
    manifest = _read(unpacked / _MANIFEST, 'manifest')
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
      raise ValueError(f'This is not a {SUFFIX} file.')
    version = manifest.get('version')
    if version not in _VERSIONS:
      raise ValueError(
          f'This {SUFFIX} file is of version {version}; this server '
          f'reads {" and ".join(str(v) for v in _VERSIONS)}.'
      )
    record = _read(unpacked / _RECORD, 'Session record')
    if not isinstance(record, dict) or not isinstance(record.get('runs'), list):
      raise ValueError('The Session record is not valid.')
    # Of VERSION 1, a Session of one chat, its capture moved to Chat 1's.
    capture = unpacked / _CAPTURES
    capture.mkdir(exist_ok=True)
    record = session_store.upgrade(record, capture)
    wanted = {
        run.get('artifact') for run in record['runs'] if isinstance(run, dict)
    }
    described = {}
    for directory in sorted((unpacked / _MODELS).glob('*')):
      metadata = _read(directory / _MODEL, 'model description')
      if (
          not isinstance(metadata, dict)
          or metadata.get('sha256') != directory.name
      ):
        raise ValueError(
            f'The description of the model {directory.name} is not valid.'
        )
      described['model-' + directory.name[:24]] = directory
    if set(described) != wanted:
      raise ValueError(
          'The file does not describe exactly the models its Session loads.'
      )
    _check_captures(capture, record)
    old_id = manifest.get('session_id')
    if not isinstance(old_id, str) or not old_id:
      raise ValueError('The manifest names no Session.')
    imported = {
        'session_id': old_id,
        'exported_at': (
            manifest['exported_at']
            if isinstance(manifest.get('exported_at'), str)
            else None
        ),
        'imported_at': clock.now(),
    }
    # The Sessions' lock too: no model adopted here is deleted, unused,
    # before the record that names it is written.
    with self._lock, self._sessions.lock:
      record_id = self._sessions.free_id(old_id)
      while (self._workspace.captures / record_id).exists():
        record_id = str(uuid.uuid4())
      for directory in described.values():
        self._models.adopt(
            json_file.read_json(directory / _MODEL),
            self._keep(
                directory / _FACTS,
                self._workspace.facts / directory.name,
            ),
        )
      for graph in sorted((unpacked / _GRAPHS).glob('*')):
        kept = self._workspace.graphs / graph.name
        if not (kept / 'manifest.json').is_file():
          shutil.rmtree(kept, ignore_errors=True)
          os.replace(graph, kept)
      for pairs in sorted((unpacked / _MAPPINGS).glob('*')):
        if not (self._workspace.mappings / pairs.name).exists():
          os.replace(pairs, self._workspace.mappings / pairs.name)
      target = self._workspace.captures / record_id
      os.replace(capture, target)
      try:
        return self._sessions.adopt(record, record_id, imported)
      except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

  def _keep(self, facts: pathlib.Path, kept: pathlib.Path) -> pathlib.Path:
    """A model's facts in the workspace: those kept, else these moved."""
    if not kept.is_dir():
      if facts.is_dir():
        os.replace(facts, kept)
      else:
        kept.mkdir()
    return kept


def _header(member: _Member, mtime: int) -> bytes:
  info = tarfile.TarInfo(member.name)
  info.size = member.size
  info.mode = 0o644
  info.mtime = mtime
  return info.tobuf(tarfile.PAX_FORMAT, 'utf-8', 'surrogateescape')


def _member_size(member: _Member, mtime: int) -> int:
  return (
      len(_header(member, mtime))
      + member.size
      + (-member.size % tarfile.BLOCKSIZE)
  )


def _json_member(name: str, value: object) -> _Member:
  data = json.dumps(value, ensure_ascii=False, indent=1).encode('utf-8')
  return _Member(name, len(data), data=data)


def _file_member(name: str, path: pathlib.Path) -> _Member:
  return _Member(name, path.stat().st_size, path=path)


def _tree(
    root: pathlib.Path, prefix: str, sort_index: bool = False
) -> list[_Member]:
  """Every regular file under `root`, named under `prefix`, sorted.

  Args:
    root: The directory.
    prefix: What the members' names start with.
    sort_index: Whether the captures' index.json files go first.

  Returns:
    List of archive members.
  """
  found = sorted(
      path
      for path in root.rglob('*')
      if path.is_file() and not path.is_symlink()
  )
  if sort_index:
    found.sort(key=lambda path: path.name != _INDEX)
  return [
      _file_member(f'{prefix}/{path.relative_to(root).as_posix()}', path)
      for path in found
  ]


def _stream(members: list[_Member], mtime: int) -> Iterator[bytes]:
  """The tar of `members`, its files read a piece at a time.

  Args:
    members: The archive members to stream.
    mtime: Timestamp for tar headers.

  Yields:
    Tar archive chunks.

  Raises:
    OSError: A file changed size while it was read.
  """
  for member in members:
    yield _header(member, mtime)
    if member.data is not None:
      yield member.data
    else:
      with open(member.path, 'rb') as file:
        left = member.size
        while left:
          chunk = file.read(min(_CHUNK, left))
          if not chunk:
            raise OSError(f'{member.name} got shorter as it was read.')
          left -= len(chunk)
          yield chunk
    if padding := -member.size % tarfile.BLOCKSIZE:
      yield bytes(padding)
  yield bytes(_END)


def _check_name(name: str) -> None:
  """Refuses a member an export does not hold, or a name that escapes.

  Args:
    name: The member name.

  Raises:
    ValueError: Not a name of a member above, or one with an empty,
      '.' or '..' part, a backslash or a control character.
  """
  parts = name.split('/')
  if (
      len(name) > _NAME_LIMIT
      or any(part in ('', '.', '..') for part in parts)
      or '\\' in name
      or any(ord(char) < 32 or ord(char) == 127 for char in name)
      or not _MEMBER.fullmatch(name)
  ):
    raise ValueError(f'The file holds {name[:200]!r}, which no export does.')


def _unpack(path: pathlib.Path, into: pathlib.Path) -> None:
  """Unpacks an export, each member checked before it is written.

  Args:
    path: Path to the archive.
    into: Directory to unpack into.

  Raises:
    ValueError: It is not an uncompressed tar, or holds a member of a
      name check_name() refuses, one that is not a regular file, or one
      twice.
  """
  try:
    with tarfile.open(path, 'r:') as tar:
      seen = set()
      for member in tar:
        _check_name(member.name)
        if not member.isreg():
          raise ValueError(f'{member.name} is not a file.')
        if member.name in seen:
          raise ValueError(f'The file holds {member.name} twice.')
        seen.add(member.name)
        target = into / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = tar.extractfile(member)
        with source, open(target, 'xb') as file:
          shutil.copyfileobj(source, file, _CHUNK)
  except tarfile.TarError as error:
    raise ValueError(f'This is not a {SUFFIX} file: {error}') from error


def _read(path: pathlib.Path, what: str) -> object:
  """A member's JSON value.

  Args:
    path: Path to the JSON file.
    what: Description of what is being read.

  Returns:
    The loaded JSON object.

  Raises:
    ValueError: It is missing or not JSON.
  """
  try:
    return json_file.read_json(path)
  except FileNotFoundError:
    raise ValueError(f'The file holds no {what}.') from None
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f'The {what} cannot be read: {error}') from error


def _check_captures(root: pathlib.Path, record: dict[str, Any]) -> None:
  """Checks that an imported Session's captures are its chats', and read.

  Args:
    root: Path to the captures directory.
    record: The Session record dictionary.

  Raises:
    ValueError: A directory is not a chat's of the record, a chat that
      kept a turn has no capture, or a capture does not read
      (_check_capture()).
  """
  chats = record.get('chats')
  if not isinstance(chats, list) or not all(
      isinstance(chat, dict) and isinstance(chat.get('n'), int)
      for chat in chats
  ):
    raise ValueError('The Session record is not valid.')
  names = {workspace.chat_directory(chat['n']): chat for chat in chats}
  for entry in root.iterdir():
    if entry.name not in names or not entry.is_dir():
      raise ValueError(
          f'The file holds captures/{entry.name}, which is no chat of '
          'its Session.'
      )
    _check_capture(entry)
  for name, chat in names.items():
    if chat.get('has_capture') and not (root / name).is_dir():
      raise ValueError(f'Chat {chat["n"]} of the Session has no capture.')


def _check_capture(root: pathlib.Path) -> None:
  """Checks that an imported capture reads, its files all inside it.

  Args:
    root: Path to the capture directory.

  Raises:
    ValueError: It has no index, names a file outside its turn's
      directory, or cannot be read (capture_index.CaptureIndex.load()).
  """
  index = _read(root / _INDEX, 'capture')
  try:
    for turn in index['turns']:
      for side in turn['sides'].values():
        for file in side['files']:
          parts = pathlib.PurePosixPath(file['path']).parts
          if (
              not parts
              or file['path'].startswith('/')
              or '..' in parts
              or '\\' in file['path']
          ):
            raise ValueError(f'The capture names the file {file["path"]!r}.')
    capture_index.CaptureIndex.load(root)
  except (KeyError, TypeError, AttributeError, errors.NotFound) as error:
    raise ValueError(f'The capture cannot be read: {error!r}') from error
