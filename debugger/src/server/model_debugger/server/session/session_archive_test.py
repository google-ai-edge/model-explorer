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

"""Tests of a Session exported as one file and imported elsewhere."""

import asyncio
import io
import json
import pathlib
import tarfile

from absl.testing import absltest
from model_debugger.server.session import model_store
from model_debugger.server.session import session_archive
from model_debugger.server.session import tap_scans
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_POINTS = ['decode:0:0']


def _archives(world: world_lib.World) -> session_archive.SessionArchives:
  return session_archive.SessionArchives(
      world.workspace,
      world.sessions,
      world.models,
      world.scans,
      world.graphs,
      world.captures,
      world.model_servers,
      world.exports,
  )


def _tar(members: list[tuple[str, bytes]], symlink: str = '') -> bytes:
  """An uncompressed tar of `members`, and of a link named `symlink`."""
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode='w') as tar:
    for name, data in members:
      info = tarfile.TarInfo(name)
      info.size = len(data)
      tar.addfile(info, io.BytesIO(data))
    if symlink:
      info = tarfile.TarInfo(symlink)
      info.type = tarfile.SYMTYPE
      info.linkname = '/etc/passwd'
      tar.addfile(info)
  return buffer.getvalue()


class SessionArchiveTest(absltest.TestCase):
  """Export on one workspace, import on another that lacks the model."""

  def setUp(self):
    """A Session of a model it observes a point of, ended after 2 chats.

    Each chat kept a turn; the second runs with settings of its own.
    """
    super().setUp()
    self.world = world_lib.create(self)
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    config['name'] = 'CPU / GPU: "tapped"'
    config['tap_profile'] = 'custom-outputs-v1'
    config['tap_points'] = {self.model.artifact: _POINTS}
    self.record = self.world.sessions.create(config)
    self.world.start(self.record)
    job = self.world.turn(self.record['id'], 'What is 2 + 2?')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.world.chat(
        self.record['id'],
        {'ref': {'systemPrompt': 'Be brief.'}, 'target': {}},
    )
    job = self.world.turn(self.record['id'], 'And 3 + 3?')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.world.model_servers.close(self.record['id'])
    self.archives = _archives(self.world)
    self.other = self._world()

  def _world(self) -> world_lib.World:
    """Another computer: an empty workspace that lacks the model."""
    made = world_lib.World(
        pathlib.Path(self.create_tempdir().full_path),
        [fake_launchers.LOCAL],
    )
    self.addCleanup(made.close)
    return made

  def _export(self, archives, record_id: str) -> pathlib.Path:
    """Writes a Session's export to a file, as the browser saves it."""
    exported = archives.export(record_id)
    path = pathlib.Path(self.create_tempdir().full_path) / 'x.mdsession'
    try:
      with open(path, 'wb') as file:
        for chunk in exported.chunks:
          file.write(chunk)
    finally:
      exported.end()
    self.assertEqual(path.stat().st_size, exported.size)
    return path

  def test_an_ended_session_is_read_where_its_model_is_not(self):
    """The import has every chat's capture and the model's facts."""
    path = self._export(self.archives, self.record['id'])
    with tarfile.open(path) as tar:
      names = tar.getnames()
    sha = self.model.sha256
    self.assertEqual(names[:2], ['manifest.json', 'record.json'])
    self.assertNotIn('bookmarks.json', names)
    self.assertIn(f'models/{sha}/model.json', names)
    self.assertIn(f'models/{sha}/facts/{model_store.DESCRIBED}', names)
    self.assertIn(f'models/{sha}/facts/{tap_scans.SCAN}', names)
    self.assertIn('captures/chat-1/index.json', names)
    self.assertIn('captures/chat-2/index.json', names)
    manifest = json.loads(
        tarfile.open(path).extractfile('manifest.json').read()
    )
    self.assertEqual(manifest['version'], session_archive.VERSION)

    imported = _archives(self.other).import_path(path)
    record_id = imported['id']
    self.assertEqual(record_id, self.record['id'])
    self.assertEqual(imported['name'], self.record['name'])
    self.assertIs(imported['execution_ended'], True)
    self.assertEqual(imported['imported']['session_id'], record_id)
    model = self.other.models.get(self.model.artifact)
    self.assertFalse(model.present())
    self.assertIsNone(model.path)
    self.assertEqual(model.described(), self.model.described())
    self.assertEqual(
        self.other.scans.scan(model), self.world.scans.scan(self.model)
    )
    for n in (1, 2):
      mine = self.world.captures.get(record_id, n)
      theirs = self.other.captures.get(record_id, n)
      self.assertEqual(theirs.turns, mine.turns)
      self.assertEqual(theirs.sides, mine.sides)
    self.assertEqual(
        imported['chats'], self.world.sessions.get(record_id)['chats']
    )
    # Its chats are read: it has ended, and never starts.
    summary = self.other.summary(record_id)
    self.assertEqual(
        [
            (chat['title'], chat['state'], chat['actions']['debug'])
            for chat in summary['chats']
        ],
        [
            ('Chat 1', 'ended', {'allowed': True, 'reason': ''}),
            ('Chat 2', 'ended', {'allowed': True, 'reason': ''}),
        ],
    )
    self.assertTrue(summary['actions']['enter']['allowed'])
    self.assertFalse(summary['actions']['new_chat']['allowed'])
    self.assertTrue(summary['actions']['export']['allowed'])
    self.assertIsNone(summary['model_server']['button'])
    self.assertEqual(
        self.other.prepared.observed(imported),
        self.world.prepared.observed(self.world.sessions.get(record_id)),
    )

    # Imported again, and again: another id each, and a name of its own.
    again = _archives(self.other).import_path(path)
    self.assertNotEqual(again['id'], record_id)
    self.assertEqual(again['name'], self.record['name'] + ' (2)')
    thrice = _archives(self.other).import_path(path)
    self.assertEqual(thrice['name'], self.record['name'] + ' (3)')

    # Exported from there, without the model file, and imported again.
    third = self._world()
    moved = _archives(third).import_path(
        self._export(_archives(self.other), record_id)
    )
    self.assertEqual(
        third.scans.scan(third.models.get(self.model.artifact)),
        self.world.scans.scan(self.model),
    )
    self.assertEqual(
        third.captures.get(moved['id'], 2).turns,
        self.world.captures.get(record_id, 2).turns,
    )

  def test_an_older_export_is_a_session_of_one_ended_chat(self):
    """Version 1, of one chat in its record: Chat 1, its capture moved.

    Its settings lose those of tool calling (session_store.upgrade()).
    """
    path = self._export(self.archives, self.record['id'])
    with tarfile.open(path) as tar:
      members = dict(
          (member.name, tar.extractfile(member).read()) for member in tar
      )
    manifest = json.loads(members.pop('manifest.json'))
    record = json.loads(members.pop('record.json'))
    first, _ = record.pop('chats')
    record.pop('notice')
    # Of a chat with tools, which this server drops.
    record.update(
        generation={
            side: {**settings, 'tools': '[]', 'constrainedDecoding': 'on'}
            for side, settings in first['generation'].items()
        },
        has_capture=True,
        chat_over=True,
        notice='Generation stopped. This chat is over.',
    )
    older = [
        ('manifest.json', json.dumps({**manifest, 'version': 1}).encode()),
        ('record.json', json.dumps(record).encode()),
    ]
    for name, data in members.items():
      if name.startswith('captures/chat-1/'):
        older.append(
            ('captures/' + name.removeprefix('captures/chat-1/'), data)
        )
      elif not name.startswith('captures/'):
        older.append((name, data))
    older_path = pathlib.Path(self.create_tempdir().full_path) / 'old.mdsession'
    older_path.write_bytes(_tar(older))

    imported = _archives(self.other).import_path(older_path)

    (chat,) = imported['chats']
    self.assertEqual(
        chat,
        {
            'n': 1,
            'created_at': record['created_at'],
            'generation': first['generation'],
            'has_capture': True,
            'over': True,
            'notice': 'Generation stopped. This chat is over.',
        },
    )
    for key in ('generation', 'has_capture', 'chat_over'):
      self.assertNotIn(key, imported)
    self.assertEqual(
        self.other.captures.get(imported['id'], 1).turns,
        self.world.captures.get(self.record['id'], 1).turns,
    )
    summary = self.other.summary(imported['id'])
    self.assertEqual(summary['status_text'], 'Ended')
    self.assertTrue(summary['chats'][0]['actions']['debug']['allowed'])
    # Exported again, it is of this version.
    again = self._export(_archives(self.other), imported['id'])
    with tarfile.open(again) as tar:
      self.assertIn('captures/chat-1/index.json', tar.getnames())

  def test_an_older_exports_bookmarks_are_ignored(self):
    """The bookmarks.json an older server exported is read past."""
    path = self._export(self.archives, self.record['id'])
    with tarfile.open(path) as tar:
      members = [
          (member.name, tar.extractfile(member).read()) for member in tar
      ]
    older = pathlib.Path(self.create_tempdir().full_path) / 'old.mdsession'
    older.write_bytes(
        _tar(
            members[:2]
            + [('bookmarks.json', b'{"bookmarks": []}')]
            + members[2:]
        )
    )
    imported = _archives(self.other).import_path(older)
    self.assertEqual(imported['id'], self.record['id'])
    self.assertFalse((self.other.workspace.root / 'bookmarks').exists())

  def test_a_session_being_exported_waits_for_its_export(self):
    """Until its stream ends: not exported again, deleted or opened."""
    record_id = self.record['id']
    exported = self.archives.export(record_id)
    summary = self.world.summary(record_id)
    self.assertTrue(summary['exporting'])
    refused = {'allowed': False, 'reason': session_archive.EXPORTING}
    for action in ('export', 'delete', 'enter'):
      self.assertEqual(summary['actions'][action], refused, action)
    with self.assertRaisesRegex(errors.Conflict, 'being exported'):
      self.archives.export(record_id)
    # Given up by the client part way: the export ends all the same.
    next(exported.chunks)
    exported.end()
    exported.end()
    summary = self.world.summary(record_id)
    self.assertFalse(summary['exporting'])
    for action in ('export', 'delete', 'enter'):
      self.assertTrue(summary['actions'][action]['allowed'], action)

  def test_only_a_session_that_ended_is_exported(self):
    """One with no turn, or whose Model Server is on, is not exported."""
    unused = self.world.create()
    with self.assertRaisesRegex(errors.Conflict, 'no turn to export'):
      self.archives.export(unused['id'])
    self.assertEqual(
        self.world.summary(unused['id'])['actions']['export'],
        {'allowed': False, 'reason': session_archive.NO_TURN},
    )
    running = self.world.create()
    self.world.start(running)
    self.world.turn(running['id'], 'Hi')
    with self.assertRaisesRegex(errors.Conflict, 'off first'):
      self.archives.export(running['id'])
    self.assertFalse(
        self.world.summary(running['id'])['actions']['export']['allowed']
    )
    self.world.model_servers.close(running['id'])
    self.assertTrue(
        self.world.summary(running['id'])['actions']['export']['allowed']
    )

  def test_a_file_that_is_no_export_is_refused_whole(self):
    """Unknown or escaping members, links, other formats: nothing kept."""
    path = self._export(self.archives, self.record['id'])
    with tarfile.open(path) as tar:
      good = [(member.name, tar.extractfile(member).read()) for member in tar]
    manifest = json.loads(dict(good)['manifest.json'])
    root = pathlib.Path(self.create_tempdir().full_path)
    for label, data, message in (
        ('not a tar', b'hello' * 200, 'not a .mdsession'),
        ('escapes', _tar(good + [('captures/../../x', b'')]), 'no export'),
        ('absolute', _tar(good + [('/etc/x', b'')]), 'no export'),
        ('unknown', _tar(good + [('notes.txt', b'')]), 'no export'),
        ('link', _tar(good, symlink='captures/link'), 'not a file'),
        ('twice', _tar(good + good[:1]), 'twice'),
        (
            'format',
            _tar(
                [(
                    'manifest.json',
                    json.dumps({**manifest, 'format': 'x'}).encode(),
                )]
                + good[1:]
            ),
            'not a .mdsession',
        ),
        (
            'version',
            _tar(
                [(
                    'manifest.json',
                    json.dumps({**manifest, 'version': 3}).encode(),
                )]
                + good[1:]
            ),
            'version 3',
        ),
        (
            'no chat',
            _tar(good + [('captures/chat-9/index.json', b'{}')]),
            'no chat of its Session',
        ),
        (
            'no model',
            _tar([m for m in good if not m[0].startswith('models/')]),
            'exactly the models',
        ),
    ):
      with self.subTest(label):
        bad = root / f'{label}.mdsession'
        bad.write_bytes(data)
        with self.assertRaisesRegex(ValueError, message):
          _archives(self.other).import_path(bad)
        self.assertEqual(self.other.sessions.list(), [])
        self.assertEqual(list(self.other.workspace.captures.iterdir()), [])
        self.assertEqual(list(self.other.workspace.uploads.iterdir()), [])

  def test_an_upload_must_be_the_size_it_announced(self):
    """The bytes are streamed to a file, then imported; none stay."""
    data = self._export(self.archives, self.record['id']).read_bytes()

    async def chunks(pieces):
      for piece in pieces:
        yield piece

    archives = _archives(self.other)
    for size, pieces, message in (
        (len(data) + 1, [data], 'ended after'),
        (len(data) - 1, [data], 'more than'),
        (0, [], 'non-empty'),
    ):
      with self.assertRaisesRegex(ValueError, message):
        asyncio.run(archives.import_file(size, chunks(pieces)))
    self.assertEqual(list(self.other.workspace.uploads.iterdir()), [])
    record = asyncio.run(
        archives.import_file(len(data), chunks([data[:5000], data[5000:]]))
    )
    self.assertEqual(record['id'], self.record['id'])
    self.assertEqual(list(self.other.workspace.uploads.iterdir()), [])
    stored = json_file.read_json(self.other.workspace.root / 'models.json')
    self.assertNotIn('path', stored['models'][self.model.artifact])

  def test_file_names_hold_no_path(self):
    """A name becomes a file name of its own, and a header of it."""
    self.assertEqual(
        session_archive.file_name('a/b: "c"'), 'a_b_ _c_.mdsession'
    )
    self.assertEqual(session_archive.file_name('..'), 'session.mdsession')
    self.assertEqual(
        session_archive.content_disposition('Gemma · CPU'),
        'attachment; filename="Gemma _ CPU.mdsession"; '
        "filename*=UTF-8''Gemma%20%C2%B7%20CPU.mdsession",
    )


if __name__ == '__main__':
  test_main.main()
