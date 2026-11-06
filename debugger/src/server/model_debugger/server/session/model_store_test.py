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

"""Tests of the model store."""

import asyncio
import hashlib
import json
import os
import pathlib
from unittest import mock

from absl.testing import parameterized
from model_debugger.protocol import runner_pb2
from model_debugger.server.session import model_store
from model_debugger.server.session import workspace
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors


async def _chunks(data, size=7):
  for start in range(0, len(data), size):
    yield data[start : start + size]


def _upload(store, name, data, size=None):
  return asyncio.run(
      store.upload(name, len(data) if size is None else size, _chunks(data))
  )


class ModelStoreTest(parameterized.TestCase):
  """Tests of the model store."""

  def setUp(self):
    """An empty directory, and an open workspace in it."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.opened = workspace.Workspace(self.tmp_path / 'ws')
    self.addCleanup(self.opened.close)

  def test_a_download_is_kept_where_it_is(self):
    """A download is recorded in place, and known after a restart."""
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'model bytes')
    model = model_store.ModelStore(self.opened).add_download(path)
    sha256 = hashlib.sha256(b'model bytes').hexdigest()
    self.assertEqual(
        model,
        model_store.Model(
            artifact='model-' + sha256[:24],
            sha256=sha256,
            name='m.litertlm',
            size=11,
            path=path.resolve(),
        ),
    )
    # Kept where the download put it, and known after a restart.
    self.assertEqual(list(self.opened.models.iterdir()), [])
    self.assertEqual(
        model_store.ModelStore(self.opened).get(model.artifact), model
    )

  def test_a_vocabularys_pieces_are_read_from_a_model_that_has_it(self):
    """By the vocabulary's digest, of a model whose file is here."""
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'model bytes')
    store = model_store.ModelStore(self.opened)
    store.add_download(path)
    with (
        mock.patch.object(
            model_store.Model,
            'described',
            return_value={'vocabulary': {'table_sha256': 'v'}},
        ),
        mock.patch.object(
            model_store.model_facts,
            'described_pieces',
            return_value=(b'a', b'b'),
        ) as read,
    ):
      self.assertEqual(store.pieces('v'), (b'a', b'b'))
      self.assertIsNone(store.pieces('w'))
      read.assert_called_once_with(path.resolve(), mock.ANY)
      path.unlink()
      self.assertIsNone(store.pieces('v'))

  def test_a_download_recorded_again_is_not_hashed_again(self):
    """Recorded again, by size and modification time, unread."""
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'model bytes')
    first = model_store.ModelStore(self.opened).add_download(path)
    self.enter_context(mock.patch.object(model_store, '_hash_file', self.fail))
    self.assertEqual(
        model_store.ModelStore(self.opened).add_download(path), first
    )

  def test_the_models_own_sampling_is_kept_as_last_reported(self):
    """Kept per model, across restarts and recording again."""
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'model bytes')
    store = model_store.ModelStore(self.opened)
    model = store.add_download(path)
    self.assertIsNone(model.sampler_defaults)
    store.report_loaded(
        model.artifact,
        runner_pb2.LoadResponse(
            start_token=2,
            sampler_defaults=runner_pb2.SamplerDefaults(
                type=runner_pb2.SamplerDefaults.TOP_P, top_k=48
            ),
        ),
    )
    reported = {'type': 'TOP_P', 'top_k': 48}
    self.assertEqual(store.get(model.artifact).sampler_defaults, reported)
    again = model_store.ModelStore(self.opened)
    self.assertEqual(again.add_download(path).sampler_defaults, reported)
    self.assertEqual(again.get(model.artifact).sampler_defaults, reported)
    # A model that reports none, once it did.
    again.report_loaded(model.artifact, runner_pb2.LoadResponse())
    self.assertEqual(again.get(model.artifact).sampler_defaults, {})

  def test_a_changed_download_replaces_it(self):
    """A file changed where it was recorded replaces its model."""
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'one')
    store = model_store.ModelStore(self.opened)
    first = store.add_download(path)
    path.write_bytes(b'two!')
    os.utime(path, ns=(1, 1))
    second = store.add_download(path)
    self.assertEqual(second.sha256, hashlib.sha256(b'two!').hexdigest())
    self.assertEqual(store.list(), [second])
    with self.assertRaises(errors.NotFound):
      store.get(first.artifact)

  def test_add_download_refuses(self):
    """Not a .litertlm file of the workspace's downloads: refused."""
    store = model_store.ModelStore(self.opened)
    with self.assertRaisesRegex(ValueError, 'not a file'):
      store.add_download(self.opened.hf / 'missing.litertlm')
    (self.opened.hf / 'm.tflite').write_bytes(b'x')
    with self.assertRaisesRegex(ValueError, '.litertlm'):
      store.add_download(self.opened.hf / 'm.tflite')
    outside = self.tmp_path / 'm.litertlm'
    outside.write_bytes(b'model bytes')
    with self.assertRaisesRegex(ValueError, 'not a download'):
      store.add_download(outside)
    with self.assertRaisesRegex(ValueError, 'not a download'):
      store.add_download(self.opened.models / '..' / '..' / 'm.litertlm')
    self.assertEqual(store.list(), [])

  def test_upload_stores_by_sha256(self):
    """Upload stores by sha256."""
    data = bytes(range(256)) * 9000  # Spans more than one write.
    store = model_store.ModelStore(self.opened)
    model = _upload(store, 'gemma.litertlm', data)
    sha256 = hashlib.sha256(data).hexdigest()
    self.assertEqual(model.sha256, sha256)
    self.assertEqual(model.name, 'gemma.litertlm')
    self.assertLen(data, model.size)
    self.assertEqual(model.path, self.opened.models / f'{sha256}.litertlm')
    self.assertEqual(model.path.read_bytes(), data)
    self.assertEqual(list(self.opened.uploads.iterdir()), [])
    self.assertEqual(model_store.ModelStore(self.opened).list(), [model])

  def test_same_bytes_twice_is_one_file(self):
    """Same bytes twice is one file."""
    store = model_store.ModelStore(self.opened)
    first = _upload(store, 'a.litertlm', b'same')
    self.assertEqual(_upload(store, 'b.litertlm', b'same'), first)
    self.assertLen(list(self.opened.models.iterdir()), 1)
    path = self.opened.hf / 'c.litertlm'
    path.write_bytes(b'other')
    downloaded = store.add_download(path)
    self.assertEqual(_upload(store, 'd.litertlm', b'other'), downloaded)

  def test_upload_size_must_match(self):
    """Upload size must match."""
    store = model_store.ModelStore(self.opened)
    with self.assertRaisesRegex(ValueError, 'ended after 3 of 5'):
      _upload(store, 'a.litertlm', b'abc', size=5)
    with self.assertRaisesRegex(ValueError, 'more than 2'):
      _upload(store, 'a.litertlm', b'abc', size=2)
    self.assertEqual(list(self.opened.uploads.iterdir()), [])
    self.assertEqual(list(self.opened.models.iterdir()), [])
    self.assertEqual(store.list(), [])

  @parameterized.parameters(
      ('',),
      ('../x.litertlm',),
      ('a/b.litertlm',),
      ('x.bin',),
      ('.litertlm',),
      ('a\n.litertlm',),
  )
  def test_upload_refuses_names(self, name):
    """Upload refuses names."""
    with self.assertRaises(ValueError):
      _upload(model_store.ModelStore(self.opened), name, b'abc')

  def test_upload_refuses_sizes(self):
    """Upload refuses sizes."""
    store = model_store.ModelStore(self.opened)
    with self.assertRaisesRegex(ValueError, 'non-empty'):
      _upload(store, 'a.litertlm', b'', size=0)
    with self.assertRaisesRegex(ValueError, 'non-empty'):
      _upload(
          store,
          'a.litertlm',
          b'',
          size=model_store.UPLOAD_LIMIT_BYTES + 1,
      )

  def test_get_unknown(self):
    """Get unknown."""
    with self.assertRaises(errors.NotFound):
      model_store.ModelStore(self.opened).get('model-0')

  def _hf_directory(self) -> pathlib.Path:
    """A Hugging Face cache snapshot: files linked to blobs, a dot file."""
    repository = self.opened.hf / 'models--google--gemma-4-E2B-it'
    blobs = repository / 'blobs'
    blobs.mkdir(parents=True)
    snapshot = repository / 'snapshots' / 'abc123'
    (snapshot / 'sub').mkdir(parents=True)
    for name, data in (
        ('config.json', b'{"architectures": []}'),
        ('model.safetensors', b'weights'),
        ('sub/tokenizer.json', b'{}'),
    ):
      blob = blobs / hashlib.sha256(data).hexdigest()
      blob.write_bytes(data)
      (snapshot / name).symlink_to(blob)
    (snapshot / '.cache').mkdir()
    (snapshot / '.cache' / 'x').write_bytes(b'not the model')
    (snapshot / '.hidden').write_bytes(b'not the model')
    return snapshot

  def test_add_download_directory_names_it_by_its_manifest(self):
    """A model directory's id is its sorted manifest's SHA-256."""
    snapshot = self._hf_directory()
    manifest = ''.join(
        f'{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\n'
        for name, data in (
            ('config.json', b'{"architectures": []}'),
            ('model.safetensors', b'weights'),
            ('sub/tokenizer.json', b'{}'),
        )
    )

    model = model_store.ModelStore(self.opened).add_download_directory(snapshot)

    sha256 = hashlib.sha256(manifest.encode()).hexdigest()
    self.assertEqual(model.sha256, sha256)
    self.assertEqual(model.artifact, 'model-' + sha256[:24])
    self.assertEqual(model.name, 'google/gemma-4-E2B-it')
    self.assertEqual(model.runtime, model_store.PYTORCH)
    self.assertEqual(model.size, 30)
    self.assertEqual(model.path, snapshot.resolve())
    self.assertTrue(model.present())

  def test_add_download_directory_is_deterministic_and_hashes_once(self):
    """Each file is hashed once: the workspace keeps its hash."""
    snapshot = self._hf_directory()
    first = model_store.ModelStore(self.opened).add_download_directory(snapshot)

    with mock.patch.object(
        model_store, '_hash_file', side_effect=AssertionError
    ):
      again = model_store.ModelStore(self.opened).add_download_directory(
          snapshot
      )

    self.assertEqual(again, first)
    self.assertEqual(
        [m.artifact for m in model_store.ModelStore(self.opened).list()],
        [first.artifact],
    )
    self.assertEqual(
        model_store.ModelStore(self.opened).get(first.artifact).runtime,
        model_store.PYTORCH,
    )

  def test_add_download_directory_sees_a_file_change(self):
    """A changed file changes the id."""
    snapshot = self._hf_directory()
    store = model_store.ModelStore(self.opened)
    first = store.add_download_directory(snapshot)
    (snapshot / 'model.safetensors').unlink()
    (snapshot / 'model.safetensors').write_bytes(b'other weights')

    again = store.add_download_directory(snapshot)

    self.assertNotEqual(again.sha256, first.sha256)
    self.assertEqual([m.artifact for m in store.list()], [again.artifact])

  def test_add_download_directory_needs_a_config(self):
    """A directory without config.json is refused."""
    (self.opened.hf / 'empty').mkdir()

    with self.assertRaisesRegex(ValueError, 'config.json'):
      model_store.ModelStore(self.opened).add_download_directory(
          self.opened.hf / 'empty'
      )

  def test_a_model_imported_without_its_file_reads_its_facts(self):
    """adopt(): no path, never present; its facts answer for its file."""
    store = model_store.ModelStore(self.opened)
    facts = self.opened.facts / ('a' * 64)
    facts.mkdir()
    described = {
        'vocabulary': {'table_sha256': 'b' * 64},
        'kv_quantization': {},
    }
    (facts / model_store.DESCRIBED).write_text(json.dumps(described))
    (facts / model_store.CONFIG).write_text('{"model_type": "gemma4"}')
    metadata = {
        'sha256': 'a' * 64,
        'name': 'gemma-4-E2B-it',
        'size': 5,
        'runtime': model_store.PYTORCH,
        'source': None,
        'sampler_defaults': {'type': 'GREEDY'},
    }
    model = store.adopt(metadata, facts)
    self.assertIsNone(model.path)
    self.assertFalse(model.present())
    self.assertEqual(model.facts, facts)
    self.assertEqual(model.described(), described)
    self.assertEqual(model.config(), {'model_type': 'gemma4'})
    self.assertEqual(model.sampler_defaults, {'type': 'GREEDY'})
    # Known after a restart, still without a file.
    again = model_store.ModelStore(self.opened).get(model.artifact)
    self.assertEqual(again, model)
    # A model without facts or file knows nothing of its file.
    bare = store.adopt(
        {**metadata, 'sha256': 'c' * 64}, self.opened.facts / 'none'
    )
    self.assertEqual(
        bare.described(), {'vocabulary': None, 'kv_quantization': None}
    )
    with self.assertRaisesRegex(OSError, 'not on this computer'):
      bare.config()
    for bad in (
        {**metadata, 'sha256': 'A' * 64},
        {**metadata, 'name': 'a\nb'},
        {**metadata, 'size': -1},
        {**metadata, 'runtime': 'TF'},
        [],
    ):
      with self.assertRaises(ValueError):
        store.adopt(bad, facts)

  def test_a_known_model_imported_keeps_its_file_and_gains_facts(self):
    """Here already: kept as it is; downloaded later: facts stay."""
    store = model_store.ModelStore(self.opened)
    path = self.opened.hf / 'm.litertlm'
    path.write_bytes(b'model bytes')
    sha256 = hashlib.sha256(b'model bytes').hexdigest()
    facts = self.opened.facts / sha256
    facts.mkdir()
    metadata = {
        'sha256': sha256,
        'name': 'renamed.litertlm',
        'size': 11,
        'runtime': model_store.LITERT_LM,
        'source': None,
        'sampler_defaults': None,
    }
    imported = store.adopt(metadata, facts)
    self.assertFalse(imported.present())
    downloaded = store.add_download(path)
    self.assertEqual(downloaded.path, path.resolve())
    self.assertEqual(downloaded.facts, facts)
    self.assertEqual(store.adopt(metadata, facts), downloaded)

  def test_an_older_servers_model_outside_the_workspace_loses_its_path(self):
    """upgrade(): read as a model without its file; inside, kept."""
    outside = self.tmp_path / 'm.litertlm'
    outside.write_bytes(b'model bytes')
    inside = self.opened.hf / 'n.litertlm'
    inside.write_bytes(b'other bytes')
    records = {
        'model-a': {
            'sha256': 'a' * 64,
            'name': 'm.litertlm',
            'size': 11,
            'path': str(outside),
            'mtime_ns': 1,
            'sampler_defaults': {},
        },
        'model-b': {
            'sha256': 'b' * 64,
            'name': 'n.litertlm',
            'size': 11,
            'path': str(inside),
            'mtime_ns': 2,
        },
    }
    (self.opened.root / 'models.json').write_text(
        json.dumps({'models': records})
    )
    store = model_store.ModelStore(self.opened)
    older = store.get('model-a')
    self.assertIsNone(older.path)
    self.assertFalse(older.present())
    self.assertEqual(older.sampler_defaults, {})
    self.assertEqual(store.get('model-b').path, inside)
    self.assertTrue(outside.is_file(), 'a file outside is never touched')
    # Written back as this server keeps it.
    kept = json.loads((self.opened.root / 'models.json').read_text())
    self.assertNotIn('path', kept['models']['model-a'])
    self.assertIs(
        model_store.upgrade(kept['models'], self.opened.root),
        kept['models'],
    )

  def test_a_model_no_session_uses_is_deleted_with_its_files(self):
    """collect(): its file, its facts and its record go; others stay."""
    store = model_store.ModelStore(self.opened)
    uploaded = _upload(store, 'a.litertlm', b'uploaded')
    kept = _upload(store, 'b.litertlm', b'kept')
    snapshot = self._hf_directory()
    pytorch = store.add_download_directory(snapshot)
    repository = snapshot.parents[1]
    facts = self.opened.facts / ('c' * 64)
    facts.mkdir()
    imported = store.adopt(
        {
            'sha256': 'c' * 64,
            'name': 'imported',
            'size': 1,
            'runtime': model_store.LITERT_LM,
            'source': None,
            'sampler_defaults': None,
        },
        facts,
    )
    used = {kept.artifact}
    # Only those named: an upload no Session chose yet stays.
    removed = store.collect(
        used, {kept.artifact, pytorch.artifact, imported.artifact}
    )
    self.assertCountEqual(
        [m.artifact for m in removed], [pytorch.artifact, imported.artifact]
    )
    self.assertFalse(repository.exists(), 'its blobs and snapshot too')
    self.assertFalse(facts.exists())
    self.assertTrue(uploaded.path.is_file())
    self.assertEqual(
        [m.artifact for m in store.list()],
        [uploaded.artifact, kept.artifact],
    )
    self.assertEqual(
        json.loads((self.opened.root / 'file_hashes.json').read_text())[
            'files'
        ],
        {},
    )
    # As the server starts: every model no Session uses.
    self.assertEqual(
        [m.artifact for m in store.collect(used)], [uploaded.artifact]
    )
    self.assertFalse(uploaded.path.exists())
    self.assertTrue(kept.path.is_file())
    self.assertEqual(
        [m.artifact for m in model_store.ModelStore(self.opened).list()],
        [kept.artifact],
    )

  def test_one_file_of_a_repository_goes_and_the_others_stay(self):
    """A blob another snapshot's link names is kept."""
    repository = self.opened.hf / 'models--org--repo'
    blobs = repository / 'blobs'
    blobs.mkdir(parents=True)
    models = []
    store = model_store.ModelStore(self.opened)
    for name, data in (('a.litertlm', b'a'), ('b.litertlm', b'b')):
      blob = blobs / hashlib.sha256(data).hexdigest()
      blob.write_bytes(data)
      link = repository / 'snapshots' / 'c1' / name
      link.parent.mkdir(parents=True, exist_ok=True)
      link.symlink_to(blob)
      models.append(store.add_download(link))
    self.assertLen(models, 2)
    first, second = models[0], models[1]
    store.collect({second.artifact}, {first.artifact})
    self.assertFalse(first.path.exists())
    self.assertTrue(second.present())
    self.assertEqual(
        sorted(path.name for path in blobs.iterdir()),
        [hashlib.sha256(b'b').hexdigest()],
    )

  def test_the_start_clears_what_no_model_holds(self):
    """collect(used): stray uploads, downloads and facts go."""
    store = model_store.ModelStore(self.opened)
    kept = _upload(store, 'a.litertlm', b'kept')
    (self.opened.models / 'stray.litertlm').write_bytes(b'x')
    (self.opened.facts / ('d' * 64)).mkdir()
    repository = self.opened.hf / 'models--org--repo'
    (repository / 'blobs').mkdir(parents=True)
    (repository / 'blobs' / 'x.incomplete').write_bytes(b'half')
    (self.opened.hf / 'loose.litertlm').write_bytes(b'x')
    (self.opened.hf / 'xet').mkdir()
    self.assertEqual(store.collect({kept.artifact}), [])
    self.assertEqual(list(self.opened.models.iterdir()), [kept.path])
    self.assertEqual(list(self.opened.facts.iterdir()), [])
    self.assertEqual(list(self.opened.hf.iterdir()), [], 'xet/ too')

  def test_a_moved_workspace_keeps_every_model(self):
    """Paths are kept relative; an older server's absolute ones too."""
    store = model_store.ModelStore(self.opened)
    uploaded = _upload(store, 'a.litertlm', b'uploaded')
    facts = self.opened.facts / ('c' * 64)
    facts.mkdir()
    (facts / model_store.DESCRIBED).write_text('{}')
    imported = store.adopt(
        {
            'sha256': 'c' * 64,
            'name': 'imported',
            'size': 1,
            'runtime': model_store.LITERT_LM,
            'source': None,
            'sampler_defaults': None,
        },
        facts,
    )
    kept = json.loads((self.opened.root / 'models.json').read_text())
    self.assertEqual(
        kept['models'][uploaded.artifact]['path'],
        f'models/{uploaded.sha256}.litertlm',
    )
    self.assertEqual(
        kept['models'][imported.artifact]['facts'], f'facts/{"c" * 64}'
    )
    # As an older server wrote them: absolute, of where it was then.
    old_root = self.opened.root
    for record in kept['models'].values():
      for key in ('path', 'facts'):
        if key in record:
          record[key] = str(old_root / record[key])
    (old_root / 'models.json').write_text(json.dumps(kept))
    self.opened.close()
    moved = self.tmp_path / 'moved'
    old_root.rename(moved)
    reopened = workspace.Workspace(moved)
    self.addCleanup(reopened.close)
    again = model_store.ModelStore(reopened)
    self.assertEqual(again.collect({uploaded.artifact, imported.artifact}), [])
    self.assertTrue(again.get(uploaded.artifact).present())
    self.assertEqual(
        again.get(imported.artifact).facts, reopened.facts / ('c' * 64)
    )
    self.assertTrue(
        (reopened.facts / ('c' * 64) / model_store.DESCRIBED).is_file()
    )
    self.assertEqual(
        list(reopened.models.iterdir()),
        [reopened.models / f'{uploaded.sha256}.litertlm'],
    )

  def test_a_path_in_the_workspace_but_no_model_directory_is_dropped(self):
    """Only models/ and hf/ hold model files; nothing else is deleted."""
    other = self.opened.root / 'captures' / 'x.litertlm'
    other.write_bytes(b'x')
    (self.opened.root / 'models.json').write_text(
        json.dumps({
            'models': {
                'model-a': {
                    'sha256': 'a' * 64,
                    'name': 'x.litertlm',
                    'size': 1,
                    'path': str(other),
                }
            }
        })
    )
    store = model_store.ModelStore(self.opened)
    self.assertIsNone(store.get('model-a').path)
    store.collect(set())
    self.assertTrue(other.is_file())

  def test_a_pytorch_snapshot_goes_but_a_file_beside_it_stays(self):
    """One commit's snapshot: the LiteRT-LM file another model is stays."""
    snapshot = self._hf_directory()
    store = model_store.ModelStore(self.opened)
    blob = snapshot.parents[1] / 'blobs' / 'litertlm'
    blob.write_bytes(b'litertlm bytes')
    (snapshot / 'm.litertlm').symlink_to(blob)
    litert = store.add_download(snapshot / 'm.litertlm')
    pytorch = store.add_download_directory(snapshot)
    store.collect({litert.artifact}, {pytorch.artifact})
    self.assertTrue(litert.present())
    self.assertFalse((snapshot / 'config.json').exists())
    self.assertEqual(
        sorted(p.name for p in (snapshot.parents[1] / 'blobs').iterdir()),
        ['litertlm'],
    )


if __name__ == '__main__':
  test_main.main()
