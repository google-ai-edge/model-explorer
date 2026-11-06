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

"""Tests of the models named on Hugging Face, over a fake hub and cache."""

import asyncio
import hashlib
import json
import os
import pathlib
import threading
import time
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import httpx
from huggingface_hub import errors as hub_errors
from huggingface_hub import file_download
from model_debugger.server.session import hf_models
from model_debugger.server.session import jobs
from model_debugger.server.session import model_store
from model_debugger.server.session import workspace
from model_debugger.server.testing import test_main

_COMMIT = 'a' * 40
_GEMMA = 'litert-community/gemma-4-E2B-it-litert-lm'


def _http_error(cls, status: int):
  request = httpx.Request('GET', 'https://huggingface.co/api/x')
  return cls('refused', response=httpx.Response(status, request=request))


class FakeHub:
  """huggingface_hub's downloads over a cache in a directory.

  Attributes:
    cache: The cache directory path.
    repos: By repository, each file's bytes at _COMMIT.
    fetched: The files written into the cache, as (repo, file).
    offline: Whether the hub cannot be reached.
    error: What every call raises, if set.
    hold: If set, a download waits for it after its first chunk.
    writing: Set once a download wrote its first chunk.
    get_paths_info: Handler for get_paths_info calls.
  """

  def __init__(self, cache: pathlib.Path) -> None:
    """A hub of no repositories, its cache in `cache`."""
    self.cache = cache
    self.repos: dict[str, dict[str, bytes]] = {}
    self.fetched: list[tuple[str, str]] = []
    self.offline = False
    self.error: BaseException | None = None
    self.hold: threading.Event | None = None
    self.writing = threading.Event()

  def _snapshot(self, repo: str) -> pathlib.Path:
    folder = 'models--' + repo.replace('/', '--')
    return self.cache / folder / 'snapshots' / _COMMIT

  def _info(self, repo: str, name: str) -> file_download.DryRunFileInfo:
    path = self._snapshot(repo) / name
    return file_download.DryRunFileInfo(
        commit_hash=_COMMIT,
        file_size=len(self.repos[repo][name]),
        filename=name,
        local_path=str(path),
        is_cached=path.is_file(),
        will_download=not path.is_file(),
    )

  def _write(self, repo: str, name: str, tqdm_class=None) -> str:
    path = self._snapshot(repo) / name
    if not path.is_file():
      if self.offline:
        raise hub_errors.LocalEntryNotFoundError('offline')
      data = self.repos[repo][name]
      blobs = path.parents[2] / 'blobs'
      blobs.mkdir(parents=True, exist_ok=True)
      digest = hashlib.sha256(data).hexdigest()
      incomplete = blobs / f'{digest}.incomplete'
      incomplete.write_bytes(data[: len(data) // 2])
      self.writing.set()
      if self.hold is not None:
        assert self.hold.wait(10)
      if tqdm_class is not None:
        # huggingface_hub's bar after the chunk: it may stop here.
        tqdm_class(total=len(data), unit='B', disable=True).update(len(data))
      time.sleep(0.05)
      incomplete.unlink()
      (blobs / digest).write_bytes(data)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.symlink_to(blobs / digest)
      self.fetched.append((repo, name))
    return str(path)

  def _check(self, repo: str) -> None:
    if self.error is not None:
      raise self.error
    if repo not in self.repos:
      raise _http_error(hub_errors.RepositoryNotFoundError, 404)

  def hf_hub_download(
      self,
      repo_id,
      filename,
      revision,
      cache_dir,
      dry_run=False,
      tqdm_class=None,
  ):
    """As huggingface_hub.hf_hub_download, into the cache only."""
    del revision  # Unused.
    assert cache_dir == self.cache, cache_dir
    self._check(repo_id)
    if filename not in self.repos[repo_id]:
      raise _http_error(hub_errors.RemoteEntryNotFoundError, 404)
    if dry_run:
      if self.offline:
        raise hub_errors.DryRunError('offline')
      return self._info(repo_id, filename)
    return self._write(repo_id, filename, tqdm_class)

  def snapshot_download(
      self,
      repo_id,
      revision,
      allow_patterns,
      cache_dir,
      dry_run=False,
      tqdm_class=None,
  ):
    """As huggingface_hub.snapshot_download, into the cache only."""
    del revision  # Unused.
    assert cache_dir == self.cache, cache_dir
    self._check(repo_id)
    if dry_run and self.offline:
      raise hub_errors.DryRunError('offline')
    names = [
        name
        for name in self.repos[repo_id]
        if any(pathlib.PurePath(name).match(p) for p in allow_patterns)
    ]
    if dry_run:
      return [self._info(repo_id, name) for name in names]
    for name in names:
      self._write(repo_id, name, tqdm_class)
    return str(self._snapshot(repo_id))

  def list_repo_files(self, repo_id, revision):
    """As huggingface_hub.list_repo_files."""
    del revision  # Unused.
    self._check(repo_id)
    return list(self.repos[repo_id])

  def get_paths_info(self, repo_id, paths, revision):
    """As huggingface_hub.get_paths_info: each file's LFS SHA-256."""
    del revision  # Unused.
    self._check(repo_id)
    if self.offline:
      raise hub_errors.OfflineModeIsEnabled('offline')
    return [
        types.SimpleNamespace(
            path=path,
            lfs=types.SimpleNamespace(
                sha256=hashlib.sha256(self.repos[repo_id][path]).hexdigest()
            ),
        )
        for path in paths
    ]


class ParseTest(parameterized.TestCase):
  """Tests of what the editor names."""

  @parameterized.named_parameters(
      (
          'blob_link',
          f'https://huggingface.co/{_GEMMA}/blob/main/gemma-4-E2B-it.litertlm',
          '',
          (_GEMMA, 'gemma-4-E2B-it.litertlm', 'main'),
      ),
      (
          'resolve_link_at_a_commit',
          f'https://huggingface.co/{_GEMMA}/resolve/{_COMMIT}/a/b.litertlm',
          '',
          (_GEMMA, 'a/b.litertlm', _COMMIT),
      ),
      ('repo_and_file', _GEMMA, 'm.litertlm', (_GEMMA, 'm.litertlm', 'main')),
      ('repo_alone', _GEMMA, '', (_GEMMA, '', 'main')),
      (
          'tree_link',
          f'https://huggingface.co/{_GEMMA}/tree/v1',
          '',
          (_GEMMA, '', 'v1'),
      ),
  )
  def test_a_litert_lm_model_is_one_file(self, link, file, expected):
    """A link to the file, or the repository and the file."""
    request = hf_models.parse(model_store.LITERT_LM, link, file)
    self.assertEqual((request.repo, request.file, request.revision), expected)

  def test_a_pytorch_model_is_a_repository(self):
    """A link to the repository, at a revision."""
    request = hf_models.parse(
        model_store.PYTORCH,
        'https://huggingface.co/google/gemma-4-E2B-it',
        '',
        _COMMIT,
    )
    self.assertEqual(
        (request.repo, request.file, request.revision),
        ('google/gemma-4-E2B-it', '', _COMMIT),
    )

  @parameterized.named_parameters(
      ('not_litertlm', 'LiteRT-LM', _GEMMA, 'config.json', 'not a LiteRT'),
      ('a_file_for_pytorch', 'PyTorch', _GEMMA, 'm.litertlm', 'whole repo'),
      ('another_host', 'LiteRT-LM', 'https://example.com/a/b', '', 'one of'),
      ('http', 'LiteRT-LM', 'http://huggingface.co/a/b', '', 'one of'),
      ('no_owner', 'PyTorch', 'gemma', '', 'owner'),
      ('traversal', 'LiteRT-LM', _GEMMA, '../x.litertlm', 'not a file'),
      ('runtime', 'TFLite', _GEMMA, '', 'LiteRT-LM or PyTorch'),
      (
          'odd_link',
          'PyTorch',
          f'https://huggingface.co/{_GEMMA}/discussions',
          '',
          'Link to the repository',
      ),
  )
  def test_refused(self, runtime, link, file, message):
    """Each says why."""
    with self.assertRaisesRegex(ValueError, message):
      hf_models.parse(runtime, link, file)


class HfModelsTest(parameterized.TestCase):
  """Tests of the download jobs."""

  def setUp(self):
    """A workspace's model store, the jobs and a fake hub."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.opened = workspace.Workspace(root / 'ws')
    self.addCleanup(self.opened.close)
    self.models = model_store.ModelStore(self.opened)
    self.jobs = jobs.Jobs()
    self.hub = FakeHub(self.opened.hf)
    self.hub.repos[_GEMMA] = {
        'gemma-4-E2B-it.litertlm': b'litertlm' * 1000,
        'README.md': b'readme',
    }
    self.hub.repos['google/gemma-4-E2B-it'] = {
        'config.json': b'{}',
        'model.safetensors': b'weights' * 100,
        'tokenizer.json': b'{}',
        'chat_template.jinja': b'{{ x }}',
        'original/consolidated.pth': b'not fetched',
    }
    self.hf = hf_models.HfModels(
        self.models, self.jobs, self.opened.hf, self.hub
    )

  def _wait(self, job: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
      job = self.jobs.get(job['id'])
      if job['status'] in jobs.TERMINAL:
        return job
      time.sleep(0.01)
    raise AssertionError(f'job {job["id"]} did not end')

  def _progress(self, job: dict[str, Any]) -> list[dict[str, Any]]:
    events = []

    async def read():
      async for event in self.jobs.events(job['id'], 0):
        if event['type'] == 'progress':
          events.append(event)

    asyncio.run(read())
    return events

  def test_a_litert_lm_file_is_fetched_once_and_recorded(self):
    """Fetched into the workspace, hashed, recorded with its source."""
    job = self._wait(
        self.hf.start({
            'runtime': 'LiteRT-LM',
            'link': (
                f'https://huggingface.co/{_GEMMA}/blob/main/'
                'gemma-4-E2B-it.litertlm'
            ),
        })
    )

    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['operation'], 'download')
    data = self.hub.repos[_GEMMA]['gemma-4-E2B-it.litertlm']
    model = self.models.get(job['model']['artifact'])
    self.assertEqual(model.sha256, hashlib.sha256(data).hexdigest())
    self.assertEqual(model.name, 'gemma-4-E2B-it.litertlm')
    self.assertEqual(model.runtime, model_store.LITERT_LM)
    self.assertTrue(model.present())
    self.assertTrue(model.path.is_relative_to(self.opened.hf))
    source = {
        'repo': _GEMMA,
        'file': 'gemma-4-E2B-it.litertlm',
        'revision': 'main',
        'commit': _COMMIT,
    }
    self.assertEqual(model.source, source)
    self.assertEqual(job['model']['source'], source)
    self.assertEqual(
        job['progress'],
        'Downloaded gemma-4-E2B-it.litertlm · '
        f'{hf_models.size_text(len(data))}',
    )
    self.assertEqual(self.hub.fetched, [(_GEMMA, 'gemma-4-E2B-it.litertlm')])
    progress = self._progress(job)
    self.assertTrue(
        any(e['message'].startswith('Downloading ') for e in progress)
    )
    # Named again, it is on the server: nothing is fetched.
    again = self._wait(
        self.hf.start({
            'runtime': 'LiteRT-LM',
            'link': _GEMMA,
            'file': 'gemma-4-E2B-it.litertlm',
        })
    )
    self.assertEqual(again['model']['artifact'], model.artifact)
    self.assertLen(self.hub.fetched, 1)
    self.assertIn(
        f'{_GEMMA} is on the Server',
        [e['message'] for e in self._progress(again)],
    )
    # Its source stays when it is recorded again.
    self.assertEqual(self.models.add_download(model.path).source, source)

  def test_a_repository_of_one_litert_lm_file_needs_no_file_named(self):
    """Of several, the one to use must be named."""
    job = self._wait(self.hf.start({'runtime': 'LiteRT-LM', 'link': _GEMMA}))
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['model']['source']['file'], 'gemma-4-E2B-it.litertlm')
    self.hub.repos[_GEMMA]['other.litertlm'] = b'x'
    job = self._wait(self.hf.start({'runtime': 'LiteRT-LM', 'link': _GEMMA}))
    self.assertEqual(job['status'], 'failed')
    self.assertIn('has 2 .litertlm files', job['error'])

  def test_a_pytorch_repository_is_recorded_as_a_directory(self):
    """What the runner reads is fetched; the id is the directory's."""
    job = self._wait(
        self.hf.start({
            'runtime': 'PyTorch',
            'link': 'https://huggingface.co/google/gemma-4-E2B-it',
            'revision': _COMMIT,
        })
    )

    self.assertEqual(job['status'], 'completed', job['error'])
    model = self.models.get(job['model']['artifact'])
    self.assertEqual(model.runtime, model_store.PYTORCH)
    self.assertEqual(model.name, 'google/gemma-4-E2B-it')
    self.assertNotIn(
        ('google/gemma-4-E2B-it', 'original/consolidated.pth'),
        self.hub.fetched,
    )
    self.assertEqual(model.source['commit'], _COMMIT)
    # The same directory recorded again is the same model.
    self.assertEqual(
        self.models.add_download_directory(model.path).artifact,
        model.artifact,
    )

  def test_a_repository_without_a_config_is_refused(self):
    """The PyTorch runner reads config.json."""
    del self.hub.repos['google/gemma-4-E2B-it']['config.json']
    job = self._wait(
        self.hf.start({'runtime': 'PyTorch', 'link': 'google/gemma-4-E2B-it'})
    )
    self.assertEqual(job['status'], 'failed')
    self.assertIn('has no config.json', job['error'])
    self.assertEqual(self.models.list(), [])

  @parameterized.named_parameters(
      (
          'gated',
          lambda: _http_error(hub_errors.GatedRepoError, 403),
          'is gated',
      ),
      (
          'missing_or_private',
          lambda: _http_error(hub_errors.RepositoryNotFoundError, 401),
          'does not exist, or it is private',
      ),
      (
          'unauthorized',
          lambda: _http_error(hub_errors.HfHubHTTPError, 401),
          r'refused .* \(401\)',
      ),
      (
          'no_revision',
          lambda: _http_error(hub_errors.RevisionNotFoundError, 404),
          'has no revision main',
      ),
  )
  def test_what_the_hub_refuses_says_why(self, error, message):
    """401, 403 and 404 say what to do."""
    self.hub.error = error()
    job = self._wait(
        self.hf.start(
            {'runtime': 'LiteRT-LM', 'link': _GEMMA, 'file': 'x.litertlm'}
        )
    )
    self.assertEqual(job['status'], 'failed')
    self.assertRegex(job['error'], message)

  def test_offline_a_cached_file_resolves_and_another_says_why(self):
    """Without the network the cache alone answers."""
    payload = {
        'runtime': 'LiteRT-LM',
        'link': _GEMMA,
        'file': 'gemma-4-E2B-it.litertlm',
    }
    self._wait(self.hf.start(payload))
    self.hub.offline = True

    cached = self._wait(self.hf.start(payload))
    self.hub.repos[_GEMMA]['new.litertlm'] = b'new'
    missing = self._wait(self.hf.start({**payload, 'file': 'new.litertlm'}))

    self.assertEqual(cached['status'], 'completed', cached['error'])
    self.assertEqual(missing['status'], 'failed')
    self.assertIn('cannot be reached', missing['error'])

  def test_the_editor_is_told_whether_the_server_has_it_and_its_size(self):
    """check(): to download, then on the server, by the hub's plan."""
    payload = {
        'runtime': 'LiteRT-LM',
        'link': _GEMMA,
        'file': 'gemma-4-E2B-it.litertlm',
    }
    size = len(self.hub.repos[_GEMMA]['gemma-4-E2B-it.litertlm'])
    self.assertEqual(
        self.hf.check(payload),
        {
            'on_server': False,
            'size': size,
            'text': 'Downloads to the Server · ' + hf_models.size_text(size),
        },
    )
    self._wait(self.hf.start(payload))
    self.assertEqual(
        self.hf.check(payload)['text'],
        f'On the Server · {hf_models.size_text(size)}',
    )
    # A whole repository: what the runner reads of it.
    pytorch = {'runtime': 'PyTorch', 'link': 'google/gemma-4-E2B-it'}
    checked = self.hf.check(pytorch)
    self.assertFalse(checked['on_server'])
    self.assertEqual(checked['size'], 700 + 2 + 2 + 7)
    # Without the hub, the model store answers; no size for the rest.
    self.hub.offline = True
    self.assertEqual(self.hf.check(payload)['text'], checked_text(size))
    self.assertEqual(
        self.hf.check(pytorch),
        {
            'on_server': False,
            'size': None,
            'text': 'Downloads to the Server',
        },
    )
    self.hub.offline = False
    self.hub.error = _http_error(hub_errors.GatedRepoError, 403)
    with self.assertRaisesRegex(ValueError, 'is gated'):
      self.hf.check(payload)

  def _uploaded(self, data: bytes) -> model_store.Model:
    """A model uploaded to the server of `data`."""

    async def chunks():
      yield data

    return asyncio.run(self.models.upload('mine.litertlm', len(data), chunks()))

  def test_a_file_the_server_has_by_its_content_is_not_downloaded(self):
    """An upload of the same bytes is the model; offline, it downloads."""
    data = self.hub.repos[_GEMMA]['gemma-4-E2B-it.litertlm']
    uploaded = self._uploaded(data)
    payload = {
        'runtime': 'LiteRT-LM',
        'link': _GEMMA,
        'file': 'gemma-4-E2B-it.litertlm',
    }
    self.assertEqual(
        self.hf.check(payload),
        {
            'on_server': True,
            'size': len(data),
            'text': checked_text(len(data)),
        },
    )
    job = self._wait(self.hf.start(payload))
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['model']['artifact'], uploaded.artifact)
    self.assertEqual(self.hub.fetched, [])
    self.assertEqual(self.models.list(), [uploaded])
    # Without the hub's answer, the file is fetched as ever.
    self.hub.get_paths_info = None
    self.assertFalse(self.hf.check(payload)['on_server'])
    job = self._wait(self.hf.start(payload))
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(self.hub.fetched, [(_GEMMA, 'gemma-4-E2B-it.litertlm')])

  def test_a_cancelled_download_stops_and_leaves_nothing(self):
    """Its job ends at once; what it wrote is deleted; nothing recorded."""
    self.hub.hold = threading.Event()
    self.addCleanup(self.hub.hold.set)
    job = self.hf.start({'runtime': 'PyTorch', 'link': 'google/gemma-4-E2B-it'})
    self.assertTrue(self.hub.writing.wait(10))
    blobs = self.opened.hf / 'models--google--gemma-4-E2B-it' / 'blobs'
    self.assertLen(list(blobs.glob('*.incomplete')), 1)
    # A link a download made to a blob that is not there.
    dangling = blobs.parent / 'snapshots' / _COMMIT / 'left.json'
    dangling.parent.mkdir(parents=True, exist_ok=True)
    dangling.symlink_to(blobs / 'gone')

    with self.assertLogs(level='INFO') as logs:
      cancelled = self.hf.cancel(job['id'])
    self.assertEqual(
        [record.getMessage() for record in logs.records],
        [f'Download {job["id"]} cancelled'],
    )

    self.assertEqual(
        (cancelled['status'], cancelled['error'], cancelled['note']),
        ('cancelled', hf_models.CANCELLED, ''),
    )
    self.hub.hold.set()
    deadline = time.monotonic() + 10
    while list(blobs.glob('*')) and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertEqual(list(blobs.glob('*')), [])
    self.assertFalse(dangling.is_symlink())
    self.assertEqual(self.hub.fetched, [])
    self.assertEqual(self.models.list(), [])
    ended = self.jobs.get(job['id'])
    self.assertEqual(ended['status'], 'cancelled')
    self.assertNotIn('model', ended)
    # Ended, it is answered as it is.
    self.assertEqual(self.hf.cancel(job['id'])['status'], 'cancelled')

  def test_what_is_not_valid_is_refused_before_any_job(self):
    """parse() refuses it as the request comes."""
    with self.assertRaisesRegex(ValueError, 'not a LiteRT-LM model'):
      self.hf.start({'runtime': 'LiteRT-LM', 'link': _GEMMA, 'file': 'a.bin'})


class ChildTransferTest(absltest.TestCase):
  """Tests of a download's child process, with stand-ins for the hub."""

  def test_what_it_returns_and_raises_comes_back(self):
    """Its result, and its error as it was raised."""
    done = hf_models._ChildTransfer('os.path:join', ['a', 'b'], {}, {})
    self.assertEqual(done.result(), os.path.join('a', 'b'))
    failed = hf_models._ChildTransfer('json:loads', ['{'], {}, {})
    with self.assertRaises(json.JSONDecodeError):
      failed.result()

  def test_its_environment_is_its_own(self):
    """What it is given is set in it alone."""
    transfer = hf_models._ChildTransfer(
        'os:getenv', ['HF_TEST_ONLY'], {}, {'HF_TEST_ONLY': 'child'}
    )
    self.assertEqual(transfer.result(), 'child')
    self.assertIsNone(os.environ.get('HF_TEST_ONLY'))

  def test_stop_ends_it_at_once(self):
    """A transfer that would run a minute is gone within seconds."""
    transfer = hf_models._ChildTransfer('time:sleep', [60], {}, {})
    transfer.wait(0.2)
    self.assertTrue(transfer.alive())
    began = time.monotonic()
    transfer.stop()
    self.assertLess(time.monotonic() - began, 6)
    self.assertFalse(transfer.alive())
    with self.assertRaisesRegex(RuntimeError, 'download stopped'):
      transfer.result()


def checked_text(size: int) -> str:
  """What check() says of a model on the server of `size` bytes."""
  return f'On the Server · {hf_models.size_text(size)}'


if __name__ == '__main__':
  test_main.main()
