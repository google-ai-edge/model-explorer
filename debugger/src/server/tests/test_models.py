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
import os

from model_debugger_server import errors, models, workspace
import pytest


@pytest.fixture
def opened(tmp_path):
  """An open workspace, closed after the test."""
  value = workspace.Workspace(tmp_path / 'ws')
  yield value
  value.close()


async def _chunks(data, size=7):
  for start in range(0, len(data), size):
    yield data[start : start + size]


def _upload(store, name, data, size=None):
  return asyncio.run(
      store.upload(name, len(data) if size is None else size, _chunks(data))
  )


def test_register_by_reference(opened, tmp_path):
  """Register by reference."""
  path = tmp_path / 'm.litertlm'
  path.write_bytes(b'model bytes')
  model = models.ModelStore(opened).register(path)
  sha256 = hashlib.sha256(b'model bytes').hexdigest()
  assert model == models.Model(
      artifact='model-' + sha256[:24],
      sha256=sha256,
      name='m.litertlm',
      size=11,
      path=path.resolve(),
  )
  # Kept where it is, and known after a restart.
  assert list(opened.models.iterdir()) == []
  assert models.ModelStore(opened).get(model.artifact) == model


def test_register_again_does_not_hash_again(opened, tmp_path, monkeypatch):
  """Register again does not hash again."""
  path = tmp_path / 'm.litertlm'
  path.write_bytes(b'model bytes')
  first = models.ModelStore(opened).register(path)
  monkeypatch.setattr(models, '_hash_file', pytest.fail)
  assert models.ModelStore(opened).register(path) == first


def test_register_a_changed_file_replaces_it(opened, tmp_path):
  """Register a changed file replaces it."""
  path = tmp_path / 'm.litertlm'
  path.write_bytes(b'one')
  store = models.ModelStore(opened)
  first = store.register(path)
  path.write_bytes(b'two!')
  os.utime(path, ns=(1, 1))
  second = store.register(path)
  assert second.sha256 == hashlib.sha256(b'two!').hexdigest()
  assert store.list() == [second]
  with pytest.raises(errors.NotFound):
    store.get(first.artifact)


def test_register_refuses(opened, tmp_path):
  """Register refuses."""
  store = models.ModelStore(opened)
  with pytest.raises(ValueError, match='not a file'):
    store.register(tmp_path / 'missing.litertlm')
  (tmp_path / 'm.tflite').write_bytes(b'x')
  with pytest.raises(ValueError, match='.litertlm'):
    store.register(tmp_path / 'm.tflite')


def test_upload_stores_by_sha256(opened):
  """Upload stores by sha256."""
  data = bytes(range(256)) * 9000  # Spans more than one write.
  store = models.ModelStore(opened)
  model = _upload(store, 'gemma.litertlm', data)
  sha256 = hashlib.sha256(data).hexdigest()
  assert model.sha256 == sha256
  assert model.name == 'gemma.litertlm'
  assert model.size == len(data)
  assert model.path == opened.models / f'{sha256}.litertlm'
  assert model.path.read_bytes() == data
  assert list(opened.uploads.iterdir()) == []
  assert models.ModelStore(opened).list() == [model]


def test_same_bytes_twice_is_one_file(opened, tmp_path):
  """Same bytes twice is one file."""
  store = models.ModelStore(opened)
  first = _upload(store, 'a.litertlm', b'same')
  assert _upload(store, 'b.litertlm', b'same') == first
  assert len(list(opened.models.iterdir())) == 1
  path = tmp_path / 'c.litertlm'
  path.write_bytes(b'other')
  registered = store.register(path)
  assert _upload(store, 'd.litertlm', b'other') == registered


def test_upload_size_must_match(opened):
  """Upload size must match."""
  store = models.ModelStore(opened)
  with pytest.raises(ValueError, match='ended after 3 of 5'):
    _upload(store, 'a.litertlm', b'abc', size=5)
  with pytest.raises(ValueError, match='more than 2'):
    _upload(store, 'a.litertlm', b'abc', size=2)
  assert list(opened.uploads.iterdir()) == []
  assert list(opened.models.iterdir()) == []
  assert store.list() == []


@pytest.mark.parametrize(
    'name',
    ['', '../x.litertlm', 'a/b.litertlm', 'x.bin', '.litertlm', 'a\n.litertlm'],
)
def test_upload_refuses_names(opened, name):
  """Upload refuses names."""
  with pytest.raises(ValueError):
    _upload(models.ModelStore(opened), name, b'abc')


def test_upload_refuses_sizes(opened):
  """Upload refuses sizes."""
  store = models.ModelStore(opened)
  with pytest.raises(ValueError, match='non-empty'):
    _upload(store, 'a.litertlm', b'', size=0)
  with pytest.raises(ValueError, match='non-empty'):
    _upload(store, 'a.litertlm', b'', size=models.UPLOAD_LIMIT_BYTES + 1)


def test_get_unknown(opened):
  """Get unknown."""
  with pytest.raises(errors.NotFound):
    models.ModelStore(opened).get('model-0')
