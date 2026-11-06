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

"""The model id: the manifest's lines, links followed, hidden files left out."""

import hashlib

from model_debugger.runner_pytorch.service import model_id
from model_debugger.runner_pytorch.testing import test_main


def test_the_manifest_lists_files_by_path_with_size_and_sha256(tmp_path):
  """The manifest lists files by path with size and sha256."""
  (tmp_path / 'blobs').mkdir()
  (tmp_path / 'blobs' / 'x').write_bytes(b'weights')
  (tmp_path / 'b.json').write_text('{}')
  (tmp_path / 'a.safetensors').symlink_to(tmp_path / 'blobs' / 'x')
  (tmp_path / '.cache').mkdir()
  (tmp_path / '.cache' / 'lock').write_text('')
  (tmp_path / '.hidden').write_text('')
  weights = hashlib.sha256(b'weights').hexdigest()
  braces = hashlib.sha256(b'{}').hexdigest()
  expected = (
      f'a.safetensors\t7\t{weights}\n'
      f'b.json\t2\t{braces}\n'
      f'blobs/x\t7\t{weights}\n'
  )
  cache = tmp_path.parent / f'{tmp_path.name}-hashes.json'
  assert model_id.manifest(tmp_path, cache) == expected
  assert cache.is_file()
  assert model_id.model_id(tmp_path, cache) == (
      hashlib.sha256(expected.encode()).hexdigest()
  )


if __name__ == '__main__':
  test_main.main()
