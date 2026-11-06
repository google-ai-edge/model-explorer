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

"""Tests of the generator: writing and checking the skeleton files."""

import pathlib
import shutil

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import generate
from model_debugger.server.testing import test_main


class GenerateTest(absltest.TestCase):
  """Tests of run(), on a copy of the package's data."""

  def setUp(self):
    """A copy of the data, as the package directory run() is given."""
    super().setUp()
    self.directory = pathlib.Path(self.create_tempdir().full_path)
    (known,) = catalog.architectures()
    shutil.copytree(known.directory, self.directory / 'data' / known.id)
    self.data = self.directory / 'data' / known.id

  def test_the_files_are_what_the_configs_make(self):
    """--check of the package's own files finds nothing wrong."""
    self.assertEqual(generate.run(self.directory, check=True), [])

  def test_check_finds_a_changed_skeleton_and_writing_restores_it(self):
    """--check fails and writes nothing; without, the file is rewritten."""
    path = self.data / 'gemma-4-E2B-it.skeleton.json'
    original = path.read_bytes()
    path.write_bytes(original + b'\n')
    (problem,) = generate.run(self.directory, check=True)
    self.assertIn('is not what its config makes', problem)
    self.assertEqual(path.read_bytes(), original + b'\n')
    self.assertEqual(generate.run(self.directory, check=False), [])
    self.assertEqual(path.read_bytes(), original)

  def test_a_config_that_is_not_the_pinned_one_is_refused(self):
    """A pinned config is checked by its SHA-256 before it is read."""
    path = self.data / 'gemma-4-E4B-it.config.json'
    path.write_bytes(path.read_bytes().replace(b'"gelu', b'"silu'))
    (problem,) = generate.run(self.directory, check=False)
    self.assertIn('the one its entry pins', problem)


if __name__ == '__main__':
  test_main.main()
