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

"""Tests of the architectures the server knows."""

import copy
import hashlib
import json
import pathlib
from unittest import mock

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import skeleton
from model_debugger.server.testing import test_main


class CatalogTest(absltest.TestCase):
  """Tests of architectures() and generate()."""

  def setUp(self):
    """Gemma 4, the one architecture known."""
    super().setUp()
    (self.gemma4,) = catalog.architectures()

  def test_gemma4_has_a_pinned_config_per_skeleton(self):
    """Each skeleton's config is pinned beside it, by SHA-256."""
    self.assertEqual((self.gemma4.id, self.gemma4.name), ('gemma4', 'Gemma 4'))
    self.assertEqual(self.gemma4.spec['model_type'], 'gemma4_text')
    entries = self.gemma4.spec['skeletons']
    self.assertEqual(
        [e['id'] for e in entries], ['gemma-4-E2B-it', 'gemma-4-E4B-it']
    )
    for entry in entries:
      pinned = self.gemma4.directory / entry['pinned']['file']
      self.assertEqual(
          hashlib.sha256(pinned.read_bytes()).hexdigest(),
          entry['pinned']['sha256'],
      )
      self.assertEqual(
          entry['pinned']['source'],
          f'https://huggingface.co/{entry["pinned"]["model"]}/blob/'
          f'{entry["pinned"]["revision"]}/config.json',
      )
    with self.assertRaisesRegex(ValueError, 'has no skeleton'):
      self.gemma4.entry('gemma-4-E9B-it')

  def test_a_skeleton_comes_from_the_generator_and_its_pinned_config(self):
    """provenance(): this package's files, by their repository paths."""
    package = pathlib.Path(catalog.__file__).parent
    self.assertTrue(
        package.as_posix().endswith(
            catalog.REPOSITORY_PACKAGE.removeprefix('src/server/')
        )
    )

    def here(path: str) -> pathlib.Path:
      prefix = catalog.REPOSITORY_PACKAGE + '/'
      self.assertTrue(path.startswith(prefix), path)
      return package / path.removeprefix(prefix)

    for entry in self.gemma4.spec['skeletons']:
      found = self.gemma4.provenance(entry['id'])
      generator = found['generator']
      self.assertTrue(here(generator['path']).is_file())
      self.assertEqual(here(generator['path']).name, 'generate.py')
      self.assertEqual(
          here(generator['definition']).name,
          catalog._DEFINITIONS['gemma4'].__module__.rsplit('.')[-1] + '.py',
      )
      self.assertTrue(here(generator['definition']).is_file())
      self.assertEqual(generator['identity'], catalog.identity())
      self.assertIn(':generate', generator['command'])
      self.assertTrue(generator['check'].endswith('-- --check'))
      # Where it was ported from is a note beside it, not its source.
      self.assertEqual(
          generator['derived_from']['path'],
          '~/model_arch_explore/mkjson.py',
      )
      config = found['config']
      self.assertEqual(
          hashlib.sha256(here(config['path']).read_bytes()).hexdigest(),
          config['sha256'],
      )
      self.assertEqual(
          {key: config[key] for key in ('model', 'revision', 'sha256')},
          {
              key: entry['pinned'][key]
              for key in ('model', 'revision', 'sha256')
          },
      )
    with self.assertRaisesRegex(ValueError, 'has no skeleton'):
      self.gemma4.provenance('gemma-4-E9B-it')

  def test_a_config_json_makes_its_skeleton(self):
    """By its text model_type; what a PyTorch runner's config would do."""
    entry = self.gemma4.entry('gemma-4-E4B-it')
    config = json.loads(
        (self.gemma4.directory / entry['pinned']['file']).read_bytes()
    )
    known, document = catalog.generate(config)
    self.assertEqual(known.id, 'gemma4')
    self.assertEqual(document, self.gemma4.skeleton('gemma-4-E4B-it'))
    # A text config alone makes the same.
    _, alone = catalog.generate(config['text_config'])
    self.assertEqual(alone, document)

  def test_an_unknown_model_type_or_a_broken_skeleton_is_refused(self):
    """Qwen3 is not defined; a skeleton that breaks the schema says so."""
    with self.assertRaisesRegex(ValueError, "model_type 'qwen3'"):
      catalog.generate({'model_type': 'qwen3'})
    entry = self.gemma4.entry('gemma-4-E2B-it')
    config = json.loads(
        (self.gemma4.directory / entry['pinned']['file']).read_bytes()
    )
    broken = copy.deepcopy(self.gemma4.skeleton(entry['id']))
    broken['layers'][3]['from']['input']['layer'] = 7
    with mock.patch.dict(catalog._DEFINITIONS, {'gemma4': lambda _: broken}):
      with self.assertRaisesRegex(ValueError, 'names layer 7'):
        catalog.generate(config)

  def test_the_files_are_named_by_a_digest(self):
    """identity(): the same files, the same digest."""
    self.assertRegex(catalog.identity(), '^[0-9a-f]{16}$')
    self.assertEqual(catalog.identity(), catalog.identity())
    self.assertEqual(skeleton.check(self.gemma4.skeleton('gemma-4-E2B-it')), [])


if __name__ == '__main__':
  test_main.main()
