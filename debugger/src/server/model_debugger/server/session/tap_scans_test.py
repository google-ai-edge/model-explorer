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

"""Tests of the scans of models, of a file or kept with its facts."""

import pathlib

from absl.testing import absltest
from model_debugger.server.session import model_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

_STEMS = {name: 'layer' for name in tap_fixtures.SIGNATURES}
_GROUPS = {'decode': {3: 0}}


class TapScansTest(absltest.TestCase):
  """A model's scans travel as files; a store without the file reads them."""

  def setUp(self):
    """A model file registered in one workspace, scanned there."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.opened = workspace.Workspace(root / 'here')
    self.addCleanup(self.opened.close)
    path = self.opened.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = model_store.ModelStore(self.opened).add_download(path)
    self.scans = tap_scans.TapScans(model_store.ModelStore(self.opened))
    self.addCleanup(self.scans.close)
    self.elsewhere = workspace.Workspace(root / 'elsewhere')
    self.addCleanup(self.elsewhere.close)

  def _imported(self, files: dict[str, bytes]) -> model_store.Model:
    """The model in the other workspace, with these files as facts."""
    facts = self.elsewhere.facts / self.model.sha256
    facts.mkdir()
    for name, data in files.items():
      (facts / name).write_bytes(data)
    return model_store.ModelStore(self.elsewhere).adopt(
        {
            'sha256': self.model.sha256,
            'name': self.model.name,
            'size': self.model.size,
            'runtime': model_store.LITERT_LM,
            'source': None,
            'sampler_defaults': None,
        },
        facts,
    )

  def test_a_model_without_its_file_reads_its_kept_scans(self):
    """The plain scan and the one by the layers it was made by."""
    files = self.scans.stored_files(self.model, _STEMS, _GROUPS)
    self.assertEqual(set(files), {tap_scans.SCAN, tap_scans.LAYERED_SCAN})
    model = self._imported(files)
    scans = tap_scans.TapScans(model_store.ModelStore(self.elsewhere))
    self.addCleanup(scans.close)
    self.assertEqual(scans.kept(model), self.scans.scan(self.model))
    self.assertEqual(scans.scan(model), self.scans.scan(self.model))
    self.assertEqual(
        scans.scan(model, _STEMS, _GROUPS),
        self.scans.scan(self.model, _STEMS, _GROUPS),
    )
    with self.assertRaisesRegex(ValueError, 'not those it was scanned by'):
      scans.scan(model, {'decode': 'block'})

  def test_a_model_with_neither_says_it_is_not_here(self):
    """No file, no kept scan: nothing to scan, and kept() says none."""
    model = self._imported({})
    scans = tap_scans.TapScans(model_store.ModelStore(self.elsewhere))
    self.addCleanup(scans.close)
    self.assertIsNone(scans.kept(model))
    with self.assertRaisesRegex(ValueError, 'hold no scan.json'):
      scans.scan(model)
    with self.assertRaisesRegex(ValueError, 'hold no scan-layers'):
      scans.scan(model, _STEMS)


if __name__ == '__main__':
  test_main.main()
