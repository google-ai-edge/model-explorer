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

"""Tests of JSON files replaced whole."""

import pathlib

from absl.testing import absltest
from model_debugger.server.testing import test_main
from model_debugger.server.util import json_file


class JsonFileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_json_round_trip_leaves_no_temporary(self):
    path = self.tmp_path / 'value.json'
    json_file.write_json(path, {'a': [1, 'ü']})
    json_file.write_json(path, {'a': [2]})
    self.assertEqual(json_file.read_json(path), {'a': [2]})
    self.assertEqual([p.name for p in self.tmp_path.iterdir()], ['value.json'])

  def test_failed_write_keeps_the_old_file(self):
    path = self.tmp_path / 'value.json'
    json_file.write_json(path, {'a': 1})
    with self.assertRaises(TypeError):
      json_file.write_json(path, {'a': object()})
    self.assertEqual(json_file.read_json(path), {'a': 1})
    self.assertEqual([p.name for p in self.tmp_path.iterdir()], ['value.json'])


if __name__ == '__main__':
  test_main.main()
