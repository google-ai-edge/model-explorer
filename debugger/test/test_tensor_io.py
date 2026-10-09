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
# ==============================================================================
"""Tests for safetensors reading, slicing, and metadata decoding."""

import json
from pathlib import Path
import tempfile
import unittest

from ml_dtypes import bfloat16
from model_explorer_debugger.metrics import compare
from model_explorer_debugger.store import SessionStore
from model_explorer_debugger.tensor_io import load_tensor
import numpy as np
from safetensors.numpy import save_file


class TensorIOTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.reference = np.array([3.0, 4.0], dtype=np.float32)
    self.target = np.array([0.0, 4.0], dtype=np.float32)
    save_file(
        {'block/output.0': self.reference, 'block/output.1': self.target},
        self.root / 'capture.safetensors',
    )
    self.record = {
        'format': 'safetensors',
        'path': 'capture.safetensors',
        'key': 'block/output.0',
        'shape': [2],
        'dtype': 'float32',
    }

  def assert_error(self, record, status):
    with self.assertRaisesRegex(ValueError, '^' + status + '$'):
      load_tensor(self.root, record)

  def test_legacy_npy_and_implicit_format_are_rejected(self):
    record = {'path': 'legacy.npy', 'shape': [2], 'dtype': 'float32'}
    for indexed in (
        record,
        {**record, 'format': 'npy'},
        {key: value for key, value in self.record.items() if key != 'format'},
    ):
      with self.subTest(indexed=indexed):
        self.assert_error(indexed, 'unsupported_tensor_format')

  def test_exact_key_selects_one_tensor_in_shared_shard(self):
    reference = load_tensor(self.root, self.record)
    target = load_tensor(self.root, {**self.record, 'key': 'block/output.1'})
    np.testing.assert_array_equal(reference, self.reference)
    np.testing.assert_array_equal(target, self.target)
    self.assertEqual(reference.dtype, np.dtype('float32'))
    self.assertAlmostEqual(compare(reference, target)['CosSim']['value'], 0.8)

  def test_missing_or_invalid_key_never_falls_back_to_first_tensor(self):
    for key in ('output.0', 'block/output', 'block/output.2', ''):
      with self.subTest(key=key):
        self.assert_error({**self.record, 'key': key}, 'tensor_key_not_found')
    for key in (None, 0, ['block/output.0']):
      with self.subTest(key=key):
        self.assert_error({**self.record, 'key': key}, 'invalid_tensor_key')
    record = self.record.copy()
    del record['key']
    self.assert_error(record, 'invalid_tensor_key')

  def test_shape_and_dtype_metadata_are_validated(self):
    for field, value in [
        ('shape', [1, 2]),
        ('shape', [True, 2]),
        ('shape', [2.0]),
        ('shape', [-2]),
        ('shape', None),
        ('dtype', 'float64'),
        ('dtype', 'F32'),
        ('dtype', None),
    ]:
      with self.subTest(field=field, value=value):
        self.assert_error(
            {**self.record, field: value}, 'tensor_metadata_mismatch'
        )

  def test_scalar_and_empty_shapes_are_preserved(self):
    for tensor in (
        np.array(2.0, dtype=np.float16),
        np.empty((0, 2), dtype=np.float64),
    ):
      with self.subTest(shape=tensor.shape):
        save_file({'tensor': tensor}, self.root / 'shape.safetensors')
        loaded = load_tensor(
            self.root,
            {
                'format': 'safetensors',
                'path': 'shape.safetensors',
                'key': 'tensor',
                'shape': list(tensor.shape),
                'dtype': tensor.dtype.name,
            },
        )
        self.assertEqual(loaded.shape, tensor.shape)
        self.assertEqual(loaded.dtype, tensor.dtype)

  def test_unknown_formats_and_suffix_mismatches_are_rejected(self):
    for tensor_format in ('npy', 'npz', 'pickle', '', None, {}):
      with self.subTest(tensor_format=tensor_format):
        self.assert_error(
            {**self.record, 'format': tensor_format},
            'unsupported_tensor_format',
        )
    self.assert_error(
        {**self.record, 'path': 'capture.npy'}, 'invalid_tensor_path'
    )

  def test_paths_cannot_escape_root_including_symlinks(self):
    outside = self.root.parent / (self.root.name + '-outside.safetensors')
    outside.write_bytes((self.root / 'capture.safetensors').read_bytes())
    self.addCleanup(outside.unlink)
    (self.root / 'escape.safetensors').symlink_to(outside)
    (self.root / 'directory-link').symlink_to(
        self.root.parent, target_is_directory=True
    )
    for path in (
        '../' + outside.name,
        str(outside),
        str(self.root / 'capture.safetensors'),
        'escape.safetensors',
        'directory-link/' + outside.name,
        None,
        '',
        '\x00',
    ):
      with self.subTest(path=path):
        self.assert_error({**self.record, 'path': path}, 'invalid_tensor_path')

  def test_in_root_symlink_remains_readable(self):
    (self.root / 'alias.safetensors').symlink_to('capture.safetensors')
    np.testing.assert_array_equal(
        load_tensor(self.root, {**self.record, 'path': 'alias.safetensors'}),
        self.reference,
    )

  def test_missing_files(self):
    with self.assertRaises(FileNotFoundError):
      load_tensor(self.root, {**self.record, 'path': 'missing.safetensors'})

  def test_corrupt_files_have_stable_errors(self):
    safetensors = (self.root / 'capture.safetensors').read_bytes()
    for content in (b'', b'not a shard', safetensors[:-1]):
      with self.subTest(content=content[:10]):
        path = 'corrupt.safetensors'
        (self.root / path).write_bytes(content)
        self.assert_error({**self.record, 'path': path}, 'invalid_tensor_file')

  def test_bfloat16_preserves_raw_bits_and_metrics_use_float64(self):
    reference = np.array([3.0, 4.0], dtype=bfloat16)
    target = np.array([0.0, 4.0], dtype=bfloat16)
    # Include signed zero and NaN payloads in a separate tensor to prove the
    # storage reader does not round-trip through float32.
    special = np.array([0, 0x8000, 0x7FC1, 0xFFC2], dtype=np.uint16).view(
        bfloat16
    )
    save_file(
        {'reference': reference, 'target': target, 'special': special},
        self.root / 'bf16.safetensors',
    )
    record = {**self.record, 'path': 'bf16.safetensors', 'dtype': 'bfloat16'}
    loaded = {}
    for key, original in [
        ('reference', reference),
        ('target', target),
        ('special', special),
    ]:
      loaded[key] = load_tensor(
          self.root, {**record, 'key': key, 'shape': list(original.shape)}
      )
      self.assertEqual(loaded[key].dtype, np.dtype(bfloat16))
      self.assertEqual(loaded[key].nbytes, original.size * 2)
      np.testing.assert_array_equal(
          loaded[key].view(np.uint16), original.view(np.uint16)
      )
    metrics = compare(loaded['reference'], loaded['target'])
    for key, expected in [
        ('CosSim', 0.8),
        ('Max abs error', 3.0),
        ('Mean abs error', 1.5),
        ('RMSE', np.sqrt(4.5)),
        ('Relative L2', 0.6),
    ]:
      self.assertAlmostEqual(metrics[key]['value'], expected)
    with self.assertRaisesRegex(ValueError, '^non_finite_tensor$'):
      compare(loaded['special'], loaded['special'])
    void = np.zeros(1, dtype='V2')
    with self.assertRaisesRegex(ValueError, '^unsupported_dtype$'):
      compare(void, void)

  def test_store_validates_execution_before_reading_and_reports_bad_shards(
      self,
  ):
    records = [
        {
            **self.record,
            'id': run,
            'run': run,
            'graph': 'g',
            'node': 'n',
            'output': '0',
            'layer': 0,
            'anchor': 'a',
            'batch': 0,
            'turn': 1,
            'phase': 'prefill',
            'step': 0,
            'sample': 'sample',
            'key': key,
        }
        for run, key in [
            ('ref', 'block/output.0'),
            ('target', 'block/output.1'),
        ]
    ]
    documents = {
        'session': {
            'batches': [{'batch': 0, 'turn': 1, 'phase': 'prefill', 'step': 0}]
        },
        'semantic': {
            'layers': [{'def': 0}],
            'semantic_graph': [{'anchors': [{'id': 'a'}]}],
        },
        'execution': {
            'executions': [
                {
                    'id': run,
                    'graphs': [{
                        'id': 'g',
                        'nodes': [
                            {'id': 'n', 'outputsMetadata': [{'id': '0'}]}
                        ],
                    }],
                }
                for run in ('ref', 'target')
            ]
        },
        'tensor_index': {'tensors': records},
    }
    for name, data in documents.items():
      (self.root / (name + '.json')).write_text(json.dumps(data))
    store = SessionStore(self.root)
    self.assertEqual(store.compare_batch(0)['rows'][0]['status'], 'ok')
    with self.assertRaisesRegex(ValueError, '^execution_identity_mismatch$'):
      store.load(
          {**records[0], 'output': 'missing', 'path': '../outside.safetensors'}
      )
    (self.root / 'capture.safetensors').write_bytes(b'invalid shard')
    row = store.compare_batch(0)['rows'][0]
    self.assertEqual(row['status'], 'invalid_tensor_file')
    self.assertEqual(row['metrics'], {})


if __name__ == '__main__':
  unittest.main()
