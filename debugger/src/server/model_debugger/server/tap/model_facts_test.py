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

"""Tests of what a model's files say: its vocabulary and KV quantization."""

import json
import pathlib
import struct
import zlib

from absl.testing import absltest
import flatbuffers
from model_debugger.server.tap import litertlm_schema_generated as schema
from model_debugger.server.tap import model_facts
from model_debugger.server.tap import tflite_schema_generated as tflite
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

PIECES = ('<pad>', '<eos>', '<bos>', '▁the', '\n', 'colors')


def _varint(value: int) -> bytes:
  out = bytearray()
  while True:
    byte = value & 0x7F
    value >>= 7
    if value:
      out.append(byte | 0x80)
    else:
      out.append(byte)
      return bytes(out)


def _field(number: int, value: bytes) -> bytes:
  return _varint(number << 3 | 2) + _varint(len(value)) + value


def _sentencepiece(pieces: list[str] | tuple[str, ...]) -> bytes:
  """A SentencePiece ModelProto: each piece's text, score and type."""
  model = b''
  for piece in pieces:
    message = (
        _field(1, piece.encode())
        + _varint(2 << 3 | 5)
        + struct.pack('<f', -1.5)
        + _varint(3 << 3)
        + _varint(1)
    )
    model += _field(1, message)
  # The trainer and normalizer specs, which hold no piece.
  return model + _field(2, b'\x0a\x03unk') + _field(3, b'\x0a\x04nmt')


def _tokenizer_json(
    pieces: list[str] | tuple[str, ...], added: tuple[int, ...] = (0, 1, 2)
) -> bytes:
  """A BPE tokenizer.json: its vocab, some ids also as added tokens."""
  return json.dumps({
      'added_tokens': [
          {'id': i, 'content': pieces[i], 'special': True} for i in added
      ],
      'model': {
          'type': 'BPE',
          'vocab': {piece: i for i, piece in enumerate(pieces)},
          'merges': [],
      },
  }).encode()


def _kv_tensor(
    name: str, kind: int, scales: list[float] | None = None
) -> tflite.TensorT:
  result = tflite.TensorT()
  result.name = name
  result.shape = [1, 1, 4, 2]
  result.type = kind
  result.buffer = 0
  if scales is not None:
    result.quantization = tflite.QuantizationParametersT()
    result.quantization.scale = scales
    result.quantization.zeroPoint = [3] * len(scales)
  return result


def _kv_model() -> bytes:
  """A TFLite model whose decode signature reads three KV caches.

  `kv_cache_k_0` is int8 of one scale, `kv_cache_v_0` int8 of a scale per
  channel, `kv_cache_k_1` float32; `embeddings` is no KV cache.

  Returns:
      The serialized model.
  """
  model = tflite.ModelT()
  model.version = 3
  model.buffers = [tflite.BufferT()]
  graph = tflite.SubGraphT()
  int8, f32 = tflite.TensorType.INT8, tflite.TensorType.FLOAT32
  graph.tensors = [
      _kv_tensor('embeddings', f32),
      _kv_tensor('kv_cache_k_0', int8, [0.25]),
      _kv_tensor('kv_cache_v_0', int8, [0.25, 0.5]),
      _kv_tensor('kv_cache_k_1', f32),
  ]
  graph.inputs = [0, 1, 2, 3]
  graph.outputs = []
  graph.operators = []
  model.subgraphs = [graph]
  definition = tflite.SignatureDefT()
  definition.signatureKey = 'decode'
  definition.subgraphIndex = 0
  definition.inputs = []
  for index, name in enumerate(
      ('embeddings', 'kv_cache_k_0', 'kv_cache_v_0', 'kv_cache_k_1')
  ):
    entry = tflite.TensorMapT()
    entry.name = name
    entry.tensorIndex = index
    definition.inputs.append(entry)
  definition.outputs = []
  model.signatureDefs = [definition]
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


class ModelFactsTest(absltest.TestCase):
  """Tests of `describe` of either kind of model, and of its parts."""

  def setUp(self):
    """An empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def _litertlm(self, tokenizer=None, text=None) -> pathlib.Path:
    path = self.tmp_path / 'model.litertlm'
    tap_fixtures.write_model(path, text=text, tokenizer=tokenizer)
    return path

  def test_the_digest_is_of_the_pieces_in_id_order(self):
    """The digest tells the order and the bounds of the pieces."""
    digest = model_facts.table_digest([b'a', b'bc'])
    self.assertEqual(digest, model_facts.table_digest([b'a', b'bc']))
    self.assertNotEqual(digest, model_facts.table_digest([b'bc', b'a']))
    # Lengths keep pieces apart: 'ab', 'c' is not 'a', 'bc'.
    self.assertNotEqual(digest, model_facts.table_digest([b'ab', b'c']))

  def test_a_sentencepiece_section_and_a_tokenizer_json_agree(self):
    """One table in either format has one digest."""
    litertlm = self._litertlm(
        (schema.AnySectionDataType.SP_Tokenizer, _sentencepiece(PIECES))
    )
    directory = self.tmp_path / 'hf'
    directory.mkdir()
    (directory / 'tokenizer.json').write_bytes(_tokenizer_json(PIECES))
    ours = model_facts.describe(litertlm)['vocabulary']
    theirs = model_facts.describe(directory)['vocabulary']
    self.assertEqual(ours['source'], model_facts.SENTENCEPIECE_SECTION)
    self.assertEqual(theirs['source'], model_facts.TOKENIZER_JSON)
    self.assertLen(PIECES, ours['size'])
    self.assertEqual(ours['table_sha256'], theirs['table_sha256'])
    self.assertEqual(
        ours['table_sha256'],
        model_facts.table_digest([piece.encode() for piece in PIECES]),
    )

  def test_another_piece_is_another_vocabulary(self):
    """One piece that differs is another vocabulary."""
    other = PIECES[:3] + ('▁a',) + PIECES[4:]
    self.assertNotEqual(
        model_facts.table_digest(
            model_facts.sentencepiece_pieces(_sentencepiece(PIECES))
        ),
        model_facts.table_digest(
            model_facts.tokenizer_json_pieces(_tokenizer_json(other))
        ),
    )

  def test_an_added_token_names_its_id(self):
    """An added token gives its id the piece decoding gives it."""
    text = json.loads(_tokenizer_json(PIECES))
    text['added_tokens'].append({'id': 3, 'content': '<start>'})
    pieces = model_facts.tokenizer_json_pieces(json.dumps(text).encode())
    self.assertEqual(pieces[3], b'<start>')

  def test_a_unigram_vocab_lists_its_pieces_in_id_order(self):
    """A Unigram vocab is a list in id order."""
    text = json.dumps(
        {'model': {'vocab': [[piece, -1.0] for piece in PIECES]}}
    ).encode()
    self.assertEqual(
        model_facts.tokenizer_json_pieces(text),
        [piece.encode() for piece in PIECES],
    )

  def test_ids_with_a_gap_are_no_vocabulary(self):
    """Ids that skip one make no table."""
    text = json.dumps({'model': {'vocab': {'a': 0, 'b': 2}}}).encode()
    with self.assertRaisesRegex(ValueError, 'without a gap'):
      model_facts.tokenizer_json_pieces(text)

  def test_a_compressed_tokenizer_section(self):
    """A .litertlm may hold its tokenizer.json compressed."""
    text = _tokenizer_json(PIECES)
    data = struct.pack('<Q', len(text)) + zlib.compress(text)
    vocabulary = model_facts.describe(
        self._litertlm((schema.AnySectionDataType.HF_Tokenizer_Zlib, data))
    )['vocabulary']
    self.assertEqual(vocabulary['source'], model_facts.HF_TOKENIZER_SECTION)
    self.assertEqual(
        vocabulary['table_sha256'],
        model_facts.table_digest([piece.encode() for piece in PIECES]),
    )

  def test_a_model_without_a_readable_tokenizer_has_no_vocabulary(self):
    """No tokenizer section, a broken one, or no tokenizer.json: None."""
    self.assertIsNone(model_facts.describe(self._litertlm())['vocabulary'])
    broken = self._litertlm(
        (schema.AnySectionDataType.SP_Tokenizer, b'\x0a\x05ab')
    )
    self.assertIsNone(model_facts.describe(broken)['vocabulary'])
    self.assertIsNone(model_facts.describe(self.tmp_path)['vocabulary'])

  def test_the_pieces_of_either_format_are_one_table(self):
    """pieces() reads what the digest is of; None without a tokenizer."""
    litertlm = self._litertlm(
        (schema.AnySectionDataType.SP_Tokenizer, _sentencepiece(PIECES))
    )
    directory = self.tmp_path / 'hf'
    directory.mkdir()
    (directory / 'tokenizer.json').write_bytes(_tokenizer_json(PIECES))
    expected = tuple(piece.encode() for piece in PIECES)
    self.assertEqual(model_facts.pieces(litertlm), expected)
    self.assertEqual(model_facts.pieces(directory), expected)
    self.assertIsNone(model_facts.pieces(self._litertlm()))
    self.assertIsNone(model_facts.pieces(self.tmp_path / 'none'))

  def test_kv_inputs_of_one_scale_are_listed(self):
    """Integer KV inputs of one scale, and only they, are listed."""
    self.assertEqual(
        model_facts.kv_quantization(_kv_model()),
        {'kv_cache_k_0': {'scale': 0.25, 'zero_point': 3}},
    )
    described = model_facts.describe(self._litertlm(text=_kv_model()))
    self.assertEqual(
        described['kv_quantization'],
        {'kv_cache_k_0': {'scale': 0.25, 'zero_point': 3}},
    )

  def test_float_kv_caches_are_not_quantized(self):
    """A float KV cache, or a directory's, has no quantization."""
    self.assertEqual(
        model_facts.describe(self._litertlm())['kv_quantization'], {}
    )
    self.assertEqual(model_facts.describe(self.tmp_path)['kv_quantization'], {})

  def test_a_file_that_is_no_model_says_nothing(self):
    """What cannot be read is None."""
    path = self.tmp_path / 'not.litertlm'
    path.write_bytes(b'not a model')
    self.assertEqual(
        model_facts.describe(path),
        {'vocabulary': None, 'kv_quantization': None},
    )

  def test_described_and_described_pieces_cache_by_path_and_sha256(self):
    """described() and described_pieces() return cached facts and pieces."""
    litertlm = self._litertlm(
        (schema.AnySectionDataType.SP_Tokenizer, _sentencepiece(PIECES))
    )
    self.assertEqual(
        model_facts.described(litertlm, 'sha_v1'),
        model_facts.describe(litertlm),
    )
    self.assertEqual(
        model_facts.described_pieces(litertlm, 'sha_v1'),
        model_facts.pieces(litertlm),
    )


if __name__ == '__main__':
  test_main.main()
