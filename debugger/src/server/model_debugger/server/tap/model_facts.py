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

"""What a model's own files say that two runtimes are compared by.

Two sides that ran different model files (Gemma 4 E2B in PyTorch and in
LiteRT-LM) are compared only by what both models' files say:

- `vocabulary`: the tokenizer's id -> piece table, as its SHA-256
  (`table_sha256`): every id from 0 up, its piece's UTF-8 bytes, each
  preceded by their length as a little-endian uint32. Two models whose tables
  have one digest name the same token by the same id, so their logits are
  vectors of one vocabulary. A `.litertlm` file holds its tokenizer as a
  section, a SentencePiece model (its pieces in id order) or a Hugging Face
  tokenizer.json compressed with zlib (after its uncompressed size, a
  little-endian uint64); a Hugging Face model directory holds its
  tokenizer.json, whose `model.vocab` and `added_tokens` give each id its
  piece (an added token's piece is the one decoding gives the id).
- `kv_quantization`: how the text model's KV cache tensors are quantized,
  by input name (`kv_cache_k_0`): each integer tensor of one scale, its
  `scale` and `zero_point`, from the decode signature of the `.litertlm`'s
  prefill and decode section, so its stored values are
  scale * (stored - zero_point). A tensor of another quantization is not
  listed. A Hugging Face model's cache is not quantized: it has none.

`describe` reads them; `described` once per file and SHA-256, for a side's
capture and the Session editor's hint alike (session/model_comparison); what
cannot be read is None, and says why in the log, so a comparison that needs
it says it is unknown.
"""

from collections.abc import Iterator
import functools
import hashlib
import json
import pathlib
import re
import struct
from typing import Any
import zlib

from absl import logging
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import litertlm_schema_generated as schema
from model_debugger.server.tap import tflite_schema_generated as tflite

# A Hugging Face model directory's tokenizer, as transformers loads it.
TOKENIZER_JSON = 'tokenizer.json'
# The sources of a vocabulary, as `vocabulary['source']` names them.
SENTENCEPIECE_SECTION = 'SentencePiece section'
HF_TOKENIZER_SECTION = 'Hugging Face tokenizer section'
# The input names of a LiteRT-LM text model's KV cache tensors.
_KV_INPUT = re.compile(r'kv_cache_[kv]_\d+')
# The signature whose KV inputs are read; every signature shares them.
_DECODE = 'decode'
_INTEGER_TYPES = (
    tflite.TensorType.INT8,
    tflite.TensorType.UINT8,
    tflite.TensorType.INT16,
    tflite.TensorType.INT4,
)
# SentencePiece's ModelProto: field 1 holds the pieces; a piece's field 1 is
# its text.
_PIECES_FIELD = 1
_PIECE_TEXT_FIELD = 1
_UINT32 = struct.Struct('<I')
_UINT64 = struct.Struct('<Q')


def table_digest(token_pieces: list[bytes]) -> str:
  """The SHA-256 of an id -> piece table, the pieces in id order."""
  digest = hashlib.sha256()
  for piece in token_pieces:
    digest.update(_UINT32.pack(len(piece)))
    digest.update(piece)
  return digest.hexdigest()


def _vocabulary(token_pieces: list[bytes], source: str) -> dict[str, Any]:
  return {
      'table_sha256': table_digest(token_pieces),
      'size': len(token_pieces),
      'source': source,
  }


@functools.lru_cache(maxsize=16)
def described(path: pathlib.Path, sha256: str) -> dict[str, Any]:
  """describe() of a model, read once per file and SHA-256; do not change.

  Args:
      path: As for describe().
      sha256: The file's SHA-256: a part of the key only, for the file at `path`
        may change.
  """
  del sha256
  return describe(path)


def describe(path: pathlib.Path) -> dict[str, Any]:
  """What a model's files say: `{vocabulary, kv_quantization}`.

  Args:
      path: A `.litertlm` file, or a Hugging Face model directory.

  Returns:
      `vocabulary`: `{table_sha256, size, source}`, or None when the
      tokenizer cannot be read; `kv_quantization`: `{input name: {scale,
      zero_point}}`, {} for a model whose KV cache is not quantized, None
      when the model cannot be read.
  """
  if path.is_dir():
    return {
        'vocabulary': _directory_vocabulary(path),
        'kv_quantization': {},
    }
  try:
    with litertlm_file.mapped(path) as file:
      return {
          'vocabulary': _litertlm_vocabulary(file, path),
          'kv_quantization': _kv_quantization(file, path),
      }
  except (OSError, ValueError) as error:
    logging.warning('The model %s cannot be read: %s', path, error)
    return {'vocabulary': None, 'kv_quantization': None}


@functools.lru_cache(maxsize=4)
def described_pieces(
    path: pathlib.Path, sha256: str
) -> tuple[bytes, ...] | None:
  """pieces() of a model, read once per file and SHA-256; do not change.

  Args:
      path: As for pieces().
      sha256: The file's SHA-256: a part of the key only, as described's.
  """
  del sha256
  return pieces(path)


def pieces(path: pathlib.Path) -> tuple[bytes, ...] | None:
  """A model's tokenizer pieces, in id order: what each id stands for.

  As the vocabulary's digest reads them (describe()); None when the
  tokenizer cannot be read, which the log says why.

  Args:
      path: A `.litertlm` file, or a Hugging Face model directory.

  Returns:
      The tokenizer's pieces in id order, or None when they cannot be read.
  """
  try:
    if path.is_dir():
      return tuple(tokenizer_json_pieces((path / TOKENIZER_JSON).read_bytes()))
    with litertlm_file.mapped(path) as file:
      found = _litertlm_pieces(file, path)
    return None if found is None else tuple(found[0])
  except (OSError, ValueError) as error:
    logging.warning('The tokenizer of %s cannot be read: %s', path, error)
    return None


def _litertlm_pieces(
    file: litertlm_file.LitertlmFile, path: pathlib.Path
) -> tuple[list[bytes], str] | None:
  """A `.litertlm` file's tokenizer pieces, and the section they are of.

  Args:
      file: The mapped `.litertlm` file.
      path: The file's path, for the log.

  Returns:
      The pieces in id order and the name of their section kind, or None
      when the file has no tokenizer section or more than one.

  Raises:
      ValueError: The tokenizer section cannot be read.
  """
  kinds = {
      schema.AnySectionDataType.SP_Tokenizer: SENTENCEPIECE_SECTION,
      schema.AnySectionDataType.HF_Tokenizer_Zlib: HF_TOKENIZER_SECTION,
  }
  found = [s for s in file.sections if s.data_type in kinds]
  if len(found) != 1:
    logging.warning(
        'The model %s has %d tokenizer sections, not one.', path, len(found)
    )
    return None
  section = found[0]
  with file.bytes_of(section) as view:
    data = bytes(view)
  if section.data_type == schema.AnySectionDataType.SP_Tokenizer:
    return sentencepiece_pieces(data), kinds[section.data_type]
  return tokenizer_json_pieces(_inflated(data)), kinds[section.data_type]


def _litertlm_vocabulary(
    file: litertlm_file.LitertlmFile, path: pathlib.Path
) -> dict[str, Any] | None:
  try:
    found = _litertlm_pieces(file, path)
  except ValueError as error:
    logging.warning('The tokenizer of %s cannot be read: %s', path, error)
    return None
  return None if found is None else _vocabulary(*found)


def _inflated(data: bytes) -> bytes:
  """A zlib-compressed tokenizer.json after its uncompressed size."""
  if len(data) < _UINT64.size:
    raise ValueError('The tokenizer section is too short.')
  (size,) = _UINT64.unpack_from(data)
  try:
    text = zlib.decompress(data[_UINT64.size :])
  except zlib.error as error:
    raise ValueError(f'It is not zlib data: {error}') from None
  if len(text) != size:
    raise ValueError('It inflates to another size than it says.')
  return text


def _directory_vocabulary(path: pathlib.Path) -> dict[str, Any] | None:
  try:
    text = (path / TOKENIZER_JSON).read_bytes()
    token_pieces = tokenizer_json_pieces(text)
  except (OSError, ValueError) as error:
    logging.warning('The tokenizer of %s cannot be read: %s', path, error)
    return None
  return _vocabulary(token_pieces, TOKENIZER_JSON)


def _varint(data: bytes, at: int) -> tuple[int, int]:
  value = shift = 0
  while True:
    if at >= len(data) or shift > 63:
      raise ValueError('A protobuf varint runs past its message.')
    byte = data[at]
    at += 1
    value |= (byte & 0x7F) << shift
    shift += 7
    if byte < 0x80:
      return value, at


def _fields(data: bytes) -> Iterator[tuple[int, int | bytes]]:
  """Yields each (field number, value) of a protobuf message, in order.

  A length-delimited value is its bytes; a varint its number; a fixed one
  its bytes.

  Args:
      data: The serialized protobuf message.

  Yields:
      Each (field number, value) pair, in order.

  Raises:
      ValueError: A field runs past the message or has an unknown wire type.
  """
  at = 0
  while at < len(data):
    key, at = _varint(data, at)
    wire = key & 7
    if wire == 0:
      value, at = _varint(data, at)
    elif wire == 2:
      length, at = _varint(data, at)
      value = data[at : at + length]
      if len(value) != length:
        raise ValueError('A protobuf field runs past its message.')
      at += length
    elif wire in (1, 5):
      width = 8 if wire == 1 else 4
      value = data[at : at + width]
      if len(value) != width:
        raise ValueError('A protobuf field runs past its message.')
      at += width
    else:
      raise ValueError(f'A protobuf field has the wire type {wire}.')
    yield key >> 3, value


def sentencepiece_pieces(data: bytes) -> list[bytes]:
  """The pieces of a SentencePiece ModelProto, in id order.

  Args:
      data: The serialized ModelProto.

  Returns:
      The text of each piece, in id order.

  Raises:
      ValueError: It is not a ModelProto with pieces.
  """
  token_pieces = []
  for number, value in _fields(data):
    if number != _PIECES_FIELD:
      continue
    if not isinstance(value, bytes):
      raise ValueError('A SentencePiece piece is not a message.')
    text = next((v for n, v in _fields(value) if n == _PIECE_TEXT_FIELD), None)
    if not isinstance(text, bytes):
      raise ValueError('A SentencePiece piece has no text.')
    token_pieces.append(text)
  if not token_pieces:
    raise ValueError('The SentencePiece model has no token_pieces.')
  return token_pieces


def tokenizer_json_pieces(text: bytes) -> list[bytes]:
  """The pieces of a Hugging Face tokenizer.json, in id order.

  Args:
      text: The tokenizer.json's bytes.

  Returns:
      Each piece, UTF-8 encoded, in id order.

  Raises:
      ValueError: It is not a tokenizer.json, or its ids do not run from 0
        without a gap.
  """
  try:
    tokenizer = json.loads(text)
    vocab = tokenizer['model']['vocab']
    added = tokenizer.get('added_tokens') or []
    if isinstance(vocab, dict):
      by_id = {int(i): piece for piece, i in vocab.items()}
    else:
      # A Unigram model lists [piece, score] in id order.
      by_id = {i: entry[0] for i, entry in enumerate(vocab)}
    for token in added:
      by_id[int(token['id'])] = token['content']
  except (KeyError, TypeError, IndexError, ValueError) as error:
    raise ValueError(f'It is not a tokenizer.json: {error}') from None
  if not by_id or sorted(by_id) != list(range(len(by_id))):
    raise ValueError('Its ids do not run from 0 without a gap.')
  return [by_id[i].encode() for i in range(len(by_id))]


def _kv_quantization(
    file: litertlm_file.LitertlmFile, path: pathlib.Path
) -> dict[str, Any] | None:
  """Returns the KV cache quantization of , or None if it is unknown."""
  try:
    section = file.prefill_decode()
  except ValueError as error:
    logging.warning('The KV cache of %s is unknown: %s', path, error)
    return None
  with file.bytes_of(section) as data:
    try:
      return kv_quantization(data)
    except (ValueError, IndexError, struct.error) as error:
      logging.warning('The KV cache of %s is unknown: %s', path, error)
      return None


def kv_quantization(model: 'bytes | memoryview[int]') -> dict[str, Any]:
  """Each KV cache input's one scale and zero point, of a TFLite model.

  Args:
      model: The TFLite model, its decode signature (or else its first) naming
        the KV inputs.

  Returns:
      `{input name: {scale, zero_point}}` of each integer KV input of one
      scale.
  """
  root = tflite.Model.GetRootAs(model, 0)
  if not root.SignatureDefsLength():
    return {}
  definitions = [
      root.SignatureDefs(i) for i in range(root.SignatureDefsLength())
  ]
  definition = next(
      (d for d in definitions if d.SignatureKey() == _DECODE.encode()),
      definitions[0],
  )
  graph = root.Subgraphs(definition.SubgraphIndex())
  result = {}
  for i in range(definition.InputsLength()):
    entry = definition.Inputs(i)
    name = entry.Name().decode()
    if not _KV_INPUT.fullmatch(name):
      continue
    tensor = graph.Tensors(entry.TensorIndex())
    quantization = tensor.Quantization()
    if (
        tensor.Type() not in _INTEGER_TYPES
        or quantization is None
        or quantization.ScaleLength() != 1
    ):
      continue
    zero_point = (
        int(quantization.ZeroPoint(0)) if quantization.ZeroPointLength() else 0
    )
    result[name] = {
        'scale': float(quantization.Scale(0)),
        'zero_point': zero_point,
    }
  return result
