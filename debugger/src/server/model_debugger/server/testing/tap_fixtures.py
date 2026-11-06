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

"""A small .litertlm model to observe tensors of, laid out as LiteRT-LM's are.

Its text model has the signatures of a LiteRT-LM model, each its own
subgraph: `decode`, `prefill_4`, `prefill_16` and `verify`. Each takes
`embeddings` [1, 1, 4] and a K cache `kv_cache_k_0` [1, 1, KV_LENGTH, 2],
and runs four operators:

  0  ADD(embeddings, weight)        -> sum      [1, 1, 4] float32
  1  MUL(sum, weight)               -> activations, the signature's output
  2  ADD(kv_cache_k_0, bias)        -> scores   [1, 1, KV_LENGTH, 2]
  3  CAST(sum)                      -> text     [1, 1, 4], a string

`weight` is inline in the model, quantized (WEIGHT_SCALE); `bias` is out
of line, after the FlatBuffer (Buffer.offset), and so are decode's CAST's
large custom options (CUSTOM_OPTIONS) and an empty buffer, at the section's
end, as LiteRT-LM's models have one. The ADDs fuse a RELU. Around the text
model are a metadata section before and an embedder section after, so a copy
that grows the text model moves the embedder.
"""

import pathlib
import struct

import flatbuffers
from model_debugger.server.tap import litertlm_schema_generated as schema
from model_debugger.server.tap import tflite_schema_generated as tflite
import numpy as np

KV_LENGTH = 10
SIGNATURES = ('decode', 'prefill_4', 'prefill_16', 'verify')
UUID = '00000000-0000-4000-8000-000000000000'
# The bytes of the sections around the text model.
METADATA = b'llm metadata, as far as the copy can tell'
EMBEDDER = b'an embedder model' * 100
_BLOCK = 16 * 1024
# The bytes of `weight`, and of `bias`, out of line at _BIAS_AT.
WEIGHT = np.array([7.5, -1.0, 3.25, 9.0], dtype=np.float32).tobytes()
BIAS = np.arange(KV_LENGTH * 2, dtype=np.float32).tobytes()
_BIAS_AT = 8192
CUSTOM_OPTIONS = b'large custom options of CAST ' * 8
_CUSTOM_AT = _BIAS_AT + 128
WEIGHT_SCALE = 0.5


def _tensor(
    name: str, shape: list[int], kind: int, buffer: int = 0
) -> tflite.TensorT:
  tensor = tflite.TensorT()
  tensor.name = name
  tensor.shape = shape
  tensor.type = kind
  tensor.buffer = buffer
  return tensor


def _op(code: int, inputs: list[int], outputs: list[int]) -> tflite.OperatorT:
  op = tflite.OperatorT()
  op.opcodeIndex = code
  op.inputs = inputs
  op.outputs = outputs
  if code == 0:
    op.builtinOptionsType = tflite.BuiltinOptions.AddOptions
    op.builtinOptions = tflite.AddOptionsT()
    op.builtinOptions.fusedActivationFunction = (
        tflite.ActivationFunctionType.RELU
    )
  return op


def _map(name: str, index: int) -> tflite.TensorMapT:
  entry = tflite.TensorMapT()
  entry.name = name
  entry.tensorIndex = index
  return entry


def text_model() -> bytes:
  """The text model: a TFLite FlatBuffer with its payloads after it."""
  model = tflite.ModelT()
  model.version = 3
  codes = []
  for builtin in (
      tflite.BuiltinOperator.ADD,
      tflite.BuiltinOperator.MUL,
      tflite.BuiltinOperator.CAST,
  ):
    code = tflite.OperatorCodeT()
    code.builtinCode = builtin
    code.deprecatedBuiltinCode = min(builtin, 127)
    codes.append(code)
  model.operatorCodes = codes
  weight = tflite.BufferT()
  weight.data = np.frombuffer(WEIGHT, dtype=np.uint8)
  bias = tflite.BufferT()
  bias.offset = _BIAS_AT
  bias.size = len(BIAS)
  # Empty, out of line at the section's end.
  empty = tflite.BufferT()
  empty.offset = _CUSTOM_AT + len(CUSTOM_OPTIONS)
  empty.size = 0
  model.buffers = [tflite.BufferT(), weight, bias, empty]
  model.subgraphs, model.signatureDefs = [], []
  f32 = tflite.TensorType.FLOAT32
  for index, name in enumerate(SIGNATURES):
    graph = tflite.SubGraphT()
    quantized = _tensor('weight', [4], f32, 1)
    quantized.quantization = tflite.QuantizationParametersT()
    quantized.quantization.scale = [WEIGHT_SCALE]
    quantized.quantization.zeroPoint = [0]
    graph.tensors = [
        _tensor('embeddings', [1, 1, 4], f32),
        quantized,
        _tensor(f'{name}/layer_0/sum', [1, 1, 4], f32),
        _tensor(f'{name}/layer_0/activations', [1, 1, 4], f32),
        _tensor('kv_cache_k_0', [1, 1, KV_LENGTH, 2], f32),
        _tensor('bias', [KV_LENGTH, 2], f32, 2),
        _tensor(f'{name}/layer_0/scores', [1, 1, KV_LENGTH, 2], f32),
        _tensor(f'{name}/layer_0/text', [1, 1, 4], tflite.TensorType.STRING),
    ]
    graph.inputs = [0, 4]
    graph.outputs = [3]
    graph.operators = [
        _op(0, [0, 1], [2]),
        _op(1, [2, 1], [3]),
        _op(0, [4, 5], [6]),
        _op(2, [2], [7]),
    ]
    if name == 'decode':
      graph.operators[3].largeCustomOptionsOffset = _CUSTOM_AT
      graph.operators[3].largeCustomOptionsSize = len(CUSTOM_OPTIONS)
    model.subgraphs.append(graph)
    definition = tflite.SignatureDefT()
    definition.signatureKey = name
    definition.subgraphIndex = index
    definition.inputs = [_map('embeddings', 0), _map('kv_cache_k_0', 4)]
    definition.outputs = [_map('activations', 3)]
    model.signatureDefs.append(definition)
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  header = bytes(builder.Output())
  assert len(header) <= _BIAS_AT
  return (
      header
      + bytes(_BIAS_AT - len(header))
      + BIAS
      + bytes(_CUSTOM_AT - _BIAS_AT - len(BIAS))
      + CUSTOM_OPTIONS
  )


def layered_model(layers: int, unnamed: bool = False) -> bytes:
  """A TFLite model of one signature, `decode`, of `layers` layers.

  Layer i reads the residual stream x and writes, in order:
    `.../layer_i/layer_i.attn/add`  ADD(x, weight)
    `.../layer_i/layer_i.attn/kv`   MUL(x, weight), read by no later layer
    `.../layer_i/layer_i.post/mul`  MUL(add, weight), the next x
  and then `final_norm/mul` MUL(x, weight) is the signature's output.

  Args:
      layers: How many.
      unnamed: Whether layer 0's `attn/kv` MUL writes a second output,
        `StatefulPartitionedCall:1`, the signature's output `kv_cache_k_0`,
        named by no scope, as Gemma 4's layer 0 KV cache writes are.

  Returns:
      The serialized model.
  """
  model = tflite.ModelT()
  model.version = 3
  codes = []
  for builtin in (tflite.BuiltinOperator.ADD, tflite.BuiltinOperator.MUL):
    code = tflite.OperatorCodeT()
    code.builtinCode = builtin
    code.deprecatedBuiltinCode = builtin
    codes.append(code)
  model.operatorCodes = codes
  weight = tflite.BufferT()
  weight.data = np.frombuffer(WEIGHT, dtype=np.uint8)
  model.buffers = [tflite.BufferT(), weight]
  f32 = tflite.TensorType.FLOAT32
  scope = 'LanguageModel.decode_graph/transformer'
  graph = tflite.SubGraphT()
  graph.tensors = [
      _tensor('embeddings', [1, 1, 4], f32),
      _tensor('w', [4], f32, 1),
  ]
  graph.operators = []
  x = 0
  for i in range(layers):
    base = f'{scope}/layer_{i}/layer_{i}'
    for name in ('attn/add', 'attn/kv', 'post/mul'):
      graph.tensors.append(_tensor(f'{base}.{name}', [1, 1, 4], f32))
    add, kv, out = range(len(graph.tensors) - 3, len(graph.tensors))
    writes = [kv]
    if unnamed and i == 0:
      graph.tensors.append(_tensor('StatefulPartitionedCall:1', [1, 1, 4], f32))
      cache = len(graph.tensors) - 1
      writes.append(cache)
    graph.operators += [_op(0, [x, 1], [add]), _op(1, [x, 1], writes)]
    graph.operators.append(_op(1, [add, 1], [out]))
    x = out
  graph.tensors.append(_tensor(f'{scope}/final_norm/mul', [1, 1, 4], f32))
  graph.operators.append(_op(1, [x, 1], [len(graph.tensors) - 1]))
  graph.inputs = [0]
  graph.outputs = [len(graph.tensors) - 1]
  model.subgraphs = [graph]
  definition = tflite.SignatureDefT()
  definition.signatureKey = 'decode'
  definition.subgraphIndex = 0
  definition.inputs = [_map('embeddings', 0)]
  definition.outputs = [_map('activations', len(graph.tensors) - 1)]
  if unnamed:
    graph.outputs.append(cache)
    definition.outputs.append(_map('kv_cache_k_0', cache))
  model.signatureDefs = [definition]
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


def _pair(key: str, value: str) -> schema.KeyValuePairT:
  pair = schema.KeyValuePairT()
  pair.key = key
  pair.valueType = schema.VData.StringValue
  pair.value = schema.StringValueT()
  pair.value.value = value
  return pair


def write_model(
    path: pathlib.Path,
    text: bytes | None = None,
    between: bytes = b'',
    after_text: bytes = b'',
    tokenizer: tuple[int, bytes] | None = None,
) -> None:
  """Writes the model to `path`: three sections, the text model second.

  Args:
      path: Where.
      text: The text model's bytes; text_model() by default.
      between: Bytes written after the header and after each section but the
        text model, before the zeros up to the next section or the file's end.
      after_text: Bytes written so after the text model.
      tokenizer: A fourth section, last: its schema.AnySectionDataType and its
        bytes; none by default.
  """
  sections = [
      (schema.AnySectionDataType.LlmMetadataProto, {}, METADATA),
      (
          schema.AnySectionDataType.TFLiteModel,
          {'model_type': 'tf_lite_prefill_decode'},
          text_model() if text is None else text,
      ),
      (
          schema.AnySectionDataType.TFLiteModel,
          {'model_type': 'tf_lite_embedder'},
          EMBEDDER,
      ),
  ]
  if tokenizer is not None:
    sections.append((tokenizer[0], {}, tokenizer[1]))
  metadata = schema.LiteRTLMMetaDataT()
  metadata.systemMetadata = schema.SystemMetadataT()
  metadata.systemMetadata.entries = [
      _pair('author', 'The tests'),
      _pair('uuid', UUID),
      _pair('creation_timestamp', '2026-10-18T00:00:00+00:00'),
  ]
  metadata.sectionMetadata = schema.SectionMetadataT()
  metadata.sectionMetadata.objects = []
  offset = _BLOCK
  for kind, items, data in sections:
    item = schema.SectionObjectT()
    item.dataType = kind
    item.items = [_pair(key, value) for key, value in items.items()]
    item.beginOffset = offset
    item.endOffset = offset + len(data)
    metadata.sectionMetadata.objects.append(item)
    offset = -(-(offset + len(data)) // _BLOCK) * _BLOCK
  builder = flatbuffers.Builder(1024)
  builder.Finish(metadata.Pack(builder))
  header = bytes(builder.Output())
  with open(path, 'wb') as out:
    out.write(b'LITERTLM' + struct.pack('<III', 1, 5, 0) + bytes(4))
    out.write(struct.pack('<Q', 32 + len(header)) + header + between)
    for index, (item, (_, _, data)) in enumerate(
        zip(metadata.sectionMetadata.objects, sections)
    ):
      out.write(bytes(item.beginOffset - out.tell()))
      out.write(data + (after_text if index == 1 else between))
    out.write(bytes(offset - out.tell()))
