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

"""The tensors of a model that can be observed: its operators' outputs.

A point is one output of one operator of one signature of the text model,
the prefill and decode TFLite section of a .litertlm file, named
`<signature>:<op>:<output>`. Observing it makes the tensor an output of its
signature (tap_model), which the runtime then captures at every step that
signature runs. The scan reads the model's header and graphs only, not its
weights, and runs nothing.

Not every signature runs in every chat. LiteRT-LM runs `decode` for every
token of a reply and covers a prompt with prefill chunks: a short prompt
with the smallest `prefill_<N>` alone, a longer one with larger ones too;
`verify` runs only with speculative decoding. A point only in a signature
that does not run captures nothing (Signature.runs).

Some tensors carry the model's KV length, the length of its KV cache inputs:
in a capture that dimension is the context length the model was loaded with
(Point.context_axes).

A tensor's layer is read from its name, as the converter wrote it, by the
name of the model's layers: the scope component `<stem>_<N>` that the
graph of the model (graph/model_graph.layer_stems) names its repeated layers
by, such as `layer` for Gemma 4 or `...Qwen3DecoderLayer` for Qwen3. The
first such component of the scope of a tensor's first name (a tensor of
fused operators names each of them, `;` between) is its layer. A tensor
whose name names no layer, such as Gemma 4's layer 0 KV cache writes, named
`StatefulPartitionedCall:1` and the like, is in the layer the graph groups
its operator in (graph/model_graph.layer_groups), if the scan is given it.
A scan made without the stems places no tensor in a layer. A layer's
output is the tensor of its last operator that an operator of the next
layer reads; the last layer of a signature has none to read it, and its
output is the tensor named as the other layers' outputs are, but for their
numbers (Point.layer_output).
"""

import collections
from collections.abc import Mapping
import dataclasses
import hashlib
import pathlib
import re
from typing import Any

from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import tflite_schema_generated as tflite

# The tensor types the runtime captures as they are.
CAPTURED_TYPES = frozenset((
    tflite.TensorType.FLOAT32,
    tflite.TensorType.FLOAT16,
    tflite.TensorType.INT32,
    tflite.TensorType.INT64,
    tflite.TensorType.INT8,
    tflite.TensorType.UINT8,
    tflite.TensorType.BOOL,
))
# Operators whose outputs are not observed: control flow.
_CONTROL_FLOW = frozenset((
    tflite.BuiltinOperator.IF,
    tflite.BuiltinOperator.WHILE,
    tflite.BuiltinOperator.CALL_ONCE,
))
# Signature.runs: when LiteRT-LM runs a signature.
RUNS_ALWAYS = 'always'  # decode, and the smallest prefill
RUNS_LONG_PROMPTS = 'long_prompts'  # a larger prefill
RUNS_SPECULATIVE = 'speculative_decoding'  # verify
RUNS_UNKNOWN = 'unknown'  # a signature LiteRT-LM is not known to run
# A K cache input: [batch, kv_head, sequence, head_dim], as the runtime
# captures it (capture_index); its sequence axis is the model's KV length.
_K_CACHE = re.compile(r'kv_cache_k_\d+')
_K_SEQUENCE_AXIS = 2
_PREFILL = re.compile(r'prefill_(\d+)')
# The four points reviewed for Gemma 4 E2B: layer 0's pre-attention RMSNorm
# in each signature. They hold for these exact bytes of its prefill and
# decode section alone.
REVIEWED_SECTION_SHA256 = (
    '37dcf27b6d3b8522891dc0d3bbaaf4f3a2e6f264ad0c6d25238355de93be85a2'
)
REVIEWED_POINTS = (
    'decode:10:0',
    'prefill_1024:12:0',
    'prefill_128:12:0',
    'verify:11:0',
)
_HASH_CHUNK = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class Signature:
  """One signature of the text model.

  Attributes:
      name: Its key, such as 'decode' or 'prefill_128'.
      subgraph: The subgraph it runs.
      runs: When LiteRT-LM runs it: one of the RUNS_* values.
  """

  name: str
  subgraph: int
  runs: str


@dataclasses.dataclass(frozen=True)
class Point:
  """One operator output of a signature, which may be observed.

  Attributes:
      id: '<signature>:<op>:<output>'.
      signature: The signature's key.
      subgraph: The subgraph the signature runs.
      op: The operator's index in that subgraph.
      output: Which of the operator's outputs.
      tensor: The tensor's index in that subgraph.
      tensor_name: The tensor's name in the model.
      shape: Its shape in the model.
      dtype: Its type, as tensor_dtype() names it.
      operator: The operator: its builtin name, and a composite's name.
      output_name: The signature output that already holds the tensor, if
        any: the runtime captures it without observing it.
      context_axes: The axes of `shape` of the model's KV length, which a
        capture holds at the context length instead.
      reason: Why it cannot be observed; '' if it can.
      layer: The transformer layer its name places it in, if any.
      layer_output: Whether it is its layer's output.
      layer_stem: What its layer is named by, the scope component
        `<layer_stem>_<layer>`; '' when it is in no layer.
  """

  id: str
  signature: str
  subgraph: int
  op: int
  output: int
  tensor: int
  tensor_name: str
  shape: tuple[int, ...]
  dtype: str
  operator: str
  output_name: str | None
  context_axes: tuple[int, ...]
  reason: str
  layer: int | None = None
  layer_output: bool = False
  layer_stem: str = ''

  @property
  def selectable(self) -> bool:
    """Whether it can be observed."""
    return not self.reason


@dataclasses.dataclass(frozen=True)
class Scan:
  """What a model offers to observe.

  Attributes:
      section_sha256: SHA-256 of the bytes of its prefill and decode section.
      kv_length: The length of its KV cache inputs, if it has any.
      signatures: Its signatures, in the model's order.
      points: Every operator output of every signature, by signature, then
        operator, then output.
      recommended: The ids of the points reviewed for these exact bytes, if any.
  """

  section_sha256: str
  kv_length: int | None
  signatures: tuple[Signature, ...]
  points: tuple[Point, ...]
  recommended: tuple[str, ...]

  def point(self, wanted_id: str) -> Point:
    """The point of an id.

    Args:
        wanted_id: The point's id, as point_id() makes it.

    Returns:
        The model's point of that id.

    Raises:
        ValueError: The model has no such point.
    """
    for point in self.points:
      if point.id == wanted_id:
        return point
    raise ValueError(f'The model has no tensor output {wanted_id}.')


def point_id(signature: str, op: int, output: int) -> str:
  """The id of an operator output of a signature."""
  return f'{signature}:{op}:{output}'


def to_json(found: Scan) -> dict[str, Any]:
  """A scan as JSON, which from_json() reads back.

  A Session moved to another computer is read there without its model
  file (session/session_archive), so its models' scans go with it.

  Args:
      found: The scan.

  Returns:
      The scan as a JSON-serializable dict.
  """
  return {
      'section_sha256': found.section_sha256,
      'kv_length': found.kv_length,
      'signatures': [dataclasses.asdict(s) for s in found.signatures],
      'points': [dataclasses.asdict(p) for p in found.points],
      'recommended': list(found.recommended),
  }


def from_json(value: dict[str, Any]) -> Scan:
  """The scan to_json() wrote.

  A key it no longer writes, such as `layer_stems` in a Session exported
  before, is ignored.

  Args:
      value: A scan as to_json() wrote it.

  Returns:
      The scan.

  Raises:
      ValueError: `value` is not one.
  """
  try:
    return Scan(
        section_sha256=str(value['section_sha256']),
        kv_length=value['kv_length'],
        signatures=tuple(Signature(**s) for s in value['signatures']),
        points=tuple(
            Point(**{
                **p,
                'shape': tuple(p['shape']),
                'context_axes': tuple(p['context_axes']),
            })
            for p in value['points']
        ),
        recommended=tuple(value['recommended']),
    )
  except (KeyError, TypeError, AttributeError) as error:
    raise ValueError(f'This is not a kept scan: {error!r}') from error


def tensor_dtype(tensor_type: int) -> str:
  """A TFLite tensor type's name, such as 'float32'."""
  for name, value in vars(tflite.TensorType).items():
    if value == tensor_type and name.isupper():
      return name.lower()
  return f'type {tensor_type}'


def builtin(model: tflite.Model, op: tflite.Operator) -> int:
  """The builtin code of an operator, old and new fields alike."""
  code = model.OperatorCodes(op.OpcodeIndex())
  return max(code.BuiltinCode(), code.DeprecatedBuiltinCode())


def scan(
    path: pathlib.Path,
    layer_stems: Mapping[str, str] | None = None,
    layer_groups: Mapping[str, Mapping[int, int]] | None = None,
) -> Scan:
  """Lists what a .litertlm model offers to observe.

  Args:
      path: The model.
      layer_stems: Per signature, the stem of the scope component its layers are
        named by (graph/model_graph.layer_stems); without, no point is in a
        layer.
      layer_groups: Per signature, per operator, by its index, the layer the
        graph groups it in, of those an output of which is named by no layer
        (graph/model_graph.layer_groups): that output is in it.

  Returns:
      Its points.

  Raises:
      ValueError: It is not a .litertlm file, or has no prefill and decode
        TFLite section of its own.
  """
  with litertlm_file.mapped(path) as container:
    section = container.prefill_decode()
    with container.bytes_of(section) as data:
      digest = hashlib.sha256()
      for begin in range(0, len(data), _HASH_CHUNK):
        digest.update(data[begin : begin + _HASH_CHUNK])
      if bytes(data[4:8]) != b'TFL3':
        raise ValueError(
            'The prefill and decode section is not a TFLite model.'
        )
      found = read_model(data, layer_stems, layer_groups)
  sha256 = digest.hexdigest()
  signatures, points, kv_length = found
  recommended = REVIEWED_POINTS if sha256 == REVIEWED_SECTION_SHA256 else ()
  return Scan(
      section_sha256=sha256,
      kv_length=kv_length,
      signatures=signatures,
      points=points,
      recommended=recommended,
  )


def read_model(
    data: 'bytes | memoryview[int]',
    layer_stems: Mapping[str, str] | None = None,
    layer_groups: Mapping[str, Mapping[int, int]] | None = None,
) -> tuple[tuple[Signature, ...], tuple[Point, ...], int | None]:
  """Lists the signatures and points of a TFLite model.

  Nothing returned refers to `data`.

  Args:
      data: The model.
      layer_stems: Per signature, the stem its layers are named by; a signature
        without one has no layers.
      layer_groups: Per signature, the layer of each operator whose outputs'
        names may name none (scan()).

  Returns:
      Its signatures, its points, and its KV length, if it has KV caches.
  """
  model = tflite.Model.GetRootAs(data, 0)
  keys = [model.SignatureDefs(i) for i in range(model.SignatureDefsLength())]
  users: dict[int, int] = {}
  for definition in keys:
    index = definition.SubgraphIndex()
    users[index] = users.get(index, 0) + 1
  kv_length = _kv_length(model, keys)
  prefill_sizes = sorted(
      int(match.group(1))
      for definition in keys
      if (match := _PREFILL.fullmatch(definition.SignatureKey().decode()))
  )
  signatures, points = [], []
  for definition in keys:
    name = definition.SignatureKey().decode()
    subgraph = definition.SubgraphIndex()
    signatures.append(Signature(name, subgraph, _runs(name, prefill_sizes)))
    graph = model.Subgraphs(subgraph)
    outputs = {
        definition.Outputs(i).TensorIndex(): (
            definition.Outputs(i).Name().decode()
        )
        for i in range(definition.OutputsLength())
    }
    producers: dict[int, int] = {}
    for op_index in range(graph.OperatorsLength()):
      op = graph.Operators(op_index)
      for slot in range(op.OutputsLength()):
        tensor = op.Outputs(slot)
        producers[tensor] = producers.get(tensor, 0) + 1
    stem = (layer_stems or {}).get(name)
    groups = (layer_groups or {}).get(name, {})
    layer_outputs = _layer_outputs(graph, stem, groups) if stem else set()
    for op_index in range(graph.OperatorsLength()):
      op = graph.Operators(op_index)
      code = builtin(model, op)
      operator = _operator(model, op, code)
      for slot in range(op.OutputsLength()):
        index = op.Outputs(slot)
        if index < 0:
          continue  # An optional output the operator does not write.
        tensor = graph.Tensors(index)
        shape = tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength()))
        dynamic = any(
            tensor.ShapeSignature(i) < 0
            for i in range(tensor.ShapeSignatureLength())
        )
        if users[subgraph] != 1:
          reason = (
              'Its graph is shared by several signatures, which '
              'this server does not observe.'
          )
        elif producers[index] != 1:
          reason = 'Several operators write this tensor.'
        elif tensor.IsVariable():
          reason = 'It is a variable tensor.'
        elif tensor.Type() not in CAPTURED_TYPES:
          reason = f'Its type, {tensor_dtype(tensor.Type())}, is not captured.'
        elif not shape or any(d <= 0 for d in shape) or dynamic:
          reason = 'Its shape is empty or dynamic.'
        elif code in _CONTROL_FLOW:
          reason = 'It is an output of control flow.'
        else:
          reason = ''
        layer = (
            _layer((tensor.Name() or b'').decode(), stem, groups, op_index)
            if stem
            else None
        )
        points.append(
            Point(
                id=point_id(name, op_index, slot),
                signature=name,
                subgraph=subgraph,
                op=op_index,
                output=slot,
                tensor=index,
                tensor_name=(tensor.Name() or b'').decode(),
                shape=shape,
                dtype=tensor_dtype(tensor.Type()),
                operator=operator,
                output_name=outputs.get(index),
                context_axes=tuple(
                    axis
                    for axis, extent in enumerate(shape)
                    if kv_length is not None and extent == kv_length
                ),
                reason=reason,
                layer=layer,
                layer_output=index in layer_outputs,
                layer_stem=stem if layer is not None else '',
            )
        )
  return tuple(signatures), tuple(points), kv_length


def layer_of(tensor_name: str, stem: str) -> int | None:
  """The layer a tensor's name places it in, if any.

  Args:
      tensor_name: The tensor's name in the model: the scope of the operator
        that writes it, or of each operator fused into it, `;` between.
      stem: What the model's layers are named by: layer N is the scope component
        `<stem>_<N>`.

  Returns:
      The layer number N, or None when the name places it in no layer.
  """
  match = _layer_component(stem).search(tensor_name.split(';')[0])
  return int(match.group(1)) if match else None


def _layer(
    tensor_name: str, stem: str, groups: Mapping[int, int], op: int
) -> int | None:
  """A tensor's layer: its name's, or else its operator's group's."""
  found = layer_of(tensor_name, stem)
  return groups.get(op) if found is None else found


def _layer_component(stem: str) -> re.Pattern[str]:
  """A layer's scope component, such as `.../layer_17/...` for `layer`."""
  return re.compile(rf'(?:^|/){re.escape(stem)}_(\d+)(?:/|$)')


def _layer_outputs(
    graph: tflite.SubGraph, stem: str, groups: Mapping[int, int]
) -> set[int]:
  """The tensors of a subgraph that are its layers' outputs.

  Of each layer that has a next one, the tensor of its last operator that
  an operator of the next layer reads, each operator in the layer of its
  first output (_layer()). Of a layer that has none, the last tensor named
  as most of those are, but for their layer numbers.

  Args:
      graph: The subgraph.
      stem: What the model's layers are named by, as for layer_of().
      groups: Operators' fallback layers, by operator index, as for _layer().

  Returns:
      The indices of the layers' output tensors.
  """
  number = re.compile(rf'{re.escape(stem)}_\d+')
  anonymous = f'{stem}_N'
  names: dict[int, str] = {}
  layers: dict[int, int | None] = {}
  readers: dict[int, list[int]] = collections.defaultdict(list)
  for op_index in range(graph.OperatorsLength()):
    op = graph.Operators(op_index)
    for slot in range(op.InputsLength()):
      if op.Inputs(slot) >= 0:
        readers[op.Inputs(slot)].append(op_index)
    written = [
        op.Outputs(slot)
        for slot in range(op.OutputsLength())
        if op.Outputs(slot) >= 0
    ]
    for tensor in written:
      names[tensor] = (graph.Tensors(tensor).Name() or b'').decode()
    layers[op_index] = (
        _layer(names[written[0]], stem, groups, op_index) if written else None
    )
  present = {layer for layer in layers.values() if layer is not None}
  last: dict[int, tuple[int, int]] = {}  # layer -> (op, tensor)
  for op_index in range(graph.OperatorsLength()):
    layer = layers[op_index]
    if layer is None or layer + 1 not in present:
      continue
    op = graph.Operators(op_index)
    for slot in range(op.OutputsLength()):
      tensor = op.Outputs(slot)
      if tensor >= 0 and any(
          layers[reader] == layer + 1 for reader in readers[tensor]
      ):
        last[layer] = (op_index, tensor)
  outputs = {tensor for _, tensor in last.values()}
  named = collections.Counter(
      number.sub(anonymous, names[tensor].split(';')[0]) for tensor in outputs
  )
  if not named:
    return outputs
  pattern = named.most_common(1)[0][0]
  ending: dict[int, int] = {}
  for op_index in range(graph.OperatorsLength()):
    layer = layers[op_index]
    if layer is None or layer + 1 in present:
      continue
    op = graph.Operators(op_index)
    for slot in range(op.OutputsLength()):
      tensor = op.Outputs(slot)
      if tensor >= 0 and (
          number.sub(anonymous, names[tensor].split(';')[0]) == pattern
      ):
        ending[layer] = tensor
  return outputs | set(ending.values())


def _runs(name: str, prefill_sizes: list[int]) -> str:
  """When LiteRT-LM runs the signature of a key."""
  if name == 'decode':
    return RUNS_ALWAYS
  if name == 'verify':
    return RUNS_SPECULATIVE
  match = _PREFILL.fullmatch(name)
  if match is None:
    return RUNS_UNKNOWN
  if int(match.group(1)) == prefill_sizes[0]:
    return RUNS_ALWAYS
  return RUNS_LONG_PROMPTS


def _kv_length(
    model: tflite.Model, keys: list[tflite.SignatureDef]
) -> int | None:
  """The sequence length of the model's K cache inputs, if they agree."""
  lengths = set()
  for definition in keys:
    graph = model.Subgraphs(definition.SubgraphIndex())
    for i in range(definition.InputsLength()):
      entry = definition.Inputs(i)
      if not _K_CACHE.fullmatch(entry.Name().decode()):
        continue
      tensor = graph.Tensors(entry.TensorIndex())
      if tensor.ShapeLength() > _K_SEQUENCE_AXIS:
        lengths.add(int(tensor.Shape(_K_SEQUENCE_AXIS)))
  return lengths.pop() if len(lengths) == 1 else None


def _operator(model: tflite.Model, op: tflite.Operator, code: int) -> str:
  """An operator's name: its builtin's, a custom code, or a composite's."""
  names = {
      value: name
      for name, value in vars(tflite.BuiltinOperator).items()
      if name.isupper()
  }
  name = names.get(code, f'operator {code}')
  if code == tflite.BuiltinOperator.CUSTOM:
    custom = model.OperatorCodes(op.OpcodeIndex()).CustomCode()
    return f'CUSTOM {custom.decode()}' if custom else name
  if (
      op.BuiltinOptions2Type()
      == tflite.BuiltinOptions2.StableHLOCompositeOptions
  ):
    table = op.BuiltinOptions2()
    options = tflite.StableHLOCompositeOptions()
    options.Init(table.Bytes, table.Pos)
    composite = options.Name()
    if composite:
      return f'{name} {composite.decode()}'
  return name
