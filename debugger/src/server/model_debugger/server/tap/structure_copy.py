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

"""A model's structure without its weights: what a graph is drawn from.

Model Explorer's converter (graph/graph_converter) reads a whole TFLite
model into memory and unpacks it, weights and all, though a graph shows
none of them. A structure copy is a TFLite model of one subgraph of the
prefill and decode section, everything the converter draws kept as it is:

- the subgraph itself, whole: its tensors, with their type, shape, shape
  signature, buffer index, quantization and sparsity, its operators, with
  their options (large custom options moved inline, as custom options), its
  inputs and outputs;
- the signature defs that run it, and no other;
- every other subgraph, by its index and name, but with no tensors,
  operators, inputs or outputs: an operator names its decomposition or
  body by those, and the server keeps no graph of them;
- every buffer, by its index: the bytes of one of at most KEPT_BYTES kept,
  inline; one larger emptied. Those are the weights; a smaller one holds a
  shape, a scalar, a norm's scale, which a graph shows as its value.
- the operator codes, metadata, description and version.

Its graph is the section's but for the values of the constants whose
buffers were emptied: the converter writes none for them. Converting Gemma
4 E2B's four signatures so, one copy at a time, takes 0.7 GB at most where
its whole section took 4.7 GB; E4B's 0.9 GB where it took 8.1 GB.
"""

from collections.abc import Iterable, Iterator
import copy

from model_debugger.server.tap import tap_model
from model_debugger.server.tap import tflite_schema_generated as tflite

# What a structure copy is: raised whenever this module writes another copy
# of the same section, so that a graph converted from an older one is not
# reused (graph_converter.identity()).
FORMAT_VERSION = 1
# The largest buffer a copy keeps the bytes of.
KEPT_BYTES = 64 * 1024


def identity() -> str:
  """What makes a structure copy, for the identity of what it converts."""
  return (
      f'structure copy {FORMAT_VERSION}, buffers over {KEPT_BYTES} bytes '
      'emptied'
  )


def copies(
    section: 'memoryview[int]',
    subgraphs: Iterable[int],
    kept_bytes: int = KEPT_BYTES,
) -> Iterator[tuple[int, bytes]]:
  """The structure copy of each of a TFLite model's subgraphs.

  The model is read once; each copy is made as it is asked for, and
  nothing it yields refers to `section`.

  Args:
      section: The TFLite model.
      subgraphs: The subgraphs, by index; one that repeats is copied once.
      kept_bytes: The largest buffer whose bytes are kept.

  Yields:
      (subgraph, its copy).

  Raises:
      ValueError: It is not a TFLite model, keeps weights in other files,
        has no such subgraph, or a payload lies outside it.
  """
  if bytes(section[4:8]) != b'TFL3':
    raise ValueError('The section is not a TFLite model.')
  model = tflite.ModelT.InitFromObj(tflite.Model.GetRootAs(section, 0))
  if model.externalBuffers or model.externalBufferGroups:
    raise ValueError(
        'The model keeps weights in other files, which this server does '
        'not read.'
    )
  model.buffers = [
      _structure_buffer(section, buffer, kept_bytes)
      for buffer in model.buffers or []
  ]
  graphs = list(model.subgraphs or [])
  definitions = list(model.signatureDefs or [])
  empty = [_emptied(graph) for graph in graphs]
  done = set()
  for index in subgraphs:
    if index in done:
      continue
    done.add(index)
    if not 0 <= index < len(graphs):
      raise ValueError(f'The model has no subgraph {index}.')
    model.subgraphs = [
        _inlined(section, graph) if i == index else empty[i]
        for i, graph in enumerate(graphs)
    ]
    model.signatureDefs = [
        definition
        for definition in definitions
        if definition.subgraphIndex == index
    ]
    yield index, tap_model.pack(model)


def _structure_buffer(
    section: 'memoryview[int]', buffer: tflite.BufferT, kept_bytes: int
) -> tflite.BufferT:
  """A buffer as a copy keeps it: inline, or emptied if too large."""
  if buffer.offset > 1:
    if buffer.size < 0 or buffer.offset + buffer.size > len(section):
      raise ValueError('A weight of the model lies outside its section.')
    size = buffer.size
    data = section[buffer.offset : buffer.offset + size]
  else:
    data = buffer.data
    size = 0 if data is None else len(data)
  kept = tap_model.AlignedBuffer()
  if 0 < size <= kept_bytes:
    kept.data = bytes(memoryview(data).cast('B'))
  return kept


def _emptied(graph: tflite.SubGraphT) -> tflite.SubGraphT:
  """A subgraph with its index's place and name, and nothing else."""
  emptied = copy.copy(graph)
  emptied.tensors = []
  emptied.operators = []
  emptied.inputs = []
  emptied.outputs = []
  return emptied


def _inlined(
    section: 'memoryview[int]', graph: tflite.SubGraphT
) -> tflite.SubGraphT:
  """A subgraph whose large custom options are inline: its copy's."""
  large = [
      op for op in graph.operators or [] if op.largeCustomOptionsOffset > 1
  ]
  if not large:
    return graph
  inlined = copy.copy(graph)
  operators = []
  for op in graph.operators:
    if op.largeCustomOptionsOffset > 1:
      begin = op.largeCustomOptionsOffset
      end = begin + op.largeCustomOptionsSize
      if end > len(section):
        raise ValueError(
            'The custom options of an operator lie outside the section.'
        )
      op = copy.copy(op)
      op.customOptions = bytes(section[begin:end])
      op.largeCustomOptionsOffset = op.largeCustomOptionsSize = 0
    operators.append(op)
  inlined.operators = operators
  return inlined
