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

"""A converter for the tests: Model Explorer's graph JSON, made in Python.

It writes what graph_converter.convert() writes, in-process and in a few
milliseconds, of the small .litertlm models the tests make (tap_fixtures):
of their prefill and decode section, whole, not of structure copies; laid out
as Model Explorer's converter lays out a TFLite model: one subgraph per
TFLite subgraph, a node per operator with its index as its id, its output
tensors' index, name and shape as its outputsMetadata, its namespace its
first output's first name; then a GraphInputs node of the subgraph's inputs
and a pseudo_const node per constant tensor read. It is not the converter:
the tests of the real one are graph_converter_test's.
"""

from collections.abc import Callable, Mapping
import json
import pathlib
from typing import Any

from model_debugger.server.graph import graph_converter
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import tap_scan
from model_debugger.server.tap import tflite_schema_generated as tflite


def converted(data: bytes) -> list[dict[str, Any]]:
  """The converter's JSON of a TFLite model: one graph, of every subgraph."""
  model = tflite.Model.GetRootAs(data, 0)
  keys = {}
  for i in range(model.SignatureDefsLength()):
    definition = model.SignatureDefs(i)
    keys[definition.SubgraphIndex()] = definition
  names = {
      value: name.lower()
      for name, value in vars(tflite.BuiltinOperator).items()
      if name.isupper()
  }
  subgraphs = []
  for index in range(model.SubgraphsLength()):
    graph = model.Subgraphs(index)
    subgraphs.append(_subgraph(model, graph, index, keys.get(index), names))
  return [{'label': '', 'subgraphs': subgraphs}]


def _subgraph(
    model: tflite.Model,
    graph: tflite.SubGraph,
    index: int,
    definition: tflite.SignatureDef | None,
    names: Mapping[int, str],
) -> dict[str, Any]:
  """Returns the Model Explorer graph of subgraph `index` of `model`."""
  count = graph.OperatorsLength()
  producer = {}
  for op_index in range(count):
    op = graph.Operators(op_index)
    for slot in range(op.OutputsLength()):
      producer[op.Outputs(slot)] = (str(op_index), str(slot))
  inputs = [graph.Inputs(i) for i in range(graph.InputsLength())]
  signature_names = {}
  if definition is not None:
    for i in range(definition.InputsLength()):
      entry = definition.Inputs(i)
      signature_names[entry.TensorIndex()] = entry.Name().decode()
  graph_inputs = str(count)
  for slot, tensor in enumerate(inputs):
    producer[tensor] = (graph_inputs, str(slot))
  constants = {}
  nodes = []
  for op_index in range(count):
    op = graph.Operators(op_index)
    edges = []
    for slot in range(op.InputsLength()):
      tensor = op.Inputs(slot)
      if tensor < 0:
        continue
      if tensor not in producer:
        constants.setdefault(tensor, str(count + 1 + len(constants)))
        source = (constants[tensor], '0')
      else:
        source = producer[tensor]
      edges.append({
          'sourceNodeId': source[0],
          'sourceNodeOutputId': source[1],
          'targetNodeInputId': str(slot),
      })
    outputs = [
        _output(graph, slot, op.Outputs(slot))
        for slot in range(op.OutputsLength())
        if op.Outputs(slot) >= 0
    ]
    first = _name(graph, op.Outputs(0)) if op.OutputsLength() else ''
    nodes.append({
        'id': str(op_index),
        'label': names.get(tap_scan.builtin(model, op), 'custom'),
        'namespace': first.split(';')[0],
        'attrs': [],
        'incomingEdges': edges,
        'inputsMetadata': [],
        'outputsMetadata': outputs,
        'subgraphIds': [],
    })
  graph_outputs = []
  for slot, tensor in enumerate(inputs):
    entry = _output(graph, slot, tensor)
    entry['attrs'].append(
        {'key': 'signature_name', 'value': signature_names.get(tensor, '')}
    )
    graph_outputs.append(entry)
  nodes.append({
      'id': graph_inputs,
      'label': model_graph.GRAPH_INPUTS,
      'namespace': model_graph.GRAPH_INPUTS,
      'attrs': [],
      'incomingEdges': [],
      'inputsMetadata': [],
      'outputsMetadata': graph_outputs,
  })
  for tensor, node_id in constants.items():
    nodes.append({
        'id': node_id,
        'label': model_graph.CONSTANT,
        'namespace': _name(graph, tensor),
        'attrs': [{'key': '__value', 'value': '0' * 100}],
        'incomingEdges': [],
        'inputsMetadata': [],
        'outputsMetadata': [_output(graph, 0, tensor)],
    })
  return {
      'id': (
          definition.SignatureKey().decode()
          if definition is not None
          else f'subgraph_{index}'
      ),
      'subgraphIndex': index,
      'nodes': nodes,
  }


def _name(graph: tflite.SubGraph, tensor: int) -> str:
  return (graph.Tensors(tensor).Name() or b'').decode()


def _output(graph: tflite.SubGraph, slot: int, tensor: int) -> dict[str, Any]:
  found = graph.Tensors(tensor)
  shape = ','.join(str(found.Shape(i)) for i in range(found.ShapeLength()))
  dtype = tap_scan.tensor_dtype(found.Type())
  return {
      'id': str(slot),
      'attrs': [
          {'key': 'tensor_index', 'value': str(tensor)},
          {'key': 'tensor_name', 'value': _name(graph, tensor)},
          {'key': 'tensor_shape', 'value': f'{dtype}[{shape}]'},
          {'key': 'quantization', 'value': 'q' * 100},
      ],
  }


class FakeConverter:
  """graph_converter.convert(), in-process; counts what it converts.

  Attributes:
      calls: How many conversions ran.
      timeouts: The timeout each was given.
      error: If set, what each conversion fails with; the log it leaves says so
        too.
      before: If set, called with `stopped` before each conversion.
  """

  def __init__(self) -> None:
    """Converts every model, until told to fail."""
    self.calls = 0
    self.timeouts: list[float] = []
    self.error: Exception | None = None
    self.before: Callable[[Callable[[], bool]], None] | None = None

  def __call__(
      self,
      model: pathlib.Path,
      subgraphs: Mapping[str, int],
      out: pathlib.Path,
      stopped: Callable[[], bool],
      timeout: float,
  ) -> None:
    """Writes the graphs of the signatures of the model at `model`."""
    self.calls += 1
    self.timeouts.append(timeout)
    if self.before is not None:
      self.before(stopped)
    if self.error is not None:
      graph_converter.log_path(out).write_text(
          f'converting {model.name}\n{self.error}\n'
      )
      raise self.error
    with litertlm_file.mapped(model) as container:
      section = container.prefill_decode()
      with container.bytes_of(section) as view:
        data = bytes(view)
    graphs = model_graph.signature_graphs(converted(data), subgraphs)
    out.write_text(json.dumps(graphs))
