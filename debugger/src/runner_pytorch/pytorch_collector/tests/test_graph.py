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

"""Tests for model explorer graph export in pytorch_collector."""

# pylint: disable=redefined-outer-name

import json

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
from pytorch_collector import to_model_explorer
from pytorch_collector import unused_ops
import torch
from torch import nn


class Block(nn.Module):
  """A single linear block with relu and residual connection."""

  def __init__(self):
    super().__init__()
    self.inner = nn.Linear(3, 3)

  def forward(self, x):
    return torch.relu(self.inner(x)) + x


class Repeated(nn.Module):
  """A module with repeated and shared block calls."""

  def __init__(self):
    super().__init__()
    self.layers = nn.ModuleList([Block()])
    self.shared = Block()
    self.kept = None

  def forward(self, x):
    hidden = self.shared(self.shared(self.layers[0](x)))
    if self.kept is not None:
      hidden = hidden + self.kept
    self.kept = hidden
    return hidden


@pytest.fixture(scope="module")
def capture():
  """Fixture providing a capture of Repeated module forwards."""
  model = Repeated().eval()
  with collect(model) as result:
    first = model(torch.ones(1, 3))
    model.outside = first.abs()
    model(torch.ones(1, 3))
  return result


def _nodes(graph):
  """Returns a mapping from node id to node."""
  return {node["id"]: node for node in graph["nodes"]}


def test_the_collection_survives_json_and_has_one_graph_per_forward(capture):
  """Tests collection JSON roundtrip and one graph per forward pass."""
  collection = to_model_explorer(capture, label="repeated")

  assert json.loads(json.dumps(collection)) == collection
  assert collection["label"] == "repeated"
  assert [graph["id"] for graph in collection["graphs"]] == [
      "forward_0",
      "forward_1",
  ]
  for forward_id, graph in enumerate(collection["graphs"]):
    operators = {
        node["id"]
        for node in graph["nodes"]
        if node["namespace"] not in ("inputs", "outputs")
    }
    assert operators == {
        str(op.call_id) for op in capture.ops if op.forward_id == forward_id
    }
    assert len(_nodes(graph)) == len(graph["nodes"])


def test_every_link_stays_inside_its_graph_and_names_real_ports(capture):
  """Tests links stay within graphs and reference valid ports."""
  for graph in to_model_explorer(capture)["graphs"]:
    nodes = _nodes(graph)
    links = []
    for node in graph["nodes"]:
      for link in node["incomingEdges"]:
        links.append((node, link))
    assert links
    for node, link in links:
      source = nodes[link["sourceNodeId"]]
      assert link["sourceNodeOutputId"] in {
          output["id"] for output in source["outputsMetadata"]
      }
      assert int(link["targetNodeInputId"]) >= 0


def test_module_paths_become_namespaces_and_repeated_calls_are_numbered(
    capture,
):
  """Tests module hierarchy namespaces and numbering of repeated calls."""
  graph = to_model_explorer(capture, forwards=[0])["graphs"][0]
  namespaces = [
      node["namespace"]
      for node in graph["nodes"]
      if node["label"] == "aten.addmm"
  ]

  assert namespaces == ["layers/0/inner", "shared#0/inner", "shared#1/inner"]
  residuals = [
      node["namespace"]
      for node in graph["nodes"]
      if node["label"] == "aten.add"
  ]
  assert residuals == ["layers/0", "shared#0", "shared#1"]


def test_operator_nodes_describe_shapes_arguments_and_unlinked_inputs(capture):
  """Tests operator node attributes, tensor shapes, and input links."""
  graph = to_model_explorer(capture, forwards=[0])["graphs"][0]
  addmm = next(node for node in graph["nodes"] if node["label"] == "aten.addmm")
  attrs = {item["key"]: item["value"] for item in addmm["attrs"]}

  assert (
      attrs["op"] == "aten.addmm.default"
      and attrs["module"] == "layers.0.inner"
  )
  assert attrs["args.0"] == "param: layers.0.inner.bias float32[3]"
  assert addmm["outputsMetadata"] == [
      {"id": "0", "attrs": [{"key": "tensor_shape", "value": "float32[1, 3]"}]}
  ]
  # The bias is an attribute; the input and the transposed weight are links.
  assert sorted(
      link["targetNodeInputId"] for link in addmm["incomingEdges"]
  ) == ["1", "2"]
  root_input = _nodes(graph)["args.0"]
  assert (root_input["namespace"], root_input["label"]) == ("inputs", "args.0")


def test_what_another_forward_computed_is_an_input_of_the_reading_graph(
    capture,
):
  """Tests cross-forward tensor dependencies are represented as graph inputs."""
  first, second = to_model_explorer(capture)["graphs"]
  kept = [node for node in second["nodes"] if node["id"].startswith("kept:")]

  assert (
      len(kept) == 1
      and kept[0]["namespace"] == "inputs"
      and not kept[0]["incomingEdges"]
  )
  attrs = {item["key"]: item["value"] for item in kept[0]["attrs"]}
  assert (
      attrs["source_graph"] == "forward_0"
      and attrs["source_output"] == "output"
  )
  # The model kept this tensor itself, so no argument path names it.
  assert kept[0]["id"] == f"kept:forward_0:{attrs['source_node']}:output"
  assert _nodes(first)[attrs["source_node"]]["label"] == "aten.add"
  reader = next(
      node
      for node in second["nodes"]
      if any(
          link["sourceNodeId"] == kept[0]["id"]
          for link in node["incomingEdges"]
      )
  )
  assert reader["label"] == "aten.add" and not reader["namespace"]


class Stateful(nn.Module):
  """Stateful module with an optional state argument."""

  def forward(self, x, state=None):
    return x * 2 if state is None else x * state


def test_a_tensor_passed_back_in_is_named_by_the_path_it_came_in_under():
  """Tests tensors passed back across forwards retain input path names."""
  model = Stateful()
  with collect(model) as capture:
    model(torch.ones(2), state=model(torch.ones(2)))
  second = to_model_explorer(capture)["graphs"][1]
  inputs = {
      node["id"]: node
      for node in second["nodes"]
      if node["namespace"] == "inputs"
  }

  assert inputs.keys() == {"args.0", "kwargs.state"}
  assert not inputs["args.0"]["attrs"]
  attrs = {
      item["key"]: item["value"] for item in inputs["kwargs.state"]["attrs"]
  }
  assert attrs == {
      "source_graph": "forward_0",
      "source_node": str(capture.ops[0].call_id),
      "source_output": "output",
  }


def test_forwards_can_be_selected_and_must_exist(capture):
  """Tests selecting specific forward passes and bounds validation."""
  assert [
      graph["id"]
      for graph in to_model_explorer(capture, forwards=[1])["graphs"]
  ] == ["forward_1"]
  with pytest.raises(ValueError, match=r"forwards \[5\]; available: \[0, 1\]"):
    to_model_explorer(capture, forwards=[0, 5])


def test_the_caller_names_forwards_it_understands(capture):
  """Tests custom naming of forward passes in exported collections."""
  collection = to_model_explorer(capture, names={0: "prefill", 1: "decode_1"})

  assert [graph["id"] for graph in collection["graphs"]] == [
      "prefill",
      "decode_1",
  ]
  named = {"key": "source_graph", "value": "prefill"}
  assert any(
      named in node["attrs"] for node in collection["graphs"][1]["nodes"]
  )
  # A selected graph still names an unselected source forward the caller's way.
  only_decode = to_model_explorer(capture, forwards=[1], names={0: "prefill"})[
      "graphs"
  ][0]
  assert only_decode["id"] == "forward_1"
  assert any(named in node["attrs"] for node in only_decode["nodes"])


@pytest.mark.parametrize(
    "names, message",
    [
        ({7: "later"}, r"Names were given for forwards \[7\]"),
        ({0: "same", 1: "same"}, "must be unique"),
        ({0: "forward_1"}, "must be unique"),
        ({0: "a/b"}, "without '/'"),
        ({0: ""}, "non-empty"),
    ],
)
def test_forward_names_must_be_usable_graph_names(capture, names, message):
  """Tests validation of custom forward names."""
  with pytest.raises(ValueError, match=message):
    to_model_explorer(capture, names=names)


def test_an_unfinished_capture_cannot_be_exported():
  """Tests that exporting an unfinished capture raises CaptureError."""
  with pytest.raises(
      CaptureError, match="readable only after successful scope exit"
  ):
    to_model_explorer(collect(nn.Linear(1, 1)))


class Wasteful(nn.Module):
  """Module with unused intermediate operations and buffer writes."""

  def __init__(self):
    super().__init__()
    self.register_buffer("seen", torch.zeros(2))

  def forward(self, x, state=None):
    _ = (x + 1).unsqueeze(0)  # Computed, then dropped.
    self.seen[:1].add_(1)  # Read by nobody, but it writes.
    hidden = x * 2 if state is None else x * state
    return {"hidden": hidden, "same": x}


@pytest.fixture(scope="module")
def wasteful():
  """Fixture providing a capture of Wasteful module forwards."""
  model = Wasteful()
  with collect(model) as result:
    model(torch.ones(2), state=model(torch.ones(2))["hidden"])
  return result


def test_every_returned_tensor_names_its_source(wasteful):
  """Tests that returned tensors attribute their source operator or input."""
  first = [edge for edge in wasteful.outputs if edge.forward_id == 0]

  assert [(edge.call_id, edge.path, edge.kind) for edge in first] == [
      (None, ("output", "hidden"), "op"),
      (None, ("output", "same"), "input"),
  ]
  producer = next(
      op for op in wasteful.ops if op.call_id == first[0].source_call_id
  )
  assert producer.op == "aten.mul.Tensor" and first[1].source_path == (
      "args",
      0,
  )


def test_returned_tensors_are_output_nodes_fed_by_their_sources(wasteful):
  """Tests that returned tensors create output nodes fed by their sources."""
  graph = to_model_explorer(wasteful)["graphs"][0]
  outputs = {
      node["id"]: node
      for node in graph["nodes"]
      if node["namespace"] == "outputs"
  }

  assert outputs.keys() == {"output.hidden", "output.same"}
  assert (
      _nodes(graph)[
          outputs["output.hidden"]["incomingEdges"][0]["sourceNodeId"]
      ]["label"]
      == "aten.mul"
  )
  assert outputs["output.same"]["incomingEdges"][0]["sourceNodeId"] == "args.0"


def test_unused_calls_reach_no_output_and_a_write_is_never_one_of_them(
    wasteful,
):
  """Tests that unused operations reach no outputs and writes are excluded."""
  ops = {op.call_id: op.op for op in wasteful.ops}
  unused = unused_ops(wasteful)

  assert unused.keys() == {0, 1}
  for calls in unused.values():
    assert [ops[call_id] for call_id in calls] == [
        "aten.add.Tensor",
        "aten.unsqueeze.default",
    ]
  # The first forward's product is used because the second forward's output
  # reads it.
  assert wasteful.outputs[0].source_call_id not in unused[0]


def test_hiding_unused_calls_removes_only_them_and_keeps_every_link_inside(
    wasteful,
):
  """Tests hiding unused operations prunes only them and preserves links."""
  shown, hidden = (
      to_model_explorer(wasteful, hide_unused=flag)["graphs"]
      for flag in (False, True)
  )
  for forward_id, (full, reduced) in enumerate(zip(shown, hidden)):
    removed = _nodes(full).keys() - _nodes(reduced).keys()
    assert removed == {
        str(call_id) for call_id in unused_ops(wasteful)[forward_id]
    }
    assert "aten.add_" in {node["label"] for node in reduced["nodes"]}
    for node in reduced["nodes"]:
      assert all(
          link["sourceNodeId"] in _nodes(reduced)
          for link in node["incomingEdges"]
      )


class PassesThrough(nn.Module):
  """Module that returns one input untouched."""

  def forward(self, x, untouched):
    return x + 1, untouched


def test_a_tensor_returned_as_it_came_adds_no_input_node_and_can_be_hidden():
  """Tests that passthrough tensors do not add inputs and can be pruned."""
  model = PassesThrough()
  with collect(model) as capture:
    model(torch.ones(2), torch.zeros(3))
  shown, hidden = (
      to_model_explorer(capture, hide_unused=flag)["graphs"][0]
      for flag in (False, True)
  )

  assert [
      node["id"] for node in shown["nodes"] if node["namespace"] == "inputs"
  ] == ["args.0"]
  passed = _nodes(shown)["output.1"]
  assert not passed["incomingEdges"] and passed["attrs"] == [
      {"key": "source", "value": "input args.1"}
  ]
  assert [
      node["id"] for node in hidden["nodes"] if node["namespace"] == "outputs"
  ] == ["output.0"]
