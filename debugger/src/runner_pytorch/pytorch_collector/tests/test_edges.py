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

"""Tests for pytorch_collector edge tracking and DAG construction."""

# pylint: disable=protected-access

import gc
import json
import weakref

import pytest
from pytorch_collector import _provenance
from pytorch_collector import CaptureError
from pytorch_collector import collect
from pytorch_collector import Edge
from pytorch_collector import Op
import torch
from torch import nn


def _ops(capture, name, forward_id=None):
  return [
      op
      for op in capture.ops
      if op.op == name and (forward_id is None or op.forward_id == forward_id)
  ]


def _edge(capture, op: Op, path) -> Edge:
  matches = [
      e for e in capture.edges if e.call_id == op.call_id and e.path == path
  ]
  assert len(matches) == 1
  return matches[0]


def _assert_source(edge: Edge, producer: Op, path=("output",)):
  assert (
      edge.kind,
      edge.source_forward_id,
      edge.source_call_id,
      edge.source_path,
  ) == (
      "op",
      producer.forward_id,
      producer.call_id,
      path,
  )


def test_operator_outputs_are_the_objects_later_operators_and_hooks_receive():
  """Tests operator outputs are the objects later operators/hooks receive."""
  model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()

  with collect(model) as capture:
    model(torch.ones(2, 4))

  first, second = _ops(capture, "aten.addmm.default")
  (relu,) = _ops(capture, "aten.relu.default")
  assert (first.module, relu.module, second.module) == ("0", "1", "2")
  assert _edge(capture, first, ("args", 1)).kind == "input"
  assert _edge(capture, first, ("args", 1)).source_path == ("args", 0)
  _assert_source(_edge(capture, relu, ("args", 0)), first)
  _assert_source(_edge(capture, second, ("args", 1)), relu)


def test_every_operator_tensor_input_has_exactly_one_edge():
  """Tests that every operator tensor input has exactly one edge."""
  model = nn.Sequential(
      nn.Linear(4, 4), nn.LayerNorm(4), nn.Softmax(dim=-1)
  ).eval()

  with collect(model) as capture:
    model(torch.ones(2, 4))

  ports = []
  for op in capture.ops:
    for port in op.inputs:
      ports.append((op.call_id, port.path))
  assert ports
  assert sorted(ports) == sorted(
      (edge.call_id, edge.path) for edge in capture.edges
  )
  # Operators without snapshots, such as a parameter transpose, are still
  # listed.
  recorded = {
      record.call_id for record in capture.records if record.op is not None
  }
  assert {op.call_id for op in capture.ops} > recorded


class Residual(nn.Module):
  """Residual block module for testing skip connections."""

  def __init__(self):
    super().__init__()
    self.inner = nn.Linear(3, 3)

  def forward(self, x):
    hidden = self.inner(x)
    return hidden + x, hidden * 2


def test_residual_inputs_and_fan_out_share_one_producer():
  """Tests that residual inputs and fan-outs share one producer."""
  model = Residual().eval()

  with collect(model) as capture:
    model(torch.ones(1, 3))

  (addmm,) = _ops(capture, "aten.addmm.default")
  (add,) = _ops(capture, "aten.add.Tensor")
  (mul,) = _ops(capture, "aten.mul.Tensor")
  assert not add.module and not mul.module
  _assert_source(_edge(capture, add, ("args", 0)), addmm)
  assert _edge(capture, add, ("args", 1)).kind == "input"
  _assert_source(_edge(capture, mul, ("args", 0)), addmm)


class SharedTwice(nn.Module):
  """Module invoking the same submodule twice."""

  def __init__(self):
    super().__init__()
    self.block = nn.Linear(3, 3)

  def forward(self, x):
    return self.block(self.block(x))


def test_a_shared_module_called_twice_chains_its_own_calls():
  model = SharedTwice().eval()

  with collect(model) as capture:
    model(torch.ones(1, 3))

  first, second = _ops(capture, "aten.addmm.default")
  assert first.module == second.module == "block"
  assert first.module_call_id != second.module_call_id
  _assert_source(_edge(capture, second, ("args", 1)), first)


class Chunked(nn.Module):
  """Module chunking input into multiple tensors."""

  def forward(self, x):
    left, right = x.chunk(2, dim=-1)
    return torch.softmax(left + right, dim=-1)


def test_multiple_outputs_are_identified_by_path_and_scalars_are_described():
  """Tests multi-output operator identification and scalar descriptions."""
  model = Chunked().eval()

  with collect(model) as capture:
    model(torch.ones(2, 6))

  (split,) = [op for op in capture.ops if op.op.startswith("aten.split")]
  (add,) = _ops(capture, "aten.add.Tensor")
  (softmax,) = _ops(capture, "aten._softmax.default")
  _assert_source(_edge(capture, add, ("args", 0)), split, ("output", 0))
  _assert_source(_edge(capture, add, ("args", 1)), split, ("output", 1))
  assert [(port.path, port.shape, port.dtype) for port in split.outputs] == [
      (("output", 0), (2, 3), torch.float32),
      (("output", 1), (2, 3), torch.float32),
  ]
  assert softmax.inputs[0].shape == (2, 3)
  assert dict(softmax.arguments) == {"dim": "-1", "half_to_float": "False"}
  json.dumps([op.arguments for op in capture.ops])


class Stateful(nn.Module):
  """Module with internal state across forward calls."""

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 2)
    self.carried = None
    self.outside = None
    self.preexisting = torch.full((2,), 3.0)

  def forward(self, x):
    hidden = self.linear(x) + self.preexisting + torch.tensor(1.0)
    if self.outside is not None:
      hidden = hidden - self.outside
    if self.carried is not None:
      hidden = hidden * self.carried
    self.carried = hidden
    return hidden


def test_sources_outside_the_current_forward_are_distinguished():
  """Tests that sources outside the current forward are distinguished."""
  model = Stateful().eval()

  with collect(model) as capture:
    first = model(torch.ones(1, 2))
    model.outside = first.abs()  # A visible operator between root calls.
    model(torch.ones(1, 2))

  add_existing, add_literal = _ops(capture, "aten.add.Tensor", 0)
  assert _edge(capture, add_existing, ("args", 1)).kind == "constant"
  (lift,) = _ops(capture, "aten.lift_fresh.default", 0)
  assert _edge(capture, lift, ("args", 0)).kind == "constant"
  _assert_source(_edge(capture, add_literal, ("args", 1)), lift)

  (sub,) = _ops(capture, "aten.sub.Tensor", 1)
  assert _edge(capture, sub, ("args", 1)) == Edge(
      1, sub.call_id, ("args", 1), "external"
  )
  (mul,) = _ops(capture, "aten.mul.Tensor", 1)
  carried = _edge(capture, mul, ("args", 1))
  assert (carried.kind, carried.source_forward_id, carried.source_call_id) == (
      "op",
      0,
      add_literal.call_id,
  )


def test_a_cache_tensor_passed_back_as_input_keeps_its_operator_source():
  """Tests that cache tensors passed back retain their operator source."""
  model = nn.Linear(2, 2).eval()

  with collect(model) as capture:
    first = model(torch.ones(1, 2))
    model(first)

  first_addmm, second_addmm = _ops(capture, "aten.addmm.default")
  _assert_source(_edge(capture, second_addmm, ("args", 1)), first_addmm)


def test_ops_and_edges_share_the_atomic_result_of_records():
  """Tests that ops and edges share the atomic result of records."""
  model = nn.Linear(2, 2).eval()
  capture = collect(model)

  with pytest.raises(CaptureError, match="ops are readable only after"):
    _ = capture.ops
  with pytest.raises(RuntimeError, match="boom"):
    with capture:
      model(torch.ones(1, 2))
      with pytest.raises(CaptureError, match="edges are readable only after"):
        _ = capture.edges
      raise RuntimeError("boom")
  for name in ("records", "ops", "edges"):
    with pytest.raises(CaptureError):
      getattr(capture, name)

  with collect(model) as empty:
    pass
  assert not empty.ops and not empty.edges


def test_tracking_does_not_retain_tensors_and_forgets_them_when_they_die():
  """Tests that tracking does not retain tensors after garbage collection."""
  provenance = _provenance.Provenance()
  tensor = torch.ones(3)
  alive = weakref.ref(tensor)
  provenance.outputs(0, 7, [(("output",), tensor)])
  assert (
      provenance.inputs(0, 8, [(("args", 0), tensor)])[1][0].source_call_id == 7
  )

  del tensor
  gc.collect()

  assert alive() is None
  assert len(provenance._producers) == 0
  # A new object cannot inherit the dead tensor's source through a reused id().
  assert (
      provenance.inputs(0, 9, [(("args", 0), torch.ones(3))])[1][0].kind
      == "constant"
  )


def test_a_tracking_failure_voids_the_whole_capture(monkeypatch):
  """Tests that a tracking failure voids the entire capture."""

  def broken(self, *args):
    raise ValueError("tracking bug")

  monkeypatch.setattr(_provenance.Provenance, "outputs", broken)
  model = nn.Linear(2, 2).eval()
  capture = collect(model)

  with pytest.raises(ValueError, match="tracking bug"):
    with capture:
      model(torch.ones(1, 2))
  with pytest.raises(CaptureError):
    _ = capture.records
  assert not any(model._forward_hooks) and not any(model._forward_pre_hooks)


class Registered(nn.Module):
  """Module with registered parameters and buffers for testing attribution."""

  def __init__(self):
    super().__init__()
    self.embed = nn.Embedding(5, 4)
    self.norm = nn.LayerNorm(4)
    self.head = nn.Linear(4, 5, bias=False)
    self.head.weight = self.embed.weight  # Tied: one tensor, two names.
    self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)
    self.register_buffer("scale", torch.full((4,), 2.0))
    self.register_buffer("offset", torch.zeros(4), persistent=False)

  def forward(self, tokens):
    hidden = self.norm(self.embed(tokens)) * self.frozen
    return self.head(hidden * self.scale + self.offset)


def test_parameters_and_buffers_are_attributed_by_their_first_registered_name():
  """Tests that parameters and buffers are attributed by first registered name."""
  model = Registered().eval()

  with collect(model) as capture:
    model(torch.tensor([[1, 2]]))

  (embedding,) = _ops(capture, "aten.embedding.default")
  assert _edge(capture, embedding, ("args", 0)) == Edge(
      0,
      embedding.call_id,
      ("args", 0),
      "param",
      name="embed.weight",
  )
  (norm,) = _ops(capture, "aten.native_layer_norm.default")
  assert [_edge(capture, norm, ("args", i)).name for i in (2, 3)] == [
      "norm.weight",
      "norm.bias",
  ]
  (transpose,) = _ops(capture, "aten.t.default")
  assert _edge(capture, transpose, ("args", 0)).name == "embed.weight"
  # The transposed weight is an ordinary operator output, not a parameter.
  (matmul,) = _ops(capture, "aten.mm.default")
  _assert_source(_edge(capture, matmul, ("args", 1)), transpose)
  by_name = {
      edge.name: edge.kind for edge in capture.edges if edge.name is not None
  }
  assert by_name == {
      "embed.weight": "param",
      "norm.weight": "param",
      "norm.bias": "param",
      "frozen": "param",
      "scale": "buffer",
      "offset": "buffer",
  }
  assert not [edge for edge in capture.edges if edge.kind == "constant"]


class Counter(nn.Module):
  """Module with a persistent counter buffer."""

  def __init__(self):
    super().__init__()
    self.register_buffer("steps", torch.zeros(1))

  def forward(self, x):
    self.steps.add_(1)
    return x * self.steps


def test_a_buffer_written_by_an_operator_reports_that_write():
  """Tests that a buffer written by an operator reports that write."""
  model = Counter().eval()

  with collect(model) as capture:
    model(torch.ones(2))

  (add,) = _ops(capture, "aten.add_.Tensor")
  (mul,) = _ops(capture, "aten.mul.Tensor")
  assert _edge(capture, add, ("args", 0)) == Edge(
      0, add.call_id, ("args", 0), "buffer", name="steps"
  )
  _assert_source(_edge(capture, mul, ("args", 1)), add)


class Writes(nn.Module):
  """Wrapper module executing arbitrary write bodies."""

  def __init__(self, body):
    super().__init__()
    self.body = body

  def forward(self, x):
    return self.body(x)


def _capture(body, *inputs):
  """Helper to run body within a Writes module and collect capture."""
  model = Writes(body).eval()
  with collect(model) as capture:
    model(*inputs)
  return capture


@pytest.mark.parametrize(
    "write, name",
    [
        (lambda y: y.add_(1), "aten.add_.Tensor"),
        (lambda y: y.copy_(torch.zeros(3)), "aten.copy_.default"),
        (
            lambda y: y.index_put_((torch.tensor([0]),), torch.tensor([9.0])),
            "aten.index_put_.default",
        ),
        (lambda y: torch.mul(y, 2, out=y), "aten.mul.out"),
    ],
)
def test_a_write_makes_its_call_the_producer_of_the_written_tensor(write, name):
  """Tests that a write makes its operator call the producer of the tensor."""

  def body(x):
    y = x * 1
    write(y)
    return y + 1

  capture = _capture(body, torch.ones(3))

  (writer,) = _ops(capture, name)
  reader = _ops(capture, "aten.add.Tensor")[-1]
  _assert_source(_edge(capture, reader, ("args", 0)), writer)


def test_written_arguments_that_are_not_returned_become_output_ports():
  """Tests that in-place written arguments become output ports."""

  def body(x):
    parts = [x * 1, x * 2]
    torch._foreach_add_(parts, 1)
    return parts[0] + parts[1]

  capture = _capture(body, torch.ones(3))

  (foreach,) = _ops(capture, "aten._foreach_add_.Scalar")
  add = _ops(capture, "aten.add.Tensor")[-1]
  assert [port.path for port in foreach.outputs] == [
      ("args", 0, 0),
      ("args", 0, 1),
  ]
  _assert_source(_edge(capture, add, ("args", 0)), foreach, ("args", 0, 0))
  _assert_source(_edge(capture, add, ("args", 1)), foreach, ("args", 0, 1))


def test_every_operator_source_names_an_output_port_of_its_producer():
  """Tests that every operator source names an output port of its producer."""

  def body(x):
    y = x * 1
    y[0] = 5.0
    torch._foreach_mul_([y], 2)
    return y.sum()

  capture = _capture(body, torch.ones(3))

  ports = set()
  for op in capture.ops:
    for port in op.outputs:
      ports.add((op.call_id, port.path))
  sources = {
      (e.source_call_id, e.source_path) for e in capture.edges if e.kind == "op"
  }
  assert sources and sources <= ports


def test_a_write_through_a_view_updates_the_base_but_not_its_other_views():
  """Tests that a write through a view updates the base tensor."""

  def body(x):
    base = x * 1
    earlier_view = base[1:]
    base[0] = 5.0  # aten.select, then aten.copy_ into that view
    return base + 1, earlier_view + 1

  capture = _capture(body, torch.ones(3))

  (mul,) = _ops(capture, "aten.mul.Tensor")
  (fill,) = _ops(capture, "aten.copy_.default") or _ops(
      capture, "aten.fill_.Scalar"
  )
  (slice_,) = _ops(capture, "aten.slice.Tensor")
  read_base, read_view = _ops(capture, "aten.add.Tensor")
  _assert_source(
      _edge(capture, fill, ("args", 0)), _ops(capture, "aten.select.int")[0]
  )
  _assert_source(_edge(capture, read_base, ("args", 0)), fill)
  # Known limit: an existing sibling view still reports the operator that
  # made it.
  _assert_source(_edge(capture, read_view, ("args", 0)), slice_)
  _assert_source(_edge(capture, slice_, ("args", 0)), mul)


class SlicedCache(nn.Module):
  """Module slicing a cache buffer during forward."""

  def __init__(self):
    super().__init__()
    self.register_buffer("cache", torch.zeros(4, 2))

  def forward(self, x, position):
    self.cache[position] = x
    return self.cache.sum(0)


def test_a_sliced_write_into_a_buffer_carries_into_the_next_forward():
  """Tests that a sliced write into a buffer carries into the next forward."""
  model = SlicedCache().eval()

  with collect(model) as capture:
    model(torch.ones(2), 0)
    model(torch.ones(2), 1)

  first_write, second_write = _ops(capture, "aten.copy_.default")
  first_select, second_select = _ops(capture, "aten.select.int")
  first_sum, second_sum = _ops(capture, "aten.sum.dim_IntList")
  assert _edge(capture, first_select, ("args", 0)).name == "cache"
  _assert_source(_edge(capture, first_sum, ("args", 0)), first_write)
  # Forward 1 slices the buffer that forward 0 wrote, then reads its own write.
  _assert_source(_edge(capture, second_select, ("args", 0)), first_write)
  _assert_source(_edge(capture, second_sum, ("args", 0)), second_write)
