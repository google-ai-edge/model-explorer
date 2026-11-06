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

"""Tests for PyTorch dispatch collector mode and tracing."""

# pylint: disable=protected-access

import collections

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils._python_dispatch import TorchDispatchMode


def ops(capture):
  return [row for row in capture.records if row.op is not None]


def op_calls(capture):
  return list(dict.fromkeys((row.call_id, row.op) for row in ops(capture)))


class RecordingMode(TorchDispatchMode):

  def __init__(self):
    super().__init__()
    self.names = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.names.append(str(func))
    return func(*args, **(kwargs or {}))


def test_visible_operations_share_module_identity_and_preserve_values():
  """Tests visible operations share module identity and preserve values."""

  class Math(nn.Module):

    def forward(self, x):
      product = torch.mm(x, x.T)
      summed = product + 2
      normalized = torch.softmax(summed, dim=-1)
      return normalized.view(-1) + torch.ones(4)

  m = nn.Sequential(Math())
  x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
  baseline = m(x)
  with collect(m) as c:
    actual = m(x)
  torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
  names = [name for _, name in op_calls(c)]
  assert names == [
      'aten.permute.default',
      'aten.mm.default',
      'aten.add.Tensor',
      'aten._softmax.default',
      'aten.view.default',
      'aten.ones.default',
      'aten.add.Tensor',
  ]
  module_rows = [row for row in c.records if row.op is None]
  child_call_id = next(row.call_id for row in module_rows if row.module == '0')
  assert all(
      row.module == '0'
      and row.module_call_id == child_call_id
      and row.forward_id == 0
      for row in ops(c)
  )
  ids = {row.call_id for row in module_rows}
  assert not ids.intersection(row.call_id for row in ops(c))
  mm_output = next(
      row.tensor
      for row in ops(c)
      if row.op == 'aten.mm.default' and row.edge == 'out'
  )
  torch.testing.assert_close(
      mm_output, torch.tensor([[5.0, 11.0], [11.0, 25.0]])
  )


@pytest.mark.parametrize('use_out', [False, True])
def test_inplace_and_out_preserve_before_values_and_original_aliases(use_out):
  """Tests in-place and out= preserve before-values and original aliases."""

  class Mutate(nn.Module):

    def forward(self, x):
      return torch.add(x, 3, out=x) if use_out else x.add_(3)

  x = torch.ones(2)
  m = Mutate()
  with collect(m) as c:
    result = m(x)
  assert result is x and x.tolist() == [4, 4]
  rows = ops(c)
  assert len({row.call_id for row in rows}) == 1
  for row in rows:
    assert row.tensor.tolist() == ([1, 1] if row.edge == 'in' else [4, 4])
  if use_out:
    assert ('kwargs', 'out') in {row.path for row in rows}
  x.zero_()
  assert rows[-1].tensor.tolist() == [4, 4]


def test_scalar_only_factory_and_scalar_return_are_observed():
  """Tests scalar-only factory and scalar return operations are captured."""

  class Factory(nn.Module):

    def forward(self):
      value = torch.arange(3)
      return value.sum().item()

  m = Factory()
  with collect(m) as c:
    assert m() == 3
  assert [name for _, name in op_calls(c)] == [
      'aten.arange.default',
      'aten.sum.default',
      'aten._local_scalar_dense.default',
  ]
  assert all(
      row.op is not None and row.module_call_id == 0 for row in c.records
  )
  assert ops(c)[0].path == ('output',) and ops(c)[0].edge == 'out'


def test_root_external_and_inter_forward_operations_do_not_get_records():
  """Tests operations outside root and between forward passes are omitted."""

  class Add(nn.Module):

    def forward(self, x):
      return x + 1

  m = Add()
  with collect(m) as c:
    x = torch.ones(1)
    m(x)
    x = torch.sin(x)
    m(x)
    torch.cos(x)
  assert [name for _, name in op_calls(c)] == [
      'aten.add.Tensor',
      'aten.add.Tensor',
  ]
  assert {row.forward_id for row in ops(c)} == {0, 1}


def test_random_operation_runs_exactly_once():
  """Tests random operation executes exactly once during capture."""

  class Random(nn.Module):

    def forward(self):
      return torch.rand(4)

  m = Random()
  torch.manual_seed(194)
  baseline = m()
  baseline_next = torch.rand(4)
  torch.manual_seed(194)
  with collect(m) as c:
    result = m()
  assert torch.equal(result, baseline)
  assert torch.equal(torch.rand(4), baseline_next)
  assert [name for _, name in op_calls(c)] == ['aten.rand.default']


def test_parameter_views_copies_and_casts_are_filtered_at_module_and_op_boundaries():
  """Tests parameter views, copies, and casts are filtered at boundaries."""

  class Weights(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.arange(6.0).reshape(2, 3))
      self.child = nn.Identity()

    def forward(self):
      copy = self.weight.T.contiguous().to(torch.float64).clone()
      return self.child(copy)

  m = Weights()
  with collect(m, memory_limit_bytes=0) as c:
    result = m()
  assert not c.records
  assert result.dtype == torch.float64
  torch.testing.assert_close(result, m.weight.T.to(torch.float64))


def test_embedding_and_linear_keep_activations_without_copying_registered_weights():
  """Tests embedding and linear keep activations without copying weights."""
  m = nn.Sequential(nn.Embedding(5, 3), nn.Linear(3, 2))
  indices = torch.tensor([1, 4])
  baseline = m(indices)
  with collect(m) as c:
    actual = m(indices)
  torch.testing.assert_close(actual, baseline)
  embedding = [row for row in ops(c) if row.op == 'aten.embedding.default']
  assert [(row.edge, row.path) for row in embedding] == [
      ('in', ('args', 1)),
      ('out', ('output',)),
  ]
  addmm = [row for row in ops(c) if row.op == 'aten.addmm.default']
  assert [(row.edge, row.path) for row in addmm] == [
      ('in', ('args', 1)),
      ('out', ('output',)),
  ]
  assert not any(
      tuple(row.tensor.shape) in ((5, 3), (2, 3), (3, 2))
      and row.op == 'aten.t.default'
      for row in c.records
  )


def test_unknown_weight_derivation_and_mutated_copy_are_captured():
  """Tests unknown weight derivation and mutated copy are captured."""

  class Derived(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(2))

    def forward(self):
      copied = self.weight.clone()
      copied.add_(4)
      return copied, self.weight * 3

  m = Derived()
  with collect(m) as c:
    m()
  assert [row.tensor.tolist() for row in ops(c) if row.edge == 'out'] == [
      [5, 5],
      [3, 3],
  ]
  assert [row.tensor.tolist() for row in c.records if row.op is None] == [
      [5, 5],
      [3, 3],
  ]


def test_copy_mutation_between_roots_invalidates_held_representation():
  """Tests copy mutation between roots invalidates held representation."""

  class Held(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(2))
      self.saved = None

    def forward(self):
      if self.saved is None:
        self.saved = self.weight.clone()
      return self.saved

  m = Held()
  with collect(m) as c:
    m()
    m.saved.add_(6)
    m()
  assert len(c.records) == 1
  assert c.records[0].forward_id == 1 and c.records[0].tensor.tolist() == [7, 7]


def test_failed_outside_root_operator_invalidates_partial_writes():
  """Tests failed outside-root operator invalidates partial writes."""

  class Held(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(1))
      self.saved = None

    def forward(self):
      if self.saved is None:
        self.saved = self.weight.clone()
      return self.saved

  m = Held()
  with collect(m) as c:
    m()
    with pytest.raises(RuntimeError):
      torch._foreach_add_(
          [m.saved, torch.ones(2)], [torch.ones(1), torch.ones(3)]
      )
    m()
  # ATen modifies the first target before the second target's shape error.
  assert m.saved.item() == 2
  assert len(c.records) == 1 and c.records[0].tensor.item() == 2
  assert c.records[0].forward_id == 1


def test_module_and_op_share_one_budget_before_inplace_execution():
  """Tests module and operator share one budget before in-place execution."""

  class Mutate(nn.Module):

    def forward(self, x):
      return x.add_(1)

  m, x = Mutate(), torch.ones(1)
  # Module input consumes 4 bytes; operator input cannot fit, so x stays 1.
  with pytest.raises(MemoryError):
    with collect(m, memory_limit_bytes=4) as c:
      m(x)
  assert x.item() == 1
  with pytest.raises(CaptureError):
    _ = c.records
  with collect(m, memory_limit_bytes=16) as retry:
    m(x)
  assert (
      sum(
          row.tensor.numel() * row.tensor.element_size()
          for row in retry.records
      )
      == 16
  )


def test_caught_operator_error_poisons_scope_and_restores_modes():
  """Tests caught operator error poisons scope and restores dispatch modes."""

  class Catch(nn.Module):

    def forward(self, x):
      try:
        torch.mm(x, x)
      except RuntimeError:
        pass
      return x

  m = Catch()
  with RecordingMode() as outer:
    before = _get_current_dispatch_mode_stack()
    with pytest.raises(CaptureError):
      with collect(m) as c:
        m(torch.ones(1, 2))
    assert _get_current_dispatch_mode_stack() == before
    torch.ones(1)
    assert outer.names[-1] == 'aten.ones.default'
  with pytest.raises(CaptureError):
    _ = c.records


@pytest.mark.parametrize('raise_original', [False, True])
def test_unbalanced_mode_scope_removes_only_collector_and_preserves_original_error(
    raise_original,
):
  """Tests unbalanced mode scope removes only collector and keeps error."""
  m = nn.Identity()
  foreign = RecordingMode()
  original = ValueError('original scope failure')
  before = _get_current_dispatch_mode_stack()
  try:
    with pytest.raises(
        ValueError if raise_original else CaptureError
    ) as caught:
      with collect(m) as c:
        foreign.__enter__()
        m(torch.ones(1))
        if raise_original:
          raise original
    if raise_original:
      assert caught.value is original
    else:
      assert 'dispatch mode nesting' in str(caught.value)
    assert _get_current_dispatch_mode_stack() == before + [foreign]
    torch.zeros(1)
    assert foreign.names[-1] == 'aten.zeros.default'
    with pytest.raises(CaptureError):
      _ = c.records
  finally:
    foreign.__exit__(None, None, None)
  assert _get_current_dispatch_mode_stack() == before


def test_internal_snapshots_bypass_modes_but_model_operations_keep_them():
  """Tests internal snapshots bypass modes while model ops retain them."""

  class AlterClone(RecordingMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      result = super().__torch_dispatch__(func, types, args, kwargs)
      return result + 100 if func is torch.ops.aten.clone.default else result

  class Add(nn.Module):

    def forward(self, x):
      return x + 1

  x, m = torch.ones(1), Add()
  with AlterClone() as mode:
    before = _get_current_dispatch_mode_stack()
    with collect(m) as c:
      output = m(x)
    assert _get_current_dispatch_mode_stack() == before
  assert mode.names == ['aten.add.Tensor']
  assert output.item() == 2
  assert [row.tensor.item() for row in c.records] == [1, 1, 2, 2]


def test_lower_mode_transformed_copy_is_not_mistaken_for_parameter_identity():
  """Tests transformed copy in lower mode is not mistaken for identity."""

  class AlterClone(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      result = func(*args, **(kwargs or {}))
      return result + 100 if func is torch.ops.aten.clone.default else result

  class WeightCopy(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(2))

    def forward(self):
      return self.weight.clone()

  m = WeightCopy()
  with AlterClone():
    with collect(m) as c:
      result = m()
  assert result.tolist() == [101, 101]
  assert len(c.records) == 2
  assert [row.op for row in c.records] == ['aten.clone.default', None]
  assert all(
      row.edge == 'out' and row.tensor.tolist() == [101, 101]
      for row in c.records
  )


def test_snapshot_helpers_also_bypass_torch_function_modes_without_removing_them():
  """Tests snapshot helpers bypass torch function modes without removing them."""

  class AlterClone(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
      result = func(*args, **(kwargs or {}))
      return result + 100 if func is torch.Tensor.clone else result

  m, x = nn.Identity(), torch.ones(1)
  with AlterClone():
    with collect(m) as c:
      result = m(x)
    assert x.clone().item() == 101  # The user's mode is still active.
  assert result is x
  assert [row.tensor.item() for row in c.records] == [1, 1]


def test_nested_disjoint_collectors_do_not_capture_each_others_snapshots():
  """Tests nested disjoint collectors do not capture each other's snapshots."""

  class Add(nn.Module):

    def forward(self, x):
      return x + 1

  inner = Add()

  class Outer(nn.Module):

    def forward(self, x):
      with collect(inner) as nested:
        value = inner(x)
      self.nested = nested
      return value + 2

  m = Outer()
  with collect(m) as c:
    assert m(torch.ones(1)).item() == 4
  assert collections.Counter(name for _, name in op_calls(c)) == {
      'aten.add.Tensor': 2
  }
  assert [name for _, name in op_calls(m.nested)] == ['aten.add.Tensor']


def test_interrupted_child_is_detected_before_the_next_operator_mutates_input():
  """Tests interrupted child is detected before next operator mutates input."""

  class Child(nn.Module):

    def forward(self, x):
      raise KeyboardInterrupt('inner')

  class Root(nn.Module):

    def __init__(self):
      super().__init__()
      self.child = Child()

    def forward(self, x):
      try:
        self.child(x)
      except KeyboardInterrupt:
        pass
      return x.add_(1)

  m, x = Root(), torch.ones(1)
  with pytest.raises(CaptureError, match='interrupted call'):
    with collect(m):
      m(x)
  assert x.item() == 1


@pytest.mark.parametrize('fail_at,expected_value', [(2, 1), (3, 2)])
def test_snapshot_clone_failure_preserves_exact_operator_execution(
    monkeypatch, fail_at, expected_value
):
  """Tests snapshot clone failure preserves exact operator execution."""
  original = torch.Tensor.clone
  original_error = RuntimeError('snapshot clone failed')
  count = 0

  def failing_clone(tensor, *args, **kwargs):
    nonlocal count
    count += 1
    if count == fail_at:
      raise original_error
    return original(tensor, *args, **kwargs)

  class Mutate(nn.Module):

    def forward(self, x):
      return x.add_(1)

  m, x = Mutate(), torch.ones(1)
  before = _get_current_dispatch_mode_stack()
  monkeypatch.setattr(torch.Tensor, 'clone', failing_clone)
  with pytest.raises(RuntimeError) as caught:
    with collect(m) as c:
      m(x)
  assert caught.value is original_error
  assert x.item() == expected_value
  assert _get_current_dispatch_mode_stack() == before
  assert not c._pending and c._snapshots is None
