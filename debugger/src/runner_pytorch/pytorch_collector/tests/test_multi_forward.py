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

"""Combined module/operator guarantees across a whole capture scope."""

import collections

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
import torch
from torch import nn
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack


def hooks(model):
  return {
      id(module): tuple(
          tuple(getattr(module, name).items())
          for name in (
              '_forward_pre_hooks',
              '_forward_pre_hooks_with_kwargs',
              '_forward_hooks',
              '_forward_hooks_with_kwargs',
              '_forward_hooks_always_called',
          )
      )
      for module in model.modules()
  }


class Branching(nn.Module):

  def __init__(self):
    super().__init__()
    self.left = self.right = nn.ReLU()

  def forward(self, x):
    y = self.left(x + 1)
    return self.right(y * 2) if x.shape[0] > 1 else y - 3


def inputs():
  return [
      torch.tensor([[-2.0, 3.0]]),
      torch.arange(6.0).reshape(3, 2),
      torch.ones(1, 2),
  ]


def test_shapes_branches_and_shared_modules_keep_forward_and_call_identity():
  """Tests shapes, branches, and shared modules keep forward and call identity."""
  model, values = Branching(), inputs()
  expected = [model(x) for x in values]
  before, stack = hooks(model), _get_current_dispatch_mode_stack()
  with collect(model) as capture:
    actual = []
    for x in values:
      torch.sin(
          x
      )  # Caller work between roots must not allocate records or IDs.
      actual.append(model(x))
  assert hooks(model) == before and _get_current_dispatch_mode_stack() == stack
  assert torch.is_grad_enabled()
  for output, baseline in zip(actual, expected, strict=True):
    torch.testing.assert_close(output, baseline, atol=0, rtol=0)

  calls, forward_bytes = collections.defaultdict(list), collections.defaultdict(
      int
  )
  for row in capture.records:
    calls[row.call_id].append(row)
    forward_bytes[row.forward_id] += (
        row.tensor.numel() * row.tensor.element_size()
    )
  expected_calls = []
  for forward_id, repeated in enumerate([False, True, False]):
    expected_calls.extend([
        (forward_id, '', None),
        (forward_id, '', 'aten.add.Tensor'),
        (forward_id, 'left', None),
        (forward_id, 'left', 'aten.relu.default'),
    ])
    expected_calls.append(
        (forward_id, '', 'aten.mul.Tensor' if repeated else 'aten.sub.Tensor')
    )
    if repeated:
      expected_calls.extend([
          (forward_id, 'left', None),
          (forward_id, 'left', 'aten.relu.default'),
      ])
  assert sorted(calls) == list(range(len(expected_calls)))
  assert [
      (calls[i][0].forward_id, calls[i][0].module, calls[i][0].op)
      for i in sorted(calls)
  ] == expected_calls
  for rows in calls.values():
    assert [row.edge for row in rows] == ['in', 'out']
    assert (
        len({(r.forward_id, r.module, r.op, r.module_call_id) for r in rows})
        == 1
    )
    first = rows[0]
    owner = calls[first.module_call_id]
    assert owner[0].op is None
    assert (owner[0].forward_id, owner[0].module) == (
        first.forward_id,
        first.module,
    )
    assert (
        rows[0].tensor.shape
        == rows[1].tensor.shape
        == values[first.forward_id].shape
    )
  assert dict(forward_bytes) == {0: 80, 1: 336, 2: 80}

  root_outputs = [
      r
      for r in capture.records
      if not r.module and r.op is None and r.edge == 'out'
  ]
  for row, baseline in zip(root_outputs, expected, strict=True):
    torch.testing.assert_close(row.tensor, baseline, atol=0, rtol=0)


def test_each_forward_fits_alone_but_cumulative_scope_exceeds_budget():
  """Tests each forward fits alone but cumulative scope exceeds budget."""
  model, values = Branching(), inputs()
  for x in values:
    with collect(model, memory_limit_bytes=336):
      model(x)
  completed = 0
  with pytest.raises(MemoryError):
    with collect(model, memory_limit_bytes=336) as failed:
      for x in values:
        model(x)
        completed += 1
  assert completed == 1
  with pytest.raises(CaptureError, match='successful scope exit'):
    _ = failed.records
  with collect(model, memory_limit_bytes=496) as retry:
    for x in inputs():
      model(x)
  assert {r.forward_id for r in retry.records} == {0, 1, 2}
  assert (
      sum(r.tensor.numel() * r.tensor.element_size() for r in retry.records)
      == 496
  )


@pytest.mark.parametrize(
    'second_budget,expected_visits,expected_value',
    [
        (0, 1, 10),  # Second module input: forward body has not run.
        (4, 2, 10),  # Second operator input: body ran, operator has not.
        (8, 2, 11),  # Second operator output: mutation already happened.
        (12, 2, 11),  # Second module output: mutation already happened.
    ],
)
def test_later_failure_invalidates_prior_forward_and_requires_caller_reset(
    second_budget,
    expected_visits,
    expected_value,
):
  """Tests later failure invalidates prior forward and requires caller reset."""

  class Stateful(nn.Module):

    def __init__(self):
      super().__init__()
      self.visits = 0

    def forward(self, x):
      self.visits += 1
      return x.add_(1)

  model = Stateful()
  values = [torch.zeros(1), torch.full((1,), 10.0)]
  before, stack, grad = (
      hooks(model),
      _get_current_dispatch_mode_stack(),
      torch.is_grad_enabled(),
  )
  with pytest.raises(MemoryError):
    with collect(model, memory_limit_bytes=16 + second_budget) as failed:
      model(values[0])
      model(values[1])
  assert hooks(model) == before and _get_current_dispatch_mode_stack() == stack
  assert torch.is_grad_enabled() == grad
  assert model.visits == expected_visits and values[1].item() == expected_value
  assert (
      values[0].item() == 1
  )  # Collector does not roll back any successful mutation.
  with pytest.raises(CaptureError, match='successful scope exit'):
    _ = failed.records

  # Retry starts from caller-rebuilt model state and inputs, with a new scope.
  model.visits = 0
  values = [torch.zeros(1), torch.full((1,), 10.0)]
  with collect(model, memory_limit_bytes=32) as recovered:
    for x in values:
      model(x)
  assert model.visits == 2 and [x.item() for x in values] == [1, 11]
  assert len(recovered.records) == 8
  assert {r.forward_id for r in recovered.records} == {0, 1}
  assert (
      sum(r.tensor.numel() * r.tensor.element_size() for r in recovered.records)
      == 32
  )
