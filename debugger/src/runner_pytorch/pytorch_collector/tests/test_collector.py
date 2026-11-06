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

"""Tests for pytorch_collector.collect context manager."""

# pylint: disable=protected-access

import collections
from concurrent import futures
import weakref

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
import torch
from torch import nn


def module_records(capture):
  """Returns module records from capture excluding operator records."""
  return tuple(row for row in capture.records if row.op is None)


def hooks(model):
  """Returns a mapping of model module IDs to their registered hook IDs."""
  return {
      id(m): tuple(
          tuple((k, id(v)) for k, v in getattr(m, name).items())
          for name in (
              '_forward_pre_hooks',
              '_forward_hooks',
              '_forward_pre_hooks_with_kwargs',
              '_forward_hooks_with_kwargs',
              '_forward_hooks_always_called',
          )
      )
      for m in model.modules()
  }


def test_multi_forward_records_values_and_order():
  """Tests recording values and order across multiple forward calls."""
  model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU())
  with torch.no_grad():
    model[0].weight.copy_(torch.eye(2))
  before = hooks(model)
  x = torch.tensor([[-2.0, 3.0]], requires_grad=True)
  with collect(model) as capture:
    assert not torch.is_grad_enabled()
    y = model(x)
    model(x + 1)
    with pytest.raises(CaptureError, match='successful scope exit'):
      _ = capture.records
  assert torch.is_grad_enabled()
  assert hooks(model) == before
  assert torch.equal(y, torch.tensor([[0.0, 3.0]]))
  rows = module_records(capture)
  module_order = {
      call_id: index
      for index, call_id in enumerate(sorted({r.call_id for r in rows}))
  }
  assert [
      (r.forward_id, r.module, r.edge, module_order[r.call_id]) for r in rows
  ] == [
      (0, '', 'in', 0),
      (0, '0', 'in', 1),
      (0, '0', 'out', 1),
      (0, '1', 'in', 2),
      (0, '1', 'out', 2),
      (0, '', 'out', 0),
      (1, '', 'in', 3),
      (1, '0', 'in', 4),
      (1, '0', 'out', 4),
      (1, '1', 'in', 5),
      (1, '1', 'out', 5),
      (1, '', 'out', 3),
  ]
  assert all(r.op is None and r.module_call_id == r.call_id for r in rows)
  assert torch.equal(rows[0].tensor, x)
  assert not any(r.tensor.requires_grad for r in rows)
  rows[0].tensor.zero_()
  assert x[0, 0] == -2 and rows[1].tensor[0, 0] == -2


def test_kwargs_mapping_and_none_outputs():
  """Tests kwargs mapping and None outputs."""

  class Model(nn.Module):

    def forward(self, x, *, extra):
      return collections.UserDict(result=[x + extra['y'], None])

  model = Model()
  with collect(model) as c:
    output = model(torch.ones(2), extra={'y': torch.ones(2)})
  rows = module_records(c)
  assert [r.path for r in rows] == [
      ('args', 0),
      ('kwargs', 'extra', 'y'),
      ('output', 'result', 0),
  ]
  assert torch.equal(rows[-1].tensor, output['result'][0])


def test_shared_module_is_hooked_once_and_calls_have_distinct_ids():
  """Tests shared module is hooked once and calls have distinct IDs."""
  shared = nn.Identity()
  model = nn.Sequential(shared, shared)
  with collect(model) as c:
    assert len(shared._forward_hooks) == 1
    model(torch.ones(1))
  child = [r for r in c.records if r.module]
  assert {r.module for r in child} == {'0'}
  assert [r.call_id for r in child] == [1, 1, 2, 2]


def test_parameter_filtering_buffers_unknown_derivations_and_dtype():
  """Tests filtering parameters, capturing buffers and derived outputs."""

  class Model(nn.Module):

    def __init__(self):
      super().__init__()
      self.unusually_named = nn.Parameter(
          torch.arange(4, dtype=torch.bfloat16), requires_grad=False
      )
      self.register_buffer('state', torch.ones(2, dtype=torch.bfloat16))

    def forward(self):
      return {
          'weight': self.unusually_named,
          'view': self.unusually_named.view(2, 2),
          'buffer': self.state,
          'derived': self.unusually_named * 2,
      }

  model = Model()
  with collect(model) as c:
    model()
  assert [r.path for r in module_records(c)] == [
      ('output', 'buffer'),
      ('output', 'derived'),
  ]
  assert all(r.tensor.dtype == torch.bfloat16 for r in c.records)


def test_inplace_forward_does_not_corrupt_input_snapshot():
  """Tests in-place forward does not corrupt input snapshot."""

  class Mutate(nn.Module):

    def forward(self, x):
      return x.add_(3)

  m, x = Mutate(), torch.ones(2)
  with collect(m) as c:
    assert m(x) is x
  rows = module_records(c)
  assert torch.equal(rows[0].tensor, torch.ones(2))
  assert torch.equal(rows[1].tensor, torch.full((2,), 4.0))


def test_preexisting_hooks_capture_final_values_and_remain_installed():
  """Tests pre-existing hooks capture final values and remain installed."""
  m = nn.Identity()
  pre = m.register_forward_pre_hook(
      lambda m, a, kw: ((a[0] + 2,), kw), with_kwargs=True
  )
  post = m.register_forward_hook(lambda m, a, out: out * 3)
  before = hooks(m)
  try:
    with collect(m) as c:
      result = m(torch.ones(1))
    rows = module_records(c)
    assert rows[0].tensor.item() == 3
    assert rows[1].tensor.item() == result.item() == 9
    assert hooks(m) == before
  finally:
    pre.remove()
    post.remove()


def test_root_prehook_can_call_registered_child():
  """Tests root pre-hook can call registered child module."""
  m = nn.Sequential(nn.Identity())
  h = m.register_forward_pre_hook(lambda module, args: (module[0](args[0]),))
  try:
    with collect(m) as c:
      m(torch.ones(1))
    assert [(r.module, r.edge, r.call_id) for r in c.records] == [
        ('0', 'in', 1),
        ('0', 'out', 1),
        ('', 'in', 0),
        ('0', 'in', 2),
        ('0', 'out', 2),
        ('', 'out', 0),
    ]
  finally:
    h.remove()


def test_normal_child_reentry_pairs_each_call():
  """Tests normal child re-entry correctly pairs call IDs."""

  class Reentrant(nn.Module):

    def forward(self, x, depth=0):
      return self(x + 1, depth=1) + 2 if depth == 0 else x

  model = nn.Sequential(Reentrant())
  with collect(model) as c:
    y = model(torch.zeros(1))
  child = [r for r in module_records(c) if r.module == '0']
  assert [(r.edge, r.tensor.item()) for r in child] == [
      ('in', 0),
      ('in', 1),
      ('out', 1),
      ('out', 3),
  ]
  assert (
      child[0].call_id
      == child[3].call_id
      < child[1].call_id
      == child[2].call_id
  )
  assert y.item() == 3


def test_late_child_output_hook_f6_is_rejected():
  """Tests registering child output hook after entry is rejected."""
  model = nn.Sequential(nn.Identity())
  before = hooks(model)
  late = None
  with pytest.raises(CaptureError, match='hook configuration'):
    with collect(model) as c:
      late = model[0].register_forward_hook(
          lambda m, a, out: torch.zeros_like(out)
      )
      model(torch.ones(1))
  assert late.id in model[0]._forward_hooks  # Third-party hook was not removed.
  late.remove()
  assert hooks(model) == before
  with pytest.raises(CaptureError):
    _ = c.records


def test_hook_added_during_forward_f6_is_rejected():
  """Tests registering a hook during forward execution is rejected."""

  class Late(nn.Module):

    def forward(self, x):
      self.late = self.register_forward_hook(lambda m, a, out: out * 0)
      return x

  m = Late()
  try:
    with pytest.raises(CaptureError, match='hook configuration'):
      with collect(m):
        m(torch.ones(1))
  finally:
    m.late.remove()


def test_caught_reentrant_baseexception_f7_cannot_mispair_results():
  """Tests caught re-entrant BaseException does not mispair results."""

  class Reentrant(nn.Module):
    """Reentrant module raising KeyboardInterrupt on inner forward."""

    def __init__(self):
      super().__init__()
      self.depth = 0

    def forward(self, x):
      if self.depth:
        raise KeyboardInterrupt('inner')
      self.depth += 1
      try:
        try:
          self(x + 10)
        except KeyboardInterrupt:
          pass
      finally:
        self.depth -= 1
      return x

  m = nn.Sequential(Reentrant())
  before = hooks(m)
  with pytest.raises(CaptureError, match='interrupted call'):
    with collect(m) as c:
      m(torch.zeros(1))
  assert hooks(m) == before
  with pytest.raises(CaptureError):
    _ = c.records


@pytest.mark.parametrize(
    'error_type', [ValueError, KeyboardInterrupt, SystemExit]
)
def test_original_model_exception_is_preserved_and_resources_released(
    error_type,
):
  """Tests original model exception is preserved and resources are released."""
  original = error_type('original')

  class Fail(nn.Module):

    def forward(self, x):
      raise original

  m = Fail()
  before = hooks(m)
  with pytest.raises(error_type) as caught:
    with collect(m) as c:
      m(torch.ones(1))
  assert caught.value is original
  assert hooks(m) == before and torch.is_grad_enabled()
  with pytest.raises(CaptureError):
    _ = c.records
  with collect(m):
    pass  # Ownership was released even for BaseException.


def test_caught_error_poison_and_model_state_not_rolled_back():
  """Tests caught error poisons capture without rolling back model state."""

  class Fail(nn.Module):

    def __init__(self):
      super().__init__()
      self.count = 0

    def forward(self, x):
      self.count += 1
      if self.count == 2:
        raise ValueError('second forward')
      return x

  m = Fail()
  with pytest.raises(CaptureError):
    with collect(m) as c:
      m(torch.ones(1))
      with pytest.raises(ValueError):
        m(torch.ones(1))
  assert m.count == 2
  with pytest.raises(CaptureError):
    _ = c.records


def test_none_return_inside_except_block_is_successful():
  """Tests None return inside an except block succeeds."""

  class Nothing(nn.Module):

    def forward(self, x):
      return None

  m = Nothing()
  try:
    raise ValueError('ambient exception')
  except ValueError:
    with collect(m) as c:
      assert m(torch.ones(1)) is None
  assert len(c.records) == 1 and c.records[0].edge == 'in'


def test_budget_accumulates_across_forward_and_failed_result_is_unreadable():
  """Tests budget accumulates across forwards and failed result is unreadable."""
  m = nn.Identity()
  with pytest.raises(MemoryError):
    with collect(m, memory_limit_bytes=8) as c:
      m(torch.ones(1))  # Input + output are exactly 8 bytes.
      m(torch.ones(1))
  with pytest.raises(CaptureError):
    _ = c.records
  with collect(m, memory_limit_bytes=16) as retry:
    m(torch.ones(1))
    m(torch.ones(1))
  assert len(retry.records) == 4


def test_empty_scope_and_direct_child_outside_root_are_empty():
  """Tests empty scope and direct child outside root produce empty records."""
  m = nn.Sequential(nn.Identity())
  with collect(m, memory_limit_bytes=0) as c:
    m[0](torch.ones(1))
  assert not c.records


@pytest.mark.parametrize('after_root', [False, True])
@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt])
def test_caught_child_error_outside_root_does_not_poison_capture(
    after_root, error_type
):
  """Tests caught child error outside root does not poison capture."""

  class Child(nn.Module):

    def forward(self, x, fail=False):
      if fail:
        raise error_type('outside root')
      return x + 1

  m = nn.Sequential(Child())
  before = hooks(m)
  with collect(m) as c:
    if after_root:
      m(torch.ones(1))
    with pytest.raises(error_type, match='outside root'):
      m[0](torch.ones(1), fail=True)
    if not after_root:
      m(torch.ones(1))
  assert hooks(m) == before
  assert len(module_records(c)) == 4
  assert {row.forward_id for row in c.records} == {0}
  assert torch.equal(c.records[-1].tensor, torch.tensor([2.0]))


def test_zero_budget_can_capture_parameter_only_call_without_rows():
  """Tests zero-budget capture for parameter-only calls without rows."""

  class Weights(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(1))

    def forward(self):
      return self.weight

  m = Weights()
  with collect(m, memory_limit_bytes=0) as c:
    m()
  assert not c.records


def test_scope_preserves_disabled_grad_and_training_flag():
  """Tests scope preserves disabled grad and training flag."""
  m = nn.Identity().train()
  with torch.no_grad():
    with collect(m):
      assert m.training and not torch.is_grad_enabled()
    assert not torch.is_grad_enabled()
  assert m.training


def test_overlapping_capture_is_rejected_without_removing_owner_hooks():
  """Tests overlapping capture is rejected without removing owner hooks."""
  m = nn.Sequential(nn.Identity())
  with collect(m) as owner:
    before = hooks(m)
    with pytest.raises(CaptureError, match='already owned'):
      with collect(m[0]):
        pass
    assert hooks(m) == before
    m(torch.ones(1))
  assert len(owner.records) == 4


def test_partial_hook_installation_failure_is_rolled_back():
  """Tests partial hook installation failure rolls back installed hooks."""

  class FailInstall(nn.Identity):

    def register_forward_hook(self, *args, **kwargs):
      raise ValueError('install failed')

  m = nn.Sequential(nn.Identity(), FailInstall())
  before = hooks(m)
  with pytest.raises(ValueError, match='install failed'):
    with collect(m):
      pass
  assert hooks(m) == before and torch.is_grad_enabled()
  with collect(m[0]):
    pass


@pytest.mark.parametrize('boundary', ['root', 'child'])
@pytest.mark.parametrize('wrap', [False, True])
def test_custom_call_cannot_publish_missing_or_incorrect_boundaries(
    boundary, wrap
):
  """Tests custom __call__ cannot publish incorrect boundaries."""

  class CustomCall(nn.Identity):

    def __call__(self, x):
      if wrap:
        return super().__call__(x) + 10
      return self.forward(x) + 10

  child = CustomCall()
  m = child if boundary == 'root' else nn.Sequential(nn.Identity(), child)
  before = hooks(m)
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(m):
      m(torch.ones(1))
  assert hooks(m) == before and torch.is_grad_enabled()
  if boundary == 'child':
    with collect(m[0]):
      pass


def test_call_binding_changed_after_last_forward_is_rejected():
  """Tests call binding changed after last forward is rejected."""

  class Model(nn.Identity):
    pass

  m = Model()
  before = hooks(m)
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(m) as c:
      m(torch.ones(1))
      Model.__call__ = lambda self, x: self.forward(x) + 10
  assert hooks(m) == before
  with pytest.raises(CaptureError):
    _ = c.records


@pytest.mark.parametrize(
    'mutation',
    ['alias', 'replace_child', 'replace_forward', 'parameter_storage'],
)
def test_topology_and_binding_changes_are_rejected(mutation):
  """Tests topology and binding changes during capture are rejected."""
  m = nn.Sequential(nn.Linear(1, 1))
  with pytest.raises(CaptureError, match='changed during capture'):
    with collect(m):
      if mutation == 'alias':
        m.alias = m[0]
      elif mutation == 'replace_child':
        m[0] = nn.Identity()
      elif mutation == 'replace_forward':
        m[0].forward = lambda x: x
      else:
        m[0].weight.data = m[0].weight.data.clone()
      m(torch.ones(1, 1))


def test_topology_change_between_last_forward_and_exit_is_detected():
  """Tests topology change between last forward and exit is detected."""
  m = nn.Sequential(nn.Identity())
  with pytest.raises(CaptureError, match='topology'):
    with collect(m):
      m(torch.ones(1))
      m.alias = m[0]


def test_global_hook_and_inference_mode_are_rejected():
  """Tests global hook and inference mode are rejected."""
  h = nn.modules.module.register_module_forward_hook(lambda *args: None)
  try:
    with pytest.raises(CaptureError, match='global module hooks'):
      with collect(nn.Identity()):
        pass
  finally:
    h.remove()
  with torch.inference_mode(), pytest.raises(
      CaptureError, match='inference_mode'
  ):
    with collect(nn.Identity()):
      pass


def test_cross_thread_root_call_poison_scope():
  """Tests cross-thread root call poisons scope."""
  m = nn.Identity()
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    with pytest.raises(CaptureError, match='thread'):
      with collect(m):
        with pytest.raises(CaptureError, match='thread'):
          executor.submit(m, torch.ones(1)).result()


def test_recursive_root_is_rejected():
  """Tests recursive root is rejected."""

  class Root(nn.Module):

    def forward(self, x, depth=0):
      return self(x, depth=1) if depth == 0 else x

  m = Root()
  with pytest.raises(CaptureError, match='recursive root'):
    with collect(m):
      m(torch.ones(1))


def test_unknown_value_failure_releases_prior_snapshots():
  """Tests unknown value failure releases prior snapshots."""

  class Unknown(nn.Module):

    def forward(self, x):
      return [x, object()]

  m = Unknown()
  with pytest.raises(TypeError, match='Unsupported value'):
    with collect(m) as c:
      m(torch.ones(1))
  assert not c._pending
  assert c._snapshots is None


def test_completed_capture_does_not_own_model_and_is_one_use():
  """Tests completed capture releases model reference and is one-use."""
  model = nn.Identity()
  reference = weakref.ref(model)
  c = collect(model)
  with c:
    model(torch.ones(1))
  del model
  assert reference() is None
  with pytest.raises(CaptureError, match='once'):
    with c:
      pass
  with pytest.raises(CaptureError, match='active'):
    c.__exit__(None, None, None)
  assert len(c.records) == 2


@pytest.mark.parametrize('limit', [-1, True, 1.5])
def test_invalid_memory_limit(limit):
  with pytest.raises((ValueError, TypeError)):
    collect(nn.Identity(), memory_limit_bytes=limit)
