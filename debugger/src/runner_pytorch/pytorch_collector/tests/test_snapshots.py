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

"""Tests for tensor snapshotting and identity tracking in pytorch_collector."""

# pylint: disable=protected-access
import collections
import gc
import weakref

import pytest
from pytorch_collector._snapshots import Snapshotter
from pytorch_collector._snapshots import walk
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode


class IdentityTracking(TorchDispatchMode):
  """Use the same prepare/execute/finish contract as the capture dispatcher."""

  def __init__(self, snapshotter, *, track_copies=True):
    super().__init__()
    self.snapshotter = snapshotter
    self.track_copies = track_copies
    self.observed = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    self.observed.append(func)
    token = self.snapshotter.prepare_operation(
        func, args, kwargs, track_copies=self.track_copies
    )
    result = func(*args, **kwargs)
    self.snapshotter.finish_operation(token, result)
    return result


def test_nested_mapping_paths_and_independent_original_dtype_snapshots():
  """Tests nested mapping paths and independent original dtype snapshots."""
  value = torch.tensor([1, 2], dtype=torch.bfloat16, requires_grad=True)
  nested = collections.UserDict({
      "args": (value,),
      "kwargs": collections.OrderedDict([(3, [None, value])]),
  })
  snapshotter = Snapshotter([], 8)

  snapshots = snapshotter.snapshot(nested, ())

  assert [path for path, _ in snapshots] == [("args", 0), ("kwargs", 3, 1)]
  assert snapshotter.bytes_used == 8
  for _, tensor in snapshots:
    assert tensor.dtype == torch.bfloat16
    assert not tensor.requires_grad
    torch.testing.assert_close(tensor, value)
  with torch.no_grad():
    value.add_(10)
  snapshots[0][1].add_(20)
  torch.testing.assert_close(
      snapshots[1][1], torch.tensor([1, 2], dtype=torch.bfloat16)
  )


def test_noncontiguous_and_expanded_values_are_copied_with_values_intact():
  """Tests noncontiguous and expanded values are copied with values intact."""
  transposed = torch.arange(12).reshape(3, 4).T
  expanded = torch.tensor([2.0]).expand(4)
  snapshots = Snapshotter([], 112).snapshot((transposed, expanded), ("output",))
  torch.testing.assert_close(snapshots[0][1], transposed)
  torch.testing.assert_close(snapshots[1][1], expanded)
  expanded[0] = 5
  assert snapshots[1][1].tolist() == [2, 2, 2, 2]


def test_parameter_identity_views_detach_and_tied_references_are_excluded():
  """Tests parameter identity views, detach, and tied references are excluded."""
  parameter = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4))
  snapshotter = Snapshotter([parameter, parameter], 0)
  aliases = [
      parameter,
      parameter.T,
      parameter.detach(),
      parameter.reshape(-1),
      parameter[1:, 1:],
  ]

  assert not snapshotter.snapshot(aliases, ("output",))
  assert snapshotter.bytes_used == 0


def test_parameter_type_without_registered_identity_and_buffer_are_captured():
  """Tests parameter type without registered identity and buffer are captured."""
  model = torch.nn.Module()
  model.register_parameter("weight", torch.nn.Parameter(torch.arange(2.0)))
  model.register_buffer("running", torch.arange(2.0))
  unrelated = torch.nn.Parameter(torch.tensor([9.0]))
  snapshotter = Snapshotter(model.parameters(), 12)

  snapshots = snapshotter.snapshot(
      (model.weight, model.running, unrelated), ("args",)
  )

  assert [path for path, _ in snapshots] == [("args", 1), ("args", 2)]
  assert snapshotter.bytes_used == 12


def test_independent_parameter_copies_and_computed_activations_are_captured():
  """Tests independent parameter copies and computed activations are captured."""
  parameter = torch.nn.Parameter(torch.ones(2, 2))
  copied = parameter.detach().clone()
  cast = parameter.to(torch.float64)
  activation = torch.ones(1, 2) @ parameter
  snapshotter = Snapshotter([parameter], 56)

  snapshots = snapshotter.snapshot([copied, cast, activation], ("output",))

  assert len(snapshots) == 3
  assert snapshotter.bytes_used == 56


def test_sharing_storage_does_not_exclude_nonparameter_or_overlapping_regions():
  """Tests sharing storage does not exclude nonparameter or overlapping regions."""
  backing = torch.arange(12.0)
  parameter = torch.nn.Parameter(backing[2:6])
  same_region = backing[3:5]
  outside = backing[6:]
  overlaps_outside = backing[1:4]
  oversized_view = parameter.as_strided((7,), (1,), storage_offset=1)
  snapshotter = Snapshotter([parameter], 64)

  snapshots = snapshotter.snapshot(
      [same_region, outside, overlaps_outside, oversized_view], ()
  )

  assert [path for path, _ in snapshots] == [(1,), (2,), (3,)]
  assert snapshotter.bytes_used == 64


def test_strided_parameter_does_not_hide_tensors_in_its_storage_gaps():
  """Tests strided parameter does not hide tensors in storage gaps."""
  backing = torch.arange(12.0).reshape(3, 4)
  parameter = torch.nn.Parameter(backing[:, ::2])
  snapshotter = Snapshotter([parameter], 24)

  snapshots = snapshotter.snapshot(
      [parameter.detach(), parameter.T, backing[:, 1::2]], ()
  )

  assert [path for path, _ in snapshots] == [(2,)]
  torch.testing.assert_close(snapshots[0][1], backing[:, 1::2])


def test_transposed_registered_parameter_covers_its_dense_region():
  """Tests transposed registered parameter covers its dense region."""
  parameter = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4).T)
  assert not Snapshotter([parameter], 0).snapshot(
      [parameter[1:], parameter.detach().T], ()
  )


def test_empty_tensors_with_distinct_storage_are_not_confused():
  """Tests empty tensors with distinct storage are not confused."""
  parameter = torch.nn.Parameter(torch.empty(0))
  other = torch.empty(0)
  snapshots = Snapshotter([parameter], 0).snapshot([parameter, other], ())
  assert [path for path, _ in snapshots] == [(1,)]


def test_budget_is_cumulative_counts_repeated_snapshots_and_clear_resets_it():
  """Tests budget is cumulative, counts repeated snapshots, and clear resets it."""
  value = torch.ones(2)
  snapshotter = Snapshotter([], 16)
  first = snapshotter.snapshot(value, ())
  second = snapshotter.snapshot(value, ())
  assert snapshotter.bytes_used == 16
  with pytest.raises(MemoryError, match="memory limit"):
    snapshotter.snapshot(value, ())
  assert snapshotter.bytes_used == 16
  snapshotter.clear()
  assert snapshotter.bytes_used == 0
  assert len(snapshotter.snapshot(value, ())) == 1
  first[0][1].zero_()
  assert second[0][1].tolist() == [1, 1]


def test_over_budget_tensor_is_rejected_before_clone(monkeypatch):
  """Tests over budget tensor is rejected before clone."""

  def forbidden_clone(*args, **kwargs):
    del args, kwargs  # Unused.
    pytest.fail("clone must not run when the tensor does not fit")

  monkeypatch.setattr(torch.Tensor, "clone", forbidden_clone)
  with pytest.raises(MemoryError):
    Snapshotter([], 3).snapshot(torch.ones(1), ())


def test_failed_call_rolls_back_its_accounting_without_erasing_previous_calls():
  """Tests failed call rolls back its accounting without erasing previous calls."""
  snapshotter = Snapshotter([], 16)
  snapshotter.snapshot(torch.ones(1), ())
  with pytest.raises(TypeError, match="Unsupported value"):
    snapshotter.snapshot([torch.ones(2), object()], ())
  assert snapshotter.bytes_used == 4


@pytest.mark.parametrize("kind", ["list", "mapping", "indirect"])
def test_cycles_fail_but_shared_containers_are_not_cycles(kind):
  """Tests cycles fail but shared containers are not cycles."""
  if kind == "list":
    value = []
    value.append(value)
  elif kind == "mapping":
    value = collections.UserDict()
    value["self"] = value
  else:
    value = []
    value.append({"back": value})
  snapshotter = Snapshotter([], 16)
  with pytest.raises(ValueError, match="Cyclic"):
    snapshotter.snapshot(value, ())
  shared = [torch.ones(1)]
  assert len(snapshotter.snapshot([shared, shared], ())) == 2


def test_metadata_and_python_scalar_atoms_do_not_make_tensor_records():
  """Tests metadata and python scalar atoms do not make tensor records."""
  atoms = [
      None,
      True,
      7,
      3.5,
      1j,
      "name",
      b"raw",
      torch.float32,
      torch.device("cpu"),
      torch.strided,
      torch.contiguous_format,
      torch.channels_last,
      torch.per_tensor_affine,
  ]
  snapshotter = Snapshotter([], 0)
  assert not snapshotter.snapshot(atoms, ())
  assert snapshotter.bytes_used == 0


@pytest.mark.parametrize("key", [1.5, ("nested",), True])
def test_unsupported_mapping_keys_fail_even_with_scalar_values(key):
  """Tests unsupported mapping keys fail even with scalar values."""
  with pytest.raises(TypeError, match="Mapping keys"):
    Snapshotter([], 0).snapshot({key: None}, ())


@pytest.mark.parametrize(
    "limit,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_memory_limits(limit, error):
  """Tests invalid memory limits."""
  with pytest.raises(error):
    Snapshotter([], limit)


def test_unsupported_tensor_kinds_fail_explicitly():
  """Tests unsupported tensor kinds fail explicitly."""

  class CustomTensor(torch.Tensor):
    """Custom Tensor subclass for testing unsupported types."""

  snapshotter = Snapshotter([], 100)
  values = [
      torch.ones(2).as_subclass(CustomTensor),
      torch.ones(2).to_sparse(),
      torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8),
  ]
  for value in values:
    with pytest.raises(TypeError):
      snapshotter.snapshot(value, ())
  with pytest.raises(ValueError, match="CPU"):
    snapshotter.snapshot(torch.empty(2, device="meta"), ())


def test_observed_clone_cast_and_contiguous_chain_excludes_only_weight_representations():
  """Tests observed clone, cast, and contiguous chain excludes only weights."""
  parameter = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
  snapshotter = Snapshotter([parameter], 1000)
  tracker = IdentityTracking(snapshotter)
  with torch.no_grad(), tracker:
    copied = parameter.clone()
    cast = copied.to(torch.float64)
    contiguous = cast.T.contiguous()
    activation = torch.ones(1, 2, dtype=torch.float64) @ cast
    unknown_derivation = cast * 2

  assert torch.ops.aten.clone.default in tracker.observed
  assert torch.ops.aten._to_copy.default in tracker.observed
  assert not snapshotter.snapshot(
      [copied, cast, contiguous, cast.detach(), copied[:, 1:]], ()
  )
  assert len(snapshotter.snapshot([activation, unknown_derivation], ())) == 2


def test_operation_source_bound_by_keyword_and_copy_out_overload_is_not_propagated():
  """Tests keyword-bound source and copy out overload are not propagated."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    copied = torch.ops.aten.clone.default(self=parameter)
    cast = torch.ops.aten._to_copy.default(self=copied, dtype=torch.float64)
    destination = torch.empty_like(parameter)
    torch.ops.aten.clone.out(parameter, out=destination)
  assert snapshotter.is_parameter(copied)
  assert snapshotter.is_parameter(cast)
  assert not snapshotter.is_parameter(destination)


def test_unobserved_copy_and_its_observed_cast_remain_normal_tensors():
  """Tests unobserved copy and observed cast remain normal tensors."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  external = parameter.detach().clone()
  snapshotter = Snapshotter([parameter], 100)
  with IdentityTracking(snapshotter):
    cast = external.to(torch.float64)
  assert not snapshotter.is_parameter(external)
  assert not snapshotter.is_parameter(cast)


def test_copy_of_nonparameter_region_sharing_parameter_storage_is_not_propagated():
  """Tests copy of nonparameter region sharing storage is not propagated."""
  backing = torch.arange(6.0)
  parameter = torch.nn.Parameter(backing[:3])
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    owned_copy = backing[1:3].clone()
    outside_copy = backing[3:].clone()
    overlap_copy = backing[2:4].clone()
  assert snapshotter.is_parameter(owned_copy)
  assert not snapshotter.is_parameter(outside_copy)
  assert not snapshotter.is_parameter(overlap_copy)


def test_write_through_derived_alias_invalidates_whole_storage_and_following_copies():
  """Tests write through derived alias invalidates storage and following copies."""
  parameter = torch.nn.Parameter(torch.arange(6.0))
  parameter = torch.nn.Parameter(torch.arange(6.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    copied = parameter.clone()
    alias = copied[1:]
    assert snapshotter.is_parameter(alias)
    alias.add_(10)
    following_copy = copied.clone()
  assert not snapshotter.is_parameter(copied)
  assert not snapshotter.is_parameter(alias)
  assert not snapshotter.is_parameter(following_copy)
  assert len(snapshotter.snapshot(copied, ())) == 1


def test_out_write_into_derived_weight_keeps_new_runtime_result():
  """Tests out write into derived weight keeps new runtime result."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    destination = parameter.clone()
    output = torch.add(torch.ones(3), torch.ones(3), out=destination)
  assert output is destination
  assert not snapshotter.is_parameter(output)
  captured = snapshotter.snapshot(output, ())
  assert captured[0][1].tolist() == [2, 2, 2]


def test_foreach_write_invalidates_tensor_list_and_preserves_registered_parameter_identity():
  """Tests foreach write invalidates tensor list and preserves parameter."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    first = parameter.clone()
    second = parameter.clone()
    torch._foreach_add_([first, second, parameter], 1)
  assert not snapshotter.is_parameter(first)
  assert not snapshotter.is_parameter(second)
  assert snapshotter.is_parameter(parameter)
  assert snapshotter.is_parameter(parameter.detach())


def test_storage_rebinding_invalidates_previous_derived_aliases():
  """Tests storage rebinding invalidates previous derived aliases."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    derived = parameter.clone()
    old_alias = derived.view(-1)
    derived.set_(torch.ones(3))
  assert not snapshotter.is_parameter(derived)
  assert not snapshotter.is_parameter(old_alias)


def test_identity_only_tracking_invalidates_writes_but_does_not_recognize_new_copies():
  """Tests identity-only tracking invalidates writes without new copies."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    derived = parameter.clone()
  with torch.no_grad(), IdentityTracking(snapshotter, track_copies=False):
    outside_copy = parameter.clone()
    derived.copy_(torch.ones(3))
  assert not snapshotter.is_parameter(outside_copy)
  assert not snapshotter.is_parameter(derived)
  assert snapshotter.is_parameter(parameter)


def test_identity_only_tracking_ignores_unsupported_tensor_kinds_and_tensorless_ops():
  """Tests identity-only tracking ignores unsupported kinds and tensorless ops."""
  snapshotter = Snapshotter([], 0)
  meta = torch.empty(3, device="meta")
  with IdentityTracking(snapshotter, track_copies=False):
    output = meta.clone()
    torch.ops.aten._assert_scalar.default(True, "ok")
  assert not snapshotter.is_parameter(output)


def test_tracking_does_not_run_tensor_operations_during_prepare_or_finish():
  """Tests tracking does not run tensor ops during prepare or finish."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 0)

  class NoTensorOperations(TorchDispatchMode):
    """Dispatch mode failing if any tensor operation is dispatched."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      pytest.fail(f"Identity bookkeeping must not execute {func}")

  with NoTensorOperations():
    token = snapshotter.prepare_operation(
        torch.ops.aten.clone.default, (parameter,), {}
    )
  result = parameter.detach().clone()
  with NoTensorOperations():
    snapshotter.finish_operation(token, result)
    assert snapshotter.is_parameter(result)


def test_derived_tracking_keeps_neither_tensor_nor_storage_data_alive():
  """Tests derived tracking keeps neither tensor nor storage data alive."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 0)
  with torch.no_grad(), IdentityTracking(snapshotter):
    derived = parameter.clone()
  tensor_ref = weakref.ref(derived)
  storage_ref = StorageWeakRef(derived.untyped_storage())
  alias = derived.detach()
  del derived
  gc.collect()
  assert tensor_ref() is None
  assert not storage_ref.expired()
  assert snapshotter.is_parameter(alias)
  del alias
  gc.collect()
  assert storage_ref.expired()
  # Empty allocations can have the same data_ptr (zero), but never inherit
  # expired provenance or the provenance of another live storage.
  assert all(not snapshotter.is_parameter(torch.empty(0)) for _ in range(100))


def test_clear_forgets_live_derived_storage_without_removing_parameter_sources():
  """Tests clear forgets live derived storage without removing parameters."""
  parameter = torch.nn.Parameter(torch.arange(3.0))
  snapshotter = Snapshotter([parameter], 100)
  with torch.no_grad(), IdentityTracking(snapshotter):
    derived = parameter.clone()
  assert snapshotter.is_parameter(derived)
  snapshotter.clear()
  assert not snapshotter.is_parameter(derived)
  assert snapshotter.is_parameter(parameter)
  assert len(snapshotter.snapshot(derived, ())) == 1


def test_walk_yields_original_tensors_including_parameters_in_traversal_order():
  """Tests walk yields original tensors in traversal order."""
  parameter = torch.nn.Parameter(torch.ones(2))
  activation = torch.zeros(3)
  value = {
      "args": (parameter, 1.5, [activation]),
      "kwargs": {"mask": None, "dtype": torch.float32},
  }

  walked = list(walk(value, ("root",)))

  assert [path for path, _ in walked] == [
      ("root", "args", 0),
      ("root", "args", 2, 0),
  ]
  assert walked[0][1] is parameter
  assert walked[1][1] is activation
  # The snapshotter applies its own parameter exclusion on top of the same walk.
  assert [
      path
      for path, _ in Snapshotter([parameter], 64).snapshot(value, ("root",))
  ] == [("root", "args", 2, 0)]


def test_walk_is_lazy_and_rejects_unsupported_values_at_their_position():
  """Tests walk is lazy and rejects unsupported values at position."""
  first = torch.ones(1)
  iterator = walk((first, object()), ())

  assert next(iterator) == ((0,), first)
  with pytest.raises(TypeError, match="Unsupported value of type object"):
    next(iterator)
  with pytest.raises(TypeError, match="path must be a tuple"):
    walk(first, ["args"])
