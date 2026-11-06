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

"""Tests for dispatch mode lifecycle and flag restoration."""

# pylint: disable=protected-access

import pytest
from pytorch_collector._dispatch import CaptureMode
import torch
import torch.utils._python_dispatch as dispatch


def flags():
  """Returns current dispatch mode flags."""
  return (
      dispatch._is_in_torch_dispatch_mode,
      dispatch._is_in_non_infra_torch_dispatch_mode,
      dispatch._is_in_any_mode_without_ignore_compile_internals,
  )


class RecordingMode(dispatch.TorchDispatchMode):
  """A TorchDispatchMode recording operations and exits."""

  def __init__(self):
    super().__init__()
    self.operations = []
    self.exits = 0

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.operations.append(str(func))
    return func(*args, **(kwargs or {}))

  def __exit__(self, *args):
    self.exits += 1
    return super().__exit__(*args)


class InfraMode(RecordingMode):
  """A RecordingMode marked as an infra mode."""

  @classmethod
  def is_infra_mode(cls):
    return True


class IgnoreCompileMode(RecordingMode):
  """A RecordingMode marked to ignore compile internals."""

  @classmethod
  def ignore_compile_internals(cls):
    return True


class Receiver:
  """A mock receiver recording operations."""

  def __init__(self):
    self.operations = []

  def _operation(self, func, args, kwargs):
    self.operations.append(str(func))
    return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def restore_modes():
  """Fixture to restore dispatch mode stack and flags after test."""
  before = dispatch._get_current_dispatch_mode_stack()
  before_flags = flags()
  yield
  stack = dispatch._get_current_dispatch_mode_stack()
  while len(stack) > len(before):
    stack[-1].__exit__(None, None, None)
    stack = dispatch._get_current_dispatch_mode_stack()
  assert stack == before
  assert flags() == before_flags


def test_normal_close_restores_outer_mode_and_its_flags():
  """Tests that normal close restores outer mode and its flags."""
  outer = InfraMode()
  outer.__enter__()
  before_flags = flags()
  mode = CaptureMode(Receiver())
  mode.__enter__()
  assert mode.close()
  assert dispatch._get_current_dispatch_mode_stack() == [outer]
  assert flags() == before_flags
  assert not mode.old_dispatch_mode_flags
  torch.ones(1)
  assert outer.operations == ['aten.ones.default']
  outer.__exit__(None, None, None)


@pytest.mark.parametrize(
    'mode_type', [RecordingMode, InfraMode, IgnoreCompileMode]
)
def test_close_removes_only_capture_below_two_foreign_activations(mode_type):
  """Tests that close removes only capture below two foreign activations."""
  before = flags()
  receiver = Receiver()
  capture = CaptureMode(receiver)
  capture.__enter__()
  foreign = mode_type()
  foreign.__enter__()
  foreign.__enter__()

  assert not capture.close()
  assert dispatch._get_current_dispatch_mode_stack() == [foreign, foreign]
  assert foreign.exits == 0
  torch.ones(1)
  assert not receiver.operations
  assert foreign.operations == ['aten.ones.default', 'aten.ones.default']
  expected_active = (
      True,
      not foreign.is_infra_mode(),
      not foreign.ignore_compile_internals(),
  )
  assert flags() == expected_active

  foreign.__exit__(None, None, None)
  assert flags() == expected_active
  foreign.__exit__(None, None, None)
  assert flags() == before
  assert not dispatch._get_current_dispatch_mode_stack()


def test_reentered_foreign_object_also_below_capture_keeps_its_lower_activation():
  """Tests re-entered foreign object below capture keeps lower activation."""
  before = flags()
  foreign = InfraMode()
  foreign.__enter__()
  below_flags = flags()
  capture = CaptureMode(Receiver())
  capture.__enter__()
  foreign.__enter__()
  foreign.__enter__()

  assert not capture.close()
  assert dispatch._get_current_dispatch_mode_stack() == [
      foreign,
      foreign,
      foreign,
  ]
  assert flags() == below_flags
  for remaining in (2, 1):
    foreign.__exit__(None, None, None)
    assert dispatch._get_current_dispatch_mode_stack() == [foreign] * remaining
    assert flags() == below_flags
  foreign.__exit__(None, None, None)
  assert flags() == before


def test_foreign_order_survives_out_of_order_close_and_base_exception():
  """Tests foreign order survives out-of-order close and base exception."""
  before = flags()
  capture = CaptureMode(Receiver())
  capture.__enter__()
  lower, upper = InfraMode(), IgnoreCompileMode()
  lower.__enter__()
  upper.__enter__()
  try:
    raise KeyboardInterrupt('original')
  except KeyboardInterrupt:
    assert not capture.close()
  assert dispatch._get_current_dispatch_mode_stack() == [lower, upper]
  assert flags() == (True, True, False)
  upper.__exit__(None, None, None)
  assert flags() == (True, False, False)
  lower.__exit__(None, None, None)
  assert flags() == before


def test_repeated_close_does_not_remove_another_mode():
  """Tests that repeated close does not remove another mode."""
  capture = CaptureMode(Receiver())
  capture.__enter__()
  assert capture.close()
  with RecordingMode() as foreign:
    before_flags = flags()
    assert not capture.close()
    assert dispatch._get_current_dispatch_mode_stack() == [foreign]
    assert flags() == before_flags
