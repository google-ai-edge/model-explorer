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

"""A rejected foreign-thread exit must leave owner-thread cleanup possible."""

# pylint: disable=protected-access

from concurrent import futures

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
import torch
from torch import nn
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.fixture(autouse=True)
def recover_test_thread_after_a_regression():
  # A broken foreign-thread exit can strand native dispatch state. Keep a
  # failing regression from corrupting the remaining tests' thread state.
  before_stack = _get_current_dispatch_mode_stack()
  before_grad = torch.is_grad_enabled()
  yield
  stack = _get_current_dispatch_mode_stack()
  while len(stack) > len(before_stack):
    stack[-1].__exit__(None, None, None)
    stack = _get_current_dispatch_mode_stack()
  torch.set_grad_enabled(before_grad)


class RecordingMode(TorchDispatchMode):

  def __init__(self):
    super().__init__()
    self.operations = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.operations.append(str(func))
    return func(*args, **(kwargs or {}))


def hook_state(model):
  return tuple(
      tuple(getattr(model, name).items())
      for name in (
          '_forward_pre_hooks',
          '_forward_pre_hooks_with_kwargs',
          '_forward_hooks',
          '_forward_hooks_with_kwargs',
          '_forward_hooks_always_called',
      )
  )


def request_foreign_exit(capture, with_error, grad_enabled):
  """Requests an exit from a foreign thread and checks state consistency."""
  with torch.set_grad_enabled(grad_enabled), RecordingMode() as foreign:
    before_stack = _get_current_dispatch_mode_stack()
    error = ValueError('foreign scope exception') if with_error else None
    with pytest.raises(CaptureError, match='entering Python thread'):
      capture.__exit__(type(error) if error else None, error, None)
    result = {
        'grad_unchanged': torch.is_grad_enabled() == grad_enabled,
        'stack_unchanged': _get_current_dispatch_mode_stack() == before_stack,
    }
    torch.ones(1)
    result['foreign_mode_usable'] = foreign.operations == ['aten.ones.default']
    return result


@pytest.mark.parametrize('with_error', [False, True])
@pytest.mark.parametrize('already_failed', [False, True])
@pytest.mark.parametrize('worker_grad', [False, True])
def test_foreign_exit_keeps_owner_resources_until_owner_unwinds(
    with_error, already_failed, worker_grad
):
  """Ensures owner resources remain intact after a rejected foreign exit."""

  class Model(nn.Module):
    fail = False

    def forward(self, value):
      if self.fail:
        raise ValueError('model failure')
      return value

  model = Model()
  before_hooks, before_stack, before_grad = (
      hook_state(model),
      _get_current_dispatch_mode_stack(),
      torch.is_grad_enabled(),
  )
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    with pytest.raises(CaptureError):
      with collect(model) as capture:
        model(torch.ones(1))
        if already_failed:
          model.fail = True
          with pytest.raises(ValueError, match='model failure'):
            model(torch.ones(1))
        owner_hooks = hook_state(model)
        owner_mode, owner_grad = capture._mode, capture._grad
        state = executor.submit(
            request_foreign_exit, capture, with_error, worker_grad
        ).result()
        assert all(state.values()), state
        assert capture._state == 'active'
        assert capture._mode is owner_mode and capture._grad is owner_grad
        assert _get_current_dispatch_mode_stack() == before_stack + [owner_mode]
        assert hook_state(model) == owner_hooks
        assert not torch.is_grad_enabled()
        assert not capture._pending
        with pytest.raises(CaptureError, match='already owned'):
          with collect(model):
            pass
        with pytest.raises(CaptureError):
          _ = capture.records
        with pytest.raises(CaptureError):
          model(torch.ones(1))
  assert hook_state(model) == before_hooks
  assert _get_current_dispatch_mode_stack() == before_stack
  assert torch.is_grad_enabled() == before_grad
  with pytest.raises(CaptureError):
    _ = capture.records
  model.fail = False
  with collect(model) as retry:
    model(torch.ones(1))
  assert len(retry.records) == 2


def test_owner_exception_wins_after_foreign_exit_request():
  """Ensures owner thread exception takes precedence after foreign exit."""
  original = KeyboardInterrupt('owner exception')
  model = nn.Identity()
  before_hooks = hook_state(model)
  # Enter manually so a regression cannot pop the leaked collector while
  # trying to exit the outer mode before the fixture repairs the test state.
  outer = RecordingMode()
  outer.__enter__()
  with futures.ThreadPoolExecutor(max_workers=1) as executor, torch.no_grad():
    before_stack = _get_current_dispatch_mode_stack()
    with pytest.raises(KeyboardInterrupt) as caught:
      with collect(model) as capture:
        model(torch.ones(1))
        state = executor.submit(
            request_foreign_exit, capture, True, False
        ).result()
        assert all(state.values()), state
        raise original
    assert caught.value is original
    assert _get_current_dispatch_mode_stack() == before_stack
    assert not torch.is_grad_enabled()
    assert hook_state(model) == before_hooks
    torch.zeros(1)
    assert outer.operations[-1] == 'aten.zeros.default'
  outer.__exit__(None, None, None)
  with pytest.raises(CaptureError):
    _ = capture.records
