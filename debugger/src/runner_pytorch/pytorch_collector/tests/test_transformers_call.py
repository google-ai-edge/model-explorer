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

"""Optional, checkpoint-free coverage of the verified Transformers call wrapper."""

import sys

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
import torch
from torch import nn

transformers = pytest.importorskip(
    'transformers', reason='install the integration dependency group'
)
# pylint: disable=g-import-not-at-top
from transformers.modeling_layers import GradientCheckpointingLayer


class Layer(GradientCheckpointingLayer):
  """Dummy layer extending GradientCheckpointingLayer for testing."""

  def forward(self, x):
    return x + 1


@pytest.mark.parametrize(
    'training,checkpointing', [(False, False), (False, True), (True, False)]
)
def test_transparent_checkpoint_wrapper_keeps_actual_boundaries(
    training, checkpointing
):
  """Tests that transparent checkpoint wrapper preserves boundaries."""
  m = Layer().train(training)
  m.gradient_checkpointing = checkpointing
  with collect(m) as c:
    output = m(torch.ones(2))
  assert m.training is training
  rows = [row for row in c.records if row.op is None]
  assert len(rows) == 2
  assert torch.equal(rows[-1].tensor, output)


def test_active_checkpoint_branch_is_rejected_before_execution():
  """Tests that active checkpoint branch is rejected before execution."""
  m = Layer().train()
  m.gradient_checkpointing = True
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(m):
      pytest.fail('unsupported checkpoint branch entered')
  # pylint: disable=protected-access
  assert not m._forward_hooks and not m._forward_pre_hooks


def test_enabling_checkpoint_branch_during_scope_cannot_publish_results():
  """Tests enabling checkpoint branch during scope cannot publish results."""
  m = Layer().train()
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(m) as c:
      m(torch.ones(1))
      m.gradient_checkpointing = True
  with pytest.raises(CaptureError):
    _ = c.records


def test_another_transformers_version_is_not_assumed_compatible(monkeypatch):
  """Tests another transformers version is not assumed compatible."""
  monkeypatch.setattr(sys.modules['transformers'], '__version__', '0.0.0')
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(Layer()):
      pass


def test_checkpoint_wrapper_cannot_hide_another_call_override_in_its_mro():
  """Tests that wrapper cannot hide another call override in MRO."""

  class After(nn.Module):
    """Module overriding __call__ after Layer."""

    def __call__(self, x):
      return super().__call__(x) + 10

  class Mixed(Layer, After):
    """Mixed inheritance hierarchy."""

  m = Mixed().eval()
  assert torch.equal(m(torch.ones(1)), torch.tensor([12.0]))
  with pytest.raises(CaptureError, match='custom __call__'):
    with collect(m):
      pass
