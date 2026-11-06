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

"""The sampler: greedy, the model's defaults, and a seed that replays."""

from model_debugger.protocol import runner_pb2
from model_debugger.runner_pytorch.llm import sampler
from model_debugger.runner_pytorch.testing import test_main
import torch

_DEFAULTS = sampler.Defaults(
    do_sample=True, temperature=1.0, top_k=64, top_p=0.95
)


def test_top_k_1_is_greedy():
  """Top k 1 is greedy."""
  chosen = sampler.Sampler(
      runner_pb2.ChatOptions(max_completion_tokens=1, top_k=1), _DEFAULTS
  )
  assert chosen.greedy
  assert chosen.choose(torch.tensor([0.0, 3.0, 1.0])) == 1


def test_unset_options_are_the_models_defaults():
  """Unset options are the models defaults."""
  chosen = sampler.Sampler(
      runner_pb2.ChatOptions(max_completion_tokens=1), _DEFAULTS
  )
  assert not chosen.greedy
  assert (chosen.temperature, chosen.top_k, chosen.top_p) == (1.0, 64, 0.95)


def test_a_seed_replays_its_samples():
  """A seed replays its samples."""
  options = runner_pb2.ChatOptions(
      max_completion_tokens=1, seed=7, temperature=2.0
  )
  logits = torch.randn(100, generator=torch.Generator().manual_seed(0))
  first = sampler.Sampler(options, _DEFAULTS)
  second = sampler.Sampler(options, _DEFAULTS)
  picks = [first.choose(logits) for _ in range(20)]
  assert picks == [second.choose(logits) for _ in range(20)]
  # top_k 64 of 100: only the 64 largest are ever picked.
  allowed = set(torch.topk(logits, 64).indices.tolist())
  assert set(picks) <= allowed


if __name__ == '__main__':
  test_main.main()
