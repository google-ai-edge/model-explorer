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

"""Picks each generated token from a row of logits.

Greedy, or sampling with temperature, top-k and top-p, as a chat's options
say; an option the chat leaves unset is the model's own default, from its
generation_config. Sampling draws from a CPU generator seeded once per chat,
so a seed replays its chat.
"""

import dataclasses

from model_debugger.protocol import runner_pb2
import torch


@dataclasses.dataclass(frozen=True)
class Defaults:
  """A model's own sampling, from its generation_config.

  Attributes:
      do_sample: Whether the model samples; greedy if not.
      temperature: Its temperature, or None.
      top_k: Its top-k, or None.
      top_p: Its top-p, or None.
  """

  do_sample: bool
  temperature: float | None
  top_k: int | None
  top_p: float | None

  def proto(self) -> runner_pb2.SamplerDefaults:
    """Returns the defaults as LoadResponse reports them."""
    defaults = runner_pb2.SamplerDefaults()
    if not self.do_sample:
      defaults.type = runner_pb2.SamplerDefaults.GREEDY
      return defaults
    top_p = self.top_p is not None and self.top_p < 1
    defaults.type = (
        runner_pb2.SamplerDefaults.TOP_P
        if top_p
        else runner_pb2.SamplerDefaults.TOP_K
    )
    if self.temperature is not None:
      defaults.temperature = self.temperature
    if self.top_k is not None:
      defaults.top_k = self.top_k
    if self.top_p is not None:
      defaults.top_p = self.top_p
    return defaults


class Sampler:
  """The sampling of one chat."""

  def __init__(
      self, options: runner_pb2.ChatOptions, defaults: Defaults
  ) -> None:
    """Takes the chat's options, the model's defaults where they are unset.

    Args:
        options: The chat's options, checked to be in range.
        defaults: The model's own sampling.
    """

    def pick(field: str, default: float | int | None) -> float | int:
      return getattr(options, field) if options.HasField(field) else default

    self.temperature = pick('temperature', defaults.temperature)
    self.top_k = pick('top_k', defaults.top_k)
    self.top_p = pick('top_p', defaults.top_p)
    chat_samples = any(
        options.HasField(field) for field in ('temperature', 'top_k', 'top_p')
    )
    self.greedy = (
        (not defaults.do_sample and not chat_samples)
        or self.top_k == 1
        or self.temperature == 0
    )
    seed = options.seed if options.HasField('seed') else 0
    self._generator = torch.Generator(device='cpu')
    self._generator.manual_seed(seed & 0xFFFF_FFFF_FFFF_FFFF)

  def choose(self, logits: torch.Tensor) -> int:
    """Returns the token picked from one row of logits.

    Args:
        logits: The row, of any dtype and device.
    """
    scores = logits.reshape(-1).detach().float().cpu()
    if self.greedy:
      return int(scores.argmax())
    if self.temperature is not None:
      scores = scores / self.temperature
    if self.top_k is not None:
      top_k_count = min(self.top_k, scores.numel())
      scores[scores < torch.topk(scores, top_k_count).values[-1]] = -float(
          'inf'
      )
    if self.top_p is not None and self.top_p < 1:
      ordered, indices = torch.sort(scores, descending=True)
      remove = torch.softmax(ordered, dim=-1).cumsum(-1) > self.top_p
      # The token that crosses top_p stays, and so the first always.
      remove[1:] = remove[:-1].clone()
      remove[0] = False
      scores[indices[remove]] = -float('inf')
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=self._generator))
