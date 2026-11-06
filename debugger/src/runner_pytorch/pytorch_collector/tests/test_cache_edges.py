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

"""Tests for cache edge attribution in pytorch_collector."""

# pylint: disable=redefined-outer-name

import pytest
from pytorch_collector import collect
import torch

# Runtime remains torch-only. Run this compatibility suite with the integration
# dependency group; release acceptance must execute it in the pinned
# environment.
pytest.importorskip(
    "transformers",
    reason="Cache edge tests require the integration dependency group",
)
# pylint: disable=g-import-not-at-top
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

# pylint: enable=g-import-not-at-top

# Entry operators that may legitimately consume a tensor no operator produced.
CONSTANT_ENTRY_OPS = {
    "aten.lift_fresh.default"
}  # torch.tensor(...) literals in model code


@pytest.fixture(scope="module")
def prefill_and_decode():
  """Fixture providing a capture of prefill and decode turns."""
  torch.manual_seed(0)
  config = LlamaConfig(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=2,
      vocab_size=32,
      max_position_embeddings=64,
      attn_implementation="eager",
  )
  model = LlamaForCausalLM(config).eval()
  cache = DynamicCache(config=config)
  with collect(model) as capture:
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        past_key_values=cache,
        use_cache=True,
    )
    token = output.logits[:, -1:].argmax(-1)  # Operators between root calls.
    model(input_ids=token, past_key_values=cache, use_cache=True)
  return capture


def test_decode_reads_the_keys_and_values_that_prefill_wrote(
    prefill_and_decode,
):
  """Tests decode reads KV cache written during prefill."""
  capture = prefill_and_decode
  ops = {op.call_id: op for op in capture.ops}
  carried = [
      edge
      for edge in capture.edges
      if edge.forward_id == 1
      and edge.kind == "op"
      and edge.source_forward_id == 0
  ]

  # Keys and values of both layers: the decode concatenation extends prefill's.
  assert len(carried) == 4
  for edge in carried:
    consumer, producer = ops[edge.call_id], ops[edge.source_call_id]
    assert (consumer.op, producer.op, edge.path) == (
        "aten.cat.default",
        "aten.cat.default",
        ("args", 0, 0),
    )
    assert consumer.module == producer.module and consumer.module.endswith(
        "self_attn"
    )
    assert (
        producer.outputs[0].shape[2] == 3 and consumer.outputs[0].shape[2] == 4
    )


def test_every_input_is_attributed_and_constants_enter_only_through_known_operators(
    prefill_and_decode,
):
  """Tests all inputs are attributed and constants enter via known ops."""
  capture = prefill_and_decode
  ops = {op.call_id: op for op in capture.ops}

  ports_list = []
  for op in capture.ops:
    for port in op.inputs:
      ports_list.append((op.call_id, port.path))
  ports = sorted(ports_list)
  assert ports == sorted((edge.call_id, edge.path) for edge in capture.edges)

  outputs = set()
  for op in capture.ops:
    for port in op.outputs:
      outputs.add((op.call_id, port.path))
  assert {
      (e.source_call_id, e.source_path) for e in capture.edges if e.kind == "op"
  } <= outputs
  assert {
      ops[e.call_id].op for e in capture.edges if e.kind == "constant"
  } <= CONSTANT_ENTRY_OPS
  # The next token was computed between root calls but enters as the root input.
  assert not [edge for edge in capture.edges if edge.kind == "external"]
  names = {edge.name for edge in capture.edges if edge.kind == "param"}
  assert (
      "model.embed_tokens.weight" in names
      and "model.layers.1.mlp.down_proj.weight" in names
  )


def test_both_decoder_layers_have_the_same_operator_sequence(
    prefill_and_decode,
):
  """Tests both decoder layers execute the identical operator sequence."""
  capture = prefill_and_decode
  for forward_id in (0, 1):
    layers = [
        [
            op.op
            for op in capture.ops
            if op.forward_id == forward_id
            and op.module.startswith(f"model.layers.{index}")
        ]
        for index in (0, 1)
    ]
    assert layers[0] and layers[0] == layers[1]
