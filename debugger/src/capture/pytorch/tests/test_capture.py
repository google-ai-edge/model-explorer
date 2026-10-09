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
# ==============================================================================

"""Unit tests for the ai_edge_debugger_pytorch package."""

import json
import pathlib
import sys
import tempfile
import types
import unittest

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent / "package"
if str(_PACKAGE_DIR) not in sys.path:
  sys.path.insert(0, str(_PACKAGE_DIR))

import safetensors.torch
import torch
import torch.nn as nn

from ai_edge_debugger_pytorch import capture
from ai_edge_debugger_pytorch import kv_cache
from ai_edge_debugger_pytorch import writer


class _MockLayer(nn.Module):

  def __init__(self):
    super().__init__()
    self.self_attn = nn.Linear(16, 16)
    self.mlp = nn.Linear(16, 16)
    self.input_layernorm = nn.LayerNorm(16)
    self.post_attention_layernorm = nn.LayerNorm(16)

  def forward(self, x):
    x = self.input_layernorm(x)
    x = x + self.self_attn(x)
    x = self.post_attention_layernorm(x)
    x = x + self.mlp(x)
    return x


class _MockModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.config = types.SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=2,
    )
    self.model = types.SimpleNamespace(
        layers=nn.ModuleList([_MockLayer(), _MockLayer()])
    )

  def forward(self, input_ids=None, past_key_values=None):
    if input_ids is None:
      input_ids = torch.zeros((1, 4), dtype=torch.long)
    x = torch.randn((input_ids.shape[0], input_ids.shape[1], 16))
    for layer in self.model.layers:
      x = layer(x)
    logits = torch.randn((input_ids.shape[0], input_ids.shape[1], 32))
    return types.SimpleNamespace(logits=logits, past_key_values=past_key_values)


class KVCacheTests(unittest.TestCase):

  def test_clone_tensor_to_cpu(self):
    t = torch.randn(2, 4)
    cloned = kv_cache.clone_tensor_to_cpu(t)
    self.assertEqual(cloned.device.type, "cpu")
    self.assertTrue(cloned.is_contiguous())
    self.assertTrue(torch.equal(t, cloned))

  def test_tuple_kv_extraction(self):
    k = torch.randn(1, 4, 16)
    v = torch.randn(1, 4, 16)
    past = ((k, v),)
    self.assertEqual(kv_cache.get_cache_num_layers(past), 1)
    ret_k, ret_v = kv_cache.get_kv_pair(past, 0)
    self.assertIsNotNone(ret_k)
    self.assertIsNotNone(ret_v)
    self.assertTrue(torch.equal(k, ret_k))

  def test_dynamic_cache_kv_extraction(self):
    k = torch.randn(1, 4, 16)
    v = torch.randn(1, 4, 16)
    past = types.SimpleNamespace(key_cache=[k], value_cache=[v])
    self.assertEqual(kv_cache.get_cache_num_layers(past), 1)
    ret_k, ret_v = kv_cache.get_kv_pair(past, 0)
    self.assertTrue(torch.equal(k, ret_k))

  def test_create_kv_snapshot_with_storage(self):
    k = torch.randn(1, 4, 16)
    v = torch.randn(1, 4, 16)
    past = ((k, v),)
    snap, tensors, rows = kv_cache.create_kv_snapshot_with_storage(
        past=past,
        moment="prefill_post",
        forward_id=0,
        processed_token_count=4,
    )
    self.assertEqual(snap["moment"], "prefill_post")
    self.assertEqual(len(rows), 2)
    self.assertIn("kv_prefill_post_0_0_key", tensors)
    self.assertIn("kv_prefill_post_0_0_value", tensors)


class WriterTests(unittest.TestCase):

  def test_manifest_and_shard_writing(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      out_dir = pathlib.Path(tmp_dir)
      tensors = {"t1": torch.randn(2, 2)}
      rows = [{"slot": "t1", "shape": [2, 2], "dtype": "torch.float32"}]

      writer.write_module_shard(
          out_dir, "shard-00000.safetensors", tensors, rows
      )
      self.assertTrue((out_dir / "shard-00000.safetensors").is_file())
      self.assertTrue((out_dir / "manifest.jsonl").is_file())

      writer.append_boundary_manifest(out_dir, rows)
      self.assertTrue((out_dir / "boundaries/manifest.jsonl").is_file())

      writer.write_boundary_shard(out_dir, "boundaries.safetensors", tensors)
      self.assertTrue((out_dir / "boundaries/boundaries.safetensors").is_file())

      writer.write_forward_index(out_dir, [{"forward_id": 0}])
      self.assertTrue((out_dir / "forward_index.json").is_file())

      writer.write_tokens(out_dir, [{"token_ids": [1, 2]}])
      self.assertTrue((out_dir / "tokens.jsonl").is_file())

      writer.write_generation(out_dir, {"status": "completed"})
      self.assertTrue((out_dir / "generation.json").is_file())

      writer.write_kv_index(out_dir, [{"snapshot_id": "snapshot_0"}])
      self.assertTrue((out_dir / "kv_index.json").is_file())


class CaptureRunTests(unittest.TestCase):

  def test_detect_topology(self):
    model = _MockModel()
    name, blocks, n_layers, width = capture.detect_topology(model)
    self.assertEqual(name, "model.layers")
    self.assertEqual(len(blocks), 2)
    self.assertEqual(n_layers, 2)
    self.assertEqual(width, 16)

  def test_capture_run_lifecycle(self):
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
      out_dir = pathlib.Path(tmp_dir)
      cap = capture.CaptureRun(
          model=model,
          granularity="sublayer",
          layers="0",
          out_dir=out_dir,
      )
      self.assertEqual(len(cap.sites), 5)  # 1 layer + 4 sublayers

      with cap:
        # Forward pass 0 (prefill)
        with cap.forward_context(phase="prefill", step=0, pos_offset=0):
          input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
          k = torch.randn(1, 4, 16)
          v = torch.randn(1, 4, 16)
          output = model(input_ids=input_ids, past_key_values=((k, v),))

        cap.observe_tokens([50], forward_id=0)

        # Forward pass 1 (decode)
        with cap.forward_context(phase="decode", step=1, pos_offset=4):
          decode_ids = torch.tensor([[50]], dtype=torch.long)
          k2 = torch.randn(1, 5, 16)
          v2 = torch.randn(1, 5, 16)
          output = model(input_ids=decode_ids, past_key_values=((k2, v2),))

        cap.observe_tokens([60], forward_id=1)
        cap.finish_generation("completed")

      # Verify artifacts on disk
      self.assertTrue((out_dir / "shard-00000.safetensors").is_file())
      self.assertTrue((out_dir / "shard-00001.safetensors").is_file())
      self.assertTrue((out_dir / "manifest.jsonl").is_file())
      self.assertTrue((out_dir / "boundaries/manifest.jsonl").is_file())
      self.assertTrue((out_dir / "boundaries/boundaries.safetensors").is_file())
      self.assertTrue((out_dir / "forward_index.json").is_file())
      self.assertTrue((out_dir / "kv_index.json").is_file())
      self.assertTrue((out_dir / "tokens.jsonl").is_file())
      self.assertTrue((out_dir / "generation.json").is_file())

      # Verify: 5 sites * 2 (in/out) = 10 module tensors per forward pass.
      module_rows = [
          json.loads(line)
          for line in (out_dir / "manifest.jsonl").read_text().splitlines()
          if line.strip()
      ]
      self.assertEqual(len(module_rows), 20)  # 10 for prefill + 10 for decode

      # Verify boundary tensors in boundaries.safetensors
      with safetensors.torch.safe_open(
          out_dir / "boundaries/boundaries.safetensors", framework="pt"
      ) as f:
        keys = set(f.keys())
        self.assertIn("boundary_in_0_input_ids", keys)
        self.assertIn("boundary_out_0_logits", keys)
        self.assertIn("boundary_in_1_input_ids", keys)
        self.assertIn("boundary_out_1_logits", keys)

      # Verify KV snapshots: prefill_pre, prefill_post, and terminal
      kv_index = json.loads((out_dir / "kv_index.json").read_text())
      moments = [s["moment"] for s in kv_index["snapshots"]]
      self.assertEqual(moments, ["prefill_pre", "prefill_post", "terminal"])


if __name__ == "__main__":
  unittest.main()
