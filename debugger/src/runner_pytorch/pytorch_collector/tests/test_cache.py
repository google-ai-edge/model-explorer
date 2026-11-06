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

"""Tests for dynamic cache capture integration with HuggingFace Transformers."""

import collections
import subprocess
import sys

import pytest
from pytorch_collector._snapshots import Snapshotter
import torch

# Runtime remains torch-only. Run this compatibility suite with the
# integration dependency group; release acceptance must execute it in the
# pinned environment.
pytest.importorskip(
    "transformers",
    reason="Cache tests require the integration dependency group",
)
# pylint: disable=g-import-not-at-top
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.cache_utils import DynamicSlidingWindowLayer
from transformers.cache_utils import LinearAttentionLayer
from transformers.cache_utils import StaticLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

# pylint: enable=g-import-not-at-top


def _hybrid_cache():
  """Returns a DynamicCache configured for hybrid Gemma4 layers."""
  config = Gemma4TextConfig(
      num_hidden_layers=4,
      layer_types=[
          "sliding_attention",
          "full_attention",
          "sliding_attention",
          "full_attention",
      ],
      num_kv_shared_layers=2,
      sliding_window=3,
  )
  return DynamicCache(config=config)


def _kv(length=2):
  keys = torch.arange(length * 2, dtype=torch.bfloat16).reshape(1, 1, length, 2)
  return keys, keys + 10


def test_empty_lazy_cache_and_uninitialized_gemma_layers_do_not_initialize_during_capture():
  """Tests that uninitialized cache layers do not initialize during capture."""
  snapshotter = Snapshotter([], 8)
  assert not snapshotter.snapshot(DynamicCache(), ("kwargs", "past_key_values"))
  cache = _hybrid_cache()
  assert [type(layer) for layer in cache.layers] == [
      DynamicSlidingWindowLayer,
      DynamicLayer,
  ]

  rows = snapshotter.snapshot(cache, ("kwargs", "past_key_values"))

  assert [path for path, _ in rows] == [
      ("kwargs", "past_key_values", "layers", 0, "_sliding_window_tensor")
  ]
  assert rows[0][1].item() == 3
  assert rows[0][1].dtype == torch.int64
  assert all(
      layer.keys is None and layer.values is None and not layer.is_initialized
      for layer in cache.layers
  )
  assert snapshotter.bytes_used == 8


def test_input_snapshot_keeps_observed_values_across_cache_update_and_later_mutation():
  """Tests that snapshots retain observed values across cache mutations."""
  cache = DynamicCache()
  keys, values = _kv()
  cache.update(keys, values, 0)
  snapshotter = Snapshotter([], 1000)

  before = dict(snapshotter.snapshot(cache, ("kwargs", "past_key_values")))
  cache.update(*_kv(length=1), 0)
  after = dict(snapshotter.snapshot(cache, ("output", "past_key_values")))
  cache.layers[0].keys.fill_(-7)
  cache.layers[0].values.fill_(-9)

  key_path = ("kwargs", "past_key_values", "layers", 0, "keys")
  value_path = ("kwargs", "past_key_values", "layers", 0, "values")
  torch.testing.assert_close(before[key_path], keys)
  torch.testing.assert_close(before[value_path], values)
  assert (
      after[("output", "past_key_values", "layers", 0, "keys")].shape[-2] == 3
  )
  assert before[key_path].dtype == torch.bfloat16
  assert not torch.any(
      after[("output", "past_key_values", "layers", 0, "keys")] == -7
  )


def test_sliding_storage_and_shared_full_kv_are_captured_separately():
  """Tests that sliding window cache and shared KV states are captured."""
  cache = _hybrid_cache()
  full_keys, full_values = cache.update(*_kv(length=4), 0)
  shared = collections.UserDict({"sliding_attention": (full_keys, full_values)})
  snapshotter = Snapshotter([], 1000)

  rows = dict(
      snapshotter.snapshot(
          {"past_key_values": cache, "shared_kv_states": shared}, ("kwargs",)
      )
  )

  stored = rows[("kwargs", "past_key_values", "layers", 0, "keys")]
  shared_keys = rows[("kwargs", "shared_kv_states", "sliding_attention", 0)]
  shared_values = rows[("kwargs", "shared_kv_states", "sliding_attention", 1)]
  assert stored.shape[-2] == 2
  assert shared_keys.shape[-2] == shared_values.shape[-2] == 4
  torch.testing.assert_close(stored, full_keys[..., -2:, :])
  full_keys.fill_(-1)
  assert not torch.any(shared_keys == -1)
  assert not torch.any(stored == -1)


def test_model_output_mapping_preserves_nested_cache_field_paths():
  cache = DynamicCache()
  cache.update(*_kv(), 0)
  output = BaseModelOutputWithPast(
      last_hidden_state=torch.ones(1, 2), past_key_values=cache
  )
  rows = Snapshotter([], 1000).snapshot(output, ("output",))
  assert [path for path, _ in rows] == [
      ("output", "last_hidden_state"),
      ("output", "past_key_values", "layers", 0, "keys"),
      ("output", "past_key_values", "layers", 0, "values"),
  ]


def test_repeated_cache_and_layer_references_each_get_independent_snapshots():
  cache = DynamicCache()
  cache.update(*_kv(), 0)
  cache.layers.append(cache.layers[0])
  rows = Snapshotter([], 1000).snapshot([cache, cache], ())
  assert len(rows) == 8
  rows[0][1].zero_()
  assert rows[2][1].flatten()[-1].item() == 3
  assert rows[4][1].flatten()[-1].item() == 3
  assert rows[0][0] == (0, "layers", 0, "keys")
  assert rows[4][0] == (1, "layers", 0, "keys")


def test_cache_cycle_fails_and_rolls_back_this_snapshot_budget():
  cache = DynamicCache()
  cache.update(*_kv(), 0)
  cache.layers.append(cache)
  snapshotter = Snapshotter([], 1000)
  with pytest.raises(ValueError, match="Cyclic"):
    snapshotter.snapshot(cache, ())
  assert snapshotter.bytes_used == 0


def test_cache_tensors_reuse_parameter_filter_and_cumulative_budget():
  """Tests that cache tensors reuse parameter filter and cumulative budget."""
  parameter = torch.nn.Parameter(torch.arange(4.0).reshape(1, 1, 2, 2))
  cache = DynamicCache()
  cache.layers = [DynamicLayer()]
  cache.layers[0].keys = parameter.detach()
  cache.layers[0].values = torch.ones_like(parameter)
  snapshotter = Snapshotter([parameter], 16)
  rows = snapshotter.snapshot(cache, ())
  assert [path for path, _ in rows] == [("layers", 0, "values")]
  assert snapshotter.bytes_used == 16
  with pytest.raises(MemoryError):
    snapshotter.snapshot(cache, ())
  assert snapshotter.bytes_used == 16


@pytest.mark.parametrize("version", ["5.16.0", "5.17.1", "5.17.0.dev0", None])
def test_unverified_transformers_versions_are_rejected(monkeypatch, version):
  # Transformers can replace its LazyModule during later test collection.
  # Patch the live module that the runtime checks, not an earlier import.
  monkeypatch.setattr(sys.modules["transformers"], "__version__", version)
  with pytest.raises(TypeError, match="requires transformers==5.17.0"):
    Snapshotter([], 100).snapshot(DynamicCache(), ())


def test_unverified_cache_version_does_not_restrict_ordinary_mapping_capture(
    monkeypatch,
):
  """Tests that unverified cache versions allow normal mapping capture."""
  cache = DynamicCache()
  cache_module = sys.modules["transformers.cache_utils"]
  monkeypatch.setattr(sys.modules["transformers"], "__version__", "4.57.6")
  monkeypatch.delattr(cache_module, "LinearAttentionCacheLayerMixin")
  snapshotter = Snapshotter([], 100)
  mapping = collections.UserDict({"x": [torch.ones(1)]})
  assert len(snapshotter.snapshot(mapping, ())) == 1
  with pytest.raises(TypeError, match="requires transformers==5.17.0"):
    snapshotter.snapshot(cache, ())


def test_cache_subclasses_and_other_cache_types_are_rejected():
  class CustomCache(DynamicCache):
    pass

  class MappingCache(DynamicCache, dict):
    pass

  for cache in [CustomCache(), MappingCache(), Cache(layers=[DynamicLayer()])]:
    with pytest.raises(TypeError, match="Unsupported Cache type"):
      Snapshotter([], 100).snapshot(cache, ())


def test_unknown_internal_layer_and_layer_subclass_are_rejected():
  class CustomLayer(DynamicLayer):
    pass

  for layer in [CustomLayer(), StaticLayer(max_cache_len=3)]:
    cache = DynamicCache()
    cache.layers = [layer]
    with pytest.raises(TypeError, match="Unsupported Cache type"):
      Snapshotter([], 100).snapshot(cache, ())


@pytest.mark.parametrize(
    "layer", [None, 1, "layer", torch.ones(1), [], {}, object()]
)
def test_non_layer_entries_cannot_silently_use_ordinary_container_traversal(
    layer,
):
  cache = DynamicCache()
  cache.layers = [layer]
  with pytest.raises(
      TypeError, match=r"Unsupported Cache type in DynamicCache.layers\[0\]"
  ):
    Snapshotter([], 100).snapshot(cache, ())


def test_self_referencing_cache_layer_list_fails_as_a_cycle():
  cache = DynamicCache()
  cache.layers.append(cache.layers)
  with pytest.raises(ValueError, match="Cyclic"):
    Snapshotter([], 100).snapshot(cache, ())


def test_unknown_future_lazy_layer_type_is_rejected_even_when_cache_is_empty():
  class CustomLayer(DynamicLayer):
    pass

  cache = DynamicCache()
  cache.layer_class_to_replicate = CustomLayer
  with pytest.raises(TypeError, match="layer_class_to_replicate"):
    Snapshotter([], 100).snapshot(cache, ())


def test_offloading_and_malformed_known_tensor_fields_fail_explicitly():
  cache = DynamicCache()
  cache.offloading = True
  with pytest.raises(TypeError, match="Offloaded"):
    Snapshotter([], 100).snapshot(cache, ())
  cache.offloading = False
  cache.layers = [DynamicLayer()]
  cache.layers[0].keys = {"unexpected": torch.ones(1)}
  with pytest.raises(TypeError, match="keys must be a Tensor or None"):
    Snapshotter([], 100).snapshot(cache, ())


def test_importing_and_using_collector_does_not_import_transformers():
  """Tests that importing and using collector does not import transformers."""
  command = (
      "import sys, torch; from pytorch_collector import collect; model ="
      " torch.nn.Identity(); capture = collect(model); capture.__enter__();"
      " model(torch.ones(1)); capture.__exit__(None, None, None); assert"
      " 'transformers' not in sys.modules"
  )
  result = subprocess.run(
      [
          sys.executable,
          "-c",
          command,
      ],
      check=False,
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stderr


def test_linear_attention_layer_states_are_captured_by_state_index():
  """Tests that LinearAttentionLayer states are captured by state index."""
  cache = DynamicCache()
  cache.layers.append(LinearAttentionLayer())
  layer = cache.layers[0]
  layer.update_conv_state(torch.arange(6, dtype=torch.float32).reshape(1, 2, 3))
  snapshotter = Snapshotter([], 1000)

  rows = dict(snapshotter.snapshot(cache, ("kwargs", "past_key_values")))

  assert list(rows) == [
      ("kwargs", "past_key_values", "layers", 0, "conv_states", 0)
  ]
  assert torch.equal(
      rows[("kwargs", "past_key_values", "layers", 0, "conv_states", 0)],
      layer.conv_states[0],
  )
  layer.update_recurrent_state(torch.ones(1, 2, 2))
  rows = dict(snapshotter.snapshot(cache, ("output", "past_key_values")))
  assert (
      "output",
      "past_key_values",
      "layers",
      0,
      "recurrent_states",
      0,
  ) in rows
  layer.conv_states = [layer.conv_states[0]]
  with pytest.raises(TypeError, match="conv_states must be a dict"):
    snapshotter.snapshot(cache, ("kwargs", "past_key_values"))
