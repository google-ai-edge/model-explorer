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

"""Dual-mode (DynamicCache + tuple) KV cache extraction and snapshotting."""

from typing import Any

import torch


def clone_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
  """Detaches, moves to CPU, clones, and makes contiguous a PyTorch tensor.

  Args:
    tensor: Input PyTorch tensor to be safely extracted.

  Returns:
    Contiguous cloned CPU tensor safe for safetensors serialization.
  """
  return tensor.detach().cpu().clone().contiguous()


def get_kv_pair(
    past: Any, layer_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Extracts (key, value) for layer_idx from DynamicCache or tuple.

  Args:
    past: Cache structure (transformers.DynamicCache or tuple of tuples).
    layer_idx: Zero-based transformer layer index.

  Returns:
    A tuple of (key_tensor, value_tensor), or (None, None) if not present.
  """
  if past is None:
    return None, None
  if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
    if layer_idx < len(past.key_cache):
      return past.key_cache[layer_idx], past.value_cache[layer_idx]
    return None, None
  try:
    pair = past[layer_idx]
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
      return pair[0], pair[1]
  except (IndexError, TypeError):
    pass
  return None, None


def get_cache_num_layers(past: Any) -> int:
  """Returns the number of layers stored in past_key_values.

  Args:
    past: Cache structure (DynamicCache, tuple, or None).

  Returns:
    Total layer count stored in the cache.
  """
  if past is None:
    return 0
  if hasattr(past, "key_cache"):
    return len(past.key_cache)
  try:
    return len(past)
  except TypeError:
    return 0


def create_kv_snapshot_with_storage(
    past: Any,
    moment: str,
    forward_id: int,
    processed_token_count: int,
    valid_length: int | None = None,
    snapshot_idx: int = 0,
    shard_name: str = "kv_00000.safetensors",
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
  """Creates a KV snapshot and returns tensors and manifest entries to store.

  Args:
    past: KV cache structure from generation output.
    moment: Lifecycle moment label (e.g. 'prefill_pre', 'prefill_post',
      'terminal').
    forward_id: Identifier of the forward pass associated with this snapshot.
    processed_token_count: Number of cumulative tokens processed up to this
      moment.
    valid_length: Optional sequence length slice to retain from cached
      keys/values.
    snapshot_idx: Zero-based sequence index for naming snapshot_id.
    shard_name: Target safetensors filename for tensor storage.

  Returns:
    A tuple containing:
      - Snapshot metadata dictionary.
      - Dictionary mapping tensor slot names to cloned CPU tensors.
      - List of manifest row dictionaries to record in kv/manifest.jsonl.
  """
  snapshot_id = snapshot_idx
  if past is None:
    snapshot = {
        "snapshot_id": snapshot_id,
        "moment": moment,
        "state": "not_allocated",
        "storage_complete": True,
        "forward_id": forward_id,
        "processed_token_count": processed_token_count,
        "layers": [],
    }
    return snapshot, {}, []

  num_layers = get_cache_num_layers(past)
  layers = []
  tensors_dict = {}
  manifest_rows = []

  for layer_idx in range(num_layers):
    k, v = get_kv_pair(past, layer_idx)
    if k is None or v is None:
      continue
    v_len = valid_length if valid_length is not None else k.shape[-2]
    k_slice = clone_tensor_to_cpu(k[..., :v_len, :])
    v_slice = clone_tensor_to_cpu(v[..., :v_len, :])

    layer_tensors = []
    for kind, tensor_slice in (("key", k_slice), ("value", v_slice)):
      slot = f"kv_{moment}_{forward_id}_{layer_idx}_{kind}"
      key = f"kv:{moment}:{forward_id}:{layer_idx}:{kind}"
      tensors_dict[slot] = tensor_slice
      manifest_rows.append({
          "scope": "kv",
          "shard": shard_name,
          "slot": slot,
          "key": key,
          "shape": list(tensor_slice.shape),
          "dtype": str(tensor_slice.dtype),
          "layer": layer_idx,
          "kind": kind,
          "moment": moment,
          "forward_id": forward_id,
          "snapshot_id": snapshot_id,
      })
      layer_tensors.append({
          "key": key,
          "storage_status": "stored",
          "shape": list(tensor_slice.shape),
          "dtype": str(tensor_slice.dtype),
          "kind": kind,
      })

    layers.append({
        "layer": layer_idx,
        "layer_index": layer_idx,
        "layer_type": "DynamicLayer",
        "state": "available",
        "layout": ["batch", "kv_head", "sequence", "head_dim"],
        "capacity": int(v_len),
        "valid_length": int(v_len),
        "logical_start": 0,
        "logical_end": int(v_len),
        "processed_token_count": int(processed_token_count),
        "tensors": layer_tensors,
    })

  snapshot = {
      "snapshot_id": snapshot_id,
      "moment": moment,
      "state": "available",
      "storage_complete": True,
      "forward_id": forward_id,
      "processed_token_count": processed_token_count,
      "preparation_status": "verified",
      **({"terminal_status": "completed"} if moment == "terminal" else {}),
      "layers": layers,
  }
  return snapshot, tensors_dict, manifest_rows
