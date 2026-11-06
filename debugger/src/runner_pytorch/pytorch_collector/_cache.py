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

"""Explicit fields for the optional, pinned Transformers dynamic cache types.

This module does not import Transformers. A real cache object means its defining
module is already loaded; ordinary captures keep PyTorch as their only runtime
dependency. These fields cover Transformers 5.17.0's non-offloaded DynamicCache,
DynamicLayer, DynamicSlidingWindowLayer and LinearAttentionLayer (the conv and
recurrent states of Qwen3.5's gated-DeltaNet layers). Other cache
implementations
must be reviewed before they can be expanded here.
"""

import sys
from typing import Any

from torch import Tensor

TRANSFORMERS_VERSION = "5.17.0"


def cache_fields(value: Any) -> tuple[tuple[str, Any], ...] | None:
  """Return known borrowed fields, or None for an ordinary non-cache value."""
  # pylint: disable=line-too-long
  cache_module = sys.modules.get("transformers.cache_utils")
  # pylint: enable=line-too-long
  if cache_module is None:
    return None
  # Earlier versions may not define every mixin. Looking at an ordinary list
  # or Mapping must remain valid even when such a version happens to be loaded.
  cache_bases = tuple(
      base
      for name in ("Cache", "CacheLayerMixin", "LinearAttentionCacheLayerMixin")
      if isinstance(base := getattr(cache_module, name, None), type)
  )
  if not isinstance(value, cache_bases):
    return None

  package = sys.modules.get("transformers")
  version = getattr(package, "__version__", None)
  if version != TRANSFORMERS_VERSION:
    raise TypeError(
        f"Cache capture requires transformers=={TRANSFORMERS_VERSION}; loaded"
        f" version is {version!r}"
    )

  kind = type(value)
  if kind is cache_module.DynamicCache:
    if value.offloading:
      raise TypeError("Offloaded DynamicCache is not supported")
    if not isinstance(value.layers, list):
      raise TypeError("DynamicCache.layers must be a list")
    if value.layer_class_to_replicate not in (None, cache_module.DynamicLayer):
      raise TypeError("Unsupported DynamicCache layer_class_to_replicate")
    for index, layer in enumerate(value.layers):
      if layer is value or layer is value.layers:
        raise ValueError(f"Cyclic DynamicCache.layers at index {index}")
      if type(layer) not in (
          cache_module.DynamicLayer,
          cache_module.DynamicSlidingWindowLayer,
          getattr(cache_module, "LinearAttentionLayer", None),
      ):
        raise TypeError(
            f"Unsupported Cache type in DynamicCache.layers[{index}]: "
            f"{type(layer).__module__}.{type(layer).__qualname__}"
        )
    return (("layers", value.layers),)

  if kind is cache_module.DynamicLayer:
    fields = (("keys", value.keys), ("values", value.values))
  elif kind is cache_module.DynamicSlidingWindowLayer:
    sliding_window_tensor = getattr(value, "_sliding_window_tensor")
    fields = (
        ("keys", value.keys),
        ("values", value.values),
        ("_sliding_window_tensor", sliding_window_tensor),
    )
    if not isinstance(sliding_window_tensor, Tensor):
      raise TypeError(
          "DynamicSlidingWindowLayer._sliding_window_tensor must be a Tensor"
      )
  elif kind is getattr(cache_module, "LinearAttentionLayer", None):
    # One conv and one recurrent state per state index, updated in place
    # by copy_(); an uninitialized state is None and yields nothing.
    for name in ("conv_states", "recurrent_states"):
      states = getattr(value, name)
      if not isinstance(states, dict) or any(
          not isinstance(k, int) or isinstance(k, bool) for k in states
      ):
        raise TypeError(
            f"LinearAttentionLayer.{name} must be a dict keyed by state index"
        )
      if any(
          item is not None and not isinstance(item, Tensor)
          for item in states.values()
      ):
        raise TypeError(
            f"LinearAttentionLayer.{name} values must be Tensors or None"
        )
    return (
        ("conv_states", value.conv_states),
        ("recurrent_states", value.recurrent_states),
    )
  else:
    raise TypeError(
        f"Unsupported Cache type: {kind.__module__}.{kind.__qualname__}"
    )

  for name, field in fields:
    if field is not None and not isinstance(field, Tensor):
      raise TypeError(f"{kind.__name__}.{name} must be a Tensor or None")
  return fields
