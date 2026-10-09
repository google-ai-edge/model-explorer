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

"""Bounded read-only previews of explicitly selected raw capture resources."""

import math


def preview_resource(store, identity, offset=0, limit=64):
  if not isinstance(identity, str) or not identity:
    raise ValueError('resource_id is required')
  if (
      type(offset) is not int
      or offset < 0
      or type(limit) is not int
      or not 1 <= limit <= 256
  ):
    raise ValueError('Expected offset >= 0 and limit between 1 and 256')
  return _preview(store.load_resource(identity), identity, offset, limit)


def preview_pair_tensor(
    store, identity, role, mode='original', offset=0, limit=64
):
  if (
      not isinstance(identity, str)
      or not identity
      or role not in ('ref', 'target')
      or mode not in ('original', 'comparison')
  ):
    raise ValueError(
        'Expected a saved pair, ref/target role and original/comparison mode'
    )
  if (
      type(offset) is not int
      or offset < 0
      or type(limit) is not int
      or not 1 <= limit <= 256
  ):
    raise ValueError('Expected offset >= 0 and limit between 1 and 256')
  return _preview(
      store.load_pair_tensor(identity, role, mode=mode),
      f'{identity}:{role}:{mode}',
      offset,
      limit,
  )


def _preview(values, identity, offset, limit):
  if offset > values.size:
    raise ValueError('Preview offset exceeds the tensor length')
  # Only the displayed window is widened to Python values. The source tensor
  # and comparison shape remain unchanged; non-finite values stay explicit.
  output = []
  for item in values.reshape(-1)[offset : offset + limit]:
    value = item.item()
    if isinstance(value, float) and not math.isfinite(value):
      value = str(value)
    elif isinstance(value, int) and abs(value) > 2**53 - 1:
      # JSON numbers are IEEE doubles in the browser. Preserve exact
      # int64/uint64 values instead of silently rounding the preview.
      value = str(value)
    output.append(value)
  return dict(
      resource_id=identity,
      shape=list(values.shape),
      dtype=str(values.dtype),
      offset=offset,
      total=int(values.size),
      values=output,
  )
