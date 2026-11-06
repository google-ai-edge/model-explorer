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

"""Scoped dispatcher observation; capture owns records and failure state."""

import torch.utils._python_dispatch as dispatch
from torch.utils._python_dispatch import TorchDispatchMode

_FLAG_FIELDS = (
    ("old_dispatch_mode_flags", "_is_in_torch_dispatch_mode"),
    (
        "old_non_infra_dispatch_mode_flags",
        "_is_in_non_infra_torch_dispatch_mode",
    ),
    (
        "old_without_ignore_compile_internals_dispatch_mode_flags",
        "_is_in_any_mode_without_ignore_compile_internals",
    ),
)


# pylint: disable=protected-access
class CaptureMode(TorchDispatchMode):
  """TorchDispatchMode subclass for capturing PyTorch operations."""

  def __init__(self, capture):
    super().__init__()
    self._capture = capture

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    return self._capture._operation(func, args, kwargs or {})

  def close(self) -> bool:
    """Removes this activation only; reports whether scopes were nested normally.

    Torch 2.9's default exit blindly pops the current top mode. A foreign
    mode left active by the caller must survive capture cleanup instead.
    Repair its saved flags as well so its eventual normal exit is safe.

    Returns:
      True if the mode was exited normally at the top of the stack,
      False otherwise.
    """
    stack = dispatch._get_current_dispatch_mode_stack()
    position = next(
        (i for i in range(len(stack) - 1, -1, -1) if stack[i] is self), None
    )
    if position is None:
      return False
    if position == len(stack) - 1:
      super().__exit__(None, None, None)
      return True

    above = stack[position + 1 :]
    # A mode object can be entered repeatedly, including below this capture.
    # Index each activation from the top of that object's saved flag deques.
    seen = {}
    activations = []
    for mode in reversed(above):
      occurrence = seen.get(id(mode), 0) + 1
      seen[id(mode)] = occurrence
      activations.append((mode, -occurrence))

    current_flags = [getattr(self, field)[-1] for field, _ in _FLAG_FIELDS]
    updates = []
    for mode, index in reversed(activations):
      contributions = (
          True,
          not mode.is_infra_mode(),
          not mode.ignore_compile_internals(),
      )
      for i, (field, _) in enumerate(_FLAG_FIELDS):
        saved = getattr(mode, field)
        _ = saved[index]  # Validate before changing any stack or flags.
        updates.append((saved, index, current_flags[i]))
        current_flags[i] = current_flags[i] or contributions[i]

    removed = []
    try:
      for _ in above:
        removed.append(dispatch._pop_mode())
      dispatch._pop_mode()  # This capture; do not invoke another mode's exit.
    finally:
      for mode in reversed(removed):
        dispatch._push_mode(mode)
    for field, _ in _FLAG_FIELDS:
      getattr(self, field).pop()
    for saved, index, value in updates:
      saved[index] = value
    for (_, global_name), value in zip(_FLAG_FIELDS, current_flags):
      setattr(dispatch, global_name, value)
    return False
