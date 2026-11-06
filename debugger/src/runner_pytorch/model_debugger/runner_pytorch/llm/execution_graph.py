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

"""The execution graph of a turn's passes, as pytorch-collector draws it.

A turn whose passes ran inside collect() (anchor_observer) can export, of
the first pass of each signature it ran (prefill, decode), the graph
pytorch-collector's own to_model_explorer() makes of it: one node per
operator call under its module path, its tensors' producers as edges, the
parameters as attributes of the node that reads them, what the pass read
from outside as `inputs` nodes. Operators whose results reach nothing
(unused_ops(): an index a mask function ignores) are left out
(`hide_unused`), as the collector's own export of Gemma 4 does. Nothing is
folded or renamed here: what the server draws of it is the server's.

Beside the graphs, the class of each module the passes' operators ran in,
and of each module above it (`modules`, by module path), as the model holds
them: which module is a Linear, say. Gemma 4 E2B's prefill and decode take
about 0.13 s to export and 0.2 MB gzipped.

The document, `execution_graph.json.gz` in the turn's capture:

  {"format": 1, "hide_unused": true,
   "graphs": [{"id": "prefill", "nodes": [...]}, {"id": "decode", ...}],
   "steps": {"prefill": <its step label>, "decode": <its step label>},
   "modules": {"<module path>": "<class name>", ...},
   "seconds": <how long the export took>}
"""

from collections.abc import Iterable, Sequence
import time
from typing import Any

import pytorch_collector
import torch

FILE = 'execution_graph.json.gz'
FORMAT = 1


def document(
    capture: pytorch_collector.Capture,
    passes: Sequence[tuple[str, int]],
    model: torch.nn.Module,
    signatures: Iterable[str],
) -> dict[str, Any] | None:
  """The graph of the first pass of each signature asked for.

  Args:
      capture: The turn's capture, exited.
      passes: Each forward pass's (signature, step), by its forward id.
      model: The model the passes ran.
      signatures: The signatures whose graphs are wanted.

  Returns:
      The document, or None if the turn ran none of them.
  """
  wanted = set(signatures)
  firsts: dict[str, tuple[int, int]] = {}
  for forward_id, (signature, step) in enumerate(passes):
    if signature in wanted and signature not in firsts:
      firsts[signature] = (forward_id, step)
  if not firsts:
    return None
  started = time.perf_counter()
  names = {forward_id: name for name, (forward_id, _) in firsts.items()}
  collection = pytorch_collector.to_model_explorer(
      capture, sorted(names), names=names, hide_unused=True
  )
  paths = {''}
  for op in capture.ops:
    if op.forward_id in names and op.module:
      parts = op.module.split('.')
      paths.update('.'.join(parts[:i]) for i in range(1, len(parts) + 1))
  modules = dict(model.named_modules())
  return {
      'format': FORMAT,
      'hide_unused': True,
      'graphs': collection['graphs'],
      'steps': {signature: step for signature, (_, step) in firsts.items()},
      'modules': {
          path: type(modules[path]).__name__
          for path in sorted(paths)
          if path in modules
      },
      'seconds': round(time.perf_counter() - started, 3),
  }
