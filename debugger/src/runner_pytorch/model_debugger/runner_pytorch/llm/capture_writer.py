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

"""Writes a turn's capture in LiteRT-LM's layout, which the server reads.

One safetensors file per tensor per forward pass, named
`<signature>_<key>_step_<step>.safetensors`, holding the one tensor `<key>`
in its own dtype (bf16 stays bf16), with `__metadata__` naming the signature
(`prefill` or `decode`) and the step. Beside them `generated_tokens.jsonl`,
one line per released token: {"token_ids": [[id]], "texts": [text]}.

A KV cache file, `(pre|post)_kv_cache_(k|v)_<layer>`, holds the cache as
Hugging Face keeps it, [batch, kv_head, sequence, head_dim] for both k and v,
which its metadata `layout` says: `k:bhsd,v:bhsd`. The sequence axis spans
the positions the layer holds: all those processed so far, or for a sliding
window layer only the last ones, which its metadata `kv_positions`,
`<first>-<last>`, names. A layer that shares another's KV (Gemma 4's last
layers) has no cache of its own and no file.

A turn that observes the decoder layers' anchors (anchor_observer) has one
file per anchor per layer per forward pass, `post_anchor_<layer>_<anchor>`,
the tensor as the model holds it there, [batch, rows, hidden] (rows: the
positions the pass fed), its `__metadata__` naming besides the `layer`, the
`anchor`, the `module` path it was taken at, its `edge` (`in` for what the
module reads, `out` for what it returns) and its `path` in the collector's
record. Each module the Load asks for beside them (`observeModules`) has
one file per forward pass too, `post_module_<module path>_<edge>`, what it
reads (`in`, its argument 0) or returns (`out`) as the model holds it,
its `__metadata__` naming the `module`, `edge` and `path`. And
`observation.json`, what the turn observed:
{"observe": "anchors", "capture": <how collect() observed: "full" or
"modules">, "modules": <the modules asked for, "<path>:<edge>">,
"memory_limit_bytes": <collect()'s budget>, "captured": <whether the
tensors are in the capture>, "reason": <why not, or "">, "snapshot_bytes":
<what collect() snapshotted of the turn, 0 if it failed>,
"execution_graph": <the signatures whose graphs the turn wrote>,
"execution_graph_reason": <why it wrote none it was to, or "">}. A turn
that observes nothing has no such file.

A turn that observed fully may write `execution_graph.json.gz` too: the
execution graph of the first pass of each signature it ran
(execution_graph), gzipped JSON.
"""

import gzip
import json
import pathlib
from typing import Any

from model_debugger.runner_pytorch.llm import anchor_observer
from model_debugger.runner_pytorch.llm import execution_graph
from safetensors import torch as safetensors_torch
import torch

TOKENS_FILE = 'generated_tokens.jsonl'
OBSERVATION_FILE = 'observation.json'
KV_LAYOUT = 'k:bhsd,v:bhsd'


class CaptureWriter:
  """The capture of one turn, written into its directory as it runs."""

  def __init__(self, directory: pathlib.Path) -> None:
    """Starts the capture in `directory`, which is created empty."""
    directory.mkdir(parents=True, exist_ok=False)
    self.directory = directory
    (directory / TOKENS_FILE).touch()

  def tensor(
      self,
      signature: str,
      step: int,
      key: str,
      tensor: torch.Tensor,
      metadata: dict[str, str] | None = None,
  ) -> None:
    """Writes one tensor of a forward pass.

    Args:
        signature: 'prefill' or 'decode'.
        step: The forward pass's step label.
        key: The tensor's name, as `pre_input_pos`.
        tensor: The tensor; copied to the CPU, its dtype kept.
        metadata: More `__metadata__` entries.
    """
    name = f'{signature}_{key}_step_{step}.safetensors'
    safetensors_torch.save_file(
        {key: tensor.detach().to('cpu').contiguous()},
        str(self.directory / name),
        metadata={'signature': signature, 'step': str(step)} | (metadata or {}),
    )

  def kv_cache(
      self, signature: str, step: int, moment: str, cache: object
  ) -> None:
    """Writes every layer's KV cache as it is now.

    Args:
        signature: 'prefill' or 'decode'.
        step: The forward pass's step label.
        moment: 'pre' (before the pass) or 'post' (after it).
        cache: The transformers DynamicCache.
    """
    for layer, entry in enumerate(cache.layers):
      keys = getattr(entry, 'keys', None)
      values = getattr(entry, 'values', None)
      if keys is None or values is None or keys.shape[-2] == 0:
        continue
      metadata = {'layout': KV_LAYOUT}
      if getattr(entry, 'is_sliding', False):
        last = int(entry.cumulative_length) - 1
        metadata['kv_positions'] = f'{last - keys.shape[-2] + 1}-{last}'
      for kind, tensor in (('k', keys), ('v', values)):
        self.tensor(
            signature,
            step,
            f'{moment}_kv_cache_{kind}_{layer}',
            tensor,
            metadata,
        )

  def anchor(self, found: anchor_observer.Tensor) -> None:
    """Writes one anchor's or one module's tensor of one forward pass."""
    metadata = {
        'module': found.module,
        'edge': found.edge,
        'path': found.path,
    }
    if found.anchor:
      metadata |= {'layer': str(found.layer), 'anchor': found.anchor}
    self.tensor(found.signature, found.step, found.key, found.tensor, metadata)

  def observation(self, facts: dict[str, Any]) -> None:
    """Writes observation.json: what the turn observed, or why not."""
    (self.directory / OBSERVATION_FILE).write_text(
        json.dumps(facts, sort_keys=True) + '\n', encoding='utf-8'
    )

  def graph(self, document: dict[str, Any]) -> None:
    """Writes execution_graph.json.gz: the turn's execution graph."""
    body = json.dumps(document, separators=(',', ':')).encode('utf-8')
    (self.directory / execution_graph.FILE).write_bytes(
        gzip.compress(body, 6, mtime=0)
    )

  def released(self, token_id: int, text: str) -> None:
    """Appends a released token to generated_tokens.jsonl."""
    line = json.dumps({'token_ids': [[token_id]], 'texts': [text]})
    with open(self.directory / TOKENS_FILE, 'a', encoding='utf-8') as file:
      file.write(line + '\n')
