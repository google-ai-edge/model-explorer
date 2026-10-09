# AI Edge Debugger PyTorch Capture (`ai_edge_debugger_pytorch`)

Standalone PyTorch activation capture hooks and manifest generator for the AI
Edge Debugger.

## Overview

Exports `CaptureRun` for non-invasive runtime tensor capture on PyTorch causal
language models.

Key invariants:

-   **Layer-0 Sublayer Scope**: Fixed to layer 0 sublayers and outer model
    boundaries to bound memory and I/O overhead.
-   **Dual-Mode KV Snapshots**: Captures KV cache snapshots across
    `prefill_pre`, `prefill_post`, and `terminal` moments, supporting both
    `transformers.DynamicCache` and tuple of tuples.
-   **Shard Export**: Safetensors tensor payloads, `manifest.jsonl`,
    `boundaries/manifest.jsonl`, `kv/manifest.jsonl`, `forward_index.json`,
    `tokens.jsonl`, and `generation.json`.

## Installation

Install into a Python ≥ 3.10 environment that provides `torch` ≥ 2.4 and
`safetensors` ≥ 0.4:

```sh
python3 -m pip install -e src/capture/pytorch
```

## Usage

`CaptureRun` attaches forward hooks to a loaded Hugging Face causal language
model for one generation turn and writes Safetensors shards and manifests under
the requested output directory:

```python
from pathlib import Path
from ai_edge_debugger_pytorch import CaptureRun

with CaptureRun(model, out_dir=Path("/tmp/capture_out")) as capture:
  with capture.forward_context(phase="prefill", step=0, pos_offset=0, turn=1):
    outputs = model(**prefill_inputs, use_cache=True)
  capture.observe_tokens(
      tokens=[next_token_id],
      forward_id=capture.forwards[-1]["forward_id"],
      texts=[next_token_text],
  )
  capture.finish_generation("max_output_tokens")
```

See [PyTorch eager worker](../../../docs/pytorch-worker.md) for how the resident
Python Runner orchestrates `CaptureRun` across multi-turn conversations.

## Testing

From the repository root (using an interpreter with `torch` and `safetensors`
installed):

```sh
PYTHONPATH=src/capture/pytorch/package python3 -m unittest discover \
  -s src/capture/pytorch/tests -v
```

