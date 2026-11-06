# pytorch-collector

> Moved here from the user's repository `~/Projects/pytorch-collector` on
> 2026-10-20: commit `fe82ff9` with its uncommitted module-only change
> (`collect(model, modules=[...])`), and from now on maintained here, the one
> source of the library. The package's files are byte for byte that working
> tree's `src/pytorch_collector/`, but for `equivalence/` (Z3 and tflite checks
> against LiteRT), which stayed there with its tests, `examples/`, their tests
> and the acceptance records `docs/verification/`. This README is that
> repository's, without its equivalence and development sections;
> `docs/DESIGN.md` and `docs/PLAN.md` are its own, with a note.

One `with` block captures, for every forward of a model, the inputs and outputs
of every module and every dispatched operator, plus the data flow between them.
The same execution can be exported as a Model Explorer graph.

```python
from pytorch_collector import collect, to_model_explorer

with collect(model) as capture:  # model(...) calls, or one generate()
    model(input_ids)

capture.records    # (forward_id, module, op, call_id, edge, path, tensor)
capture.ops        # every operator call, with or without a snapshot
capture.edges      # exactly one source per tensor input of every operator
capture.outputs    # the source of every tensor a forward returned
to_model_explorer(capture, names={0: "prefill", 1: "decode_1"})  # json.dump
```

Version `0.2.0`. Pinned to PyTorch 2.9.0, CPU eager (a module-only capture runs
on any device), single thread; Transformers 5.17.0 for the cache types. Details
and evidence: [design](docs/DESIGN.md), [plan](docs/PLAN.md), and the acceptance
records for stages 1–5, which stayed in pytorch-collector's
`docs/verification/`.

## How capture works

1. **Enter**: claim the model, enter `no_grad`, install hooks on every module
   from `named_modules()`, enable a `TorchDispatchMode`.
2. **Observe**: at each module entry/exit and around each operator call, walk
   the values (tensors, tuples, lists, Mappings, `DynamicCache`), drop
   parameters and proven parameter views, check the budget, `detach().clone()`.
3. **Attribute**: each operator is assigned to the innermost module call; each
   tensor input is traced to its producer by object identity. Writes (`add_`,
   `out=`, through a view) move the producer to the writing call.
4. **Exit**: verify pairing and hook integrity, remove only our own hooks and
   mode, restore state. Any failure anywhere makes the whole result unreadable;
   the model and cache are not rolled back.
5. **Read**: everything is readable only after a successful exit.

Rules that matter:

- Parameters and their views/transposes/casts are never snapshotted; everything
  else that reaches an observation point is (activations, buffers, cache
  tensors, unrecognized `W * scale`).
- One snapshot budget across all forwards, default 1 GiB, set with
  `memory_limit_bytes`. Exceeding it fails the capture.
- Nothing outside a root forward is recorded.

## Capturing only some modules

```python
parts = ("", ".pre_feedforward_layernorm", ".per_layer_input_gate")
anchors = [
    f"model.language_model.layers.{i}{part}"
    for i in range(35)
    for part in parts
]
with collect(model, modules=anchors) as capture:  # paths or module objects
    model.generate(**inputs, max_new_tokens=24)
capture.records    # only these modules' inputs/outputs, bit-identical to full
capture.modules    # the selected paths; None for a full capture
```

Only the selected modules (and the root, to number forwards) get hooks and no
dispatch mode is enabled, so there are no operator records: `ops`, `edges` and
`outputs` raise, and `to_model_explorer` refuses the capture. The value walk,
`DynamicCache` expansion, parameter exclusion and budget are the same; a
parameter copied or cast inside the forward is not recognized without operators
and is saved. Snapshots stay on the device they were observed on (CPU, MPS; CUDA
untested). Unknown paths raise `ValueError`. Gemma 4 E2B, 13-token prompt + 24
decodes, 105 anchor modules on CPU: 0.46 GB and 3.1 s against 4.81 GB and 13.3 s
for a full capture
([measurement](docs/PLAN.md#step-6-capture-only-selected-modules),
[design](docs/DESIGN.md#step-7-capture-only-selected-modules)).

## The graph

`to_model_explorer` gives one graph per forward. Node rules:

-   operator call (its `call_id`): under its module path
-   something read but not computed here (root input, KV passed back in) (the
    path it was passed under, e.g. `kwargs.input_ids`): `inputs`; a KV entry
    also names `source_graph` / `source_node`
-   something returned (its output path, e.g. `output.logits`): `outputs`
-   parameter, buffer, constant (—): attributes of the consuming node

`unused_ops(capture)` walks backwards from the outputs and from every writing
operator; `hide_unused=True` leaves out what it does not reach (for Gemma: the
`head_idx` a mask function ignores, and cache fields passed through untouched).

Edge kinds: `op` (an operator output, possibly from an earlier forward),
`input`, `param` / `buffer`, `external` (made between two root calls),
`constant`.

## Exporting a model

`examples/export_graph.py` stayed in `~/Projects/pytorch-collector` with the
other examples; these commands run there, in its own environment (uv).

```sh
uv sync --group integration

# Gemma 4 E2B-it: prefill + 3 decodes, ~20 s
uv run --group integration python examples/export_graph.py \
    --model google/gemma-4-E2B-it \
    --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
    --model-class Gemma4ForConditionalGeneration \
    --prompt "The capital of France is" --steps 4 \
    --hide-unused --run-dir output/gemma4-e2b

# Qwen3.5-2B: prefill + 1 decode, ~16 s, ~1.9 GB of snapshots
uv run --group integration python examples/export_graph.py \
    --model Qwen/Qwen3.5-2B --model-class Qwen3_5ForConditionalGeneration \
    --input-ids 785,6722,315,9625,374 --steps 2 --hide-unused \
    --memory-limit-bytes $((8 * 1024**3)) --run-dir output/qwen3.5-2b

model-explorer output/gemma4-e2b/test_0/graph.json --port 8090
```

Each run creates `output/<model>/test_N` with `graph.json`, one
`<graph>.safetensors` of raw operator values keyed `<call_id>:<in|out>:<path>`,
`values.json` and `summary.json`. Weights come from the local HF cache only.
`--model-class` is needed where `AutoModelForCausalLM` does not fit;
`--input-ids` replaces the tokenizer. Qwen3.5's gated-DeltaNet cache
(`LinearAttentionLayer`) is recorded as `conv_states.<i>` /
`recurrent_states.<i>`.

## Development

The library is the runner's Bazel package `//pytorch_collector`; it needs torch
alone. Its tests, those of the capture library as that repository has them, are
`//pytorch_collector/tests` (one `py_test` per file, pytest), on the runner's
pinned packages, transformers among them, so none is skipped but the one that
needs an MPS device, without one:

```sh
cd src/runner_pytorch
bazel test //pytorch_collector/...    # 212 tests in 12 files
```

ruff does not check it: it keeps its own style (double quotes, long lines), not
the runner's (`ci/check_runner_pytorch.sh` checks `model_debugger/` only).
