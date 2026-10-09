# LogicGraph (`model-explorer-hfgraph`)

See a Hugging Face model in Model Explorer **the way the source is written**:
module hierarchy, every `if` / `for` in `modeling_*.py` (including the regions
the tracer never took), loops folded to one iteration, and a source line on
every op.

`torch.export` alone gives you one path. LogicGraph gives you the whole program,
with the traced path filled in.

LogicGraph ships inside `ai-edge-model-explorer`: the built-in
`builtin_hfgraph` adapter opens `.hfgraph` and `.hfrun` files, and the
`model-explorer-hfgraph` command builds them.

## Requirements

-   Python ≥ 3.10 to build graphs (run the pipeline or open a `.hfrun` spec).
    Opening an existing `.hfgraph` file works on any Python version supported
    by `ai-edge-model-explorer`.
-   About 9 GB of RAM to trace a 2B-parameter model with real weights.
-   Network access to the Hugging Face Hub, and authentication for gated
    models. `--tiny` still downloads the model config.

## Install

```bash
pip install "ai-edge-model-explorer[hfgraph]"   # adds torch, transformers, pydantic
```

Viewing an existing `.hfgraph` file only needs the base package
(`pip install ai-edge-model-explorer`).

## Quick start

`<hf_id>` accepts either a Hugging Face Hub repo ID (`Qwen/Qwen3.5-2B`) or a
local checkpoint directory.

```bash
# Seconds: shrunken random-weight Qwen3.5 with the real layer pattern,
# prefill + one decode step.
model-explorer-hfgraph Qwen/Qwen3.5-2B out/tiny --tiny --decode

# Real weights (a few minutes on CPU).
model-explorer-hfgraph Qwen/Qwen3.5-2B out/qwen3_5_2b --decode

# Open the result in Model Explorer.
model-explorer-hfgraph-view out/qwen3_5_2b --open
```

Model Explorer also opens LogicGraph files directly, with no `--extensions`
flag:

```bash
cp out/qwen3_5_2b/model_explorer.json qwen3_5_2b.hfgraph
model-explorer qwen3_5_2b.hfgraph

# A .hfrun spec runs the pipeline when opened, then caches the result.
echo '{"model": "Qwen/Qwen3.5-2B", "out": "qwen3_5_tiny.out", "tiny": true, "decode": true}' > qwen3_5_tiny.hfrun
model-explorer qwen3_5_tiny.hfrun
```

## Reading the graph

The namespace tree reads like the source:

```text
model / layers.0 / linear_attn / if@574 use_precomputed_states and seq_len == 1 ▸ else / …
model / layers.0 / linear_attn / torch_chunk_gated_delta_rule / for@399 range(1, chunk_size) ×63
```

Every region has one of six statuses. Regions without ops get a colored
placeholder node whose attributes (`reason`, `needs`, `evidence`) explain the
status; the group's side panel ("Layer attributes") shows the condition, its
kind, status and evidence.

<!-- mdformat off(preserve GFM table layout) -->
| color | status | meaning |
|---|---|---|
| light green | covered | executed by at least one trace, or decided as taken inside an executed region and contains no tensor ops |
| grey | pruned | decided by config or export constants and never taken (`self.block_type == 'linear_attention'`) |
| yellow | guarded | value-dependent but forced by `is_tracing()` / `is_compiling()`: eager and export differ here |
| light grey | uncovered | reachable, but no trace took it; the `needs` attribute says what input would |
| purple | data_dependent | depends on tensor values (`nonzero`, `.item()` …); traced through a dense stand-in or not at all |
| red | conflict | evidence says the region was not taken in a trace that nevertheless has ops there; points at a classification-rule bug |
<!-- mdformat on -->

The graph selector offers `merged` plus one graph per trace (`seq8`, `decode1`,
…). A group with a single child gets an extra info node, because Model Explorer
collapses single-child layers by default. `coverage.md` in the output directory
lists every region.

## Command reference

### `model-explorer-hfgraph`

```shell
model-explorer-hfgraph <hf_id> <out_dir> [options]
```

Runs the whole pipeline (skeleton → prune → plan → trace → merge → emit) and
writes the output directory described under [Output](#output).

<!-- mdformat off(preserve GFM table layout) -->
| option | effect |
|---|---|
| `--tiny` | shrink the config (4 layers, small hidden size, real `layer_types` pattern, random weights); seconds instead of minutes, structure only |
| `--decode` | add the trace `decode1`: an eager prefill of 8 tokens fills a `DynamicCache`, then one cached step is exported; covers decode-only branches |
| `--full-coverage` | trace several static prefill lengths derived from the shape conditions (1, 2, 8, 64, 65, 129 …), at most 8; implies `--decode` |
| `--primary <trace id>` | the trace whose shapes the merged graph shows (default: the first planned trace) |
| `--draft` | also run `torch.export.draft_export` and keep its report next to the trace |
| `--skip-trace` | reuse `traces/*.json` that already exist; rerun only merge and emit |
<!-- mdformat on -->

Progress lines are printed per stage, for example:

```text
[A/B] templates: 9 forwards, 32 functions; branches: 104 ({...}); unknown ratio 0.0%; module instances: 153; region paths: 833 (pruned 181)
[C] trace plan: ['seq8', 'decode1']
[D] seq8 done in 40.9s
[E] seq8: ops 18359 → 3851 after folding 18 loops; paths: 535
[E] coverage: {'covered': 416, 'pruned': 181, 'uncovered': 231, 'guarded': 5}
[F] out/qwen3_5_2b/model_explorer.json: graphs=3 merged nodes=6038 groups=834 placeholders=491  check: 0 problems
```

`check: N problems` is the emitter's self-check: every static region path must
be a group in the merged graph, and every non-covered one must have a
placeholder.

### `model-explorer-hfgraph-view`

```shell
model-explorer-hfgraph-view <out_dir | model_explorer.json> [--port 8085] [--open]
```

Starts Model Explorer with the graph registered as an in-memory source
(`graphs://…`), so no file renaming is needed. Prints the URL; `--open` also
opens the browser.

### Opening files in Model Explorer

```shell
model-explorer <file.hfgraph | file.hfrun>
```

<!-- mdformat off(preserve GFM table layout) -->
| file | content |
|---|---|
| `.hfgraph` | a `model_explorer.json` written by `model-explorer-hfgraph` (rename or copy it) |
| `.hfrun` | `{"model": "...", "out": "...", "tiny": bool, "decode": bool, "full_coverage": bool, "draft": bool, "primary": "...", "rerun": bool}`; runs `model-explorer-hfgraph` first and caches the result in `out` (see below) |
<!-- mdformat on -->

A `.hfrun` spec resolves `out` relative to the spec file (default:
`<spec name>.out` next to it). `out` must be a directory strictly below the
spec's directory: `.`, `..`, absolute paths elsewhere and symlinks that lead
out are refused, because every run replaces `out` as a whole. The pipeline
writes into a fresh hidden directory next to `out` and swaps it in only when it
succeeds, so files or symlinks already in `out` are never written through.
`model` must not start with `-`. After a run the spec's output-affecting fields
(`model`, `primary` and the boolean flags) and the Model Explorer version are
recorded in `out/hfrun_spec.json`; the cached result is reused while they match
(symlinked cache files are ignored), and `"rerun": true` forces a new run. A
failed run keeps the previous output but drops its record. Output directories
written by running `model-explorer-hfgraph` directly have no record, so opening
a spec that points at one runs the pipeline once.

The adapter also appears in Model Explorer's "Select Models" dialog.

### Trace subprocess

Stage D runs once per planned trace as a subprocess, so that `TORCH_LOGS=guards`
applies to a fresh `torch` import:

```shell
python -m model_explorer.hfgraph.trace <hf_id> <skeleton.json> <out.json> --seq 8 [--tiny] [--decode N] [--draft]
```

### Programmatic use

```python
from model_explorer.hfgraph.models import build_text_model, config_hash
from model_explorer.hfgraph.prune import expand_instances
from model_explorer.hfgraph.skeleton import build_skeleton

model, tcfg = build_text_model("Qwen/Qwen3.5-2B", tiny=True, device="meta")
sk, registry = build_skeleton(model, "Qwen/Qwen3.5-2B", tcfg, config_hash(tcfg))
sk = expand_instances(sk, registry, model)
print(len(sk.branches), sum(1 for i in sk.instances if i.static == "pruned"))
```

## Output

```text
out/<name>/
  skeleton.json           Skeleton: modules, branches (all regions), loops    stages A + B
  traces/<id>.json        Trace: ops and edges of one strict torch.export     stages D + E
  traces/<id>.guards.log  raw Dynamo guard log (evidence for branch decisions)
  traces/<id>.draft.txt   draft_export report / export failure text, if any
  coverage.json           Coverage: status of every region path              stage E
  coverage.md             the same as a table
  model_explorer.json     Model Explorer GraphCollection                      stage F
```

All stages exchange the pydantic models in `hfgraph/schema.py`;
`meta.schema_version` is an integer.

```text
Skeleton
  meta          schema_version, hf_id, config_hash, torch, transformers, source_file_hashes{rel path: sha1}
  functions[]   FunctionTemplate      one per class forward / helper function
  branches[]    Branch                one per if / for / while / ternary in those functions
  instances[]   Instance              every statically known path

Trace
  trace   TraceInfo {id, inputs, plan {seq_len, cache}, config_overrides, strict, primary, specialized[], stats}
  ops[]   Op {id '<path>|<file>:<line>|<ordinal>', path, fx_name, aten, label, file, line, loop_iter, inputs[], outputs[], attrs}
  edges[] Edge {src, src_out, dst, dst_in}
  paths[] PathRecord {path, ops, iterations}

Coverage
  paths[]   PathCoverage {path, branch, status, reason, evidence, covered_by[trace ids], needs}
  summary   {status: count}
```

A **path** is one string that says where something is in the source structure.
It is both the identity of an instance and, after rendering, the Model Explorer
namespace:

```text
model/layers.0/if@888#then/linear_attn/torch_chunk_gated_delta_rule/if@394#else/for@399#body
```

<!-- mdformat off(preserve GFM table layout) -->
| segment | form | example |
|---|---|---|
| module instance | attribute name; `ModuleList` children merged into the parent segment | `linear_attn`, `layers.0` |
| function call | template id | `torch_chunk_gated_delta_rule`, `sdpa_attention_forward` |
| region | `<kind>@<line>#<then\|else\|body>` | `if@888#then`, `for@399#body` |
<!-- mdformat on -->

Templates hold every source construct once; instances are the paths that exist
statically, so `layers.0/if@888#else` can be `pruned` while
`layers.3/if@888#else` is `taken`. In `model_explorer.json`, node ids are op
ids, `ph:<graph>:<path>` for placeholders, and `info:<graph>:<namespace>` for
single-child info nodes; every node carries its raw `path` as an attribute.

## Models & limits

-   Supported models: any `qwen3_5` / `qwen4_exp` checkpoint through
    `models.FAMILIES`; other causal LMs fall back to `AutoModelForCausalLM`.
-   Only the text decoder (`model.language_model`) of multimodal checkpoints is
    exported.
-   Traces are static-shape; dynamic sequence dims fail on GatedDeltaNet models
    (chunk-loop guards), so the plan traces several static lengths instead.
-   The `--decode` graph lifts cache tensors as constants: fine for coverage,
    not reusable for inference.
-   AST line numbers and export stack traces must come from the same installed
    `transformers`; `skeleton.json` records the source file hashes.
-   Placeholders are defined per function and expanded per instance, so a
    24-layer model shows the same uncovered region 24 times.

## Troubleshooting

-   **`Running a .hfrun spec requires ...`** when opening a `.hfrun` file:
    install the extra with `pip install "ai-edge-model-explorer[hfgraph]"` and
    restart Model Explorer. The extra needs Python ≥ 3.10; on Python 3.9 the
    message also says so, and `model-explorer-hfgraph` prints the same hint.
    Opening `.hfgraph` files never needs it.
-   **`trace <id> failed`**: the last lines of `traces/<id>.guards.log` are
    printed; the full log is in the output directory. Rerun with `--tiny` to
    check the model structure without real weights.
-   **Hub errors on `--tiny`**: the config is still downloaded; log in to the
    Hub for gated models, or pass a local checkpoint directory.

## Internals

`cli.py` parses the command line and checks for the `hfgraph` extra using only
the standard library; `pipeline.py` runs the stages below in order.

1.  **skeleton** (`skeleton.py`): AST of every module `forward` and the helper
    functions it calls (decorators and the config-resolved attention / mask /
    experts interfaces included). Each condition is classified
    `config | shape | value | guarded | python | unknown` (see
    [Branch classification](#branch-classification)); loops get `loop`.
2.  **prune** (`prune.py`): conditions that depend only on config or export
    constants are evaluated statically, per module instance; if instances
    disagree, the ops decide.
3.  **plan** (`plan.py`): which static input lengths to trace, plus the decode
    step.
4.  **trace** (`trace.py`): strict `torch.export` in a subprocess with
    `TORCH_LOGS=guards`; falls back to non-strict, then to a skeleton-only
    graph. Data-dependent code (sparse-attention indexer, MoE routing) is traced
    through dense stand-ins.
5.  **merge** (`merge.py`): ops are joined to branch regions by source line,
    unrolled loops are folded, and coverage is computed from ops, the guard log
    and shape evaluation. When evidence contradicts the rule table, the region
    is marked `conflict` instead of being silently trusted.
6.  **emit** (`emit.py`): Model Explorer JSON with placeholders, group
    attributes and a self-check.

The JSON schemas for `skeleton.json`, `traces/<id>.json` and `coverage.json`
live under `hfgraph/schema/*.schema.json`; regenerate them with
`python -m model_explorer.hfgraph.schema` after changing `hfgraph/schema.py`
(a unit test checks that they match).

### Branch classification

`skeleton.classify` gives every `if` / ternary condition a `cond_kind`. The
first matching rule wins:

<!-- mdformat off(preserve GFM table layout) -->
| # | condition | `cond_kind` |
|---|---|---|
| 1 | calls an export guard (`is_tracing`, `is_torchdynamo_compiling`, …) | `guarded` if it also calls a value method, else `config` |
| 2 | calls a value method (`.item()`, `.any()`, `.sum()`, `.nonzero()`, …) | `value` |
| 3 | names a module constant (`is_*_available`, `_is_torch*`, …) or `_attn_implementation` | `config` |
| 4 | calls `hasattr` / `isinstance` / `callable`, or the builtins `all` / `any` without method calls | `python` |
| 5 | is itself a call (or `not <call>`) | `shape` |
| 6 | reads `self.config…` or `self.training` and no shape name | `config` |
| 7 | names a shape attribute (`shape`, `size`, `ndim`, …) or shape variable (`seq_len`, `num_chunks`, …) | `shape` |
| 8 | names a plain call argument (`use_cache`, `return_dict`, `output_attentions`, …) | `python` |
| 9 | uses `is` / `is not` | `shape` on a tensor argument (`attention_mask`, `past_key_values`, …), else `python` |
| 10 | reads only `self.…` attributes | `config` |
| 11 | anything else | `unknown` |
<!-- mdformat on -->

`config`, `python` and `guarded` conditions are evaluated in stage B;
`shape` conditions are evaluated per trace from its input shapes in stage E;
`value` and `unknown` regions are only decided by the ops a trace executed.
