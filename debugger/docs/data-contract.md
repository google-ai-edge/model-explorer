# Saved-session data contract

The server reads one configured directory. The UI uses HTTP only. Restart the
server after changing a session's JSON index; tensor files are read for each
comparison request. A session directory is an immutable saved-run snapshot, not
a capture workspace.

## Files

-   `semantic.json`: existing canonical `semantic_graph` definitions, `layers`
    instances, and optional `semantic_ops`. Nodes retain `incomingEdges`,
    `outputsMetadata`, and `attrs`. Each layer selects a definition through
    `def`, supplies operation parameters in `attrs`, and may bind inputs through
    `from`.
-   `execution.json`: `{ "executions": [...] }`; each execution has an `id`,
    runtime metadata, and unchanged `graphs` with GraphNodes and output ports.
    No inferred semantic-to-op mapping is added.
-   `session.json`: model display name, `runs` (reference `ref`, target
    `target`), `turns`, `phases`, and `batches` with unique `batch`, `turn`,
    `phase`, `step`, `index`. The UI derives layers from semantic JSON. See the
    checked-in example for full metadata. `notice` describes dataset provenance.
-   `tensor_index.json`: `{ "tensors": [...] }`. Empty is valid. Each record
    identifies one concrete execution output and one recorded semantic anchor:

```json
{
  "id": "ref-layer0-input-b0",
  "run": "ref",
  "graph": "prefill_1",
  "node": "Op12",
  "output": "0",
  "layer": 0,
  "anchor": "input",
  "batch": 0,
  "turn": 1,
  "phase": "prefill",
  "step": 0,
  "sample": "sample-0",
  "format": "safetensors",
  "path": "tensors/ref-prefill-step0.safetensors",
  "key": "post_layer0_input",
  "shape": [1, 1, 1536],
  "dtype": "float32"
}
```

This record illustrates the format, not a verified mapping for the bundled demo.
The dataset producer must supply a correct semantic anchor and concrete
graph/node/output identity. The backend validates that execution identity
exists; it cannot infer that a tensor is semantically equivalent from its
numeric shape. Paths are relative to the session directory. Every record
requires `format: "safetensors"` and the exact stored tensor `key`; several
records may share one shard path. Safetensors is the only raw storage format.
Missing formats, `.npy`, pickle/object arrays, packed quantized buffers and
implicit layout conversions are rejected.

The shared reader preserves the stored dtype and shape, including BF16 through
`ml_dtypes`; dtype names are canonical names such as `float32`, `int8`, and
`bfloat16`. A key is a storage identity, distinct from the graph output port or
semantic anchor. PyTorch shard slot names must be resolved through its capture
manifest. Missing keys, corrupt shards, metadata mismatches, and paths escaping
the session directory produce explicit errors, never a fallback to another
tensor.

LiteRT-LM workers write a version-2 `export/capture_index.json` with
`tensor_root: "run"` and paths relative to that run directory. This is an index
over original runtime shards and does not write converted tensors.
Standalone/iOS exports use `tensor_root: "export"` with paths relative to the
export directory. Publication validates the declared boundary and copies each
referenced shard once into the immutable saved capture, preserving its bytes.
Appended turns also retain each earlier shard once. The final
`tensor_index.json` always uses session-relative paths and does not depend on
the job directory remaining present.

## Pairing and computation

Pair by layer + anchor + batch + turn + phase + step, with matching explicit
sample IDs. Require exactly one reference and target record; duplicates are
ambiguous and never silently picked. Validate graph/node/output identity, dtype
and shape metadata against the actual file. Require equal tensor shapes. Do not
reshape, transpose, slice, filter NaN/Inf, or add epsilon.

Computations accumulate in float64 over corresponding elements, including BF16
inputs. This calculation does not rewrite or change the stored tensor dtype. Let
`d = reference - target`:

<!-- mdformat off(preserve GFM table layout) -->
| Metric | Definition | Worst aggregation |
| --- | --- | --- |
| CosSim | dot(ref,target) / (norm(ref) × norm(target)) | minimum |
| Max abs error | max(abs(d)) | maximum |
| Mean abs error | mean(abs(d)) | maximum |
| RMSE | sqrt(mean(d²)) | maximum |
| Relative L2 | norm(d) / norm(ref) | maximum |
<!-- mdformat on -->

CosSim is undefined for either zero norm. Relative L2 is undefined for a zero
reference norm. Other metrics can still be valid. Empty, non-finite,
unequal-shape, missing or unreadable tensors produce explicit per-anchor
statuses. Invalid/undefined values are excluded independently per metric;
coverage always includes the full expected anchor count. Numeric overflow is an
error, not a finite substituted value. KL is excluded by agreement.

## API

-   `GET /api/session`: display metadata and available batch coordinates.
-   `GET /api/semantic`: canonical semantic JSON, unchanged.
-   `GET /api/execution`: execution JSON, unchanged.
-   `GET /api/comparisons?batch=N`: compute all layer/anchor pairs for the
    selected batch and per-layer worst/coverage summaries.
-   `GET /api/overview`: compute session batch worst values and coverage. This
    currently scans the saved session; large runs will need incremental
    summaries.

The UI aborts superseded batch requests and discards late responses. Metrics
remain backend results. Timeline bars use the selected metric's relative range;
missing values use a short marker. Worst jumps to the worst comparable batch.
First divergence stays disabled until an explicit threshold policy is provided;
no threshold is invented.

## Bundled example scope

Graph Diff renders graph definitions, layer parameters, external inputs, anchors
and dataflow dynamically, with node inspection dialogs, execution graph
selection and manual tensor mappings.

The bundled [`examples/gemma4-e2b/`](../examples/gemma4-e2b/README.md) fixture
provides canonical Gemma 4 E2B semantic and execution graph definitions plus
session metadata. No raw tensor captures are bundled in the repository fixture,
so the default example intentionally has zero tensor-anchor metric coverage.
Tests create small synthetic Safetensors shards in temporary directories and
never present them as Gemma measurements.

## Node inspection and explicit mappings

Click a semantic node or anchor (or press Enter/Space on it) to open a modal
pinned to the current layer and batch. Background selection is inert while the
modal is open. The modal opens on Parameters; Source and Execution mapping are
explicit tabs. Escape/Close restores focus to the originating graph node.

Source uses optional `semantic_evidence.json` and its referenced local source
files. The server resolves the selected node's JSON pointer, verifies the pinned
source-file SHA256, and shows the enclosing Python function where available.
Evidence line ranges are highlighted within that context. “Source hash verified”
verifies the file bytes, not the correctness of a semantic interpretation.
Missing or changed source is reported explicitly. The frontend renders source as
text, never as HTML.

Mapping requires independently selecting a reference and target execution graph,
node and indexed output tensor. No graph or tensor is chosen by default.
Selecting both outputs computes the comparison; only a successful comparison
enables Save. Errors clear prior comparison results. Load saved selection is an
explicit action and restores graph/node/output controls; merely opening the
modal does not apply a new selection.

Additional endpoints:

-   `GET /api/node?layer=N&batch=N&semantic=ID`: parameters, verified source
    contexts, execution graphs, captured tensor catalog, and saved mapping
    status.
-   `POST /api/selection/compare`: compare one explicit pair.
-   `POST /api/mappings`: validate, recompute, and persist one explicit pair.
-   `POST /api/mappings/remove`: remove this context's saved mapping and restore
    automatic anchor pairing.

POST payloads use `layer`, `batch`, `semantic`, `reference`, and `target`. The
last two fields are concrete tensor-index IDs, not op IDs. Removal only uses the
first three. A `semantic` value of `anchor:<anchor ID>` updates that report
anchor. Nodes whose output is exactly an anchor open that anchor context. Other
node mappings are standalone comparisons and do not enter layer/anchor coverage;
the modal states this distinction.

Mappings are saved atomically in the configured data directory's
`mappings.json`, scoped by layer, batch and semantic target. Records include a
dataset fingerprint (session, semantic graph, execution graph and tensor index),
both complete tensor identities, and raw-file SHA256 digests. Restarting with a
changed dataset or changing raw-file bytes invalidates the mapping. A stale
mapping is shown as stale and excluded from summaries; no cross-session or
cross-run reuse occurs. Replacing or removing a mapping is explicit. This local
backend uses a single-process, thread-safe writer.

Saved anchor mappings refresh the table, layer coverage and session summary.
Undefined individual metrics remain excluded. Neither automatic pairing nor
manual mapping performs hidden shape transformations.

## Runner run export layout

A sealed native run export (`Documents/Runs/<job id>/` on the Runner) contains
only:

-   metadata files: `job.json`, `result.json`, `terminal.json` (mandatory),
    `runtime-build.json`, and `failure.json` for failed or stopped runs;
-   `tensors/*.safetensors`;
-   `raw/*.safetensors`, `raw/generated_tokens.jsonl` and
    `raw/runtime_trace.jsonl`.

`RunnerFiles.swift` (`run_files` / `safeFile`) builds the transfer manifest from
this list and the Server's `runtime/runner_artifacts.py` (`RUN_METADATA_FILES`)
accepts exactly these paths. Change both together.

## Capture index v2

The machine-readable contract is
`src/contracts/python/package/model_debugger_contracts/schemas/capture-index-v2.schema.json`;
both writers and the importer validate against it with
`model_debugger_contracts.schema.validate`. `capture-job.schema.json` and
`tap-manifest.schema.json` describe the request the Server sends to a native
Runner and the tap manifest inside it; `src/contracts/fixtures/capture-job.json`
is decoded by the Apple Runner's lifecycle tests.

Each run directory of a job (`<job>/<run id>/`) may contain
`export/capture_index.json`. The Server rejects any `format_version` other than
`2` and any `tensor_root` other than:

-   `"run"`: paths are relative to the run directory and reference the runtime's
    original shards; the export holds metadata only (LiteRT-LM workers and the
    PyTorch Runner).
-   `"export"`: paths are relative to `export/`; standalone and iOS exports copy
    each selected shard once into `export/tensors/NNNNNN.safetensors`.

`tensors[]` records always carry `format: "safetensors"`, `path`, `key`, `dtype`
and `shape`, plus the coordinates of the producing runtime:

<!-- mdformat off(preserve GFM table layout) -->
| Runtime | Record fields | Index fields |
|---|---|---|
| LiteRT-LM (`runtime/litert_capture.py`) | the tap manifest entry (`signature`, `output_name`, `tensor_type`, …), `backend_requested`, `session` (capture subdirectory), `step`, `phase` (`prefill` when the signature names prefill, else `decode`) | `tapped_sha256`, `backend_requested`, `uncaptured_signatures`; with `export_scope: "all"` also `tokens`, `admitted_input_ids` and `native_trace: {path, sha256}` |
| LiteRT-LM without a native trace (`runtime/litert_dump_inference.py`) | the same tap manifest fields plus `forward_id` and `turn` | `runtime: "LiteRT-LM"`, `export_scope: "all"`, `capture_scope: "native_dump_inferred"`, `forwards[]` (one per dumped `(signature, step)`, boundary `inputs`/`outputs` for `input_pos`, `logits`, `activations`, an `input_identity` over the serialized conversation plus released prefix, `vocab_identity`), `resources[]`, `token_records[]` (only when every check passed), `tokens`, `generation`, `inference {basis, status, checks}` |
| PyTorch (`model_debugger_runner/pytorch_capture.py`) | `module_path`, `module_type`, `edge`, `invocation`, `call_seq`, `output`, `output_tree`, `layer`, `kind`, `phase`, `step`, `capture_key`, `scope: "module"`, and `forward_id`, `module_call_id`, `output_path`, `turn`, `pos_offset` when recorded | `runtime: "PyTorch"`, `export_scope: "all"`, `capture_scope: "generation"`, `input_identity`, `topology {blocks_path, n_layers, width, sites[]}`, `skipped`, `forwards`, `kv_snapshots`, `token_records`, `generation`, `resources[]` (boundary and KV shards with their `capture_key`), `forward_input_proofs` |
<!-- mdformat on -->

Publication (`capture_importer.py`) validates the declared root, requires the
PyTorch `input_identity` to match the run result, requires the native trace
named by `native_trace` to exist under the tensor root with the recorded digest,
and copies each referenced shard once into the immutable saved capture. Appended
turns retain each earlier shard once. The saved `tensor_index.json` always uses
session-relative paths and never depends on the job directory remaining present.

## Inferred native generation evidence

When a native run returns no `raw/runtime_trace.jsonl` but its dump carries
Decode logits, `import_capture.py` calls
`litert_dump_inference.infer_generation`. It derives the evidence Token Diff
needs from the dump alone, which is sound only for the configuration the native
Runner admits today: greedy (`temperature 0`, `topK 1`), single-candidate,
non-speculative decoding. Each dumped `(signature, step)` becomes one forward;
the k-th released token in `generated_tokens.jsonl` is bound to the k-th Decode
forward only if its logits argmax equals that token, Decode `input_pos` equals
`step - 1` and advances by one per forward, and the Decode count is the released
count or one more (the trailing unreleased sample, usually a stop token, is kept
as `generation.unreleased_argmax`). Any failed check keeps forwards and
resources, writes no `token_records`, marks Decode forwards ineligible for
pairing and records the reason in `inference.status`. Publication re-runs the
same checks (`verify_inferred_index`) before accepting the index. Logical
contexts are the serialized conversation plus the released prefix, so two sides
pair up to and including the first diverging token. The identity of the
unreleased stop sample is not derivable. Records carry `basis:
"inferred_greedy_argmax"`; `token-analysis` reports it per pair. Each side's
distribution summary also carries `selected_probability` and `margin` (most
likely minus second most likely probability): properties of one runtime's own
step, valid in any context. Once the two generations have diverged a pair
reports `distribution.context: "different"`: paired metrics and `delta` stay
null, while `rows` keep each side's own candidates (one vocabulary is enough for
rows; one context is required only to subtract them).

KV cache shards in the dump (`<signature>_post_kv_cache_{k,v}_<slot>_step_<N>`,
written by the runtime after each Prefill invocation and once more at generation
end) become `kv_snapshots` through `runtime/litert_kv_dump.py`: `prefill_post`
on the last Prefill forward and `terminal` on the last Decode forward; nothing
exists before Prefill or per Decode step. `processed_token_count` is the first
dumped `input_pos` plus the valid rows of that Prefill (checked against the
mask's attention columns) and, at generation end, the last Decode position plus
one (checked against the Decode count and, when reported, the Runner's token
count). `logical_token_ids` is `null`; `logical_context_identity` digests the
serialized conversation, admitted input IDs, processed count and, for
`terminal`, the released token IDs, so two sides pair only for the same context.
Each shard is bound to the model's `odml.cache_update` descriptor (owner layer,
key/value layout, INT8 scale) from `litert_evidence_model.describe_model`; an
unknown descriptor or a failed count check keeps no snapshot (`inference.kv`
records the reason). CPU host tensors are logical as dumped. GPU shards are the
WebGPU delegate's host download (unsigned bytes in its physical layout): the
Server normalizes them with the pinned conversion in `litert_storage` into
`<shard>_logical.safetensors` and records a storage contract with `binding:
"server_dump_normalization"` that names the untouched dump shard, the pinned
LiteRT revision from `runtime-build.json` and the Runner's WebGPU backend
evidence; every value is re-validated against the dump at publication and on
read. Without that evidence the GPU shard stays `unavailable` with its reason.
When the Apple Runner reports the admitted input (`result.json` `renderedInput`
and `inputTokens`, the rendered template as the engine tokenized it, BOS first),
the index carries `admitted_input_ids`, `input_tokens` and `serialized_input`,
Prefill identities are built from those IDs, and `tokenCount - tokenCountBefore`
must equal the admitted count plus the Decode count
(`inference.checks.input_count_reconciles`). The conversation entry then shows
`input_token_count`, `input_tokens` and `serialized_input`. Each admitted token
carries `kind` (`template`, `text` or `special`) when the token texts tile the
rendered input. With the Runner's `stopTokens`, a trailing Decode invocation
whose argmax is a configured single-ID stop token becomes a `token_records`
entry and a conversation token with `kind: "template"` when the run admitted the
same ID as a chat-template token (`<turn|>`), otherwise `kind: "special"`
(`<eos>`), `released: false`, `stop: true` (`generation.unreleased_sample`); the
runtime filtered it from the text, but it is a real sample with its own logits.
Publication spells control tokens the Runner could not (`<bos>`, `<turn|>`) from
the saved tokenizer (`text_basis: "saved tokenizer"`).

Every full-raw native import also copies the container's exact `SP_Tokenizer`
section next to the export (`index.tokenizer`), and publication keeps it once
per digest under `tokenizers/<sha256>.model`, referenced from `session.json` as
`runs[].tokenizer`. Token Diff uses it to label any vocabulary ID in the
per-step distribution; generated tokens keep the text the runtime released. The
tokenizer bytes are evidence of identity as well: their digest is the
`vocab_sha256` of the native input proofs.

## Native capture Safetensors layout

Native LiteRT-LM captures are indexed, never converted. `litert_capture.collect`
walks every `*.safetensors` file below the capture directory and reads only the
Safetensors header and metadata:

-   File metadata carries `signature` and `step`; a missing or non-integer
    `step` fails the export.
-   A stored key matches a tap when `(signature, "post_" + output_name)` is in
    the tap manifest; other keys are ignored. Matching keys must have the tap's
    dtype (TFLite tensor type mapped to the Safetensors header type, for example
    `0 → F32`, `18 → BF16`) and exact shape.
-   The same `(capture subdirectory, signature, key, step)` may appear once; a
    signature whose step is missing any expected key fails as an incomplete
    capture, and a capture with no matching tensors fails even when inference
    succeeded.

The Apple Runner's sealed run export keeps the files listed under
[Runner run export layout](#runner-run-export-layout); `raw/*.safetensors` are
the untouched runtime dumps and `raw/runtime_trace.jsonl` is the trace that
`native_trace` points at.
