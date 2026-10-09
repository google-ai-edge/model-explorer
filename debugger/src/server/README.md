# AI Edge Debugger Server

From the repository root:

```bash
python3 -m venv src/server/.venv
src/server/.venv/bin/python -m pip install \
  -e src/contracts/python -e src/runner/python -e 'src/server[apple,test]'
npm --prefix src/ui ci && npm --prefix src/ui run build
src/server/.venv/bin/python -m model_explorer_debugger.server
```

The `[apple,test]` extras and the Runner package make `bash ci/test_server.sh`,
the native Runner transport, and the Runner integration tests runnable from this
environment.

Without flags the Server resolves everything from the checkout and prints the
result at startup, with the origin of each value:

<!-- mdformat off(preserve GFM table layout) -->
| Setting | Default | Override |
|---|---|---|
| Workspace | `.local-runtime/workspace` in the checkout | `--workspace DIR`, or `--data DIR` to read one saved session |
| UI | `src/ui/dist/model_explorer_debugger/browser` when built | `--ui DIR`; API only when no build exists |
| Port | `8080`, matching `src/ui/src/proxy.conf.json`; when it is busy the next free port up to `8099` is used and shown in the summary | `--port N`, which must be free |
| LiteRT preparation tools | the root remembered in the workspace, else the sibling `../debugger_runtime` checkout with its `.venv` | `--runtime-root DIR`, remembered in `server-settings.json` |
| PyTorch environment | the root remembered in the workspace | `--pytorch-root DIR`, remembered |
| Local models | registrations persist in the workspace | `--model FILE.litertlm`, `--pytorch-model DIR` |
<!-- mdformat on -->

Paths given on the command line are the only way to point the Server at local
files; browser requests use opaque artifact IDs.

Open `http://127.0.0.1:8080/`. For UI development, keep the API on `8080` and
run `npm --prefix src/ui start`; Angular serves the UI on `4201` and proxies
`/api/`. See [UI setup](../ui/README.md). `src/runner/ui/` is the separate
renderer bundled by the Runner App.

Use `--data /absolute/path/to/session` to read another saved session. See
[the data contract](../../docs/data-contract.md) for JSON and raw tensor
formats. Run tests from the root with `SERVER_PYTHON=src/server/.venv/bin/python
bash ci/test_server.sh` (editable installs plus `unittest discover -s test`).
Editable installs leave no `src/server/build/`; delete that directory if an
earlier non-editable install created it.

The Server owns session APIs, scheduling, Runner transport, capture imports and
numeric analysis. Model execution lives in `src/runner/apple/` and the
independent `src/runner/python/` package. The Server depends on shared contracts
in `src/contracts/python/`; it does not require the Runner package for normal
HTTP operation. Execution code imports `model_debugger_runner`; shared errors
and model identity use `model_debugger_contracts`. `python -m
model_explorer_debugger.runtime.python_runner_host` provides a CLI launch bridge
for registered Mac Apps that invoke that module, starting the independent Runner
supervisor. The App reads only the version-2 local configuration written by the
Runner setup tool. See [Runner setup](../runner/python/README.md) for the
separate Python environment, canonical module and configuration command.

### KV position analysis

`GET /api/kv-analysis?turn=1` (with `session_id` in workspace mode) returns
stored snapshot and explicit-comparison contexts. Each available layer/K/V row
contains exact logical-position and KV-head cells; metrics reduce only the batch
and head dimension axes. `relative_l2` is a ratio, `max_abs` is absolute error,
and `cosine_distance` is `1 - CosSim`. Whole-tensor summaries use the same
metric keys. Undefined or non-finite values are `null`. Explicit cross-runtime
cells use the verified comparison view, including declared permutation and
dequantization; original tensors remain unchanged. Snapshot cells require proven
correspondence, a recorded four-axis layout and a sequence dimension equal to
the valid logical range. Larger capacity buffers are not assigned guessed
logical positions.

Responses expose analysis budgets: 100,000 cells, 32 million compared elements,
and 128 million declared evidence elements per request. A row exceeding a budget
returns `analysis_limit` with no cells, retaining its raw evidence identity.
Positions are never silently sampled or truncated. Pair declarations are
filtered by Turn and scope before proof validation; corrupt pairs do not hide
other saved KV evidence, and ambiguous pair coordinates are reported without
choosing a pair.

For long contexts, use the bounded APIs instead of the legacy full-cell
response:

-   `GET /api/kv-analysis?turn=1&mode=metadata`: the same context identities,
    separate Ref/Target snapshot identities, layers and per-side
    `token_structure`, with empty `cells` and `token_metrics`. Snapshot metadata
    does not read tensor payloads.
-   `GET
    /api/kv-range?turn=1&context=...&start=0&end=131072&bins=128&kind=key&head=max`:
    shared half-open position bins for every layer. Each chart metric has `min`,
    `max`, `min_position`, `max_position`, `valid_count`; missing coverage stays
    null. `head` accepts `max`, `mean`, or a head index. Optional `formula` adds
    each bin's `match_count`, independently of the displayed head.
-   `GET /api/kv-selection?turn=1&context=...&start=65537&end=65538&layer=0`:
    directly compares all stored batch/head/channel elements at each token, then
    reports each K/V metric's worst finite value, original position and valid
    count across the selection. CosSim uses its minimum; error metrics use their
    maximum. Omit `layer` for separate per-layer rows. A single-position
    selection also returns exact `token_metrics`, including zero-norm/non-finite
    status.
-   `GET
    /api/kv-find?turn=1&context=...&formula=relative_l2%20%3E%20100%25&offset=0&limit=40`:
    safe three-valued formula evaluation across the context, with sorted
    position / layer / K/V results and exact counts for readable comparable
    rows. `complete` and `unavailable_rows` identify evidence outside that
    readable scope. A cached bounded block-count index lets subsequent pages
    read only their matching blocks.

Each side's slice is limited to 262,144 elements; charts use at most 128 bins
and Find returns at most 40 results. The serialized reply/index LRU holds at
most 8 MiB and 8 entries. Identical concurrent requests share one calculation,
with two numerical workers and a bounded waiting queue. File identities and
declarations invalidate these caches. These endpoints do not raise the legacy
full-cell limits. Explicit comparisons still run the existing proof validator,
cached only while all registered dependencies and session bindings are
unchanged. Proofs exceeding the new conservative 8-million-element validation
cap remain `analysis_limit`; bounded snapshot access does not imply that every
oversized explicit proof is supported.

`GET /api/kv-head?turn=1&context_id=...&layer=0&kind=key&position=0&head=0`
returns the exact selected head vector, source dtype/shape, explicit comparison
transforms, channel Ref/Target values, `delta = Target - Reference`, absolute
delta and the largest absolute-delta channel. Its `metrics` are computed across
channels in this selected vector: `cosine_similarity`, `cosine_distance`,
`relative_l2`, `mean_abs`, `rmse`, `max_abs`. Source BF16/INT8 storage is
distinct from a dequantized FP64 comparison view. Raw files are never rewritten.
Add `session_id` in workspace mode. If the tensor has more than one batch,
`batch` is required and an omitted batch returns `batch_selection_required`; no
batch is implicitly selected or aggregated. Missing comparison sides retain
readable single-side vectors with null differences/metrics. Positions outside a
layer's recorded range return `position_unavailable` with source metadata.
Unproven layouts retain metadata and produce no guessed channel coordinates. A
selected head is limited to 16,384 channels; larger vectors return
`analysis_limit`. The head endpoint now slices the selected vector directly; it
does not load the entire cache or compare every stored position first. File
SHA256 verification is streamed and cached by immutable file identity, with
changes invalidating the cache.

The legacy tensor API reads Safetensors directly using each tensor index
record's `format`, relative `path`, and exact `key`. Bounded KV reads use
Safetensors slices or explicitly indexed `.npy` files with read-only memory
mapping. BF16 is retained on disk and in the reader; metrics accumulate in
float64. After updating this checkout, repeat the editable contracts/server
installation above to install the Safetensors and `ml_dtypes` reader
dependencies.

The data directory must be writable to save/remove mappings. Source and tensor
reads stay within this configured directory. Browser requests never provide
filesystem paths. A successful mapping save revalidates tensor contents and
atomically replaces `mappings.json`.

## Homepage metadata

`GET /api/sessions` returns the single configured capture summary plus supported
operation flags. `POST /api/sessions/rename` takes `{id, name}` (1–160 nonblank
characters) and saves a separate `.debugger-session.json` atomically. The ID is
derived from the saved session manifest, not an invented capture-run ID;
original creation time is `null` unless recorded. Original session JSON, runtime
settings, tensors and mappings remain unchanged. Session creation, duplication
and deletion are not available in this saved-capture server.

For source-checkout development, prefix the server/test command with
`PYTHONPATH=src/server/package` to use current source rather than an older
installed package.

Homepage management uses `GET /api/sessions` and `POST
/api/sessions/{create,update,duplicate,rename,delete,restore}`. Saved captures
are immutable except their display name; copies are drafts without execution
evidence. Delete hides a record; restore recovers it. `.debugger-workspace.json`
stores draft metadata independently of the original manifest. An empty list
retains a configuration template.

`POST /api/artifacts/upload` accepts an octet-stream body and URL-encoded
`X-File-Name`; bytes are streamed to a unique `.debugger-artifacts/` directory
with size/disk/path checks and interrupted-upload cleanup. Uploaded files and
captured files are retained when sessions are hidden. Hugging Face references
are metadata only; no download or inference is claimed. Session management
remains usable without model execution; initialization/generation additionally
require an available Runner.

## Local LiteRT-LM on Apple Silicon macOS

For local Hugging Face models, see
[PyTorch eager worker setup](../../docs/pytorch-worker.md). Both runtimes use
the same workspace/job APIs and Safetensors reader. The Server routes
initialization, generation and close to the selected Runner App. It does not
launch a LiteRT-LM or PyTorch model worker. Its remaining runtime subprocesses
prepare and inspect LiteRT-LM capture models.

Build the UI and configure the [Apple Runner](../runner/apple/README.md), then
start from the repository root:

```sh
npm --prefix src/ui ci
npm --prefix src/ui run build
src/server/.venv/bin/python -m model_explorer_debugger.server \
  --model ../debugger_runtime/artifacts/gemma4-e2b-tapped.litertlm \
  --tap-manifest ../debugger_runtime/artifacts/tap-manifest.json \
  --semantic examples/gemma4-e2b/semantic.json
```

Open `http://127.0.0.1:8080/`. Create a Session with available Runner devices
and registered or uploaded model files. Creating it starts its Model Server: the
Session list shows **Starting**, then **On**, and **Open** unlocks once it is
on. Open the Session and enter a prompt. For a saved Session, turn the **Model
Server** switch on in the list or in the Session toolbar. Both Runner slots
initialize before the Session becomes active. A local Mac can provide Reference
and Target slots; an iPhone provides one. Unavailable Runners fail explicitly,
with no Server execution fallback.

The sibling `debugger_runtime` checkout is found automatically; pass
`--runtime-root` once for another location and the workspace remembers it.
`tools/run_local_litert.sh` is an environment-variable wrapper over the same
command (`LITERT_LM_ROOT`, `MODEL_DEBUGGER_WORKSPACE`, `MODEL_DEBUGGER_PORT`,
`MODEL_DEBUGGER_UI_ROOT`). The runtime checkout supplies its `.venv`,
`debugger_tap`, FlatBuffers and builder dependencies. The App uses its bundled
native C API for LiteRT-LM execution. Setting `--runtime-root` does not
configure an App's Python interpreter or start model execution in the Server.
Registering a model with startup flags avoids uploading another copy;
registrations persist.

Use the Runner's advertised options and supported model/capture configuration.
The current Apple LiteRT-LM adapter supports CPU and its native capture limits;
older Python-worker GPU/custom settings do not enable a fallback. `--semantic`
associates a reviewed graph with a verified tap manifest. Arbitrary selected
outputs do not gain inferred semantic identities.

Workspace APIs:

-   `GET /api/runtime/capabilities`: supported options and registered models.
-   `POST /api/sessions/{id}/initialize`: `{request_id}`; reserves and starts
    both Runners.
-   `POST /api/sessions/{id}/turns`: `{request_id, prompt, max_output_tokens}`.
-   `GET /api/jobs/{id}`: persisted job status and current streamed text.
-   `GET /api/jobs/{id}/events`: SSE with event IDs; reconnect using
    `Last-Event-ID` or `?after=N`.
-   `POST /api/jobs/{id}/cancel`: `{}`; Stop ends the writable Chat and retains
    loaded models.
-   `POST /api/sessions/{id}/close`: releases both Runner sides.
-   `GET /api/sessions/{id}/capture`: latest completed capture metadata.
-   Existing capture and mapping routes use `?session_id=ID` in workspace mode.

`request_id` makes retries idempotent; a changed payload with the same ID is
rejected. Browser requests use registered artifact IDs. The service binds to
loopback and validates Host/Origin. Captures, saved text and model files remain
independent from active Runner ownership. New Chat starts empty Conversations;
saved read-only Chat history is not replayed into a new execution.

See [Runner architecture](../../docs/runner-architecture.md) for the current
execution contract.

### Select LiteRT-LM capture points in Create Session

Choose or upload a `.litertlm` file in Create Session. The text graph is scanned
automatically in a background process while the rest of the form remains
editable. The scanner reads container metadata and the prefill/decode TFLite
structure; it does not run inference or unpack/copy weight sections. SHA256
verification reads model bytes. Identical registered models share a scan, cached
under `<workspace>/tap-scans/`.

Enable **Capture intermediate tensors**, then **Add capture points**. Search
names, operator types, or indices; optionally filter by signature. Select up to
16 concrete output tensors per model. Details show the full tensor name, builtin
operator type and tensor index. Search terms are ANDed; `layer_1/
pre_attention_norm` selects Layer 1 without matching Layer 10. The first 80
matching outputs are shown; refine search to narrow larger results. Selected
points can be removed directly or changed with the checkbox/keyboard. The
reviewed E2B Layer 0 RMSNorm configuration remains available as **Use
recommended** when its exact section hash matches.

Reference and Target share selections only when their artifact IDs match.
Different model files require independent selections. Each signature remains an
explicit selection; the UI does not infer correspondence between Prefill and
Decode. Changing the file cannot transfer coordinates from another model.

**Create session** saves the configuration, starts preparation and queues the
Model Server start behind it; the page returns to the Session list, which shows
**Starting** until the Model Server is on. Preparation failure preserves the
draft and reads as **Failure**; the switch retries from the preparation.
Initialize and Generate are blocked until preparation succeeds. The existing
fixed `gemma4-e2b-layer0-rmsnorm-v1` API profile remains compatible; the
selector saves `custom-outputs-v1` and `tap_points` keyed by model artifact ID.

The preparation worker reuses the configured runtime's `debugger_tap` /
FlatBuffers implementation and `litert_lm_builder` package. It verifies the
scan's exact section hash and selected output identities, promotes those
outputs, checks every buffer payload, repacks into a separate model, and
verifies the resulting signature bindings. Original model files are never
rewritten. Prepared caches include the source hash and selected-point
configuration. Both run artifact IDs are replaced atomically after preparation
completes. A single active prepare/inference task and a separate one-worker scan
queue bound concurrency.

Current scope: macOS arm64; one text `prefill_decode` TFLite section.
Audio/vision sections, PyTorch, shared-subgraph signature aliases, dynamic/empty
shapes, variable tensors, and control-flow outputs are not exposed as supported
custom capture points. Structural eligibility does not prove native backend
support; initialization and actual capture must verify that. Custom captures
retain concrete execution identities and original Safetensors data; automatic
semantic anchoring remains limited to the reviewed mapping, so arbitrary points
do not acquire inferred layer/semantic labels.

API: `POST /api/artifacts/<registered-id>/scan` starts/reuses a scan, and `GET`
reads its state/results. Create/update a session with `tap_profile:
"custom-outputs-v1"` and `tap_points: {"<artifact-id>": ["decode:101:0",
"prefill_128:112:0"]}`, using IDs from its completed scan. Start `POST
/api/sessions/<id>/prepare` with a unique `request_id`; existing job JSON/SSE
and cancellation endpoints apply. Paths are resolved server-side from registered
IDs, never browser-supplied filesystem paths.

Generated models and manifests live in `<workspace>/prepared-models/`. Peak free
space for preparation is approximately four model copies plus 1 GiB.
Completion/cancellation cleans temporary preparation files; abrupt process/OS
termination can retain unregistered partial files. Added outputs can affect
backend fusion/timing.

## Session-owned Runner lifecycle

The Server speaks Runner control protocol v5 and still accepts v4 Runners. Each
active Session owns two Runner slots, one Reference and one Target. One physical
Mac can provide both slots for the same Session; an iPhone provides one. The
**Model Server** switch, in the Session list and in the Session toolbar, is the
only lifecycle control: it reads Failure, Starting, On or Off, turns the Model
Server on, cancels a start, retries after a failure and, after a confirmation,
turns it off, which releases both sides. Stop generation ends the Chat while
retaining models and ownership. Runner binaries older than v4 need rebuilding.
See [Runner architecture](../../docs/runner-architecture.md).

## Server robustness

The normal CLI runs FastAPI/Uvicorn with one workspace coordinator. See
[architecture, recovery and operational limits](../../docs/server-robustness.md).
Workspace metadata, admitted jobs and ordered events are authoritative in
`workspace.sqlite3`. The first start migrates legacy JSON metadata and keeps its
original files under `.legacy-metadata/`. Stop the old server before upgrading a
workspace; never run old and new binaries against the same directory.

Analysis runs in supervised child processes. Defaults: 2 workers, 8 waiting
requests, 60-second deadline, 8192 MiB per worker. Override with
`--analysis-workers`, `--analysis-waiting`, `--analysis-timeout` and
`--analysis-memory-mib`. Overload and worker interruption return HTTP 503 with
`Retry-After: 1`. `/api/health` checks HTTP availability; `/api/diagnostics`
reports coordinator and analysis status. The HTTP server binds only to loopback.
