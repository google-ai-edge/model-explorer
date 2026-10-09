# PyTorch eager worker

The workspace runs local Hugging Face text causal models through the independent
`model_debugger_runner` execution package in `src/runner/python/`, using the
in-tree `ai_edge_debugger_pytorch` capture package in `src/capture/pytorch/`.
The HTTP server remains independent of `torch`. The local Runner App owns the
supervisor and workers; the server connects to that App over loopback WebSocket.
Initialize starts a dedicated process for each session run and loads the model
once; Generate reuses that process, model and verified KV prefix.

Install the in-tree capture, contracts, and runner packages into a Python
runtime environment (`.venv`) that provides `torch`, `transformers`,
`tokenizers`, and `safetensors`:

```sh
/path/to/pytorch-runtime/.venv/bin/python -m pip install \
  -e src/capture/pytorch -e src/contracts/python -e src/runner/python
```

Then configure the App's Python adapter and restart/open the local Runner App:

```sh
python3 src/runner/python/tools/configure_python.py --pytorch-root /path/to/pytorch-runtime
```

See [Runner setup](../src/runner/python/README.md). `--pytorch-root` on the
server does not choose the Runner interpreter or enable a server execution
fallback. The setup tool writes version 2 with the Runner and shared-contract
source roots. The App rejects version-1 configurations; rerun the setup tool
after upgrading.

From this repository, after building the UI:

```sh
src/server/.venv/bin/python -m model_explorer_debugger.server \
  --pytorch-model /path/to/local-huggingface-model
```

`--pytorch-model` can repeat; registrations persist in the workspace. A
directory must contain `config.json`, its tokenizer and Safetensors weights.
Cached HF snapshot directories with blob symlinks work. The worker verifies
registered file hashes on every load, uses `local_files_only=True`, requires
Safetensors, and does not load custom remote code. LiteRT-LM preparation tools
come from the sibling `debugger_runtime` checkout or a remembered
`--runtime-root`, so both runtimes are available from the same command.

In **New session**, choose **This computer**, select **PyTorch**, choose the
registered model and configure **CPU**, **MPS** or **CUDA** when the worker
environment reports that backend. **Model dtype** supports Default (checkpoint
dtype), float32, float16 and bfloat16; hardware/model support is checked during
execution. CPU thread count applies only to CPU. Context length is capped by
model/tokenizer capacity; overflow produces an error instead of silently
truncating the prompt.

Create the session: its Model Server starts on its own and the Session list
shows it as **Starting**, then **On**. Open the session once it is on and send a
message. For a saved session, turn the **Model Server** switch on in the list or
in the session toolbar. Generation streams text and saves exact token IDs.
Temperature, top K, top P, seed, system prompt and output token limit are
applied, with greedy defaults (temperature 0, top K 1, seed 0) unless
configured. The tokenizer's chat template is used when available, otherwise
explicit role lines are serialized. Template defaults (including a model's
thinking behavior) remain active; explicit Thinking controls are unavailable.
The resident runtime keeps the conversation messages, actual token history,
consumed length and live KV cache until Close. A new Turn uses the actual chat
template again and reuses the cache only when its consumed token IDs and mask
are an exact prefix of that serialized input. Only the uncached suffix is then
fed to the model. The worker slices this suffix before calling HF input
preparation; it does not rely on version-specific `cache_position` slicing. Each
actual forward must return the expected consumed-token cache length before
sampling continues. A mismatch fails the Turn and invalidates the resident
state. Template/prefix changes, hidden tokens lost during text serialization,
unavailable cache lengths or additional tokenizer inputs cause a full Prefill
replay with an explicit `cache_reuse.reason`; the loaded model is still reused.

The Runner's Session lifecycle initializes an empty Chat. Subsequent wire
requests send only the new message; the worker owns resident history. Stop ends
the Chat and resets its Conversation/KV while retaining the loaded model. New
Chat starts empty Conversations; saved read-only history is not replayed.
Execution failure invalidates the Session execution. The server records
successful paired text before Debug export; export failure retains that text
with Debug data unavailable. The disposable worker helper's history input
remains available to standalone callers and is not the Server's Session
restoration path.

The capture package supports multiple forwards with explicit execution
identities and separate model-boundary records. Without explicit generation
context it records an unknown phase; it does not infer Prefill or Decode from
input/cache length. This worker supplies an explicit Prefill context around each
Turn's first actual forward. `pos_offset` is zero for a full replay or the
verified reused-prefix length for a suffix Prefill. Every subsequent actual
cached forward has an explicit Decode context and the actual Turn number. The
first Decode has step 0; its position offset is the actual cache position
prepared for that call. One temporary `CaptureRun` covers the whole Turn and
releases its hooks afterward. The loaded model and cache remain resident. After
sampling, the worker records the chosen token ID and text against the forward
that produced its logits. It does not patch PyTorch, compile the model or
replace its forward methods. The model's selected attention implementation (for
example SDPA) is recorded; eager execution does not force the attention
implementation named `eager`.

The raw capture includes module and model-boundary records for those forwards,
KV snapshots before/after explicit Prefill, and a terminal KV snapshot on normal
EOS or output-limit completion. Decode updates cache metadata and token
consumption without saving a full KV snapshot at every step. The terminal
snapshot contains the last cache the model actually produced: the final
generated token has not been fed back into the model and remains pending. There
is no extra forward to fill that gap. Cancellation and errors have explicit
terminal reasons and do not claim normal completion. Unsupported cache layouts
retain an unavailable state and reason instead of fabricated tensor values.

**Debug → Graph Diff** displays observed module boundaries and their numeric
comparisons. These nodes have no inferred mathematical dataflow edges. Automatic
pairing requires the same registered model identity and verified logical input
context: actual call inputs, tokenizer/serialization proof and the complete
consumed token prefix behind its cache. Different runtimes have separate
execution identities; no cross-runtime equivalence is inferred from matching
text. The worker exports every observed Prefill/Decode layer-0 module boundary,
with outer-boundary and KV resources kept separate. Unsupported module topology
fails initialization rather than pretending capture succeeded. Initialize
validates topology without running a fabricated prompt or claiming any KV
capture.

The existing job, cancellation, SSE and saved-session APIs are reused. Capture
index v2 stores runtime/module path/invocation/input-or-output coordinates plus
Safetensors path/key/shape/dtype. Exported module records also retain their
actual `forward_id`, `module_call_id`, invocation and available output-path
metadata when the capture recorded them. The worker uses `export_scope='all'`
when producing its capture index; see
[Capture index v2](data-contract.md#capture-index-v2). The result's
`capture_scope` and `published_scope` are `all_observed_forwards`;
`decode_capture` and `kv_capture` describe whether those observations actually
exist. A separate `raw_capture` declares `scope=whole_generation` with actual
raw indices, counts and KV snapshot states; it is null for Initialize.

The historical `raw/prefill/` directory name is retained, but its scope now
covers the entire generation. It contains `forward_index.json`, `kv_index.json`,
`tokens.jsonl` and `generation.json`, with model-boundary and KV shards under
`boundaries/` and `kv/` respectively. Token records identify their source
forward and whether a later forward consumed each ID, including its actual input
position. The generation record includes its status, stop reason,
generated/processed token counts and pending token IDs. Processed cache length
is distinct from generated token count and can include prompt positions.
`forward_input_proofs` identifies the actual prepared input tensors and prior
consumed prefix of every call; equal suffix tokens alone do not establish equal
cache context. Scores remain unknown unless the sampler supplies them.

The complete export includes forward records, KV snapshots, token events,
generation state and independent raw resource references for immutable
publication. A KV tensor or model-boundary tensor is not presented as a module
graph node. Original Safetensors bytes and explicit resource identities remain
the source of numeric evidence; no `.npy` storage or implicit layout conversion
is included.

`python -m model_debugger_runner.pytorch_worker --serve` accepts the stdin JSON
request-file protocol and a Close command. EOF also releases the model. The App
supervisor entry point is `python -m model_debugger_runner.python_runner_host
--root <pytorch-root>`. These private IPC entry points belong to the Runner
package (`model_explorer_debugger.runtime.python_runner_host` remains as a CLI
launch bridge for Apps registered with that module name). The Python
`execute(request, emit)` helper is a disposable wrapper over the same resident
implementation. Results include `runtime_instance`, `worker_pid`,
`model_load_count`, `turn_sequence`, `context_token_ids`,
`processed_token_count`, `pending_token_ids` and cache-reuse details.
`token_count` is the number of newly generated tokens in that result.

Run the Python Runner and capture regression suite with
`bash ci/test_python_runner.sh`.
