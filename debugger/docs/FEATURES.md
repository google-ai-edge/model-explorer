# Debugger, Server and Runner features and scope

The repository includes the main Debugger UI in `src/ui`, the HTTP Server in
`src/server`, the shared contracts in `src/contracts`, the PyTorch capture
package in `src/capture/pytorch`, and the execution/capture Runners in
`src/runner`.

## Implemented workflows

<!-- mdformat off(preserve GFM table layout) -->
| Workflow | Implementation | Current limits |
|---|---|---|
| Main Debugger UI | Session management, Chat/Token Diff, Graph Diff, KV Diff, capture navigation and saved view state | Analysis depends on captured evidence; missing values remain unavailable. Model execution requires an available Runner. |
| Server orchestration | Workspace/session APIs, pairing, job preparation, capture imports and bounded numerical analysis | Loopback coordinator; model execution belongs to Runner. |
| Native model execution | Apple app, LiteRT-LM Engine, resident Session and per-Chat conversation/cache | Text only. macOS arm64: CPU and conditional GPU support. iOS: CPU. |
| Python model execution | App-owned supervisor and resident PyTorch workers | Local Hugging Face causal text models. CPU/MPS/CUDA depend on the external environment. |
| Session lifecycle | Initialize, input preflight, generate, cancel, reset Chat, end Session; disconnect/heartbeat cleanup | Server-driven ownership uses the Server in `src/server`. Reset retains the loaded model while clearing conversation state. |
| Native capture | Prepared job/model import, selected taps, raw capture validation, export and sealed transfer | Native jobs accept 1024/4096 context, 1–32 output tokens and 1–16 taps. Manual capture remains available without Server ownership. |
| Native Token Diff evidence | Without a runtime trace, the Server infers forwards, full-vocabulary logits, final activations and token bindings from the RuntimeDebugger dump (greedy, single-candidate, non-speculative runs), re-checking every binding against the logits argmax | Marked `basis: inferred_greedy_argmax`; pairs stop at the first diverging token; the unreleased stop sample is unavailable. KV snapshots exist only after Prefill and at generation end, with derived contexts; GPU KV bytes are normalized with the pinned WebGPU conversion and stay unavailable without the Runner's build and backend evidence. See the [data contract](data-contract.md#inferred-native-generation-evidence). |
| Python capture | Prefill/Decode module tensors, token records, forward input proofs and available KV snapshots | In-tree `ai_edge_debugger_pytorch` package (`src/capture/pytorch`). Module capture is fixed to layer-0 sublayers; it is not a full model graph. |
| Runner UI | Connection/activity, owner, resident model, latest output, capture count, transfer progress, connection/environment settings, manual capture and appearance | Offline renderer of host state. File dialogs, execution and export are owned by the native host. |
| Model identity | Shared file-content hashing and verification, including tokenizer/template inputs | Registration checks identity; it does not establish backend residency or capture completeness. |
<!-- mdformat on -->

### Native backend evidence

The macOS adapter advertises GPU only when its three required WebGPU libraries
are present. It computes library hashes, compares them with available build
provenance and records Engine initialization. `acceleratorResidencyVerified`
remains false: these facts do not prove that model operations ran on the GPU.
iOS GPU execution is rejected by this implementation. See
[NativeBackendSupport.swift](../src/runner/apple/Core/NativeBackendSupport.swift).

### Python generation and capture

Generation supports a system prompt, text streaming, seed, greedy or
temperature/top-K/top-P sampling and 1–256 output tokens. Context overflow is
rejected without truncation. Verified token-prefix and cache-length checks
govern cross-turn reuse. Multimodal input, custom stop sequences and thinking
controls are unsupported.

Export includes every observed generation forward and keeps raw Safetensors
shards. Unavailable evidence is omitted or reported as unavailable. A capture
export error preserves successfully generated text and resident conversation
state. See the
[Python Runner scope](../src/runner/python/README.md#supported-scope).

## Outside this repository

-   Android and Linux native hosts are not implemented here.
-   Model weights, Python runtime virtual environments (`torch`, `transformers`,
    `tokenizers`), and native LiteRT-LM runtime source/libraries are external
    dependencies.

## Server

The Server provides workspace/session APIs, authenticated Runner coordination,
transactional SQLite jobs and events, recoverable capture publication, SSE, and
bounded supervised numerical analysis. See
[Server architecture](server-robustness.md).

Explicit cross-runtime comparison (`cross_runtime_pairs`, `cross_runtime_kv`,
`cross_runtime_positions`) requires the LiteRT side to record a
`token_replay_proof`. No Runner in this repository produces that proof, so
registering such a pair fails with an explicit error; the analysis code and its
tests are retained as the reference for a future Runner implementation.
