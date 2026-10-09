# Python Runner

`model_debugger_runner` contains the App-owned JSON-lines supervisor, resident
PyTorch workers, conversation/cache handling and capture export. Its local
dependency is `model_debugger_contracts`; it does not import a Server package.

## Install

Use a configured environment that provides `torch`, `transformers`,
`tokenizers`, and `safetensors`. A runtime root contains `.venv/bin/python` and
installs this repository's local capture, contracts, and runner packages into
that interpreter.

From this repository's root, install the three local packages into that
environment:

```sh
/path/to/runtime/.venv/bin/python -m pip install \
  -e src/capture/pytorch -e src/contracts/python -e src/runner/python
```

The package itself declares only its lightweight contracts dependency. Import
and protocol checks work without loading model libraries. Real model execution
requires the configured environment described above.

## Entrypoints

```sh
/path/to/runtime/.venv/bin/python -m model_debugger_runner.python_runner_host \
  --root /path/to/runtime
```

The supervisor receives private App commands on stdin and emits JSON lines on
stdout. It owns one resident worker for each Session/role. Its worker entrypoint
is `python -m model_debugger_runner.pytorch_worker --serve`; the supervisor
normally starts it. Neither command opens an HTTP listener or loads a model
until an initialization request arrives. This package does not provide an App
window or Server transport.

## Supported scope

-   Local Hugging Face text causal language models with Safetensors weights and
    tokenizer files; models requiring remote code are unsupported.
-   CPU, MPS and CUDA when available in the configured environment; default,
    float32, float16 and bfloat16 precision. Runtime tests below exercise CPU.
-   Text streaming, system prompts, greedy or temperature/top-K/top-P sampling,
    a seed and 1–256 output tokens. Multimodal input, custom stop sequences and
    thinking controls are unsupported.
-   Resident Session workers, verified conversation cache reuse, input
    preflight, cancellation and Chat reset. Context overflow rejects input
    without truncation.
-   One export covering every observed Prefill and Decode forward, token
    records, forward input proofs and available KV snapshots. Module tensor
    capture is currently fixed to sublayers in layer 0; it is not a complete
    model graph. Raw tensors retain their Safetensors format. Export failure
    preserves generated text and resident conversation state.

The worker CLI accepts `--serve` only. The App-owned supervisor controls its
lifecycle; individual request files are not a separate execution mode. Capture
export always covers the complete observed generation, with no selected-Prefill
or selected-forward mode.

## App configuration

Generate a reviewable configuration file without changing the App's saved
settings:

```sh
python3 src/runner/python/tools/configure_python.py \
  --pytorch-root /path/to/runtime --output /tmp/python-runner.json
```

The tool probes that environment and writes version 2 with `executable`,
`runtimeRoot`, detected `backends`, independent `packageRoot`,
`contractPackageRoot`, and `module=model_debugger_runner.python_runner_host`.
`--package-root` and `--contracts-root` override the default source locations.
Omitting `--output` intentionally writes the macOS App's saved configuration; an
App that supports version 2 is required to use it.

## Checks

Lightweight package/protocol tests need only Python:

```sh
bash ci/test_python_runner.sh
```

To require all 6 package-boundary checks and 31 runtime tests, including the 21
real tiny-model tests:

```sh
bash ci/test_python_runner.sh --runtime-root /path/to/runtime
```

The script runs parent tests and child workers with that root's same `.venv`.
The fixture reads `RUNNER_TEST_RUNTIME_ROOT` explicitly and does not infer an
interpreter from the capture package's physical location. Models and captures
are generated in temporary directories; no model download is required. The real
checks use CPU and cover BF16 captures, cache reuse, input rejection, reset,
cancellation, failure recovery and worker close. They do not establish GPU
execution.
