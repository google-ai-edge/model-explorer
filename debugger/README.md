# AI Edge Debugger

> [!NOTE]
> **Status: preview.** The code is published for review. End-to-end runs need
> the debugger runtime, which is coming soon.

Find out where two runs of the same on-device language model stop agreeing, and
why. A Session runs one model twice, as a **Reference** and a **Target** (for
example LiteRT-LM on CPU and on GPU), sends both the same messages and records
what each one computed:

-   **Chat**: both conversations side by side.
-   **Token Diff**: the generated tokens aligned across the two runs, with each
    step's probabilities, logits divergence and activation error up to the first
    step that differs.
-   **Graph Diff**: the captured tensors of one forward pass, placed on the
    model graph.
-   **KV Diff**: the two KV caches compared position by position.

Values that were not captured read as not captured, and evidence that is derived
rather than recorded is labelled as derived. See
[implemented features and limits](docs/FEATURES.md) for the exact scope.

This repository contains the Debugger web app, its HTTP Server, the Runner Apps
that execute models (Apple native and Python), the PyTorch capture package, the
shared Runner UI, and the contracts between them.

## Requirements and platform support

<!-- mdformat off(preserve GFM table layout) -->
| Mode | Supported platforms | Required tools |
|---|---|---|
| **Saved-session viewer & UI development** | Linux, macOS | Python ≥ 3.10, Node.js ≥ 24, npm |
| **Live LiteRT-LM & PyTorch Runner execution** | Apple Silicon macOS (`arm64`), iOS 17+ (LiteRT-LM CPU) | Xcode, `debugger_runtime` checkout (Python ≥ 3.11, `torch` ≥ 2.4 for PyTorch capture), local `.litertlm` or Hugging Face Safetensors model files |
<!-- mdformat on -->

## Start the Debugger

From the repository root, build the main UI and install the Python packages into
`src/server/.venv`:

```sh
npm --prefix src/ui ci
npm --prefix src/ui run build
python3 -m venv src/server/.venv
src/server/.venv/bin/python -m pip install \
  -e src/contracts/python -e src/runner/python -e 'src/server[apple,test]'
```

To explore the bundled Gemma 4 E2B saved session immediately (no model weights
or Runner App required):

```sh
src/server/.venv/bin/python -m model_explorer_debugger.server --data examples/gemma4-e2b
```

To start a writable workspace for live comparisons:

```sh
src/server/.venv/bin/python -m model_explorer_debugger.server
```

Open `http://127.0.0.1:8080/`. Without flags the Server opens
`.local-runtime/workspace` in this checkout, serves the UI you just built,
listens on `8080` (or the next free port when `8080` is busy) and prints the
effective configuration, including the URL, at startup. LiteRT capture
preparation needs the `debugger_runtime` checkout: a sibling directory is found
automatically; otherwise pass `--runtime-root /path/to/debugger_runtime` once
and the workspace remembers it. `--model /path/to/model.litertlm` registers a
local model once instead of uploading it. `--help` lists the remaining flags.

The `debugger_runtime` checkout's own `.venv` publishes the captured data, so it
needs the contracts package once. Without it, turns still save their text but no
Debug data appears:

```sh
/path/to/debugger_runtime/.venv/bin/python -m pip install -e src/contracts/python
```

Reading saved Sessions works without a running model. Running one needs a
registered Runner App and model files; see the next section.

Restart the Server after you pull or edit Server code. It imports some modules
on first use, so a long-running process ends up mixing old and new code.

For UI development, start the same Server on port `8080`, then run `npm --prefix
src/ui start` and open `http://127.0.0.1:4201/`. The Angular development server
proxies `/api/` requests to `127.0.0.1:8080`. See
[UI setup and source ownership](src/ui/README.md) for checks and build details.

## Run a first comparison on a Mac

1.  Build the Mac Runner App. `LITERT_LM_ROOT` is the `debugger_runtime`
    checkout:

    ```sh
    LITERT_LM_ROOT=/absolute/path/to/debugger_runtime bash src/runner/apple/Tools/build_macos.sh
    ```

2.  Register the build in `.local-runtime/workspace/runner-builds.json`. The
    `id` is the App's `CFBundleVersion`, an `@`, and the `litertLMCommit` from
    its `Contents/Resources/runtime-build.json`:

    ```json
    {"version": 1, "builds": [{
      "id": "1@a0f1e97361155c8bb9cd3d6aa045c04320d4334b",
      "label": "Debug build · CPU/GPU · macOS",
      "kind": "macos-app", "devices": ["local"], "platform": "macOS", "protocolVersion": 5,
      "path": "/absolute/path/to/ai-edge-debugger/src/runner/apple/.build/mac/Build/Products/Debug/ModelDebuggerMac.app"
    }]}
    ```

3.  Start the Server. `--model` registers a local `.litertlm` file so it is not
    uploaded through the browser; add `--tap-manifest` and `--semantic` for a
    model that already carries capture points (see
    [Server setup](src/server/README.md)).

4.  In the UI choose **New session**, pick the device, the Model Server build,
    the runtime options and the model for each side, then **Create session**.
    Creating a Session starts its Model Server. The list shows **Starting**,
    then **On**; a refusal shows as **Failure** with the Server's reason.

5.  **Open** the Session once its Model Server is on and send a message in Chat.
    Switch to **Debug** for Token Diff, Graph Diff and KV Diff.

6.  Turn the Model Server off with the same switch, in the list or in the
    Session toolbar. This ends the running chat and frees the device; captured
    turns stay. A saved Session can be opened for reading at any time, and the
    toolbar switch turns its Model Server on again.

Good to know:

-   One Mac serves both sides of one Session at a time. A second Session cannot
    start until the first one's Model Server is off.
-   A native LiteRT-LM turn generates at most 32 tokens, the context is 1024 or
    4096 tokens, and a model can carry up to 16 capture points. These limits are
    enforced by the Server and by the Runner App.
-   Without a runtime trace, the native Token Diff and KV Diff evidence is
    derived from the runtime's debug dump and is marked as such; it covers
    greedy, non-speculative runs. The [data contract](docs/data-contract.md)
    states what is derived and how it is checked.
-   Rebuild the Runner App after pulling Runner changes. The Server accepts
    Runner protocol 4 and 5; protocol 5 adds file links and binary transfers,
    which cut a local Model Server start from about two minutes to well under
    one.
-   An iPhone provides one side. See the
    [Apple build instructions](src/runner/apple/README.md).
-   For Hugging Face models through the Python Runner, see
    [PyTorch worker](docs/pytorch-worker.md).

## Preview the Runner UI

From the repository root, preview the offline Runner UI:

```sh
python3 tools/serve_runner_ui.py
```

Open `http://127.0.0.1:8872/`. Without an App host, this is a waiting renderer;
the preview server does not execute models.

For execution, use the [Apple build instructions](src/runner/apple/README.md) or
configure the [Python Runner](src/runner/python/README.md). Both require
explicit external runtime dependencies and local model files.

The Server runs API-only when no UI build exists or `--ui` points elsewhere. See
[Server setup](src/server/README.md) and
[architecture and recovery](docs/server-robustness.md).

## Source organization

<!-- mdformat off(preserve GFM table layout) -->
| Location | Responsibility |
|---|---|
| `src/ui` | Main Angular Debugger application, build and browser assets |
| `src/server` | HTTP/SSE, workspace transactions, Runner coordination and isolated analysis |
| `test` | Server behavior and integration tests |
| `src/capture/pytorch` | In-tree `ai_edge_debugger_pytorch` forward-hook tensor and KV cache capture package |
| `src/contracts/python` | Shared model identity and error contracts, the capture-job and Runner wire schemas and their validator |
| `src/contracts/fixtures` | Schema-checked example messages that the Server and Swift tests both read |
| `src/runner/python` | App-owned Python supervisor and resident model workers |
| `src/runner/ui` | Shared, offline Runner renderer |
| `src/runner/apple` | Apple native host, execution implementation and build tools |
| `examples/gemma4-e2b` | A small saved Session the browser tests run against |
| `docs` | Features and limits, data contract, Runner architecture, Server robustness |
| `ci`, `tools` | Check scripts, repository checks and the local LiteRT launcher |
<!-- mdformat on -->

## Checks

Before running the check suite for the first time, ensure `src/server/.venv` is
installed (see [Start the Debugger](#start-the-debugger)) and download the
Playwright Chromium binary used by `ci/build_ui.sh`:

```sh
npm --prefix src/ui run e2e:browsers
```

Then run the verification scripts from the repository root:

```sh
bash ci/check_repository.sh
bash ci/build_ui.sh
bash ci/test_contracts.sh
SERVER_PYTHON=src/server/.venv/bin/python bash ci/test_server.sh
bash ci/test_python_runner.sh
bash ci/check_apple.sh
```

`ci/build_ui.sh` ends with the browser smoke (`npm --prefix src/ui run
test:e2e`): Playwright starts the Server on `examples/gemma4-e2b` with the built
UI and drives Chromium through the Session, Token Diff, Graph Diff and KV Diff
views.

`ci/check_apple.sh` compiles the Swift host checks on macOS; it needs the
vendored native libraries (or `LITERT_LM_LIBS`) and runs the native validation
only when `LITERT_LM_LIBS` or `LITERT_LM_ROOT` is set.

Require the Python real-model CPU checks with an existing capture environment:

```sh
bash ci/test_python_runner.sh --runtime-root /path/to/runtime
```

The Apple README documents isolated native validation, storage and lifecycle
checks. Builds and tests do not establish GPU execution unless a real model and
backend evidence are explicitly verified.

Build caches, model weights, private settings, captures and native `Vendor/`
libraries stay outside Git. Required third-party notices remain with their
assets.
