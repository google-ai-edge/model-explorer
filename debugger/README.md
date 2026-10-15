# AI Edge Debugger

The Model Debugger: a server that drives runners on devices, loads a model
there, chats with it, and reads the tensors the runtime captures. One
repository, four parts:

<!-- mdformat off(preserve GFM table layout) -->
| Part | What | Language |
|---|---|---|
| [`src/protocol/`](src/protocol/README.md) | the server ↔ runner protocol: one `.proto`, one README | — |
| [`src/runner/`](src/runner/README.md) | the runner: hosts LiteRT-LM, serves the protocol; a command line and Apps for macOS, iOS and Android | C++, Swift and Kotlin, Bazel |
| [`src/server/`](src/server/README.md) | the server | Python |
| [`src/ui/`](src/ui/README.md) | the web UI the server serves: sessions, chat, and the token, logits and KV diffs | TypeScript, Angular |
| `ci/` | one check script per part, and one for all | shell |
<!-- mdformat on -->

## Prerequisites

- **Python ≥ 3.11** (for `src/server/`)
- **Node.js ≥ 24** and **npm** (for `src/ui/`)
- **Apple Silicon macOS**, **Xcode 26.4**, **Bazelisk** (`bazel`), **Git LFS** (`git lfs install`), and **`clang-format`** (for `src/runner/`; optional **Android SDK/NDK** via `ANDROID_HOME` and `ANDROID_NDK_HOME`)
- **`shellcheck`** (for `ci/check_all.sh`)

## Quickstart

```sh
# 1. Set up the Python server and compile the gRPC protocol
cd src/server
python3 -m venv .venv
.venv/bin/pip install -e '.[test,proto]'
.venv/bin/python -m grpc_tools.protoc -I../protocol \
    --python_out=src --pyi_out=src --grpc_python_out=src \
    ../protocol/model_debugger/v1/runner.proto
cd ../..

# 2. Build the Angular web UI
(cd src/ui && npm ci && npm run build)

# 3. Build the local C++ CLI runner (first run fetches and builds LiteRT-LM)
(cd src/runner && bazelisk build //model_debugger/runner/cli:runner)

# 4. Start the server with a .litertlm model and open the printed URL
src/server/.venv/bin/model-debugger-server --model /path/to/model.litertlm
```

## Check

```sh
RUNNER_MODEL=/path/to/model.litertlm ci/check_all.sh    # the first run fetches and builds LiteRT-LM
```

`ci/check_all.sh` runs each part's check: `ci/check_server.sh` (ruff,
pytest), `ci/check_ui.sh` (`npm ci` once, the production build, `npm test`,
`npm run test:unit`) and `ci/check_runner.sh` (clang-format, swift-format,
ktfmt for the Android App's Kotlin, buildifier's format and lint for the
Starlark files, `bazel test` on the CPU and on the GPU, the iOS App for the
simulator, and the Android App when `ANDROID_HOME` and `ANDROID_NDK_HOME`
name an SDK and an NDK; a `.litertlm` model is needed). The protocol has no
check of its own: both parts compile its `.proto`.
