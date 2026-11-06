# AI Edge Debugger

The Model Debugger: a server that drives runners on devices, loads a model
there, chats with it, and reads the tensors the runtime captures. One
repository, four parts:

- `src/protocol/`: the server <-> runner protocol: one `.proto`, one README.
- `src/runner/`: the runner (C++, Swift and Kotlin, Bazel): hosts LiteRT-LM,
  serves the protocol; a command line and Apps for macOS, iOS and Android.
- `src/runner_pytorch/`: the PyTorch runner (Python, Bazel): hosts a Hugging
  Face Gemma 4 E2B on this computer's CPU or GPU, serves the same protocol;
  and, in `pytorch_collector/`, `pytorch-collector`, the capture library it
  captures with, which lives here.
- `src/server/`: the server (Python, Bazel).
- `src/ui/`: the web UI the server serves (TypeScript, Angular): sessions,
  chat, and the token, logits and KV diffs.
- `ci/`: one check script per part, and one for all; the server's layer and
  metric catalog checks (shell, Python).

## Check

```
# The first run fetches and builds LiteRT-LM.
RUNNER_MODEL=/path/to/model.litertlm ci/check_all.sh
```

`ci/check_all.sh` runs each part's check: `ci/check_runner.sh` (clang-format,
swift-format, ktfmt for the Android App's Kotlin, buildifier's format and
lint for the Starlark files, `bazel test` on the CPU and on the GPU, the iOS
App for the simulator, and the Android App when `ANDROID_HOME` and
`ANDROID_NDK_HOME` name an SDK and an NDK; a `.litertlm` model is needed),
then `ci/check_runner_pytorch.sh` (buildifier, ruff, and `bazel test`, with
real two-turn runs when `RUNNER_PYTORCH_MODEL` names a Gemma 4 E2B
directory), then `ci/check_server.sh` (buildifier, ruff's check and format,
the layer check `ci/check_server_layers.py`, `ci/check_metric_catalog.py`,
which checks the UI's copy of the metric catalog against the server's,
`bazel test`, which runs the server with the runner just built and the model,
and the wheel, which carries the UI: it builds the UI first and checks the
wheel holds its `index.html`), then `ci/check_ui.sh` (`npm ci` once, the
format and layering checks, the production build, `npm test`, and the
structure-graph prototype's checks). The runner and the server are Bazel
workspaces of their own. The protocol has no check of its own: both parts
compile its `.proto`.
