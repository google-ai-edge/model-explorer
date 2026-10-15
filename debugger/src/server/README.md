# Model Debugger Server

The server of the Model Debugger, in Python. It serves the web UI
(`src/ui`) and its HTTP API on `127.0.0.1`, and drives two runners per
Session, Reference and Target, by the protocol in `src/protocol`: it launches
each runner, uploads the model if the runner lacks it, loads it, holds one
chat, sends each turn to both sides, and stores what both runtimes captured
for the token, logits and KV diffs.

A runner serves one connection for its whole life, and one chat per
connection. So a new chat starts fresh runners, and a Session whose Model
Server was turned off, or whose server exited, never runs again: duplicate
it instead.

## Layout

<!-- mdformat off(preserve GFM table layout) -->
| Module | What |
|---|---|
| `main.py` | the command line: flags, the workspace, uvicorn |
| `http_app.py` | every route the UI calls; loopback and same-origin only |
| `execution.py` | each Session's Model Server: start, turns, new chat, cancel, close; devices |
| `jobs.py` | the jobs the UI follows over server-sent events |
| `sessions.py`, `models.py`, `options.py`, `workspace.py` | records, model files, the UI's options as protocol messages, the workspace on disk |
| `runner_client.py`, `launchers.py` | one runner over one gRPC channel; starting runners on this computer, Android devices and iOS simulators |
| `capture_index.py`, `capture_view.py`, `tensors.py` | a chat's capture on disk and the UI's view of it |
| `analysis/` | the token, logits and KV comparisons |
<!-- mdformat on -->

## Build

Requires Python 3.11 or later. From `src/server`, once:

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
```

Then compile the protocol into `src/model_debugger/v1` (generated, not
committed; `ci/check_server.sh` does it too):

```
.venv/bin/python -m grpc_tools.protoc -I ../protocol --python_out=src \
  --grpc_python_out=src --pyi_out=src model_debugger/v1/runner.proto
```

The UI is built in `src/ui` (`npm ci && npm run build`); the command-line
runner in `src/runner` (`bazelisk build //model_debugger/runner/cli:runner`).

## Run

```
.venv/bin/model-debugger-server --model /path/to/model.litertlm
```

or `.venv/bin/python -m model_debugger_server`. Flags:

- `--workspace DIR`: where Sessions, models and captures are kept; by default
  `~/Library/Application Support/model_debugger/server` on macOS, else
  `${XDG_DATA_HOME:-~/.local/share}/model_debugger/server`. One server at a
  time may use a workspace.
- `--port N`: by default 8080, or the next free port up to 8099.
- `--ui DIR`: the built UI; by default `src/ui/dist/model_debugger/browser`
  when it is built.
- `--runner FILE`: the command-line runner, which makes this computer a
  device; by default the one Bazel built last,
  `src/runner/bazel-out/<configuration>/bin/model_debugger/runner/cli/runner`
  (not `bazel-bin`, which follows the last build, such as an App's).
- `--model FILE`: a `.litertlm` file offered as a model, by reference;
  repeatable. Models can also be uploaded from the UI.

Android devices (`adb`) and booted iOS simulators with the Runner App
installed are devices too. The server prints its URL; open it in a browser
on the same computer.

## Test

```
.venv/bin/python -m pytest -q tests
```

The tests drive in-process fake runners (`tests/fake_runner.py`); the
launcher's test runs the real command-line runner when it is built. With a
model, `tests/test_end_to_end.py` also runs the whole server with two real
runners on this computer's CPU, through the HTTP API:

```
RUNNER_MODEL=/path/to/model.litertlm .venv/bin/python -m pytest -q tests
```

`RUNNER_BINARY` names a runner other than the Bazel build. `REAL_SESSIONS`
names a directory of sessions the real runner recorded (`sessionA`,
`sessionD`) for the capture index's tests of the real format. The repository's
`ci/check_server.sh` compiles the protocol, then runs ruff (check and
format) and pytest.
