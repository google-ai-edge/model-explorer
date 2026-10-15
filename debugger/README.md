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

From this `debugger/` directory:

```sh
# 1. Set up the Python server and compile the gRPC protocol
cd src/server
python3 -m venv .venv
.venv/bin/pip install -e '.[test,proto]'
.venv/bin/python -m grpc_tools.protoc -I ../protocol \
    --python_out=src --grpc_python_out=src --pyi_out=src \
    model_debugger/v1/runner.proto
cd ../..

# 2. Build the Angular web UI
(cd src/ui && npm ci && npm run build)

# 3. Build the local C++ CLI runner (first run fetches and builds LiteRT-LM)
git lfs install
(cd src/runner && bazelisk build //model_debugger/runner/cli:runner)

# 4. Start the server and open the printed URL (http://127.0.0.1:8080/)
src/server/.venv/bin/model-debugger-server
```

Models (`.litertlm`) can be uploaded directly from the web UI when creating a
Session, or pre-registered by reference on startup without copying the file:

```sh
src/server/.venv/bin/model-debugger-server --model /path/to/model.litertlm
```

## Setting Up Target Devices

The server (`src/server/src/model_debugger_server/launchers.py`) automatically
discovers and launches runners on four target environments:

### 1. This Computer (macOS CLI Runner — CPU & Metal GPU)

`"This computer"` in the UI device selector runs the headless C++ CLI runner
(`//model_debugger/runner/cli:runner`), spawned on demand by the server on a
free localhost port:

```sh
cd src/runner
bazelisk build //model_debugger/runner/cli:runner
```

- The server automatically finds the binary under
  `src/runner/bazel-out/<config>/bin/model_debugger/runner/cli/runner` (or you
  can pass `--runner /path/to/runner` explicitly).
- If `"This computer · Unchecked"` is disabled in the UI, build
  `//model_debugger/runner/cli:runner` above. Note that building only the
  standalone macOS SwiftUI app (`//model_debugger/runner/app/apple:macos`) does
  not populate the CLI binary path used by the server.

### 2. iOS Device (`devicectl` — CPU & Metal GPU)

1. **Enable Developer Mode**: Connect the iPhone or iPad to your Mac via USB,
   trust the computer, and turn on **Settings > Privacy & Security > Developer
   Mode** on the device.
2. **Create a local provisioning profile in Xcode (once per 7 days for free
   Personal Teams)**:
   - Open **Xcode > Settings > Accounts** (`⌘,`) and sign in with your Apple ID.
   - Create a temporary iOS App project in Xcode (`File > New > Project... >
     iOS > App`) with Bundle Identifier
     `com.google.ai.edge.modeldebugger.runner`.
   - Under **Signing & Capabilities**, check **Automatically manage signing**,
     select your Team, and choose your **connected iPhone/iPad** (not a
      simulator) as the run destination in the top toolbar so Xcode generates
     `iOS Team Provisioning Profile: com.google.ai.edge.modeldebugger.runner`
     under `~/Library/MobileDevice/Provisioning Profiles/`.
3. **Build and install the iOS Runner App**:
   ```sh
   cd src/runner
   bazelisk build //model_debugger/runner/app/apple:ios --ios_multi_cpus=arm64
   unzip -o "$(bazelisk cquery --output=files //model_debugger/runner/app/apple:ios \
       --ios_multi_cpus=arm64)" -d /tmp/runner-ios
   xcrun devicectl device install app --device <DEVICE_NAME_OR_UDID> \
       "/tmp/runner-ios/Payload/Model Debugger Runner.app"
   ```
4. **Trust the developer certificate on the device**: On first install with a
   Personal Team, open **Settings > General > VPN & Device Management** on the
   iPhone/iPad and trust your developer certificate.
5. **Run from the UI**: Keep the device connected and unlocked. The server
   automatically detects the paired device and installed app via
   `xcrun devicectl`, launches `com.google.ai.edge.modeldebugger.runner` over
   Xcode's CoreDevice IPv6 tunnel (`[$ADDRESS]:<port>`), and supports both
   **CPU** and **GPU (Metal)** backends.

### 3. iOS Simulator (`simctl` — CPU)

```sh
cd src/runner
bazelisk build //model_debugger/runner/app/apple:ios --ios_multi_cpus=sim_arm64
unzip -o "$(bazelisk cquery --output=files //model_debugger/runner/app/apple:ios \
    --ios_multi_cpus=sim_arm64)" -d /tmp/runner-ios-sim
xcrun simctl boot "iPhone 17"
xcrun simctl install booted "/tmp/runner-ios-sim/Payload/Model Debugger Runner.app"
```

Any booted simulator with the Runner App installed appears automatically in the
UI device selector (`CPU` backend).

### 4. Android Device (`adb` — CPU & OpenCL GPU)

```sh
cd src/runner
bazelisk build --config=android_app //model_debugger/runner/app/android:runner
adb install -r bazel-bin/model_debugger/runner/app/android/runner.apk
```

Any connected Android device (`arm64-v8a`) with the Runner APK installed
appears automatically in the UI device selector and is port-forwarded over
`adb` on demand.

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
