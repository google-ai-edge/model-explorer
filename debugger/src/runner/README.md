# Model Debugger Runner

The Runner hosts a model runtime and serves the Model Debugger Protocol
(`../protocol/`) over gRPC. The Model Debugger server connects to a Runner,
uploads and loads a model, chats with it, and reads the tensors the runtime
captures during generation.

The runtime sits behind one interface (`model_debugger/runner/llm/runtime.h`).
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) is the runtime today;
PyTorch is planned.

The Runner is written in C++. It ships as a command-line program and as Apps
for macOS and iOS (SwiftUI) and Android (Kotlin, Jetpack Compose).

## Platform support

<!-- mdformat off(preserve GFM table layout) -->
| Platform | Runs as | Status | Backends |
|---|---|---|---|
| macOS (Apple silicon) | command line, App | supported | CPU, GPU (Metal) |
| iOS Simulator | App | supported | CPU |
| iOS device | App | supported | CPU, GPU (Metal) |
| Android (arm64-v8a) | App | supported | CPU, GPU (OpenCL) |
<!-- mdformat on -->

## Prerequisites

- macOS 26.4 or later on Apple silicon
- Xcode 26.4 or later
- [Bazelisk](https://github.com/bazelbuild/bazelisk), which runs the Bazel
  version pinned in `.bazelversion`
- [Git LFS](https://git-lfs.com)

To build the Android App, you also need:

- Android SDK Platform 35, Build-Tools 35.0.0 and Platform-Tools (`adb`),
  located by `ANDROID_HOME`
- Android NDK r28b, located by `ANDROID_NDK_HOME`

To run the iOS App on a device, you also need:

- The device paired with this Mac, with Developer Mode on (**Settings >
  Privacy & Security > Developer Mode**)
- A provisioning profile for `com.google.ai.edge.modeldebugger.runner` under
  `~/Library/MobileDevice/Provisioning Profiles/`. To generate one:
  1. Sign in to **Xcode > Settings > Accounts** (`⌘,`) with your Apple ID (a
     free Personal Team works and lasts 7 days).
  2. Create a temporary iOS App project in Xcode with Bundle Identifier
     `com.google.ai.edge.modeldebugger.runner`, enable **Automatically manage
     signing**, select your Team, and select your **connected iPhone/iPad** as
     the run destination so Xcode provisions the device and writes `iOS Team
     Provisioning Profile: com.google.ai.edge.modeldebugger.runner`.

## Usage

Every form runs the same Runner (`model_debugger/runner/runner.h`) and
listens on `127.0.0.1:8769` by default.

### macOS

#### Command line

```shell
bazel run //model_debugger/runner/cli:runner -- --listen 127.0.0.1:8769
```

#### App

```shell
bazel run //model_debugger/runner/app/apple:macos -- --listen 127.0.0.1:8769
```

Note: The server's automatic device launcher (`launchers.py`) starts the macOS
command-line runner (`//model_debugger/runner/cli:runner`), the Android App
over `adb`, the iOS device App over `devicectl`, and the iOS Simulator App.
The macOS App is launched manually as shown here.

### iOS

#### Simulator

```shell
bazel build //model_debugger/runner/app/apple:ios --ios_multi_cpus=sim_arm64
unzip -o "$(bazel cquery --output=files //model_debugger/runner/app/apple:ios \
    --ios_multi_cpus=sim_arm64)" -d /tmp/runner-ios
xcrun simctl boot "iPhone 17"
xcrun simctl install booted "/tmp/runner-ios/Payload/Model Debugger Runner.app"
xcrun simctl launch booted com.google.ai.edge.modeldebugger.runner \
    --listen 127.0.0.1:8769
```

#### Device

```shell
bazel build //model_debugger/runner/app/apple:ios --ios_multi_cpus=arm64
unzip -o "$(bazel cquery --output=files //model_debugger/runner/app/apple:ios \
    --ios_multi_cpus=arm64)" -d /tmp/runner-ios
xcrun devicectl device install app --device DEVICE \
    "/tmp/runner-ios/Payload/Model Debugger Runner.app"
ADDRESS=$(xcrun devicectl list devices --json-output - 2>/dev/null | jq -r \
    --arg d DEVICE '.result.devices[] | select(.deviceProperties.name == $d)
    | .connectionProperties.tunnelIPAddress')
xcrun devicectl device process launch --device DEVICE \
    com.google.ai.edge.modeldebugger.runner --listen "[$ADDRESS]:8769"
```

The server connects to `[$ADDRESS]:8769`, the device's address on the tunnel
Xcode keeps to a paired device.

- `DEVICE` is the device's name in `xcrun devicectl list devices`.
- Once installed and trusted on the device (in Settings > General > VPN &
  Device Management), the server's `launchers.py` automatically discovers a
  paired device and launches the App on a free port when the device is
  unlocked.
- The App listens on the tunnel address alone, so only this Mac reaches it.
- Use the address, not `<device>.coredevice.local`: gRPC does not resolve
  `.local` names.

### Android

```shell
bazel build --config=android_app //model_debugger/runner/app/android:runner
adb install -r bazel-bin/model_debugger/runner/app/android/runner.apk
adb shell am start -n com.google.ai.edge.modeldebugger.runner/.RunnerActivity \
    --es listen 127.0.0.1:8769
adb forward tcp:8769 tcp:8769
```

`adb forward` makes the App reachable from the Mac at `127.0.0.1:8769`.

## Testing

```shell
RUNNER_MODEL=/path/to/model.litertlm bazel test //...                    # CPU and GPU
RUNNER_MODEL=/path/to/model.litertlm bazel test //... --config=thread    # ThreadSanitizer
RUNNER_MODEL=/path/to/model.litertlm bazel test //... --config=address   # AddressSanitizer
```

`RUNNER_MODEL` must be the absolute path of a `.litertlm` model; the tests
are written against Gemma 4 E2B. Without it, the tests that load a model are
skipped.

The full check, including formatting and the App builds, is the repository's
`ci/check_runner.sh`:

```shell
RUNNER_MODEL=/path/to/model.litertlm ../../ci/check_runner.sh
```

## Project structure

```
model_debugger/runner/
├── runner.h, runner_c.h   the Runner, and its C API for the Apps
├── cli/                   the command line
├── app/apple/             the macOS and iOS App (SwiftUI)
├── app/android/           the Android App (Kotlin, Jetpack Compose)
├── service/               the protocol's gRPC service, cache directory, host info
├── llm/                   the runtime interface
│   └── litert_lm/         LiteRT-LM behind it, and the rule that builds LiteRT-LM
├── util/                  JSON helpers
└── testing/               test utilities and the CPU/GPU test macro
```
