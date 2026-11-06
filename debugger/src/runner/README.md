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

| Platform | Runs as | Status | Backends |
|---|---|---|---|
| macOS (Apple silicon) | command line, App | supported | CPU, GPU (Metal) |
| iOS Simulator | App | supported | CPU |
| iOS device | App | supported | CPU, GPU (Metal) |
| Android (arm64-v8a) | App | supported | CPU, GPU (OpenCL) |

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

- The device paired with this Mac, with Developer Mode on
- A provisioning profile for `com.google.ai.edge.modeldebugger.runner`. Xcode
  creates it when you sign any project with this bundle ID for your team, with
  the device connected. A profile of a free personal team lasts 7 days.

## Usage

Every form runs the same Runner (`model_debugger/runner/runner.h`) and
listens on `127.0.0.1:8769` by default.

### macOS

#### Command line

```shell
bazel run //model_debugger/runner/cli:runner -- --listen 127.0.0.1:8769
```

#### Keeping a build per LiteRT-LM commit

A Runner is built against one LiteRT-LM commit, `LITERT_LM_COMMIT` in
`MODULE.bazel`; `bazel-bin` holds only the last build. `keep` builds the
command-line Runner and keeps it, so that the server can launch any kept
commit:

```shell
bazel run //model_debugger/runner/cli:keep               # the pinned commit
bazel run --repo_env=LITERT_LM_COMMIT_MACOS_ARM64=COMMIT \
    //model_debugger/runner/cli:keep                     # another commit
```

Each commit gets a directory of its own:

```
~/Library/Application Support/model_debugger/runners/macos_arm64/COMMIT/
├── runner                         finds the libraries beside it
├── liblitert-lm.dylib, libLiteRt.dylib, libGemmaModelConstraintProvider.dylib
├── libwebgpu_dawn.dylib, libLiteRtWebGpuAccelerator.dylib, libLiteRtTopKWebGpuSampler.dylib
└── build.json                     commit, commit_date, cl, built_at, sha256
```

- `COMMIT` is the whole 40-digit commit. `LITERT_LM_COMMIT_<PLATFORM>`
  overrides the pinned commit for any build of that platform.
- Before a build is kept, it is started from its directory with a HOME of its
  own and must report the commit in `GetInfo`. Keeping a commit again replaces
  its directory in one atomic swap. To forget a commit, delete its directory.
- Each commit LiteRT-LM is built from also takes about 6.5 GB under
  `~/Library/Caches/model_debugger/litert_lm/COMMIT/`; delete it once the
  commit is kept and no longer built.
- The Runner builds only against LiteRT-LM commits made after its C API
  change of 2026-10-01; older ones do not compile with it.

#### App

```shell
bazel run //model_debugger/runner/app/apple:macos -- --listen 127.0.0.1:8769
```

### Phone Apps' versions

Each phone App carries the LiteRT-LM commit it is built against, read from
its LiteRT-LM repository's `commit.json`, in its own version, where the
server reads it without launching the App: Android's `versionName` is
`<commit> (<commit date>)`, its `versionCode` 1; iOS takes numbers alone, so
`CFBundleShortVersionString` is the commit's date, `YYYY.MM.DD`, and
`CFBundleVersion` the commit's first 12 hex digits in decimal, with the
whole commit as `LiteRTLMCommit`. `aapt2 dump badging runner.apk` and
`plutil -p "Model Debugger Runner.app/Info.plist"` show them.

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
- The App listens on the tunnel address alone, so only this Mac reaches it.
- Use the address, not `<device>.coredevice.local`: gRPC does not resolve
  `.local` names.
- The first launch needs the developer to be trusted on the device, in
  Settings > General > VPN & Device Management.

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
├── service/               the gRPC service, model store, cache directory, host info
├── llm/                   the runtime interface
│   └── litert_lm/         LiteRT-LM behind it, and the rule that builds LiteRT-LM
├── util/                  JSON helpers; the exit for a disk with no room
└── testing/               test utilities and the CPU/GPU test macro
```
