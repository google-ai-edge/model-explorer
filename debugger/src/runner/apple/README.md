# Apple Runner

The macOS and iOS targets share the Swift C API runtime, Session lifecycle and
bundled [Runner UI](../ui/README.md). Native execution stays in `Core/`; the
window, WebKit bridge, file handling and connection transport stay in `Runner/`.

## Dependencies

-   Xcode with the macOS/iOS SDKs and an Apple Silicon Mac for these targets.
-   An explicitly selected debugger-enabled LiteRT-LM checkout. The build
    scripts retain the pinned native revision and header checks; native
    dependencies are staged into ignored `Vendor/`, never committed with
    application source.
-   Python 3 for the dependency-free Xcode generator and staging tools.

No model, saved connection, Python environment or previous build is included.
The optional macOS Python adapter is configured separately using the Runner
Python package and shared contracts. This app accepts only the canonical version
2 configuration and `model_debugger_runner.python_runner_host` module. Use this
repository's `src/runner/python/tools/configure_python.py` to create that
configuration when enabling Python; old version 1 configurations are rejected.
No configuration is written by building or migrating the app. Python is not
required to compile it.

## Build macOS

From the repository root, select an existing runtime checkout explicitly:

```bash
LITERT_LM_ROOT=/absolute/path/to/debugger_runtime \
  bash src/runner/apple/Tools/build_macos.sh
```

This stages native libraries, checks the local Runner logo, generates the Xcode
project and builds
`src/runner/apple/.build/mac/Build/Products/Debug/ModelDebuggerMac.app`. Use
`RUNNER_DERIVED_DATA=/absolute/build/path` for another output directory. The
deployment target follows the staged libraries' minimum supported macOS version.
`build_macos_runtime.sh` is a separate, explicit native Bazel rebuild; it is not
needed when the selected runtime already supplies compatible debugger-enabled
libraries.

To build an already staged package without rewriting Vendor:

```bash
python3 src/runner/apple/Tools/generate_project.py
xcodebuild -project src/runner/apple/ModelDebuggerRunner.xcodeproj \
  -scheme ModelDebuggerMac -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath src/runner/apple/.build/mac build
```

## iOS

`Tools/build_native.sh` builds and stages the pinned iOS XCFramework and
companion libraries from `LITERT_LM_ROOT`. Generate the project, select the
`ModelDebuggerRunner` scheme in Xcode, and configure your development team for a
physical device. Generated project files do not contain developer identities.
The iOS target requires iOS 17 or later. Device execution is a separate
validation step.

## Isolated checks

Run validation and lifecycle tests with a fresh test home and build cache:

```bash
mkdir -p /tmp/runner-check-home
CFFIXED_USER_HOME=/tmp/runner-check-home \
  LITERT_LM_LIBS="$PWD/src/runner/apple/Vendor/macos_arm64" \
  RUNNER_BUILD_ROOT=/tmp/runner-validation \
  bash src/runner/apple/Tools/check_macos.sh
CFFIXED_USER_HOME=/tmp/runner-check-home \
  RUNNER_BUILD_ROOT=/tmp/runner-lifecycle \
  bash src/runner/apple/Tools/check_runner.sh
```

The default macOS check runs 16 validation and 10 capture-storage filesystem
checks; the Runner check runs 55 lifecycle checks. They use synthetic inputs and
do not prove model inference. `CFFIXED_USER_HOME` isolates Foundation's
user-home lookup so these tests do not load an existing Python Runner
configuration. Real native smoke checks require an explicitly provided model/job
and new output directory.

### Capture smoke fixture (DEBUG builds)

A DEBUG build launched with `--capture-smoke-test` loads `smoke-job.json` and
`smoke-model.litertlm` from the App's Documents directory and runs one capture
through the real C API without a Server. Stage both files in the container first
(Xcode > Devices, or the simulator's Documents folder); release builds omit this
hook. The CLI smokes in `Tests/*Smoke.swift` cover the same path on macOS.

## Resources and integration boundary

`../ui/index.html` contains its fonts and their license. The bundle resource
folder is `ui`; no Debugger web frontend is needed. `Brand/` contains the
approved raster, SVG hash and attribution; `CLiteRTLM/` contains pinned headers
and their license. The project generator preserves a single local logo source in
`../ui/debugger-logo.svg`.

Server-side pairing, job export, and capture import tools live in
[`src/server/`](../../server/README.md). Application storage remains outside the
repository; building the app does not copy or initialize connection keys, Python
configuration, model caches, or captures.
