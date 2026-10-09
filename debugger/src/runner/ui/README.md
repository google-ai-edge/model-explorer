# Shared Runner UI

`index.html` is the production UI, shared by platform shells. It loads from the
app bundle, needs no frontend build step, and has no network requests or CDN
dependencies. `debugger-logo.svg` is the shared source checked against the Apple
brand hash. The main Debugger web application lives in
[`src/ui`](../../ui/README.md).

## Host contract v1

Before loading the page, provide:

```js
window.RunnerHost = {
  postMessage(serializedCommand) { /* deliver to the native host */ }
};
```

The page sends `{ "version": 1, "action": "ready" }` when its renderer is ready.
The host then pushes a full state snapshot using
`window.RunnerUI.update(state)`. Push subsequent changes after native state
mutations. The schema's Apple implementation is
[`RunnerUIState.swift`](../apple/Runner/RunnerUIState.swift).

<!-- mdformat off(preserve GFM table layout) -->
| Field | Source / meaning |
| --- | --- |
| `version` | `1`; reject unsupported versions |
| `device` | Name, platform label, foreground requirement |
| `connection` | Typed `unconfigured`, `waiting`, `connected`, `failed`; display text, transport, optional address |
| `activity` | Typed `idle`, `importing`, `verifying`, `loading`, `generating`, `validating`, `stopping`, `releasing`, `failed`; native detail, output, optional capture count / transfer |
| `activity.transfer` | Acknowledged `received` and `total` bytes; absent for transports without progress events |
| `owner` | Optional activated Server/Session identity and `runId` (`ref` or `target`); one owner per Runner |
| `lifecycle` | Native phase (`waiting`, `active`, `ending`, `ended`) |
| `build` | Actual native build identity, label and optional version/CL |
| `sessions[]` | Successful resident model for the single owner/role; `key`, `id`, `role`, `modelHash`, optional `tokens`, `contextLength`, `capturePoints`; `runtime` and `backend` identify the adapter |
| `runtimes[]` | Advertised `id`, `backends`, and adapter transport; render actual capabilities |
| `interfaces[]` | Host-approved `address` / `label` choices |
| `manual` | Prepared job / cached model availability and job metadata |
| `actions` | Named capability flags including current busy/ownership restrictions |
| `environment` | Optional native device/build facts; platform, model, chip, architecture, OS, physical RAM, logical processors and available build identity |
| `environmentState` | Optional timestamped thermal/power/process memory readings; unavailable measurements omitted |
<!-- mdformat on -->

Context-token counts describe the last completed request. PyTorch uses its
verified processed cache length, which is unavailable before its first forward.
Eager capture has no fixed selected capture-point count. No polling of an
executing C handle is required. Remove a resident record on close, disconnect or
worker failure. A pending worker is not a loaded Session. Unknown optional
values may be omitted; do not replace them with zero.

Actions use `{version:1, action:<name>}`. Supported names: `importJob`,
`importModel`, `run`, `stop` (manual capture), `stopGeneration`, `endSession`,
`exportCapture`, `refreshInterfaces`, `exportConnection`, `refreshEnvironment`,
and `setAddress` (adds `address`). Omit unsupported action keys; send `false`
for supported but currently unavailable actions. The host rechecks capabilities
at execution time and validates addresses against actual interfaces. UI
disabling is not a security boundary.

File selection, credential export and capture export use host dialogs. Never
pass connection tokens, file bytes, raw tensors or native pointers to the UI.
Render text as text; serialize state as structured arguments, not interpolated
JavaScript. Accept commands only from the bundled main frame; block navigation
to remote pages. A UI reload receives the current snapshot and must not create,
initialize or close a native Session.

Environment is a collapsed Settings section. `refreshEnvironment` samples native
readings without changing runtime ownership. `processResidentBytes` is process
resident memory; physical-iOS `processMemoryHeadroomBytes` is process allocation
headroom, not free system RAM. Hosts without environment support render a clear
unavailable message. Never synthesize zero values or infer hardware from the
selected platform.

## Platform adapters

-   **macOS / iOS — implemented:** `RunnerWebView.swift` embeds
    [WKWebView](https://developer.apple.com/documentation/webkit/wkwebview),
    with a script-message handler for actions. SwiftUI only owns the app/window
    and native file dialogs. C API engines and Conversations remain in Swift.
    macOS can also own a local Python supervisor and resident PyTorch workers.
-   **Android — future:** package these same assets; use Android's
    [local-content WebView guidance](https://developer.android.com/develop/ui/views/layout/webapps/load-local-content)
    with `WebViewAssetLoader`, then implement this host contract and
    runtime/file capabilities. Android runtime, USB transport and lifecycle
    handling are not supplied by this UI work.
-   **Linux — future:** a WebKitGTK shell can connect actions through
    [UserContentManager](https://webkitgtk.org/reference/webkit2gtk/stable/method.UserContentManager.register_script_message_handler.html).
    A native runtime/transport adapter and packaging still need implementation.

## Develop / preview

From the repository root:

```sh
python3 tools/serve_runner_ui.py
```

Open `http://127.0.0.1:8872/` for the standalone renderer. It starts in the
waiting state without a host. This server only exposes the renderer and its
logo; it has no model execution, state simulation or access to local data.

Build Apple apps using `src/runner/apple/Tools/generate_project.py` and the
existing Xcode schemes. The generator includes `src/runner/ui` as a resource
folder in both targets, so changes to this renderer ship to both platforms.

## Active ownership

The owner is assigned by the Server control protocol, independently of UI
contract v1. Native Stop generation requests cancellation of the current
two-sided turn; End Session requests release of both Runners. Neither action can
be implemented by changing this renderer's snapshot. The native coordinator
validates owner and connection identity and waits for resource cleanup. Closing
WebKit does not end a Session. The Session list owns lifecycle actions in the
main web application.

The Gallery-derived renderer uses the platform system font stack and bundles no
fonts. It renders only the host contract above. Model-library previews and
simulated file imports are not part of the production renderer. Appearance
changes are local UI state, remembered in the renderer's `localStorage`, and do
not send commands to the native runtime.
