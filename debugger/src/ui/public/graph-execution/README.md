# Captured execution renderer

This same-origin frame embeds the pinned Model Explorer renderer used by Graph
Diff style-r6. It has no demo data or report-loading API. The Angular
`ExecutionGraph` component supplies current `NodeDetails` through a
source/origin-checked `postMessage` channel, `model-debugger-execution`.

## Host API

-   `render` carries `{details, reference, target, metric, metrics, mapping,
    operation, theme}`. `details` is the projection `{tensors, executions}` of
    the current `NodeDetails`; sources, parameters and saved mappings never
    cross into the frame. `reference` and `target` are real TensorRecord IDs;
    `metrics` is metric-name → number/null for that selected pair.
-   `action` accepts `fit` or `readable`. Search lives in the host workspace
    toolbar, which selects the hit through the normal inspection path and the
    frame reveals it.
-   The frame emits `ready`, `tensorSelected: {side, id}`, and
    `operationSelected: {side, nodeId, graphId}`. Side is `ref` or `target`.

Execution run roles follow selected records' `run` fields, then the remaining
execution order. Each pane has an independent graph selector when its run
contains multiple graphs. Graph output metadata and actual tensor records
establish output ports; context operations do not receive fabricated outputs.
Outputs without a unique actual record remain visible and inspect their
producer. Mapping accepts captured tensor records only.

Only the selected captured pair receives numerical metrics. Operations have
neutral gray fills, unmeasured tensors have a pale blue fill, and the selected
pair retains the prototype's white/red severity endpoints. Selection uses the
local blue-outline bridge independently of numeric fills. Metric updates redraw
colors without replacing topology or moving the cameras.

## Required runtime files

-   `index.html`, `execution.css`, `execution-data.js`, `execution-view.js`.
-   `upstream/model-explorer/dist/main_browser.js` (the exact locally extended
    bundle), `dist/worker.js`, and all nine files under `dist/static_files/`:
    three font atlas PNG/JSON pairs, icon atlas PNG/JSON, and `styles.css`.
-   `model-explorer-fonts.css` and `fonts/material_icon.woff2` for the upstream
    UI icons.
-   The retained Apache `LICENSE`, upstream package metadata,
    [PROVENANCE.json](upstream/model-explorer/PROVENANCE.json), and upstream
    README accompany the runtime.

The main Angular configuration already copies `public/**`; it needs no
additional scripts or assets entry. The iframe keeps the bundled Angular/Zone
runtime separate from the application's Angular runtime. No Plotly, Dagre,
prototype shell, fixture, mapping store, or screenshot is a renderer runtime
dependency.

## DOM and observer lifecycle

This entrypoint runs deferred scripts after the DOM exists and observes its
actual host element via `ResizeObserver` and media-query observers. It does not
suppress exceptions or patch native observer behavior.

## Verification

From the repository root:

```sh
node --test src/ui/src/features/graph/execution_graph/tests/execution_graph.test.cjs
```

These deterministic adapter tests use actual data/controller source with
controlled DOM and renderer doubles; they do not replace the parent workflow's
browser/WebGL checks.

## Asset versions

The Angular iframe URL is bound to the frame HTML SHA-256. The HTML similarly
binds its CSS, adapter, controller and pinned upstream bundle to their content
hashes; the controller binds the worker URL. Run `node
scripts/version_graph_execution.mjs` after editing these assets. `npm run build`
performs this step automatically, and `npm test` checks the complete chain with
`--check`. Generated URLs are checked in so direct Angular builds use the same
versions. This prevents a new bridge consumer from loading an older cached
unversioned upstream URL.

A missing or throwing theme bridge leaves a visible reload action while
preserving available graph navigation. Graph/viewport initialization exceptions
replace the loading message with an error and reload action.

## Trust boundary

The frame is same-origin with the application, so it is a runtime isolation of
the pinned renderer's Angular/Zone runtime, not a security boundary.
`index.html` carries a Content-Security-Policy: scripts, styles, fonts, images,
fetches and the worker load only from this origin (inline styles are allowed
because the renderer injects them; images may also be `data:`/`blob:` textures).
Both sides check `origin` and `source` on every message. The pinned upstream
bundle is `ai-edge-model-explorer-visualizer@0.1.2` plus the local changes in
`upstream/model-explorer/local-changes.patch`; `PROVENANCE.json` records the
hashes, `npm test` verifies them, and `node
scripts/replay_model_explorer_patch.mjs` rebuilds the served bundle from the
upstream tarball to prove the patch is complete.
