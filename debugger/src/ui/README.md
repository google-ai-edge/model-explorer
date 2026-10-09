# AI Edge Debugger UI

Angular + TypeScript + SCSS. Start the [Python server](../server/README.md) on
port 8080, then run `npm ci` and `npm start` here. Open http://127.0.0.1:4201/;
the development server proxies `/api/**` to port 8080. Use `npm run verify` for
format checks, contract type checks, all regression scripts, and a production
build. Output: `dist/model_explorer_debugger/browser/`. The existing
Angular project and package names are retained for build compatibility.
Requires Node.js ≥ 24 and npm ≥ 10 (regression scripts use native Node
TypeScript support).

## Source ownership

<!-- mdformat off(preserve GFM table layout) -->
| Directory                    | Responsibility                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `src/app/`                   | Thin ReportPage shell, toolbar, workspace navigation and application preferences   |
| `src/data/`                  | HTTP requests, current capture loading and runtime request lifecycle               |
| `src/data/contracts/`        | Explicit wire types; independent of components and demo fixtures                   |
| `src/features/sessions/`     | Session list, create/edit/configuration, Runner options and model pickers          |
| `src/features/conversation/` | Chat and Token Diff, alignment, search, bookmarks and reading state                |
| `src/features/graph/`        | Architecture layout, batch/layer metrics, mapping and node inspector               |
| `src/features/kv/`           | KV range, selection and head evidence                                              |
| `src/shared/`                | Controls and presentation primitives with multiple real callers                    |
| `src/theme/`                 | Semantic tokens, Material theme, typography and application-wide interaction rules |
| `src/fixtures/`              | Frozen example data validated against the explicit contracts                       |
<!-- mdformat on -->

Keep business state and component layout with their feature. Add a shared
abstraction only when existing callers need the same behavior. Component files
use `.ts`, `.ng.html`, and `.scss`, following Model Explorer. `npm run format`
uses the pinned formatter and Angular template parser.

Production styles are owned by their current components. All runtime assets and
verification dependencies are included in this repository.

## Capture identity

`WorkspaceStateService.context()` distinguishes the parent Session, active Chat,
and actual capture record. Capture-scoped API methods receive that record ID
explicitly; missing IDs fail before a request. Runtime and report state attach
to the same resolved context, and async completion must still belong to the
current attachment. Management requests such as listing or creating Sessions do
not use a capture ID.

## Loading and returning to a view

Session loading publishes the original capture independently of Graph. Graph
enrichment is a separate projection: entering Graph loads its semantic model,
overview and current comparison. Leaving cancels unfinished Graph requests;
completed results remain cached for the current capture. Late responses cannot
replace newer capture or observation data.

The shared capture, Turn and phase remain the navigation authority. Ordinary
view restoration never replays an old Turn. Conversation v3 snapshots preserve
display, selection and independent Chat/Debug reading anchors; Graph v1
snapshots preserve filters, panels and architecture viewport by capture and
batch; KV v1 snapshots preserve display, Find, selection, zoom and head panels
by capture and Turn. Each feature validates saved identifiers against current
captured metadata before rendering. Explicit bookmarks and Token-to-KV jumps are
navigation actions and take precedence over ordinary restoration.

`ConversationReadingController` owns rendered reading positions and virtualizer
cleanup. `KvDataController` owns cancellation, loading/error states and identity
validation for metadata, range, selection and Find requests. View snapshots
store UI choices, not tensor values or comparison results. Missing evidence
remains unavailable.

`npm test` discovers all `scripts/check_*.mjs` regressions and feature
`*.test.mjs`/`*.test.cjs` suites. `npm run test:e2e` runs the Playwright smoke
in `e2e/`: it starts the Python Server on `examples/gemma4-e2b` with `dist/`, so
build first and run `npm run e2e:browsers` once; `SERVER_PYTHON` overrides the
interpreter. `npm run test:graph` runs the Graph subset.

The execution iframe and its dependencies use content versions. After editing
`public/graph-execution/`, run `node scripts/version_graph_execution.mjs`; `npm
run build` also does this in its prebuild step. Tests reject a stale version
chain. When publishing, copy nested iframe HTML with the other assets and
publish only the root application `index.html` last. Keep
`public/graph-execution/upstream/model-explorer/dist/`: it contains the pinned
renderer runtime and is included in Git despite the general build-cache
exclusions.

## Fonts and third-party notices

`public/fonts/` ships only the Material Icons font (Apache-2.0); text uses
Roboto when the system provides it, otherwise the platform system font. No
external font requests are made. `npm run build` appends the licenses of the
lazy-loaded Plotly.js asset and everything served from `public/` (the modified
Model Explorer visualizer with its provenance, Material Icons) to
`3rdpartylicenses.txt`, which the About dialog links to.

## Theme

`src/theme/_tokens.scss` writes every token once with `light-dark()`; the only
switch is the `data-theme` attribute that `ThemeService` sets, which selects
`color-scheme` there. Material's dark mode re-emits only its color layer. Global
stylesheets live under `src/theme/` and `src/shared/styles/`; feature
directories hold component styles only. The heat palette is a set of root tokens
that the Token Diff mixins and the trend chart both read.

## Tests

`npm test` runs the esbuild-bundled node checks (`scripts/check_*.mjs`) and the
`*.test.mjs` files next to the code. Scripts that exercise services import real
Angular through `scripts/lib/angular_harness.mjs` (signals, effects, dependency
injection, a flushable effect scheduler) and replace only browser transport and
time. `npm run test:unit` (`ng test`, vitest + jsdom) renders components with
TestBed; `src/test_providers.ts` supplies the providers every spec gets. `npm
run verify` runs all of it plus the build; `npm run verify:e2e` adds the
Playwright smoke against the served production build.
