# Model Debugger UI

The web UI of the Model Debugger: Angular, TypeScript and SCSS, with Plotly
for the charts. The Python server (`src/server/README.md`) serves it and its
`/api`.

## Run

Node 26 (`.nvmrc`). Start the server on port 8080, then, here:

```shell
npm ci
# http://127.0.0.1:4201/, /api/** proxied to 127.0.0.1:8080
# (src/proxy.conf.json)
npm start
npm run build
```

`npm run build` writes `dist/model_debugger/browser/`, which the server
serves by default, and appends the licenses of everything served from
`public/` to its `3rdpartylicenses.txt`, which the About dialog links to.

## Layout

- `src/server/`: The only door to the Python server: `ServerClient`, one
  request file per `/api` topic, the stores.
- `src/server/contracts/`: The wire types, by `/api` topic.
- `src/state/`: Where the page is and what is attached there
  (`WorkspaceStateService`), page addresses, drafts, preferences.
- `src/theme/`: Tokens, the Material theme, typography, global stylesheets
  and partials, `ThemeService`.
- `src/widgets/`: Controls and presentation with several callers; `charts/`
  holds the Plotly charts and their models.
- `src/analyses/`: One folder per analysis (`kv`, `final_logits`, `sampling`,
  `intermediate_tensors`), `shared/`, `registry.ts`, `panel_state.ts`.
- `src/home/`: The home page: Session list, editor, configuration, Runner
  options, the model, HuggingFace and tap pickers.
- `src/conversation/`: The conversation page; what its parts share sits in
  the folder itself.
- `src/conversation/conversation_panel/`: The page that puts the parts
  together.
- `src/conversation/chat/`: The Chat reading.
- `src/conversation/debug/`: The Debug reading: Token Diff, stage analysis
  rails, details panel, find, saved views.
- `src/conversation/composer/`: The message box, Send and Stop, the Debug
  switch.
- `src/app/`: The top: the shell, routes, Session page, toolbar, settings,
  About.
- `scripts/`: The rule check, the structure-graph prototype's checks and
  preview builder, the license step.
- `public/`: Served as is: fonts, Plotly, the Model Explorer visualizer files
  and their licenses.

Component files are `.ts`, `.ng.html` and `.scss`, as in Model Explorer. A
module with one caller lives beside it, not in `widgets/`. Components in
`widgets/` are named for what they are, with no page or analysis prefix, and
a selector is its class name in kebab case (`RangeChart` is `range-chart`).

In `widgets/charts/`, `channel-chart` and `scatter-chart` either reduce the
rows they are given (KV, intermediate tensors) or draw what the server
reduced (bins, view); `waiting` keeps the last drawing until the next answer
arrives (Final Logits Inspection).

## Rules

`scripts/check_layering.mjs` checks these, and `ci/check_ui.sh` fails when
one breaks:

- Layers, lowest first: `server/`, `widgets/`, `analyses/`, the pages `home/`
  and `conversation/`, then `app/`. A layer imports only from layers below
  it.
- Only `src/server/` sends requests: no other file uses `fetch`,
  `EventSource` or `XMLHttpRequest`.
- `widgets/` imports no `state/`, analysis, page or `app/`.
- `home/` and `conversation/` import nothing of each other.
- No module outside `app/` but `main.ts` imports `app/`.
- `state/` imports only `server/`; `widgets/` and `analyses/` import none of
  it.
- `theme/` imports only `state/` (`ThemeService` keeps its choice with
  `PreferencesService`). `widgets/` and `analyses/` use only its stylesheets;
  `server/` and `state/` none of it.
- No analysis imports another. What several share is in `analyses/shared/`,
  which imports none of them, or in `widgets/`.
- The pages and `app/` reach `analyses/` only through `analyses/registry.ts`
  or `analyses/panel_state.ts`, and what `panel_state.ts` runs declares no
  component or directive.
- In the conversation page, `chat/`, `debug/` and `composer/` import nothing
  of each other, and `conversation/` itself imports none of them nor
  `conversation_panel/`.
- A details body (`analyses/<name>/<name>_details.ts`) and `analyses/shared/`
  import no chart, directly or through the modules of `analyses/` they
  import. A chart is `widgets/plotly_loader.ts` or a module of
  `widgets/charts/` that declares a component or directive or imports a
  chart.
- A CSS custom property that `server/`, `widgets/`, `analyses/` or a low
  module (`LOW_MODULES` in the check) reads without a fallback is set on the
  root element in `theme/`, or at or below the reader: an analysis reads no
  property only the page around it sets.
- Every folder and file directly under `src/` is in the check's `LAYOUT`; a
  new one fails until it is given its place and its rules.

An import is any reference the check can read: tests, `.mjs` and `.cjs`
modules, side-effect imports, `import(...)` with a literal specifier (one
built at run time fails the check), `require(...)`,
`new URL(..., import.meta.url)`, `styleUrl`, `templateUrl` and `styleUrls`,
and Sass's `@use`, `@forward` and `@import`.

## Checks and tests

From the repository root, `ci/check_ui.sh` runs, in order, each step failing
the check:

1. `npm ci`, when `node_modules/` is missing;
2. `npm run format:check`: Prettier over everything it formats here, all but
   the generated and third-party files `.prettierignore` names
   (`npm run format` writes it);
3. `npm run check:layering`: the rules above;
4. `npm run build`;
5. `npm test`: the unit tests, `ng test` (vitest and jsdom): every
   `*.spec.ts` under `src/`, each beside the file it tests, with the
   providers of `src/test_providers.ts`; specs of services and components
   build them with `TestBed`, the server and the browser's transport replaced
   by fakes (`src/server/testing/` holds the shared ones);
6. the structure-graph prototype's checks, each run with `node` on its own:
   `scripts/check_graph_pairs.mjs`, `check_mapping_pair_edit.mjs` and
   `check_model_tree_metrics.mjs`. They test the plain scripts of
   `public/graph-arch-preview/`, which `ng test` does not build.

`npm test -- --include <path>` runs the specs of one folder or one file, such
as `npm test -- --include src/analyses/kv/` or
`npm test -- --include src/conversation/token_query.spec.ts`.
`npm run verify` runs `ci/check_ui.sh`.

`src/server/testing/metric_catalog_fixture.json`, the catalog the specs use,
is the server's answer to GET /api/metrics: `ci/check_server.sh` checks that
it is, and, from the repository root, running
`python3 -I ci/check_metric_catalog.py --write` with
`src/ui/src/server/testing/metric_catalog_fixture.json` writes it.

## How the UI works

- **The server decides.** Metrics, their labels, formats and thresholds come
  from GET /api/metrics (`MetricCatalogStore`); comparisons, alignments, what
  each stage offers and every reason (`{code, side, text}`) come from GET
  /api/session. The UI aligns, measures and decides nothing of its own; an
  analysis request names rows of the server's comparison. Which step a column
  of an analysis is, each token of the Session says in the server's words:
  `produced_text` for Final Logits and the intermediate tensors,
  `written_text` for the KV cache.
- **The address is the navigation state.** `/` is home, `/sessions/{id}` a
  Session's live chat (else the New chat form, or its latest chat with a
  turn), `.../new` the New chat form, `.../chats/{c}` chat c, `.../debug`
  Debug mode, `.../debug/turns/{n}` a turn and
  `.../debug/turns/{n}/{stage}/{analysis}` an analysis open. A Session holds
  several chats, one live at a time; each has its own capture, which every
  capture-scoped request names (`session_id`, `chat`;
  `server/capture_id.ts`). `state/page_address/page_address.ts` reads and
  writes the addresses, `app/routes.ts` maps them to pages, and navigation
  keeps the query (`?debugPalette`). Which modes a chat is read in is the
  server's (its `debug` action): every chat is read in Chat, one with a turn
  kept in Debug too; `WorkspaceStateService` replaces the address of a mode
  the server refuses, however the page arrived there.
- **Theme.** `src/theme/_tokens.scss` writes every token once with
  `light-dark()`; `ThemeService` sets `data-theme`, which picks the scheme.
  The charts read the `--chart-*` tokens and the heat palettes
  (`--heat-semantic-*`, `--heat-activation-*`).
- **Fonts.** `public/fonts/` ships Material Icons; the UI makes no external
  font requests.
