/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import '@angular/compiler';
import {
  createEnvironmentInjector,
  runInInjectionContext,
  signal,
  ElementRef,
  ɵEffectScheduler as EffectScheduler,
  ɵChangeDetectionScheduler as ChangeDetectionScheduler,
} from '@angular/core';
import {strict as assert} from 'node:assert';
import {build} from 'esbuild';
import {mkdir, mkdtemp, readFile, rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root = fileURLToPath(new URL('..', import.meta.url));
execFileSync(
  process.execPath,
  [path.join(root, 'scripts/version_graph_execution.mjs'), '--check'],
  {stdio: 'inherit'},
);
const cache = path.join(root, 'node_modules/.cache');
await mkdir(cache, {recursive: true});
const temp = await mkdtemp(path.join(cache, 'graph-presentation-'));
const originals = {
  window: globalThis.window,
  ResizeObserver: globalThis.ResizeObserver,
  location: globalThis.location,
};
const resizeCallbacks = [];
globalThis.window = {innerWidth: 1200, addEventListener() {}, removeEventListener() {}};
globalThis.location = {origin: 'http://test'};
globalThis.ResizeObserver = class {
  constructor(callback) {
    resizeCallbacks.push(callback);
  }
  observe() {}
  disconnect() {}
};
globalThis.__graphRenderHooks = [];
let injector;
try {
  const bundle = path.join(temp, 'components.mjs');
  await build({
    stdin: {
      contents: `export {GraphWorkspace} from './src/features/graph/graph_workspace/graph_workspace'; export {ExecutionGraph} from './src/features/graph/execution_graph/execution_graph'; export {ReportStateService} from './src/data/report_state_service'; export {GraphInspectionService} from './src/features/graph/graph_inspection_service'; export {GraphViewStore} from './src/features/graph/graph_workspace/graph_view_store'; export {PreferencesService} from './src/app/preferences_service'; export {WorkspaceStateService} from './src/app/workspace_state_service';`,
      resolveDir: root,
      loader: 'ts',
    },
    bundle: true,
    packages: 'external',
    platform: 'node',
    format: 'esm',
    outfile: bundle,
    logLevel: 'silent',
    plugins: [
      {
        name: 'render-hook',
        setup(builder) {
          builder.onResolve({filter: /\/workspace_state_service$/}, () => ({
            path: 'navigation',
            namespace: 'navigation-stub',
          }));
          builder.onLoad({filter: /.*/, namespace: 'navigation-stub'}, () => ({
            contents: 'export class WorkspaceStateService {}',
          }));
          builder.onResolve({filter: /^@angular\/core$/}, (args) =>
            args.namespace === 'render-hook'
              ? {path: args.path, external: true}
              : {path: 'core', namespace: 'render-hook'},
          );
          builder.onLoad({filter: /.*/, namespace: 'render-hook'}, () => ({
            contents: `export * from '@angular/core'; export const afterNextRender = callback => {globalThis.__graphRenderHooks.push(callback); return {destroy(){}};};`,
          }));
        },
      },
    ],
  });
  const {
    GraphWorkspace,
    ExecutionGraph,
    ReportStateService,
    GraphInspectionService,
    GraphViewStore,
    PreferencesService,
    WorkspaceStateService,
  } = await import(pathToFileURL(bundle));
  const host = {
    clientWidth: 1200,
    clientHeight: 720,
    querySelector() {
      return {};
    },
  };
  const state = {
    captureId: signal('capture'),
    batchId: signal(11),
    layer: signal(0),
    metric: signal('CosSim'),
    semantic: signal({semantic_graph: [{nodes: [], anchors: []}], layers: [{def: 0}]}),
    comparison: signal(null),
    dark: signal(false),
    metrics: ['CosSim'],
  };
  const scheduler = EffectScheduler.ɵprov.factory();
  injector = createEnvironmentInjector([
    {provide: ReportStateService, useValue: state},
    {provide: WorkspaceStateService, useValue: {graphTarget: signal(null)}},
    {provide: GraphInspectionService, useValue: {context: signal(null)}},
    {provide: GraphViewStore, useValue: {read: () => null, save() {}}},
    {provide: PreferencesService, useValue: {read: () => null}},
    {provide: ElementRef, useValue: {nativeElement: host}},
    {provide: EffectScheduler, useValue: scheduler},
    {provide: ChangeDetectionScheduler, useValue: {notify() {}}},
  ]);
  const workspace = runInInjectionContext(injector, () => new GraphWorkspace());
  scheduler.flush();
  for (const callback of globalThis.__graphRenderHooks.splice(0)) callback();
  assert.equal(resizeCallbacks.length, 1);
  workspace.detailsOpen.set(true);
  workspace.detailsWidth.set(560);
  globalThis.window.innerWidth = 820;
  host.clientWidth = 820;
  resizeCallbacks[0]();
  assert.equal(
    workspace.detailsWidth(),
    480,
    'Details leaves room for the graph on smaller desktops',
  );
  assert.equal(workspace.detailsOpen(), true);
  globalThis.window.innerWidth = 720;
  host.clientWidth = 720;
  resizeCallbacks[0]();
  assert.equal(workspace.detailsOpen(), true, 'narrow layouts preserve the selected Details state');
  workspace.toggleDetails();
  assert.equal(workspace.detailsOpen(), false);
  workspace.toggleDetails();
  assert.equal(workspace.detailsOpen(), true);
  globalThis.window.innerWidth = 1200;
  host.clientWidth = 1200;
  resizeCallbacks[0]();
  assert.equal(workspace.detailsWidth(), 480, 'widening preserves the user reading width');
  assert.equal(workspace.detailsOpen(), true);
  console.log(
    'PASS: actual GraphWorkspace resize preserves graph room and Details visibility across layouts',
  );

  const messages = [],
    dark = signal(false);
  const execution = runInInjectionContext(injector, () => new ExecutionGraph());
  execution.dark = dark;
  execution.frame = () => ({
    nativeElement: {contentWindow: {postMessage: (message) => messages.push(message)}},
  });
  scheduler.flush();
  assert.equal(messages.at(-1).payload.theme, 'light');
  dark.set(true);
  scheduler.flush();
  assert.equal(messages.at(-1).payload.theme, 'dark');
  execution.send();
  assert.equal(messages.at(-1).payload.theme, 'dark');
  console.log('PASS: actual ExecutionGraph effect and frame-load send propagate the current theme');

  const bundlePath = path.join(
    root,
    'public/graph_execution/upstream/model_explorer/dist/main_browser.js',
  );
  const source = await readFile(bundlePath, 'utf8');
  const helper = source.match(
    /  function mePaneBackgroundSet\(target, background\) \{[\s\S]*?(?=\n  function meViewportRead)/,
  )?.[0];
  assert.ok(helper);
  const context = {};
  vm.createContext(context);
  vm.runInContext(helper + ';globalThis.paint=mePaneBackgroundSet;', context);
  const calls = [],
    camera = {zoom: 3};
  const target = {
    service: {
      camera,
      scene: {background: {set: (color) => calls.push(color)}},
      render: () => calls.push('render'),
    },
  };
  assert.equal(context.paint(target, '#202124'), true);
  assert.deepEqual(calls, ['#202124', 'render']);
  assert.equal(target.service.camera, camera);
  assert.equal(context.paint(null, '#202124'), false);
  assert.equal(context.paint(target, 'bad-color'), false);
  assert.equal(context.paint({service: {scene: {background: null}}}, '#ffffff'), false);
  const provenance = JSON.parse(
    await readFile(path.join(root, 'public/graph_execution/upstream/model_explorer/PROVENANCE.json')),
  );
  assert.equal(
    provenance.sha256['dist/main_browser.js'],
    createHash('sha256').update(source).digest('hex'),
  );
  console.log(
    'PASS: pinned background bridge only repaints the scene, rejects unavailable panes, and matches provenance hash',
  );
} finally {
  injector?.destroy();
  Object.assign(globalThis, originals);
  delete globalThis.__graphRenderHooks;
  await rm(temp, {recursive: true, force: true});
}
