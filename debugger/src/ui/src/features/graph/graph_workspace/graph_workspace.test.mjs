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

import {build} from 'esbuild';
import assert from 'node:assert/strict';
import {test} from 'node:test';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const names = {
  report_state_service: 'ReportStateService',
  architecture_view: 'ArchitectureView',
  execution_graph: 'ExecutionGraph',
  graph_inspection_service: 'GraphInspectionService',
  graph_details: 'GraphDetails',
  explicit_pairs: 'ExplicitPairs',
  graph_view_store: 'GraphViewStore',
  resize_handle: 'ResizeHandle',
  find_navigator: 'FindNavigator',
};
const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('./graph_workspace.ts', import.meta.url))],
  bundle: true,
  write: false,
  platform: 'node',
  format: 'cjs',
  plugins: [
    {
      name: 'workspace-lifecycle-dependencies',
      setup(builder) {
        builder.onResolve({filter: /.*/}, ({path}) => {
          const name = path.split('/').at(-1);
          if (path.startsWith('@angular/') || names[name] || name === 'plotly_loader')
            return {path: name, namespace: 'stub'};
        });
        builder.onLoad({filter: /.*/, namespace: 'stub'}, ({path}) => ({
          contents:
            path === 'core'
              ? `
        export const Component=()=>value=>value, HostListener=()=>()=>{};
        export const ChangeDetectionStrategy={OnPush:0};
        export class DestroyRef {} export class ElementRef {} export class Injector {}
        export const inject=token=>globalThis.dependencies[token.name];
        export const computed=read=>read, untracked=read=>read();
        export const effect=read=>globalThis.effects.push(read);
        export const afterNextRender=read=>globalThis.renders.push(read);
        export const viewChild=()=>()=>undefined;
        export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}
      `
              : path === 'forms'
                ? 'export class FormsModule {}'
                : path === 'common'
                  ? 'export class TitleCasePipe {}'
                  : path === 'icon'
                    ? 'export class MatIconModule {}'
                    : path === 'plotly_loader'
                      ? 'export const loadTokenPlotly=async()=>{};'
                      : `export class ${names[path]} {}`,
        }));
      },
    },
  ],
});
function signal(value) {
  const read = () => value;
  read.set = (next) => (value = next);
  read.update = (fn) => (value = fn(value));
  return read;
}
const model = {
  layers: [{def: 0}, {def: 0}],
  semantic_graph: [{nodes: [], anchors: [{id: 'output'}]}],
};
function fixture(initial = null) {
  const effects = [],
    renders = [],
    destroys = [],
    values = new Map(),
    requests = [];
  const key = (capture, batch) => JSON.stringify([capture, batch]);
  if (initial) values.set(key('a', 0), structuredClone(initial));
  const views = {
    read: (capture, batch) => structuredClone(values.get(key(capture, batch)) ?? null),
    save: (capture, batch, view) => values.set(key(capture, batch), structuredClone(view)),
  };
  const state = {
    captureId: signal('a'),
    batchId: signal(0),
    layer: signal(0),
    metric: signal('CosSim'),
    metrics: ['CosSim', 'RMSE'],
    semantic: () => model,
    comparison: signal(null),
    dark: signal(false),
    selectBatch(value) {
      this.batchId.set(value);
    },
    selectLayer(value) {
      this.layer.set(value);
    },
  };
  const inspection = {
    context: signal(null),
    mapping: signal(false),
    select(context) {
      this.context.set({...context});
      requests.push({...context});
    },
    close() {
      this.context.set(null);
    },
  };
  const module = {exports: {}};
  vm.runInNewContext(outputFiles[0].text, {
    module,
    exports: module.exports,
    effects,
    renders,
    window: {innerWidth: 1000},
    ResizeObserver: class {
      observe() {}
      disconnect() {}
    },
    dependencies: {
      ReportStateService: state,
      GraphInspectionService: inspection,
      GraphViewStore: views,
      ElementRef: {nativeElement: {querySelector: () => null, clientWidth: 1000}},
      Injector: {},
      DestroyRef: {onDestroy: (callback) => destroys.push(callback)},
    },
  });
  const workspace = new module.exports.GraphWorkspace();
  const render = () => {
    for (const callback of renders.splice(0)) callback();
  };
  effects[0]();
  effects[1]();
  render();
  const changeBatch = (batch) => {
    state.batchId.set(batch);
    inspection.close();
    effects[0]();
    effects[1]();
    render();
  };
  return {
    workspace,
    state,
    inspection,
    values,
    requests,
    effects,
    renders,
    render,
    key,
    changeBatch,
    destroy: () => {
      inspection.close();
      for (const callback of destroys) callback();
    },
  };
}
const snapshot = () => ({
  version: 1,
  query: {
    text: 'saved',
    anchor: '',
    metric: 'CosSim',
    operator: 'lt',
    threshold: '',
    withMetrics: false,
  },
  metric: 'CosSim',
  details: true,
  results: true,
  trends: true,
  detailsWidth: 350,
  trendHeight: 140,
  context: {layer: 0, semantic: 'anchor:output'},
  execution: true,
  viewport: {layer: 0, zoom: 1.2, left: 30, top: 200},
});

test('in-place batch changes save outgoing state and restore independent query, selection and viewport', () => {
  const f = fixture();
  f.workspace.query.set({...f.workspace.query(), text: 'batch A'});
  f.workspace.viewport = {layer: 0, zoom: 1.2, left: 30, top: 200};
  f.inspection.select({layer: 0, batch: 0, semantic: 'anchor:output'});
  f.effects[0]();
  f.workspace.executionOpen.set(true);
  f.changeBatch(1);
  assert.equal(f.values.get(f.key('a', 0)).execution, true);
  assert.equal(f.values.get(f.key('a', 0)).query.text, 'batch A');
  assert.equal(f.workspace.executionOpen(), false);
  assert.equal(f.workspace.restoredViewport().top, 0);
  f.workspace.query.set({...f.workspace.query(), text: 'batch B'});
  f.workspace.viewport = {layer: 0, zoom: 0.8, left: 4, top: 400};
  f.changeBatch(0);
  assert.equal(f.workspace.query().text, 'batch A');
  assert.equal(f.workspace.viewport.top, 200);
  assert.equal(f.workspace.executionOpen(), true);
  assert.equal(f.inspection.context().batch, 0);
  f.changeBatch(1);
  assert.equal(f.workspace.query().text, 'batch B');
  assert.equal(f.workspace.viewport.top, 400);
  assert.equal(f.workspace.executionOpen(), false);
});

test('provider close before destruction retains reading context without caching capture data', () => {
  const f = fixture(snapshot());
  f.destroy();
  const saved = f.values.get(f.key('a', 0));
  assert.equal(saved.context.semantic, 'anchor:output');
  assert.equal(saved.execution, true);
  assert.equal(saved.viewport.top, 200);
  assert.equal('detailsPayload' in saved, false);
});

test('capture replacement saves the outgoing identity once and cannot overwrite it after reset', () => {
  const f = fixture(snapshot());
  f.state.captureId.set('b');
  f.state.batchId.set(0);
  f.inspection.close();
  f.effects[0]();
  f.effects[1]();
  f.workspace.query.set({...f.workspace.query(), text: 'new capture reset'});
  f.destroy();
  assert.equal(f.values.get(f.key('a', 0)).query.text, 'saved');
  assert.equal(f.values.get(f.key('a', 0)).execution, true);
  assert.equal(f.values.has(f.key('b', 0)), false);
});

test('bookmark navigation wins over an incoming ordinary snapshot and preserves the outgoing metric', () => {
  const f = fixture(snapshot());
  f.values.set(f.key('a', 1), {
    ...snapshot(),
    metric: 'CosSim',
    query: {...snapshot().query, text: 'ordinary B'},
  });
  const bookmark = {
    capture: 'a',
    batch: 1,
    layer: 0,
    semantic: 'anchor:output',
    metric: 'RMSE',
    execution: true,
  };
  f.workspace.restoreBookmark(bookmark);
  f.inspection.close();
  f.effects[0]();
  f.effects[1]();
  f.render();
  f.effects[0]();
  assert.equal(f.values.get(f.key('a', 0)).metric, 'CosSim');
  assert.equal(f.state.metric(), 'RMSE');
  assert.equal(f.workspace.executionOpen(), true);
  assert.equal(f.inspection.context().batch, 1);
});

test('a queued restore cannot select a node after the capture is replaced', () => {
  const f = fixture();
  f.values.set(f.key('a', 1), snapshot());
  f.state.batchId.set(1);
  f.effects[0]();
  f.state.captureId.set('b');
  f.effects[0]();
  f.render();
  assert.equal(f.requests.length, 0);
});
