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

import {createEnvironmentInjector, runInInjectionContext, signal, ɵChangeDetectionScheduler as ChangeDetectionScheduler, ɵEffectScheduler as EffectScheduler,} from '@angular/core';
import {build} from 'esbuild';
import {strict as assert} from 'node:assert';
import {mkdir, mkdtemp, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const cache = path.join(root, 'node_modules/.cache');
await mkdir(cache, {recursive: true});
const temp = await mkdtemp(path.join(cache, 'graph-loading-'));
const pending = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
};
const makeSession = (model = 'Current', batches) => ({
  model,
  runs: [],
  turns: [
    {n: 1, prefill_tokens: 8, decode_tokens: 2},
    {n: 2, prefill_tokens: null, decode_tokens: null},
  ],
  phases: [{id: 'prefill'}, {id: 'decode'}],
  batches: batches ??
      [
        {batch: 11, turn: 1, phase: 'prefill', index: 0, step: null},
        {batch: 21, turn: 1, phase: 'decode', index: 0, step: 0},
        {batch: 22, turn: 1, phase: 'decode', index: 1, step: 1},
        {batch: 31, turn: 2, phase: 'prefill', index: 0, step: null},
        {batch: 41, turn: 2, phase: 'decode', index: 0, step: 0},
      ],
  layers: [{index: 0, label: 'Recorded layer', hidden: null}],
  anchors: [],
  conversation: [{turn: 1, run: 'ref', output: 'Saved answer'}],
});
const makeSemantic = (dimension = '256') => ({
  semantic_graph: [
    {
      kind: 'decoder',
      inputs: [{id: 'input', shape: ['tokens', dimension]}],
      nodes: [],
      anchors: [],
    },
  ],
  layers: [
    {def: 0, attrs: {attn: {ops: {attend: {span: {kind: 'local'}}}}}},
    {def: 0, attrs: {attn: {ops: {k_proj: {skip: true}}}}},
  ],
});
const makeComparison = (batch) => ({batch, rows: [], layers: []});
const makeOverview = () => ({
  batches: [
    {batch: 11, metrics: {CosSim: {value: 0.9}, RMSE: {value: 0.1}}},
    {batch: 21, metrics: {CosSim: {value: 0.5}, RMSE: {value: 0.4}}},
    {batch: 22, metrics: {CosSim: {value: null}, RMSE: {value: null}}},
    {batch: 31, metrics: {CosSim: {value: 0.8}, RMSE: {value: 0.2}}},
    {batch: 41, metrics: {CosSim: {value: 0.7}, RMSE: {value: 0.3}}},
  ],
});
function freeze(value) {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

try {
  const bundle = path.join(temp, 'state.mjs');
  await build({
    stdin: {
      contents:
          `export {ReportStateService} from './src/data/report_state_service'; export {ReportApiService} from './src/data/report_api_service'; export {ThemeService} from './src/theme/theme_service';`,
      resolveDir: root,
      loader: 'ts',
    },
    bundle: true,
    packages: 'external',
    platform: 'node',
    format: 'esm',
    outfile: bundle,
    logLevel: 'silent',
  });
  const {ReportStateService, ReportApiService, ThemeService} =
      await import(pathToFileURL(bundle));
  async function scenario(name, check) {
    const calls = [];
    const api = Object.fromEntries(
        ['session', 'semantic', 'overview', 'comparison'].map(
            (kind) =>
                [kind,
                 (...args) => {
                   const response = pending();
                   const request = {
                     kind,
                     capture: args[0],
                     batch: kind === 'comparison' ? args[1] : null,
                     signal: args.at(-1),
                     ...response,
                   };
                   assert.ok(
                       request.signal instanceof AbortSignal,
                       `${kind} forwards AbortSignal`);
                   calls.push(request);
                   // This transport can resolve after abort: response guards
                   // must independently reject it.
                   return response.promise;
                 },
    ]),
    );
    // Run Angular's real root effect scheduler and DestroyRef. No signal/effect
    // mocks. A browser change-detection notifier is unnecessary for these
    // service-level assertions.
    const scheduler = EffectScheduler.ɵprov.factory();
    const injector = createEnvironmentInjector([
      {provide: ReportApiService, useValue: api},
      {provide: ThemeService, useValue: {dark: signal(false)}},
      {provide: EffectScheduler, useValue: scheduler},
      {provide: ChangeDetectionScheduler, useValue: {notify() {}}},
    ]);
    const state =
        runInInjectionContext(injector, () => new ReportStateService());
    const flush = async () => {
      for (let i = 0; i < 8; i++) {
        await Promise.resolve();
        scheduler.flush();
      }
    };
    const requests = (kind) => calls.filter((request) => request.kind === kind);
    const last = (kind) => requests(kind).at(-1);
    const load = async (capture = 'capture-a', response = makeSession()) => {
      state.attach(capture);
      const loading = state.load();
      last('session').resolve(response);
      await loading;
      await flush();
    };
    const graphReady = async () => {
      state.setGraphActive(true);
      last('semantic').resolve(makeSemantic());
      await flush();
      last('overview').resolve(makeOverview());
      last('comparison').resolve(makeComparison(state.batchId()));
      await flush();
    };
    try {
      await check(
          {state, calls, requests, last, load, graphReady, flush, injector});
      console.log('PASS: ' + name);
    } finally {
      if (!injector.destroyed) injector.destroy();
    }
  }

  await scenario(
      'Chat publishes raw Session independently; Graph failure/retry and immutable projection',
      async (h) => {
        const raw = freeze(makeSession()), serialized = JSON.stringify(raw);
        await h.load('capture-a', raw);
        h.state.selectPhase('decode');
        h.state.selectBatch(22);
        h.state.refreshMetrics();
        await h.flush();
        assert.deepEqual(
            h.calls.map((r) => r.kind),
            ['session'],
        );
        assert.equal(h.state.session(), raw);
        assert.equal(h.state.loading(), false);
        h.state.setGraphActive(true);
        h.state.setGraphActive(true);
        assert.equal(h.requests('semantic').length, 1);
        assert.equal(h.state.graphLoading(), true);
        h.last('semantic').reject(new Error('semantic offline'));
        await h.flush();
        assert.match(h.state.graphError(), /offline/);
        assert.equal(h.state.error(), '');
        assert.equal(h.state.session().conversation[0].output, 'Saved answer');
        assert.equal(h.state.graphLoading(), false);
        assert.equal(h.requests('comparison').length, 0);
        h.state.retryGraph();
        h.last('semantic').resolve(makeSemantic('hidden_size'));
        await h.flush();
        assert.equal(h.state.graphError(), '');
        assert.equal(h.state.session(), raw);
        assert.equal(
            JSON.stringify(raw), serialized,
            'Graph never mutates API Session evidence');
        assert.equal(h.state.preview.layers[0].hidden, null);
        assert.equal(h.state.preview.layers[1].label, 'decoder · shared KV');
        assert.equal(
            h.state.batchId(), 22,
            'Graph arrival does not reset Chat-selected batch');
        assert.equal(h.last('comparison').batch, 22);
      },
  );

  await scenario(
      'leaving Graph aborts all Graph work; stale results ignored; reentry resumes and caches',
      async (h) => {
        await h.load();
        h.state.setGraphActive(true);
        const oldSemantic = h.last('semantic');
        h.state.setGraphActive(false);
        assert.equal(oldSemantic.signal.aborted, true);
        h.state.setGraphActive(true);
        oldSemantic.resolve(makeSemantic('999'));
        await h.flush();
        assert.equal(h.state.semantic(), null);
        h.last('semantic').resolve(makeSemantic());
        await h.flush();
        const oldOverview = h.last('overview'),
              oldComparison = h.last('comparison');
        h.state.setGraphActive(false);
        assert.equal(oldOverview.signal.aborted, true);
        assert.equal(oldComparison.signal.aborted, true);
        oldOverview.resolve(makeOverview());
        oldComparison.resolve(makeComparison(11));
        await h.flush();
        assert.equal(h.state.overview(), null);
        assert.equal(h.state.comparison(), null);
        h.state.setGraphActive(true);
        await h.flush();
        assert.equal(
            h.requests('semantic').length, 2,
            'completed semantic cache remains valid');
        assert.notEqual(h.last('overview'), oldOverview);
        assert.notEqual(h.last('comparison'), oldComparison);
        h.last('overview').resolve(makeOverview());
        h.last('comparison').resolve(makeComparison(11));
        await h.flush();
        const count = h.calls.length;
        h.state.setGraphActive(false);
        await h.flush();
        h.state.setGraphActive(true);
        await h.flush();
        assert.equal(
            h.calls.length, count,
            'complete Graph cache avoids duplicate fetches');
        h.state.setGraphActive(false);
        h.state.selectPhase('decode');
        h.state.selectBatch(22);
        await h.flush();
        assert.equal(
            h.calls.length, count, 'Chat selection does not fetch Graph data');
        const semanticCount = h.requests('semantic').length;
        const overviewCount = h.requests('overview').length;
        h.state.setGraphActive(true);
        await h.flush();
        assert.equal(
            h.last('comparison').batch, 22,
            'returning Graph uses the current Chat batch');
        assert.equal(h.requests('semantic').length, semanticCount);
        assert.equal(h.requests('overview').length, overviewCount);
      },
  );

  await scenario(
      'capture switch and repeated retry reject stale session/Graph responses; destroy aborts',
      async (h) => {
        h.state.attach('old');
        const oldLoad = h.state.load(), oldSession = h.last('session');
        await h.load('new', makeSession('New'));
        assert.equal(oldSession.signal.aborted, true);
        oldSession.resolve(makeSession('Old'));
        await oldLoad;
        assert.equal(h.state.session().model, 'New');
        h.state.setGraphActive(true);
        const oldSemantic = h.last('semantic');
        await h.load('third', makeSession('Third'));
        assert.equal(oldSemantic.signal.aborted, true);
        oldSemantic.reject(new Error('late semantic failure'));
        h.last('semantic').resolve(makeSemantic());
        await h.flush();
        const oldOverview = h.last('overview'),
              oldComparison = h.last('comparison');
        await h.load('fourth', makeSession('Fourth'));
        assert.equal(oldOverview.signal.aborted, true);
        assert.equal(oldComparison.signal.aborted, true);
        oldOverview.resolve(makeOverview());
        oldComparison.reject(new Error('late comparison failure'));
        await h.flush();
        assert.equal(h.state.overview(), null);
        assert.equal(h.state.comparisonError(), '');
        h.state.retryGraph();
        const retried = h.last('semantic');
        h.state.retryGraph();
        assert.equal(retried.signal.aborted, true);
        retried.resolve(makeSemantic('999'));
        h.last('semantic').resolve(makeSemantic());
        await h.flush();
        assert.equal(h.state.preview.layers[0].hidden, 256);
        assert.equal(h.state.session().model, 'Fourth');
        h.injector.destroy();
        assert.equal(h.last('overview').signal.aborted, true);
        assert.equal(h.last('comparison').signal.aborted, true);
      },
  );

  await scenario(
      'same-capture reload preserves valid batch/layer; invalid selection falls back within capture',
      async (h) => {
        await h.load();
        await h.graphReady();
        h.state.selectTurn(2);
        h.state.selectPhase('decode');
        h.state.selectLayer(1);
        await h.flush();
        const oldComparison = h.last('comparison');
        const reload = h.state.load();
        assert.equal(oldComparison.signal.aborted, true);
        const interrupted = h.last('session');
        const latest = h.state.load();
        assert.equal(interrupted.signal.aborted, true);
        h.last('session').resolve(makeSession('Refreshed'));
        await latest;
        interrupted.resolve(makeSession('Stale refresh'));
        await reload;
        h.last('semantic').resolve(makeSemantic());
        await h.flush();
        assert.equal(h.state.session().model, 'Refreshed');
        assert.equal(h.state.batchId(), 41);
        assert.equal(h.state.layer(), 1);
        assert.equal(h.state.batch().turn, 2);
        assert.equal(h.state.batch().phase, 'decode');
        assert.equal(h.last('comparison').batch, 41);
        const missing = h.state.load();
        h.last('session').resolve(
            makeSession('Reduced', [makeSession().batches[0]]));
        await missing;
        assert.equal(h.state.batchId(), 11);
        h.last('semantic')
            .resolve({...makeSemantic(), layers: [makeSemantic().layers[0]]});
        await h.flush();
        assert.equal(h.state.layer(), 0);
      },
  );

  await scenario(
      'Turn/Prefill/Decode stay independent; comparisons retry and unknown metrics stay unknown',
      async (h) => {
        await h.load();
        await h.graphReady();
        h.state.metric.set('Mean abs error');
        assert.equal(h.state.format(null), '—');
        h.state.selectPhase('decode');
        await h.flush();
        const old = h.last('comparison');
        assert.equal(old.batch, 21);
        h.state.selectBatch(22);
        await h.flush();
        assert.equal(old.signal.aborted, true);
        assert.equal(h.last('comparison').batch, 22);
        old.resolve(makeComparison(21));
        await h.flush();
        assert.equal(h.state.comparison(), null);
        h.last('comparison').resolve(makeComparison(22));
        await h.flush();
        h.state.selectTurn(2);
        await h.flush();
        assert.equal(h.last('comparison').batch, 41);
        h.state.selectPhase('prefill');
        await h.flush();
        assert.equal(h.last('comparison').batch, 31);
        assert.equal(h.state.batch().step, null);
        h.state.selectBatch(99);
        assert.equal(
            h.state.batchId(), 31,
            'unknown batch ids leave the selection alone');
        h.state.refreshMetrics();
        await h.flush();
        h.last('comparison').reject(new Error('comparison offline'));
        h.last('overview').reject(new Error('overview offline'));
        await h.flush();
        assert.match(h.state.comparisonError(), /offline/);
        assert.match(h.state.overviewError(), /offline/);
        assert.equal(h.state.comparing(), false);
        assert.equal(h.state.error(), '');
        h.state.refreshMetrics();
        await h.flush();
        h.last('overview').resolve(makeOverview());
        h.last('comparison').resolve(makeComparison(31));
        await h.flush();
        assert.equal(h.state.comparisonError(), '');
        assert.equal(h.state.overviewError(), '');
        h.state.setGraphActive(false);
        const count = h.calls.length;
        h.state.refreshMetrics();
        await h.flush();
        assert.equal(h.calls.length, count);
      },
  );

  await scenario(
      'no-data profiles and null capture have safe projections without Graph requests',
      async (h) => {
        await h.load('empty', makeSession('No tensors', []));
        h.state.setGraphActive(true);
        await h.flush();
        assert.equal(h.calls.length, 1);
        await h.load('unprofiled');
        h.last('semantic').resolve({semantic_graph: [], layers: []});
        await h.flush();
        assert.equal(h.state.graphLoading(), false);
        assert.equal(h.requests('overview').length, 0);
        assert.equal(h.requests('comparison').length, 0);
        h.state.attach(null);
        await h.flush();
        assert.equal(h.state.session(), null);
        assert.deepEqual(h.state.preview.layers, []);
        assert.deepEqual(h.state.definition().anchors, []);
        h.state.selectTurn(2);
        h.state.selectPhase('decode');
        h.state.selectBatch(1);
        assert.equal(h.state.batchId(), 0);
      },
  );

  // Test actual API forwarding separately from the mocked service transport.
  const fetchBefore = globalThis.fetch;
  try {
    const calls = [];
    globalThis.fetch = async (url, options) => {
      calls.push({url, options});
      return {ok: true, json: async () => ({})};
    };
    const api = new ReportApiService(), controller = new AbortController();
    await api.session('capture&1', controller.signal);
    await api.semantic('capture&1', controller.signal);
    await api.overview('capture&1', controller.signal);
    assert.deepEqual(
        calls.map((c) => c.url),
        [
          '/api/session?session_id=capture%261',
          '/api/semantic?session_id=capture%261',
          '/api/overview?session_id=capture%261',
        ],
    );
    assert.ok(calls.every((c) => c.options.signal === controller.signal));
    console.log(
        'PASS: session/semantic/overview API preserves capture identity and AbortSignal');
  } finally {
    globalThis.fetch = fetchBefore;
  }
} finally {
  await rm(temp, {recursive: true, force: true});
}
