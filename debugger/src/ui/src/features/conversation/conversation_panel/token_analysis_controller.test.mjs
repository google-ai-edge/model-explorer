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

import assert from 'node:assert/strict';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('./token_analysis_controller.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}`,
        }));
      },
    },
  ],
});
const {TokenAnalysisController, ANALYSIS_CONCURRENCY} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const flush = async () => {
  for (let i = 0; i < 12; i++) await Promise.resolve();
};
function fakeApi() {
  const calls = [];
  return {
    calls,
    inFlight: () => calls.filter((c) => !c.done).length,
    peak: 0,
    tokenAnalysis(captureId, turn, pairs, signal) {
      const call = {captureId, turn, pairs, signal, done: false};
      calls.push(call);
      this.peak = Math.max(this.peak, this.inFlight());
      return new Promise((resolve, reject) => {
        call.resolve = () => {
          call.done = true;
          resolve({
            turn,
            pairs: pairs.map((p) => ({ref_step: p.ref, target_step: p.target, metrics: {}})),
          });
        };
        call.reject = (error) => {
          call.done = true;
          reject(error);
        };
      });
    },
  };
}
const pairs = (n, from = 0) =>
  Array.from({length: n}, (_, i) => ({ref: from + i, target: from + i}));

test('at most four batches are in flight and every pair lands in the results', async () => {
  const api = fakeApi();
  const controller = new TokenAnalysisController(api);
  controller.load('cap', [{turn: 1, pairs: pairs(1280)}], new AbortController().signal);
  await flush();
  assert.equal(api.inFlight(), ANALYSIS_CONCURRENCY);
  assert.equal(controller.loading(), true);
  while (api.calls.some((c) => !c.done)) {
    api.calls.find((c) => !c.done).resolve();
    await flush();
  }
  assert.equal(api.calls.length, 10);
  assert.equal(api.peak, ANALYSIS_CONCURRENCY);
  assert.equal(controller.results().size, 1280);
  assert.equal(controller.loading(), false);
  assert.equal(controller.error(), '');
});

test('a second alignment only fetches pairs never seen; a new capture starts over', async () => {
  const api = fakeApi();
  const controller = new TokenAnalysisController(api);
  controller.load('cap', [{turn: 1, pairs: pairs(100)}], new AbortController().signal);
  await flush();
  api.calls[0].resolve();
  await flush();
  controller.load(
    'cap',
    [{turn: 1, pairs: [...pairs(100), ...pairs(10, 100)]}],
    new AbortController().signal,
  );
  await flush();
  assert.equal(api.calls.length, 2);
  assert.equal(api.calls[1].pairs.length, 10, 'only the ten unseen pairs are requested');
  assert.equal(controller.results().size, 100, 'cached metrics show before the fetch lands');
  api.calls[1].resolve();
  await flush();
  assert.equal(controller.results().size, 110);
  controller.load('other', [{turn: 1, pairs: pairs(5)}], new AbortController().signal);
  await flush();
  assert.equal(controller.results().size, 0);
  assert.equal(api.calls[2].captureId, 'other');
});

test('aborted loads change nothing; failures report once and never stick to a later load', async () => {
  const api = fakeApi();
  const controller = new TokenAnalysisController(api);
  const aborter = new AbortController();
  controller.load('cap', [{turn: 1, pairs: pairs(3)}], aborter.signal);
  await flush();
  aborter.abort();
  api.calls[0].resolve();
  await flush();
  assert.equal(controller.results().size, 0);
  assert.equal(controller.loading(), true, 'the superseding load owns the loading flag');
  controller.load('cap', [{turn: 2, pairs: pairs(3)}], new AbortController().signal);
  await flush();
  api.calls[1].reject(new Error('offline'));
  await flush();
  assert.equal(controller.error(), 'Token metrics could not be loaded.');
  assert.equal(controller.loading(), false);
  controller.load('cap', [{turn: 2, pairs: pairs(3)}], new AbortController().signal);
  assert.equal(controller.error(), '');
});

test('no capture or nothing missing resolves immediately without a request', () => {
  const api = fakeApi();
  const controller = new TokenAnalysisController(api);
  controller.load(null, [{turn: 1, pairs: pairs(3)}], new AbortController().signal);
  controller.load('cap', [], new AbortController().signal);
  assert.equal(api.calls.length, 0);
  assert.equal(controller.loading(), false);
});
