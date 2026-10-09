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
import {strict as assert} from 'node:assert';
import {fileURLToPath} from 'node:url';

async function common(name) {
  const feature = name === 'batch_forward' ? 'graph' : 'conversation';
  const {outputFiles} = await build({
    entryPoints: [fileURLToPath(
        new URL(`../src/features/${feature}/${name}.ts`, import.meta.url))],
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'node',
  });
  return import(
      'data:text/javascript;base64,' +
      Buffer.from(outputFiles[0].text).toString('base64'));
}
const {batchForwardId, tokenMatchesBatch} = await common('batch_forward');
const paired = {
  batch: 12,
  turn: 1,
  phase: 'decode',
  forward_ids: {ref: 9, target: 5}
};
assert.equal(batchForwardId(paired, 'ref'), 9);
assert.equal(batchForwardId(paired, 'target'), 5);
assert.equal(
    tokenMatchesBatch({batch: 12, source_forward_id: 5}, paired, 'target'),
    true);
assert.equal(
    tokenMatchesBatch({batch: 12, source_forward_id: 9}, paired, 'target'),
    false);
assert.equal(
    tokenMatchesBatch({batch: 12, source_forward_id: 9}, paired, 'ref'), true);
const partial = {
  ...paired,
  forward_id: 9,
  forward_ids: {ref: 9}
};
assert.equal(batchForwardId(partial, 'target'), undefined);
assert.equal(
    tokenMatchesBatch({batch: 12, source_forward_id: 9}, partial, 'target'),
    false);
assert.equal(
    batchForwardId({...paired, forward_ids: {ref: 9, target: null}}, 'target'),
    null);
assert.equal(
    tokenMatchesBatch(
        {batch: 12, source_forward_id: 9}, {batch: 12, forward_id: 9},
        'target'),
    true,
);
console.log(
    'PASS: independent native forward IDs, missing-side mapping never falls back, legacy shared forward retained',
);

const {capturedTokenLabel} = await common('conversation_content');
assert.equal(capturedTokenLabel({text: null, id: 106, step: 5}), 'Token #106');
assert.equal(capturedTokenLabel({text: null, step: 5}), 'Not captured');
assert.equal(capturedTokenLabel({text: '', id: 106, step: 5}), '');
assert.equal(
    capturedTokenLabel({text: '<turn|>', id: 106, step: 5}), '<turn|>');
const {pairIdentity, samePairIdentity, validViewState, readViewState} =
    await common('conversation_view_state');
const pair = {
  index: 5,
  ref: {step: 5, text: null, id: 106},
  target: {step: 6, text: null, id: 106},
  match: null,
};
assert.equal(samePairIdentity(pair, pairIdentity(pair)), true);
const saved = {
  version: 2,
  formula: 'NOT token_match',
  alignment: 'steps',
  selection: {kind: 'token', turn: 1, index: 5},
  identity: pairIdentity(pair),
  details: true,
  boundaries: true,
  whitespace: false,
  color: 'token_match',
  collapsed: [],
  thinking: [],
  infoExpanded: true,
  groups: {},
  panelWidth: 390,
  scrollTop: 0,
  anchor: null,
  trends: {},
};
assert.equal(validViewState(readViewState(saved)), true);
assert.equal(readViewState(saved).identity.target.text, null);
const {virtualConversation} = await common('conversation_virtualization');
const long = Array.from({length: 2050}, (_, step) => ({
                                          text: step === 2049 ? null : 'a',
                                          id: step === 2049 ? 106 : 818,
                                          step,
                                        }));
const virtual = virtualConversation(
    {turn: 1, run: 'ref', tokens: long, output: 'a'.repeat(2049)},
    {
      turn: 1,
      run: 'target',
      tokens: long.map((t) => ({...t})),
      output: 'a'.repeat(2049)
    },
    'steps',
    true,
);
assert.equal(virtual.steps.ref.at(-1), 'Token #106');
assert.equal(virtual.pairs.at(-1).match, null);
assert.equal(virtual.responseTextComplete, false);
console.log(
    'PASS: unknown text label, nullable bookmark identity and >2048-token virtualization');
