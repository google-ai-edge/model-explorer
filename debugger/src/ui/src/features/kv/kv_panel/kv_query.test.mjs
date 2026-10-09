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
  entryPoints: [fileURLToPath(new URL('./kv_query.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {compileKvQuery} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const captured = {relative_l2: 0.12, max_abs: 0.5, cosine_distance: 0.03};
const missing = {relative_l2: null, max_abs: null, cosine_distance: null};
const matches = (formula) => compileKvQuery(formula)(captured);

test('comparisons, numeric literals, percent ratios, aliases, precedence and nested boolean filters', () => {
  assert.equal(matches(''), true);
  assert.equal(compileKvQuery(' \n\t')(missing), true);
  for (const formula of [
    'relative_l2 > 10%',
    'relative_l2 >= .12',
    'relative_l2 < .2',
    'relative_l2 <= .12',
    'relative_l2 = .12',
    'relative_l2 == .12',
    'relative_l2 != .2',
    '10% < relative_l2',
    'max_abs_delta = max_abs',
    'cosine_distance = 3E-2',
  ]) {
    assert.equal(matches(formula), true, formula);
  }
  assert.equal(matches('relative_l2 > 0.1'), true);
  assert.equal(matches('relative_l2 > 12%'), false);
  assert.equal(
    matches('NOT (relative_l2 > 10% AND (max_abs < .1 OR cosine_distance > .02))'),
    false,
  );
  assert.equal(matches('relative_l2 > .1 OR max_abs < 0 AND cosine_distance < 0'), true);
  assert.equal(matches('(relative_l2 > .1 OR max_abs < 0) AND cosine_distance < 0'), false);
  assert.equal(matches('nOt max_abs < 0 aNd cosine_distance >= 0'), true);
  assert.equal(matches('max_abs > -2e-3 AND true != false'), true);
});

test('missing and nonfinite metrics stay unknown through NOT, AND and OR', () => {
  for (const unknown of [null, undefined, NaN, Infinity, -Infinity]) {
    const row = {...captured, relative_l2: unknown};
    for (const formula of [
      'relative_l2 > 0',
      'relative_l2 != 0',
      'NOT relative_l2 > 0',
      'NOT NOT relative_l2 > 0',
      'relative_l2 > 0 AND true',
      'relative_l2 > 0 OR false',
    ]) {
      assert.equal(compileKvQuery(formula)(row), false, `${formula}, unknown=${unknown}`);
    }
    assert.equal(compileKvQuery('relative_l2 > 0 OR max_abs > 0')(row), true);
    assert.equal(compileKvQuery('NOT (relative_l2 > 0 AND max_abs < 0)')(row), true);
    assert.equal(compileKvQuery('NOT (relative_l2 > 0 OR max_abs > 0)')(row), false);
  }
});

test('helpful errors, type and unit checks, injection rejection and size/token/depth bounds', () => {
  for (const [formula, message] of [
    ['relative_l2 >', /Expected a metric or value at column 14/],
    ['(relative_l2 > .1', /Expected closing parenthesis/],
    ['relative_l2', /Add a comparison/],
    ['unknown > .1', /Unknown metric: unknown/],
    ['max_abs > 10%', /% is only valid with relative_l2/],
    ['cosine_distance > 10%', /% is only valid with relative_l2/],
    ['10% > .1', /% is only valid with relative_l2/],
    ['relative_l2 > 1e999', /Number must be finite/],
    ['relative_l2 == true', /same type/],
    ['true > false', /Use =, == or !=/],
    ['relative_l2 > 0 > 1', /Unexpected term/],
    ['relative_l2 > .1 && true', /Unexpected character/],
    ['relative_l2 > .1; globalThis.attacked = true', /Unexpected character/],
    ['constructor = 0', /Unknown metric/],
    ['__proto__ = 0', /Unknown metric/],
    ['globalThis.alert(1)', /Unexpected character/],
    ['max_abs[0] > .1', /Unexpected character/],
    ['max_abs > "0"', /Unexpected character/],
  ])
    assert.throws(() => compileKvQuery(formula), message, formula);
  assert.throws(() => compileKvQuery(' '.repeat(1001)), /1,000 characters/);
  assert.throws(() => compileKvQuery(Array(51).fill('1=1').join(' OR ')), /200 terms/);
  assert.throws(() => compileKvQuery('('.repeat(33) + 'true' + ')'.repeat(33)), /32 levels/);
  assert.throws(() => compileKvQuery('NOT '.repeat(33) + 'true'), /32 levels/);
  assert.equal(matches('('.repeat(32) + 'true' + ')'.repeat(32)), true);
  assert.equal(globalThis.attacked, undefined);
});
