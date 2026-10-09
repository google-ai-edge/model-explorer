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
  entryPoints: [fileURLToPath(new URL('./metric_query.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {compileMetricQuery} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const ratio = {
  fields: {
    a: {type: 'number'},
    b: {type: 'number'},
    b_alias: {type: 'number', key: 'b'},
    ok: {type: 'boolean'},
  },
  percentField: 'a',
  percentScale: 0.01,
};
const percent = {...ratio, percentScale: 1};
const run = (formula, row, language = ratio) => compileMetricQuery(formula, language)(row);

test('precedence, grouping, aliases and case-insensitive keywords', () => {
  const row = {a: 0.12, b: 0.5, ok: true};
  assert.equal(run('a > .1 OR b < 0 AND a < 0', row), true);
  assert.equal(run('(a > .1 OR b < 0) AND a < 0', row), false);
  assert.equal(run('nOt b < 0 aNd a >= 0', row), true);
  assert.equal(run('b_alias = b', row), true);
  assert.equal(run('ok AND NOT ok = false', row), true);
});

test('percent literals scale per language and are tied to the percent field', () => {
  assert.equal(run('a > 10%', {a: 0.12}), true);
  assert.equal(run('a > 12%', {a: 0.12}), false);
  assert.equal(run('a > 10%', {a: 0.12}, percent), false);
  assert.equal(run('a > 0.1%', {a: 0.12}, percent), true);
  assert.equal(run('10% < a', {a: 0.5}), true);
  for (const formula of ['b > 10%', '10% > .1'])
    assert.throws(() => compileMetricQuery(formula, ratio), /% is only valid with a/);
});

test('unknown metrics propagate as unknown through NOT, AND and OR', () => {
  for (const unknown of [null, undefined, NaN, Infinity, -Infinity]) {
    const row = {a: unknown, b: 0.5, ok: null};
    for (const formula of [
      'a > 0',
      'a != 0',
      'NOT a > 0',
      'NOT NOT a > 0',
      'a > 0 AND true',
      'a > 0 OR false',
      'ok',
      'NOT ok',
    ])
      assert.equal(run(formula, row), null, `${formula}, unknown=${unknown}`);
    assert.equal(run('a > 0 OR b > 0', row), true);
    assert.equal(run('NOT (a > 0 AND b < 0)', row), true);
    assert.equal(run('NOT (a > 0 OR b > 0)', row), false);
  }
});

test('requiresMetrics flags numeric fields only; a blank formula matches everything', () => {
  assert.equal(compileMetricQuery('NOT ok', ratio).requiresMetrics, false);
  assert.equal(compileMetricQuery('ok AND a > 1', ratio).requiresMetrics, true);
  const blank = compileMetricQuery(' \n\t', ratio);
  assert.equal(blank({}), true);
  assert.equal(blank.requiresMetrics, false);
});

test('helpful errors, type checks, injection rejection and size bounds', () => {
  for (const [formula, message] of [
    ['a >', /Expected a metric or value at column 4/],
    ['(a > .1', /Expected closing parenthesis/],
    ['a', /Add a comparison after a/],
    ['unknown > .1', /Unknown metric: unknown/],
    ['a > 1e999', /Number must be finite/],
    ['a == true', /same type/],
    ['true > false', /Use =, == or !=/],
    ['ok > true', /Use =, == or !=/],
    ['a > 0 > 1', /Unexpected term/],
    ['a > .1 && true', /Unexpected character/],
    ['a > .1; globalThis.attacked = true', /Unexpected character/],
    ['constructor = 0', /Unknown metric/],
    ['__proto__ = 0', /Unknown metric/],
    ['hasOwnProperty = 0', /Unknown metric/],
    ['globalThis.alert(1)', /Unexpected character/],
    ['b[0] > .1', /Unexpected character/],
    ['b > "0"', /Unexpected character/],
  ])
    assert.throws(() => compileMetricQuery(formula, ratio), message, formula);
  assert.throws(() => compileMetricQuery(' '.repeat(1001), ratio), /1,000 characters/);
  assert.throws(() => compileMetricQuery(Array(51).fill('1=1').join(' OR '), ratio), /200 terms/);
  assert.throws(
    () => compileMetricQuery('('.repeat(33) + 'true' + ')'.repeat(33), ratio),
    /32 levels/,
  );
  assert.throws(() => compileMetricQuery('NOT '.repeat(33) + 'true', ratio), /32 levels/);
  assert.equal(run('('.repeat(32) + 'true' + ')'.repeat(32), {}), true);
  assert.equal(globalThis.attacked, undefined);
});
