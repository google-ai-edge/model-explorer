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
  entryPoints: [fileURLToPath(new URL('./format.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {formatSignificant, formatFixed, formatMetric, formatPercent, formatGiB, formatCount} =
  await import(
    'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
  );

test('significant and fixed digits, with placeholders for missing values', () => {
  assert.equal(formatSignificant(0.123456), '0.1235');
  assert.equal(formatSignificant(1.5, 5), '1.5');
  assert.equal(formatSignificant(null), '—');
  assert.equal(formatSignificant(NaN, 4, 'Unavailable'), 'Unavailable');
  assert.equal(formatFixed(1.5, 3), '1.500');
  assert.equal(formatFixed(undefined, 2), '—');
});

test('metric readouts switch to exponent notation outside the readable range and keep −0', () => {
  assert.equal(formatMetric(0.123456), '0.1235');
  assert.equal(formatMetric(0.0001234), '1.23e-4');
  assert.equal(formatMetric(12345.6), '1.23e+4');
  assert.equal(formatMetric(0), '0');
  assert.equal(formatMetric(-0), '−0');
  assert.equal(formatMetric(Infinity), 'Unavailable');
  assert.equal(formatMetric(null, {placeholder: '—'}), '—');
  assert.equal(
    formatMetric(0.00005, {digits: 4, exponentDigits: 3, small: 0.0001, large: 1000}),
    '5.000e-5',
  );
  assert.equal(formatMetric(999.99, {small: 0.0001, large: 1000}), '1000');
});

test('percentages scale ratios and never append % to a placeholder', () => {
  assert.equal(formatPercent(0.1234), '12.34%');
  assert.equal(formatPercent(0.0000012), '1.20e-4%');
  assert.equal(formatPercent(null), 'Unavailable');
  assert.equal(formatPercent(undefined, {placeholder: '—'}), '—');
});

test('bytes and counts use en-US grouping', () => {
  assert.equal(formatGiB(16 * 1024 ** 3), '16 GiB');
  assert.equal(formatGiB(1.5 * 1024 ** 3, 2), '1.5 GiB');
  assert.equal(formatGiB(20 * 1024 ** 3 + 123456789, 2), '20.11 GiB');
  assert.equal(formatCount(1234567), '1,234,567');
});
