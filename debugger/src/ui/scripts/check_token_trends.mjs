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

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL(
      '../src/features/conversation/token_trends.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node'
});
const {
  normalizeTrendRange,
  sampleTokenTrend,
  trendTickStep,
  trendTokenLine,
  plotText,
  SIDE_TREND_METRICS
} =
    await import(
        'data:text/javascript;base64,' +
        Buffer.from(outputFiles[0].text).toString('base64'));
assert.deepEqual(normalizeTrendRange(11, [8, 3]), [3, 8]);
assert.deepEqual(normalizeTrendRange(11, [-4, 999]), [0, 10]);
assert.deepEqual(normalizeTrendRange(0, [0, 0]), [0, 1]);
assert.equal(trendTickStep([0, 131071]), 20000);
const values = Array(131072).fill(.1);
values[65536] = 100;
values[70000] = null;
values[131071] = 20;
const sampled = sampleTokenTrend(values, [65000, 71000], 900);
assert.ok(sampled.x.length < 3000);
assert.ok(sampled.x.includes(65536));
assert.ok(sampled.x.includes(70000));
assert.ok(sampled.x.includes(131071));
assert.equal(sampled.max, 100);
assert.equal(sampled.available, 6000);
const missing = sampleTokenTrend(Array(131072).fill(null), [0, 131071], 900);
assert.equal(missing.available, 0);
assert.ok(missing.y.every(v => v === null));
assert.deepEqual(sampleTokenTrend([0, null, 2], [0, 2], 900).y, [0, null, 2]);
console.log(
    'PASS: range clamping, integer ticks, 128k extrema/gaps, bounded Plotly sample and unknown values');
// Hover lines carry the token as text: control-token brackets and whitespace
// must survive Plotly's HTML subset.
assert.equal(plotText('<turn|> & "x"'), '&lt;turn|&gt; &amp; "x"');
assert.equal(
    trendTokenLine('Reference', {text: ' body', step: 7}, '11.2%'),
    'Reference · step 7 " body": <b>11.2%</b>');
assert.equal(
    trendTokenLine('Target', {text: '<turn|>', step: 23}),
    'Target · step 23 "&lt;turn|&gt;"');
assert.equal(
    trendTokenLine('Target', {text: '\n', step: 2}), 'Target · step 2 "\\n"');
assert.equal(
    trendTokenLine('Reference', {text: null, id: 5, step: 0}),
    'Reference · step 0 "Token #5"');
assert.equal(trendTokenLine('Reference', undefined, '—'), 'Reference: ∅');
assert.ok(SIDE_TREND_METRICS.every((metric) => metric.reading));
console.log(
    'PASS: hover token lines escape markup, keep whitespace legible and read ∅ at a gap');
