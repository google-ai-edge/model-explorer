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
import {build} from 'esbuild';
const {outputFiles} = await build({
  entryPoints: [new URL('./stage_state_model.ts', import.meta.url).pathname],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {summarizeStageValues} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
test('stage statistics exclude missing/nonfinite evidence and retain global extrema positions', () => {
  const result = summarizeStageValues([
    {position: 128, value: null},
    {position: 129, value: 0},
    {position: 130, value: 2},
    {position: 131, value: NaN},
    {position: 132, value: Infinity},
  ]);
  assert.deepEqual(result, {
    mean: 1,
    min: 0,
    max: 2,
    min_position: 129,
    max_position: 130,
    valid_count: 2,
  });
  assert.equal(summarizeStageValues([]).mean, null);
});
test('finite large values do not overflow the stage mean', () => {
  assert.equal(
    summarizeStageValues([
      {position: 0, value: 1e308},
      {position: 1, value: 1e308},
    ]).mean,
    1e308,
  );
});
