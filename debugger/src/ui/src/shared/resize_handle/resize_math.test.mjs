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
  entryPoints: [fileURLToPath(new URL('./resize_math.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {clampSize, draggedSize, steppedSize} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const bounds = {min: 280, max: 600};

test('clampSize rounds and bounds', () => {
  assert.equal(clampSize(279.6, bounds), 280);
  assert.equal(clampSize(1000, bounds), 600);
  assert.equal(clampSize(350.4, bounds), 350);
});

test('draggedSize grows toward the page start for right-anchored panels and away otherwise', () => {
  assert.equal(draggedSize(350, 900, 850, bounds, true), 400);
  assert.equal(draggedSize(350, 900, 950, bounds, true), 300);
  assert.equal(draggedSize(120, 400, 450, {min: 80, max: 320}, false), 170);
  assert.equal(draggedSize(350, 900, 0, bounds, true), 600, 'clamped at max');
  assert.equal(draggedSize(350, 900, 2000, bounds, true), 280, 'clamped at min');
});

test('steppedSize maps arrows by orientation and anchoring, Home/End to the bounds', () => {
  const right = {orientation: 'vertical', reverse: true, step: 20};
  assert.equal(steppedSize(350, 'ArrowLeft', bounds, right), 370);
  assert.equal(steppedSize(350, 'ArrowRight', bounds, right), 330);
  assert.equal(steppedSize(350, 'ArrowUp', bounds, right), null);
  assert.equal(steppedSize(350, 'Home', bounds, right), 280);
  assert.equal(steppedSize(350, 'End', bounds, right), 600);
  assert.equal(steppedSize(590, 'ArrowLeft', bounds, right), 600, 'clamped');
  const bottom = {orientation: 'horizontal', reverse: false, step: 10};
  const heights = {min: 80, max: 320};
  assert.equal(steppedSize(120, 'ArrowDown', heights, bottom), 130);
  assert.equal(steppedSize(120, 'ArrowRight', heights, bottom), 130);
  assert.equal(steppedSize(120, 'ArrowUp', heights, bottom), 110);
  assert.equal(steppedSize(120, 'Tab', heights, bottom), null);
});
