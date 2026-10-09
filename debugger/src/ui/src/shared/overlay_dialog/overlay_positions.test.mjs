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
  entryPoints: [fileURLToPath(new URL('./overlay_positions.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {overlayPositions} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);

test('below-start opens under the origin and flips above as the fallback', () => {
  assert.deepEqual(overlayPositions('below-start', 8), [
    {originX: 'start', originY: 'bottom', overlayX: 'start', overlayY: 'top', offsetY: 8},
    {originX: 'start', originY: 'top', overlayX: 'start', overlayY: 'bottom', offsetY: -8},
  ]);
});

test('above-end keeps the end edges aligned, negates the gap and applies the inset', () => {
  assert.deepEqual(overlayPositions('above-end', 8, -4), [
    {originX: 'end', originY: 'top', overlayX: 'end', overlayY: 'bottom', offsetY: -8, offsetX: -4},
    {originX: 'end', originY: 'bottom', overlayX: 'end', overlayY: 'top', offsetY: 8, offsetX: -4},
  ]);
});
