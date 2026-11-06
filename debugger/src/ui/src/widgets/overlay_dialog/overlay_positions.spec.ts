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

import {overlayPositions} from './overlay_positions';

describe('overlayPositions', () => {
  it('below-start opens under the origin and flips above as the fallback', () => {
    expect(overlayPositions('below-start', 8)).toStrictEqual([
      {originX: 'start', originY: 'bottom', overlayX: 'start', overlayY: 'top', offsetY: 8},
      {originX: 'start', originY: 'top', overlayX: 'start', overlayY: 'bottom', offsetY: -8},
    ]);
  });

  it('above-end keeps the end edges aligned, negates the gap and applies the inset', () => {
    expect(overlayPositions('above-end', 8, -4)).toStrictEqual([
      {
        originX: 'end',
        originY: 'top',
        overlayX: 'end',
        overlayY: 'bottom',
        offsetY: -8,
        offsetX: -4,
      },
      {
        originX: 'end',
        originY: 'bottom',
        overlayX: 'end',
        overlayY: 'top',
        offsetY: 8,
        offsetX: -4,
      },
    ]);
  });
});
