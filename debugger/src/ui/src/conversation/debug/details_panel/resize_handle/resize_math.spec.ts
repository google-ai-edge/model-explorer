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

import {clampSize, draggedSize, steppedSize} from './resize_math';

describe('resize math', () => {
  const bounds = {min: 280, max: 600};

  it('clampSize rounds and bounds', () => {
    expect(clampSize(279.6, bounds)).toBe(280);
    expect(clampSize(1000, bounds)).toBe(600);
    expect(clampSize(350.4, bounds)).toBe(350);
  });

  it('draggedSize grows toward the page start for right-anchored panels and away otherwise', () => {
    expect(draggedSize(350, 900, 850, bounds, true)).toBe(400);
    expect(draggedSize(350, 900, 950, bounds, true)).toBe(300);
    expect(draggedSize(120, 400, 450, {min: 80, max: 320}, false)).toBe(170);
    expect(draggedSize(350, 900, 0, bounds, true), 'clamped at max').toBe(600);
    expect(draggedSize(350, 900, 2000, bounds, true), 'clamped at min').toBe(280);
  });

  it('steppedSize maps arrows by orientation and anchoring, Home/End to the bounds', () => {
    const right = {orientation: 'vertical' as const, reverse: true, step: 20};
    expect(steppedSize(350, 'ArrowLeft', bounds, right)).toBe(370);
    expect(steppedSize(350, 'ArrowRight', bounds, right)).toBe(330);
    expect(steppedSize(350, 'ArrowUp', bounds, right)).toBe(null);
    expect(steppedSize(350, 'Home', bounds, right)).toBe(280);
    expect(steppedSize(350, 'End', bounds, right)).toBe(600);
    expect(steppedSize(590, 'ArrowLeft', bounds, right), 'clamped').toBe(600);
    const bottom = {orientation: 'horizontal' as const, reverse: false, step: 10};
    const heights = {min: 80, max: 320};
    expect(steppedSize(120, 'ArrowDown', heights, bottom)).toBe(130);
    expect(steppedSize(120, 'ArrowRight', heights, bottom)).toBe(130);
    expect(steppedSize(120, 'ArrowUp', heights, bottom)).toBe(110);
    expect(steppedSize(120, 'Tab', heights, bottom)).toBe(null);
  });
});
