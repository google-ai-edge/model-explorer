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

/** Pure sizing rules shared by every draggable panel edge. */
export interface ResizeBounds {
  min: number;
  max: number;
}
export type ResizeOrientation = 'vertical' | 'horizontal';

export function clampSize(value: number, bounds: ResizeBounds): number {
  return Math.max(bounds.min, Math.min(bounds.max, Math.round(value)));
}

/** Size after a pointer drag. `reverse` panels (right/bottom anchored) grow toward the page start. */
export function draggedSize(
  startSize: number,
  startPosition: number,
  position: number,
  bounds: ResizeBounds,
  reverse: boolean,
): number {
  const delta = position - startPosition;
  return clampSize(startSize + (reverse ? -delta : delta), bounds);
}

/** Size after a keyboard step, or null when the key does not resize. */
export function steppedSize(
  value: number,
  key: string,
  bounds: ResizeBounds,
  options: {orientation: ResizeOrientation; reverse: boolean; step: number},
): number | null {
  if (key === 'Home') return clampSize(bounds.min, bounds);
  if (key === 'End') return clampSize(bounds.max, bounds);
  const vertical = options.orientation === 'vertical';
  const forward = vertical ? ['ArrowRight'] : ['ArrowDown', 'ArrowRight'];
  const backward = vertical ? ['ArrowLeft'] : ['ArrowUp', 'ArrowLeft'];
  const grow = options.reverse ? backward : forward;
  const shrink = options.reverse ? forward : backward;
  if (grow.includes(key)) return clampSize(value + options.step, bounds);
  if (shrink.includes(key)) return clampSize(value - options.step, bounds);
  return null;
}
