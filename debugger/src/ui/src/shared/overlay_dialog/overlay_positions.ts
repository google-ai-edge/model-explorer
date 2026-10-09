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

import type {ConnectedPosition} from '@angular/cdk/overlay';

/** Where a panel opens relative to its origin; the vertical flip is the fallback. */
export type OverlayPlacement =
  | 'below-start'
  | 'below-end'
  | 'above-start'
  | 'above-end';

export function overlayPositions(
  placement: OverlayPlacement,
  gap: number,
  inset = 0,
): ConnectedPosition[] {
  const [vertical, horizontal] = placement.split('-') as [
    'below' | 'above',
    'start' | 'end',
  ];
  const position = (side: 'below' | 'above'): ConnectedPosition => ({
    originX: horizontal,
    originY: side === 'below' ? 'bottom' : 'top',
    overlayX: horizontal,
    overlayY: side === 'below' ? 'top' : 'bottom',
    offsetY: side === 'below' ? gap : -gap,
    ...(inset ? {offsetX: inset} : {}),
  });
  return [
    position(vertical),
    position(vertical === 'below' ? 'above' : 'below'),
  ];
}
