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

import {DestroyRef, Directive, inject, output} from '@angular/core';

/** Keep Plotly data clicks and range gestures separate from background inspection. */
@Directive({
  standalone: true,
  host: {
    '(pointerdown)': 'start($event)',
    '(pointermove)': 'move($event)',
    '(pointercancel)': 'handled()',
    '(click)': 'click($event)',
  },
})
export class ChartBackground {
  readonly backgroundClick = output<MouseEvent>();
  private pointer: {x: number; y: number} | null = null;
  private consumed = false;
  private timer?: ReturnType<typeof setTimeout>;
  constructor() {
    inject(DestroyRef).onDestroy(() => clearTimeout(this.timer));
  }
  start(event: PointerEvent) {
    clearTimeout(this.timer);
    this.pointer = {x: event.clientX, y: event.clientY};
    this.consumed = event.button !== 0;
  }
  move(event: PointerEvent) {
    if (
      event.buttons &&
      this.pointer &&
      Math.hypot(event.clientX - this.pointer.x, event.clientY - this.pointer.y) > 4
    )
      this.handled();
  }
  handled() {
    this.consumed = true;
  }
  click(event: MouseEvent) {
    if (
      event.button !== 0 ||
      event.defaultPrevented ||
      (event.target as Element).closest(
        'button, a, input, select, textarea, [role="button"], .rangeslider-container, .modebar, .legend, kv-scatter-overview, analysis-range-axis-selection',
      )
    )
      return;
    // Plotly reports point clicks from document listeners after the host click.
    this.timer = setTimeout(() => {
      if (!this.consumed && !window.getSelection()?.toString()) this.backgroundClick.emit(event);
    });
  }
}
