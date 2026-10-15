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

import {Component, input, output} from '@angular/core';
interface Range {
  start: number;
  end: number;
}
/** UI ends are inclusive; data and Plotly adapters use half-open intervals. */
@Component({
  selector: 'analysis-position-range',
  template: `@if (range(); as r) {
    <span title="Visible range, including both endpoints">{{ label() }}</span>
    <span class="range-inputs" role="group" [attr.aria-label]="label() + ' inputs'">
      <input
        [attr.aria-label]="label() + ' start'"
        type="number"
        step="1"
        [min]="full()?.start ?? 0"
        [max]="r.end - 1"
        [value]="r.start"
        (change)="commit($event, 'start')"
      />
      <span>–</span>
      <input
        [attr.aria-label]="label() + ' end'"
        type="number"
        step="1"
        [min]="r.start"
        [max]="(full()?.end ?? r.end) - 1"
        [value]="r.end - 1"
        (change)="commit($event, 'end')"
      />
      <span class="range-location-frame" aria-hidden="true"></span>
    </span>
  }`,
  styles: [
    `
      @use '../../theme/explorer_controls' as controls;
      :host {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font: 12px/20px var(--me-font-family);
        color: var(--me-on-surface-color);
      }
      .range-inputs {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .range-location-frame {
        position: absolute;
        inset: 0;
        border-radius: 3px;
        pointer-events: none;
      }
      input {
        @include controls.inline-control-density;
        width: 78px;
        border: 1px solid var(--me-outline-variant-color);
        background: var(--me-surface-color);
        color: inherit;
      }
      input:focus-visible {
        outline: 2px solid var(--me-primary-color);
        outline-offset: 1px;
      }
    `,
  ],
})
export class PositionRangeControl {
  readonly label = input('Token Range');
  readonly range = input<Range | null>(null);
  readonly full = input<Range | null>(null);
  readonly rangeChange = output<Range>();
  commit(event: Event, edge: 'start' | 'end') {
    const node = event.target as HTMLInputElement,
      r = this.range(),
      full = this.full();
    if (!r || !full) return;
    const value = node.valueAsNumber;
    const next = !Number.isFinite(value)
      ? r
      : edge === 'start'
        ? {start: Math.max(full.start, Math.min(r.end - 1, Math.floor(value))), end: r.end}
        : {start: r.start, end: Math.min(full.end, Math.max(r.start + 1, Math.floor(value) + 1))};
    node.value = String(edge === 'start' ? next.start : next.end - 1);
    if (next.start !== r.start || next.end !== r.end) this.rangeChange.emit(next);
  }
}
