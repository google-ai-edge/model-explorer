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

import {formatSignificant} from '../../shared/format/format';
import {Component, input, output} from '@angular/core';
import {validValueRange, type ValueRange} from '../kv/kv_panel/kv_scatter_model';
@Component({
  selector: 'analysis-value-range',
  template: `<span title="Visible value range on both axes">Value Range</span>
    <input
      type="number"
      step="any"
      aria-label="Value range minimum"
      [value]="formatValue(range().start)"
      (change)="commit($event, 'start')"
      (keydown.enter)="commit($event, 'start')"
    />
    <span>–</span>
    <input
      type="number"
      step="any"
      aria-label="Value range maximum"
      [value]="formatValue(range().end)"
      (change)="commit($event, 'end')"
      (keydown.enter)="commit($event, 'end')"
    />`,
  styles: `
    @use '../../theme/explorer_controls' as controls;
    :host {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font: 12px/20px var(--me-font-family);
      color: var(--me-on-surface-color);
    }
    input {
      @include controls.inline-control-density;
      width: 90px;
      border: 1px solid var(--me-outline-variant-color);
      background: var(--me-surface-color);
      color: inherit;
    }
    input:focus-visible {
      outline: 2px solid var(--me-primary-color);
      outline-offset: 1px;
    }
  `,
})
export class ValueRangeControl {
  readonly formatValue = formatSignificant;
  readonly range = input.required<ValueRange>();
  readonly rangeChange = output<ValueRange>();
  commit(event: Event, edge: 'start' | 'end') {
    const input = event.target as HTMLInputElement;
    const next = {...this.range(), [edge]: input.valueAsNumber};
    if (validValueRange(next)) this.rangeChange.emit(next);
    else input.value = String(this.range()[edge]);
  }
}
