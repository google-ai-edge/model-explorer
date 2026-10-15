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

import {ChangeDetectionStrategy, Component, input, output} from '@angular/core';

/** The same bounded column strip for Position and Channel selection. */
@Component({
  selector: 'analysis-range-axis-selection',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<button
    class="position-column-axis"
    [attr.aria-label]="label()"
    [style.left.px]="axis().left"
    [style.top.px]="axis().top"
    [style.width.px]="axis().width"
    [style.height.px]="height()"
    [disabled]="disabled()"
    (mousemove)="pointerHover.emit($event)"
    (mouseleave)="pointerLeave.emit()"
    (click)="choose.emit($event)"
    (keydown)="navigate.emit($event)"
  >
    @for (cell of cells(); track $index) {
      <span
        class="axis-bin"
        aria-hidden="true"
        [style.left.px]="cell.left"
        [style.width.px]="cell.width"
      ></span>
    }
    @if (hover(); as box) {
      <span
        class="axis-hover"
        aria-hidden="true"
        [style.left.px]="box.left - axis().left"
        [style.width.px]="box.width"
      ></span>
    }
    @if (selected(); as box) {
      <span
        class="axis-selection"
        aria-hidden="true"
        [style.left.px]="box.left - axis().left"
        [style.width.px]="box.width"
      ></span>
    }
  </button>`,
  styles: [
    `
      @use './range_chart_style' as chart;
      @include chart.selection;
      :host {
        display: contents;
      }
    `,
  ],
})
export class RangeAxisSelection {
  readonly axis = input.required<{left: number; top: number; width: number}>();
  readonly cells = input<readonly {left: number; width: number}[]>([]);
  readonly hover = input<{left: number; width: number} | null>(null);
  readonly selected = input<{left: number; width: number} | null>(null);
  readonly label = input.required<string>();
  readonly height = input(8);
  readonly disabled = input(false);
  readonly pointerHover = output<MouseEvent>();
  readonly pointerLeave = output<void>();
  readonly choose = output<MouseEvent>();
  readonly navigate = output<KeyboardEvent>();
}
