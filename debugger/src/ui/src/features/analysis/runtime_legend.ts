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

import {Component, computed, input} from '@angular/core';
import {runtimeSeries, runtimeSelectionBackground} from './runtime_series';

@Component({
  selector: 'analysis-runtime-legend',
  template: `<div
    class="tensor-runtime-legend"
    [class.selection-legend]="appearance() === 'selection'"
    [attr.aria-label]="
      appearance() === 'selection' ? 'Selected token legend' : 'Tensor runtime legend'
    "
  >
    @for (side of visible(); track side) {
      <span [attr.title]="runtimeStyles[side].name">
        <i
          [class]="side"
          [style.border-color]="runtimeStyles[side].color"
          [style.background-color]="appearance() === 'selection' ? selectionBackground(side) : null"
          aria-hidden="true"
        ></i
        >{{ runtimeStyles[side].label }}
      </span>
    }
  </div>`,
  styles: `
    :host {
      display: block;
      flex: none;
    }
    .tensor-runtime-legend {
      display: flex;
      align-items: center;
      height: 32px;
      gap: 12px;
      white-space: nowrap;
      color: var(--me-on-surface-variant-color);
      font: 11px/20px var(--me-font-family);
    }
    span {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    i {
      width: 18px;
      border-top: 1.5px solid;
    }
    .selection-legend i {
      box-sizing: border-box;
      width: 12px;
      height: 12px;
      border: 0;
      border-radius: 2px;
    }
    .tensor-runtime-legend:not(.selection-legend) i.target {
      border-top-style: dashed;
    }
  `,
})
export class AnalysisRuntimeLegend {
  readonly appearance = input<'line' | 'selection'>('line');
  readonly selectionBackground = runtimeSelectionBackground;
  readonly runtimes = input<readonly string[]>(['ref', 'target']);
  readonly visible = computed(() =>
    (['ref', 'target'] as const).filter((side) => this.runtimes().includes(side)),
  );
  readonly runtimeStyles = {ref: runtimeSeries('ref'), target: runtimeSeries('target')};
}
