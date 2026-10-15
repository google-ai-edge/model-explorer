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

import {
  afterRenderEffect,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  input,
} from '@angular/core';

export interface ChartHoverContent {
  properties: readonly {label: string; value: string}[];
  title?: string;
  columns?: readonly string[];
  rows?: readonly {label: string; values: readonly string[]}[];
  activeColumn?: number;
}
export interface ChartHoverPosition {
  left: number;
  top: number;
  content: ChartHoverContent;
}
export function hoverBin(start: number, end: number, reduction: 'min' | 'max' | 'mean') {
  return end - start > 1 ? {min: 'Minimum', max: 'Maximum', mean: 'Mean'}[reduction] : '-';
}
export function hoverInterval(start: number, end: number) {
  return end - start > 1 ? `${start}–${end - 1}` : String(start);
}

/** Shared non-interactive readout; measuring the rendered table keeps it inside the chart and viewport. */
@Component({
  selector: 'analysis-chart-hover',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {role: 'tooltip'},
  template: `
    <dl>
      @for (property of info().content.properties; track property.label) {
        <dt>{{ property.label }}</dt>
        <dd>{{ property.value }}</dd>
      }
    </dl>
    @if (info().content.title; as title) {
      <div class="title">{{ title }}</div>
    }
    @if (info().content.columns; as columns) {
      <table>
        <thead>
          <tr>
            <th></th>
            @for (column of columns; track $index) {
              <th scope="col" [class.active]="info().content.activeColumn === $index">
                {{ column }}
              </th>
            }
          </tr>
        </thead>
        <tbody>
          @for (row of info().content.rows; track row.label) {
            <tr>
              <th scope="row">{{ row.label }}</th>
              @for (value of row.values; track $index) {
                <td>{{ value }}</td>
              }
            </tr>
          }
        </tbody>
      </table>
    }
  `,
  styles: `
    :host {
      position: absolute;
      z-index: 5;
      pointer-events: none;
      padding: 6px 8px;
      border: 1px solid var(--me-outline-variant-color);
      border-radius: 3px;
      background: var(--me-surface-color);
      color: var(--me-on-surface-color);
      font: 12px/18px var(--me-font-family);
      box-shadow: 0 2px 6px #0002;
      width: max-content;
      max-width: calc(100% - 16px);
      box-sizing: border-box;
    }
    dl {
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 0 16px;
      margin: 0;
    }
    dt,
    thead th {
      color: var(--me-on-surface-variant-color);
    }
    dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    .title {
      margin-top: 6px;
    }
    table {
      border-collapse: collapse;
      margin-top: 6px;
      width: 100%;
      font: inherit;
    }
    .title + table {
      margin-top: 0;
    }
    th,
    td {
      padding: 0 0 0 12px;
      text-align: left;
      font-weight: 400;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }
    th:first-child {
      padding-left: 0;
      white-space: nowrap;
    }
    thead th.active {
      color: var(--me-primary-color);
      text-decoration: underline;
      text-underline-offset: 3px;
    }
  `,
})
export class ChartHover {
  readonly info = input.required<ChartHoverPosition>();
  private readonly element = inject(ElementRef<HTMLElement>);
  constructor() {
    afterRenderEffect(() => {
      const info = this.info(),
        element = this.element.nativeElement;
      const parent = element.parentElement?.getBoundingClientRect();
      if (!parent) return;
      const rect = element.getBoundingClientRect();
      const left = Math.max(
        8,
        Math.min(
          info.left + 12,
          parent.width - rect.width - 8,
          window.innerWidth - parent.left - rect.width - 8,
        ),
      );
      const bottom = Math.min(parent.height, window.innerHeight - parent.top);
      const top = Math.max(8, Math.min(info.top + 12, bottom - rect.height - 8));
      element.style.left = `${left}px`;
      element.style.top = `${top}px`;
    });
  }
}
