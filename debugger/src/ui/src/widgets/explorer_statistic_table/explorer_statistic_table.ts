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

/** Shared statistics-table presentation following Model Explorer's summary tables. */
import {NgTemplateOutlet} from '@angular/common';
import {Component, TemplateRef, input} from '@angular/core';

@Component({
  selector: 'table[explorer-statistic-table]',
  imports: [NgTemplateOutlet],
  host: {'[attr.aria-label]': 'label()', '[class.six-columns]': 'columnCount() === 6'},
  template: `
    <colgroup>
      @if (columnCount() === 3) {
        <col class="numeric-column" />
        <col class="numeric-column" />
        <col class="numeric-column" />
      } @else if (columnCount() === 6) {
        <col class="identity-column" />
        <col />
        <col />
        <col />
        <col />
        <col />
      } @else {
        <col class="identity-column" />
        <col />
        <col />
        <col />
        <col />
      }
    </colgroup>
    <thead>
      <ng-container [ngTemplateOutlet]="header()" />
    </thead>
    @for (item of rows(); track item.label) {
      <tbody [class]="rowClass()">
        <ng-container [ngTemplateOutlet]="row()" [ngTemplateOutletContext]="{$implicit: item}" />
      </tbody>
    } @empty {
      <tbody>
        <tr>
          <td [attr.colspan]="columnCount() === 3 ? 3 : columnCount() === 6 ? 6 : 5">
            {{ emptyText() }}
          </td>
        </tr>
      </tbody>
    }
  `,
  styles: `
    :host {
      font-variant-numeric: tabular-nums;
    }
    // All inspector tables share five tracks; four-column tables span the last two.
    col.identity-column {
      width: var(--info-table-identity-width, 78px);
    }
    col:not(.identity-column) {
      width: calc((100% - var(--info-table-identity-width, 78px)) / 4);
    }
    :host(.six-columns) col:not(.identity-column) {
      width: calc((100% - var(--info-table-identity-width, 78px)) / 5);
    }
    col.numeric-column {
      width: calc(100% / 3);
    }
    td {
      padding: 1px 0;
      font: 12px / normal var(--me-font-family);
      color: var(--me-on-surface-variant-color);
    }
  `,
})
export class ExplorerStatisticTable<T extends {label: string}> {
  readonly label = input.required<string>();
  readonly rows = input.required<readonly T[]>();
  readonly header = input.required<TemplateRef<unknown>>();
  readonly row = input.required<TemplateRef<{$implicit: T}>>();
  readonly columnCount = input.required<number>();
  readonly rowClass = input('');
  readonly emptyText = input('No statistics available.');
}
