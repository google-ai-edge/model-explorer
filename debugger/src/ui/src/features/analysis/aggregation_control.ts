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

import {ChangeDetectionStrategy, Component, input, output, signal} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule} from '@angular/cdk/a11y';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import type {HeatmapReduction} from './heatmap_model';

export interface Aggregation {
  dimension: string;
  label: string;
  options: readonly {id: string; label: string}[];
  selected: readonly string[];
  function: HeatmapReduction;
  /** Show the concrete tensor point when there is no cross-point reduction. */
  showSingleSelection?: boolean;
}
export interface AggregationChange {
  selection: string[] | null;
  function: HeatmapReduction;
}

@Component({
  selector: 'analysis-aggregation',
  imports: [
    OverlayModule,
    A11yModule,
    MatButtonModule,
    MatIconModule,
    OverlayDialog,
    OverlayPanel,
    ConfigPicker,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <button
      mat-button
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      [attr.aria-label]="value() ? value()!.label + ' aggregation' : 'Aggregation'"
      aria-haspopup="dialog"
      [attr.aria-expanded]="open()"
      [disabled]="!value()?.options?.length"
      [attr.title]="value()?.options?.length ? null : 'No aggregation dimension available'"
      (click)="edit()"
    >
      <span class="value">
        @if (value()?.options?.length) {
          @let aggregation = value()!;
          @if (aggregation.showSingleSelection && aggregation.selected.length === 1) {
            {{ selectionLabel(aggregation) }}
          } @else {
            {{ names[aggregation.function] }} · {{ aggregation.selected.length }}
            {{ aggregation.label }}
          }
        } @else {
          -
        }
      </span>
      <mat-icon iconPositionEnd>arrow_drop_down</mat-icon>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      [width]="240"
      [gap]="0"
      (closed)="open.set(false)"
    >
      @if (value(); as aggregation) {
        <section
          class="picker"
          overlayPanel
          [attr.aria-label]="value() ? value()!.label + ' aggregation' : 'Aggregation'"
        >
          <strong>{{ aggregation.label }}</strong>
          <label
            ><input
              type="checkbox"
              [attr.aria-label]="'All ' + aggregation.label.toLowerCase()"
              [checked]="draft().length === aggregation.options.length"
              [indeterminate]="draft().length > 0 && draft().length < aggregation.options.length"
              (change)="toggleAll()"
            />All</label
          >
          <div class="options">
            @for (option of aggregation.options; track option.id) {
              <label
                ><input
                  type="checkbox"
                  [attr.aria-label]="option.label"
                  [checked]="draft().includes(option.id)"
                  (change)="toggle(option.id)"
                />{{ option.label }}</label
              >
            }
          </div>
          <div class="function-row">
            Function
            <config-picker
              label="Aggregation function"
              [compact]="true"
              [value]="names[function()]"
              [options]="['Minimum', 'Maximum', 'Mean']"
              (valueChange)="chooseFunction($event)"
            />
          </div>
          <div class="actions">
            <button mat-button (click)="open.set(false)">Cancel</button>
            <button mat-button [disabled]="!draft().length" (click)="apply()">Apply</button>
          </div>
        </section>
      }
    </ng-template>
  `,
  styleUrls: ['./layer_legend.scss'],
  styles: `
    :host {
      margin-left: 0;
      width: 160px;
    }
    .function-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }
    config-picker {
      flex: 1;
      min-width: 0;
    }
  `,
})
export class AggregationControl {
  readonly value = input<Aggregation | null>(null);
  readonly valueChange = output<AggregationChange>();
  readonly open = signal(false);
  readonly draft = signal<string[]>([]);
  readonly function = signal<HeatmapReduction>('min');
  readonly names = {min: 'Minimum', max: 'Maximum', mean: 'Mean'};
  selectionLabel(value: Aggregation) {
    return (
      value.options.find((option) => option.id === value.selected[0])?.label ?? value.selected[0]
    );
  }
  edit() {
    const value = this.value();
    if (!value?.options.length) return;
    this.draft.set([...value.selected]);
    this.function.set(value.function);
    this.open.set(true);
  }
  toggle(id: string) {
    this.draft.update((ids) =>
      ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id],
    );
  }
  toggleAll() {
    const options = this.value()?.options ?? [];
    this.draft.set(
      this.draft().length === options.length ? [] : options.map((option) => option.id),
    );
  }
  chooseFunction(label: string) {
    this.function.set(label === 'Minimum' ? 'min' : label === 'Maximum' ? 'max' : 'mean');
  }
  apply() {
    const value = this.value();
    if (!value || !this.draft().length) return;
    const selected = value.options
      .map((option) => option.id)
      .filter((id) => this.draft().includes(id));
    this.valueChange.emit({
      selection: selected.length === value.options.length ? null : selected,
      function: this.function(),
    });
    this.open.set(false);
  }
}
