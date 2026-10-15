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

import {ChangeDetectionStrategy, Component, computed, input, output, signal} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule} from '@angular/cdk/a11y';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {ConfigPicker} from '../../shared/config_picker/config_picker';

export interface TensorIndexDimension {
  id: string;
  label: string;
  value: string | null;
  options: readonly string[];
  optionLabels?: Readonly<Record<string, string>>;
  disabled?: boolean;
}
export interface TensorIndexChange {
  dimension: string;
  value: string;
}

/** Named coordinates for any tensor slice; the owner validates dependent dimensions. */
@Component({
  selector: 'analysis-tensor-index',
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
  template: ` <button
      mat-button
      type="button"
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      aria-label="Tensor Index"
      aria-haspopup="dialog"
      [attr.aria-expanded]="open()"
      [disabled]="!dimensions().length"
      [title]="summary()"
      (click)="open.set(!open())"
    >
      <span class="value">{{ summary() }}</span
      ><mat-icon iconPositionEnd>arrow_drop_down</mat-icon>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      [width]="260"
      [gap]="0"
      (closed)="open.set(false)"
    >
      <section class="picker" overlayPanel aria-label="Tensor Index selection">
        @for (dimension of dimensions(); track dimension.id) {
          <div class="dimension-row">
            <span>{{ dimension.label }}</span>
            <config-picker
              [label]="'Tensor ' + dimension.label.toLowerCase()"
              [plural]="dimension.label.toLowerCase() + ' indices'"
              [value]="dimension.value ?? 'Select'"
              [displayValue]="dimension.optionLabels?.[dimension.value ?? '']"
              [options]="dimension.options.slice()"
              [optionLabels]="dimension.optionLabels ?? {}"
              [disabled]="dimension.disabled || !dimension.options.length"
              [compact]="true"
              (valueChange)="choose(dimension.id, $event)"
            />
          </div>
        }
      </section>
    </ng-template>`,
  styles: `
    @use '../../theme/explorer_controls' as controls;
    @include controls.configuration-picker-trigger;
    :host {
      display: block;
      width: 190px;
      min-width: 0;
    }
    .picker {
      @include controls.configuration-popup;
      width: 100%;
      padding: 8px;
      border: 1px solid var(--me-outline-variant-color);
      border-radius: 0 0 4px 4px;
      font: 12px/20px var(--me-font-family);
    }
    .dimension-row {
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }
    .dimension-row + .dimension-row {
      margin-top: 8px;
    }
  `,
})
export class TensorIndexControl {
  readonly dimensions = input<readonly TensorIndexDimension[]>([]);
  readonly valueChange = output<TensorIndexChange>();
  readonly open = signal(false);
  readonly summary = computed(
    () =>
      '[' +
      this.dimensions()
        .map((d) => `${d.label}: ${d.optionLabels?.[d.value ?? ''] ?? d.value ?? '-'}`)
        .join(', ') +
      ']',
  );
  choose(id: string, value: string) {
    const dimension = this.dimensions().find((d) => d.id === id);
    if (!dimension || dimension.disabled || !dimension.options.includes(value)) return;
    this.valueChange.emit({dimension: id, value});
  }
}
