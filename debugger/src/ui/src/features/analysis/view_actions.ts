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

import {Component, computed, input, output} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
@Component({
  selector: 'analysis-view-actions',
  imports: [MatIconModule, MatTooltipModule],
  template: `@if (showReset()) {
      <button
        aria-label="Reset view"
        matTooltip="Reset view"
        [disabled]="!changed()"
        (click)="resetView.emit()"
      >
        <mat-icon>center_focus_strong</mat-icon>
      </button>
    }
    @if (expandable() && !expanded()) {
      <button
        class="expand-action"
        aria-label="Expand analysis"
        matTooltip="Expand analysis"
        (click)="expandView.emit()"
      >
        <mat-icon aria-hidden="true">zoom_out_map</mat-icon>
      </button>
    }`,
  styles: [
    `
      :host {
        display: inline-flex;
        align-items: center;
        gap: 2px;
      }
      button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        padding: 0;
        border: 0;
        border-radius: 4px;
        background: transparent;
        color: var(--me-on-surface-color);
        cursor: pointer;
      }
      button.expand-action {
        width: 32px;
        height: 32px;
        color: var(--me-primary-color);
      }
      button.expand-action mat-icon {
        width: 20px;
        height: 20px;
        font-size: 20px;
      }
      button:hover:not(:disabled) {
        background: var(--me-surface-container-highest-color);
      }
      button:disabled {
        opacity: 0.35;
        cursor: default;
      }
      button:focus-visible {
        outline: 2px solid var(--me-primary-color);
      }
      mat-icon {
        width: 18px;
        height: 18px;
        font-size: 18px;
      }
    `,
  ],
})
export class AnalysisViewActions {
  readonly showReset = input(true);
  readonly range = input<{start: number; end: number} | null>(null);
  readonly full = input<{start: number; end: number} | null>(null);
  readonly expanded = input(false);
  readonly expandable = input(true);
  readonly resetView = output<void>();
  readonly expandView = output<void>();
  readonly changed = computed(
    () =>
      !!this.range() &&
      !!this.full() &&
      (this.range()!.start !== this.full()!.start || this.range()!.end !== this.full()!.end),
  );
}
