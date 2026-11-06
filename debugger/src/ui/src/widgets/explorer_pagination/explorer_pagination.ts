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

/** Compact pager shared by token pairs and tensor values. */
import {Component, input, output} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
@Component({
  selector: 'explorer-pagination',
  imports: [MatButtonModule, MatIconModule],
  template: ` <div role="group" [attr.aria-label]="label()">
    <button
      type="button"
      mat-icon-button
      [attr.aria-label]="previousLabel()"
      [disabled]="index() === 0"
      (click)="change.emit(-1)"
    >
      <mat-icon>chevron_left</mat-icon>
    </button>
    <span aria-live="polite">{{ index() + 1 }} / {{ pages() }}</span>
    <button
      type="button"
      mat-icon-button
      [attr.aria-label]="nextLabel()"
      [disabled]="index() + 1 >= pages()"
      (click)="change.emit(1)"
    >
      <mat-icon>chevron_right</mat-icon>
    </button>
  </div>`,
  styles: `
    :host {
      display: block;
    }
    div {
      display: flex;
      align-items: center;
      gap: 2px;
      color: var(--me-on-surface-variant-color);
      font: 400 11px/16px var(--me-font-family);
      white-space: nowrap;
    }
    button.mat-mdc-icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 16px;
      padding: 0 4px;
      --mdc-icon-button-state-layer-size: 24px;
    }
    mat-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
    }
  `,
})
export class ExplorerPagination {
  readonly index = input.required<number>();
  readonly pages = input.required<number>();
  readonly label = input.required<string>();
  readonly previousLabel = input.required<string>();
  readonly nextLabel = input.required<string>();
  readonly change = output<number>();
}
