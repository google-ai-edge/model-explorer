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

import {Component, input} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

/** Inspector locator: Model Explorer's small, secondary icon in a stable hit area. */
@Component({
  selector: 'button[explorer-locator], div[explorer-locator]',
  imports: [MatIconModule],
  host: {
    '[attr.type]': 'row() ? null : "button"',
    '[class.with-label]': '!!label() || row()',
    '[class.locator-row]': 'row()',
  },
  template: `@if (label()) {
      <span>{{ label() }}</span>
    }
    <ng-content />
    <mat-icon aria-hidden="true">my_location</mat-icon>`,
  styles: `
    :host {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 20px;
      width: 20px;
      height: 20px;
      box-sizing: border-box;
      padding: 0;
      border: 0;
      border-radius: 3px;
      background: transparent;
      color: var(--me-on-surface-variant-color);
      vertical-align: middle;
      cursor: pointer;
    }
    :host(.with-label) {
      flex: 0 1 auto;
      width: auto;
      height: auto;
      min-height: 20px;
      gap: 4px;
      color: var(--me-primary-color);
      font: 500 12px / normal var(--me-font-family);
      text-align: left;
    }
    :host(.locator-row) {
      flex: 1 1 0;
      min-width: 0;
      justify-content: flex-start;
      align-items: flex-start;
      line-height: 20px;
    }
    :host(.locator-row) mat-icon {
      margin-top: 4px;
    }
    :host(.with-label) mat-icon {
      color: var(--me-on-surface-variant-color);
      flex-shrink: 0;
    }
    mat-icon {
      width: 12px;
      height: 12px;
      font-size: 12px;
      line-height: 12px;
      opacity: 0.7;
    }
    :host(:hover),
    :host(:focus-visible) {
      background: var(--me-surface-container-high-color);
    }
    :host(:hover) mat-icon,
    :host(:focus-visible) mat-icon {
      opacity: 1;
    }
    :host(:focus-visible) {
      outline: 2px solid var(--me-primary-color);
      outline-offset: -2px;
    }
    :host(:disabled) {
      opacity: 0.4;
      cursor: default;
    }
  `,
})
export class ExplorerLocator {
  readonly label = input('');
  readonly row = input(false);
}
