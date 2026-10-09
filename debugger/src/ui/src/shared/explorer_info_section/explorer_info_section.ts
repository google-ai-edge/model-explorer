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

/** Presentation adapted from Model Explorer info_panel (Apache-2.0). */
import {Component, input, model} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
@Component({
  selector: 'explorer-info-section',
  standalone: true,
  imports: [MatIconModule, MatButtonModule],
  template: ` <section [attr.aria-label]="label()">
    <header class="header">
      <button
        mat-icon-button
        class="toggle"
        [attr.aria-label]="(expanded() ? 'Collapse ' : 'Expand ') + label()"
        [attr.aria-expanded]="expanded()"
        (click)="expanded.set(!expanded())"
      >
        <mat-icon>{{ expanded() ? 'expand_more' : 'chevron_right' }}</mat-icon></button
      ><strong>{{ label() }}</strong
      ><span class="filler"></span><ng-content select="[section-action]" />
    </header>
    <div class="section-content" [hidden]="!expanded()"><ng-content /></div>
  </section>`,
  styles: `
    :host {
      display: block;
      font: 11px / normal var(--me-font-family);
      color: var(--me-on-surface-color);
      // Header: 12px padding, 24px toggle pulled 8px left, 2px gap: the title text starts at 30px.
      --info-indent: 30px;
    }
    section {
      padding-bottom: 8px;
      box-sizing: border-box;
    }
    .header {
      display: flex;
      align-items: center;
      font: 700 11px/24px var(--me-font-family);
      padding: 4px 12px 0;
      text-transform: uppercase;
      position: sticky;
      top: 0;
      z-index: 1;
      box-sizing: border-box;
      background: var(--me-surface-color);
      user-select: none;
    }
    strong {
      font: inherit;
    }
    .filler {
      flex-grow: 1;
    }
    button.toggle.mat-mdc-button-base {
      padding: 0;
      width: 24px;
      height: 24px;
      margin-left: -8px;
      margin-right: 2px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: inherit;
      border-radius: 50%;
    }
    button.toggle mat-icon {
      font-size: 20px;
      width: 20px;
      height: 20px;
    }
    button.toggle:hover {
      background: var(--me-surface-container-high-color);
    }
    button.toggle ::ng-deep .mat-mdc-button-touch-target {
      display: none;
    }
    .section-content[hidden] {
      display: none;
    }
  `,
})
export class ExplorerInfoSection {
  readonly label = input.required<string>();
  readonly expanded = model(true);
}
