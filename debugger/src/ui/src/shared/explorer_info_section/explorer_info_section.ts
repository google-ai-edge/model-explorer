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
import {MatIconModule} from '@angular/material/icon';
import {MatButtonModule} from '@angular/material/button';
@Component({
  selector: 'explorer-info-section',
  standalone: true,
  host: {'[class.compact]': 'compact()'},
  imports: [MatIconModule, MatButtonModule],
  template: ` <section [attr.aria-label]="label()">
    <header class="header">
      @if (rowToggle() && collapsible()) {
        <button
          class="row-toggle-icon"
          tabindex="-1"
          [attr.aria-label]="'Toggle ' + label()"
          [attr.aria-expanded]="expanded()"
          (click)="expanded.set(!expanded())"
        >
          <mat-icon>{{ expanded() ? 'expand_more' : 'chevron_right' }}</mat-icon>
        </button>
        <button
          class="disclosure-row"
          [attr.aria-label]="(expanded() ? 'Collapse ' : 'Expand ') + label()"
          [attr.aria-expanded]="expanded()"
          (click)="expanded.set(!expanded())"
        >
          <strong>{{ label() }}{{ count() === null ? '' : ' (' + count() + ')' }}</strong>
        </button>
      } @else {
        @if (collapsible()) {
          <button
            mat-icon-button
            class="toggle"
            [attr.aria-label]="(expanded() ? 'Collapse ' : 'Expand ') + label()"
            [attr.aria-expanded]="expanded()"
            (click)="expanded.set(!expanded())"
          >
            <mat-icon>{{ expanded() ? 'expand_more' : 'chevron_right' }}</mat-icon>
          </button>
        } @else {
          <span class="toggle-placeholder"></span>
        }
        <strong>{{ label() }}{{ count() === null ? '' : ' (' + count() + ')' }}</strong>
        <span class="filler"></span>
      }
      <ng-content select="[section-action]" />
    </header>
    <div class="section-content items-container" [hidden]="!expanded()"><ng-content /></div>
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
    .toggle-placeholder {
      width: 24px;
      margin-left: -8px;
      margin-right: 2px;
    }
    .filler {
      flex: var(--info-action-spacer, 1 1 auto);
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
    :host(.compact) {
      --info-indent: 24px;
    }
    :host(.compact) section {
      padding-bottom: 0;
    }
    :host(.compact) .header {
      position: static;
      padding: 0;
      font: var(--info-heading-weight, 400) var(--info-heading-size, 12px) /
        var(--info-compact-row-height, 24px) var(--me-font-family);
      color: var(--info-heading-color, inherit);
      text-transform: none;
    }
    :host(.compact) button.toggle.mat-mdc-button-base {
      width: var(--info-toggle-width, 24px);
      height: var(--info-compact-row-height, 24px);
      margin-left: 0;
      margin-right: var(--info-toggle-gap, 0px);
      transform: translateX(var(--info-toggle-offset, 0px));
    }
    :host(.compact) button.toggle mat-icon {
      font-size: var(--info-toggle-icon-size, 16px);
      width: var(--info-toggle-icon-size, 16px);
      height: var(--info-toggle-icon-size, 16px);
    }
    // Model Explorer Inputs: the label and trailing space are the locator row;
    // the leading index/icon gutter remains outside its rectangular hover surface.
    .disclosure-row {
      display: flex;
      align-items: center;
      flex: 1;
      min-width: 0;
      min-height: var(--info-compact-row-height, 20px);
      margin-right: 16px;
      border: 0;
      padding: 0;
      border-radius: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      line-height: var(--info-compact-row-height, 20px);
      text-align: left;
      cursor: pointer;
    }
    .disclosure-row:hover {
      background: var(--me-surface-container-high-color);
    }
    .disclosure-row:focus-visible {
      outline: 2px solid var(--me-primary-color);
      outline-offset: -2px;
    }
    .row-toggle-icon {
      flex: 0 0 var(--info-toggle-width, 24px);
      height: var(--info-compact-row-height, 24px);
      margin-right: var(--info-toggle-gap, 0px);
      display: flex;
      align-items: center;
      justify-content: center;
      border: 0;
      padding: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
    }
    .row-toggle-icon mat-icon {
      font-size: var(--info-toggle-icon-size, 16px);
      width: var(--info-toggle-icon-size, 16px);
      height: var(--info-toggle-icon-size, 16px);
      transform: translateX(var(--info-toggle-offset, 0px));
    }
    // Copied from Model Explorer info_panel.scss (.items-container), Apache-2.0.
    // Keep the upstream content spacing here for every inspector section.
    .items-container {
      display: flex;
      flex-direction: column;
      gap: var(--info-item-gap, 4px);
      margin-top: var(--info-content-spacing, 10px);
      margin-bottom: var(--info-content-bottom-spacing, 6px);
    }
    .section-content[hidden] {
      display: none;
    }
  `,
})
export class ExplorerInfoSection {
  readonly label = input.required<string>();
  readonly count = input<number | null>(null);
  readonly expanded = model(true);
  readonly collapsible = input(true);
  /** Nested disclosures share the same state, keyboard and icon implementation. */
  readonly compact = input(false);
  readonly rowToggle = input(false);
}
