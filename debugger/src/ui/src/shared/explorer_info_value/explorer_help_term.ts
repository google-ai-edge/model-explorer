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

import {Component, Injectable, OnDestroy, forwardRef, inject, input, signal} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerInfoValue} from './explorer_info_value';

export interface ExplorerHelp {
  title: string;
  text?: string;
  rows: readonly {label: string; value: string}[];
}
export interface ExplorerValueSegment {
  text: string;
  help?: ExplorerHelp;
}

/** Explanations across the inspector share a single open overlay. */
@Injectable({providedIn: 'root'})
class ExplorerHelpCoordinator {
  active?: ExplorerHelpTerm;
  activate(term: ExplorerHelpTerm) {
    if (this.active !== term) this.active?.dismiss();
    this.active = term;
  }
}
let nextHelpId = 0;

@Component({
  selector: 'explorer-help-term',
  imports: [OverlayModule, MatIconModule, forwardRef(() => ExplorerInfoValue)],
  template: `
    <button
      type="button"
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      [attr.aria-label]="text()"
      [attr.aria-describedby]="open() ? helpId : null"
      (mouseenter)="show()"
      (mouseleave)="scheduleClose()"
      (focus)="show()"
      (blur)="scheduleClose()"
      (click)="toggle($event)"
      (keydown.escape)="dismiss($event)"
      (keydown.enter)="$event.stopPropagation()"
    >
      <span class="term-label" [textContent]="text()"></span
      ><mat-icon aria-hidden="true">help_outline</mat-icon>
    </button>
    <ng-template
      cdkConnectedOverlay
      [cdkConnectedOverlayOrigin]="origin"
      [cdkConnectedOverlayOpen]="open()"
      [cdkConnectedOverlayOffsetY]="4"
      [cdkConnectedOverlayViewportMargin]="8"
      [cdkConnectedOverlayPush]="true"
      (overlayOutsideClick)="dismiss()"
      (overlayKeydown)="onOverlayKey($event)"
    >
      <div
        class="explanation"
        [attr.role]="pinned() ? 'dialog' : 'tooltip'"
        [attr.aria-label]="help().title"
        [id]="helpId"
        (mouseenter)="cancelClose()"
        (mouseleave)="scheduleClose()"
        (focusin)="cancelClose()"
        (focusout)="scheduleClose()"
      >
        <div class="title">{{ help().title }}</div>
        @if (help().text !== undefined) {
          <div class="body" [textContent]="help().text"></div>
        }
        @if (help().rows.length) {
          <dl>
            @for (row of help().rows; track row.label) {
              <dt>{{ row.label }}</dt>
              <dd>
                <explorer-info-value [label]="row.label" [value]="row.value" [inline]="true" />
              </dd>
            }
          </dl>
        }
      </div>
    </ng-template>
  `,
  styles: `
    :host {
      display: inline;
    }
    button {
      appearance: none;
      padding: 0;
      margin: 0;
      border: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      line-height: inherit;
      text-align: inherit;
      cursor: help;
      display: inline-flex;
      align-items: center;
      gap: var(--explorer-help-gap, 2px);
      vertical-align: baseline;
    }
    .term-label {
      text-decoration-line: underline;
      text-decoration-style: dotted;
      text-decoration-color: var(--me-on-surface-variant-color);
      text-decoration-thickness: 1px;
      text-underline-offset: 2px;
    }
    button mat-icon {
      flex: 0 0 var(--explorer-help-icon-size, 14px);
      width: var(--explorer-help-icon-size, 14px);
      height: var(--explorer-help-icon-size, 14px);
      font-size: var(--explorer-help-icon-size, 14px);
      line-height: var(--explorer-help-icon-size, 14px);
      color: var(--me-on-surface-low-color);
      opacity: 1;
    }
    button:focus-visible {
      outline: 1px solid var(--me-primary-color);
      outline-offset: 1px;
    }
    .explanation {
      box-sizing: border-box;
      width: 300px;
      max-width: calc(100vw - 32px);
      padding: 8px 10px;
      border: 1px solid var(--me-outline-variant-color);
      border-radius: 4px;
      background: var(--me-surface-container-low-color);
      color: var(--me-on-surface-color);
      font: 12px/1.5 var(--me-font-family);
      white-space: normal;
    }
    .title {
      font-weight: 500;
      margin-bottom: 6px;
    }
    .body {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .body + dl {
      margin-top: 6px;
    }
    dl {
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 4px 8px;
      margin: 0;
    }
    dt {
      color: var(--me-on-surface-variant-color);
    }
    dd {
      min-width: 0;
      margin: 0;
      align-self: center;
    }
  `,
})
export class ExplorerHelpTerm implements OnDestroy {
  readonly text = input.required<string>();
  readonly help = input.required<ExplorerHelp>();
  readonly helpId = `explorer-help-${nextHelpId++}`;
  readonly open = signal(false);
  readonly pinned = signal(false);
  private readonly coordinator = inject(ExplorerHelpCoordinator);
  private closeTimer?: ReturnType<typeof setTimeout>;
  show() {
    this.cancelClose();
    this.coordinator.activate(this);
    this.open.set(true);
  }
  cancelClose() {
    clearTimeout(this.closeTimer);
  }
  scheduleClose() {
    this.cancelClose();
    this.closeTimer = setTimeout(() => {
      if (!this.pinned()) this.dismiss();
    }, 150);
  }
  toggle(event: Event) {
    event.stopPropagation();
    if (this.pinned()) this.dismiss();
    else {
      this.show();
      this.pinned.set(true);
    }
  }
  dismiss(event?: Event) {
    event?.stopPropagation();
    this.cancelClose();
    this.open.set(false);
    this.pinned.set(false);
    if (this.coordinator.active === this) this.coordinator.active = undefined;
  }
  onOverlayKey(event: KeyboardEvent) {
    if (event.key === 'Escape') this.dismiss(event);
  }
  ngOnDestroy() {
    this.dismiss();
  }
}
