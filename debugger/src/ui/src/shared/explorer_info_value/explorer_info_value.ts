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

/** ME hoverable_label overflow contract, using the existing Material tooltip/dialog. */
import {
  afterNextRender,
  Component,
  effect,
  ElementRef,
  inject,
  Injector,
  input,
  OnDestroy,
  signal,
  viewChild,
} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {MatTooltipModule} from '@angular/material/tooltip';
@Component({
  selector: 'explorer-info-value',
  standalone: true,
  imports: [MatTooltipModule],
  template: `
    <span
      #value
      class="value"
      [matTooltip]="text()"
      [matTooltipDisabled]="!truncated()"
      matTooltipClass="explorer-value-tooltip"
      tabindex="0"
      [attr.aria-label]="label() + ': ' + text()"
      (keydown.enter)="read()"
      >{{ text() }}</span
    >
    @if (truncated()) {
      <button
        type="button"
        class="read-value"
        [attr.aria-label]="'Read full ' + label()"
        (click)="read()"
      >
        …
      </button>
    }
  `,
  styles: `
    :host {
      display: flex;
      min-width: 0;
      align-items: center;
      gap: 2px;
      font: 12px/18px var(--me-font-family);
    }
    .value {
      display: block;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    .value:focus-visible {
      outline: 1px solid var(--me-primary-color);
      outline-offset: -1px;
    }
    button.read-value {
      flex: none;
      width: 18px;
      height: 18px;
      padding: 0;
      border: 0;
      border-radius: 3px;
      background: transparent;
      color: var(--me-primary-color);
      font: 12px/18px var(--me-font-family);
      cursor: pointer;
    }
    button:hover {
      background: var(--me-surface-container-low-color);
    }
    ::ng-deep .explorer-value-tooltip .mdc-tooltip__surface {
      padding: 2px;
      font: 12px/12px var(--me-font-family);
      border: 1px solid var(--me-outline-variant-color);
      border-radius: 4px;
      background: var(--me-surface-container-low-color);
      color: var(--me-on-surface-low-color);
      max-width: 360px;
      overflow-wrap: anywhere;
    }
  `,
})
export class ExplorerInfoValue implements OnDestroy {
  readonly value = input<string | number | null | undefined>();
  readonly label = input('value');
  readonly truncated = signal(false);
  readonly element = viewChild<ElementRef<HTMLElement>>('value');
  private readonly dialog = inject(MatDialog);
  private readonly injector = inject(Injector);
  private observer?: ResizeObserver;
  text() {
    return this.value() == null ? 'Not captured' : String(this.value());
  }
  private measure() {
    const e = this.element()?.nativeElement;
    if (e) this.truncated.set(e.scrollWidth > e.clientWidth);
  }
  constructor() {
    afterNextRender(() => {
      this.observer = new ResizeObserver(() => this.measure());
      this.observer.observe(this.element()!.nativeElement);
      this.measure();
    });
    effect(() => {
      this.value();
      afterNextRender(() => this.measure(), {injector: this.injector});
    });
  }
  async read() {
    if (!this.truncated()) return;
    const {FullTextDialog} = await import(
      '../full_text_dialog/full_text_dialog'
    );
    this.dialog.open(FullTextDialog, {
      width: '760px',
      maxWidth: 'calc(100vw - 24px)',
      data: {title: this.label(), text: this.text()},
      ariaLabel: 'Read full property value',
    });
  }
  ngOnDestroy() {
    this.observer?.disconnect();
  }
}
