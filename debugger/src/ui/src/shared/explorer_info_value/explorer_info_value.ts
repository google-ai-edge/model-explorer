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

/** Model Explorer overflow presentation: expandable_info_text inline, hoverable_label dialog. */
import {
  Component,
  computed,
  forwardRef,
  input,
  signal,
  viewChild,
  ElementRef,
  afterNextRender,
  OnDestroy,
  inject,
  effect,
  Injector,
  TemplateRef,
} from '@angular/core';
import {NgTemplateOutlet} from '@angular/common';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ExplorerHelpTerm, type ExplorerValueSegment} from './explorer_help_term';
import {MatDialog} from '@angular/material/dialog';
@Component({
  selector: 'explorer-info-value',
  standalone: true,
  imports: [MatTooltipModule, MatIconModule, NgTemplateOutlet, forwardRef(() => ExplorerHelpTerm)],
  host: {
    '[class.inline]': 'inline()',
    '[class.expanded]': 'expanded()',
    '[class.has-overflow]': 'truncated()',
    '[class.reserve-action]': 'reserveAction() && !(inline() && textAction())',
    '[class.text-action]': 'inline() && textAction()',
    '[class.wrap]': 'wrap()',
    '(click)': 'expandInline($event)',
  },
  template: `
    <ng-template #segmentedValue>
      @for (segment of segments(); track $index) {
        @if (segment.help) {
          <explorer-help-term [text]="segment.text" [help]="segment.help" />
        } @else {
          <ng-container>{{ segment.text }}</ng-container>
        }
      }
    </ng-template>
    <span
      #value
      class="value"
      [matTooltip]="text()"
      [matTooltipDisabled]="inline() || !truncated()"
      matTooltipClass="explorer-value-tooltip"
      [attr.tabindex]="inline() ? (textActionAvailable() ? 0 : null) : 0"
      [attr.role]="textActionAvailable() ? 'button' : null"
      [attr.aria-expanded]="textActionAvailable() ? expanded() : null"
      [attr.aria-label]="label() + ': ' + text()"
      (keydown.enter)="valueKey($event)"
      (keydown.space)="valueKey($event)"
    >
      @if (contentTemplate(); as content) {
        <ng-container [ngTemplateOutlet]="content" />
      } @else if (segments().length) {
        <ng-container [ngTemplateOutlet]="segmentedValue" />
      } @else {
        {{ text() }}
      }
    </span>
    @if (inline() && expanded() && !textAction()) {
      @if (contentTemplate(); as content) {
        <span class="expanded-value"><ng-container [ngTemplateOutlet]="content" /></span>
      } @else if (segments().length) {
        <span class="expanded-value"><ng-container [ngTemplateOutlet]="segmentedValue" /></span>
      } @else {
        <span class="expanded-value" [textContent]="text()"></span>
      }
    }
    @if ((truncated() || expanded()) && !(inline() && textAction())) {
      <button
        type="button"
        class="read-value"
        [attr.aria-label]="
          inline() ? (expanded() ? 'Collapse ' : 'Expand ') + label() : 'Read full ' + label()
        "
        [attr.aria-expanded]="inline() ? expanded() : null"
        (click)="read($event)"
      >
        @if (inline()) {
          <mat-icon aria-hidden="true">{{ expanded() ? 'unfold_less' : 'unfold_more' }}</mat-icon>
        } @else {
          …
        }
      </button>
    }
  `,
  styles: `
    :host {
      display: flex;
      min-width: 0;
      align-items: center;
      gap: 2px;
      font: inherit;
    }
    .value {
      display: block;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    :host(.inline) {
      position: relative;
      align-items: flex-start;
    }
    :host(.wrap) .value {
      white-space: normal;
      overflow: visible;
      overflow-wrap: anywhere;
    }
    :host(.expanded) .value {
      position: absolute;
      inset: 0;
      visibility: hidden;
    }
    :host(.inline.text-action.expanded) .value {
      position: static;
      inset: auto;
      visibility: visible;
      white-space: pre-wrap;
      overflow: visible;
      overflow-wrap: anywhere;
      cursor: pointer;
    }
    .expanded-value {
      flex: 1;
      min-width: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      cursor: text;
    }
    // Adapted from Model Explorer expandable_info_text.scss (Apache-2.0).
    :host(.inline.has-overflow:not(.expanded)) .value {
      cursor: pointer;
    }
    :host(.inline) button.read-value {
      display: flex;
      align-items: center;
      justify-content: center;
      box-sizing: content-box;
      width: 12px;
      height: 12px;
      border: 1px solid var(--me-on-secondary-container-color);
      border-radius: 2px;
      background: var(--me-secondary-container-color);
      opacity: 0.75;
    }
    :host(.inline) button.read-value mat-icon {
      color: var(--me-primary-color);
      font-size: 11px;
      width: 11px;
      height: 11px;
      font-weight: 700;
    }
    :host(.inline:hover) button.read-value,
    :host(.inline) button.read-value:focus-visible {
      opacity: 1;
    }
    :host(.inline.expanded) button.read-value {
      background: var(--me-surface-container-high-color);
      border-color: var(--me-outline-color);
    }
    :host(.inline.expanded) button.read-value mat-icon {
      color: var(--me-on-surface-variant-color);
    }
    // Inspector headers reserve the same action column for sorting/expansion.
    :host(.inline.reserve-action) {
      display: grid;
      grid-template-columns: minmax(0, 1fr) var(--info-table-action-size, 20px);
      gap: var(--info-table-action-gap, 4px);
    }
    :host(.inline.reserve-action) .value,
    :host(.inline.reserve-action) .expanded-value {
      grid-column: 1;
    }
    :host(.inline.reserve-action) button.read-value {
      grid-column: 2;
      justify-self: center;
    }
    .read-value:focus-visible {
      outline: 2px solid var(--me-primary-color);
    }
    .value:focus-visible {
      outline: 1px solid var(--me-primary-color);
      outline-offset: -1px;
    }
    button.read-value {
      flex: none;
      width: 18px;
      height: 16px;
      padding: 0;
      border: 0;
      border-radius: 3px;
      background: transparent;
      color: var(--me-primary-color);
      font: inherit;
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
  readonly segments = input<readonly ExplorerValueSegment[]>([]);
  /** Optional formatted value; the plain value remains the accessible text and dialog fallback. */
  readonly contentTemplate = input<TemplateRef<unknown> | null>(null);
  readonly inline = input(false);
  readonly wrap = input(false);
  readonly reserveAction = input(false);
  /** Inline statistics toggle on their text, leaving their action column for locating. */
  readonly textAction = input(false);
  readonly expanded = signal(false);
  readonly truncated = signal(false);
  readonly textActionAvailable = computed(
    () => this.inline() && this.textAction() && (this.truncated() || this.expanded()),
  );
  readonly element = viewChild<ElementRef<HTMLElement>>('value');
  private readonly dialog = inject(MatDialog);
  private readonly injector = inject(Injector);
  private observer?: ResizeObserver;
  text() {
    return this.value() == null ? 'Not captured' : String(this.value());
  }
  private measure() {
    if (this.wrap()) {
      this.truncated.set(false);
      return;
    }
    const e = this.element()?.nativeElement;
    if (this.expanded()) return;
    if (e) {
      // Inline toggles consume width themselves. Test the full value slot so a
      // previous toggle cannot keep a value truncated after the panel grows.
      if (this.inline()) {
        const textRange = document.createRange();
        textRange.selectNodeContents(e);
        // scrollWidth/clientWidth round to integers and can miss subpixel clipping.
        this.truncated.set(
          textRange.getBoundingClientRect().width >
            (this.reserveAction()
              ? e.getBoundingClientRect().width
              : e.parentElement!.getBoundingClientRect().width),
        );
      } else {
        this.truncated.set(e.scrollWidth > e.clientWidth);
      }
    }
  }
  constructor() {
    afterNextRender(() => {
      this.observer = new ResizeObserver(() => this.measure());
      this.observer.observe(this.element()!.nativeElement);
      this.measure();
    });
    effect(() => {
      this.value();
      this.wrap();
      this.expanded.set(false);
      afterNextRender(() => this.measure(), {injector: this.injector});
    });
  }
  expandInline(event: MouseEvent) {
    if (this.inline() && this.textAction()) {
      if (this.textActionAvailable()) void this.read(event);
      return;
    }
    if (!this.inline() || !this.truncated() || this.expanded()) return;
    event.stopPropagation();
    this.expanded.set(true);
  }
  valueKey(event: Event) {
    if (this.inline() && this.textAction()) {
      if (!this.textActionAvailable()) return;
      event.preventDefault();
      void this.read(event);
    } else if ((event as KeyboardEvent).key === 'Enter') void this.read();
  }
  async read(event?: Event) {
    event?.stopPropagation();
    if (this.inline()) {
      this.expanded.update((value) => !value);
      afterNextRender(() => this.measure(), {injector: this.injector});
      return;
    }
    if (!this.truncated()) return;
    const {FullTextDialog} = await import('../full_text_dialog/full_text_dialog');
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
