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

import {Component, input, output, signal, computed, inject} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule, FocusMonitor} from '@angular/cdk/a11y';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MAT_TOOLTIP_DEFAULT_OPTIONS, MatTooltipModule} from '@angular/material/tooltip';
import {OverlayDialog, OverlayPanel} from '../overlay_dialog/overlay_dialog';
/** CDK positioning/focus used by the existing controls, with the prototype's model-search behavior. */
@Component({
  selector: 'config-picker',
  imports: [
    OverlayDialog,
    OverlayPanel,
    OverlayModule,
    A11yModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  providers: [
    {
      provide: MAT_TOOLTIP_DEFAULT_OPTIONS,
      useFactory: () => ({
        ...inject(MAT_TOOLTIP_DEFAULT_OPTIONS, {skipSelf: true}),
        position: 'right',
        disableTooltipInteractivity: true,
      }),
    },
  ],
  template: ` <button
      type="button"
      mat-button
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      [attr.aria-label]="label()"
      [matTooltip]="'Select ' + label()"
      matTooltipPosition="below"
      aria-haspopup="listbox"
      [attr.aria-expanded]="open()"
      [disabled]="disabled()"
      (keydown)="triggerKeydown($event)"
      (click)="query.set(''); open.set(!open())"
    >
      <span class="value">{{ displayValue() ?? value() }}</span
      ><mat-icon iconPositionEnd>arrow_drop_down</mat-icon>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      [width]="popupWidth() ?? origin.elementRef.nativeElement.getBoundingClientRect().width"
      [gap]="0"
      (closed)="open.set(false)"
      (overlayKeydown)="keydown($event)"
    >
      <section
        class="picker"
        [class.compact]="compact()"
        overlayPanel
        [attr.aria-label]="'Select ' + label()"
      >
        @if (searchable()) {
          <input
            type="search"
            cdkFocusInitial
            [attr.aria-label]="'Search ' + plural()"
            [placeholder]="'Search ' + plural()"
            [value]="query()"
            (input)="query.set($any($event.target).value)"
          />
        }
        <div role="listbox" [attr.aria-label]="plural()">
          @for (group of filteredGroups(); track group.label) {
            <div [attr.role]="group.label ? 'group' : null" [attr.aria-label]="group.label || null">
              @if (group.label) {
                <div class="option-group-label">{{ group.label }}</div>
              }
              @for (option of group.options; track option) {
                <span
                  class="option-hint"
                  [matTooltip]="
                    compact() && unavailable().includes(option)
                      ? (descriptions()[option] ?? '')
                      : ''
                  "
                  ><button
                    mat-button
                    type="button"
                    role="option"
                    [attr.aria-label]="optionLabels()[option] ? option : null"
                    [matTooltip]="
                      compact() && !unavailable().includes(option)
                        ? (descriptions()[option] ?? '')
                        : ''
                    "
                    [class.described]="
                      !compact() && !inlineDescriptions() && !!descriptions()[option]
                    "
                    [class.inline-description]="inlineDescriptions()"
                    [attr.aria-description]="compact() ? descriptions()[option] : null"
                    [disabled]="unavailable().includes(option)"
                    [attr.aria-selected]="option === value()"
                    [attr.cdkFocusInitial]="!searchable() && option === value() ? '' : null"
                    (click)="valueChange.emit(option); open.set(false)"
                  >
                    <span class="option-text"
                      ><span class="option-name">{{ optionLabels()[option] ?? option }}</span>
                      @if (!compact() && descriptions()[option]) {
                        <small>{{ descriptions()[option] }}</small>
                      }
                    </span>
                    @if (option === value()) {
                      <mat-icon iconPositionEnd>check</mat-icon>
                    }
                  </button></span
                >
              }
            </div>
          } @empty {
            <p>No matching {{ plural() }}</p>
          }
        </div>
      </section>
    </ng-template>`,
  styleUrl: './config_picker.scss',
})
export class ConfigPicker {
  private readonly focusMonitor = inject(FocusMonitor);
  readonly popupWidth = input<number>();
  readonly compact = input(false);
  readonly inlineDescriptions = input(false);
  readonly descriptions = input<Partial<Record<string, string>>>({});
  readonly unavailable = input<string[]>([]);
  readonly label = input.required<string>();
  readonly plural = input('options');
  readonly value = input.required<string>();
  readonly displayValue = input<string>();
  readonly options = input.required<string[]>();
  readonly groups = input<readonly {label: string; options: readonly string[]}[]>([]);
  readonly optionLabels = input<Partial<Record<string, string>>>({});
  readonly disabled = input(false);
  readonly searchable = input(false);
  readonly valueChange = output<string>();
  readonly open = signal(false);
  readonly query = signal('');
  readonly filtered = computed(() =>
    this.options().filter((o) => o.toLowerCase().includes(this.query().trim().toLowerCase())),
  );
  readonly filteredGroups = computed(() => {
    const options = this.filtered();
    if (!options.length) return [];
    if (!this.groups().length) return [{label: '', options}];
    return this.groups()
      .map((group) => ({
        label: group.label,
        options: options.filter((option) => group.options.includes(option)),
      }))
      .filter((group) => group.options.length);
  });
  triggerKeydown(event: KeyboardEvent) {
    if (event.isComposing) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      this.query.set('');
      this.open.set(true);
    }
  }
  keydown(event: KeyboardEvent) {
    if (event.isComposing) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const panel = (event.target as HTMLElement).closest('.picker');
      const options = Array.from(
        panel?.querySelectorAll<HTMLButtonElement>('[role=option]:not(:disabled)') ?? [],
      );
      const index = options.indexOf(event.target as HTMLButtonElement);
      const next =
        index < 0
          ? event.key === 'ArrowDown'
            ? 0
            : options.length - 1
          : (index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
      if (options[next]) this.focusMonitor.focusVia(options[next], 'keyboard');
    } else if (event.key === 'Tab') this.open.set(false);
  }
}
