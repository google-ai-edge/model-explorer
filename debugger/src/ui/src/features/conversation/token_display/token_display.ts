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

import {OverlayModule} from '@angular/cdk/overlay';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
  signal,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {TokenAnalysisPair} from '../../../data/contracts/token_analysis';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
import {tokenMarkup} from '../conversation_content';
import {
  TokenColor,
  colorLabel,
  heatLevel,
  heatPalette,
  metricText,
} from '../token_analysis';
import {TOKEN_METRICS} from '../token_metric_metadata';
export interface TokenDisplayPreview {
  ref: string;
  target: string;
  match: boolean | null;
  analysis?: TokenAnalysisPair;
}
export interface TokenDisplaySettings {
  color: TokenColor;
  boundaries: boolean;
}
@Component({
  selector: 'token-display',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    OverlayDialog,
    OverlayPanel,
    MatTooltipModule,
    OverlayModule,
    MatButtonModule,
    MatIconModule,
    FormsModule,
    ConfigPicker,
  ],
  template: ` <button
      mat-icon-button
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      aria-label="Display"
      matTooltip="Display"
      [attr.aria-expanded]="open()"
      (click)="show()"
    >
      <mat-icon>visibility</mat-icon>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      placement="above-end"
      [inset]="-4"
      (closed)="close()"
    >
      <section overlayPanel class="settings-panel" aria-label="Display settings">
        <header>
          <strong>Display</strong
          ><button mat-icon-button aria-label="Close display settings" (click)="close()">
            <mat-icon>close</mat-icon>
          </button>
        </header>
        <div class="preview" aria-label="Captured token preview">
          <strong>Preview</strong>
          @if (preview(); as sample) {
            <div class="preview-pair">
              @for (side of ['Ref', 'Target']; track side) {
                <div [class.target]="side === 'Target'">
                  <span class="side-label">{{ side }}</span>
                  <div class="preview-tokens" [class.boundaries]="draft().boundaries">
                    <span
                      [class.mismatch]="draft().color === 'token_match' && sample.match === false"
                      [class.numeric-token]="
                        side === 'Target' && heatLevel(draft().color, sample.analysis) !== null
                      "
                      [class.numeric-missing]="
                        side === 'Target' &&
                        numeric() &&
                        heatLevel(draft().color, sample.analysis) === null
                      "
                      [attr.data-palette]="heatPalette(draft().color)"
                      [style.--heat-level]="heatLevel(draft().color, sample.analysis)"
                      [innerHTML]="previewHtml(side === 'Ref' ? sample.ref : sample.target)"
                    ></span>
                  </div>
                </div>
              }
            </div>
            <div class="preview-legend">
              {{
                numeric()
                  ? previewMetric()
                  : sample.match === null
                    ? 'Comparison unavailable'
                    : sample.match
                      ? 'Match'
                      : 'Mismatch'
              }}
            </div>
          } @else {
            <p class="preview-empty">No captured tokens to preview.</p>
          }
        </div>
        <div class="setting">
          <label>Color by</label
          ><config-picker
            label="Color by"
            plural="color metrics"
            [compact]="true"
            [popupWidth]="360"
            [value]="colorLabel(draft().color)"
            [options]="colorOptions"
            [unavailable]="unavailableMetrics()"
            [descriptions]="metricDescriptions()"
            (valueChange)="setColor($event)"
          /><button
            mat-button
            [disabled]="draft().color === 'token_match'"
            aria-label="Reset Color by to default"
            (click)="resetColor()"
          >
            Reset to default
          </button>
        </div>
        <div class="setting">
          <label
            ><input
              type="checkbox"
              [ngModel]="draft().boundaries"
              (ngModelChange)="setBoundaries($event)"
            />Token outlines</label
          ><button
            mat-button
            [disabled]="!draft().boundaries"
            aria-label="Reset Token outlines to default"
            (click)="setBoundaries(false)"
          >
            Reset to default
          </button>
        </div>
        <footer>
          <span role="status" class="sr-only">{{
            dirty() ? 'Unapplied changes' : 'Display settings applied'
          }}</span
          ><button mat-flat-button color="primary" [disabled]="!dirty()" (click)="apply()">
            Apply
          </button>
        </footer>
      </section>
    </ng-template>`,
  styleUrl: './token_display.scss',
})
export class TokenDisplay {
  readonly preview = input<TokenDisplayPreview | null>(null);
  previewHtml(text: string) {
    return tokenMarkup(text);
  }
  readonly settings = input.required<TokenDisplaySettings>();
  readonly settingsChange = output<TokenDisplaySettings>();
  readonly open = signal(false);
  readonly draft = signal<TokenDisplaySettings>({
    color: 'token_match',
    boundaries: false,
  });
  readonly colorLabel = colorLabel;
  readonly heatLevel = heatLevel;
  readonly heatPalette = heatPalette;
  readonly available = input<Record<string, boolean>>({});
  readonly unavailableMetrics = computed(() =>
    TOKEN_METRICS.filter((m) => !this.available()[m.key]).map((m) => m.label),
  );
  readonly colorOptions = [
    'Token · Match',
    'None',
    ...TOKEN_METRICS.map((m) => m.label),
  ];
  readonly metricDescriptions = computed(() =>
    Object.fromEntries(
      TOKEN_METRICS.map((m) => [
        m.label,
        this.available()[m.key]
          ? m.description + ' ' + m.unit
          : 'Not captured for these tokens',
      ]),
    ),
  );
  setColor(label: string) {
    const color =
      label === 'None'
        ? 'none'
        : (TOKEN_METRICS.find((m) => m.label === label)?.key ?? 'token_match');
    this.draft.update((current) => ({...current, color}));
  }
  resetColor() {
    this.draft.update((current) => ({...current, color: 'token_match'}));
  }
  setBoundaries(boundaries: boolean) {
    this.draft.update((current) => ({...current, boundaries}));
  }
  numeric() {
    const color = this.draft().color;
    return color !== 'none' && color !== 'token_match';
  }
  previewMetric() {
    const color = this.draft().color;
    return color === 'none' || color === 'token_match'
      ? ''
      : colorLabel(color) +
          ': ' +
          metricText(this.preview()?.analysis?.metrics[color], color);
  }
  show() {
    this.draft.set({...this.settings()});
    this.open.set(true);
  }
  close() {
    this.open.set(false);
  }
  dirty() {
    const s = this.settings();
    const d = this.draft();
    return s.color !== d.color || s.boundaries !== d.boundaries;
  }
  apply() {
    this.settingsChange.emit({...this.draft()});
  }
}
