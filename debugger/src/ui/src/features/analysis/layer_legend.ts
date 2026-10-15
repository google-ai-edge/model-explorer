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

import {ChangeDetectionStrategy, Component, computed, input, output, signal} from '@angular/core';
import {NgTemplateOutlet} from '@angular/common';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule} from '@angular/cdk/a11y';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';

export const layerColor = (layer: number) => `hsl(${(layer * 137.508 + 210) % 360} 62% 46%)`;

@Component({
  selector: 'analysis-layer-legend',
  imports: [
    NgTemplateOutlet,
    OverlayModule,
    A11yModule,
    MatButtonModule,
    MatIconModule,
    OverlayDialog,
    OverlayPanel,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: ` @if (inline()) {
      <section class="inline-picker" aria-label="Layers">
        <strong>Layers {{ selected().length }}/{{ layers().length }}</strong>
        <ng-container [ngTemplateOutlet]="choices" />
      </section>
    } @else {
      <button
        mat-button
        cdkOverlayOrigin
        #origin="cdkOverlayOrigin"
        aria-label="Layers"
        title="Layer legend: search and select visible curves"
        aria-haspopup="dialog"
        [attr.aria-expanded]="open()"
        (click)="query.set(''); open.set(!open())"
      >
        @if (compact()) {
          Layers {{ selected().length }}/{{ layers().length }}
        } @else {
          Layers · {{ selected().length }} / {{ layers().length }}
          <mat-icon iconPositionEnd>arrow_drop_down</mat-icon>
        }
      </button>

      <ng-template
        overlayDialog
        [origin]="origin"
        [open]="open()"
        [width]="260"
        [gap]="0"
        placement="below-end"
        (closed)="close()"
      >
        <section class="picker" overlayPanel aria-label="Layer legend">
          <strong>Layer legend</strong>
          <p class="help">Select layers to show. Hover a layer to highlight its curves.</p>
          <ng-container [ngTemplateOutlet]="choices" />
          <button mat-button (click)="close()">Done</button>
        </section>
      </ng-template>
    }
    <ng-template #choices>
      <input
        type="search"
        cdkFocusInitial
        aria-label="Search layers"
        placeholder="Search layers"
        [value]="query()"
        (input)="query.set($any($event.target).value)"
      />
      <div class="actions">
        <button mat-button (click)="reset()">Default layers</button>
        <button mat-button (click)="selectionChange.emit([])">Clear</button>
      </div>
      <div class="options" (mouseleave)="highlight.emit(null)">
        @for (layer of filtered(); track layer) {
          <label
            (mouseenter)="highlight.emit(layer)"
            (focusin)="highlight.emit(layer)"
            (focusout)="highlight.emit(null)"
          >
            <input
              type="checkbox"
              [attr.aria-label]="'Layer ' + layer"
              [checked]="selected().includes(layer)"
              (change)="toggle(layer)"
            />
            <i [style.background]="color(layer)"></i><span>Layer {{ layer }}</span>
          </label>
        } @empty {
          <p>No matching layers</p>
        }
      </div>
    </ng-template>`,
  styleUrl: './layer_legend.scss',
  host: {'[class.compact]': 'compact()', '[class.inline]': 'inline()'},
})
export class LayerLegend {
  readonly compact = input(false);
  readonly inline = input(false);
  readonly layers = input<readonly number[]>([]);
  readonly selected = input<readonly number[]>([]);
  readonly defaults = input<readonly number[]>([]);
  readonly selectionChange = output<number[]>();
  readonly highlight = output<number | null>();
  readonly open = signal(false);
  readonly query = signal('');
  readonly color = layerColor;
  readonly filtered = computed(() =>
    this.layers().filter((layer) =>
      `layer ${layer}`.includes(
        this.query()
          .trim()
          .toLowerCase()
          .replace(/^l(?=\d)/, 'layer '),
      ),
    ),
  );
  reset() {
    this.selectionChange.emit([...this.defaults()]);
  }
  toggle(layer: number) {
    this.selectionChange.emit(
      this.selected().includes(layer)
        ? this.selected().filter((id) => id !== layer)
        : [...this.selected(), layer].sort((a, b) => a - b),
    );
  }
  close() {
    this.open.set(false);
    this.highlight.emit(null);
  }
}
