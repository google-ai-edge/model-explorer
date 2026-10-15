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

import {animateAnalysisLocationFeedback} from '../../shared/locate_feedback';
import {PositionRangeControl} from './position_range_control';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  inject,
  input,
  output,
} from '@angular/core';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import type {HeatmapChoice, HeatmapControl} from './heatmap_model';
import {AggregationControl, type Aggregation, type AggregationChange} from './aggregation_control';
@Component({
  selector: 'analysis-heatmap-controls',
  imports: [ConfigPicker, PositionRangeControl, AggregationControl],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `@if (showRangeControls()) {
      <div class="range-group" role="group" aria-label="Token Range controls">
        <analysis-position-range
          label="Token Range"
          [range]="range()"
          [full]="fullRange()"
          (rangeChange)="rangeChange.emit($event)"
        />
      </div>
    }
    <label data-control="metric"
      >Metric<config-picker
        [attr.title]="label(metrics(), metric())"
        label="Analysis metric"
        [popupWidth]="240"
        [compact]="true"
        [value]="label(metrics(), metric())"
        [options]="labels(metrics())"
        (valueChange)="chooseMetric($event)"
    /></label>
    @if (showAggregation()) {
      <div
        class="dimension-group aggregation-group"
        [attr.data-dimension]="aggregation()?.dimension ?? 'none'"
      >
        <span>Aggregation</span>
        <analysis-aggregation
          [value]="aggregation()"
          (valueChange)="aggregationChange.emit($event)"
        />
      </div>
    }
    @if (showRangeControls()) {
      <div class="bin-group">
        @for (control of dimensions(); track control.id) {
          @if (control.id === 'position') {
            <label
              >{{ control.label
              }}<config-picker
                [label]="control.label"
                [popupWidth]="160"
                [disabled]="control.disabled ?? false"
                [attr.title]="
                  control.disabled ? 'Each cell contains one Position; no aggregation needed' : null
                "
                [compact]="true"
                [value]="control.disabled ? '-' : label(control.choices, control.value)"
                [options]="labels(control.choices)"
                (valueChange)="chooseDimension(control, $event)"
            /></label>
          }
        }
      </div>
    }`,
  styles: [
    `
      :host {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
        flex-wrap: wrap;
      }
      .range-group,
      .bin-group {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
      }
      .range-group {
        border-right: 1px solid var(--me-outline-hairline-color);
        padding-right: 12px;
      }
      label,
      .aggregation-group {
        display: flex;
        align-items: center;
        gap: 6px;
        font: 12px/20px var(--me-font-family);
        color: var(--me-on-surface-color);
      }
      config-picker {
        width: 160px;
        min-width: 160px;
        max-width: 160px;
        flex: 0 0 160px;
      }
      .bin-group config-picker {
        width: 80px;
        min-width: 80px;
        max-width: 80px;
        flex-basis: 80px;
      }
    `,
  ],
})
export class HeatmapControls {
  readonly showRangeControls = input(true);
  readonly showAggregation = input(true);
  readonly aggregation = input<Aggregation | null>(null);
  readonly aggregationChange = output<AggregationChange>();
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private locationAnimation?: Animation;
  constructor() {
    inject(DestroyRef).onDestroy(() => this.locationAnimation?.cancel());
  }
  locateMetric() {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLButtonElement>('[data-control="metric"] button'),
    );
  }
  locateDimension(id: string) {
    const group = Array.from(
      this.host.nativeElement.querySelectorAll<HTMLElement>('[data-dimension]'),
    ).find((element) => element.dataset['dimension'] === id);
    this.locateControl(group?.querySelector('button'));
  }
  private locateControl(target: HTMLButtonElement | null | undefined) {
    if (!target) return;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    target.scrollIntoView({
      block: 'center',
      inline: 'nearest',
      behavior: reducedMotion ? 'instant' : 'smooth',
    });
    this.locationAnimation?.cancel();
    this.locationAnimation = animateAnalysisLocationFeedback(target, reducedMotion);
  }
  readonly range = input<{start: number; end: number} | null>(null);
  readonly fullRange = input<{start: number; end: number} | null>(null);
  readonly rangeChange = output<{start: number; end: number}>();
  readonly metrics = input.required<readonly HeatmapChoice[]>();
  readonly metric = input.required<string>();
  readonly dimensions = input<readonly HeatmapControl[]>([]);
  readonly metricChange = output<string>();
  readonly dimensionChange = output<{id: string; value: string}>();
  labels(choices: readonly HeatmapChoice[]) {
    return choices.map((choice) => choice.label);
  }
  label(choices: readonly HeatmapChoice[], id: string) {
    return choices.find((choice) => choice.id === id)?.label ?? id;
  }
  chooseMetric(label: string) {
    const choice = this.metrics().find((choice) => choice.label === label);
    if (choice) this.metricChange.emit(choice.id);
  }
  chooseDimension(control: HeatmapControl, label: string) {
    const choice = control.choices.find((choice) => choice.label === label);
    if (choice) this.dimensionChange.emit({id: control.id, value: choice.id});
  }
}
