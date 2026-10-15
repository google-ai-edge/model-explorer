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

import {
  Component,
  DestroyRef,
  ElementRef,
  computed,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MatIconModule} from '@angular/material/icon';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../shared/locate_feedback';
import {PositionRangeControl} from './position_range_control';
import {AnalysisViewActions} from './view_actions';
import {KvRangeChart} from '../kv/kv_panel/kv_range_chart';
import {tensorBins} from './position_tensor_model';
import {reduceMetricValues, type HeatmapReduction} from './heatmap_model';
import type {KvRangeBin, KvRangeRow, KvRangeMetricSummary} from '../../data/contracts/kv';
import type {KvRangeSelection} from '../kv/kv_panel/kv_range_types';
import type {TokenTrendState} from '../conversation/token_trends';
import {representativePosition} from '../conversation/analysis_token_strip/analysis_token_model';
import {AnalysisRuntimeLegend} from './runtime_legend';
import type {AnalysisRuntime} from './runtime_series';
export interface LogitPositionSeries {
  name: string;
  runtime?: AnalysisRuntime;
  legendLabel?: string;
  color: string;
  points: {position: number; index: number; value: number | null}[];
}
@Component({
  selector: 'final-logits-trends',
  imports: [
    MatTooltipModule,
    MatIconModule,
    MatButtonModule,
    OverlayModule,
    OverlayDialog,
    OverlayPanel,
    ConfigPicker,
    PositionRangeControl,
    AnalysisViewActions,
    KvRangeChart,
    AnalysisRuntimeLegend,
  ],
  template: `<div class="controls" role="toolbar" [attr.aria-label]="analysisLabel() + ' controls'">
      <div class="toolbar-scroll">
        <div class="toolbar-view">
          <span>View</span>
          <config-picker
            class="view-picker"
            [attr.title]="visualizationLabel()"
            label="Visualization"
            plural="visualizations"
            [value]="visualizationLabel()"
            [options]="visualizationOptions()"
            [groups]="visualizationGroups()"
            [optionLabels]="visualizationLabels()"
            [compact]="true"
            [searchable]="false"
            (valueChange)="chooseView($event)"
          />
        </div>
        <span class="tool-divider" aria-hidden="true"></span>
        <div class="toolbar-range" role="group" aria-label="Token Range controls">
          <analysis-position-range
            label="Token Range"
            [range]="range()"
            [full]="fullRange()"
            (rangeChange)="zoom($event)"
          />
          <analysis-view-actions
            [range]="range()"
            [full]="fullRange()"
            [expandable]="false"
            (resetView)="zoom(fullRange())"
          />
        </div>
        <span class="tool-divider" aria-hidden="true"></span>
        <label class="toolbar-metric"
          >Metric
          <config-picker
            label="Analysis metric"
            plural="metrics"
            [popupWidth]="240"
            [compact]="true"
            [value]="metricChoiceLabel()"
            [options]="metricOptions()"
            (valueChange)="chooseMetric($event)"
          />
        </label>
      </div>
      <div class="toolbar-actions">
        <div class="toolbar-legend">
          @if (runtimeSeries().length) {
            <analysis-runtime-legend [runtimes]="runtimeSeries()" />
          } @else {
            <span class="series-legend" aria-label="Metric series legend">
              @for (s of positionSeries(); track s.name) {
                <span [matTooltip]="s.name" [attr.aria-label]="s.name"
                  ><i [style.background]="s.color"></i>{{ s.legendLabel ?? s.name }}</span
                >
              }
            </span>
          }
        </div>
        <button
          mat-icon-button
          cdkOverlayOrigin
          #displayOrigin="cdkOverlayOrigin"
          class="view-display-button"
          aria-label="Display"
          [matTooltip]="
            'Display · Position bins: ' + (binsDisabled() ? '-' : reductionLabels[reduction()])
          "
          [attr.aria-expanded]="displayOpen()"
          (click)="toggleDisplay()"
        >
          <mat-icon>visibility</mat-icon>
        </button>
        @if (!expanded()) {
          <analysis-view-actions
            class="panel-expand"
            [showReset]="false"
            [expanded]="expanded()"
            (expandView)="expand.emit()"
          />
        }
      </div>
    </div>
    <ng-template
      overlayDialog
      [origin]="displayOrigin"
      [open]="displayOpen()"
      placement="below-end"
      [width]="360"
      (closed)="displayOpen.set(false)"
    >
      <div class="chart-display">
        <section class="settings-panel" overlayPanel aria-label="Chart display">
          <header>
            <strong>Display</strong>
            <button
              mat-icon-button
              aria-label="Close chart display"
              (click)="displayOpen.set(false)"
            >
              <mat-icon>close</mat-icon>
            </button>
          </header>
          <div class="setting">
            <label>Position bins</label>
            <config-picker
              label="Position bins"
              plural="functions"
              [compact]="true"
              [searchable]="false"
              [options]="reductionOptions"
              [value]="binsDisabled() ? '-' : reductionLabels[displayDraft()]"
              [disabled]="binsDisabled()"
              [attr.title]="
                binsDisabled() ? 'Each bin contains one position; no reduction needed' : null
              "
              (valueChange)="draftReduction($event)"
            />
            <button
              mat-button
              [disabled]="binsDisabled() || displayDraft() === 'max'"
              aria-label="Reset Position bins to default"
              (click)="displayDraft.set('max')"
            >
              Reset to default
            </button>
          </div>
          <footer>
            <span role="status" class="sr-only">{{
              displayDirty() ? 'Unapplied changes' : 'Display settings applied'
            }}</span>
            <button
              mat-flat-button
              color="primary"
              [disabled]="!displayDirty()"
              (click)="applyDisplay()"
            >
              Apply
            </button>
          </footer>
        </section>
      </div>
    </ng-template>
    @if (divergenceLabel()) {
      <p class="metric-note" role="note" aria-label="Softmax before divergence">
        Note: {{ divergenceLabel() }} is computed after applying softmax (T = 1) to the full Ref and
        Target logits.
      </p>
    }
    <div class="chart" [style.height]="expanded() ? 'calc(100dvh - 294px)' : '320px'">
      <kv-range-chart
        [analysisLabel]="analysisLabel()"
        kind="logits"
        [showLayers]="false"
        view="trends"
        metric="max_abs"
        [metricLabel]="metricLabel()"
        [metricPercent]="percent()"
        [format]="format"
        [layers]="layers()"
        [bins]="bins()"
        [visibleRange]="range()"
        [fullRange]="fullRange()"
        [rows]="chartRows()"
        [overview]="overview()"
        [series]="series()"
        [positionReduction]="reduction()"
        [selected]="selection()"
        (select)="select($event)"
        (backgroundClick)="backgroundClick.emit($event)"
        (positionHover)="positionHover.emit($event)"
        (zoom)="zoom($event)"
        (capacity)="capacity.set($event)"
      />
    </div>`,
  styleUrl: './final_logits_trends.scss',
})
export class FinalLogitsTrends {
  private readonly chart = viewChild(KvRangeChart);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private locationAnimation?: Animation;
  constructor() {
    inject(DestroyRef).onDestroy(() => this.locationAnimation?.cancel());
  }
  locateView() {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locateRange() {
    const control = this.host.nativeElement.querySelector<HTMLElement>('analysis-position-range');
    this.locateControl(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
    );
  }
  locateMetric() {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLElement>('.toolbar-metric config-picker > button'),
    );
  }
  private locateControl(target: HTMLElement | null | undefined, feedbackTarget = target) {
    if (!target) return;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    target.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
      behavior: reduced ? 'instant' : 'smooth',
    });
    this.locationAnimation?.cancel();
    this.locationAnimation = animateAnalysisLocationFeedback(feedbackTarget ?? target, reduced);
  }
  locate(side?: string, position?: number) {
    const layer = side === 'target' || side === 'Target' ? 1 : 0;
    this.chart()?.locateKind('logits', position, position === undefined ? undefined : layer);
  }
  readonly analysisLabel = input.required<string>();
  readonly inspectionEnabled = input(false);
  readonly inspect = output<'index' | 'scatter'>();
  chooseView(value: string) {
    if (value === 'Tensor Inspection · Line chart') this.inspect.emit('index');
    else if (value === 'Tensor Inspection · Scatter plot') this.inspect.emit('scatter');
  }
  readonly modeLabel = input('Tensor Comparison');
  readonly visualizationLabel = computed(() => this.modeLabel() + ' · Line chart');
  readonly inspectionChoices = computed(() =>
    this.modeLabel() === 'Distribution Comparison'
      ? []
      : ['Tensor Inspection · Line chart', 'Tensor Inspection · Scatter plot'],
  );
  readonly visualizationOptions = computed(() => [
    this.visualizationLabel(),
    ...(this.inspectionEnabled() ? this.inspectionChoices() : []),
  ]);
  readonly visualizationGroups = computed(() => [
    {label: this.modeLabel(), options: [this.visualizationLabel()]},
    ...(this.inspectionEnabled() && this.inspectionChoices().length
      ? [
          {
            label: 'Tensor Inspection',
            options: this.inspectionChoices(),
          },
        ]
      : []),
  ]);
  readonly visualizationLabels = computed(() => ({
    [this.visualizationLabel()]: 'Line chart',
    'Tensor Inspection · Line chart': 'Line chart',
    'Tensor Inspection · Scatter plot': 'Scatter plot',
  }));
  readonly metrics = input.required<readonly {id: string; label: string}[]>();
  readonly metricOptions = computed(() => this.metrics().map((metric) => metric.label));
  readonly metricChoiceLabel = computed(
    () =>
      this.metrics().find((metric) => metric.id === this.settings().metric)?.label ??
      this.settings().metric,
  );
  readonly metricLabel = input.required<string>();
  readonly settings = input.required<TokenTrendState>();
  readonly divergenceLabel = computed(() => {
    const metric = this.settings().metric;
    return metric === 'kl' ? 'KL divergence' : metric === 'js' ? 'JS divergence' : null;
  });
  readonly positionSeries = input.required<readonly LogitPositionSeries[]>();
  readonly expanded = input(false);
  readonly selectedIndex = input<number | null>(null);
  readonly selectedRange = input<KvRangeSelection | null>(null);
  readonly settingsChange = output<TokenTrendState>();
  readonly metricChange = output<string>();
  readonly expand = output<void>();
  readonly selectPosition = output<number>();
  readonly selectRange = output<{index: number; range: KvRangeSelection}>();
  readonly backgroundClick = output<MouseEvent>();
  readonly positionHover = output<KvRangeSelection | null>();
  readonly capacity = signal(160);
  readonly picked = signal<KvRangeSelection | null>(null);
  readonly reduction = computed(() => this.settings().positionReduction ?? 'max');
  readonly reductionLabels: Record<HeatmapReduction, string> = {
    min: 'Minimum',
    max: 'Maximum',
    mean: 'Mean',
  };
  readonly reductionOptions = ['Minimum', 'Maximum', 'Mean'];
  readonly displayOpen = signal(false);
  readonly displayDraft = signal<HeatmapReduction>('max');
  readonly binsDisabled = computed(() => this.bins().every((bin) => bin.end - bin.start === 1));
  readonly displayDirty = computed(
    () => !this.binsDisabled() && this.displayDraft() !== this.reduction(),
  );
  toggleDisplay() {
    if (!this.displayOpen()) this.displayDraft.set(this.reduction());
    this.displayOpen.update((open) => !open);
  }
  draftReduction(label: string) {
    const choice = (Object.keys(this.reductionLabels) as HeatmapReduction[]).find(
      (key) => this.reductionLabels[key] === label,
    );
    if (choice) this.displayDraft.set(choice);
  }
  applyDisplay() {
    if (this.displayDirty()) this.reduce(this.displayDraft());
  }
  chooseMetric(label: string) {
    const choice = this.metrics().find((metric) => metric.label === label);
    if (choice) this.metricChange.emit(choice.id);
  }
  readonly percent = computed(() => this.settings().metric === 'selected_probability');
  readonly fullRange = computed(() => {
    let start = Infinity;
    let end = -Infinity;
    for (const series of this.positionSeries()) {
      for (const point of series.points) {
        if (point.position < start) start = point.position;
        if (point.position > end) end = point.position;
      }
    }
    return start <= end ? {start, end: end + 1} : {start: 0, end: 1};
  });
  readonly range = computed(() => {
    const full = this.fullRange(),
      saved = this.settings().positionRange;
    if (!saved) return full;
    const start = Math.max(full.start, saved.start),
      end = Math.min(full.end, saved.end);
    return start < end ? {start, end} : full;
  });
  readonly bins = computed(() => tensorBins(this.range(), this.capacity()));
  readonly layers = computed(() => this.positionSeries().map((_, index) => index));
  private summarize(points: readonly LogitPositionSeries['points'][number][], bin: KvRangeBin) {
    const values = points.filter(
      (p) =>
        p.position >= bin.start &&
        p.position < bin.end &&
        p.value !== null &&
        Number.isFinite(p.value),
    );
    const numbers = values.map((p) => p.value!);
    const extreme = (reduction: 'min' | 'max') =>
      values.reduce<(typeof values)[number] | null>((a, b) => {
        if (!a) return b;
        const better = reduction === 'min' ? b.value! < a.value! : b.value! > a.value!;
        return better || (b.value === a.value && b.position < a.position) ? b : a;
      }, null)?.position ?? null;
    return {
      min: reduceMetricValues(numbers, 'min'),
      max: reduceMetricValues(numbers, 'max'),
      mean: reduceMetricValues(numbers, 'mean'),
      valid_count: numbers.length,
      min_position: extreme('min'),
      max_position: extreme('max'),
    } satisfies KvRangeMetricSummary;
  }
  private rowsFor(bins: readonly KvRangeBin[]): KvRangeRow[] {
    return this.positionSeries().map((series, layer) => ({
      layer,
      kind: 'logits',
      bins: bins.map((bin) => {
        const raw = this.summarize(series.points, bin);
        // All chart geometry and hover readouts use the displayed unit. Keep source points
        // and extrema positions unchanged so selection still locates the original token.
        const summary =
          this.settings().metric === 'margin'
            ? {
                ...raw,
                min: raw.min === null ? null : raw.min * 100,
                max: raw.max === null ? null : raw.max * 100,
                mean: raw.mean === null ? null : raw.mean * 100,
              }
            : raw;
        return {
          ...bin,
          metrics: {
            max_abs: summary,
            relative_l2: summary,
            cosine_similarity: summary,
            cosine_distance: summary,
          },
        };
      }),
    }));
  }
  readonly chartRows = computed(() => this.rowsFor(this.bins()));
  readonly overview = computed(() => {
    const bins = tensorBins(this.fullRange(), this.capacity());
    return {...this.fullRange(), context_id: '', bins, rows: this.rowsFor(bins)};
  });
  readonly series = computed(() =>
    this.positionSeries().map((s, layer) => ({
      layer,
      kind: 'logits',
      name: s.name,
      runtime: s.runtime,
      color: s.color,
      rows: this.chartRows(),
    })),
  );
  readonly runtimeSeries = computed(() =>
    this.positionSeries().flatMap((series) => (series.runtime ? [series.runtime] : [])),
  );
  readonly selection = computed(() => {
    if (this.selectedRange()) return this.selectedRange();
    const index = this.selectedIndex();
    const point = this.positionSeries()
      .flatMap((s) => s.points)
      .find((p) => p.index === index);
    const picked = this.picked();
    if (picked && point && point.position >= picked.start && point.position < picked.end)
      return picked;
    return point ? {start: point.position, end: point.position + 1, layer: null} : null;
  });
  readonly format = (v: number | null) =>
    v == null
      ? 'Unavailable'
      : this.percent()
        ? `${(v * 100).toFixed(2)}%`
        : this.settings().metric === 'margin'
          ? `${v.toFixed(2)} pp`
          : Number(v.toPrecision(5)).toString();
  zoom(range: KvRangeBin) {
    this.settingsChange.emit({...this.settings(), positionRange: range});
  }
  reduce(value: string) {
    this.settingsChange.emit({...this.settings(), positionReduction: value as HeatmapReduction});
  }
  representativeIndex(range: KvRangeSelection): number | null {
    const source =
      range.layer === null
        ? this.positionSeries().flatMap((s) => s.points)
        : (this.positionSeries()[range.layer]?.points ?? []);
    const position = representativePosition(range, this.reduction(), this.summarize(source, range));
    const point =
      source.find((p) => p.position === position) ??
      source
        .filter((p) => p.position >= range.start && p.position < range.end)
        .reduce<
          (typeof source)[number] | null
        >((a, b) => (!a || b.position < a.position ? b : a), null);
    return point?.index ?? null;
  }
  select(range: KvRangeSelection) {
    this.picked.set(range);
    const index = this.representativeIndex(range);
    if (index !== null) {
      this.selectPosition.emit(index);
      this.selectRange.emit({index, range});
    }
  }
}
