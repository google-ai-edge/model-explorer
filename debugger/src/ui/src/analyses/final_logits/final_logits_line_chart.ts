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
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MatIconModule} from '@angular/material/icon';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../widgets/locate_feedback';
import {PositionRangeControl} from '../../widgets/charts/position_range_control';
import {ViewActions} from '../../widgets/charts/view_actions';
import {RangeChart} from '../../widgets/charts/range_chart';
import type {ColumnSteps} from '../../widgets/format/step_labels';
import {representativePosition} from '../../widgets/charts/range_bins';
import {rangeBinCount} from '../../widgets/charts/range_geometry';
import type {HeatmapReduction} from '../../widgets/charts/heatmap_model';
import type {KvRangeBin, KvRangeRow} from '../../server/contracts/kv';
import type {ComparisonAlignment} from '../../server/contracts/capture';
import type {
  TokenRangeCell,
  TokenRangeRequest,
  TokenRangeResponse,
} from '../../server/contracts/token_analysis';
import type {RangeSelection} from '../../widgets/charts/range_selection';
import {catalogMetricId, type LogitPanelState} from '../shared/logit_panel_state';
import {CaptureStore} from '../../server/capture_store';
import {LatestRequest} from '../../server/latest_request';
import {MetricCatalogStore} from '../../server/metric_catalog';
import {TokenAnalysisApi} from '../../server/token_analysis_api';
import {formatMetricValue} from '../../widgets/format/format';
import {RuntimeLegend} from '../../widgets/charts/runtime_legend';
import type {Runtime} from '../../widgets/charts/runtime_series';
/** One line of the chart: its name and color, and the rows it has a point at. The server bins
 *  the values. */
export interface LogitPositionSeries {
  name: string;
  runtime?: Runtime;
  legendLabel?: string;
  color: string;
  points: {position: number; index: number}[];
}
/** Whether two ranges hold the same positions: a new series of the same rows asks nothing anew. */
function sameRange(a: KvRangeBin, b: KvRangeBin): boolean {
  return a.start === b.start && a.end === b.end;
}
/** A token-range request of a capture. */
interface RangeQuery {
  capture: string;
  request: TokenRangeRequest;
}
/** Whether two queries ask the same: the server's answer to them is the same. */
function sameQuery(a: RangeQuery | null, b: RangeQuery | null): boolean {
  return a === b || (!!a && !!b && JSON.stringify(a) === JSON.stringify(b));
}
@Component({
  selector: 'final-logits-line-chart',
  imports: [
    MatTooltipModule,
    MatIconModule,
    MatButtonModule,
    OverlayModule,
    OverlayDialog,
    OverlayPanel,
    ConfigPicker,
    PositionRangeControl,
    ViewActions,
    RangeChart,
    RuntimeLegend,
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
          <position-range-control
            label="Token Range"
            [range]="range()"
            [full]="fullRange()"
            (rangeChange)="zoom($event)"
          />
          <view-actions
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
            <runtime-legend [runtimes]="runtimeSeries()" />
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
          <view-actions
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
    @if (description(); as desc) {
      @if (desc.inputs) {
        <p class="metric-note" role="note" aria-label="Metric inputs">
          Inputs of {{ desc.label }}: {{ desc.inputs }}.
        </p>
      }
    }
    <div
      class="chart"
      [style.height]="expanded() ? 'calc(100dvh - 294px)' : '320px'"
      [attr.aria-busy]="pending()"
    >
      <range-chart
        [class.chart-pending]="!shown() && (pending() || !!failure())"
        [label]="analysisLabel()"
        [columnSteps]="columnSteps()"
        kind="logits"
        [showLayers]="false"
        view="lineChart"
        metric="max_abs"
        [metricLabel]="metricLabel()"
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
      @if (waiting()) {
        <p class="chart-state" role="status">Loading {{ analysisLabel() }}…</p>
      } @else if (failure()) {
        <div [class]="shown() ? 'chart-status error' : 'chart-state error'" role="alert">
          <span>{{ failure() }}</span>
          <button mat-button (click)="retry.set(retry() + 1)">Retry</button>
        </div>
      }
    </div>`,
  styleUrl: './final_logits_line_chart.scss',
})
export class FinalLogitsLineChart {
  private readonly chart = viewChild(RangeChart);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly api = inject(TokenAnalysisApi);
  private readonly state = inject(CaptureStore);
  private locationAnimation?: Animation;
  /** The server's bins of the visible range, and of the whole stage for the overview; and, for a
   *  selected range that is none of the drawn bins (it outlived a zoom or a resize), its one
   *  bin. The visible range is the overview's own answer while the chart shows the whole stage. */
  readonly rangeData = new LatestRequest<TokenRangeResponse>();
  readonly overviewData = new LatestRequest<TokenRangeResponse>();
  readonly selectionData = new LatestRequest<TokenRangeResponse>();
  /** The requests of the whole stage and of the visible range; each changes only when its answer
   *  can, so nothing is asked anew for an equal request. */
  private readonly overviewQuery = computed(() => this.query(this.fullRange()), {
    equal: sameQuery,
  });
  private readonly rangeQuery = computed(() => this.query(this.range()), {equal: sameQuery});
  private readonly whole = computed(() => sameQuery(this.rangeQuery(), this.overviewQuery()));
  /** The answer to the visible range's request; null until it arrives. */
  private readonly answer = computed(() =>
    this.whole() ? this.overviewData.data() : this.rangeData.data(),
  );
  /** What the chart draws: the visible range's answer, or until it arrives the last one drawn. */
  readonly shown = computed(() =>
    this.whole() ? this.overviewData.last() : (this.rangeData.last() ?? this.overviewData.last()),
  );
  /** Whether the server is computing what the chart draws: until its first answer arrives, the
   *  chart shows that instead of empty axes; later, it keeps its drawing meanwhile. */
  readonly pending = computed(() => this.overviewData.loading() || this.rangeData.loading());
  /** Whether the chart waits for its first answer, with nothing drawn yet. */
  readonly waiting = computed(() => !this.shown() && this.pending());
  /** Whether what the chart draws is not all in yet: an answer it asks for, of both alignments
   *  (AnalysisCache), or one it is about to ask for with nothing drawn. A failure ends it. */
  readonly busy = computed(
    () =>
      this.pending() ||
      this.selectionData.loading() ||
      (!!this.overviewQuery() && !this.shown() && !this.failure()),
  );
  /** Why the server could not answer what the chart draws, in its words. */
  readonly failure = computed(() =>
    this.whole() ? this.overviewData.error() : this.rangeData.error() || this.overviewData.error(),
  );
  /** Asks again after a failure. */
  readonly retry = signal(0);
  constructor() {
    // A resize changes the bins many times: only the last asks the server.
    effect((onCleanup) => {
      this.retry();
      const query = this.overviewQuery();
      onCleanup(
        this.overviewData.load(query ? (signal) => this.read(query, signal) : null, undefined, 100),
      );
    });
    effect((onCleanup) => {
      this.retry();
      const query = this.whole() ? null : this.rangeQuery();
      onCleanup(
        this.rangeData.load(query ? (signal) => this.read(query, signal) : null, undefined, 100),
      );
    });
    effect((onCleanup) => {
      const range = this.selectedRange();
      const data = this.answer();
      const drawn = !range || !data || data.bins.some((bin) => sameRange(bin, range));
      const query = drawn ? null : this.query(range, 1);
      onCleanup(this.selectionData.load(query ? (signal) => this.read(query, signal) : null));
    });
    inject(DestroyRef).onDestroy(() => {
      this.locationAnimation?.cancel();
      this.rangeData.destroy();
      this.overviewData.destroy();
      this.selectionData.destroy();
    });
  }
  /** Whether any line has a point: a stage without rows asks the server nothing. */
  private readonly hasPoints = computed(() =>
    this.positionSeries().some((series) => series.points.length),
  );
  private readonly metricId = computed(() => catalogMetricId(this.settings().metric));
  /** The token-range request of a range, in `bins` bins or at the chart's capacity, with its
   *  capture; none without a capture, a turn or points. */
  private query(
    range: KvRangeBin,
    bins = rangeBinCount(range, this.capacity()),
  ): RangeQuery | null {
    const capture = this.state.captureId();
    const turn = this.turn();
    if (!capture || turn === null || !this.hasPoints()) {
      return null;
    }
    const request: TokenRangeRequest = {
      turn,
      stage: 'decode',
      alignment: this.alignment(),
      analysis: this.analysis(),
      metric: this.metricId(),
      start: range.start,
      end: range.end,
      bins,
    };
    return {capture, request};
  }
  private async read({capture, request}: RangeQuery, signal: AbortSignal): Promise<TokenRangeResponse> {
    const data = await this.api.tokenRange(capture, request, signal, untracked(this.capacity));
    if (data.start !== request.start || data.end !== request.end || data.metric !== request.metric) {
      throw new Error('Token range response did not match the requested range.');
    }
    return data;
  }
  locateView(): void {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locateRange(): void {
    const control = this.host.nativeElement.querySelector<HTMLElement>('position-range-control');
    this.locateControl(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
    );
  }
  locateMetric(): void {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLElement>('.toolbar-metric config-picker > button'),
    );
  }
  private locateControl(target: HTMLElement | null | undefined, feedbackTarget = target): void {
    if (!target) {
      return;
    }
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    target.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
      behavior: reduced ? 'instant' : 'smooth',
    });
    this.locationAnimation?.cancel();
    this.locationAnimation = animateAnalysisLocationFeedback(feedbackTarget ?? target, reduced);
  }
  locate(side?: string, position?: number): void {
    const layer = side === 'target' || side === 'Target' ? 1 : 0;
    this.chart()?.locateKind('logits', position, position === undefined ? undefined : layer);
  }
  /** The turn, its alignment and the analysis whose metric the chart draws, as the server's
   *  token-range names them; without a turn, the chart asks for nothing. */
  readonly turn = input<number | null>(null);
  readonly alignment = input<ComparisonAlignment>('content');
  readonly analysis = input<'final_logits' | 'sampling'>('final_logits');
  readonly analysisLabel = input.required<string>();
  /** Which step produced the token at a column, on each side, in the server's words. */
  readonly columnSteps = input<((column: number) => ColumnSteps | null) | null>(null);
  readonly inspectionEnabled = input(false);
  readonly inspect = output<'index' | 'scatter'>();
  chooseView(value: string): void {
    if (value === 'Tensor Inspection · Line chart') {
      this.inspect.emit('index');
    } else if (value === 'Tensor Inspection · Scatter plot') {
      this.inspect.emit('scatter');
    }
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
  readonly settings = input.required<LogitPanelState>();
  private readonly catalog = inject(MetricCatalogStore);
  /** The metric as the server's catalog describes it. */
  readonly description = computed(() =>
    this.catalog.lookup().metric(catalogMetricId(this.settings().metric)),
  );
  /** What a value is multiplied by on the chart, as a probability drawn in percent. */
  private readonly scale = computed(() => this.description()?.format.scale ?? 1);
  readonly positionSeries = input.required<readonly LogitPositionSeries[]>();
  readonly expanded = input(false);
  readonly selectedIndex = input<number | null>(null);
  readonly selectedRange = input<RangeSelection | null>(null);
  readonly settingsChange = output<LogitPanelState>();
  readonly metricChange = output<string>();
  readonly expand = output<void>();
  readonly selectPosition = output<number>();
  readonly selectRange = output<{index: number; range: RangeSelection}>();
  readonly backgroundClick = output<MouseEvent>();
  readonly positionHover = output<RangeSelection | null>();
  readonly capacity = signal(160);
  readonly picked = signal<RangeSelection | null>(null);
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
  toggleDisplay(): void {
    if (!this.displayOpen()) {
      this.displayDraft.set(this.reduction());
    }
    this.displayOpen.update((open) => !open);
  }
  draftReduction(label: string): void {
    const choice = (['min', 'max', 'mean'] as const).find(
      (key) => this.reductionLabels[key] === label,
    );
    if (choice) {
      this.displayDraft.set(choice);
    }
  }
  applyDisplay(): void {
    if (this.displayDirty()) {
      this.reduce(this.displayDraft());
    }
  }
  chooseMetric(label: string): void {
    const choice = this.metrics().find((metric) => metric.label === label);
    if (choice) {
      this.metricChange.emit(choice.id);
    }
  }
  readonly fullRange = computed(
    () => {
      const p = this.positionSeries().flatMap((s) => s.points.map((p) => p.position));
      return p.length ? {start: Math.min(...p), end: Math.max(...p) + 1} : {start: 0, end: 1};
    },
    {equal: sameRange},
  );
  readonly range = computed(
    () => {
      const full = this.fullRange();
      const saved = this.settings().positionRange;
      if (!saved) {
        return full;
      }
      const start = Math.max(full.start, saved.start);
      const end = Math.min(full.end, saved.end);
      return start < end ? {start, end} : full;
    },
    {equal: sameRange},
  );
  readonly bins = computed(() => this.shown()?.bins ?? []);
  readonly layers = computed(() => this.positionSeries().map((_, index) => index));
  /** A server cell as the range chart reads it: in the displayed unit of its answer's metric
   *  (`scale`), every metric key the chart may name reading the one metric. Extrema positions
   *  stay the tokens'. */
  private cell(cell: TokenRangeCell, scale: number) {
    const {start, end, ...raw} = cell;
    const summary =
      scale !== 1
        ? {
            ...raw,
            min: raw.min === null ? null : raw.min * scale,
            max: raw.max === null ? null : raw.max * scale,
            mean: raw.mean == null ? null : raw.mean * scale,
          }
        : raw;
    return {
      start,
      end,
      metrics: {
        max_abs: summary,
        relative_l2: summary,
        cosine_similarity: summary,
        cosine_distance: summary,
      },
    };
  }
  private rowsFor(data: TokenRangeResponse | null): KvRangeRow[] {
    // An answer shown while another metric's loads keeps its own unit.
    const scale = (data && this.catalog.lookup().metric(data.metric)?.format.scale) || 1;
    return (data?.rows ?? []).map((row, layer) => ({
      layer,
      kind: 'logits',
      bins: row.bins.map((cell) => this.cell(cell, scale)),
    }));
  }
  readonly chartRows = computed(() => this.rowsFor(this.shown()));
  readonly overview = computed(() => {
    const data = this.overviewData.last();
    const full = this.fullRange();
    const whole = data && data.start === full.start && data.end === full.end;
    return {...full, bins: whole ? data.bins : [], rows: whole ? this.rowsFor(data) : []};
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
    if (this.selectedRange()) {
      return this.selectedRange();
    }
    const index = this.selectedIndex();
    const point = this.positionSeries()
      .flatMap((s) => s.points)
      .find((p) => p.index === index);
    const picked = this.picked();
    if (picked && point && point.position >= picked.start && point.position < picked.end) {
      return picked;
    }
    return point ? {start: point.position, end: point.position + 1, layer: null} : null;
  });
  /** A value on the chart, which is in the displayed unit, as the catalog writes it. */
  readonly format = (v: number | null): string =>
    v == null ? 'Unavailable' : formatMetricValue(this.description(), v / this.scale());
  zoom(range: KvRangeBin): void {
    this.settingsChange.emit({...this.settings(), positionRange: range});
  }
  reduce(value: HeatmapReduction): void {
    this.settingsChange.emit({...this.settings(), positionReduction: value});
  }
  /** The row a selected range stands for: the token at its extremum, of the selected line or of
   *  all of them, as the server's statistic of the range gives it (a drawn bin's, else the
   *  selected range's own); a mean, or a range the server has not answered yet, stands for its
   *  first token. */
  representativeIndex(range: RangeSelection): number | null {
    const source =
      range.layer === null
        ? this.positionSeries().flatMap((s) => s.points)
        : (this.positionSeries()[range.layer]?.points ?? []);
    const cell = [this.answer(), this.selectionData.data()]
      .flatMap((data) => (range.layer === null ? data?.all : data?.rows[range.layer]?.bins) ?? [])
      .find((bin) => sameRange(bin, range));
    const position = representativePosition(range, this.reduction(), cell);
    const point =
      source.find((p) => p.position === position) ??
      source
        .filter((p) => p.position >= range.start && p.position < range.end)
        .reduce<
          (typeof source)[number] | null
        >((a, b) => (!a || b.position < a.position ? b : a), null);
    return point?.index ?? null;
  }
  select(range: RangeSelection): void {
    this.picked.set(range);
    const index = this.representativeIndex(range);
    if (index !== null) {
      this.selectPosition.emit(index);
      this.selectRange.emit({index, range});
    }
  }
}
