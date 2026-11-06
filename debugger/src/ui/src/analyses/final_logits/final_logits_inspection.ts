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
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {
  FinalLogitsAddress,
  FinalLogitsBins,
  FinalLogitsBinsRequest,
  FinalLogitsBinsResponse,
  FinalLogitsOrder,
  FinalLogitsPointsRequest,
  FinalLogitsPointsResponse,
  FinalLogitsRequest,
  FinalLogitsResponse,
  FinalLogitsRow,
  FinalLogitsSummaryResponse,
  FinalLogitsTransform,
  FinalLogitsZoomedPointsResponse,
} from '../../server/contracts/final_logits';
import type {TokenAnalysisPair} from '../../server/contracts/token_analysis';
import type {Reason} from '../../server/contracts/metrics';
import {reasonText, sideReasonText} from '../../server/reasons';
import type {KvHeadChannel, KvRangeBin} from '../../server/contracts/kv';
import {FinalLogitsApi} from '../../server/final_logits_api';
import {CaptureStore} from '../../server/capture_store';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {formatTensorValue} from '../../widgets/format/format';
import {animateAnalysisLocationFeedback} from '../../widgets/locate_feedback';
import type {LogitPanelState} from '../shared/logit_panel_state';
import {LatestRequest} from '../../server/latest_request';
import {ChannelChart} from '../../widgets/charts/channel_chart';
import {ScatterChart} from '../../widgets/charts/scatter_chart';
import {
  channelBinSize,
  type ChannelReduction,
  type ReducedChannelBin,
  type ReducedChannelBins,
} from '../../widgets/charts/channel_bins';
import {
  isPairedChannel,
  validValueRange,
  type PairedChannel,
  type ScatterView,
  type ValueRange,
} from '../../widgets/charts/scatter_model';
import {PositionRangeControl} from '../../widgets/charts/position_range_control';
import {ValueRangeControl} from '../../widgets/charts/value_range_control';
import {ViewActions} from '../../widgets/charts/view_actions';
import {RuntimeLegend} from '../../widgets/charts/runtime_legend';

/** How long a chart's view must rest before its request is sent, in ms: a drag, a resize or a held
 *  key changes the view many times, and only where it rests asks the server. */
const SETTLE_MS = 100;

/** A notice of reasons: the server's text, and the codes and sides that tell two notices apart. */
interface Notice {
  key: string;
  text: string;
}

/** An answer about a row's whole vectors, with the capture and address it is of (`key`) and the
 *  request it answers (`request`). */
interface Answer<T> {
  key: string;
  request: string;
  value: T;
}

/** Whether an answer names the row its request named. */
function answers(
  value: FinalLogitsAddress & {first_divergence: boolean},
  query: FinalLogitsAddress,
): boolean {
  return (
    value.turn === query.turn &&
    value.stage === query.stage &&
    value.alignment === query.alignment &&
    value.row === query.row &&
    value.first_divergence === !!query.first_divergence &&
    value.transform === query.transform
  );
}

/** The line chart's bins, as the server listed them. */
function reducedBins(bins: FinalLogitsBins): ReducedChannelBin[] {
  const result: ReducedChannelBin[] = [];
  let k = 0;
  for (let start = bins.start; start < bins.end; start += bins.size) {
    result.push({
      start,
      end: Math.min(bins.end, start + bins.size),
      ref: bins.ref[k],
      target: bins.target[k],
      refIndex: bins.ref_index[k],
      targetIndex: bins.target_index[k],
    });
    k++;
  }
  return result;
}

/** An element of both vectors, as the charts read one. */
function element(
  channel: number,
  ref: KvHeadChannel['ref'],
  target: KvHeadChannel['target'],
): KvHeadChannel {
  const delta =
    typeof ref === 'number' && typeof target === 'number' && Number.isFinite(target - ref)
      ? target - ref
      : null;
  return {
    channel,
    ref,
    target,
    delta,
    abs_delta: delta === null ? null : Math.abs(delta),
    status: delta === null ? 'unavailable' : 'available',
  };
}

/** The scatter plot's points, as the server listed them. */
function pointRows(points: FinalLogitsZoomedPointsResponse['points']): PairedChannel[] {
  return points.index
    .map((channel, k) => element(channel, points.ref[k], points.target[k]))
    .filter(isPairedChannel);
}

/** The notice of a side's own reasons, or of the pair's (`side` null); null for none. */
function notice(side: 'ref' | 'target' | null, reasons: readonly Reason[]): Notice | null {
  if (!reasons.length) {
    return null;
  }
  return {
    key: JSON.stringify([side, reasons.map((reason) => [reason.code, reason.side])]),
    text:
      side === null
        ? reasonText(reasons)
        : (side === 'ref' ? 'Reference: ' : 'Target: ') + sideReasonText(reasons),
  };
}

@Component({
  selector: 'final-logits-inspection',
  imports: [
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    ConfigPicker,
    OverlayModule,
    OverlayDialog,
    OverlayPanel,
    ViewActions,
    RuntimeLegend,
    PositionRangeControl,
    ValueRangeControl,
    ChannelChart,
    ScatterChart,
  ],
  template: ` <div class="controls" role="toolbar" aria-label="Final logits inspection controls">
      <div class="toolbar-scroll">
        <div class="toolbar-view">
          <span>View</span>
          <config-picker
            class="view-picker"
            label="Visualization"
            plural="visualizations"
            [value]="visualizationLabel()"
            [attr.title]="visualizationLabel()"
            [options]="visualizationOptions"
            [groups]="visualizationGroups"
            [optionLabels]="visualizationLabels"
            [compact]="true"
            [searchable]="false"
            (valueChange)="chooseView($event)"
          />
        </div>
        <span class="tool-divider" aria-hidden="true"></span>
        <label class="position-control"
          >Token Position
          <button
            type="button"
            class="icon-button"
            aria-label="Previous logits position"
            [disabled]="position() <= fullRange().start"
            (click)="setPosition(position() - 1)"
          >
            <mat-icon>chevron_left</mat-icon>
          </button>
          <input
            type="number"
            step="1"
            aria-label="Token Position"
            [min]="fullRange().start"
            [max]="fullRange().end - 1"
            [value]="position()"
            (change)="commitPosition($event)"
            (keydown.enter)="commitPosition($event)"
          />
          <button
            type="button"
            class="icon-button"
            aria-label="Next logits position"
            [disabled]="position() >= fullRange().end - 1"
            (click)="setPosition(position() + 1)"
          >
            <mat-icon>chevron_right</mat-icon>
          </button>
        </label>
        <div
          class="toolbar-range"
          [attr.aria-label]="plot() === 'index' ? 'Element Range controls' : 'Value Range controls'"
          role="group"
        >
          @if (plot() === 'index') {
            <position-range-control
              label="Element Range"
              [range]="elementRange()"
              [full]="elementFull()"
              (rangeChange)="zoomElements($event)"
            />
            <view-actions
              [range]="elementRange()"
              [full]="elementFull()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          } @else {
            <value-range-control [range]="valueRange()" (rangeChange)="zoomValues($event)" />
            <view-actions
              [range]="valueRange()"
              [full]="valueFull()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          }
        </div>
      </div>
      <div class="toolbar-actions">
        <div class="toolbar-legend">
          @if (plot() === 'index') {
            <runtime-legend />
          } @else {
            <div class="tensor-axis-legend" aria-label="Tensor scatter axes">
              X: Ref · Y: Target
            </div>
          }
        </div>
        <button
          mat-icon-button
          class="view-display-button"
          aria-label="Display"
          [disabled]="plot() === 'scatter'"
          [matTooltip]="
            plot() === 'scatter'
              ? 'No display settings for Scatter plot'
              : 'Display · Element bins: ' +
                (displayBinsDisabled() ? '-' : reductionLabel(reduction()))
          "
          cdkOverlayOrigin
          #displayOrigin="cdkOverlayOrigin"
          [attr.aria-expanded]="displayOpen() && plot() === 'index'"
          (click)="toggleDisplay()"
        >
          <mat-icon>visibility</mat-icon>
        </button>
        @if (!expanded()) {
          <view-actions class="panel-expand" [showReset]="false" (expandView)="expand.emit()" />
        }
      </div>
      <ng-template
        overlayDialog
        [origin]="displayOrigin"
        [open]="displayOpen() && plot() === 'index'"
        placement="below-end"
        [width]="360"
        (closed)="displayOpen.set(false)"
      >
        <div class="chart-display">
          <section class="settings-panel" overlayPanel aria-label="Final logits display settings">
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
              <label>Element bins</label>
              <config-picker
                label="Element bins"
                plural="reductions"
                [value]="displayBinsDisabled() ? '-' : reductionLabel(displayDraft())"
                [disabled]="displayBinsDisabled()"
                [options]="['Minimum', 'Maximum', 'Mean']"
                [compact]="true"
                [searchable]="false"
                (valueChange)="setDisplayDraft($event)"
              />
              <button
                mat-button
                [disabled]="displayBinsDisabled() || displayDraft() === 'mean'"
                aria-label="Reset Element bins to default"
                (click)="setDisplayDraft('Mean')"
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
    </div>
    <section
      class="results"
      aria-label="Final logits tensor visualization"
      [attr.aria-busy]="loading()"
    >
      @if (loading()) {
        <p class="state-message" role="status">Loading captured logits…</p>
      } @else if (error() || drawingError()) {
        <p class="state-message" role="alert">
          {{ error() || drawingError() }} <button mat-button (click)="retry()">Retry</button>
        </p>
      } @else if (!analysis() && analysisLoading()) {
        <p class="state-message" role="status">Loading captured logits…</p>
      } @else if (!analysis()) {
        <p class="state-message">No captured logits selection at this token position.</p>
      } @else if (!total()) {
        <p class="state-message">No captured logits elements are available.</p>
      } @else {
        @for (notice of notices(); track notice) {
          <p class="state-message">{{ notice }}</p>
        }
        <div
          class="tensor-line-canvas"
          [class.expanded]="expanded()"
          [attr.aria-busy]="plot() === 'index' ? !lineBins() : !scatterView()"
        >
          @if (plot() === 'index' ? !lineBins() : !scatterView()) {
            <p class="chart-state" role="status">Loading captured logits…</p>
          }
          @if (plot() === 'index') {
            <channel-chart
              [class.chart-pending]="!lineBins()"
              dimensionLabel="Vocabulary index"
              [groupLabels]="['Final logits']"
              [comparisonAllowed]="summaryRequest.data()?.compatible ?? false"
              [bins]="lineBins()"
              [waiting]="lineWaiting()"
              [range]="lineRange()"
              [full]="elementFull()"
              [reduction]="lineReduction()"
              [capacity]="capacity()"
              (capacityChange)="capacity.set($event); capacityMeasured.set(true)"
              [selected]="selectedIndex()"
              [channelSelection]="channelSelection()"
              (channelSelectionChange)="channelSelection.set($event)"
              (zoom)="zoomElements($event)"
              (select)="showElement($event)"
              (backgroundClick)="clearElementSelection()"
            />
          } @else {
            <scatter-chart
              [class.chart-pending]="!scatterView()"
              dimensionLabel="Vocabulary index"
              [groupLabels]="['Final logits']"
              [singlePlane]="true"
              [showGroupingNotice]="true"
              [view]="scatterView()"
              [waiting]="scatterWaiting()"
              [range]="scatterRange()"
              [selected]="selectedIndex()"
              [messages]="scatterMessages()"
              (select)="showElement($event)"
              (zoom)="zoomValues($event)"
              (backgroundClick)="clearElementSelection()"
            />
          }
        </div>
        @if (plot() === 'index' && lineBins() && binSize() > 1) {
          <p class="plot-caption">
            {{ reductionLabel(reduction()) }} · up to {{ binSize() }} vocabulary indices per bin
          </p>
        }
      }
      <span class="sr-only" aria-live="polite">{{ announcement() }}</span>
    </section>`,
  styleUrl: '../shared/tensor_inspection.scss',
})
export class FinalLogitsInspection {
  readonly turn = input.required<number>();
  readonly position = input.required<number>();
  readonly fullRange = input.required<KvRangeBin>();
  readonly analysis = input<TokenAnalysisPair | undefined>();
  readonly settings = input.required<LogitPanelState>();
  readonly expanded = input(false);
  /** Whether the token analysis that names the pair (`analysis`) is still loading. */
  readonly analysisLoading = input(false);
  readonly settingsChange = output<LogitPanelState>();
  readonly positionChange = output<number>();
  readonly expand = output<void>();
  readonly pageSize = 16;
  /** The table's page. */
  readonly request = new LatestRequest<FinalLogitsResponse>();
  /** What is known of the whole vectors. */
  readonly summaryRequest = new LatestRequest<FinalLogitsSummaryResponse>();
  /** What the charts draw, the server's answers for the view: the line chart's bins, the scatter
   *  plot's points of the whole extent and of a zoomed range; the selected element's values. */
  readonly binsRequest = new LatestRequest<Answer<FinalLogitsBinsResponse>>();
  readonly pointsRequest = new LatestRequest<Answer<FinalLogitsPointsResponse>>();
  readonly zoomedPointsRequest = new LatestRequest<Answer<FinalLogitsZoomedPointsResponse>>();
  readonly elementRequest = new LatestRequest<Answer<FinalLogitsResponse>>();
  readonly loading = this.summaryRequest.loading;
  readonly error = this.summaryRequest.error;
  readonly valuesLoading = this.request.loading;
  readonly valuesError = this.request.error;
  readonly data = computed<FinalLogitsResponse | null>(() => {
    const summary = this.summaryRequest.data();
    const page = this.request.data();
    return summary
      ? {...summary, offset: page?.offset ?? 0, limit: this.pageSize, rows: page?.rows ?? []}
      : null;
  });
  private readonly api = inject(FinalLogitsApi);
  private readonly state = inject(CaptureStore);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly captureId = this.state.captureId;
  private readonly retryCount = signal(0);
  readonly selectedIndex = signal<number | null>(null);
  readonly channelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  readonly announcement = signal('');
  readonly lineChart = viewChild(ChannelChart);
  readonly scatterChart = viewChild(ScatterChart);
  readonly capacity = signal(128);
  /** Whether the line chart has measured its width: its bins are asked for only then. */
  readonly capacityMeasured = signal(false);
  readonly displayOpen = signal(false);
  readonly displayDraft = signal<ChannelReduction>('mean');
  private locationAnimation?: Animation;
  readonly transform = computed<FinalLogitsTransform>(() => 'raw');
  readonly plot = computed(() => this.settings().logitPlot ?? 'index');
  readonly reduction = computed<ChannelReduction>(
    () => this.settings().logitElementReduction ?? 'mean',
  );
  readonly order = computed<FinalLogitsOrder>(() => this.settings().logitOrder ?? 'index');
  readonly sortDirection = computed<'ascending' | 'descending'>(
    () =>
      this.settings().logitSortDirection ??
      (this.order() === 'absolute_delta' ? 'descending' : 'ascending'),
  );
  readonly total = computed(
    () =>
      this.summaryRequest.data()?.total ??
      Math.max(
        0,
        this.analysis()?.distribution.ref.size ?? 0,
        this.analysis()?.distribution.target.size ?? 0,
      ),
  );
  readonly pages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));
  readonly offset = computed(() => this.request.data()?.offset ?? this.savedOffset());
  readonly pageIndex = computed(() =>
    Math.min(this.pages() - 1, Math.floor(this.offset() / this.pageSize)),
  );
  readonly rows = computed(() =>
    (this.request.data()?.rows ?? []).map((row) => ({
      ...row,
      label: String(row.index),
      tokenLabel: row.label === null ? '—' : JSON.stringify(row.label),
    })),
  );
  readonly elementFull = computed(() => ({start: 0, end: Math.max(1, this.total())}));
  readonly elementRange = computed(() => {
    const full = this.elementFull();
    const saved = this.settings().logitElementRange;
    if (!saved) {
      return full;
    }
    const start = Math.max(0, Math.min(full.end - 1, saved.start));
    return {start, end: Math.max(start + 1, Math.min(full.end, saved.end))};
  });
  readonly binSize = computed(() => channelBinSize(this.elementRange(), this.capacity()));
  readonly displayBinsDisabled = computed(() => this.binSize() === 1);
  readonly displayDirty = computed(
    () => !this.displayBinsDisabled() && this.displayDraft() !== this.reduction(),
  );
  /** The capture and row the charts' answers must be of. */
  private readonly answerKey = computed(() => JSON.stringify([this.captureId(), this.query()]));
  /** A request's last answer, if it is of the current capture and row. */
  private current<T>(request: LatestRequest<Answer<T>>): T | null {
    const answer = request.last();
    return answer && answer.key === this.answerKey() ? answer.value : null;
  }
  private readonly bins = computed(() => this.current(this.binsRequest));
  /** The line chart's bins, range and reduction: the last answer's, kept while the next loads. */
  readonly lineBins = computed<ReducedChannelBins | null>(() => {
    const bins = this.bins();
    return bins ? {overview: reducedBins(bins.overview), visible: reducedBins(bins.visible)} : null;
  });
  readonly lineRange = computed(() => this.bins()?.range ?? this.elementRange());
  readonly lineReduction = computed(() => this.bins()?.reduction ?? this.reduction());
  private readonly points = computed(() => this.current(this.pointsRequest));
  private readonly zoomedPoints = computed(() => this.current(this.zoomedPointsRequest));
  /** The selected element's values, looked up when no point holds them. */
  private readonly lookedUp = computed(() => {
    const row = this.current(this.elementRequest)?.rows[0];
    return row ? element(row.index, row.ref, row.target) : null;
  });
  /** The elements whose values are known, by vocabulary index: the points, and, of a compared
   *  pair, the table page's rows and the looked-up element. */
  private readonly elements = computed(() => {
    const known = new Map<number, KvHeadChannel>();
    for (const answer of [this.points(), this.zoomedPoints()]) {
      for (const row of answer ? pointRows(answer.points) : []) {
        known.set(row.channel, row);
      }
    }
    if (this.points()?.compatible) {
      // A page is the current pair's: it is cleared while the next loads.
      for (const row of [
        ...(this.request.data()?.rows ?? []).map((row) => element(row.index, row.ref, row.target)),
        ...[this.lookedUp() ?? []].flat(),
      ]) {
        if (!known.has(row.channel)) {
          known.set(row.channel, row);
        }
      }
    }
    return known;
  });
  /** The scatter plot's points: the zoomed range's, else the whole extent's, kept while the next
   *  loads. */
  readonly scatterView = computed<ScatterView | null>(() => {
    const base = this.points();
    if (!base) {
      return null;
    }
    const drawn = this.zoomedPoints() ?? base;
    return {
      full: base.extent,
      visible: {
        rows: pointRows(drawn.points),
        total: drawn.paired,
        visible: drawn.visible,
        rendered: drawn.points.index.length,
        grouped: drawn.grouped,
      },
      overview: pointRows(base.points),
      paired: base.paired_runs.map(([start, end]) => ({start, end})),
      elements: this.elements(),
    };
  });
  readonly scatterRange = computed(
    () => (this.zoomedPoints() ?? this.points())?.range ?? this.valueRange(),
  );
  /** What the charts ask the server for their current view, null when they ask nothing. */
  private readonly binsPayload = computed(() => {
    const query = this.query();
    const saved = this.settings().logitElementRange;
    return query
      ? {
          ...query,
          ...(saved ? {start: saved.start, end: saved.end} : {}),
          capacity: this.capacity(),
          reduction: this.reduction(),
        }
      : null;
  });
  private readonly zoomedPointsPayload = computed(() => {
    const query = this.query();
    const saved = this.settings().logitValueRange;
    return query && saved ? {...query, start: saved.start, end: saved.end} : null;
  });
  /** Each chart's request for its current view, as the JSON of its answer key and payload; null
   *  when it asks nothing. A string, so an effect reruns only when the request changes. */
  private asked(payload: object | null): string | null {
    return payload && this.captureId() ? JSON.stringify([this.answerKey(), payload]) : null;
  }
  private readonly binsAsked = computed(() => this.asked(this.binsPayload()));
  private readonly pointsAsked = computed(() => this.asked(this.query()));
  private readonly zoomedPointsAsked = computed(() => this.asked(this.zoomedPointsPayload()));
  private readonly elementAsked = computed(() => {
    const query = this.query();
    const index = this.selectedIndex();
    return this.plot() === 'scatter' && query && index !== null && !this.elements().has(index)
      ? this.asked({...query, offset: index, limit: 1, order: 'index', direction: 'ascending'})
      : null;
  });
  /** Whether a chart's drawing is not yet its current view's: it keeps it until the answer. */
  readonly lineWaiting = computed(() => this.binsRequest.last()?.request !== this.binsAsked());
  readonly scatterWaiting = computed(() => {
    const zoomed = this.zoomedPointsAsked();
    return (
      !this.points() || (zoomed !== null && this.zoomedPointsRequest.last()?.request !== zoomed)
    );
  });
  readonly drawingError = computed(() =>
    this.plot() === 'index'
      ? this.binsRequest.error()
      : this.pointsRequest.error() || this.zoomedPointsRequest.error(),
  );
  /** The vectors' statistics, as the server gives them with the summary. */
  readonly statistics = computed(() => this.summaryRequest.data()?.statistics ?? null);
  readonly scatterMessages = computed(() =>
    this.summaryRequest.data()?.compatible
      ? []
      : [
          reasonText(this.summaryRequest.data()?.reasons) ||
            'Matching vocabulary and token context are required for a paired plot.',
        ],
  );
  readonly valueFull = computed(() => this.points()?.extent ?? {start: -1, end: 1});
  readonly valueRange = computed(() => this.settings().logitValueRange ?? this.valueFull());
  /** An element whose values decide, once known, whether locating it resets the value range. */
  private readonly revealed = signal<number | null>(null);
  /** The vector's notices: its pair's reasons, then each side's own that the pair's do not
   *  hold, a reason being held by its code and side; then why the compared vectors read different
   *  tokens on the two sides. */
  private readonly vectorNotices = computed(() => {
    const data = this.summaryRequest.data();
    if (!data) {
      return [];
    }
    const held = (reason: Reason) =>
      data.reasons.some((given) => given.code === reason.code && given.side === reason.side);
    return [
      notice(null, data.reasons),
      notice(null, data.notes ?? []),
      ...(['ref', 'target'] as const).map((side) =>
        notice(
          side,
          (data[side].reasons ?? []).filter((reason) => !held(reason)),
        ),
      ),
    ].filter((item): item is Notice => item !== null);
  });
  readonly notices = computed(() => this.vectorNotices().map((item) => item.text));
  /** The vector's notices and the page's own, the page's last; a vector notice of the same codes
   *  and sides as the page's is shown once. */
  readonly pageValuesNotices = computed(() => {
    const page = notice(null, this.request.data()?.reasons ?? []);
    const notices = this.vectorNotices().filter((item) => item.key !== page?.key);
    return [...notices, ...(page ? [page] : [])].map((item) => item.text);
  });
  readonly visualizationLabel = computed(() =>
    this.plot() === 'scatter'
      ? 'Tensor Inspection · Scatter plot'
      : 'Tensor Inspection · Line chart',
  );
  readonly visualizationOptions = [
    'Tensor Comparison · Line chart',
    'Tensor Inspection · Line chart',
    'Tensor Inspection · Scatter plot',
  ];
  readonly visualizationGroups = [
    {label: 'Tensor Comparison', options: ['Tensor Comparison · Line chart']},
    {
      label: 'Tensor Inspection',
      options: ['Tensor Inspection · Line chart', 'Tensor Inspection · Scatter plot'],
    },
  ];
  readonly visualizationLabels = {
    'Tensor Comparison · Line chart': 'Line chart',
    'Tensor Inspection · Line chart': 'Line chart',
    'Tensor Inspection · Scatter plot': 'Scatter plot',
  };
  /** The pair's row, as the token analysis named it: the server looks up its steps again. */
  private readonly query = computed<FinalLogitsAddress | null>(() => {
    const analysis = this.analysis();
    return analysis
      ? {
          turn: this.turn(),
          stage: analysis.stage,
          alignment: analysis.alignment,
          row: analysis.row,
          ...(analysis.first_divergence ? {first_divergence: true} : {}),
          transform: this.transform(),
        }
      : null;
  });
  constructor() {
    effect(() => {
      if (this.settings().logitTransform === 'centered') {
        untracked(() =>
          this.updateSettings({
            logitTransform: 'raw',
            logitValueRange: undefined,
            logitOffset: 0,
          }),
        );
      }
    });
    let previousSelection: string | null = null;
    let previousPair: string | null = null;
    effect((onCleanup) => {
      const capture = this.captureId();
      const analysis = this.analysis();
      const query = this.query();
      this.retryCount();
      const selection = JSON.stringify([capture, this.turn(), this.position(), this.transform()]);
      const pair = analysis ? JSON.stringify([analysis.ref_step, analysis.target_step]) : null;
      const selectionChanged = previousSelection !== null && selection !== previousSelection;
      const changed =
        selectionChanged || (pair !== null && previousPair !== null && pair !== previousPair);
      // Loading a restored selection establishes its pair without discarding its viewport.
      if (selectionChanged || pair !== null) {
        previousPair = pair;
      }
      previousSelection = selection;
      if (changed) {
        untracked(() => {
          this.clearElementSelection();
          this.updateSettings({
            logitOffset: 0,
            logitElementRange: undefined,
            logitValueRange: undefined,
          });
        });
      }
      onCleanup(
        this.summaryRequest.load(
          capture && query
            ? async (signal) => {
                const summary = await this.api.finalLogitsSummary(capture, query, signal);
                if (!answers(summary, query)) {
                  throw new Error('Logits response did not match the selected token position.');
                }
                return summary;
              }
            : null,
        ),
      );
    });
    // The line chart's bins, for its range, width and reduction, once its width is known. A
    // resize changes the width many times: only the last asks the server.
    let askedCapacity: number | null = null;
    effect((onCleanup) => {
      this.retryCount();
      const asked = this.binsAsked();
      const capacity = this.capacity();
      if (this.plot() !== 'index' || !this.capacityMeasured() || !asked) {
        return;
      }
      const resized = askedCapacity !== null && askedCapacity !== capacity;
      askedCapacity = capacity;
      onCleanup(
        this.answer(
          this.binsRequest,
          asked,
          (capture, payload: FinalLogitsBinsRequest, signal) =>
            this.api.finalLogitsBins(capture, payload, signal),
          resized ? SETTLE_MS : 1,
        ),
      );
    });
    // The scatter plot's points of the whole extent (with the overview's), and of a zoomed range.
    effect((onCleanup) => {
      this.retryCount();
      const asked = this.pointsAsked();
      if (this.plot() !== 'scatter' || !asked) {
        return;
      }
      onCleanup(
        this.answer(this.pointsRequest, asked, (capture, payload: FinalLogitsAddress, signal) =>
          this.api.finalLogitsPoints(capture, payload, signal),
        ),
      );
    });
    // A drag of the overview's window changes the range at every pointer move: only where it
    // rests asks the server.
    effect((onCleanup) => {
      this.retryCount();
      const asked = this.zoomedPointsAsked();
      if (this.plot() !== 'scatter') {
        return;
      }
      if (!asked) {
        this.zoomedPointsRequest.load(null);
        return;
      }
      onCleanup(
        this.answer(
          this.zoomedPointsRequest,
          asked,
          (capture, payload: FinalLogitsPointsRequest, signal) =>
            this.api.finalLogitsZoomedPoints(capture, payload, signal),
          SETTLE_MS,
        ),
      );
    });
    // The values of a selected element no point holds: its row of the table, by index. A held
    // arrow key selects many: only the one it rests on is looked up.
    effect((onCleanup) => {
      this.retryCount();
      const asked = this.elementAsked();
      if (!asked) {
        return;
      }
      onCleanup(
        this.answer(
          this.elementRequest,
          asked,
          (capture, payload: FinalLogitsRequest, signal) =>
            this.api.finalLogits(capture, payload, signal),
          SETTLE_MS,
        ),
      );
    });
    // A located element's values, once known: reset a value range that leaves it out.
    effect(() => {
      const index = this.revealed();
      if (index === null) {
        return;
      }
      if (this.selectedIndex() !== index) {
        untracked(() => this.revealed.set(null));
        return;
      }
      const row = this.elements().get(index) ?? this.lookedUp();
      if (row?.channel !== index) {
        return;
      }
      untracked(() => {
        this.revealed.set(null);
        const range = this.valueRange();
        if (
          typeof row.ref === 'number' &&
          typeof row.target === 'number' &&
          (row.ref < range.start ||
            row.target < range.start ||
            row.ref > range.end ||
            row.target > range.end)
        ) {
          this.updateSettings({logitValueRange: undefined});
        }
      });
    });
    effect((onCleanup) => {
      const capture = this.captureId();
      const pair = this.query();
      const query = pair && {
        ...pair,
        offset: this.savedOffset(),
        limit: this.pageSize,
        order: this.order(),
        direction: this.sortDirection(),
      };
      this.retryCount();
      onCleanup(
        this.request.load(
          capture && query ? (signal) => this.api.finalLogits(capture, query, signal) : null,
        ),
      );
    });
    inject(DestroyRef).onDestroy(() => {
      this.request.destroy();
      this.summaryRequest.destroy();
      this.binsRequest.destroy();
      this.pointsRequest.destroy();
      this.zoomedPointsRequest.destroy();
      this.elementRequest.destroy();
      this.locationAnimation?.cancel();
    });
  }
  /** Loads a chart's answer for a request, unless its last answer is already that request's,
   *  which is then taken again, forgetting why a newer one failed. The request is sent `delay`
   *  ms later, and never if another replaces it meanwhile: at least the task after the change,
   *  so that changes made together (a new token position and the ranges it resets) ask once. */
  private answer<P, T extends FinalLogitsAddress & {first_divergence: boolean}>(
    request: LatestRequest<Answer<T>>,
    asked: string,
    read: (capture: string, payload: P, signal: AbortSignal) => Promise<T>,
    delay = 1,
  ): () => void {
    if (untracked(() => request.last()?.request) === asked) {
      untracked(() => request.reuse());
      return () => {};
    }
    const [key, payload] = JSON.parse(asked) as [string, P];
    const [capture, query] = JSON.parse(key) as [string, FinalLogitsAddress];
    return request.load(
      async (signal) => {
        const value = await read(capture, payload, signal);
        if (!answers(value, query)) {
          throw new Error('Logits response did not match the selected token position.');
        }
        return {key, request: asked, value};
      },
      undefined,
      delay,
    );
  }
  private readonly savedOffset = computed(() => {
    const offset = this.settings().logitOffset ?? 0;
    return Number.isFinite(offset)
      ? Math.max(0, Math.floor(offset / this.pageSize) * this.pageSize)
      : 0;
  });
  private updateSettings(patch: Partial<LogitPanelState>): void {
    this.settingsChange.emit({...this.settings(), ...patch});
  }
  chooseView(label: string): void {
    this.displayOpen.set(false);
    if (label === 'Tensor Comparison · Line chart') {
      this.updateSettings({logitView: 'comparison'});
    } else if (
      label === 'Tensor Inspection · Line chart' ||
      label === 'Tensor Inspection · Scatter plot'
    ) {
      this.updateSettings({logitPlot: label.endsWith('Scatter plot') ? 'scatter' : 'index'});
    }
  }
  setPosition(position: number): void {
    if (!Number.isFinite(position) || this.fullRange().end <= this.fullRange().start) {
      return;
    }
    const next = Math.max(
      this.fullRange().start,
      Math.min(this.fullRange().end - 1, Math.floor(position)),
    );
    if (next === this.position()) {
      return;
    }
    this.updateSettings({logitPosition: next, logitOffset: 0});
    this.positionChange.emit(next);
  }
  commitPosition(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.setPosition(input.valueAsNumber);
    input.value = String(this.position());
  }
  columnSortDirection(column: FinalLogitsOrder): 'ascending' | 'descending' | 'none' {
    return column === this.order() ? this.sortDirection() : 'none';
  }
  sortBy(column: FinalLogitsOrder): void {
    const direction =
      column === this.order() && this.sortDirection() === 'ascending' ? 'descending' : 'ascending';
    this.updateSettings({logitOrder: column, logitSortDirection: direction, logitOffset: 0});
  }
  changePage(delta: number): void {
    if (this.valuesLoading()) {
      return;
    }
    const page = Math.max(0, Math.min(this.pages() - 1, this.pageIndex() + delta));
    if (page !== this.pageIndex()) {
      this.updateSettings({logitOffset: page * this.pageSize});
    }
  }
  retry(): void {
    this.retryCount.update((n) => n + 1);
  }
  private setElementSelection(index: number): void {
    this.selectedIndex.set(index);
    this.channelSelection.set(null);
    this.announcement.set(`Vocabulary index ${index} selected.`);
  }
  selectElement(index: number): void {
    this.setElementSelection(index);
    const ordinal = this.sortDirection() === 'descending' ? this.total() - 1 - index : index;
    if (
      this.order() === 'index' &&
      (ordinal < this.offset() || ordinal >= this.offset() + this.pageSize)
    ) {
      this.updateSettings({logitOffset: Math.floor(ordinal / this.pageSize) * this.pageSize});
    }
  }
  /** Reveal a chart selection in the table, without changing the chart viewport. */
  showElement(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) {
      return;
    }
    this.setElementSelection(index);
    const offset = Math.floor(index / this.pageSize) * this.pageSize;
    if (
      this.order() !== 'index' ||
      this.sortDirection() !== 'ascending' ||
      this.offset() !== offset
    ) {
      this.updateSettings({
        logitOrder: 'index',
        logitSortDirection: 'ascending',
        logitOffset: offset,
      });
    }
  }
  clearElementSelection(): void {
    this.selectedIndex.set(null);
    this.channelSelection.set(null);
    this.announcement.set('');
  }
  formatValue(value: FinalLogitsRow['ref']): string {
    return typeof value === 'string' ? value : formatTensorValue(value);
  }
  zoomElements(range: KvRangeBin): void {
    this.updateSettings({logitElementRange: range});
  }
  zoomValues(range: ValueRange): void {
    if (validValueRange(range)) {
      this.updateSettings({logitValueRange: range});
    }
  }
  resetRange(): void {
    this.updateSettings(
      this.plot() === 'index' ? {logitElementRange: undefined} : {logitValueRange: undefined},
    );
  }
  reductionLabel(value: ChannelReduction): 'Minimum' | 'Maximum' | 'Mean' {
    return value === 'min' ? 'Minimum' : value === 'max' ? 'Maximum' : 'Mean';
  }
  toggleDisplay(): void {
    if (this.plot() !== 'index') {
      return;
    }
    this.displayDraft.set(this.reduction());
    this.displayOpen.update((v) => !v);
  }
  setDisplayDraft(label: string): void {
    if (this.displayBinsDisabled()) {
      return;
    }
    this.displayDraft.set(label === 'Minimum' ? 'min' : label === 'Maximum' ? 'max' : 'mean');
  }
  applyDisplay(): void {
    if (this.displayDirty()) {
      this.updateSettings({logitElementReduction: this.displayDraft()});
    }
  }
  locateView(): void {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locatePosition(): void {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLElement>('.position-control input'),
    );
  }
  locateElement(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) {
      return;
    }
    this.selectElement(index);
    const patch: Partial<LogitPanelState> = {
      logitOrder: 'index',
      logitSortDirection: 'ascending',
      logitOffset: Math.floor(index / this.pageSize) * this.pageSize,
    };
    if (this.plot() === 'index') {
      const width = Math.min(this.total(), Math.max(16, this.capacity()));
      const start = Math.max(0, Math.min(this.total() - width, index - Math.floor(width / 2)));
      patch.logitElementRange = {start, end: start + width};
      this.channelSelection.set({start: index, end: index + 1, plane: null});
    } else {
      this.revealed.set(index);
    }
    this.updateSettings(patch);
    (this.plot() === 'index' ? this.lineChart() : this.scatterChart())?.locateKind('key');
  }
  private locateControl(target: HTMLElement | null): void {
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
    this.locationAnimation = animateAnalysisLocationFeedback(target, reduced);
  }
}
