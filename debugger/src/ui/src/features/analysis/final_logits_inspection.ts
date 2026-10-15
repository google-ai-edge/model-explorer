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
  FinalLogitsOrder,
  FinalLogitsResponse,
  FinalLogitsRow,
  FinalLogitsTransform,
  FinalLogitsVectorRequest,
  FinalLogitsVectorResponse,
} from '../../data/contracts/final_logits';
import type {TokenAnalysisPair} from '../../data/contracts/token_analysis';
import type {KvHeadChannel, KvRangeBin} from '../../data/contracts/kv';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {formatAnalysisMetric} from '../../shared/format/format';
import {animateAnalysisLocationFeedback} from '../../shared/locate_feedback';
import type {TokenTrendState} from '../conversation/token_trends';
import {KvRequest} from '../kv/kv_panel/kv_data';
import {KvChannelChart} from '../kv/kv_panel/kv_channel_chart';
import {KvScatterChart} from '../kv/kv_panel/kv_scatter_chart';
import {channelBinSize, type ChannelReduction} from '../kv/kv_panel/kv_channel_bins';
import {scatterExtent, validValueRange, type ValueRange} from '../kv/kv_panel/kv_scatter_model';
import {PositionRangeControl} from './position_range_control';
import {ValueRangeControl} from './value_range_control';
import {AnalysisViewActions} from './view_actions';
import {AnalysisRuntimeLegend} from './runtime_legend';

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
    AnalysisViewActions,
    AnalysisRuntimeLegend,
    PositionRangeControl,
    ValueRangeControl,
    KvChannelChart,
    KvScatterChart,
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
            <analysis-position-range
              label="Element Range"
              [range]="elementRange()"
              [full]="elementFull()"
              (rangeChange)="zoomElements($event)"
            />
            <analysis-view-actions
              [range]="elementRange()"
              [full]="elementFull()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          } @else {
            <analysis-value-range [range]="valueRange()" (rangeChange)="zoomValues($event)" />
            <analysis-view-actions
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
            <analysis-runtime-legend />
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
          <analysis-view-actions
            class="panel-expand"
            [showReset]="false"
            (expandView)="expand.emit()"
          />
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
      } @else if (error()) {
        <p class="state-message" role="alert">
          {{ error() }} <button mat-button (click)="retry()">Retry</button>
        </p>
      } @else if (!analysis()) {
        <p class="state-message">No captured logits selection at this token position.</p>
      } @else if (!total()) {
        <p class="state-message">No captured logits elements are available.</p>
      } @else {
        @for (notice of notices(); track notice) {
          <p class="state-message">{{ notice }}</p>
        }
        <div class="tensor-line-canvas" [class.expanded]="expanded()">
          @if (plot() === 'index') {
            <kv-channel-chart
              dimensionLabel="Vocabulary index"
              [groupLabels]="['Final logits']"
              [comparisonAllowed]="vectorRequest.data()?.compatible ?? false"
              [rows]="channels()"
              [range]="elementRange()"
              [full]="elementFull()"
              [reduction]="reduction()"
              [capacity]="capacity()"
              (capacityChange)="capacity.set($event)"
              [selected]="selectedIndex()"
              [channelSelection]="channelSelection()"
              (channelSelectionChange)="channelSelection.set($event)"
              (zoom)="zoomElements($event)"
              (select)="showElement($event)"
              (backgroundClick)="clearElementSelection()"
            />
          } @else {
            <kv-scatter-chart
              dimensionLabel="Vocabulary index"
              [groupLabels]="['Final logits']"
              [singlePlane]="true"
              [showGroupingNotice]="true"
              [rows]="scatterChannels()"
              [range]="valueRange()"
              [selected]="selectedIndex()"
              [messages]="scatterMessages()"
              (select)="showElement($event)"
              (zoom)="zoomValues($event)"
              (backgroundClick)="clearElementSelection()"
            />
          }
        </div>
        @if (plot() === 'index' && binSize() > 1) {
          <p class="plot-caption">
            {{ reductionLabel(reduction()) }} · up to {{ binSize() }} vocabulary indices per bin
          </p>
        }
      }
      <span class="sr-only" aria-live="polite">{{ announcement() }}</span>
    </section>`,
  styleUrl: './final_logits_inspection.scss',
})
export class FinalLogitsInspection {
  readonly panelId = input('');
  readonly turn = input.required<number>();
  readonly position = input.required<number>();
  readonly fullRange = input.required<KvRangeBin>();
  readonly analysis = input<TokenAnalysisPair | undefined>();
  readonly settings = input.required<TokenTrendState>();
  readonly expanded = input(false);
  readonly settingsChange = output<TokenTrendState>();
  readonly positionChange = output<number>();
  readonly expand = output<void>();
  readonly pageSize = 16;
  readonly request = new KvRequest<FinalLogitsResponse>();
  readonly vectorRequest = new KvRequest<FinalLogitsVectorResponse>();
  readonly loading = this.vectorRequest.loading;
  readonly error = this.vectorRequest.error;
  readonly valuesLoading = this.request.loading;
  readonly valuesError = this.request.error;
  readonly data = computed<FinalLogitsResponse | null>(() => {
    const vector = this.vectorRequest.data();
    const page = this.request.data();
    return vector
      ? {...vector, offset: page?.offset ?? 0, limit: this.pageSize, rows: page?.rows ?? []}
      : null;
  });
  private readonly api = inject(ReportApiService);
  private readonly state = inject(ReportStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly captureId = this.state.captureId;
  private readonly retryCount = signal(0);
  readonly selectedIndex = signal<number | null>(null);
  readonly channelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  readonly announcement = signal('');
  readonly lineChart = viewChild(KvChannelChart);
  readonly scatterChart = viewChild(KvScatterChart);
  readonly capacity = signal(128);
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
      this.vectorRequest.data()?.total ??
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
  readonly channels = computed<KvHeadChannel[]>(() => {
    const vector = this.vectorRequest.data();
    if (!vector) {
      return [];
    }
    return Array.from({length: vector.total}, (_, channel) => {
      const ref = vector.values.ref?.[channel] ?? null;
      const target = vector.values.target?.[channel] ?? null;
      const delta =
        vector.compatible &&
        typeof ref === 'number' &&
        typeof target === 'number' &&
        Number.isFinite(target - ref)
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
    });
  });
  readonly scatterChannels = computed(() =>
    this.vectorRequest.data()?.compatible ? this.channels() : [],
  );
  readonly scatterMessages = computed(() =>
    this.vectorRequest.data()?.compatible
      ? []
      : [
          this.vectorRequest.data()?.reason ??
            'Matching vocabulary and token context are required for a paired plot.',
        ],
  );
  readonly valueFull = computed(() => scatterExtent([this.scatterChannels()]));
  readonly valueRange = computed(() => this.settings().logitValueRange ?? this.valueFull());
  readonly notices = computed(() => {
    const data = this.vectorRequest.data();
    if (!data) {
      return [];
    }
    return [
      ...new Set(
        [
          data.reason,
          ...(['ref', 'target'] as const).flatMap((side) =>
            data[side].reason && !data.reason?.includes(data[side].reason!)
              ? [(side === 'ref' ? 'Reference: ' : 'Target: ') + data[side].reason]
              : [],
          ),
        ].filter((reason): reason is string => !!reason),
      ),
    ];
  });
  readonly pageValuesNotices = computed(() => {
    const notices = this.notices();
    const reason = this.request.data()?.reason;
    return reason ? [...notices.filter((notice) => !reason.includes(notice)), reason] : notices;
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
  private readonly query = computed<FinalLogitsVectorRequest>(() => ({
    turn: this.turn(),
    ref_step: this.analysis()?.ref_step ?? null,
    target_step: this.analysis()?.target_step ?? null,
    transform: this.transform(),
  }));
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
      const selection = JSON.stringify([capture, query.turn, this.position(), query.transform]);
      const pair = analysis ? JSON.stringify([query.ref_step, query.target_step]) : null;
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
        this.vectorRequest.load(
          capture && analysis
            ? async (signal) => {
                const vector = await this.api.finalLogitsVector(capture, query, signal);
                if (
                  vector.turn !== query.turn ||
                  vector.ref_step !== query.ref_step ||
                  vector.target_step !== query.target_step ||
                  vector.transform !== query.transform ||
                  !vector.complete
                ) {
                  throw new Error('Logits response did not match the selected token position.');
                }
                return vector;
              }
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const capture = this.captureId();
      const analysis = this.analysis();
      const query = {
        ...this.query(),
        offset: this.savedOffset(),
        limit: this.pageSize,
        order: this.order(),
        direction: this.sortDirection(),
      };
      this.retryCount();
      onCleanup(
        this.request.load(
          capture && analysis ? (signal) => this.api.finalLogits(capture, query, signal) : null,
        ),
      );
    });
    inject(DestroyRef).onDestroy(() => {
      this.request.destroy();
      this.vectorRequest.destroy();
      this.locationAnimation?.cancel();
    });
  }
  private readonly savedOffset = computed(() => {
    const offset = this.settings().logitOffset ?? 0;
    return Number.isFinite(offset)
      ? Math.max(0, Math.floor(offset / this.pageSize) * this.pageSize)
      : 0;
  });
  private updateSettings(patch: Partial<TokenTrendState>) {
    this.settingsChange.emit({...this.settings(), ...patch});
  }
  chooseView(label: string) {
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
  setPosition(position: number) {
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
  commitPosition(event: Event) {
    const input = event.target as HTMLInputElement;
    this.setPosition(input.valueAsNumber);
    input.value = String(this.position());
  }
  columnSortDirection(column: FinalLogitsOrder): 'ascending' | 'descending' | 'none' {
    return column === this.order() ? this.sortDirection() : 'none';
  }
  sortBy(column: FinalLogitsOrder) {
    const direction =
      column === this.order() && this.sortDirection() === 'ascending' ? 'descending' : 'ascending';
    this.updateSettings({logitOrder: column, logitSortDirection: direction, logitOffset: 0});
  }
  changePage(delta: number) {
    if (this.valuesLoading()) {
      return;
    }
    const page = Math.max(0, Math.min(this.pages() - 1, this.pageIndex() + delta));
    if (page !== this.pageIndex()) {
      this.updateSettings({logitOffset: page * this.pageSize});
    }
  }
  retry() {
    this.retryCount.update((n) => n + 1);
  }
  private setElementSelection(index: number) {
    this.selectedIndex.set(index);
    this.channelSelection.set(null);
    this.announcement.set(`Vocabulary index ${index} selected.`);
  }
  selectElement(index: number) {
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
  showElement(index: number) {
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
  clearElementSelection() {
    this.selectedIndex.set(null);
    this.channelSelection.set(null);
    this.announcement.set('');
  }
  formatValue(value: FinalLogitsRow['ref']) {
    return typeof value === 'string' ? value : formatAnalysisMetric('raw', value);
  }
  zoomElements(range: KvRangeBin) {
    this.updateSettings({logitElementRange: range});
  }
  zoomValues(range: ValueRange) {
    if (validValueRange(range)) {
      this.updateSettings({logitValueRange: range});
    }
  }
  resetRange() {
    this.updateSettings(
      this.plot() === 'index' ? {logitElementRange: undefined} : {logitValueRange: undefined},
    );
  }
  reductionLabel(value: ChannelReduction) {
    return value === 'min' ? 'Minimum' : value === 'max' ? 'Maximum' : 'Mean';
  }
  toggleDisplay() {
    if (this.plot() !== 'index') {
      return;
    }
    this.displayDraft.set(this.reduction());
    this.displayOpen.update((v) => !v);
  }
  setDisplayDraft(label: string) {
    if (this.displayBinsDisabled()) {
      return;
    }
    this.displayDraft.set(label === 'Minimum' ? 'min' : label === 'Maximum' ? 'max' : 'mean');
  }
  applyDisplay() {
    if (this.displayDirty()) {
      this.updateSettings({logitElementReduction: this.displayDraft()});
    }
  }
  locateView() {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locatePosition() {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLElement>('.position-control input'),
    );
  }
  locateIndex() {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.toolbar-range'));
  }
  locateElement(index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) {
      return;
    }
    this.selectElement(index);
    const patch: Partial<TokenTrendState> = {
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
      const row = this.channels()[index];
      const range = this.valueRange();
      if (
        typeof row?.ref === 'number' &&
        typeof row.target === 'number' &&
        (row.ref < range.start ||
          row.target < range.start ||
          row.ref > range.end ||
          row.target > range.end)
      ) {
        patch.logitValueRange = undefined;
      }
    }
    this.updateSettings(patch);
    (this.plot() === 'index' ? this.lineChart() : this.scatterChart())?.locateKind('key');
  }
  private locateControl(target: HTMLElement | null) {
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
