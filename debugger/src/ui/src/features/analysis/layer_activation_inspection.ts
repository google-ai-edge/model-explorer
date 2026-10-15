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
import type {PositionTensorResponse} from '../../data/contracts/position_analysis';
import type {KvHeadChannel, KvPositionAlignment, KvRangeBin} from '../../data/contracts/kv';
import type {KvRangeSelection} from '../kv/kv_panel/kv_range_types';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../shared/locate_feedback';
import {KvRequest} from '../kv/kv_panel/kv_data';
import {KvChannelChart} from '../kv/kv_panel/kv_channel_chart';
import {KvScatterChart} from '../kv/kv_panel/kv_scatter_chart';
import {channelBinSize, type ChannelReduction} from '../kv/kv_panel/kv_channel_bins';
import {scatterExtent, validValueRange, type ValueRange} from '../kv/kv_panel/kv_scatter_model';
import {PositionRangeControl} from './position_range_control';
import {ValueRangeControl} from './value_range_control';
import {AnalysisViewActions} from './view_actions';
import {AnalysisRuntimeLegend} from './runtime_legend';
import {TENSOR_POINT_LABELS} from './tensor_labels';
import {
  TensorIndexControl,
  type TensorIndexChange,
  type TensorIndexDimension,
} from './tensor_index_control';

export interface LayerActivationInspectionState {
  plot: 'index' | 'scatter';
  position: number;
  layer: number | null;
  anchor?: string;
  elementRange?: KvRangeBin | null;
  valueRange?: ValueRange | null;
  selection?: {channel: number} | null;
  reduction?: ChannelReduction;
  visibleRuntimes?: readonly string[];
}

@Component({
  selector: 'layer-activation-inspection',
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
    TensorIndexControl,
    PositionRangeControl,
    ValueRangeControl,
    KvChannelChart,
    KvScatterChart,
  ],
  template: ` <div
      class="controls"
      role="toolbar"
      aria-label="Intermediate Tensors inspection controls"
    >
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
            aria-label="Previous tensor position"
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
            aria-label="Next tensor position"
            [disabled]="position() >= fullRange().end - 1"
            (click)="setPosition(position() + 1)"
          >
            <mat-icon>chevron_right</mat-icon>
          </button>
        </label>
        <div class="toolbar-view layer-control">
          <span>Tensor Index</span>
          <analysis-tensor-index
            [dimensions]="tensorIndexDimensions()"
            (valueChange)="setTensorIndex($event)"
          />
        </div>
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
            <analysis-runtime-legend [runtimes]="visibleRuntimes()" />
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
              : 'Display · Bins: ' + (displayBinsDisabled() ? '-' : reductionLabel(reduction()))
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
              <label>Bins</label>
              <config-picker
                label="Bins"
                plural="functions"
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
                aria-label="Reset Bins to default"
                (click)="setDisplayDraft('Mean')"
              >
                Reset to default
              </button>
            </div>
            <div class="setting">
              <div class="series-heading">
                <label>Runtimes</label
                ><span class="series-count">{{ displayRuntimes().length }}/2</span>
              </div>
              <div class="series-options" role="group" aria-label="Visible runtimes">
                @for (series of runtimeSeries; track series.id) {
                  <label>
                    <input
                      type="checkbox"
                      [attr.aria-label]="'Show ' + series.label"
                      [checked]="displayRuntimes().includes(series.id)"
                      (change)="toggleDisplayRuntime(series.id)"
                    />
                    <i [style.border-top-color]="series.color" [class.dashed]="series.dashed"></i>
                    <span>{{ series.label }}</span>
                  </label>
                }
              </div>
              <div class="series-actions">
                <button
                  mat-button
                  [disabled]="displayRuntimes().length === 2"
                  aria-label="Reset Runtimes to default"
                  (click)="displayRuntimes.set(['ref', 'target'])"
                >
                  Reset to default
                </button>
                <button
                  mat-button
                  [disabled]="!displayRuntimes().length"
                  aria-label="Clear Runtimes"
                  (click)="displayRuntimes.set([])"
                >
                  Clear
                </button>
              </div>
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
      aria-label="Intermediate Tensors tensor visualization"
      [attr.aria-busy]="loading()"
    >
      @if (loading()) {
        <p class="state-message" role="status">Loading intermediate tensor…</p>
      } @else if (error()) {
        <p class="state-message" role="alert">
          {{ error() }} <button mat-button (click)="retry()">Retry</button>
        </p>
      } @else if (!total()) {
        <p class="state-message">No intermediate tensor elements are available.</p>
      } @else {
        @for (notice of notices(); track notice) {
          <p class="state-message">{{ notice }}</p>
        }
        <div class="tensor-line-canvas" [class.expanded]="expanded()">
          @if (plot() === 'index') {
            <kv-channel-chart
              dimensionLabel="Element index"
              [groupLabels]="['Intermediate Tensors']"
              [comparisonAllowed]="compatible()"
              [rows]="channels()"
              [visibleRuntimes]="visibleRuntimes()"
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
              (backgroundClick)="inspectStage.emit($event)"
            />
          } @else {
            <kv-scatter-chart
              dimensionLabel="Element index"
              [groupLabels]="['Intermediate Tensors']"
              [singlePlane]="true"
              [showGroupingNotice]="true"
              [rows]="scatterChannels()"
              [range]="valueRange()"
              [selected]="selectedIndex()"
              [messages]="scatterMessages()"
              (select)="showElement($event)"
              (zoom)="zoomValues($event)"
              (backgroundClick)="inspectStage.emit($event)"
            />
          }
        </div>
        @if (plot() === 'index' && binSize() > 1) {
          <p class="plot-caption">
            {{ reductionLabel(reduction()) }} · up to {{ binSize() }} elements per bin
          </p>
        }
      }
      <span class="sr-only" aria-live="polite">{{ announcement() }}</span>
    </section>`,
  styleUrl: './final_logits_inspection.scss',
  styles: '.layer-control analysis-tensor-index { width: 320px; }',
})
export class LayerActivationInspection {
  readonly turn = input.required<number>();
  readonly stage = input.required<string>();
  readonly alignment = input<KvPositionAlignment | null>(null);
  readonly fullRange = input.required<KvRangeBin>();
  readonly layers = input.required<number[]>();
  readonly anchors = input<readonly string[]>([]);
  readonly initial = input<LayerActivationInspectionState | null>(null);
  readonly initialPosition = input<number | null>(null);
  readonly initialLayer = input<number | null>(null);
  readonly expanded = input(false);
  readonly compare = output<'heatmap' | 'trends' | 'graph'>();
  readonly inspect = output<LayerActivationInspection>();
  readonly stateChange = output<LayerActivationInspection>();
  readonly expand = output<void>();
  readonly inspectStage = output<MouseEvent>();
  readonly positionHover = output<KvRangeSelection | null>();
  readonly request = new KvRequest<PositionTensorResponse>();
  readonly data = this.request.data;
  readonly loading = this.request.loading;
  readonly error = this.request.error;
  readonly valuesLoading = this.loading;
  readonly valuesError = this.error;
  private readonly api = inject(ReportApiService);
  private readonly state = inject(ReportStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly retryCount = signal(0);
  private readonly chosenPosition = signal<number | null>(null);
  private readonly chosenLayer = signal<number | null>(null);
  readonly anchor = signal('output');
  readonly plot = signal<'index' | 'scatter'>('index');
  readonly position = computed(() => {
    const range = this.fullRange(),
      value = this.chosenPosition();
    return Math.max(
      range.start,
      Math.min(
        Math.max(range.start, range.end - 1),
        value !== null && Number.isFinite(value) ? Math.floor(value) : range.start,
      ),
    );
  });
  readonly layer = computed(() => {
    const selected = this.chosenLayer(),
      layers = this.layers();
    return selected !== null && layers.includes(selected) ? selected : (layers[0] ?? null);
  });
  readonly tensorIndexDimensions = computed<TensorIndexDimension[]>(() => [
    {
      id: 'layer',
      label: 'Layer',
      value: this.layer() === null ? null : String(this.layer()),
      options: this.layers().map(String),
    },
    {
      id: 'anchor',
      label: 'Sync point',
      value: this.anchor(),
      options: this.anchors().length ? this.anchors() : ['output'],
      optionLabels: TENSOR_POINT_LABELS,
    },
  ]);
  readonly selectedIndex = signal<number | null>(null);
  readonly selection = computed(() =>
    this.selectedIndex() === null ? null : {channel: this.selectedIndex()!},
  );
  readonly channelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  readonly announcement = signal('');
  readonly lineChart = viewChild(KvChannelChart);
  readonly scatterChart = viewChild(KvScatterChart);
  readonly capacity = signal(128);
  readonly displayOpen = signal(false);
  readonly displayDraft = signal<ChannelReduction>('mean');
  readonly visibleRuntimes = signal<readonly string[]>(['ref', 'target']);
  readonly displayRuntimes = signal<string[]>(['ref', 'target']);
  readonly runtimeSeries = [
    {id: 'ref', label: 'Ref', color: 'var(--me-primary-color)', dashed: false},
    {id: 'target', label: 'Target', color: 'var(--me-on-surface-variant-color)', dashed: true},
  ];
  readonly reduction = signal<ChannelReduction>('mean');
  private readonly savedElementRange = signal<KvRangeBin | null>(null);
  private readonly savedValueRange = signal<ValueRange | null>(null);
  private locationAnimation?: Animation;
  readonly total = computed(() =>
    Math.max(this.data()?.reference?.values.length ?? 0, this.data()?.target?.values.length ?? 0),
  );
  readonly compatible = computed(() => {
    const ref = this.data()?.reference,
      target = this.data()?.target;
    return (
      !!ref &&
      !!target &&
      ref.values.length === target.values.length &&
      JSON.stringify(ref.shape) === JSON.stringify(target.shape)
    );
  });
  readonly channels = computed<KvHeadChannel[]>(() => {
    const data = this.data();
    if (!data) return [];
    const finite = (value: number | null | undefined) =>
      typeof value === 'number' && Number.isFinite(value) ? value : null;
    return Array.from({length: this.total()}, (_, channel) => {
      const ref = finite(data.reference?.values[channel]),
        target = finite(data.target?.values[channel]);
      const difference = ref === null || target === null ? null : target - ref;
      const delta =
        this.compatible() && difference !== null && Number.isFinite(difference) ? difference : null;
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
  readonly selectedChannel = computed(
    () => this.channels().find((row) => row.channel === this.selectedIndex()) ?? null,
  );
  readonly scatterChannels = computed(() => (this.compatible() ? this.channels() : []));
  readonly scatterMessages = computed(() =>
    this.compatible()
      ? []
      : ['Matching Reference and Target tensor shapes are required for a paired plot.'],
  );
  readonly elementFull = computed<KvRangeBin>(() => ({start: 0, end: Math.max(1, this.total())}));
  readonly elementRange = computed(() => {
    const full = this.elementFull(),
      saved = this.savedElementRange();
    if (!saved) return full;
    const start = Math.max(0, Math.min(full.end - 1, Math.floor(saved.start)));
    return {start, end: Math.max(start + 1, Math.min(full.end, Math.ceil(saved.end)))};
  });
  readonly valueFull = computed(() => scatterExtent([this.scatterChannels()]));
  readonly valueRange = computed(() => this.savedValueRange() ?? this.valueFull());
  readonly binSize = computed(() => channelBinSize(this.elementRange(), this.capacity()));
  readonly displayBinsDisabled = computed(() => this.binSize() === 1);
  readonly displayDirty = computed(
    () =>
      (!this.displayBinsDisabled() && this.displayDraft() !== this.reduction()) ||
      this.displayRuntimes().length !== this.visibleRuntimes().length ||
      this.displayRuntimes().some((side) => !this.visibleRuntimes().includes(side)),
  );
  readonly notices = computed(() => {
    const data = this.data();
    if (!data) return [];
    const notices: string[] = [];
    if (!data.reference) notices.push('Reference tensor is unavailable at this aligned position.');
    if (!data.target) notices.push('Target tensor is unavailable at this aligned position.');
    if (data.reference && data.target && !this.compatible())
      notices.push('Reference and Target tensor shapes differ. Values are shown independently.');
    return notices;
  });
  readonly visualizationLabel = computed(() =>
    this.plot() === 'scatter'
      ? 'Tensor Inspection · Scatter plot'
      : 'Tensor Inspection · Line chart',
  );
  readonly visualizationOptions = [
    'Tensor Comparison · Heatmap',
    'Tensor Comparison · Line chart',
    'Tensor Inspection · Line chart',
    'Tensor Inspection · Scatter plot',
    'Tensor Comparison · Graph Diff',
  ];
  readonly visualizationGroups = [
    {
      label: 'Tensor Comparison',
      options: ['Tensor Comparison · Heatmap', 'Tensor Comparison · Line chart', 'Tensor Comparison · Graph Diff'],
    },
    {
      label: 'Tensor Inspection',
      options: ['Tensor Inspection · Line chart', 'Tensor Inspection · Scatter plot'],
    },
  ];
  readonly visualizationLabels = {
    'Tensor Comparison · Graph Diff': 'Graph Diff',
    'Tensor Comparison · Heatmap': 'Heatmap',
    'Tensor Comparison · Line chart': 'Line chart',
    'Tensor Inspection · Line chart': 'Line chart',
    'Tensor Inspection · Scatter plot': 'Scatter plot',
  };
  constructor() {
    effect(() => {
      const saved = this.initial();
      untracked(() => {
        this.plot.set(saved?.plot === 'scatter' ? 'scatter' : 'index');
        this.chosenPosition.set(saved?.position ?? this.initialPosition());
        this.chosenLayer.set(saved?.layer ?? this.initialLayer());
        this.anchor.set(saved?.anchor ?? 'output');
        this.savedElementRange.set(
          saved?.elementRange && validValueRange(saved.elementRange) ? saved.elementRange : null,
        );
        this.savedValueRange.set(
          saved?.valueRange && validValueRange(saved.valueRange) ? saved.valueRange : null,
        );
        this.selectedIndex.set(saved?.selection?.channel ?? null);
        this.reduction.set(saved?.reduction ?? 'mean');
        const runtimes =
          saved?.visibleRuntimes?.filter((side) => side === 'ref' || side === 'target') ?? [
            'ref',
            'target',
          ];
        const current = this.visibleRuntimes();
        if (
          current.length !== runtimes.length ||
          current.some((side, index) => side !== runtimes[index])
        ) {
          this.visibleRuntimes.set(runtimes);
        }
      });
    });
    effect(() => {
      const position = this.initialPosition();
      if (position !== null) untracked(() => this.chosenPosition.set(position));
    });
    effect(() => {
      const layer = this.initialLayer();
      if (layer !== null) untracked(() => this.chosenLayer.set(layer));
    });
    let previous: string | null = null;
    effect((onCleanup) => {
      const capture = this.state.captureId(),
        turn = this.turn(),
        phase = this.stage(),
        alignment = this.alignment(),
        position = this.position(),
        layer = this.layer(),
        anchor = this.anchor();
      this.state.revision();
      this.retryCount();
      const identity = JSON.stringify([capture, turn, phase, alignment, position, layer, anchor]);
      if (previous !== null && identity !== previous)
        untracked(() => {
          this.clearElementSelection();
        });
      previous = identity;
      onCleanup(
        this.request.load(
          capture && layer !== null && this.fullRange().end > this.fullRange().start
            ? async (signal) => {
                const response = await this.api.positionTensor(
                  capture,
                  turn,
                  phase,
                  {position, layer, anchor, alignment},
                  signal,
                );
                if (
                  response.position !== position ||
                  response.layer !== layer ||
                  response.anchor !== anchor
                )
                  throw new Error(
                    'Tensor response did not match the selected position, layer, and sync point.',
                  );
                return response;
              }
            : null,
          (data) => {
            const selected = this.selectedIndex(),
              total = Math.max(data.reference?.values.length ?? 0, data.target?.values.length ?? 0);
            if (
              selected !== null &&
              (!Number.isInteger(selected) || selected < 0 || selected >= total)
            )
              this.clearElementSelection();
          },
        ),
      );
    });
    effect(() => {
      this.captureView();
      this.data();
      this.loading();
      this.error();
      this.channelSelection();
      untracked(() => this.stateChange.emit(this));
    });
    inject(DestroyRef).onDestroy(() => {
      this.request.destroy();
      this.locationAnimation?.cancel();
    });
  }
  captureView(): LayerActivationInspectionState {
    return {
      plot: this.plot(),
      position: this.position(),
      layer: this.layer(),
      anchor: this.anchor(),
      elementRange: this.savedElementRange(),
      valueRange: this.savedValueRange(),
      selection: this.selection(),
      reduction: this.reduction(),
      visibleRuntimes: [...this.visibleRuntimes()],
    };
  }
  chooseView(label: string) {
    this.displayOpen.set(false);
    if (label === 'Tensor Comparison · Graph Diff') this.compare.emit('graph');
    else if (label === 'Tensor Comparison · Heatmap') this.compare.emit('heatmap');
    else if (label === 'Tensor Comparison · Line chart') this.compare.emit('trends');
    else if (label === 'Tensor Inspection · Line chart') this.plot.set('index');
    else if (label === 'Tensor Inspection · Scatter plot') {
      this.plot.set('scatter');
      const selection = this.channelSelection();
      if (selection && selection.end - selection.start > 1) this.channelSelection.set(null);
    }
    if (label.startsWith('Tensor Inspection · ')) this.inspect.emit(this);
  }
  setPosition(value: number, inspect = true) {
    const range = this.fullRange();
    if (!Number.isSafeInteger(value) || value < range.start || value >= range.end) return;
    this.chosenPosition.set(value);
    if (inspect) this.inspect.emit(this);
  }
  commitPosition(event: Event) {
    const input = event.target as HTMLInputElement;
    this.setPosition(input.valueAsNumber);
    input.value = String(this.position());
  }
  setTensorIndex(change: TensorIndexChange) {
    if (change.dimension === 'layer') this.setLayer(Number(change.value));
    else if (change.dimension === 'anchor') this.setAnchor(change.value);
  }
  setLayer(layer: number) {
    if (!this.layers().includes(layer)) return;
    this.chosenLayer.set(layer);
    this.inspect.emit(this);
  }
  setAnchor(anchor: string, inspect = true) {
    const available = this.anchors().length ? this.anchors() : ['output'];
    if (!available.includes(anchor)) return;
    if (anchor !== this.anchor()) {
      this.clearElementSelection();
      this.anchor.set(anchor);
    }
    if (inspect) this.inspect.emit(this);
  }
  retry() {
    this.retryCount.update((value) => value + 1);
  }
  showElement(index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) return;
    this.selectedIndex.set(index);
    this.channelSelection.set(null);
    this.announcement.set(`Element ${index} selected.`);
    this.inspect.emit(this);
  }
  clearElementSelection() {
    this.selectedIndex.set(null);
    this.channelSelection.set(null);
    this.announcement.set('');
  }
  zoomElements(range: KvRangeBin) {
    if (validValueRange(range)) this.savedElementRange.set(range);
  }
  zoomValues(range: ValueRange) {
    if (validValueRange(range)) this.savedValueRange.set(range);
  }
  resetRange() {
    if (this.plot() === 'index') this.savedElementRange.set(null);
    else this.savedValueRange.set(null);
  }
  reductionLabel(value: ChannelReduction) {
    return value === 'min' ? 'Minimum' : value === 'max' ? 'Maximum' : 'Mean';
  }
  toggleDisplay() {
    if (this.plot() !== 'index') return;
    if (this.displayOpen()) {
      this.displayOpen.set(false);
      return;
    }
    this.displayDraft.set(this.reduction());
    this.displayRuntimes.set([...this.visibleRuntimes()]);
    this.displayOpen.set(true);
  }
  setDisplayDraft(label: string) {
    if (!this.displayBinsDisabled())
      this.displayDraft.set(label === 'Minimum' ? 'min' : label === 'Maximum' ? 'max' : 'mean');
  }
  toggleDisplayRuntime(side: string) {
    if (side !== 'ref' && side !== 'target') return;
    this.displayRuntimes.update((visible) =>
      visible.includes(side) ? visible.filter((value) => value !== side) : [...visible, side],
    );
  }
  applyDisplay() {
    if (!this.displayBinsDisabled()) this.reduction.set(this.displayDraft());
    this.visibleRuntimes.set([...this.displayRuntimes()]);
  }
  locateView() {
    this.locateControl(this.host.nativeElement.querySelector('.view-picker > button'));
  }
  locatePosition() {
    this.locateControl(this.host.nativeElement.querySelector('.position-control input'));
  }
  locateLayer() {
    this.locateControl(this.host.nativeElement.querySelector('analysis-tensor-index > button'));
  }
  locateIndex() {
    this.locateControl(this.host.nativeElement.querySelector('.toolbar-range'));
  }
  locateElements() {
    this.locateIndex();
  }
  locateValues() {
    this.locateIndex();
  }
  locateElement(index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) return;
    this.showElement(index);
    if (this.plot() === 'index') {
      const range = this.elementRange();
      if (index < range.start || index >= range.end || this.binSize() > 1) {
        const width = Math.max(1, Math.min(range.end - range.start, this.capacity()));
        const start = Math.max(0, Math.min(this.total() - width, index - Math.floor(width / 2)));
        this.savedElementRange.set({start, end: start + width});
      }
    } else {
      const row = this.channels()[index],
        range = this.valueRange();
      if (
        typeof row.ref === 'number' &&
        typeof row.target === 'number' &&
        (row.ref < range.start ||
          row.target < range.start ||
          row.ref > range.end ||
          row.target > range.end)
      )
        this.savedValueRange.set(null);
    }
    this.locateTensor();
  }
  locateTensor() {
    (this.plot() === 'index' ? this.lineChart() : this.scatterChart())?.locateKind('key');
  }
  private locateControl(target: HTMLElement | null) {
    if (!target) return;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!this.controlVisible(target)) {
      target.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
        behavior: reduced ? 'instant' : 'smooth',
      });
    }
    this.locationAnimation?.cancel();
    this.locationAnimation = animateAnalysisLocationFeedback(target, reduced);
  }
  private controlVisible(target: HTMLElement): boolean {
    const rect = target.getBoundingClientRect();
    let top = 0,
      left = 0,
      right = window.innerWidth,
      bottom = window.innerHeight;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent),
        bounds = parent.getBoundingClientRect();
      if (/auto|scroll|hidden|clip/.test(style.overflowX)) {
        left = Math.max(left, bounds.left + parent.clientLeft);
        right = Math.min(right, bounds.left + parent.clientLeft + parent.clientWidth);
      }
      if (/auto|scroll|hidden|clip/.test(style.overflowY)) {
        top = Math.max(top, bounds.top + parent.clientTop);
        bottom = Math.min(bottom, bounds.top + parent.clientTop + parent.clientHeight);
      }
    }
    return rect.top >= top && rect.bottom <= bottom && rect.left >= left && rect.right <= right;
  }
}
