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
  TensorIndexControl,
  type TensorIndexChange,
  type TensorIndexDimension,
} from '../../widgets/charts/tensor_index_control';
import {animateAnalysisLocationFeedback} from '../../widgets/locate_feedback';
import {representativePosition} from '../../widgets/charts/range_bins';
import {ScatterChart} from '../../widgets/charts/scatter_chart';
import {ValueRangeControl} from '../../widgets/charts/value_range_control';
import {scatterExtent, type ValueRange} from '../../widgets/charts/scatter_model';
import {defaultTensorLocation} from './kv_tensor_location';
import {ChannelChart} from '../../widgets/charts/channel_chart';
import {
  KV_VISUALIZATIONS,
  KV_VISUALIZATION_OPTIONS,
  KV_VISUALIZATION_GROUPS,
  KV_VISUALIZATION_LABELS,
} from './kv_visualization';
import {ViewActions} from '../../widgets/charts/view_actions';
import {RuntimeLegend} from '../../widgets/charts/runtime_legend';
import {heatmapChartHeight} from '../../widgets/charts/heatmap_plot';
import {
  afterNextRender,
  Injector,
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  linkedSignal,
  signal,
  untracked,
  viewChild,
  input,
  output,
} from '@angular/core';
import {NgTemplateOutlet} from '@angular/common';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatIconModule} from '@angular/material/icon';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {KvApi} from '../../server/kv_api';
import {CaptureStore} from '../../server/capture_store';
import type {
  KvAnalysisLayer,
  KvMetric,
  KvPositionAlignment,
  KvRangeBin,
  KvRangeResponse,
  KvTokenStructure,
} from '../../server/contracts/kv';
import type {TurnStage} from '../../server/contracts/capture';
import {reasonText} from '../../server/reasons';
import {KvDataController} from './kv_data';
import type {KvCacheKind, KvPanelSource, KvTensorGroup} from './kv_details';
import {LatestRequest} from '../../server/latest_request';
import {KvHeadDetails} from './kv_head_details';
import {PositionRangeControl} from '../../widgets/charts/position_range_control';
import {channelBinSize, type ChannelReduction} from '../../widgets/charts/channel_bins';
import {readTensorHeads, type KvTensorHead} from './kv_tensor_data';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {
  kvContextRange,
  restoreKvView,
  type KvViewSnapshot,
  type KvViewIdentity,
} from './kv_view_state';
import {HeatmapControls} from '../../widgets/charts/heatmap_controls';
import {
  HEATMAP_DEFINITIONS,
  heatmapControls,
  defaultPositionReduction,
  type HeatmapReduction,
} from '../../widgets/charts/heatmap_model';
import {RangeChart} from '../../widgets/charts/range_chart';
import {RANGE_MAX_BINS, rangeBinCount} from '../../widgets/charts/range_geometry';
import {RangeSelection} from '../../widgets/charts/range_selection';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {formatMetricValue} from '../../widgets/format/format';
import type {ColumnSteps} from '../../widgets/format/step_labels';
import {MetricCatalogStore} from '../../server/metric_catalog';
import {columnSteps} from '../shared/column_steps';
import {metricHeat, metricScaleLabels, steppedScale} from '../../widgets/metric_scale/metric_scale';

/** The answers of one view of the chart: the selected cache's range, the other cache's, and the
 *  scale of the whole range that auto colors are relative to. */
interface KvDrawn {
  range: KvRangeResponse | null;
  other: KvRangeResponse | null;
  scale: KvRangeResponse | null;
}

/**
 * Interactive KV cache analysis panel rendering Tensor Comparison heatmaps and line charts
 * alongside per-head Tensor Inspection channel and scatter plots for a stage's captured KV state.
 */
@Component({
  selector: 'kv-panel',
  standalone: true,
  imports: [
    TensorIndexControl,
    PositionRangeControl,
    ChannelChart,
    ScatterChart,
    ValueRangeControl,
    ConfigPicker,
    KvHeadDetails,
    OverlayDialog,
    OverlayPanel,
    NgTemplateOutlet,
    RangeChart,
    HeatmapControls,
    ViewActions,
    RuntimeLegend,
    OverlayModule,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
  ],
  templateUrl: './kv_panel.ng.html',
  styleUrl: './kv_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvPanel implements KvPanelSource {
  readonly positionAlignment = input<KvPositionAlignment | null>(null);
  readonly panelId = input('');
  readonly selectedCurve = signal<string | null>(null);
  readonly stageTurn = input<number | null>(null);
  /** The server's text of a row's reasons. */
  readonly reasonText = reasonText;
  /** The server's stage: its KV snapshot and each side's positions. */
  readonly stage = input<TurnStage | null>(null);
  readonly initialView = input<KvViewSnapshot | null>(null);
  readonly inspect = output<KvPanel>();
  readonly inspectOverview = output<void>();
  readonly inspectStage = output<MouseEvent>();
  readonly chartSelection = input<RangeSelection | null>();
  readonly positionHover = output<RangeSelection | null>();
  private readonly rangeChart = viewChild(RangeChart);
  private readonly tensorLineChart = viewChild(ChannelChart);
  private readonly tensorScatterChart = viewChild(ScatterChart);
  private readonly heatmapControls = viewChild(HeatmapControls);
  locateMetric(): void {
    this.heatmapControls()?.locateMetric();
  }
  locateHeads(): void {
    this.heatmapControls()?.locateDimension('head');
  }
  private readonly locationInjector = inject(Injector);
  private scopeAnimation?: Animation;
  private readonly layerDisplay = viewChild<ElementRef<HTMLElement>>('layerDisplay');
  private flashScope(
    target: HTMLElement | null | undefined,
    feedbackTarget = target,
    block: ScrollLogicalPosition = 'center',
  ): void {
    if (!target) {
      return;
    }
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    // Even nearest scrolling can move the page when a containing scroll box is clipped.
    // A visible range needs only feedback, with no scroll request at all.
    if (block !== 'nearest' || !this.scopeVisible(target)) {
      target.scrollIntoView({
        block,
        inline: 'nearest',
        behavior: reduced ? 'instant' : 'smooth',
      });
    }
    this.scopeAnimation?.cancel();
    this.scopeAnimation = animateAnalysisLocationFeedback(feedbackTarget ?? target, reduced);
  }
  private scopeVisible(target: HTMLElement): boolean {
    const rect = target.getBoundingClientRect();
    let top = 0;
    let left = 0;
    let right = window.innerWidth;
    let bottom = window.innerHeight;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      const bounds = parent.getBoundingClientRect();
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
  locatePositions(): void {
    const control = this.host.nativeElement.querySelector<HTMLElement>(
      '.toolbar-range position-range-control',
    );
    this.flashScope(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
      'nearest',
    );
  }
  locateView(): void {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('.kv-view-picker > button'),
      undefined,
      'nearest',
    );
  }
  clearElementSelection(): void {
    this.tensorChannelsState.set({key: null, value: null});
    this.tensorChannelSelection.set(null);
  }
  locateTensorPosition(): void {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('.tensor-position-control input'),
      undefined,
      'nearest',
    );
  }
  locateTensorIndex(): void {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('tensor-index-control > button'),
      undefined,
      'nearest',
    );
  }
  locateLayers(): void {
    if (this.popup() !== 'view-display') {
      this.toggleChartDisplay();
    }
    afterNextRender(() => this.flashScope(this.layerDisplay()?.nativeElement), {
      injector: this.locationInjector,
    });
  }
  private readonly pendingCacheLocation = signal<{
    kind: KvCacheKind;
    position?: number;
    layer?: number;
  } | null>(null);
  locateCache(kind: KvCacheKind, position?: number, layer?: number): void {
    this.pendingCacheLocation.set({kind, position, layer});
  }

  readonly state = inject(CaptureStore);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly data = new KvDataController(inject(KvApi));
  readonly overviewData = new KvDataController(inject(KvApi));
  readonly otherOverviewData = new KvDataController(inject(KvApi));
  private readonly valueData = new KvDataController(inject(KvApi));
  readonly expanded = input(false);
  readonly expand = output<void>();
  resetView(): void {
    const full = this.fullRange();
    if (full) {
      this.zoomTo(full);
    }
  }
  private readonly highlightedLayerState = signal<number | null>(null);
  readonly highlightedLayer = this.highlightedLayerState.asReadonly();
  readonly lineChartLayers = signal<number[] | null>(null);
  private readonly heatmapLayers = signal<number[] | null>(null);
  readonly selectedHeatmapLayers = computed(() => {
    const heatmapLayers = this.heatmapLayers();
    return this.layerIds().filter((id) => heatmapLayers === null || heatmapLayers.includes(id));
  });
  readonly visibleMetricLayers = computed(() =>
    this.view() === 'heatmap' ? this.selectedHeatmapLayers() : this.selectedLineChartLayers(),
  );
  private readonly bothTensorsState = signal(true);
  readonly bothTensors = this.bothTensorsState.asReadonly();
  readonly selectedLineChartLayers = computed(() =>
    (this.lineChartLayers() ?? this.layerIds()).filter((id) => this.layerIds().includes(id)),
  );
  layerColor(layer: number): string {
    return `hsl(${(layer * 137.508 + 210) % 360} 62% 46%)`;
  }
  readonly otherChartRows = computed(() => this.drawn().other);
  readonly chartSeries = computed(() =>
    this.selectedLineChartLayers().flatMap((layer) => [
      {layer, kind: this.kind(), rows: this.chartRows(), color: this.layerColor(layer)},
      ...(this.bothTensors()
        ? [
            {
              layer,
              kind: this.kind() === 'key' ? ('value' as const) : ('key' as const),
              rows: this.otherChartRows()?.rows ?? [],
              color: this.layerColor(layer),
            },
          ]
        : []),
    ]),
  );
  chooseView(view: 'heatmap' | 'lineChart'): void {
    const wasTensor = this.tensorView();
    this.tensorView.set(false);
    this.viewState.set(view);
    const position = this.tensorPosition();
    if (wasTensor && this.expanded() && position !== null) {
      const bin = this.chartBins().find((b) => b.start <= position && b.end > position);
      this.position.set(bin?.start ?? position);
      this.selectionEnd.set(bin?.end ?? position + 1);
      this.layer.set(this.tensorLayer() ?? this.layer());
      this.column.set(false);
    }
    if (wasTensor && !this.tensorReady()) {
      this.inspectOverview.emit();
    } else if (wasTensor) {
      this.inspect.emit(this);
    }
  }
  readonly tensorView = signal(false);
  readonly tensorVisibleRuntimes = signal<readonly string[]>(['ref', 'target']);
  readonly tensorPlot = signal<'index' | 'scatter'>('index');
  private readonly tensorPositionState = signal<number | null>(null);
  readonly tensorPosition = this.tensorPositionState.asReadonly();
  private readonly tensorLayerState = signal<number | null>(null);
  readonly tensorLayer = this.tensorLayerState.asReadonly();
  private readonly tensorHeadState = signal<number | null>(null);
  readonly tensorHead = this.tensorHeadState.asReadonly();
  readonly tensorRetry = signal(0);
  private readonly pendingTensorLocation = signal<KvCacheKind | null>(null);
  private readonly tensorChannelsState = signal<Record<string, number | null>>({
    key: null,
    value: null,
  });
  readonly tensorChannels = this.tensorChannelsState.asReadonly();
  readonly tensorChannelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  selectTensorChannel(channel: number | null): void {
    this.tensorChannelSelection.set(null);
    this.tensorChannelsState.set({key: channel, value: channel});
    this.inspect.emit(this);
  }
  readonly tensorSelectedRange = computed(() => {
    const bin = this.tensorChannelSelection();
    if (bin) {
      return bin;
    }
    const channel = this.tensorChannels()['key'];
    return channel === null ? null : {start: channel, end: channel + 1};
  });
  tensorPlotMessage(kind: KvCacheKind): string {
    const request = kind === 'key' ? this.tensorKey : this.tensorValue;
    if (request.loading()) {
      return 'Loading head dimension values…';
    }
    if (request.error() || this.tensorEvidence(kind)?.error) {
      return 'Head dimension values unavailable';
    }
    if (!this.tensorEvidence(kind)?.evidence?.channels.length) {
      return 'No captured head dimension values';
    }
    return '';
  }

  readonly tensorValueZoom = signal<ValueRange | null>(null);
  readonly tensorValueFull = computed(() =>
    scatterExtent(
      (['key', 'value'] as const).map((kind) => this.tensorEvidence(kind)?.evidence?.channels ?? []),
    ),
  );
  readonly tensorValueRange = computed(() => this.tensorValueZoom() ?? this.tensorValueFull());
  readonly tensorChannelZoom = signal<KvRangeBin | null>(null);
  private readonly tensorChannelReductionState = signal<ChannelReduction>('mean');
  readonly tensorChannelReduction = this.tensorChannelReductionState.asReadonly();
  readonly tensorChannelCapacity = signal(128);
  readonly channelBinOptions = ['Minimum', 'Maximum', 'Mean'];
  readonly tensorChannelFull = computed(() => {
    const count = Math.max(
      0,
      ...(['key', 'value'] as const).flatMap((kind) =>
        this.sides.map((side) => this.tensorSource(kind, side)?.channel_count ?? 0),
      ),
    );
    return count > 0 && this.tensorHead() !== null ? {start: 0, end: count} : null;
  });
  readonly tensorChannelRange = computed(() => {
    const full = this.tensorChannelFull();
    const range = this.tensorChannelZoom();
    if (!full || !range || range.start >= full.end) {
      return full;
    }
    return {start: Math.max(full.start, range.start), end: Math.min(full.end, range.end)};
  });
  readonly tensorChannelBinSize = computed(() => {
    const range = this.tensorChannelRange();
    return range ? channelBinSize(range, this.tensorChannelCapacity()) : 1;
  });
  setChannelReduction(value: string): void {
    const reductions: Readonly<Record<string, ChannelReduction>> = {
      Minimum: 'min',
      Maximum: 'max',
      Mean: 'mean',
    };
    const reduction = reductions[value];
    if (reduction) {
      this.tensorChannelReductionState.set(reduction);
    }
  }

  readonly incrementTensorRetry = (value: number): number => value + 1;
  readonly tensorKey = new LatestRequest<KvTensorHead[]>();
  readonly tensorValue = new LatestRequest<KvTensorHead[]>();
  readonly tensorGroups = computed<readonly KvTensorGroup[]>(() => [
    {kind: 'key', label: 'Key cache', request: this.tensorKey},
    {kind: 'value', label: 'Value cache', request: this.tensorValue},
  ]);
  readonly tensorReady = computed(
    () => this.tensorPosition() !== null && this.tensorLayer() !== null,
  );
  readonly tensorHeads = computed(() =>
    Array.from(
      {length: Math.max(this.tensorHeadCount('key'), this.tensorHeadCount('value'))},
      (_, index) => index,
    ),
  );
  readonly tensorLayerOptions = computed(() => this.layerIds().map(String));
  readonly tensorHeadOptions = computed(() => this.tensorHeads().map(String));
  readonly visualizationOptions = KV_VISUALIZATION_OPTIONS;
  readonly visualizationGroups = KV_VISUALIZATION_GROUPS;
  readonly visualizationLabels = KV_VISUALIZATION_LABELS;
  readonly visualizationLabel = computed(() =>
    this.tensorView()
      ? 'Tensor Inspection · ' + (this.tensorPlot() === 'index' ? 'Line chart' : 'Scatter plot')
      : 'Tensor Comparison · ' + (this.view() === 'heatmap' ? 'Heatmap' : 'Line chart'),
  );
  chooseVisualization(label: string): void {
    const choice = KV_VISUALIZATIONS.find((option) => option.label === label);
    if (!choice) {
      return;
    }
    if (choice.data === 'Metric') {
      this.chooseView(choice.chart === 'Heatmap' ? 'heatmap' : 'lineChart');
    } else {
      this.showTensorValues();
      this.tensorPlot.set(choice.chart === 'Line chart' ? 'index' : 'scatter');
      const selection = this.tensorChannelSelection();
      if (this.tensorPlot() === 'scatter' && selection && selection.end - selection.start > 1) {
        this.tensorChannelSelection.set(null);
      }
    }
  }
  tokenRepresentative(bin: KvRangeBin): number {
    const kind = this.selectedCurve() ?? this.kind();
    const rows = kind === this.kind() ? this.chartRows() : (this.otherChartRows()?.rows ?? []);
    const row = rows.find((r) => r.layer === this.layer()) ?? rows[0];
    const cell = row?.bins.find((b) => b.start === bin.start && b.end === bin.end);
    return representativePosition(bin, this.positionReduction(), cell?.metrics[this.metric()]);
  }
  private showTensorValues(): void {
    if (this.tensorView()) {
      return;
    }
    const selected =
      this.chartSelection() === undefined ? this.inspectedRange() : this.chartSelection();
    const position =
      selected && (this.expanded() || selected.end - selected.start === 1)
        ? this.tokenRepresentative(selected)
        : this.tensorPosition();
    const layer = selected?.layer ?? this.tensorLayer();
    this.initializeTensorLocation(position, layer);
    this.tensorView.set(true);
    this.inspect.emit(this);
  }
  private initializeTensorLocation(
    position = this.tensorPosition(),
    layer = this.tensorLayer(),
  ): void {
    const range = this.visibleRange();
    if (!range) {
      return;
    }
    const full = this.fullRange();
    const location = defaultTensorLocation(
      this.context()?.layers ?? [],
      range,
      this.positionAlignment(),
      {
        position:
          position !== null && full && position >= full.start && position < full.end
            ? position
            : null,
        layer,
        head: this.tensorHead(),
      },
      this.selectedHeads(),
    );
    this.tensorPositionState.set(location?.position ?? null);
    this.tensorLayerState.set(location?.layer ?? null);
    this.tensorHeadState.set(location?.head ?? null);
  }
  setTensorPosition(value: string, inspect = true): void {
    const position = Number(value);
    const range = this.fullRange();
    if (
      !value.trim() ||
      !Number.isSafeInteger(position) ||
      !range ||
      position < range.start ||
      position >= range.end
    ) {
      return;
    }
    this.tensorPositionState.set(position);
    this.syncTensorSelection(inspect);
  }
  moveTensorPosition(delta: number): void {
    const position = this.tensorPosition();
    if (position !== null) {
      this.setTensorPosition(String(position + delta));
    }
  }
  readonly tensorIndexDimensions = computed<TensorIndexDimension[]>(() => [
    {
      id: 'layer',
      label: 'Layer',
      value: this.tensorLayer() === null ? null : String(this.tensorLayer()),
      options: this.tensorLayerOptions(),
    },
    {
      id: 'head',
      label: 'Head',
      value: this.tensorHead() === null ? null : String(this.tensorHead()),
      options: this.tensorHeadOptions(),
      disabled: !this.tensorReady(),
    },
  ]);
  setTensorIndex(change: TensorIndexChange): void {
    if (change.dimension === 'layer') {
      this.setTensorLayer(change.value);
    } else if (change.dimension === 'head') {
      this.setTensorHead(change.value);
    }
  }
  setTensorLayer(value: string): void {
    const layer = Number(value);
    if (!this.layerIds().includes(layer)) {
      return;
    }
    this.tensorLayerState.set(layer);
    if (!this.tensorHeads().includes(this.tensorHead() ?? -1)) {
      this.tensorHeadState.set(null);
      this.initializeTensorLocation();
    }
    this.syncTensorSelection();
  }
  setTensorHead(value: string): void {
    const head = Number(value);
    if (!this.tensorHeads().includes(head)) {
      return;
    }
    this.tensorHeadState.set(head);
    this.inspect.emit(this);
  }
  private syncTensorSelection(inspect = true): void {
    const position = this.tensorPosition();
    const layer = this.tensorLayer();
    if (position !== null && layer !== null) {
      this.selectPoint(layer, position);
      this.selectedCurve.set(null);
      if (!this.selectedLineChartLayers().includes(layer)) {
        this.lineChartLayers.set([...this.selectedLineChartLayers(), layer].sort((a, b) => a - b));
      }
    }
    if (inspect) {
      this.inspect.emit(this);
    }
  }
  tensorSource(kind: KvCacheKind, side: 'ref' | 'target'): KvTokenStructure | null {
    return (
      this.context()?.layers.find((row) => row.kind === kind && row.layer === this.tensorLayer())
        ?.token_structure?.[side] ?? null
    );
  }
  tensorHeadCount(kind: KvCacheKind): number {
    return Math.max(0, ...this.sides.map((side) => this.tensorSource(kind, side)?.head_count ?? 0));
  }
  tensorEvidence(kind: KvCacheKind): KvTensorHead | null {
    return (
      (kind === 'key' ? this.tensorKey : this.tensorValue)
        .data()
        ?.find((row) => row.head === this.tensorHead()) ?? null
    );
  }
  locateTensor(kind: KvCacheKind): void {
    this.pendingTensorLocation.set(kind);
  }
  locateTensorElement(kind: KvCacheKind, index: number): void {
    if (!this.tensorEvidence(kind)?.evidence?.channels.some((row) => row.channel === index)) {
      return;
    }
    if (this.tensorPlot() === 'index') {
      const full = this.tensorChannelFull();
      const range = this.tensorChannelRange();
      if (
        full &&
        range &&
        (index < range.start || index >= range.end || this.tensorChannelBinSize() > 1)
      ) {
        const width = Math.max(1, Math.min(range.end - range.start, this.tensorChannelCapacity()));
        const start = Math.max(
          full.start,
          Math.min(index - Math.floor(width / 2), full.end - width),
        );
        this.tensorChannelZoom.set({start, end: start + width});
      }
    } else {
      const range = this.tensorValueRange();
      const outside = (['key', 'value'] as const).some((cache) => {
        const row = this.tensorEvidence(cache)?.evidence?.channels.find(
          (c) => c.channel === index,
        );
        return (
          row &&
          [row.ref, row.target].some(
            (value) =>
              typeof value === 'number' &&
              Number.isFinite(value) &&
              (value < range.start || value > range.end),
          )
        );
      });
      if (outside) {
        this.tensorValueZoom.set(null);
      }
    }
    this.selectTensorChannel(index);
    this.locateTensor(kind);
  }

  private readonly resolvedIdentity = signal<KvViewIdentity | null>(null);
  private readonly stateNoticeState = signal('');
  readonly stateNotice = this.stateNoticeState.asReadonly();
  private readonly catalog = inject(MetricCatalogStore);
  /** The metrics the catalog's KV cache analysis offers, in its order. */
  readonly metrics = computed(() =>
    this.catalog
      .lookup()
      .choices('kv')
      .map((choice) => ({id: choice.id as KvMetric, label: choice.label})),
  );
  /** The selected metric as the catalog describes it. */
  readonly description = computed(() => this.catalog.lookup().metric(this.metric()));
  /** Whether the four thresholds of the metric color the chart. */
  readonly stepped = computed(() => steppedScale(this.description(), this.colorScale()));
  readonly analysis = this.data.metadata.data;
  readonly error = this.data.metadata.error;
  readonly loading = this.data.metadata.loading;
  private readonly retry = signal(0);
  retryAnalysis(): void {
    this.retry.update((n) => n + 1);
  }
  private readonly selectedContextId = signal('');
  private readonly kindState = signal<KvCacheKind>('key');
  readonly kind = this.kindState.asReadonly();
  private readonly metricState = signal<KvMetric>('cosine_similarity');
  readonly metric = this.metricState.asReadonly();
  private readonly colorScaleState = signal<'auto' | 'thresholds'>('auto');
  readonly colorScale = this.colorScaleState.asReadonly();
  readonly head = signal('min');
  readonly aggregationHeads = signal<number[] | null>(null);
  readonly selectedHeads = computed(() =>
    ['min', 'max', 'mean'].includes(this.head())
      ? (this.aggregationHeads() ?? this.heads())
      : [Number(this.head())],
  );
  readonly aggregation = computed(() => {
    const head = this.head();
    const fn: HeatmapReduction =
      head === 'min' || head === 'max' || head === 'mean' ? head : 'min';
    return {
      dimension: 'head',
      label: 'Heads',
      options: this.heads().map((h) => ({id: String(h), label: `Head ${h}`})),
      selected: this.selectedHeads().map(String),
      function: fn,
    };
  });
  setAggregation(value: {selection: string[] | null; function: HeatmapReduction}): void {
    this.aggregationHeads.set(value.selection?.map(Number) ?? null);
    this.head.set(value.function);
  }
  readonly positionReductionOverride = signal<HeatmapReduction | null>(null);
  readonly positionReduction = computed(
    () => this.positionReductionOverride() ?? defaultPositionReduction(this.description()),
  );
  readonly heatmapDimensions = computed(() =>
    heatmapControls(
      HEATMAP_DEFINITIONS.kv,
      {position: this.positionReduction(), head: this.head()},
      {head: this.heads().map((head) => ({id: String(head), label: `Head ${head}`}))},
    ).map((control) => ({
      ...control,
      disabled:
        control.id === 'position' && !this.chartBins().some((bin) => bin.end - bin.start > 1),
    })),
  );
  readonly positionBinControl = computed(() =>
    this.heatmapDimensions().find((control) => control.id === 'position'),
  );
  readonly displayBinsDisabled = computed(() =>
    this.tensorView()
      ? this.tensorChannelBinSize() === 1
      : (this.positionBinControl()?.disabled ?? true),
  );
  readonly displayBins = computed(() =>
    this.displayBinsDisabled()
      ? '-'
      : ({min: 'Minimum', max: 'Maximum', mean: 'Mean'} as const)[
          this.tensorView() ? this.tensorChannelReduction() : this.positionReduction()
        ],
  );
  readonly displayDefaultBins = computed(() =>
    this.displayBinsDisabled()
      ? '-'
      : ({min: 'Minimum', max: 'Maximum', mean: 'Mean'} as const)[
          this.tensorView() ? 'mean' : defaultPositionReduction(this.description())
        ],
  );
  readonly displaySeriesLabel = computed(() => (this.tensorView() ? 'Runtimes' : 'Layers'));
  readonly displaySeries = computed(() =>
    this.tensorView()
      ? [
          {id: 'ref', label: 'Ref', color: 'var(--me-primary-color)', dashed: false},
          {
            id: 'target',
            label: 'Target',
            color: 'var(--me-on-surface-variant-color)',
            dashed: true,
          },
        ]
      : this.layerIds().map((layer) => ({
          id: String(layer),
          label: `Layer ${layer}`,
          color: this.layerColor(layer),
          dashed: false,
        })),
  );
  readonly displayVisible = computed(() =>
    this.tensorView() ? [...this.tensorVisibleRuntimes()] : this.visibleMetricLayers().map(String),
  );
  readonly displayDefaultVisible = computed(() =>
    this.tensorView() ? ['ref', 'target'] : this.layerIds().map(String),
  );
  private readonly displayDraftState = signal<{bins: string; visible: string[]}>({
    bins: '-',
    visible: [],
  });
  readonly displayDraft = this.displayDraftState.asReadonly();
  private readonly displayQueryState = signal('');
  readonly displayQuery = this.displayQueryState.asReadonly();
  setDisplayQuery(query: string): void {
    this.displayQueryState.set(query);
  }
  readonly displayFilteredSeries = computed(() => {
    const query = this.displayQuery()
      .trim()
      .toLowerCase()
      .replace(/^l(?=\d)/, 'layer ');
    return this.displaySeries().filter((series) => series.label.toLowerCase().includes(query));
  });
  private sameDisplaySeries(a: readonly string[], b: readonly string[]): boolean {
    return a.length === b.length && a.every((id) => b.includes(id));
  }
  readonly displaySeriesDefault = computed(() =>
    this.sameDisplaySeries(this.displayDraft().visible, this.displayDefaultVisible()),
  );
  readonly displayDirty = computed(
    () =>
      (!this.displayBinsDisabled() && this.displayDraft().bins !== this.displayBins()) ||
      (this.displaySeries().length > 0 &&
        !this.sameDisplaySeries(this.displayDraft().visible, this.displayVisible())),
  );
  readonly displaySummary = computed(() => 'Display · Bins: ' + this.displayBins());
  toggleChartDisplay(): void {
    if (this.popup() === 'view-display') {
      this.closePopup();
      return;
    }
    this.displayDraftState.set({bins: this.displayBins(), visible: this.displayVisible()});
    this.displayQueryState.set('');
    this.popupState.set('view-display');
  }
  draftDisplayBins(bins: string): void {
    this.displayDraftState.update((draft) => ({...draft, bins}));
  }
  draftDisplayVisible(visible: string[]): void {
    this.displayDraftState.update((draft) => ({...draft, visible}));
  }
  toggleDisplaySeries(id: string): void {
    const visible = this.displayDraft().visible;
    this.draftDisplayVisible(
      visible.includes(id) ? visible.filter((item) => item !== id) : [...visible, id],
    );
  }
  highlightDisplaySeries(id: string | null): void {
    if (!this.tensorView()) {
      this.highlightedLayerState.set(id === null ? null : Number(id));
    }
  }
  applyChartDisplay(): void {
    const draft = this.displayDraft();
    if (!this.displayBinsDisabled()) {
      if (this.tensorView()) {
        this.setChannelReduction(draft.bins);
      } else {
        const binMap: Readonly<Record<string, HeatmapReduction>> = {
          Minimum: 'min',
          Maximum: 'max',
          Mean: 'mean',
        };
        const value = binMap[draft.bins];
        if (value) {
          this.setHeatmapDimension({id: 'position', value});
        }
      }
    }
    if (this.displaySeries().length) {
      if (this.tensorView()) {
        this.tensorVisibleRuntimes.set(
          draft.visible.filter((id) => id === 'ref' || id === 'target'),
        );
      } else {
        const visible = draft.visible.map(Number).filter((id) => this.layerIds().includes(id));
        if (this.view() === 'heatmap') {
          this.heatmapLayers.set(visible);
        } else {
          this.lineChartLayers.set(visible);
        }
        // Only clear an actual selected cell, never promote the default tensor cursor
        // into a one-position chart selection when changing display settings.
        const selected =
          this.chartSelection() === undefined ? this.inspectedRange() : this.chartSelection();
        if (selected?.layer != null && !visible.includes(selected.layer)) {
          this.position.set(selected.start);
          this.selectionEnd.set(selected.end);
          this.column.set(true);
          this.inspect.emit(this);
        }
      }
    }
    this.displayDraftState.set({bins: this.displayBins(), visible: this.displayVisible()});
  }
  setHeatmapMetric(metric: string): void {
    const match = this.metrics().find((choice) => choice.id === metric);
    if (match) {
      this.metricState.set(match.id);
    }
  }
  setHeatmapDimension(change: {id: string; value: string}): void {
    const control = this.heatmapDimensions().find((c) => c.id === change.id);
    if (
      !control ||
      control.disabled ||
      !control.choices.some((choice) => choice.id === change.value)
    ) {
      return;
    }
    if (change.id === 'head') {
      this.head.set(change.value);
    }
    if (
      change.id === 'position' &&
      (change.value === 'min' || change.value === 'max' || change.value === 'mean')
    ) {
      this.positionReductionOverride.set(change.value);
    }
  }

  private readonly viewState = signal<'heatmap' | 'lineChart'>('heatmap');
  readonly view = this.viewState.asReadonly();
  readonly layer = signal(0);
  readonly position = signal(0);
  readonly column = signal(false);
  private readonly popupState = signal<'view-display' | null>(null);
  readonly popup = this.popupState.asReadonly();
  /** The stage's Turn, or else the first Turn the capture index read: a rejected one has no KV. */
  readonly selectedTurn = computed(
    () => this.stageTurn() ?? this.state.session()?.turns.find((turn) => !turn.rejected)?.n ?? 1,
  );
  /** Which step wrote the K and V at a column, on each side, in the server's words. */
  readonly columnSteps = (column: number): ColumnSteps | null =>
    columnSteps(
      this.state.session(),
      this.selectedTurn(),
      column,
      this.positionAlignment(),
      'written',
      'kv',
    );
  private readonly identity = computed<KvViewIdentity | null>(() => {
    const captureId = this.state.captureId();
    return captureId ? {captureId, turn: this.selectedTurn()} : null;
  });
  private readonly queryIdentity = computed(() => {
    const current = this.identity();
    const loaded = this.resolvedIdentity();
    return current &&
      loaded &&
      current.captureId === loaded.captureId &&
      current.turn === loaded.turn
      ? {...loaded, alignment: this.positionAlignment()}
      : null;
  });
  readonly context = computed(() =>
    this.queryIdentity()
      ? this.analysis()?.contexts.find(
          (context) =>
            context.id === this.selectedContextId() && context.turn === this.selectedTurn(),
        )
      : undefined,
  );
  readonly layers = computed(() =>
    (this.context()?.layers.filter((l) => l.kind === this.kind()) ?? []).sort(
      (a, b) => a.layer - b.layer,
    ),
  );
  readonly selectedLayer = computed<KvAnalysisLayer | undefined>(
    () =>
      this.layers().find((l) => l.layer === this.layer()) ??
      this.context()?.layers.find((l) => l.layer === this.layer()) ??
      this.layers()[0] ??
      this.context()?.layers[0],
  );
  readonly layerIds = computed(() => this.layers().map((l) => l.layer));
  readonly selectionEnd = signal<number | null>(null);
  private readonly zoomRange = signal<KvRangeBin | null>(null);
  private readonly binCapacity = signal(24);
  readonly rangeData = this.data.range.data;
  readonly rangeLoading = this.data.range.loading;
  readonly rangeError = this.data.range.error;
  private readonly rangeRetry = signal(0);
  retryRange(): void {
    this.rangeRetry.update((n) => n + 1);
  }
  readonly scaleReference = computed(() => this.drawn().scale);
  readonly scaleLoading = this.data.scale.loading;
  readonly scaleError = computed(() =>
    this.data.scale.error() ? 'Color scale unavailable: ' + this.data.scale.error() : '',
  );
  readonly chartLoading = computed(
    () =>
      this.rangeLoading() ||
      this.valueData.range.loading() ||
      (this.colorScale() === 'auto' && this.scaleLoading()),
  );
  readonly chartError = computed(
    () =>
      this.rangeError() ||
      this.valueData.range.error() ||
      (this.colorScale() === 'auto' ? this.scaleError() : ''),
  );
  readonly chartHeight = computed(() =>
    heatmapChartHeight(this.selectedHeatmapLayers().length, this.bothTensors() ? 2 : 1),
  );
  readonly chartViewportHeight = computed(() =>
    this.expanded() ? 'calc(100dvh - 294px)' : this.chartHeight() + 'px',
  );
  readonly fullRange = computed(() => {
    const captured = kvContextRange(this.context());
    if (!captured) {
      return captured;
    }
    const alignment = this.positionAlignment();
    if (alignment) {
      // The columns the server says the stage's KV snapshot holds, of the stage's rows.
      const shown = alignment.columns?.kv;
      if (shown) {
        return shown.end > shown.start ? {start: shown.start, end: shown.end} : null;
      }
      return alignment.pairs.length
        ? {start: alignment.start, end: alignment.start + alignment.pairs.length}
        : null;
    }
    // A column is a position on both sides: from where the stage starts on either side.
    const positions = this.stage()?.positions;
    const starts = [positions?.ref, positions?.target].flatMap((side) =>
      side ? [side.start] : [],
    );
    if (!starts.length) {
      return captured;
    }
    const lo = Math.max(captured.start, Math.min(...starts));
    return lo < captured.end ? {...captured, start: lo} : null;
  });
  readonly visibleRange = computed(() => this.zoomRange() ?? this.fullRange());
  readonly visibleBins = computed<KvRangeBin[]>(() => {
    const range = this.visibleRange();
    if (!range) {
      return [];
    }
    const n = rangeBinCount(range, this.binCapacity());
    return Array.from({length: n}, (_, i) => ({
      start: range.start + Math.floor((i * (range.end - range.start)) / n),
      end: range.start + Math.floor(((i + 1) * (range.end - range.start)) / n),
    }));
  });
  /** The answers the chart draws, all of one view: the current view's once every one has arrived,
   *  until then the last view's. No answer is drawn alone, and a new view never empties it. */
  private readonly drawn = linkedSignal<{loading: boolean; view: KvDrawn}, KvDrawn>({
    source: () => ({
      loading: this.chartLoading(),
      view: {
        range: this.rangeData(),
        other: this.valueData.range.data(),
        scale: this.data.scale.data(),
      },
    }),
    computation: (now, previous) => (now.loading && previous ? previous.value : now.view),
  });
  /** Whether the chart waits for its first answer: it says so instead of drawing empty cells. */
  readonly chartPending = computed(() => this.chartLoading() && !this.drawn().range);
  /** Whether the panel waits for its first data: the stage's KV analysis, then every answer its
   *  first view draws (both caches' bins at the chart's capacity, their color scales and the
   *  overview), each of both alignments (AnalysisCache). A failure ends the wait. */
  readonly waiting = computed(
    () =>
      this.loading() ||
      this.chartLoading() ||
      this.valueData.scale.loading() ||
      this.overviewData.range.loading() ||
      this.otherOverviewData.range.loading() ||
      (!this.chartError() && !!this.fullRange() && !this.drawn().range),
  );
  readonly chartBins = computed(() => this.drawn().range?.bins ?? this.visibleBins());
  readonly chartRows = computed(() =>
    this.colorScale() === 'auto' && !this.scaleReference() ? [] : (this.drawn().range?.rows ?? []),
  );
  readonly selectedPosition = computed<number | undefined>(() => {
    const range = this.fullRange();
    const position = this.position();
    return range
      ? position >= range.start && position < range.end
        ? position
        : range.start
      : undefined;
  });
  readonly inspectedRange = computed<RangeSelection | null>(() => {
    const start = this.selectedPosition();
    const full = this.fullRange();
    if (start === undefined || !full) {
      return null;
    }
    return {
      start,
      end: Math.min(full.end, Math.max(start + 1, this.selectionEnd() ?? start + 1)),
      layer: this.column() ? null : (this.selectedLayer()?.layer ?? null),
    };
  });
  readonly heads = computed(() =>
    Array.from(
      {length: Math.max(0, ...(this.context()?.layers ?? []).map((row) => row.head_count ?? 0))},
      (_, i) => i,
    ),
  );
  readonly sides = ['ref', 'target'] as const;
  readonly chartFormat = (value: number | null): string => this.format(value);
  constructor() {
    effect(() => {
      const selected = this.chartSelection();
      if (!this.tensorView() || !selected || selected.end - selected.start !== 1) {
        return;
      }
      untracked(() => {
        // A selected token changes the inspected tensor, not which sidebar owns focus.
        if (selected.start !== this.tensorPosition()) {
          this.setTensorPosition(String(selected.start), false);
        }
      });
    });
    effect(() => {
      // Initial Tensor views can be restored before their metadata arrives.
      if (
        this.tensorView() &&
        this.context() &&
        this.fullRange() &&
        (this.tensorPosition() === null ||
          this.tensorLayer() === null ||
          this.tensorHead() === null)
      ) {
        this.initializeTensorLocation();
      }
    });
    effect(() => {
      this.tensorPosition();
      this.tensorLayer();
      this.tensorHead();
      this.context();
      this.tensorChannelsState.set({key: null, value: null});
      this.tensorChannelSelection.set(null);
    });
    effect(() => {
      const kind = this.pendingTensorLocation();
      const chart =
        this.tensorPlot() === 'scatter' ? this.tensorScatterChart() : this.tensorLineChart();
      if (kind && chart) {
        chart.locateKind(kind);
        this.pendingTensorLocation.set(null);
      }
    });
    const tensorApi = inject(KvApi);
    effect((onCleanup) => {
      const capture = this.state.captureId();
      const context = this.context();
      const position = this.tensorPosition();
      const layer = this.tensorLayer();
      const active = this.tensorView();
      const alignment = this.positionAlignment();
      const turn = this.selectedTurn();
      this.tensorRetry();
      for (const group of this.tensorGroups()) {
        // Heads only the metadata lists: a layer and kind of none is not read.
        const count = this.tensorHeadCount(group.kind);
        onCleanup(
          group.request.load(
            active && capture && context && position !== null && layer !== null && count
              ? (signal) =>
                  readTensorHeads(
                    tensorApi,
                    capture,
                    {turn, contextId: context.id, position, layer, kind: group.kind, alignment},
                    signal,
                  )
              : null,
          ),
        );
      }
    });

    effect(() => {
      const chart = this.rangeChart();
      const location = this.pendingCacheLocation();
      if (chart && location) {
        untracked(() => chart.locateKind(location.kind, location.position, location.layer));
        this.pendingCacheLocation.set(null);
      }
    });
    let previousAlignment: unknown = undefined;
    effect(() => {
      const alignment = this.positionAlignment();
      if (previousAlignment !== undefined && previousAlignment !== alignment) {
        this.zoomRange.set(null);
        this.selectionEnd.set(null);
      }
      previousAlignment = alignment;
    });
    inject(DestroyRef).onDestroy(() => {
      this.scopeAnimation?.cancel();
      this.data.destroy();
      this.valueData.destroy();
      this.overviewData.destroy();
      this.otherOverviewData.destroy();
    });
    effect((onCleanup) => {
      const identity = this.identity();
      const contextId = this.stage()?.kv.context_id;
      const saved = this.initialView() ?? undefined;
      this.retry();
      untracked(() => {
        this.resolvedIdentity.set(null);
        this.popupState.set(null);
        onCleanup(
          this.data.loadMetadata(identity, (analysis) => {
            if (!identity) {
              return;
            }
            const restored = restoreKvView(saved, analysis, identity);
            if (!saved) {
              // The catalog's defaults: the metric shown first and how KV heads combine.
              const kv = this.catalog.lookup().analysis('kv');
              restored.state.metric = (kv?.default_metric as KvMetric) ?? restored.state.metric;
              restored.state.head = kv?.default_head ?? restored.state.head;
            }
            this.applyView(restored.state);
            if (!saved) {
              const context = analysis.contexts.find((c) => c.id === contextId);
              this.selectedContextId.set(context?.id ?? '');
            }
            this.stateNoticeState.set(restored.notice);
            this.resolvedIdentity.set(identity);
          }),
        );
      });
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity();
      const contextId = this.context()?.id;
      const range = this.fullRange();
      const kind = this.kind();
      const head = this.head();
      const heads = this.aggregationHeads();
      const enabled = this.colorScale() === 'auto';
      this.rangeRetry();
      onCleanup(
        this.data.loadScale(
          identity && contextId && range && enabled
            ? {...identity, contextId, range, kind, head, heads, bins: 1}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity();
      const contextId = this.context()?.id;
      const range = this.fullRange();
      const kind = this.kind();
      const head = this.head();
      const heads = this.aggregationHeads();
      this.rangeRetry();
      const query =
        identity && contextId && range
          ? {
              ...identity,
              contextId,
              range,
              kind,
              head,
              heads,
              bins: 128,
              capacity: 'fixed' as const,
            }
          : null;
      onCleanup(this.overviewData.loadRange(query));
      onCleanup(
        this.otherOverviewData.loadRange(
          query && this.bothTensors() ? {...query, kind: kind === 'key' ? 'value' : 'key'} : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity();
      const contextId = this.context()?.id;
      const range = this.visibleRange();
      const capacity = this.binCapacity();
      const bins = rangeBinCount(range, capacity);
      const kind = this.kind();
      const head = this.head();
      const heads = this.aggregationHeads();
      this.rangeRetry();
      onCleanup(
        this.data.loadRange(
          identity && contextId && range
            ? {...identity, contextId, range, bins, kind, head, heads, capacity}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity();
      const contextId = this.context()?.id;
      const range = this.visibleRange();
      const full = this.fullRange();
      const kind = this.kind() === 'key' ? 'value' : 'key';
      const head = this.head();
      const heads = this.aggregationHeads();
      const capacity = this.binCapacity();
      const bins = rangeBinCount(range, capacity);
      this.rangeRetry();
      const enabled = this.bothTensors();
      onCleanup(
        this.valueData.loadRange(
          enabled && identity && contextId && range
            ? {...identity, contextId, range, kind, head, heads, bins, capacity}
            : null,
        ),
      );
      onCleanup(
        this.valueData.loadScale(
          this.bothTensors() && identity && contextId && full
            ? {...identity, contextId, range: full, kind, head, heads, bins: 1}
            : null,
        ),
      );
    });
  }
  captureView(): KvViewSnapshot {
    return {
      version: 1,
      contextId: this.selectedContextId(),
      kind: this.kind(),
      metric: this.metric(),
      colorScale: this.colorScale(),
      head: this.head(),
      headSelection: this.aggregationHeads(),
      positionReduction: this.positionReductionOverride(),
      view: this.view(),
      tensor: {
        active: this.tensorView(),
        plot: this.tensorPlot(),
        position: this.tensorPosition(),
        layer: this.tensorLayer(),
        head: this.tensorHead(),
        channelRange: this.tensorChannelZoom(),
        valueRange: this.tensorValueZoom(),
        channelReduction: this.tensorChannelReduction(),
        visibleRuntimes: [...this.tensorVisibleRuntimes()],
      },
      layer: this.layer(),
      position: this.position(),
      selectionEnd: this.selectionEnd(),
      lineChartLayers: this.lineChartLayers(),
      heatmapLayers: this.heatmapLayers(),
      bothTensors: this.bothTensors(),
      column: this.column(),
      zoomRange: this.zoomRange(),
    };
  }
  private applyView(view: KvViewSnapshot): void {
    this.selectedContextId.set(view.contextId);
    this.lineChartLayers.set(view.lineChartLayers ?? null);
    this.heatmapLayers.set(view.heatmapLayers ?? null);
    // The panel always plots the Key cache with the Value cache beside it.
    this.bothTensorsState.set(true);
    this.kindState.set('key');
    // The catalog may not have arrived yet; a saved metric it does not offer falls back then.
    const offered = this.metrics();
    this.metricState.set(
      !offered.length || offered.some((choice) => choice.id === view.metric)
        ? view.metric
        : ((this.catalog.lookup().analysis('kv')?.default_metric as KvMetric | undefined) ??
            view.metric),
    );
    this.colorScaleState.set(view.colorScale);
    this.head.set(view.head);
    this.aggregationHeads.set(view.headSelection ?? null);
    this.positionReductionOverride.set(view.positionReduction ?? null);
    this.viewState.set(view.view);
    this.tensorView.set(view.tensor?.active ?? false);
    this.tensorPlot.set(view.tensor?.plot ?? 'index');
    this.tensorPositionState.set(view.tensor?.position ?? null);
    this.tensorLayerState.set(view.tensor?.layer ?? null);
    this.tensorHeadState.set(view.tensor?.head ?? null);
    this.tensorChannelZoom.set(view.tensor?.channelRange ?? null);
    this.tensorValueZoom.set(view.tensor?.valueRange ?? null);
    this.tensorChannelReductionState.set(view.tensor?.channelReduction ?? 'mean');
    this.tensorVisibleRuntimes.set(view.tensor?.visibleRuntimes ?? ['ref', 'target']);
    this.layer.set(view.layer);
    this.position.set(view.position);
    this.selectionEnd.set(view.selectionEnd);
    this.column.set(view.column);
    this.zoomRange.set(view.zoomRange);
  }
  setCapacity(capacity: number): void {
    this.binCapacity.set(Math.max(1, Math.min(RANGE_MAX_BINS, Math.floor(capacity))));
  }
  zoomTo(range: KvRangeBin): void {
    const current = this.visibleRange();
    const full = this.fullRange();
    if (!current || !full) {
      return;
    }
    // The Plotly range slider can expand or pan outside the current viewport.
    const next = {
      start: Math.max(full.start, range.start),
      end: Math.min(full.end, range.end),
    };
    if (next.end <= next.start || (next.start === current.start && next.end === current.end)) {
      return;
    }
    this.zoomRange.set(next);
  }
  selectRange(selection: RangeSelection): void {
    if (this.chartLoading() || this.chartError()) {
      return;
    }
    this.position.set(selection.start);
    this.selectionEnd.set(selection.end);
    this.column.set(selection.layer === null);
    if (selection.layer !== null) {
      this.layer.set(selection.layer);
      if (this.view() === 'heatmap') {
        this.lineChartLayers.set(
          [...new Set([...this.selectedLineChartLayers(), selection.layer])].sort((a, b) => a - b),
        );
      }
    }
    this.inspect.emit(this);
  }
  metricLabel(value = this.metric()): string {
    return this.catalog.lookup().label(value);
  }
  format(value: number | null | undefined, metric = this.metric()): string {
    return formatMetricValue(this.catalog.lookup().metric(metric), value);
  }
  readonly scaleMax = computed(() =>
    (this.scaleReference()?.rows ?? []).reduce(
      (max, row) =>
        row.bins.reduce((value, bin) => Math.max(value, bin.metrics[this.metric()]?.max ?? 0), max),
      0.000001,
    ),
  );
  paletteColor(percent: number): string {
    return `color-mix(in srgb,var(--heat-high) ${Math.min(100, Math.max(0, percent))}%,var(--heat-low))`;
  }
  color(value: number | null): string {
    const heat = metricHeat(this.description(), value, this.colorScale(), this.scaleMax());
    return heat === null ? 'var(--kv-missing)' : this.paletteColor(heat * 100);
  }
  /** The labels along the color legend, from the catalog's scale of the metric. */
  readonly scale = computed(() =>
    metricScaleLabels(
      this.description(),
      this.colorScale(),
      this.scaleReference() ? this.scaleMax() : null,
    ),
  );
  chooseLayer(layer: number): void {
    this.layer.set(layer);
  }
  selectPoint(layer: number, position: number | undefined): void {
    if (position === undefined) {
      return;
    }
    this.layer.set(layer);
    this.position.set(position);
    this.selectionEnd.set(null);
    this.column.set(false);
  }
  closePopup(): void {
    this.highlightedLayerState.set(null);
    this.popupState.set(null);
  }
  statusLabel(status: string): string {
    return status === 'ok' ? 'Comparable' : status.replace(/_/g, ' ');
  }
}
