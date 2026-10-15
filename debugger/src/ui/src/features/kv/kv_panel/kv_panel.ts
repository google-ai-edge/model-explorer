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
} from '../../analysis/tensor_index_control';
import {animateAnalysisLocationFeedback} from '../../../shared/locate_feedback';
import {representativePosition} from '../../conversation/analysis_token_strip/analysis_token_model';
import {KvScatterChart} from './kv_scatter_chart';
import {ValueRangeControl} from '../../analysis/value_range_control';
import {scatterExtent, type ValueRange} from './kv_scatter_model';
import {defaultTensorLocation} from './kv_tensor_location';
import {KvChannelChart} from './kv_channel_chart';
import {
  KV_VISUALIZATIONS,
  KV_VISUALIZATION_OPTIONS,
  KV_VISUALIZATION_GROUPS,
  KV_VISUALIZATION_LABELS,
} from './kv_visualization';
import {AnalysisViewActions} from '../../analysis/view_actions';
import {AnalysisRuntimeLegend} from '../../analysis/runtime_legend';
import {heatmapChartHeight} from '../../analysis/heatmap_plot';
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
  signal,
  untracked,
  viewChild,
  input,
  output,
  TemplateRef,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {NgTemplateOutlet} from '@angular/common';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule} from '@angular/cdk/a11y';
import {MatIconModule} from '@angular/material/icon';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import type {
  KvAnalysisContext,
  KvAnalysisLayer,
  KvMetric,
  KvSnapshotIdentity,
  KvTokenStructure,
  KvTokenMetrics,
} from '../../../data/contracts/kv';
import {KvDataController, KvRequest} from './kv_data';
import {KvHeadDetails} from './kv_head_details';
import {PositionRangeControl} from '../../analysis/position_range_control';
import {channelBins, channelBinSize, type ChannelReduction} from './kv_channel_bins';
import {readTensorHeads, type KvTensorHead} from './kv_tensor_data';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {KvViewStore} from './kv_view_store';
import {
  kvContextRange,
  restoreKvView,
  type KvViewSnapshot,
  type KvExplicitEntry,
  type KvViewIdentity,
  type KvHeadViewState,
} from './kv_view_state';
import {HeatmapControls} from '../../analysis/heatmap_controls';
import {
  HEATMAP_DEFINITIONS,
  heatmapControls,
  defaultPositionReduction,
  type HeatmapReduction,
} from '../../analysis/heatmap_model';
import {KvRangeChart} from './kv_range_chart';
import {KV_RANGE_MAX_BINS, rangeBinCount} from './kv_range_geometry';
import type {KvRangeBin, KvRangeDetailRow, KvFindResult} from '../../../data/contracts/kv';
import {KvRangeSelection} from './kv_range_types';
import {KvHeadInspector, KvHeadSelection} from './kv_head_inspector';
import {KvMetricValues, KV_DETAIL_METRICS, formatKvMetric} from './kv_metric_values';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';
import {FindNavigator} from '../../../shared/find_navigator/find_navigator';
import {BookmarkStore} from '../../../shared/bookmark_store/bookmark_store';
import {KV_BOOKMARK_CODEC, SavedPoint, kvBookmarkKey, kvBookmarkLabel} from './kv_bookmarks';
import {OverlayDialog, OverlayPanel} from '../../../shared/overlay_dialog/overlay_dialog';
import {formatSignificant} from '../../../shared/format/format';
import {KvFindController} from './kv_find_controller';

const DEFAULT_HEAD_VIEW: KvHeadViewState = {
  view: 'index',
  numericalExpanded: true,
  channelsExpanded: true,
};
export const KV_METRICS: {id: KvMetric; label: string}[] = [
  {id: 'cosine_similarity', label: 'Cosine similarity'},
  {id: 'relative_l2', label: 'Relative L2'},
  {id: 'max_abs', label: 'Max abs error'},
];

@Component({
  selector: 'kv-panel',
  standalone: true,
  imports: [
    TensorIndexControl,
    PositionRangeControl,
    KvChannelChart,
    KvScatterChart,
    ValueRangeControl,
    ConfigPicker,
    KvHeadDetails,
    OverlayDialog,
    OverlayPanel,
    FindNavigator,
    ResizeHandle,
    FormsModule,
    NgTemplateOutlet,
    KvRangeChart,
    HeatmapControls,
    AnalysisViewActions,
    AnalysisRuntimeLegend,
    KvHeadInspector,
    KvMetricValues,
    ExplorerInfoSection,
    ExplorerInfoValue,
    OverlayModule,
    A11yModule,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
  ],
  templateUrl: './kv_panel.ng.html',
  styleUrl: './kv_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvPanel {
  readonly positionAlignment = input<
    import('../../../data/contracts/kv').KvPositionAlignment | null
  >(null);
  readonly panelId = input('');
  readonly embedded = input(false);
  readonly explicitEntry = input<KvExplicitEntry | null>(null);
  readonly selectedCurve = signal<string | null>(null);
  readonly stageTurn = input<number | null>(null);
  readonly stageMoment = input<string | null>(null);
  readonly initialView = input<KvViewSnapshot | null>(null);
  readonly inspect = output<KvPanel>();
  readonly inspectOverview = output<void>();
  readonly inspectStage = output<MouseEvent>();
  readonly chartSelection = input<KvRangeSelection | null>();
  readonly positionHover = output<KvRangeSelection | null>();
  private readonly rangeChart = viewChild(KvRangeChart);
  private readonly tensorLineChart = viewChild(KvChannelChart);
  private readonly tensorScatterChart = viewChild(KvScatterChart);
  private readonly heatmapControls = viewChild(HeatmapControls);
  locateMetric() {
    this.heatmapControls()?.locateMetric();
  }
  locateHeads() {
    this.heatmapControls()?.locateDimension('head');
  }
  private readonly locationInjector = inject(Injector);
  private scopeAnimation?: Animation;
  private readonly layerDisplay = viewChild<ElementRef<HTMLElement>>('layerDisplay');
  private flashScope(
    target: HTMLElement | null | undefined,
    feedbackTarget = target,
    block: ScrollLogicalPosition = 'center',
  ) {
    if (!target) return;
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
  locatePositions() {
    const control = this.host.nativeElement.querySelector<HTMLElement>(
      '.toolbar-range analysis-position-range',
    );
    this.flashScope(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
      'nearest',
    );
  }
  locateView() {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('.kv-view-picker > button'),
      undefined,
      'nearest',
    );
  }
  clearElementSelection() {
    this.tensorChannels.set({key: null, value: null});
    this.tensorChannelSelection.set(null);
  }
  locateTensorPosition() {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('.tensor-position-control input'),
      undefined,
      'nearest',
    );
  }
  locateTensorIndex() {
    this.flashScope(
      this.host.nativeElement.querySelector<HTMLElement>('analysis-tensor-index > button'),
      undefined,
      'nearest',
    );
  }
  locateLayers() {
    if (this.popup() !== 'view-display') this.toggleChartDisplay();
    afterNextRender(() => this.flashScope(this.layerDisplay()?.nativeElement), {
      injector: this.locationInjector,
    });
  }
  private readonly pendingCacheLocation = signal<{
    kind: 'key' | 'value';
    position?: number;
    layer?: number;
  } | null>(null);
  locateCache(kind: 'key' | 'value', position?: number, layer?: number) {
    this.pendingCacheLocation.set({kind, position, layer});
  }
  readonly detailsTemplate = viewChild<TemplateRef<unknown>>('sharedDetails');

  readonly state = inject(ReportStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly data = new KvDataController(inject(ReportApiService));
  readonly overviewData = new KvDataController(inject(ReportApiService));
  readonly otherOverviewData = new KvDataController(inject(ReportApiService));
  private readonly valueData = new KvDataController(inject(ReportApiService));
  readonly expanded = input(false);
  readonly expand = output<void>();
  resetView() {
    const full = this.fullRange();
    if (full) this.zoomTo(full);
  }
  readonly highlightedLayer = signal<number | null>(null);
  readonly trendLayers = signal<number[] | null>(null);
  readonly heatmapLayers = signal<number[] | null>(null);
  readonly selectedHeatmapLayers = computed(() =>
    this.layerIds().filter(
      (id) => this.heatmapLayers() === null || this.heatmapLayers()!.includes(id),
    ),
  );
  readonly visibleMetricLayers = computed(() =>
    this.view() === 'heatmap' ? this.selectedHeatmapLayers() : this.selectedTrendLayers(),
  );
  readonly bothTensors = signal(true);
  readonly layerSearch = signal('');
  readonly visibleLayerIds = computed(() =>
    this.layerIds().filter((id) =>
      ('L' + id).toLowerCase().includes(this.layerSearch().toLowerCase()),
    ),
  );
  readonly selectedTrendLayers = computed(() =>
    (this.trendLayers() ?? this.layerIds()).filter((id) => this.layerIds().includes(id)),
  );
  layerColor(layer: number) {
    return `hsl(${(layer * 137.508 + 210) % 360} 62% 46%)`;
  }
  readonly otherChartRows = this.valueData.range.data;
  readonly chartSeries = computed(() =>
    this.selectedTrendLayers().flatMap((layer) => [
      {layer, kind: this.kind(), rows: this.chartRows(), color: this.layerColor(layer)},
      ...(this.bothTensors()
        ? [
            {
              layer,
              kind: this.kind() === 'key' ? 'value' : 'key',
              rows: this.valueData.range.data()?.rows ?? [],
              color: this.layerColor(layer),
            },
          ]
        : []),
    ]),
  );
  toggleTrendLayer(layer: number) {
    const current = this.selectedTrendLayers();
    this.trendLayers.set(
      current.includes(layer)
        ? current.filter((l) => l !== layer)
        : [...current, layer].sort((a, b) => a - b),
    );
  }
  chooseView(view: 'heatmap' | 'trends') {
    const wasTensor = this.tensorView();
    this.tensorView.set(false);
    this.view.set(view);
    if (wasTensor && this.expanded() && this.tensorPosition() !== null) {
      const position = this.tensorPosition()!;
      const bin = this.chartBins().find((b) => b.start <= position && b.end > position);
      this.position.set(bin?.start ?? position);
      this.selectionEnd.set(bin?.end ?? position + 1);
      this.layer.set(this.tensorLayer() ?? this.layer());
      this.column.set(false);
    }
    if (wasTensor && !this.tensorReady()) this.inspectOverview.emit();
    else if (wasTensor) this.inspect.emit(this);
  }
  readonly tensorView = signal(false);
  readonly tensorVisibleRuntimes = signal<readonly string[]>(['ref', 'target']);
  toggleTensorRuntime(side: string) {
    if (side !== 'ref' && side !== 'target') return;
    this.tensorVisibleRuntimes.update((visible) =>
      visible.includes(side) ? visible.filter((value) => value !== side) : [...visible, side],
    );
  }
  readonly tensorPlot = signal<'index' | 'scatter'>('index');
  readonly tensorPosition = signal<number | null>(null);
  readonly tensorLayer = signal<number | null>(null);
  readonly tensorHead = signal<number | null>(null);
  readonly tensorRetry = signal(0);
  private readonly pendingTensorLocation = signal<string | null>(null);
  readonly tensorChannels = signal<Record<string, number | null>>({key: null, value: null});
  readonly tensorChannelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  selectTensorChannel(kind: string, channel: number | null) {
    this.tensorChannelSelection.set(null);
    this.tensorChannels.set({key: channel, value: channel});
    this.inspect.emit(this);
  }
  readonly tensorSelectedRange = computed(() => {
    const bin = this.tensorChannelSelection();
    if (bin) return bin;
    const channel = this.tensorChannels()['key'];
    return channel === null ? null : {start: channel, end: channel + 1};
  });
  readonly tensorSelectionProperties = computed(() => {
    const range = this.tensorSelectedRange();
    const binned = range !== null && range.end - range.start > 1;
    return [
      {label: 'Head', value: range ? String(this.tensorHead() ?? '-') : '-'},
      {label: 'Dimension', value: 'Head dimension'},
      {
        label: 'Range',
        value: range ? (binned ? `${range.start}–${range.end - 1}` : String(range.start)) : '-',
      },
      {
        label: 'Aggregation',
        value: binned
          ? ({min: 'Minimum', max: 'Maximum', mean: 'Mean'} as const)[this.tensorChannelReduction()]
          : '-',
      },
    ];
  });
  readonly tensorSelectionValues = computed(() => {
    const range = this.tensorSelectedRange();
    return this.tensorGroups().map(({kind, label, request}) => {
      const channels =
        !request.loading() && !request.error()
          ? (this.tensorEvidence(kind)?.evidence?.channels ?? [])
          : [];
      const bin = range ? channelBins(channels, range, 1, this.tensorChannelReduction())[0] : null;
      const ref = bin?.ref ?? null,
        target = bin?.target ?? null;
      // A bin's extrema can come from different channels on each side.
      const delta =
        range && range.end - range.start === 1 && ref !== null && target !== null
          ? target - ref
          : null;
      return {kind, label, values: [ref, target, delta, delta === null ? null : Math.abs(delta)]};
    });
  });
  tensorPlotMessage(kind: string) {
    const request = kind === 'key' ? this.tensorKey : this.tensorValue;
    if (request.loading()) return 'Loading head dimension values…';
    if (request.error() || this.tensorEvidence(kind)?.error)
      return 'Head dimension values unavailable';
    if (!this.tensorEvidence(kind)?.evidence?.channels.length)
      return 'No captured head dimension values';
    return '';
  }

  readonly tensorValueZoom = signal<ValueRange | null>(null);
  readonly tensorValueFull = computed(() =>
    scatterExtent(
      ['key', 'value'].map((kind) => this.tensorEvidence(kind)?.evidence?.channels ?? []),
    ),
  );
  readonly tensorValueRange = computed(() => this.tensorValueZoom() ?? this.tensorValueFull());
  readonly tensorChannelZoom = signal<KvRangeBin | null>(null);
  readonly tensorChannelReduction = signal<ChannelReduction>('mean');
  readonly tensorChannelCapacity = signal(128);
  readonly channelBinOptions = ['Minimum', 'Maximum', 'Mean'];
  readonly tensorChannelFull = computed(() => {
    const count = Math.max(
      0,
      ...['key', 'value'].flatMap((kind) =>
        this.sides.map((side) => this.tensorSource(kind, side)?.channel_count ?? 0),
      ),
    );
    return count > 0 && this.tensorHead() !== null ? {start: 0, end: count} : null;
  });
  readonly tensorChannelRange = computed(() => {
    const full = this.tensorChannelFull(),
      range = this.tensorChannelZoom();
    if (!full || !range || range.start >= full.end) return full;
    return {start: Math.max(full.start, range.start), end: Math.min(full.end, range.end)};
  });
  readonly tensorChannelBinSize = computed(() => {
    const range = this.tensorChannelRange();
    return range ? channelBinSize(range, this.tensorChannelCapacity()) : 1;
  });
  setChannelReduction(value: string) {
    const reduction = ({Minimum: 'min', Maximum: 'max', Mean: 'mean'} as const)[
      value as 'Minimum' | 'Maximum' | 'Mean'
    ];
    if (reduction) this.tensorChannelReduction.set(reduction);
  }

  readonly incrementTensorRetry = (value: number) => value + 1;
  readonly tensorKey = new KvRequest<KvTensorHead[]>();
  readonly tensorValue = new KvRequest<KvTensorHead[]>();
  readonly tensorGroups = computed(() => [
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
  chooseVisualization(label: string) {
    const choice = KV_VISUALIZATIONS.find((option) => option.label === label);
    if (!choice) return;
    if (choice.data === 'Metric')
      this.chooseView(choice.chart === 'Heatmap' ? 'heatmap' : 'trends');
    else {
      this.showTensorValues();
      this.tensorPlot.set(choice.chart === 'Line chart' ? 'index' : 'scatter');
      const selection = this.tensorChannelSelection();
      if (this.tensorPlot() === 'scatter' && selection && selection.end - selection.start > 1)
        this.tensorChannelSelection.set(null);
    }
  }
  tokenRepresentative(bin: KvRangeBin) {
    const kind = this.selectedCurve() ?? this.kind();
    const rows = kind === this.kind() ? this.chartRows() : (this.otherChartRows()?.rows ?? []);
    const row = rows.find((r) => r.layer === this.layer()) ?? rows[0];
    const cell = row?.bins.find((b) => b.start === bin.start && b.end === bin.end);
    return representativePosition(bin, this.positionReduction(), cell?.metrics[this.metric()]);
  }
  private showTensorValues() {
    if (this.tensorView()) return;
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
  private initializeTensorLocation(position = this.tensorPosition(), layer = this.tensorLayer()) {
    const range = this.visibleRange();
    if (!range) return;
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
    this.tensorPosition.set(location?.position ?? null);
    this.tensorLayer.set(location?.layer ?? null);
    this.tensorHead.set(location?.head ?? null);
  }
  setTensorPosition(value: string, inspect = true) {
    const position = Number(value),
      range = this.fullRange();
    if (
      !value.trim() ||
      !Number.isSafeInteger(position) ||
      !range ||
      position < range.start ||
      position >= range.end
    )
      return;
    this.tensorPosition.set(position);
    this.syncTensorSelection(inspect);
  }
  moveTensorPosition(delta: number) {
    const position = this.tensorPosition();
    if (position !== null) this.setTensorPosition(String(position + delta));
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
  setTensorIndex(change: TensorIndexChange) {
    if (change.dimension === 'layer') this.setTensorLayer(change.value);
    else if (change.dimension === 'head') this.setTensorHead(change.value);
  }
  setTensorLayer(value: string) {
    const layer = Number(value);
    if (!this.layerIds().includes(layer)) return;
    this.tensorLayer.set(layer);
    if (!this.tensorHeads().includes(this.tensorHead() ?? -1)) {
      this.tensorHead.set(null);
      this.initializeTensorLocation();
    }
    this.syncTensorSelection();
  }
  setTensorHead(value: string) {
    const head = Number(value);
    if (!this.tensorHeads().includes(head)) return;
    this.tensorHead.set(head);
    this.inspect.emit(this);
  }
  private syncTensorSelection(inspect = true) {
    const position = this.tensorPosition(),
      layer = this.tensorLayer();
    if (position !== null && layer !== null) {
      this.selectPoint(layer, position);
      this.selectedCurve.set(null);
      if (!this.selectedTrendLayers().includes(layer))
        this.trendLayers.set([...this.selectedTrendLayers(), layer].sort((a, b) => a - b));
    }
    if (inspect) this.inspect.emit(this);
  }
  tensorSource(kind: string, side: 'ref' | 'target') {
    return (
      this.context()?.layers.find((row) => row.kind === kind && row.layer === this.tensorLayer())
        ?.token_structure?.[side] ?? null
    );
  }
  tensorSourcePosition(side: 'ref' | 'target') {
    const position = this.tensorPosition(),
      alignment = this.positionAlignment();
    return position === null
      ? null
      : alignment
        ? (alignment.pairs[position - alignment.start]?.[side === 'ref' ? 0 : 1] ?? null)
        : position;
  }
  tensorSourceStatus(kind: string, side: 'ref' | 'target') {
    const source = this.tensorSource(kind, side),
      position = this.tensorSourcePosition(side);
    if (position === null) return 'Not aligned';
    if (!source) return 'Not captured';
    if (source.status !== 'ok') return source.status.replaceAll('_', ' ');
    if (
      source.position_start === null ||
      source.position_count === null ||
      position < source.position_start ||
      position >= source.position_start + source.position_count
    )
      return 'Position not captured';
    return 'Captured';
  }
  tensorHeadCount(kind: string) {
    return Math.max(0, ...this.sides.map((side) => this.tensorSource(kind, side)?.head_count ?? 0));
  }
  tensorEvidence(kind: string) {
    return (
      (kind === 'key' ? this.tensorKey : this.tensorValue)
        .data()
        ?.find((row) => row.head === this.tensorHead()) ?? null
    );
  }
  locateTensorHead(kind: string, head: number) {
    this.selectedCurve.set(kind);
    this.setTensorHead(String(head));
    this.locateTensor(kind);
  }
  locateTensor(kind: string) {
    this.pendingTensorLocation.set(kind);
  }
  locateTensorElement(kind: string, index: number) {
    if (!this.tensorEvidence(kind)?.evidence?.channels.some((row) => row.channel === index)) return;
    if (this.tensorPlot() === 'index') {
      const full = this.tensorChannelFull(),
        range = this.tensorChannelRange();
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
      const outside = ['key', 'value'].some((cache) => {
        const row = this.tensorEvidence(cache)?.evidence?.channels.find(
          (row) => row.channel === index,
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
      if (outside) this.tensorValueZoom.set(null);
    }
    this.selectTensorChannel(kind, index);
    this.locateTensor(kind);
  }

  setRangeEdge(edge: 'start' | 'end', value: number) {
    const full = this.fullRange(),
      range = this.visibleRange();
    if (!full || !range || !Number.isFinite(value)) return;
    this.zoomRange.set(
      edge === 'start'
        ? {start: Math.max(full.start, Math.min(range.end - 1, Math.floor(value))), end: range.end}
        : {
            start: range.start,
            end: Math.min(full.end, Math.max(range.start + 1, Math.floor(value) + 1)),
          },
    );
  }

  private readonly views = inject(KvViewStore);
  private readonly resolvedIdentity = signal<KvViewIdentity | null>(null);
  readonly stateNotice = signal('');
  readonly entryMissing = signal(false);
  readonly metrics = KV_METRICS;
  readonly detailMetrics = KV_DETAIL_METRICS;
  readonly formatDetailMetric = formatKvMetric;
  readonly analysis = this.data.metadata.data;
  readonly error = this.data.metadata.error;
  readonly loading = this.data.metadata.loading;
  readonly retry = signal(0);
  readonly selectedContextId = signal('');
  readonly kind = signal('key');
  readonly metric = signal<KvMetric>('cosine_similarity');
  readonly colorScale = signal<'auto' | 'thresholds'>('auto');
  readonly head = signal('min');
  readonly aggregationHeads = signal<number[] | null>(null);
  readonly selectedHeads = computed(() =>
    ['min', 'max', 'mean'].includes(this.head())
      ? (this.aggregationHeads() ?? this.heads())
      : [Number(this.head())],
  );
  readonly aggregation = computed(() => ({
    dimension: 'head',
    label: 'Heads',
    options: this.heads().map((head) => ({id: String(head), label: `Head ${head}`})),
    selected: this.selectedHeads().map(String),
    function: (['min', 'max', 'mean'].includes(this.head())
      ? this.head()
      : 'min') as HeatmapReduction,
  }));
  setAggregation(value: {selection: string[] | null; function: HeatmapReduction}) {
    this.aggregationHeads.set(value.selection?.map(Number) ?? null);
    this.head.set(value.function);
  }
  readonly positionReductionOverride = signal<HeatmapReduction | null>(null);
  readonly positionReduction = computed(
    () => this.positionReductionOverride() ?? defaultPositionReduction(this.metric()),
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
  readonly positionBinLabels = computed(
    () => this.positionBinControl()?.choices.map((choice) => choice.label) ?? [],
  );
  readonly positionBinLabel = computed(() => {
    const control = this.positionBinControl();
    return !control || control.disabled
      ? '-'
      : (control.choices.find((choice) => choice.id === control.value)?.label ?? '-');
  });
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
          this.tensorView() ? 'mean' : defaultPositionReduction(this.metric())
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
  readonly displayDraft = signal<{bins: string; visible: string[]}>({bins: '-', visible: []});
  readonly displayQuery = signal('');
  readonly displayFilteredSeries = computed(() => {
    const query = this.displayQuery()
      .trim()
      .toLowerCase()
      .replace(/^l(?=\d)/, 'layer ');
    return this.displaySeries().filter((series) => series.label.toLowerCase().includes(query));
  });
  private sameDisplaySeries(a: readonly string[], b: readonly string[]) {
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
  toggleChartDisplay() {
    if (this.popup() === 'view-display') {
      this.closePopup();
      return;
    }
    this.displayDraft.set({bins: this.displayBins(), visible: this.displayVisible()});
    this.displayQuery.set('');
    this.popup.set('view-display');
  }
  draftDisplayBins(bins: string) {
    this.displayDraft.update((draft) => ({...draft, bins}));
  }
  draftDisplayVisible(visible: string[]) {
    this.displayDraft.update((draft) => ({...draft, visible}));
  }
  toggleDisplaySeries(id: string) {
    const visible = this.displayDraft().visible;
    this.draftDisplayVisible(
      visible.includes(id) ? visible.filter((item) => item !== id) : [...visible, id],
    );
  }
  highlightDisplaySeries(id: string | null) {
    if (!this.tensorView()) this.highlightedLayer.set(id === null ? null : Number(id));
  }
  applyChartDisplay() {
    const draft = this.displayDraft();
    if (!this.displayBinsDisabled()) {
      if (this.tensorView()) this.setChannelReduction(draft.bins);
      else {
        const value = ({Minimum: 'min', Maximum: 'max', Mean: 'mean'} as Record<string, string>)[
          draft.bins
        ];
        if (value) this.setHeatmapDimension({id: 'position', value});
      }
    }
    if (this.displaySeries().length) {
      if (this.tensorView())
        this.tensorVisibleRuntimes.set(
          draft.visible.filter((id) => id === 'ref' || id === 'target'),
        );
      else {
        const visible = draft.visible.map(Number).filter((id) => this.layerIds().includes(id));
        if (this.view() === 'heatmap') this.heatmapLayers.set(visible);
        else this.trendLayers.set(visible);
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
    this.displayDraft.set({bins: this.displayBins(), visible: this.displayVisible()});
  }
  setPositionBinLabel(label: string) {
    const choice = this.positionBinControl()?.choices.find((choice) => choice.label === label);
    if (choice) this.setHeatmapDimension({id: 'position', value: choice.id});
  }
  setHeatmapMetric(metric: string) {
    if (KV_METRICS.some((choice) => choice.id === metric)) this.metric.set(metric as KvMetric);
  }
  setHeatmapDimension(change: {id: string; value: string}) {
    const control = this.heatmapDimensions().find((control) => control.id === change.id);
    if (
      !control ||
      control.disabled ||
      !control.choices.some((choice) => choice.id === change.value)
    )
      return;
    if (change.id === 'head') this.head.set(change.value);
    if (change.id === 'position')
      this.positionReductionOverride.set(change.value as HeatmapReduction);
  }

  readonly view = signal<'heatmap' | 'trends'>('heatmap');
  readonly layer = signal(0);
  readonly position = signal(0);
  readonly column = signal(false);
  readonly showDetails = signal(window.matchMedia('(min-width: 801px)').matches);
  readonly popup = signal<'context' | 'display' | 'find' | 'layers' | 'view-display' | null>(null);
  draftScale: 'auto' | 'thresholds' = 'auto';
  draftMetric: KvMetric = 'relative_l2';
  draftKind = 'key';
  draftHead = 'max';
  readonly selectedTurn = computed(
    () =>
      this.stageTurn() ??
      this.state.session()?.batches.find((b) => b.batch === this.state.batchId())?.turn ??
      this.state.session()?.turns[0]?.n ??
      1,
  );
  private readonly identity = computed<KvViewIdentity | null>(() => {
    const captureId = this.state.captureId();
    return captureId ? {captureId, turn: this.selectedTurn()} : null;
  });
  private readonly queryIdentity = computed(() => {
    const current = this.identity(),
      loaded = this.resolvedIdentity();
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
  readonly zoomRange = signal<KvRangeBin | null>(null);
  readonly zoomHistory = signal<KvRangeBin[]>([]);
  readonly binCapacity = signal(24);
  readonly rangeData = this.data.range.data;
  readonly rangeLoading = this.data.range.loading;
  readonly rangeError = this.data.range.error;
  readonly rangeRetry = signal(0);
  readonly scaleReference = this.data.scale.data;
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
  readonly selectionData = this.data.selection.data;
  readonly selectionLoading = this.data.selection.loading;
  readonly selectionError = this.data.selection.error;
  readonly selectionRetry = signal(0);
  readonly findData = this.data.find.data;
  readonly findLoading = this.data.find.loading;
  readonly findError = this.data.find.error;
  readonly find = new KvFindController({
    data: this.findData,
    loading: this.findLoading,
    select: (result) => {
      this.kind.set(result.kind);
      this.selectPoint(result.layer, result.position);
    },
    setResults: (show) => this.setResults(show),
    openPopup: () => this.popup.set('find'),
    closePopup: () => this.popup.set(null),
  });
  readonly findRetry = this.find.retry;
  readonly resultIndex = this.find.resultIndex;
  readonly resultTotal = this.find.resultTotal;
  readonly resultPage = this.find.resultPage;
  readonly showResults = this.find.showResults;
  readonly formula = this.find.formula;
  readonly queryError = this.find.queryError;
  get draftFormula() {
    return this.find.draftFormula;
  }
  set draftFormula(value: string) {
    this.find.draftFormula = value;
  }
  readonly chartHeight = computed(() =>
    heatmapChartHeight(this.selectedHeatmapLayers().length, this.bothTensors() ? 2 : 1),
  );
  readonly chartViewportHeight = computed(() =>
    this.expanded() ? 'calc(100dvh - 294px)' : this.chartHeight() + 'px',
  );
  readonly fullRange = computed(() => {
    const captured = kvContextRange(this.context());
    if (!captured || !this.embedded()) return captured;
    const alignment = this.positionAlignment();
    if (alignment)
      return alignment.pairs.length
        ? {start: alignment.start, end: alignment.start + alignment.pairs.length}
        : null;
    const turn = this.state.session()?.turns.find((t) => t.n === this.stageTurn());
    if (turn?.step_start == null) return captured;
    const start =
      this.stageMoment() === 'prefill_post'
        ? turn.step_start - (turn.prefill_tokens ?? 0)
        : turn.step_start;
    const lo = Math.max(captured.start, start);
    return lo < captured.end ? {...captured, start: lo} : null;
  });
  readonly visibleRange = computed(() => this.zoomRange() ?? this.fullRange());
  readonly visibleBins = computed<KvRangeBin[]>(() => {
    const range = this.visibleRange();
    if (!range) return [];
    const n = rangeBinCount(range, this.binCapacity());
    return Array.from({length: n}, (_, i) => ({
      start: range.start + Math.floor((i * (range.end - range.start)) / n),
      end: range.start + Math.floor(((i + 1) * (range.end - range.start)) / n),
    }));
  });
  readonly chartBins = computed(() => this.rangeData()?.bins ?? this.visibleBins());
  readonly chartRows = computed(() =>
    this.colorScale() === 'auto' && !this.scaleReference() ? [] : (this.rangeData()?.rows ?? []),
  );
  readonly selectedPositions = computed(() => {
    const position = this.selectedPosition(),
      alignment = this.positionAlignment();
    return alignment && position !== undefined
      ? (alignment.pairs[position - alignment.start] ?? [null, null])
      : [position, position];
  });
  readonly selectedPosition = computed<number | undefined>(() => {
    const range = this.fullRange(),
      position = this.position();
    return range
      ? position >= range.start && position < range.end
        ? position
        : range.start
      : undefined;
  });
  readonly inspectedRange = computed<KvRangeSelection | null>(() => {
    const start = this.selectedPosition();
    if (start === undefined) return null;
    return {
      start,
      end: Math.min(this.fullRange()!.end, Math.max(start + 1, this.selectionEnd() ?? start + 1)),
      layer: this.column() ? null : (this.selectedLayer()?.layer ?? null),
    };
  });
  readonly isRange = computed(() => {
    const range = this.inspectedRange();
    return !!range && range.end - range.start > 1;
  });
  readonly heads = computed(() =>
    Array.from(
      {length: Math.max(0, ...(this.context()?.layers ?? []).map((row) => row.head_count ?? 0))},
      (_, i) => i,
    ),
  );
  readonly results = this.find.results;
  readonly resultPages = this.find.resultPages;
  readonly pageResults = this.results;
  readonly infoExpanded = signal(true);
  readonly actionsExpanded = signal(true);
  readonly savedExpanded = signal(false);
  readonly expandedHeads = signal<Record<string, boolean>>({});
  readonly headViews = signal<Record<string, KvHeadViewState>>({});
  readonly detailHeadPages = signal<Record<string, number>>({key: 0, value: 0});
  readonly tensorExpanded = signal<Record<string, boolean>>({key: true, value: true});
  readonly sides = ['ref', 'target'] as const;
  readonly tokenTensors = computed(() =>
    ['key', 'value'].map((kind) => {
      const layer = this.context()?.layers.find(
        (row) => row.layer === this.selectedLayer()?.layer && row.kind === kind,
      );
      return {
        kind,
        label: kind === 'key' ? 'Key cache' : 'Value cache',
        layer,
        sources: {
          ref: this.selectedPositions()[0] == null ? null : (layer?.token_structure?.ref ?? null),
          target:
            this.selectedPositions()[1] == null ? null : (layer?.token_structure?.target ?? null),
        },
        reason:
          this.selectionData()?.rows.find((row) => row.kind === kind && row.layer === layer?.layer)
            ?.reason ?? layer?.reason,
        comparison: this.exactComparison(kind, this.selectedLayer()?.layer),
      };
    }),
  );
  readonly columnTensors = computed(() =>
    ['key', 'value'].map((kind) => ({
      kind,
      label: kind === 'key' ? 'Key cache' : 'Value cache',
      rows: [...new Set(this.context()?.layers.map((row) => row.layer) ?? [])]
        .sort((a, b) => a - b)
        .map((id) => {
          const row = this.context()?.layers.find((row) => row.layer === id && row.kind === kind);
          return {
            layer: id,
            status: row?.status ?? 'not_captured',
            comparison: this.exactComparison(kind, id),
          };
        }),
    })),
  );
  readonly snapshotLabel = computed(() => {
    const context = this.context();
    if (!context) return 'Not captured';
    const moments: Record<string, string> = {
      prefill_pre: 'Before Prefill',
      prefill_post: 'After Prefill',
      terminal: 'Generation end',
    };
    return `Turn ${context.turn} · ${moments[context.moment] ?? context.moment ?? 'Stored cache'}`;
  });
  readonly snapshotRows = computed(() =>
    this.sides.map((side) => ({
      side,
      label: side === 'ref' ? 'Ref' : 'Target',
      value:
        (this.context()?.snapshots?.[side] ?? [])
          .map((snapshot) => this.snapshotIdentity(snapshot))
          .join('; ') || 'Not captured',
    })),
  );
  private readonly bookmarks = new BookmarkStore(KV_BOOKMARK_CODEC);
  readonly saved = computed(() => this.bookmarks.read(kvBookmarkKey(this.state.captureId())));
  readonly bookmarkError = this.bookmarks.error;
  readonly savedHere = computed(() => this.saved().filter((p) => p.context === this.context()?.id));
  readonly bookmarkSelection = computed<SavedPoint | null>(() => {
    const context = this.context(),
      position = this.selectedPosition(),
      layer = this.selectedLayer();
    return !this.isRange() && context && position !== undefined && (this.column() || layer)
      ? {context: context.id, position, layer: this.column() ? null : layer!.layer}
      : null;
  });
  readonly isSaved = computed(() => {
    const selection = this.bookmarkSelection();
    return (
      !!selection &&
      this.saved().some((point) => this.bookmarkKey(point) === this.bookmarkKey(selection))
    );
  });
  readonly panelWidth = signal(350);
  readonly workspaceWidth = signal(window.innerWidth);
  readonly resultsOverlay = computed(() => this.workspaceWidth() < 755);
  readonly detailsOverlay = computed(() => this.workspaceWidth() <= 800);
  readonly panelMaxWidth = computed(() =>
    Math.max(100, Math.floor(this.workspaceWidth()) - (this.detailsOverlay() ? 40 : 390)),
  );
  readonly panelMinWidth = computed(() => Math.min(220, this.panelMaxWidth()));
  readonly actualPanelWidth = computed(() => this.clampPanelWidth(this.panelWidth()));
  readonly canShowBothPanels = computed(
    () => this.workspaceWidth() - this.actualPanelWidth() - 365 >= 390,
  );
  readonly resizing = signal(false);
  readonly chartColor = (value: number | null) => this.color(value);
  readonly chartFormat = (value: number | null) => this.format(value);
  readonly rangeTensors = computed(() =>
    ['key', 'value'].map((kind) => ({
      kind,
      label: kind === 'key' ? 'Key cache' : 'Value cache',
      rows: (this.selectionData()?.rows ?? [])
        .filter((row) => row.kind === kind)
        .sort((a, b) => a.layer - b.layer),
    })),
  );
  constructor() {
    effect(() => {
      const selected = this.chartSelection();
      if (!this.tensorView() || !selected || selected.end - selected.start !== 1) return;
      untracked(() => {
        // A selected token changes the inspected tensor, not which sidebar owns focus.
        if (selected.start !== this.tensorPosition())
          this.setTensorPosition(String(selected.start), false);
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
      this.tensorChannels.set({key: null, value: null});
      this.tensorChannelSelection.set(null);
    });
    effect(() => {
      const kind = this.pendingTensorLocation(),
        chart =
          this.tensorPlot() === 'scatter' ? this.tensorScatterChart() : this.tensorLineChart();
      if (kind && chart) {
        chart.locateKind(kind);
        this.pendingTensorLocation.set(null);
      }
    });
    const tensorApi = inject(ReportApiService);
    effect((onCleanup) => {
      const capture = this.state.captureId(),
        context = this.context(),
        position = this.tensorPosition(),
        layer = this.tensorLayer(),
        active = this.tensorView(),
        alignment = this.positionAlignment(),
        turn = this.selectedTurn();
      this.tensorRetry();
      for (const group of this.tensorGroups()) {
        const count = this.tensorHeadCount(group.kind);
        onCleanup(
          group.request.load(
            active && capture && context && position !== null && layer !== null
              ? (signal) =>
                  readTensorHeads(
                    tensorApi,
                    capture,
                    {
                      turn,
                      contextId: context.id,
                      position,
                      layer,
                      kind: group.kind,
                      head: 0,
                      alignment,
                    },
                    count,
                    signal,
                  )
              : null,
          ),
        );
      }
    });

    effect(() => {
      const chart = this.rangeChart(),
        location = this.pendingCacheLocation();
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
        this.zoomHistory.set([]);
        this.selectionEnd.set(null);
      }
      previousAlignment = alignment;
    });
    inject(DestroyRef).onDestroy(() => {
      this.scopeAnimation?.cancel();
      this.saveView();
      this.data.destroy();
      this.valueData.destroy();
      this.overviewData.destroy();
      this.otherOverviewData.destroy();
    });
    effect((onCleanup) => {
      const observer = new ResizeObserver(([entry]) =>
        this.workspaceWidth.set(entry.contentRect.width),
      );
      observer.observe(this.host.nativeElement);
      onCleanup(() => observer.disconnect());
    });
    effect(() => {
      if (this.showDetails() && this.showResults() && !this.canShowBothPanels())
        this.showResults.set(false);
    });
    effect((onCleanup) => {
      const identity = this.identity();
      const moment = this.stageMoment();
      const initial = this.initialView();
      const explicitEntry = this.explicitEntry();
      this.state.revision();
      this.retry();
      untracked(() => {
        this.saveView();
        const saved = this.embedded()
          ? (initial ?? undefined)
          : identity
            ? this.views.read(identity)
            : undefined;
        this.resolvedIdentity.set(null);
        this.popup.set(null);
        onCleanup(
          this.data.loadMetadata(identity, (analysis) => {
            if (!identity) return;
            const entry = this.embedded() ? (saved ? null : explicitEntry) : this.state.kvEntry();
            const restored = restoreKvView(
              saved,
              analysis,
              identity,
              entry,
              window.matchMedia('(min-width: 801px)').matches,
            );
            if (this.embedded() && !saved) {
              restored.state.metric = 'cosine_similarity';
              restored.state.head = 'min';
            }
            this.applyView(restored.state);
            if (this.embedded() && !saved && !entry) {
              const context = analysis.contexts.find(
                (c) => c.turn === identity.turn && c.moment === moment,
              );
              this.selectedContextId.set(context?.id ?? '');
              this.showDetails.set(false);
            }
            this.stateNotice.set(restored.notice);
            this.entryMissing.set(restored.entryMissing);
            this.resolvedIdentity.set(identity);
            // An entry is consumed once its capture resolves; one from another capture is stale.
            if (
              !this.embedded() &&
              entry &&
              (entry.sessionId !== identity.captureId || entry.turn === identity.turn)
            )
              this.state.kvEntry.set(null);
          }),
        );
      });
    });
    effect(() => {
      this.context()?.id;
      this.formula();
      this.resultPage.set(0);
      this.resultTotal.set(0);
      this.find.reset();
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.fullRange(),
        kind = this.kind(),
        head = this.head(),
        heads = this.aggregationHeads(),
        enabled = this.colorScale() === 'auto';
      this.rangeRetry();
      onCleanup(
        this.data.loadScale(
          identity && contextId && range && enabled
            ? {...identity, contextId, range, kind, head, heads, bins: 1, formula: ''}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.fullRange(),
        kind = this.kind(),
        head = this.head(),
        heads = this.aggregationHeads();
      this.rangeRetry();
      const query =
        identity && contextId && range
          ? {...identity, contextId, range, kind, head, heads, bins: 128, formula: ''}
          : null;
      onCleanup(this.overviewData.loadRange(query));
      onCleanup(
        this.otherOverviewData.loadRange(
          query && this.bothTensors() ? {...query, kind: kind === 'key' ? 'value' : 'key'} : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.visibleRange(),
        bins = rangeBinCount(range, this.binCapacity()),
        kind = this.kind(),
        head = this.head(),
        heads = this.aggregationHeads(),
        formula = this.formula();
      this.rangeRetry();
      onCleanup(
        this.data.loadRange(
          identity && contextId && range
            ? {...identity, contextId, range, bins, kind, head, heads, formula}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.visibleRange(),
        full = this.fullRange(),
        kind = this.kind() === 'key' ? 'value' : 'key',
        head = this.head(),
        heads = this.aggregationHeads(),
        bins = rangeBinCount(range, this.binCapacity());
      this.rangeRetry();
      const enabled = this.bothTensors();
      onCleanup(
        this.valueData.loadRange(
          enabled && identity && contextId && range
            ? {...identity, contextId, range, kind, head, heads, bins, formula: this.formula()}
            : null,
        ),
      );
      onCleanup(
        this.valueData.loadScale(
          this.bothTensors() && identity && contextId && full
            ? {...identity, contextId, range: full, kind, head, heads, bins: 1, formula: ''}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        selection = this.inspectedRange();
      this.selectionRetry();
      onCleanup(
        this.data.loadSelection(
          identity && contextId && selection ? {...identity, contextId, selection} : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        formula = this.formula(),
        page = this.resultPage();
      this.findRetry();
      onCleanup(
        this.data.loadFind(
          identity && contextId && formula ? {...identity, contextId, formula, page} : null,
          (data) => untracked(() => this.find.receive(data, page)),
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
      formula: this.formula(),
      layer: this.layer(),
      position: this.position(),
      selectionEnd: this.selectionEnd(),
      trendLayers: this.trendLayers(),
      heatmapLayers: this.heatmapLayers(),
      bothTensors: this.bothTensors(),
      column: this.column(),
      zoomRange: this.zoomRange(),
      zoomHistory: this.zoomHistory(),
      panelWidth: this.panelWidth(),
      showDetails: this.showDetails(),
      showResults: this.showResults(),
      infoExpanded: this.infoExpanded(),
      actionsExpanded: this.actionsExpanded(),
      savedExpanded: this.savedExpanded(),
      expandedHeads: this.expandedHeads(),
      headViews: this.headViews(),
      detailHeadPages: this.detailHeadPages(),
      tensorExpanded: this.tensorExpanded(),
    };
  }
  private saveView() {
    const identity = this.resolvedIdentity();
    if (!this.embedded() && identity && !this.entryMissing())
      this.views.save(identity, this.captureView());
  }
  private applyView(view: KvViewSnapshot) {
    this.selectedContextId.set(view.contextId);
    this.trendLayers.set(view.trendLayers ?? null);
    this.heatmapLayers.set(view.heatmapLayers ?? null);
    this.bothTensors.set(this.embedded() || (view.bothTensors ?? true));
    this.kind.set(this.embedded() ? 'key' : view.kind);
    this.metric.set(
      KV_METRICS.some((choice) => choice.id === view.metric) ? view.metric : 'cosine_similarity',
    );
    this.colorScale.set(view.colorScale);
    this.head.set(view.head);
    this.aggregationHeads.set(view.headSelection ?? null);
    this.positionReductionOverride.set(view.positionReduction ?? null);
    this.view.set(view.view);
    this.tensorView.set(view.tensor?.active ?? false);
    this.tensorPlot.set(view.tensor?.plot ?? 'index');
    this.tensorPosition.set(view.tensor?.position ?? null);
    this.tensorLayer.set(view.tensor?.layer ?? null);
    this.tensorHead.set(view.tensor?.head ?? null);
    this.tensorChannelZoom.set(view.tensor?.channelRange ?? null);
    this.tensorValueZoom.set(view.tensor?.valueRange ?? null);
    this.tensorChannelReduction.set(view.tensor?.channelReduction ?? 'mean');
    this.tensorVisibleRuntimes.set(view.tensor?.visibleRuntimes ?? ['ref', 'target']);
    this.formula.set(this.embedded() ? '' : view.formula);
    this.layer.set(view.layer);
    this.position.set(view.position);
    this.selectionEnd.set(view.selectionEnd);
    this.column.set(view.column);
    this.zoomRange.set(view.zoomRange);
    this.zoomHistory.set(view.zoomHistory);
    this.panelWidth.set(view.panelWidth);
    this.showDetails.set(view.showDetails);
    this.showResults.set(!this.embedded() && view.showResults);
    this.infoExpanded.set(view.infoExpanded);
    this.actionsExpanded.set(view.actionsExpanded);
    this.savedExpanded.set(view.savedExpanded);
    this.expandedHeads.set(view.expandedHeads);
    this.headViews.set(view.headViews);
    this.detailHeadPages.set(view.detailHeadPages);
    this.tensorExpanded.set(view.tensorExpanded);
  }
  snapshotIdentity(snapshot: KvSnapshotIdentity) {
    const identity =
      snapshot.snapshot_id !== null
        ? `Snapshot ${snapshot.snapshot_id}`
        : (snapshot.signature ?? snapshot.id);
    return (
      [
        snapshot.runtime,
        identity,
        snapshot.forward_id !== null ? `Forward ${snapshot.forward_id}` : null,
        snapshot.step !== null ? `Step ${snapshot.step}` : null,
        snapshot.edge,
      ]
        .filter(Boolean)
        .join(' · ') || 'Not captured'
    );
  }
  rowContainsPosition(row: KvAnalysisLayer, position: number) {
    return (
      row.cells.some((cell) => cell.position === position) ||
      [row.token_structure?.ref, row.token_structure?.target].some(
        (source) =>
          source?.status === 'ok' &&
          source.position_start !== null &&
          source.position_count !== null &&
          position >= source.position_start &&
          position < source.position_start + source.position_count,
      )
    );
  }
  structureAvailable(source: KvTokenStructure | null) {
    const position = this.selectedPosition();
    return (
      !!source &&
      source.status === 'ok' &&
      position !== undefined &&
      source.position_start !== null &&
      source.position_count !== null &&
      position >= source.position_start &&
      position < source.position_start + source.position_count
    );
  }
  structureShape(source: KvTokenStructure | null) {
    return this.structureAvailable(source) && source?.shape
      ? '[' + source.shape.join(', ') + ']'
      : 'Unavailable';
  }
  structureDtype(source: KvTokenStructure | null) {
    return this.structureAvailable(source) ? (source?.dtype ?? 'Not captured') : 'Unavailable';
  }
  structureStatus(source: KvTokenStructure | null) {
    if (!source) return 'Not captured';
    if (source.status !== 'ok') return source.reason ?? source.status.replaceAll('_', ' ');
    const position = this.selectedPosition();
    if (
      position === undefined ||
      source.position_start === null ||
      source.position_count === null ||
      position < source.position_start ||
      position >= source.position_start + source.position_count
    )
      return 'Token position not captured';
    return 'Captured';
  }
  structureHeadCount(kind: string) {
    const sources = this.tokenTensors().find((tensor) => tensor.kind === kind)?.sources;
    return sources
      ? Math.max(
          0,
          ...this.sides.map((side) =>
            this.structureAvailable(sources[side]) ? (sources[side]?.head_count ?? 0) : 0,
          ),
        )
      : 0;
  }
  headPages(kind: string) {
    return Math.max(1, Math.ceil(this.structureHeadCount(kind) / 64));
  }
  headPage(kind: string) {
    return Math.min(this.detailHeadPages()[kind] ?? 0, this.headPages(kind) - 1);
  }
  visibleHeads(kind: string) {
    const start = this.headPage(kind) * 64;
    return Array.from(
      {length: Math.min(64, this.structureHeadCount(kind) - start)},
      (_, i) => start + i,
    );
  }
  moveHeadPage(kind: string, delta: number) {
    this.detailHeadPages.update((pages) => ({
      ...pages,
      [kind]: Math.max(0, Math.min(this.headPages(kind) - 1, this.headPage(kind) + delta)),
    }));
  }
  headOpen(kind: string, head: number) {
    return this.expandedHeads()[`${kind}:${head}`] ?? false;
  }
  toggleHead(kind: string, head: number) {
    this.expandedHeads.update((state) => ({
      ...state,
      [`${kind}:${head}`]: !this.headOpen(kind, head),
    }));
  }
  headView(kind: string, head: number) {
    return this.headViews()[`${kind}:${head}`] ?? DEFAULT_HEAD_VIEW;
  }
  setHeadView(kind: string, head: number, patch: Partial<KvHeadViewState>) {
    this.headViews.update((state) => ({
      ...state,
      [`${kind}:${head}`]: {...this.headView(kind, head), ...patch},
    }));
  }
  headSelection(kind: string, head: number): KvHeadSelection | null {
    const context = this.context(),
      layer = this.selectedLayer(),
      position = this.selectedPosition();
    return context && layer && position !== undefined
      ? {
          turn: this.selectedTurn(),
          contextId: context.id,
          layer: layer.layer,
          kind,
          position,
          head,
          alignment: this.positionAlignment(),
        }
      : null;
  }
  setTensorExpanded(kind: string, expanded: boolean) {
    this.tensorExpanded.update((state) => ({...state, [kind]: expanded}));
  }
  columnRows(kind: string) {
    return this.columnTensors().find((tensor) => tensor.kind === kind)?.rows ?? [];
  }
  exactComparison(kind: string, layer: number | undefined): KvTokenMetrics | undefined {
    if (this.isRange()) return undefined;
    const row = this.selectionData()?.rows.find((row) => row.kind === kind && row.layer === layer),
      position = this.selectedPosition();
    if (!row || position === undefined) return undefined;
    if (row.token_metrics?.[0]?.position === position) return row.token_metrics[0];
    return {
      position,
      status: row.status,
      metrics: {
        cosine_similarity: row.metrics.cosine_similarity.value,
        relative_l2: row.metrics.relative_l2.value,
        rmse: row.metrics.rmse.value,
        max_abs: row.metrics.max_abs.value,
      },
      metric_status: {
        cosine_similarity: row.metrics.cosine_similarity.status,
        relative_l2: row.metrics.relative_l2.status,
        rmse: row.metrics.rmse.status,
        max_abs: row.metrics.max_abs.status,
      },
    };
  }
  selectionLabel() {
    const range = this.inspectedRange();
    return range
      ? range.end - range.start === 1
        ? String(range.start)
        : `${range.start}–${range.end - 1}`
      : 'Not captured';
  }
  rangeMetricTitle(row: KvRangeDetailRow, key: (typeof this.detailMetrics)[number]['key']) {
    const metric = row.metrics[key],
      range = this.inspectedRange();
    return `${metric.valid_count} / ${range ? range.end - range.start : 0} valid positions · ${metric.status.replaceAll('_', ' ')}`;
  }
  setCapacity(capacity: number) {
    this.binCapacity.set(Math.max(1, Math.min(KV_RANGE_MAX_BINS, Math.floor(capacity))));
  }
  zoomTo(range: KvRangeBin) {
    const current = this.visibleRange(),
      full = this.fullRange();
    if (!current || !full) return;
    // The Plotly range slider can expand or pan outside the current viewport.
    const next = {
      start: Math.max(full.start, range.start),
      end: Math.min(full.end, range.end),
    };
    if (next.end <= next.start || (next.start === current.start && next.end === current.end))
      return;
    this.zoomHistory.update((history) => [...history, current]);
    this.zoomRange.set(next);
  }
  undoZoom() {
    const history = this.zoomHistory();
    if (!history.length) return;
    this.zoomRange.set(history[history.length - 1]);
    this.zoomHistory.set(history.slice(0, -1));
  }
  selectRange(selection: KvRangeSelection) {
    if (this.chartLoading() || this.chartError()) return;
    this.find.reset();
    this.position.set(selection.start);
    this.selectionEnd.set(selection.end);
    this.column.set(selection.layer === null);
    if (selection.layer !== null) {
      this.layer.set(selection.layer);
      if (this.view() === 'heatmap')
        this.trendLayers.set(
          [...new Set([...this.selectedTrendLayers(), selection.layer])].sort((a, b) => a - b),
        );
    }
    this.inspect.emit(this);
  }
  metricLabel(value = this.metric()) {
    return KV_METRICS.find((m) => m.id === value)!.label;
  }
  headLabel(value = this.head()) {
    return value === 'min'
      ? 'Min over heads'
      : value === 'max'
        ? 'Max over heads'
        : value === 'mean'
          ? 'Mean over heads'
          : `Head ${value}`;
  }
  format(value: number | null | undefined, metric = this.metric()) {
    return formatKvMetric(metric, value);
  }
  readonly scaleMax = computed(() =>
    (this.scaleReference()?.rows ?? []).reduce(
      (max, row) =>
        row.bins.reduce((value, bin) => Math.max(value, bin.metrics[this.metric()]?.max ?? 0), max),
      0.000001,
    ),
  );
  paletteColor(percent: number) {
    return `color-mix(in srgb,var(--heat-high) ${Math.min(100, Math.max(0, percent))}%,var(--heat-low))`;
  }
  color(value: number | null) {
    if (value == null) return 'var(--kv-missing)';
    if (this.metric() === 'cosine_similarity') return this.paletteColor((1 - value) * 50);
    if (this.colorScale() === 'auto') return this.paletteColor((value / this.scaleMax()) * 100);
    const limits =
      this.metric() === 'relative_l2' ? [0.0001, 0.001, 0.01, 0.1] : [0.00001, 0.0001, 0.001, 0.01];
    return this.paletteColor(limits.filter((n) => value >= n).length * 25);
  }
  readonly scale = computed(() =>
    this.metric() === 'cosine_similarity'
      ? ['1', '0.6', '0.2', '−0.2', '−0.6', '−1']
      : this.colorScale() === 'auto'
        ? this.scaleReference()
          ? [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => {
              return formatSignificant(
                f * this.scaleMax() * (this.metric() === 'relative_l2' ? 100 : 1),
                3,
              );
            })
          : []
        : this.metric() === 'relative_l2'
          ? ['0', '0.01', '0.1', '1', '10', '+']
          : ['0', '1e−5', '1e−4', '1e−3', '0.01', '+'],
  );
  chooseLayer(layer: number) {
    this.find.reset();
    this.layer.set(layer);
  }
  selectPoint(layer: number, position: number | undefined) {
    if (position === undefined) return;
    this.find.reset();
    this.layer.set(layer);
    this.position.set(position);
    this.selectionEnd.set(null);
    this.column.set(false);
  }
  selectColumn(position: number) {
    this.find.reset();
    this.position.set(position);
    this.selectionEnd.set(null);
    this.column.set(true);
  }
  selectResult(result: KvFindResult, index: number) {
    this.find.select(result, index);
  }
  setResults(show: boolean) {
    if (show && !this.canShowBothPanels()) this.setDetails(false);
    this.showResults.set(show);
  }
  setDetails(show: boolean) {
    if (show && !this.canShowBothPanels()) this.showResults.set(false);
    this.showDetails.set(show);
  }
  moveResult(delta: number) {
    this.find.move(delta);
  }
  firstResult() {
    this.find.first();
  }
  openDisplay() {
    this.draftScale = this.colorScale();
    this.draftMetric = this.metric();
    this.draftKind = this.kind();
    this.draftHead = this.head();
    this.popup.set('display');
  }
  applyDisplay() {
    this.colorScale.set(this.draftScale);
    this.metric.set(this.draftMetric);
    this.kind.set(this.draftKind);
    this.head.set(this.draftHead);
    this.aggregationHeads.set(null);
    this.popup.set(null);
  }
  openFind() {
    this.find.open();
  }
  applyFind() {
    this.find.apply();
  }
  clearFind() {
    this.find.clear();
  }
  closePopup() {
    this.highlightedLayer.set(null);
    this.popup.set(null);
  }
  chooseTurn(value: number) {
    // Explicit navigation may change shared Turn; ordinary view restoration never does.
    this.state.selectTurn(value);
  }
  chooseContext(context: KvAnalysisContext) {
    const analysis = this.analysis(),
      identity = this.identity();
    if (!analysis || !identity || context.turn !== identity.turn) return;
    const restored = restoreKvView(
      {
        ...this.captureView(),
        contextId: context.id,
        position: kvContextRange(context)?.start ?? 0,
        column: false,
        selectionEnd: null,
        zoomRange: null,
        zoomHistory: [],
      },
      analysis,
      identity,
    );
    this.applyView(restored.state);
    this.stateNotice.set('');
    this.entryMissing.set(false);
    this.popup.set(null);
  }
  statusLabel(status: string) {
    return status === 'ok' ? 'Comparable' : status.replace(/_/g, ' ');
  }
  bookmarkKey(point: SavedPoint) {
    return KV_BOOKMARK_CODEC.identity(point);
  }
  readonly bookmarkLabel = kvBookmarkLabel;
  toggleBookmark() {
    const point = this.bookmarkSelection();
    if (!point) return;
    const next = this.isSaved()
      ? this.saved().filter((p) => this.bookmarkKey(p) !== this.bookmarkKey(point))
      : [...this.saved(), point];
    this.bookmarks.set(kvBookmarkKey(this.state.captureId()), next);
  }
  openSaved(point: SavedPoint) {
    const context = this.analysis()?.contexts.find(
      (context) => context.id === point.context && context.turn === this.selectedTurn(),
    );
    if (
      !context ||
      !context.layers.some(
        (row) =>
          (point.layer === null || row.layer === point.layer) &&
          this.rowContainsPosition(row, point.position),
      )
    ) {
      this.stateNotice.set(
        'This bookmarked KV location is no longer available. The bookmark was kept.',
      );
      return;
    }
    this.chooseContext(context);
    const candidates = (this.context()?.layers ?? []).filter(
      (row) =>
        (point.layer === null || row.layer === point.layer) &&
        this.rowContainsPosition(row, point.position),
    );
    if (!candidates.some((row) => row.kind === this.kind()) && candidates[0])
      this.kind.set(candidates[0].kind);
    if (point.layer === null) this.selectColumn(point.position);
    else this.selectPoint(point.layer, point.position);
  }
  private clampPanelWidth(width: number) {
    return Math.max(this.panelMinWidth(), Math.min(this.panelMaxWidth(), Math.round(width)));
  }
}
