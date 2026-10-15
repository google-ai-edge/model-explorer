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

import {IntermediateGraphArch, type GraphArchSection} from '../analysis/intermediate_graph_arch';
import {ExplorerInfoSection} from '../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoList, type ExplorerInfoItem} from '../../shared/explorer_info_list/explorer_info_list';
import {AnalysisViewActions} from '../analysis/view_actions';
import {tensorPointLabel} from '../analysis/tensor_labels';
import {
  LayerActivationInspection,
  type LayerActivationInspectionState,
} from '../analysis/layer_activation_inspection';
import {MatIconModule} from '@angular/material/icon';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {OverlayModule} from '@angular/cdk/overlay';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {ExplorerInfoValue} from '../../shared/explorer_info_value/explorer_info_value';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../shared/locate_feedback';
import {PositionRangeControl} from '../analysis/position_range_control';
import {heatmapChartHeight} from '../analysis/heatmap_plot';
import type {StageAnalysisPanel} from './stage_analysis';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  Injector,
  TemplateRef,
  afterNextRender,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import type {PositionAnalysis} from '../../data/contracts/position_analysis';
import type {KvMetric, KvRangeBin} from '../../data/contracts/kv';
import {KvRequest} from '../kv/kv_panel/kv_data';
import {KvRangeChart} from '../kv/kv_panel/kv_range_chart';
import type {KvRangeSelection} from '../kv/kv_panel/kv_range_types';
import {layerColor} from '../analysis/layer_legend';
import {HeatmapControls} from '../analysis/heatmap_controls';
import type {AggregationChange} from '../analysis/aggregation_control';
import {formatAnalysisMetric} from '../../shared/format/format';
import {
  HEATMAP_DEFINITIONS,
  heatmapControls,
  defaultPositionReduction,
  type HeatmapReduction,
} from '../analysis/heatmap_model';
import {TENSOR_METRICS, tensorBins, tensorRangeRows} from '../analysis/position_tensor_model';
import {representativePosition} from './analysis_token_strip/analysis_token_model';

@Component({
  selector: 'stage-graph-metrics',
  standalone: true,
  imports: [
    IntermediateGraphArch,
    ExplorerInfoSection,
    ExplorerInfoList,
    LayerActivationInspection,
    AnalysisViewActions,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
    OverlayModule,
    OverlayDialog,
    OverlayPanel,
    ConfigPicker,
    ExplorerInfoValue,
    PositionRangeControl,
    HeatmapControls,
    KvRangeChart,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './graph_metric_panel.ng.html',
  styleUrl: './graph_metric_panel.scss',
})
export class GraphMetricPanel {
  readonly expanded = input(false);
  readonly chartSelection = input<KvRangeSelection | null>();
  readonly expand = output<void>();
  readonly heatmapChartHeight = heatmapChartHeight;
  readonly increment = (n: number) => n + 1;
  readonly panelId = input('');
  /** Kept to restore saved Intermediate panels; both categories now share one analysis. */
  readonly category = input<'activations' | 'intermediate'>('activations');
  readonly initial = input<NonNullable<StageAnalysisPanel['graph']> | null>(null);
  readonly turn = input.required<number>();
  readonly stage = input.required<string>();
  readonly positionAlignment = input<import('../../data/contracts/kv').KvPositionAlignment | null>(
    null,
  );
  readonly inspect = output<GraphMetricPanel>();
  readonly backgroundClick = output<MouseEvent>();
  readonly positionHover = output<KvRangeSelection | null>();
  readonly detailsTemplate = viewChild<TemplateRef<unknown>>('details');
  readonly inspection = viewChild(LayerActivationInspection);
  readonly inspectionMode = signal(false);
  readonly inspectionInitial = signal<LayerActivationInspectionState | null>(null);
  private readonly inspectionState = signal<LayerActivationInspectionState | null>(null);
  readonly state = inject(ReportStateService);
  private readonly api = inject(ReportApiService);
  readonly request = new KvRequest<PositionAnalysis>();
  readonly retry = signal(0);
  readonly metric = signal<KvMetric>('cosine_similarity');
  readonly anchor = signal('min');
  readonly anchorSelection = signal<string[] | null>(null);
  readonly anchorScope = signal<'output' | 'internal' | 'all'>('output');
  private readonly heatmapControls = viewChild(HeatmapControls);
  private readonly rangeChart = viewChild(KvRangeChart);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly locationInjector = inject(Injector);
  private readonly layerDisplay = viewChild<ElementRef<HTMLElement>>('layerDisplay');
  private locationAnimation?: Animation;
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
  locateView() {
    if (this.inspectionMode()) return this.inspection()?.locateView();
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locateRange() {
    if (this.inspectionMode()) return this.inspection()?.locatePosition();
    if (this.view() === 'graph')
      return this.locateControl(
        this.host.nativeElement.querySelector<HTMLElement>('.position-control input'),
      );
    const control = this.host.nativeElement.querySelector<HTMLElement>('analysis-position-range');
    this.locateControl(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
    );
  }
  locateLayers() {
    if (this.inspectionMode()) return this.inspection()?.locateLayer();
    if (this.view() !== 'trends') return;
    if (!this.displayOpen()) this.toggleDisplay();
    afterNextRender(() => this.locateControl(this.layerDisplay()?.nativeElement), {
      injector: this.locationInjector,
    });
  }
  locateMetric() {
    this.heatmapControls()?.locateMetric();
  }
  locateSummary() {
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.toolbar-summary button'));
  }
  locateAggregation() {
    this.heatmapControls()?.locateDimension('syncPoint');
  }
  locateChart(position?: number, layer?: number) {
    this.rangeChart()?.locateKind(this.label(), position, layer);
  }
  readonly view = signal<'heatmap' | 'trends' | 'graph'>('heatmap');
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
      options: [
        this.visualizationOptions[0],
        this.visualizationOptions[1],
        this.visualizationOptions[4],
      ],
    },
    {label: 'Tensor Inspection', options: this.visualizationOptions.slice(2, 4)},
  ];
  readonly visualizationLabels = {
    'Tensor Comparison · Graph Diff': 'Graph Diff',
    'Tensor Comparison · Heatmap': 'Heatmap',
    'Tensor Comparison · Line chart': 'Line chart',
    'Tensor Inspection · Line chart': 'Line chart',
    'Tensor Inspection · Scatter plot': 'Scatter plot',
  };
  readonly visualizationLabel = computed(() =>
    this.view() === 'graph'
      ? this.visualizationOptions[4]
      : this.view() === 'heatmap'
        ? this.visualizationOptions[0]
        : this.visualizationOptions[1],
  );
  tokenRepresentative(bin: KvRangeBin, layer = this.selected()?.layer) {
    const row = this.rows().find((row) => row.layer === layer) ?? this.rows()[0];
    const cell = row?.bins.find((cell) => cell.start === bin.start && cell.end === bin.end);
    return representativePosition(bin, this.positionReduction(), cell?.metrics[this.metric()]);
  }
  /** A layer graph can focus a real point; an aggregate mean is not a tensor identity. */
  layerGraphContext() {
    const inspection = this.inspectionMode()
      ? (this.inspection()?.captureView() ?? this.inspectionState())
      : null;
    if (inspection)
      return {
        layer: inspection.layer ?? undefined,
        anchor: inspection.anchor ?? 'output',
        position: inspection.position,
      };
    const selected = this.chartSelection() === undefined ? this.selected() : this.chartSelection();
    const layers = this.selectedLayers();
    const layer = selected?.layer ?? (layers.length === 1 ? layers[0] : undefined);
    const position =
      selected && layer !== undefined ? this.tokenRepresentative(selected, layer) : undefined;
    const anchors = this.selectedAnchors();
    const anchor =
      anchors.length === 1
        ? anchors[0]
        : position !== undefined && layer !== undefined && this.anchor() !== 'mean'
          ? this.inspectionAnchor(position, layer)
          : undefined;
    return {layer, anchor, position};
  }
  chooseView(value: string) {
    if (!this.visualizationOptions.includes(value)) return;
    this.closeDisplay();
    if (value === 'Tensor Comparison · Graph Diff') {
      this.inspectionMode.set(false);
      this.view.set('graph');
      this.setGraphPosition(this.graphPosition() ?? 0);
      return;
    }
    if (value.startsWith('Tensor Inspection')) {
      const previous = this.inspectionState();
      const selected =
        this.chartSelection() === undefined ? this.selected() : this.chartSelection();
      const range = this.range();
      const candidates = this.metricSource().filter(
        (row) => range && row.position >= range.start && row.position < range.end,
      );
      const fallback =
        candidates.find((row) => row.status === 'ok' && row.shape?.length) ??
        candidates.find((row) => row.shape?.length) ??
        candidates[0];
      const position = selected
        ? this.tokenRepresentative(selected, selected.layer)
        : (previous?.position ?? fallback?.position ?? range?.start ?? 0);
      const layer = selected?.layer ?? previous?.layer ?? fallback?.layer ?? this.layers()[0] ?? 0;
      const anchor = this.inspectionAnchor(position, layer, previous?.anchor);
      const initial: LayerActivationInspectionState = {
        position,
        layer,
        anchor,
        elementRange: previous?.elementRange ?? null,
        valueRange: previous?.valueRange ?? null,
        selection:
          previous?.position === position &&
          previous?.layer === layer &&
          (previous?.anchor ?? 'output') === anchor
            ? (previous.selection ?? null)
            : null,
        reduction: previous?.reduction ?? 'mean',
        visibleRuntimes: previous?.visibleRuntimes ?? ['ref', 'target'],
        plot: value.endsWith('Scatter plot') ? 'scatter' : 'index',
      };
      this.inspectionInitial.set(initial);
      this.inspectionState.set(initial);
      this.inspectionMode.set(true);
      this.inspect.emit(this);
      return;
    }
    this.showComparison(value === this.visualizationOptions[0] ? 'heatmap' : 'trends');
  }
  private inspectionAnchor(position: number, layer: number, previous?: string): string {
    const anchors = this.selectedAnchors();
    if (anchors.length === 1) return anchors[0];
    const rows = this.metricSource().filter(
      (row) => row.position === position && row.layer === layer,
    );
    const metric = this.metrics.find((m) => m.id === this.metric())!.source;
    if (this.anchor() === 'min' || this.anchor() === 'max') {
      const candidates = rows.filter(
        (row) => row.status === 'ok' && Number.isFinite(row.metrics[metric]?.value),
      );
      candidates.sort((a, b) => {
        const difference = a.metrics[metric]!.value! - b.metrics[metric]!.value!;
        return (
          (this.anchor() === 'min' ? difference : -difference) ||
          anchors.indexOf(a.anchor) - anchors.indexOf(b.anchor)
        );
      });
      if (candidates.length) return candidates[0].anchor;
    }
    // A mean has no source tensor: retain the browsing point or start at an available point.
    if (previous && anchors.includes(previous)) return previous;
    return rows.find((row) => row.shape?.length)?.anchor ?? anchors[0] ?? 'output';
  }
  showComparison(view: 'heatmap' | 'trends' | 'graph') {
    const inspection = this.inspection();
    if (inspection) {
      this.cacheInspection(inspection);
      if (this.expanded()) {
        const position = inspection.position();
        const bin = this.bins().find((bin) => bin.start <= position && position < bin.end);
        this.selected.set({
          start: bin?.start ?? position,
          end: bin?.end ?? position + 1,
          layer: inspection.layer(),
        });
      }
    }
    this.inspectionMode.set(false);
    this.view.set(view);
    this.inspect.emit(this);
  }
  cacheInspection(inspection: LayerActivationInspection) {
    const next = inspection.captureView(),
      previous = this.inspectionState();
    this.inspectionState.set(next);
    if (previous && (next.anchor ?? 'output') !== (previous.anchor ?? 'output')) {
      this.anchorScope.set('all');
      this.anchorSelection.set([next.anchor ?? 'output']);
      this.anchor.set(next.anchor ?? 'output');
    }
    if (previous && (next.position !== previous.position || next.layer !== previous.layer)) {
      this.selected.set({start: next.position, end: next.position + 1, layer: next.layer});
      if (next.layer !== null && !this.selectedLayers().includes(next.layer))
        this.trendLayers.set([...this.selectedLayers(), next.layer].sort((a, b) => a - b));
    }
  }
  inspectionChanged(inspection: LayerActivationInspection) {
    this.cacheInspection(inspection);
    this.inspect.emit(this);
  }
  readonly positionOverride = signal<HeatmapReduction | null>(null);
  readonly positionReduction = computed(
    () => this.positionOverride() ?? defaultPositionReduction(this.metric()),
  );
  readonly selected = signal<KvRangeSelection | null>(null);
  readonly viewport = signal<KvRangeBin | null>(null);
  readonly capacity = signal(160);
  readonly layerChoice = signal('First 3');
  readonly metrics = TENSOR_METRICS;
  readonly label = computed(() => 'Intermediate Tensors');
  readonly source = computed(() => this.request.data()?.rows ?? []);
  readonly layers = computed(() =>
    [...new Set(this.source().map((r) => r.layer))].sort((a, b) => a - b),
  );
  readonly anchors = computed(() =>
    [...new Set(this.source().map((r) => r.anchor))].sort((a, b) =>
      a === 'output' ? -1 : b === 'output' ? 1 : a.localeCompare(b),
    ),
  );
  readonly selectedAnchors = computed(() => {
    const anchors = this.anchors();
    const defaults =
      this.anchorScope() === 'output'
        ? [anchors.includes('output') ? 'output' : anchors[0]].filter((id): id is string => !!id)
        : this.anchorScope() === 'internal'
          ? anchors.filter((id) => id !== 'output')
          : anchors;
    return (
      ['min', 'max', 'mean'].includes(this.anchor())
        ? (this.anchorSelection() ?? defaults)
        : [this.anchor()]
    ).filter((id) => anchors.includes(id));
  });
  readonly aggregation = computed(() => ({
    dimension: 'syncPoint',
    label: 'Sync points',
    options: this.anchors().map((id) => ({id, label: tensorPointLabel(id)})),
    selected: this.selectedAnchors(),
    showSingleSelection: true,
    function: (['min', 'max', 'mean'].includes(this.anchor())
      ? this.anchor()
      : defaultPositionReduction(this.metric())) as HeatmapReduction,
  }));
  readonly metricSource = computed(() =>
    this.source().filter((row) => this.selectedAnchors().includes(row.anchor)),
  );
  setAggregation(value: AggregationChange) {
    this.anchorScope.set('all');
    this.anchorSelection.set(value.selection);
    this.anchor.set(value.function);
  }
  readonly fullRange = computed(() => {
    const rows = this.source();
    if (!rows.length) return null;
    let start = rows[0].position;
    let end = start;
    for (const row of rows) {
      if (row.position < start) start = row.position;
      if (row.position > end) end = row.position;
    }
    return {start, end: end + 1};
  });
  readonly range = computed(() => this.viewport() ?? this.fullRange());
  readonly bins = computed(() => (this.range() ? tensorBins(this.range()!, this.capacity()) : []));
  readonly rows = computed(() =>
    tensorRangeRows(this.metricSource(), this.layers(), this.bins(), this.anchor(), this.label()),
  );
  readonly metricName = computed(
    () => this.metrics.find((m) => m.id === this.metric())?.label ?? this.metric(),
  );
  readonly graphPosition = computed(() => {
    const full = this.fullRange();
    if (!full) return null;
    return Math.max(
      full.start,
      Math.min(full.end - 1, this.selected()?.start ?? this.range()?.start ?? full.start),
    );
  });
  setGraphPosition(value: number, inspect = true) {
    const full = this.fullRange();
    if (!full || !Number.isFinite(value)) return;
    const position = Math.max(full.start, Math.min(full.end - 1, Math.trunc(value)));
    this.selected.set({start: position, end: position + 1, layer: null});
    if (inspect) this.inspect.emit(this);
  }
  commitGraphPosition(event: Event) {
    const input = event.target as HTMLInputElement;
    this.setGraphPosition(input.valueAsNumber);
    input.value = this.graphPosition()?.toString() ?? '';
  }
  readonly graphRows = computed(() => {
    const position = this.graphPosition();
    if (position === null || this.request.loading() || this.request.error()) return [];
    return tensorRangeRows(
      this.source().filter((row) => row.anchor === 'output'),
      this.layers(),
      [{start: position, end: position + 1}],
      'output',
      this.label(),
    );
  });
  readonly moduleSummary = signal<'min' | 'max'>('max');
  readonly moduleSummaryLabels = {min: 'Min', max: 'Max'};
  setModuleSummary(value: string) {
    this.moduleSummary.set(value === 'Min' ? 'min' : 'max');
  }
  readonly graphDisplay = signal({legend: true});
  readonly graphDisplayDraft = signal({...this.graphDisplay()});
  toggleGraphDisplayOption(option: 'legend') {
    this.graphDisplayDraft.update((draft) => ({...draft, [option]: !draft[option]}));
  }
  readonly archDetails = signal<GraphArchSection[]>([]);
  readonly archView = viewChild(IntermediateGraphArch);
  mappingPairShape(reference?: string, target?: string) {
    if (reference && reference === target) return reference;
    return `Ref ${reference ?? '—'} · Target ${target ?? '—'}`;
  }
  mappingSourceLabel(name: string) {
    return name ? name.split('/').filter(Boolean).slice(-2).join(' / ') : '—';
  }
  readonly mappingTensorDetails = computed<ExplorerInfoItem[]>(() => {
    const selected = (this.archView()?.mapping()?.pairs ?? []).filter(row => row.selected);
    return selected.map(pair => ({
      label: `${pair.reference || '—'} ↔ ${pair.target || '—'}`,
      children: [
        {label: 'Reference source', value: pair.referenceSource ?? '—'},
        {label: 'Reference shape', value: pair.referenceShape ?? '—'},
        {label: 'Target source', value: pair.targetSource ?? '—'},
        {label: 'Target shape', value: pair.targetShape ?? '—'},
      ],
    }));
  });
  inspectArch(sections: GraphArchSection[]) {
    this.archDetails.set(sections);
    this.inspect.emit(this);
  }
  readonly archContext = computed(() => {
    const metric = this.metric(),
      rows = this.graphRows();
    const values = rows.map((row) => ({
      layer: row.layer,
      value: row.bins[0]?.metrics[metric]?.max ?? null,
    }));
    const layerCount = values.reduce((max, v) => Math.max(max, v.layer + 1), 35);
    const layers: (number | null)[] = Array(layerCount).fill(null);
    const colors: Record<string, {bgColor: string; textColor: string}> = {};
    const formatted: Record<string, string> = {};
    for (const {layer, value} of values) {
      if (layer < 0 || value === null || !Number.isFinite(value)) continue;
      layers[layer] = value;
      colors[String(value)] = {bgColor: this.color(value), textColor: 'var(--text)'};
      formatted[String(value)] = this.formatMetric(metric, value);
    }
    // Generated fixture data belongs to the packaged preview, not Display preferences.
    const simulatedMetrics = Array.from({length: 101}, (_, i) => {
      const value = metric === 'cosine_similarity' ? 1 - (2 * i) / 100 : (this.maximum() * i) / 100;
      return {
        value,
        text: this.formatMetric(metric, value),
        bgColor: this.color(value),
        textColor: 'var(--text)',
      };
    });
    return {
      stage: this.stage(),
      metricId: metric,
      summary: this.moduleSummary(),
      layers,
      colors,
      formatted,
      position: this.graphPosition(),
      legend: this.graphDisplay().legend,
      simulatedMetrics,
    };
  });
  readonly overview = computed(() => {
    const full = this.fullRange();
    if (!full) return null;
    const bins = tensorBins(full, this.capacity());
    return {
      ...full,
      context_id: '',
      bins,
      rows: tensorRangeRows(this.metricSource(), this.layers(), bins, this.anchor(), this.label()),
    };
  });
  readonly controls = computed(() =>
    heatmapControls(
      HEATMAP_DEFINITIONS.intermediate,
      {position: this.positionReduction(), syncPoint: this.anchor()},
      {syncPoint: this.anchors().map((id) => ({id, label: id}))},
    ).map((c) => ({
      ...c,
      disabled:
        c.id === 'position' &&
        (this.view() === 'graph'
          ? !this.range() || this.range()!.end - this.range()!.start <= 1
          : this.bins().every((b) => b.end - b.start === 1)),
    })),
  );
  readonly trendLayers = signal<number[] | null>(null);
  readonly highlightedLayer = signal<number | null>(null);
  readonly defaultLayers = computed(() => [...this.layers()]);
  readonly selectedLayers = computed(() =>
    (this.trendLayers() ?? this.defaultLayers()).filter((l) => this.layers().includes(l)),
  );
  readonly layerColor = layerColor;
  readonly reductionLabels: Record<HeatmapReduction, string> = {
    min: 'Minimum',
    max: 'Maximum',
    mean: 'Mean',
  };
  readonly reductionOptions = ['Minimum', 'Maximum', 'Mean'];
  readonly displayOpen = signal(false);
  readonly displayDraft = signal<{bins: HeatmapReduction | null; layers: number[]}>({
    bins: null,
    layers: [],
  });
  readonly displayQuery = signal('');
  readonly displayReduction = computed(
    () => this.displayDraft().bins ?? defaultPositionReduction(this.metric()),
  );
  readonly binsDisabled = computed(() => this.bins().every((bin) => bin.end - bin.start === 1));
  readonly displayFilteredLayers = computed(() => {
    const query = this.displayQuery()
      .trim()
      .toLowerCase()
      .replace(/^l(?=\d)/, 'layer ');
    return this.layers().filter((layer) => `layer ${layer}`.includes(query));
  });
  private sameLayers(a: readonly number[], b: readonly number[]) {
    return a.length === b.length && a.every((layer) => b.includes(layer));
  }
  readonly displayLayersDefault = computed(() =>
    this.sameLayers(this.displayDraft().layers, this.defaultLayers()),
  );
  readonly displayDirty = computed(() =>
    this.view() === 'graph'
      ? JSON.stringify(this.graphDisplayDraft()) !== JSON.stringify(this.graphDisplay())
      : (!this.binsDisabled() && this.displayDraft().bins !== this.positionOverride()) ||
        (this.view() === 'trends' &&
          !this.sameLayers(this.displayDraft().layers, this.selectedLayers())),
  );
  toggleDisplay() {
    if (this.displayOpen()) {
      this.closeDisplay();
      return;
    }
    this.displayDraft.set({bins: this.positionOverride(), layers: [...this.selectedLayers()]});
    this.graphDisplayDraft.set({...this.graphDisplay()});
    this.displayQuery.set('');
    this.displayOpen.set(true);
  }
  closeDisplay() {
    this.displayOpen.set(false);
    this.highlightedLayer.set(null);
  }
  draftReduction(label: string) {
    const bins = (Object.keys(this.reductionLabels) as HeatmapReduction[]).find(
      (key) => this.reductionLabels[key] === label,
    );
    if (bins) this.displayDraft.update((draft) => ({...draft, bins}));
  }
  resetDisplayReduction() {
    this.displayDraft.update((draft) => ({...draft, bins: null}));
  }
  draftLayers(layers: number[]) {
    this.displayDraft.update((draft) => ({...draft, layers}));
  }
  toggleDisplayLayer(layer: number) {
    const layers = this.displayDraft().layers;
    this.draftLayers(
      layers.includes(layer)
        ? layers.filter((id) => id !== layer)
        : [...layers, layer].sort((a, b) => a - b),
    );
  }
  applyDisplay() {
    if (!this.displayDirty()) return;
    if (this.view() === 'graph') {
      this.graphDisplay.set({...this.graphDisplayDraft()});
      return;
    }
    const draft = this.displayDraft();
    if (!this.binsDisabled()) this.positionOverride.set(draft.bins);
    if (this.view() === 'trends')
      this.trendLayers.set(draft.layers.filter((layer) => this.layers().includes(layer)));
    this.displayDraft.set({bins: this.positionOverride(), layers: [...this.selectedLayers()]});
  }
  readonly series = computed(() =>
    this.selectedLayers().map((layer) => ({
      layer,
      kind: this.label(),
      rows: this.rows(),
      color: layerColor(layer),
    })),
  );
  readonly maximum = computed(() =>
    Math.max(
      1e-12,
      ...(this.view() === 'graph' ? this.graphRows() : this.rows()).flatMap((r) =>
        r.bins.map((b) => b.metrics[this.metric()]?.max ?? 0),
      ),
    ),
  );
  readonly detailsRows = computed(() => {
    if (this.request.loading() || this.request.error()) return [];
    const selected = this.selected();
    return selected
      ? this.metricSource().filter(
          (r) =>
            r.position >= selected.start &&
            r.position < selected.end &&
            (this.view() !== 'trends' || this.selectedLayers().includes(r.layer)) &&
            (selected.layer === null || selected.layer === r.layer),
        )
      : [];
  });
  readonly formatMetric = (metric: string, value: number | null | undefined) =>
    value == null || !Number.isFinite(value) ? '-' : formatAnalysisMetric(metric, value);
  readonly formatChart = (value: number | null) => this.formatMetric(this.metric(), value);
  color(value: number) {
    const percent =
      this.metric() === 'cosine_similarity' ? (1 - value) * 50 : (value / this.maximum()) * 100;
    return `color-mix(in srgb,var(--heat-high) ${Math.max(0, Math.min(100, percent))}%,var(--heat-low))`;
  }
  changeMetric(value: string) {
    this.metric.set(value as KvMetric);
  }
  changeDimension(change: {id: string; value: string}) {
    if (change.id === 'position') this.positionOverride.set(change.value as HeatmapReduction);
    else this.anchor.set(change.value);
  }
  select(value: KvRangeSelection) {
    if (this.view() === 'graph' && !this.inspectionMode()) {
      this.setGraphPosition(value.start);
      return;
    }
    if (this.inspectionMode()) {
      const inspection = this.inspection();
      inspection?.setPosition(value.start);
      if (value.layer !== null) inspection?.setLayer(value.layer);
      this.inspect.emit(this);
      return;
    }
    this.selected.set(value);
    this.inspect.emit(this);
  }
  captureView() {
    const inspection =
      this.inspection()?.captureView() ?? this.inspectionState() ?? this.inspectionInitial();
    return {
      ...(inspection
        ? {
            mode: this.inspectionMode() ? ('inspection' as const) : ('comparison' as const),
            inspection,
          }
        : {}),
      batchId: null,
      metric: this.metric(),
      anchor: this.anchor(),
      anchorSelection: this.anchorSelection(),
      anchorScope: this.anchorScope(),
      view: this.view(),
      range: this.viewport(),
      positionReduction: this.positionOverride(),
      moduleSummary: this.moduleSummary(),
      ...(this.view() === 'graph' ? {graphDisplay: this.graphDisplay()} : {}),
      layerChoice: this.layerChoice(),
      trendLayers: this.trendLayers() ?? (this.layers().length ? this.selectedLayers() : undefined),
      selection: this.selected(),
    };
  }
  constructor() {
    effect(() => {
      const selected = this.chartSelection(),
        full = this.fullRange();
      if (
        !full ||
        this.view() !== 'graph' ||
        this.inspectionMode() ||
        !selected ||
        selected.end - selected.start !== 1
      )
        return;
      // External selection is an input event, not an authority that resets toolbar edits.
      untracked(() => {
        if (selected.start !== this.graphPosition()) this.setGraphPosition(selected.start, false);
      });
    });
    effect(() => {
      const selected = this.chartSelection(),
        inspection = this.inspection();
      if (!this.inspectionMode() || !inspection || !selected || selected.end - selected.start !== 1)
        return;
      untracked(() => {
        if (selected.start !== inspection.position()) inspection.setPosition(selected.start, false);
      });
    });
    effect(() => {
      if (this.request.loading() && this.inspectionMode())
        untracked(() =>
          this.inspectionInitial.set(this.inspectionState() ?? this.inspectionInitial()),
        );
    });
    effect(() => {
      if (this.view() === 'trends' && this.trendLayers() === null && this.layers().length) {
        const legacy = this.layerChoice();
        this.trendLayers.set(
          legacy === 'All layers' ? [...this.layers()] : [...this.defaultLayers()],
        );
      }
    });
    effect(() => {
      const saved = this.initial();
      if (!saved) {
        this.anchorScope.set(this.category() === 'intermediate' ? 'internal' : 'output');
        return;
      }
      const scope =
        saved.anchorScope ?? (this.category() === 'intermediate' ? 'internal' : 'output');
      this.anchorScope.set(scope);
      this.metric.set(
        TENSOR_METRICS.find((m) => m.id === saved.metric || m.source === saved.metric)?.id ??
          'cosine_similarity',
      );
      this.anchor.set(!saved.anchorScope && scope === 'output' ? 'min' : saved.anchor || 'min');
      this.anchorSelection.set(
        !saved.anchorScope && scope === 'output' ? null : (saved.anchorSelection ?? null),
      );
      this.view.set(saved.view ?? 'heatmap');
      this.viewport.set(saved.range ?? null);
      this.positionOverride.set(saved.positionReduction ?? null);
      // Migrate the former Output summary without discarding saved panel state.
      this.moduleSummary.set(saved.moduleSummary === 'min' ? 'min' : 'max');
      if (saved.graphDisplay)
        this.graphDisplay.set({
          legend: saved.graphDisplay.legend,
        });
      this.layerChoice.set(saved.layerChoice ?? 'First 3');
      this.trendLayers.set(
        saved.trendLayers ??
          (saved.layerChoice?.startsWith('L') ? [Number(saved.layerChoice.slice(1))] : null),
      );
      this.selected.set(saved.selection ?? null);
      this.inspectionInitial.set(saved.inspection ?? null);
      this.inspectionState.set(saved.inspection ?? null);
      this.inspectionMode.set(saved.mode === 'inspection');
    });
    inject(DestroyRef).onDestroy(() => {
      this.request.destroy();
      this.locationAnimation?.cancel();
    });
    effect((onCleanup) => {
      const capture = this.state.captureId(),
        turn = this.turn(),
        stage = this.stage(),
        alignment = this.positionAlignment();
      this.state.revision();
      this.retry();
      onCleanup(
        this.request.load(
          capture
            ? (signal) => this.api.positionAnalysis(capture, turn, stage, signal, alignment)
            : null,
        ),
      );
    });
  }
}
