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
  IntermediateGraphArch,
  type GraphArchSection,
  type GraphPairSave,
} from './intermediate_graph_arch';
import {ExplorerInfoSection} from '../../widgets/explorer_info_section/explorer_info_section';
import {
  ExplorerInfoList,
  type ExplorerInfoItem,
} from '../../widgets/explorer_info_list/explorer_info_list';
import {ViewActions} from '../../widgets/charts/view_actions';
import {tensorPointLabel} from './tensor_labels';
import {
  LayerActivationInspection,
  type LayerActivationInspectionState,
} from './layer_activation_inspection';
import {MatIconModule} from '@angular/material/icon';
import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';
import {OverlayModule} from '@angular/cdk/overlay';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {ExplorerInfoValue} from '../../widgets/explorer_info_value/explorer_info_value';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../widgets/locate_feedback';
import {PositionRangeControl} from '../../widgets/charts/position_range_control';
import {heatmapChartHeight} from '../../widgets/charts/heatmap_plot';
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
import {AnalysisCache} from '../../server/analysis_cache';
import {PositionAnalysisApi, type PositionRangeQuery} from '../../server/position_analysis_api';
import {ArtifactsApi} from '../../server/artifacts_api';
import {parseCaptureId} from '../../server/capture_id';
import {MappingsApi} from '../../server/mappings_api';
import type {ManualPairs} from '../../server/contracts/mappings';
import type {ExecutionGraphCollection} from '../../server/contracts/artifacts';
import {graphArchModel, type GraphSide, type ModelGraphFacts} from './graph_arch_data';
import {CaptureStore} from '../../server/capture_store';
import type {
  PositionAnalysis,
  PositionRangeResponse,
} from '../../server/contracts/position_analysis';
import type {KvMetric, KvPositionAlignment, KvRangeBin} from '../../server/contracts/kv';
import {reasonText} from '../../server/reasons';
import {LatestRequest} from '../../server/latest_request';
import {RangeChart} from '../../widgets/charts/range_chart';
import type {RangeSelection} from '../../widgets/charts/range_selection';
import {layerColor} from '../../widgets/charts/layer_color';
import {HeatmapControls} from '../../widgets/charts/heatmap_controls';
import type {AggregationChange} from '../../widgets/charts/aggregation_control';
import {formatMetricValue} from '../../widgets/format/format';
import {MetricCatalogStore} from '../../server/metric_catalog';
import {metricHeat, metricScaleLabels, scaleValue} from '../../widgets/metric_scale/metric_scale';
import {
  HEATMAP_DEFINITIONS,
  heatmapControls,
  defaultPositionReduction,
  type HeatmapReduction,
} from '../../widgets/charts/heatmap_model';
import {TENSOR_METRIC_SOURCES, tensorMetricId} from './position_tensor_model';
import {columnSteps} from '../shared/column_steps';
import {representativePosition} from '../../widgets/charts/range_bins';
import {rangeBinCount} from '../../widgets/charts/range_geometry';

/** The saved settings of an Intermediate Tensors panel, kept with its stage's analysis panels. */
export interface GraphPanelState {
  mode?: 'comparison' | 'inspection';
  inspection?: LayerActivationInspectionState;
  view?: 'heatmap' | 'lineChart' | 'graph';
  range?: {start: number; end: number} | null;
  positionReduction?: 'min' | 'max' | 'mean' | null;
  moduleSummary?: 'min' | 'max';
  graphDisplay?: {tensorMode?: string; simulation?: boolean; legend: boolean};
  lineChartLayers?: number[];
  selection?: {start: number; end: number; layer: number | null} | null;
  metric: string;
  anchor: string;
  anchorSelection?: string[] | null;
  /** Missing in older state, where it meant output anchors. */
  anchorScope?: 'output' | 'internal' | 'all';
  selected?: {layer: number; anchor: string} | null;
}

@Component({
  selector: 'stage-graph-metrics',
  standalone: true,
  imports: [
    IntermediateGraphArch,
    ExplorerInfoSection,
    ExplorerInfoList,
    LayerActivationInspection,
    ViewActions,
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
    RangeChart,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './graph_metric_panel.ng.html',
  styleUrl: './graph_metric_panel.scss',
})
export class GraphMetricPanel {
  readonly expanded = input(false);
  readonly chartSelection = input<RangeSelection | null>();
  readonly expand = output<void>();
  readonly heatmapChartHeight = heatmapChartHeight;
  readonly increment = (n: number) => n + 1;
  readonly panelId = input('');
  readonly initial = input<GraphPanelState | null>(null);
  readonly turn = input.required<number>();
  readonly stage = input.required<string>();
  readonly positionAlignment = input<KvPositionAlignment | null>(null);
  readonly inspect = output<GraphMetricPanel>();
  readonly backgroundClick = output<MouseEvent>();
  readonly positionHover = output<RangeSelection | null>();
  readonly detailsTemplate = viewChild<TemplateRef<unknown>>('details');
  readonly inspection = viewChild(LayerActivationInspection);
  readonly inspectionMode = signal(false);
  readonly inspectionInitial = signal<LayerActivationInspectionState | null>(null);
  private readonly inspectionState = signal<LayerActivationInspectionState | null>(null);
  readonly state = inject(CaptureStore);
  /** Which step produced the token at a column, on each side, in the server's words: the step
   *  whose tensors the column shows. */
  readonly columnSteps = (column: number) =>
    columnSteps(
      this.state.session(),
      this.turn(),
      column,
      this.positionAlignment(),
      'produced',
      'intermediate_tensors',
    );
  private readonly api = inject(PositionAnalysisApi);
  private readonly analysisCache = inject(AnalysisCache);
  /** The stage's places and cells; its bins over the visible range, over the whole stage, and at
   *  the Graph Diff's position: all the server's. */
  readonly request = new LatestRequest<PositionAnalysis>();
  /** The stage's analysis while the server works it out again after a pair made by hand changed:
   *  the one before, so that the Graph Diff stays where it is meanwhile. */
  private readonly refreshing = signal<PositionAnalysis | null>(null);
  private readonly analysisRevision = signal(0);
  private analysisKey = '';
  /** The pairs made by hand of the Session's two models, as the server keeps them. */
  readonly manualRequest = new LatestRequest<ManualPairs>();
  private readonly manualRevision = signal(0);
  private readonly mappingsApi = inject(MappingsApi);
  /** Why the server refused the last pair saved or deleted, in its words. */
  readonly pairError = signal('');
  readonly rangeRequest = new LatestRequest<PositionRangeResponse>();
  readonly overviewRequest = new LatestRequest<PositionRangeResponse>();
  readonly graphRequest = new LatestRequest<PositionRangeResponse>();
  readonly retry = signal(0);
  /** A failed request of the chart, its bins or its overview, in the server's words. */
  readonly comparisonError = computed(
    () => this.request.error() || this.rangeRequest.error() || this.overviewRequest.error(),
  );
  /** A failed request of the Graph Diff: the stage's places, or its layers at its position. */
  readonly graphError = computed(() => this.request.error() || this.graphRequest.error());
  /** Whether the panel waits for its first data: the stage's places and cells, then every answer
   *  its first view draws (the bins at the chart's capacity, the overview, the Graph Diff's), each
   *  of both alignments (AnalysisCache). A failure ends the wait. */
  readonly waiting = computed(
    () =>
      this.request.loading() ||
      this.rangeRequest.loading() ||
      this.overviewRequest.loading() ||
      this.graphRequest.loading() ||
      this.manualRequest.loading() ||
      this.modelRequest.loading() ||
      (!this.comparisonError() &&
        !!this.range() &&
        !!this.selectedAnchors().length &&
        !this.rangeRequest.data()),
  );
  /** Whether the chart waits for its first bins: laid out but unseen meanwhile, so that it asks
   *  for as many as it draws (chart_state). */
  readonly chartPending = computed(() => this.rangeRequest.loading() && !this.rows().length);
  readonly metric = signal<KvMetric>('cosine_similarity');
  readonly anchor = signal('min');
  readonly anchorSelection = signal<string[] | null>(null);
  readonly anchorScope = signal<'output' | 'internal' | 'all'>('output');
  private readonly heatmapControls = viewChild(HeatmapControls);
  private readonly rangeChart = viewChild(RangeChart);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly locationInjector = inject(Injector);
  private readonly layerDisplay = viewChild<ElementRef<HTMLElement>>('layerDisplay');
  private locationAnimation?: Animation;
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
  locateView(): void {
    if (this.inspectionMode()) {
      this.inspection()?.locateView();
      return;
    }
    this.locateControl(this.host.nativeElement.querySelector<HTMLElement>('.view-picker > button'));
  }
  locateRange(): void {
    if (this.inspectionMode()) {
      this.inspection()?.locatePosition();
      return;
    }
    if (this.view() === 'graph') {
      this.locateControl(
        this.host.nativeElement.querySelector<HTMLElement>('.position-control input'),
      );
      return;
    }
    const control = this.host.nativeElement.querySelector<HTMLElement>('position-range-control');
    this.locateControl(
      control?.querySelector<HTMLElement>('.range-inputs'),
      control?.querySelector<HTMLElement>('.range-location-frame'),
    );
  }
  locateLayers(): void {
    if (this.inspectionMode()) {
      this.inspection()?.locateLayer();
      return;
    }
    if (this.view() !== 'lineChart') {
      return;
    }
    if (!this.displayOpen()) {
      this.toggleDisplay();
    }
    afterNextRender(() => this.locateControl(this.layerDisplay()?.nativeElement), {
      injector: this.locationInjector,
    });
  }
  locateMetric(): void {
    this.heatmapControls()?.locateMetric();
  }
  locateSummary(): void {
    this.locateControl(
      this.host.nativeElement.querySelector<HTMLElement>('.toolbar-summary button'),
    );
  }
  locateAggregation(): void {
    this.heatmapControls()?.locateDimension('syncPoint');
  }
  locateChart(position?: number, layer?: number): void {
    this.rangeChart()?.locateKind(this.label(), position, layer);
  }
  readonly view = signal<'heatmap' | 'lineChart' | 'graph'>('heatmap');
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
  tokenRepresentative(bin: KvRangeBin, layer = this.selected()?.layer): number {
    const row = this.rows().find((row) => row.layer === layer) ?? this.rows()[0];
    const cell = row?.bins.find((cell) => cell.start === bin.start && cell.end === bin.end);
    return representativePosition(bin, this.positionReduction(), cell?.metrics[this.metric()]);
  }

  chooseView(value: string): void {
    if (!this.visualizationOptions.includes(value)) {
      return;
    }
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
    this.showComparison(value === this.visualizationOptions[0] ? 'heatmap' : 'lineChart');
  }
  /** The sync point Inspection opens: for a min or max over sync points, the one the server names
   *  as the extremum's at the position; else the one browsed, or the first that holds a tensor. */
  private inspectionAnchor(position: number, layer: number, previous?: string): string {
    const anchors = this.selectedAnchors();
    if (anchors.length === 1) {
      return anchors[0];
    }
    if (this.anchor() === 'min' || this.anchor() === 'max') {
      const summary = this.rows()
        .find((row) => row.layer === layer)
        ?.bins.find((bin) => bin.start <= position && position < bin.end)?.metrics[this.metric()];
      const named =
        summary?.min_position === position
          ? summary.min_anchor
          : summary?.max_position === position
            ? summary.max_anchor
            : null;
      if (named && anchors.includes(named)) {
        return named;
      }
    }
    // A mean has no source tensor: retain the browsing point or start at an available point.
    if (previous && anchors.includes(previous)) {
      return previous;
    }
    const rows = this.metricSource().filter(
      (row) => row.position === position && row.layer === layer,
    );
    return rows.find((row) => row.shape?.length)?.anchor ?? anchors[0] ?? 'output';
  }
  showComparison(view: 'heatmap' | 'lineChart' | 'graph'): void {
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
  cacheInspection(inspection: LayerActivationInspection): void {
    const next = inspection.captureView();
    const previous = this.inspectionState();
    this.inspectionState.set(next);
    if (previous && (next.anchor ?? 'output') !== (previous.anchor ?? 'output')) {
      this.anchorScope.set('all');
      this.anchorSelection.set([next.anchor ?? 'output']);
      this.anchor.set(next.anchor ?? 'output');
    }
    if (previous && (next.position !== previous.position || next.layer !== previous.layer)) {
      this.selected.set({start: next.position, end: next.position + 1, layer: next.layer});
      if (next.layer !== null && !this.selectedLayers().includes(next.layer)) {
        this.lineChartLayers.set([...this.selectedLayers(), next.layer].sort((a, b) => a - b));
      }
    }
  }
  inspectionChanged(inspection: LayerActivationInspection): void {
    this.cacheInspection(inspection);
    this.inspect.emit(this);
  }
  readonly positionOverride = signal<HeatmapReduction | null>(null);
  readonly positionReduction = computed(
    () => this.positionOverride() ?? defaultPositionReduction(this.description()),
  );
  readonly selected = signal<RangeSelection | null>(null);
  readonly viewport = signal<KvRangeBin | null>(null);
  readonly capacity = signal(160);
  private readonly catalog = inject(MetricCatalogStore);
  /** The metrics the catalog's intermediate tensors analysis offers. */
  readonly metrics = computed(() =>
    this.catalog
      .lookup()
      .choices('intermediate_tensors')
      .filter((choice) => choice.id in TENSOR_METRIC_SOURCES)
      .map((choice) => ({id: choice.id, label: choice.label})),
  );
  /** The selected metric as the catalog describes it. */
  readonly description = computed(() => this.catalog.lookup().metric(this.metric()));
  /** The two ends of the color legend. */
  readonly scaleEnds = computed(() => {
    const labels = metricScaleLabels(this.description(), 'auto', this.maximum());
    return [labels[0] ?? '', this.formatChart(scaleValue(this.description(), 1, this.maximum()))];
  });
  readonly label = computed(() => 'Intermediate Tensors');
  readonly source = computed(() => this.request.data()?.rows ?? []);
  /** The stage's layers and sync points, in the server's order (layer outputs first). */
  readonly layers = computed(() => this.request.data()?.layers ?? []);
  readonly anchors = computed(() => (this.request.data()?.anchors ?? []).map((a) => a.id));
  /** The server's label of each sync point. */
  readonly anchorLabels = computed<Record<string, string>>(() =>
    Object.fromEntries((this.request.data()?.anchors ?? []).map((a) => [a.id, a.label])),
  );
  anchorLabel(anchor: string | null | undefined): string {
    return (anchor && this.anchorLabels()[anchor]) || tensorPointLabel(anchor);
  }
  /** Whether the server's batches hold observed tensors of this stage: only then is it asked. */
  readonly captured = computed(
    () =>
      this.state
        .session()
        ?.batches.some((batch) => batch.turn === this.turn() && batch.phase === this.stage()) ??
      false,
  );
  /** Why the stage shows no tensors, in the server's words; '' when it has some. */
  readonly unavailable = computed(() => {
    const answer = this.request.data();
    if (answer?.reasons.length) {
      return reasonText(answer.reasons);
    }
    if (answer || this.captured()) {
      return '';
    }
    const stage = this.state.session()?.turns.find((turn) => turn.n === this.turn())?.stages?.[
      this.stage() as 'prefill' | 'decode'
    ];
    return reasonText(stage?.analyses.intermediate_tensors.reasons);
  });
  /** Why each point chosen of a LiteRT-LM side is paired with nothing on the PyTorch side, in the
   *  server's words. */
  readonly unpaired = computed(() =>
    (this.request.data()?.unpaired ?? []).map((reason) => reasonText([reason])),
  );
  /** The sync points Tensor Comparison aggregates: the architecture's anchors, each layer's
   *  input, its blocks' outputs and its output, as the server marks them; every one without. */
  readonly comparisonAnchors = computed(() => {
    const marked = (this.request.data()?.anchors ?? []).filter((a) => a.default).map((a) => a.id);
    return marked.length ? marked : this.anchors();
  });
  readonly selectedAnchors = computed(() => {
    const anchors = this.anchors();
    const listed = this.comparisonAnchors();
    const defaults =
      this.anchorScope() === 'output'
        ? [anchors.includes('output') ? 'output' : anchors[0]].filter((id): id is string => !!id)
        : this.anchorScope() === 'internal'
          ? listed.filter((id) => id !== 'output')
          : listed;
    return (
      ['min', 'max', 'mean'].includes(this.anchor())
        ? (this.anchorSelection() ?? defaults)
        : [this.anchor()]
    ).filter((id) => anchors.includes(id));
  });
  readonly aggregation = computed(() => ({
    dimension: 'syncPoint',
    label: 'Sync points',
    // Another sync point stays listed while it is chosen (Inspection's).
    options: [...new Set([...this.comparisonAnchors(), ...this.selectedAnchors()])].map((id) => ({
      id,
      label: this.anchorLabel(id),
    })),
    selected: this.selectedAnchors(),
    showSingleSelection: true,
    function: (['min', 'max', 'mean'].includes(this.anchor())
      ? this.anchor()
      : defaultPositionReduction(this.description())) as HeatmapReduction,
  }));
  readonly metricSource = computed(() =>
    this.source().filter((row) => this.selectedAnchors().includes(row.anchor)),
  );
  setAggregation(value: AggregationChange): void {
    this.anchorScope.set('all');
    this.anchorSelection.set(value.selection);
    this.anchor.set(value.function);
  }
  /** The stage's columns, as the server's analysis gives them. */
  readonly fullRange = computed(() => {
    const answer = this.request.data();
    return answer && answer.rows.length && answer.end > answer.start
      ? {start: answer.start, end: answer.end}
      : null;
  });
  readonly range = computed(() => this.viewport() ?? this.fullRange());
  /** How the selected sync points' values combine at a position, as the request names it. */
  private readonly anchorReduction = computed(() => this.aggregation().function);
  private rangeQuery(range: KvRangeBin, bins: number, anchors = this.selectedAnchors()) {
    return {
      ...range,
      bins,
      anchors,
      reduction: this.anchorReduction(),
      layers: null,
    } satisfies PositionRangeQuery;
  }
  /** The server's bins of the visible range: the last answer's while the next one loads. */
  private readonly drawn = computed(() => this.rangeRequest.data() ?? this.rangeRequest.last());
  readonly bins = computed(() => this.drawn()?.bins ?? []);
  readonly rows = computed(() => this.drawn()?.rows ?? []);
  /** Each layer's statistics over the visible range, one bin each, of the answer the chart draws:
   *  the list and the chart show one view. */
  readonly totals = computed(() => this.drawn()?.totals ?? []);
  readonly metricName = computed(() => this.catalog.lookup().label(this.metric()));
  readonly graphPosition = computed(() => {
    const full = this.fullRange();
    if (!full) {
      return null;
    }
    return Math.max(
      full.start,
      Math.min(full.end - 1, this.selected()?.start ?? this.range()?.start ?? full.start),
    );
  });
  setGraphPosition(value: number, inspect = true): void {
    const full = this.fullRange();
    if (!full || !Number.isFinite(value)) {
      return;
    }
    const position = Math.max(full.start, Math.min(full.end - 1, Math.trunc(value)));
    this.selected.set({start: position, end: position + 1, layer: null});
    if (inspect) {
      this.inspect.emit(this);
    }
  }
  commitGraphPosition(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.setGraphPosition(input.valueAsNumber);
    input.value = this.graphPosition()?.toString() ?? '';
  }
  /** Each layer's output at the Graph Diff's position, as the server compared it. */
  readonly graphRows = computed(() => {
    if (this.graphPosition() === null || this.request.loading() || this.request.error()) {
      return [];
    }
    return this.graphRequest.data()?.rows ?? [];
  });
  readonly moduleSummary = signal<'min' | 'max'>('max');
  readonly moduleSummaryLabels = {min: 'Min', max: 'Max'};
  setModuleSummary(value: string): void {
    this.moduleSummary.set(value === 'Min' ? 'min' : 'max');
  }
  readonly graphDisplay = signal({legend: true});
  readonly graphDisplayDraft = signal({...this.graphDisplay()});
  toggleGraphDisplayOption(option: 'legend'): void {
    this.graphDisplayDraft.update((draft) => ({...draft, [option]: !draft[option]}));
  }
  readonly archDetails = signal<GraphArchSection[]>([]);
  readonly archView = viewChild(IntermediateGraphArch);
  mappingPairShape(reference?: string, target?: string): string {
    if (reference && reference === target) {
      return reference;
    }
    return `Ref ${reference ?? '—'} · Target ${target ?? '—'}`;
  }
  mappingSourceLabel(name: string): string {
    return name ? name.split('/').filter(Boolean).slice(-2).join(' / ') : '—';
  }
  readonly mappingTensorDetails = computed<ExplorerInfoItem[]>(() => {
    const selected = (this.archView()?.mapping()?.pairs ?? []).filter((row) => row.selected);
    return selected.map((pair) => ({
      label: `${pair.reference || '—'} ↔ ${pair.target || '—'}`,
      children: [
        {label: 'Reference source', value: pair.referenceSource ?? '—'},
        {label: 'Reference shape', value: pair.referenceShape ?? '—'},
        {label: 'Target source', value: pair.targetSource ?? '—'},
        {label: 'Target shape', value: pair.targetShape ?? '—'},
      ],
    }));
  });
  inspectArch(sections: GraphArchSection[]): void {
    this.archDetails.set(sections);
    this.inspect.emit(this);
  }
  private readonly artifactsApi = inject(ArtifactsApi);
  /** Each side's model, as the Session's runs name it. */
  readonly artifacts = computed<Record<GraphSide, string> | null>(() => {
    const runs = this.state.session()?.runs ?? [];
    const ref = runs.find((run) => run.id === 'ref')?.artifact;
    const target = runs.find((run) => run.id === 'target')?.artifact ?? ref;
    return ref && target ? {ref, target} : null;
  });
  /** What the server says of each side's model's graph: where it is, and its architecture. */
  readonly modelRequest = new LatestRequest<Partial<Record<GraphSide, ModelGraphFacts>>>();
  /** The model at the Graph Diff's column, as the server's answers give it (graph_arch_data). */
  readonly archModel = computed(() =>
    graphArchModel({
      facts: this.modelRequest.data(),
      loading: this.modelRequest.loading(),
      error: this.modelRequest.error(),
      analysis: this.request.data() ?? this.refreshing(),
      manual: this.manualRequest.data() ?? this.manualRequest.last(),
      stage: this.stage(),
      position: this.graphPosition(),
      metric: this.metric(),
      reason: (code) => this.catalog.lookup().reason(code),
      format: (value) => this.formatMetric(this.metric(), value),
      color: (value) => this.color(value),
    }),
  );
  readonly archContext = computed(() => {
    const metric = this.metric();
    const rows = this.graphRows();
    const model = this.archModel();
    const count = Math.max(model.layerCount, ...rows.map((row) => row.layer + 1));
    const layers: (number | null)[] = Array(count).fill(null);
    const colors: Record<string, {bgColor: string; textColor: string}> = {};
    const formatted: Record<string, string> = {};
    for (const row of rows) {
      const value = row.bins[0]?.metrics[metric]?.max ?? null;
      if (row.layer < 0 || value === null || !Number.isFinite(value)) {
        continue;
      }
      layers[row.layer] = value;
      colors[String(value)] = {bgColor: this.color(value), textColor: 'var(--text)'};
      formatted[String(value)] = this.formatMetric(metric, value);
    }
    return {
      stage: this.stage(),
      metricId: metric,
      metricName: this.metricName(),
      summary: this.moduleSummary(),
      layers,
      colors,
      formatted,
      position: this.graphPosition(),
      legend: this.graphDisplay().legend,
      model,
    };
  });
  /** What the Mapping pane says of pairs made by hand: the server's refusal of the last one, or why
   *  none is made (one model on both sides needs none); '' otherwise. */
  readonly mappingNote = computed(() => {
    const manual = this.manualRequest.data() ?? this.manualRequest.last();
    return (
      this.pairError() ||
      (manual && manual.status !== 'pairs' ? manual.text : '') ||
      this.manualRequest.error()
    );
  });
  /** Asks the server to save a pair made in the Mapping pane; it checks and keeps it for the two
   *  models, and the stage's analysis is asked again, which places it. */
  async savePair(pair: GraphPairSave): Promise<void> {
    const session = this.sessionId();
    if (!session) {
      return;
    }
    try {
      await this.mappingsApi.save(session, pair);
      this.pairError.set('');
    } catch (error) {
      this.pairError.set(error instanceof Error ? error.message : String(error));
      return;
    }
    this.pairsChanged();
  }
  /** Asks the server to delete pairs made by hand, by id. */
  async deletePairs(ids: string[]): Promise<void> {
    const session = this.sessionId();
    if (!session) {
      return;
    }
    try {
      await Promise.all(ids.map((id) => this.mappingsApi.delete(session, id)));
      this.pairError.set('');
    } catch (error) {
      this.pairError.set(error instanceof Error ? error.message : String(error));
    }
    this.pairsChanged();
  }
  /** The Session of the chat shown: the pairs made by hand are its two models', not a chat's. */
  private sessionId(): string | null {
    const capture = this.state.captureId();
    return (capture && parseCaptureId(capture)?.session) || null;
  }
  private pairsChanged(): void {
    // The kept answers were compared under the pairs before.
    this.analysisCache.forget();
    this.manualRevision.update((n) => n + 1);
    this.analysisRevision.update((n) => n + 1);
  }
  /** Each signature's execution graph, asked once per model. */
  private readonly executionGraphs = new Map<string, Promise<ExecutionGraphCollection>>();
  /** Answers the Graph Diff's ask for a side's execution graph of a signature. */
  loadExecutionGraph(request: {side: GraphSide; signature: string}): void {
    const artifact = this.artifacts()?.[request.side];
    const view = this.archView();
    if (!artifact || !view) {
      return;
    }
    const key = artifact + '\0' + request.signature;
    let graph = this.executionGraphs.get(key);
    if (!graph) {
      graph = this.artifactsApi.signatureGraph(artifact, request.signature);
      this.executionGraphs.set(key, graph);
      graph.catch(() => this.executionGraphs.delete(key));
    }
    graph.then(
      (collection) => view.deliverGraph({...request, artifact, collection}),
      (error: unknown) =>
        view.deliverGraph({
          ...request,
          artifact,
          error: error instanceof Error ? error.message : String(error),
        }),
    );
  }
  readonly overview = computed(() => {
    const full = this.fullRange();
    const answer = this.overviewRequest.data() ?? this.overviewRequest.last();
    if (!full || !answer) {
      return null;
    }
    return {...full, context_id: '', bins: answer.bins, rows: answer.rows};
  });
  readonly controls = computed(() => {
    const range = this.range();
    return heatmapControls(
      HEATMAP_DEFINITIONS.intermediate,
      {position: this.positionReduction(), syncPoint: this.anchor()},
      {syncPoint: this.anchors().map((id) => ({id, label: id}))},
    ).map((c) => ({
      ...c,
      disabled:
        c.id === 'position' &&
        (this.view() === 'graph'
          ? !range || range.end - range.start <= 1
          : this.bins().every((b) => b.end - b.start === 1)),
    }));
  });
  readonly lineChartLayers = signal<number[] | null>(null);
  readonly highlightedLayer = signal<number | null>(null);
  readonly defaultLayers = computed(() => [...this.layers()]);
  readonly selectedLayers = computed(() =>
    (this.lineChartLayers() ?? this.defaultLayers()).filter((l) => this.layers().includes(l)),
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
    () => this.displayDraft().bins ?? defaultPositionReduction(this.description()),
  );
  readonly binsDisabled = computed(() => this.bins().every((bin) => bin.end - bin.start === 1));
  readonly displayFilteredLayers = computed(() => {
    const query = this.displayQuery()
      .trim()
      .toLowerCase()
      .replace(/^l(?=\d)/, 'layer ');
    return this.layers().filter((layer) => `layer ${layer}`.includes(query));
  });
  private sameLayers(a: readonly number[], b: readonly number[]): boolean {
    return a.length === b.length && a.every((layer) => b.includes(layer));
  }
  readonly displayLayersDefault = computed(() =>
    this.sameLayers(this.displayDraft().layers, this.defaultLayers()),
  );
  readonly displayDirty = computed(() =>
    this.view() === 'graph'
      ? JSON.stringify(this.graphDisplayDraft()) !== JSON.stringify(this.graphDisplay())
      : (!this.binsDisabled() && this.displayDraft().bins !== this.positionOverride()) ||
        (this.view() === 'lineChart' &&
          !this.sameLayers(this.displayDraft().layers, this.selectedLayers())),
  );
  toggleDisplay(): void {
    if (this.displayOpen()) {
      this.closeDisplay();
      return;
    }
    this.displayDraft.set({bins: this.positionOverride(), layers: [...this.selectedLayers()]});
    this.graphDisplayDraft.set({...this.graphDisplay()});
    this.displayQuery.set('');
    this.displayOpen.set(true);
  }
  closeDisplay(): void {
    this.displayOpen.set(false);
    this.highlightedLayer.set(null);
  }
  draftReduction(label: string): void {
    const bins = (['min', 'max', 'mean'] as const).find(
      (key) => this.reductionLabels[key] === label,
    );
    if (bins) {
      this.displayDraft.update((draft) => ({...draft, bins}));
    }
  }
  resetDisplayReduction(): void {
    this.displayDraft.update((draft) => ({...draft, bins: null}));
  }
  draftLayers(layers: number[]): void {
    this.displayDraft.update((draft) => ({...draft, layers}));
  }
  toggleDisplayLayer(layer: number): void {
    const layers = this.displayDraft().layers;
    this.draftLayers(
      layers.includes(layer)
        ? layers.filter((id) => id !== layer)
        : [...layers, layer].sort((a, b) => a - b),
    );
  }
  applyDisplay(): void {
    if (!this.displayDirty()) {
      return;
    }
    if (this.view() === 'graph') {
      this.graphDisplay.set({...this.graphDisplayDraft()});
      return;
    }
    const draft = this.displayDraft();
    if (!this.binsDisabled()) {
      this.positionOverride.set(draft.bins);
    }
    if (this.view() === 'lineChart') {
      this.lineChartLayers.set(draft.layers.filter((layer) => this.layers().includes(layer)));
    }
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
    if (this.request.loading() || this.request.error()) {
      return [];
    }
    const selected = this.selected();
    return selected
      ? this.metricSource().filter(
          (r) =>
            r.position >= selected.start &&
            r.position < selected.end &&
            (this.view() !== 'lineChart' || this.selectedLayers().includes(r.layer)) &&
            (selected.layer === null || selected.layer === r.layer),
        )
      : [];
  });
  readonly formatMetric = (metric: string, value: number | null | undefined): string =>
    value == null || !Number.isFinite(value)
      ? '-'
      : formatMetricValue(this.catalog.lookup().metric(metric), value);
  readonly formatChart = (value: number | null): string => this.formatMetric(this.metric(), value);
  color(value: number): string {
    const percent = (metricHeat(this.description(), value, 'auto', this.maximum()) ?? 0) * 100;
    return `color-mix(in srgb,var(--heat-high) ${Math.max(0, Math.min(100, percent))}%,var(--heat-low))`;
  }
  changeMetric(value: string): void {
    this.metric.set(value as KvMetric);
  }
  select(value: RangeSelection): void {
    if (this.view() === 'graph' && !this.inspectionMode()) {
      this.setGraphPosition(value.start);
      return;
    }
    if (this.inspectionMode()) {
      const inspection = this.inspection();
      inspection?.setPosition(value.start);
      if (value.layer !== null) {
        inspection?.setLayer(value.layer);
      }
      this.inspect.emit(this);
      return;
    }
    this.selected.set(value);
    this.inspect.emit(this);
  }
  captureView(): GraphPanelState {
    const inspection =
      this.inspection()?.captureView() ?? this.inspectionState() ?? this.inspectionInitial();
    return {
      ...(inspection
        ? {
            mode: this.inspectionMode() ? ('inspection' as const) : ('comparison' as const),
            inspection,
          }
        : {}),
      metric: this.metric(),
      anchor: this.anchor(),
      anchorSelection: this.anchorSelection(),
      anchorScope: this.anchorScope(),
      view: this.view(),
      range: this.viewport(),
      positionReduction: this.positionOverride(),
      moduleSummary: this.moduleSummary(),
      ...(this.view() === 'graph' ? {graphDisplay: this.graphDisplay()} : {}),
      lineChartLayers:
        this.lineChartLayers() ?? (this.layers().length ? this.selectedLayers() : undefined),
      selection: this.selected(),
    };
  }
  constructor() {
    effect(() => {
      const selected = this.chartSelection();
      const full = this.fullRange();
      if (
        !full ||
        this.view() !== 'graph' ||
        this.inspectionMode() ||
        !selected ||
        selected.end - selected.start !== 1
      ) {
        return;
      }
      // External selection is an input event, not an authority that resets toolbar edits.
      untracked(() => {
        if (selected.start !== this.graphPosition()) {
          this.setGraphPosition(selected.start, false);
        }
      });
    });
    effect(() => {
      const selected = this.chartSelection();
      const inspection = this.inspection();
      if (
        !this.inspectionMode() ||
        !inspection ||
        !selected ||
        selected.end - selected.start !== 1
      ) {
        return;
      }
      untracked(() => {
        if (selected.start !== inspection.position()) {
          inspection.setPosition(selected.start, false);
        }
      });
    });
    effect(() => {
      if (this.request.loading() && this.inspectionMode()) {
        untracked(() =>
          this.inspectionInitial.set(this.inspectionState() ?? this.inspectionInitial()),
        );
      }
    });
    effect(() => {
      // Opening the Line chart with no layers chosen keeps the layers of the current data.
      if (this.view() === 'lineChart' && this.lineChartLayers() === null && this.layers().length) {
        this.lineChartLayers.set([...this.defaultLayers()]);
      }
    });
    effect(() => {
      const saved = this.initial();
      if (!saved) {
        this.anchorScope.set('output');
        return;
      }
      const scope = saved.anchorScope ?? 'output';
      this.anchorScope.set(scope);
      const metric = tensorMetricId(saved.metric ?? '');
      this.metric.set(
        (metric in TENSOR_METRIC_SOURCES
          ? metric
          : (this.catalog.lookup().analysis('intermediate_tensors')?.default_metric ??
            'cosine_similarity')) as KvMetric,
      );
      this.anchor.set(!saved.anchorScope && scope === 'output' ? 'min' : saved.anchor || 'min');
      this.anchorSelection.set(
        !saved.anchorScope && scope === 'output' ? null : (saved.anchorSelection ?? null),
      );
      this.view.set(saved.view ?? 'heatmap');
      this.viewport.set(saved.range ?? null);
      this.positionOverride.set(saved.positionReduction ?? null);
      this.moduleSummary.set(saved.moduleSummary ?? 'max');
      if (saved.graphDisplay) {
        this.graphDisplay.set({
          legend: saved.graphDisplay.legend,
        });
      }
      this.lineChartLayers.set(saved.lineChartLayers ?? null);
      this.selected.set(saved.selection ?? null);
      this.inspectionInitial.set(saved.inspection ?? null);
      this.inspectionState.set(saved.inspection ?? null);
      this.inspectionMode.set(saved.mode === 'inspection');
    });
    inject(DestroyRef).onDestroy(() => {
      for (const request of [
        this.request,
        this.rangeRequest,
        this.overviewRequest,
        this.graphRequest,
        this.modelRequest,
        this.manualRequest,
      ]) {
        request.destroy();
      }
      this.locationAnimation?.cancel();
    });
    effect((onCleanup) => {
      const capture = this.state.captureId();
      const turn = this.turn();
      const stage = this.stage();
      const alignment = this.positionAlignment();
      const captured = this.captured();
      this.retry();
      this.analysisRevision();
      // Asked again for the same stage (a pair made by hand changed), the last answer stays shown
      // until the new one arrives.
      const key = JSON.stringify([capture, turn, stage, alignment, captured]);
      untracked(() =>
        this.refreshing.set(
          key === this.analysisKey ? (this.request.data() ?? this.refreshing()) : null,
        ),
      );
      this.analysisKey = key;
      onCleanup(
        this.request.load(
          capture && captured
            ? (signal) => this.api.positionAnalysis(capture, turn, stage, signal, alignment)
            : null,
          () => this.refreshing.set(null),
        ),
      );
    });
    // The pairs made by hand of the Session's two models, for the Graph Diff's Mapping pane.
    effect((onCleanup) => {
      const session = this.sessionId();
      const shown = this.view() === 'graph' && !this.inspectionMode();
      this.manualRevision();
      onCleanup(
        this.manualRequest.load(
          session && shown ? (signal) => this.mappingsApi.pairs(session, signal) : null,
        ),
      );
    });
    // The Graph Diff's model: each side's graph status and architecture, asked once it is shown.
    effect((onCleanup) => {
      const artifacts = this.artifacts();
      const shown = this.view() === 'graph' && !this.inspectionMode();
      this.retry();
      onCleanup(
        this.modelRequest.load(
          artifacts && shown
            ? async (signal) => {
                const facts = async (artifact: string): Promise<ModelGraphFacts> => {
                  const status = await this.artifactsApi.graphStatus(artifact, signal);
                  const architecture =
                    status.status === 'completed'
                      ? await this.artifactsApi.architecture(artifact, signal)
                      : null;
                  return {artifact, status, architecture};
                };
                const ref = await facts(artifacts.ref);
                const target =
                  artifacts.target === artifacts.ref ? ref : await facts(artifacts.target);
                return {ref, target};
              }
            : null,
        ),
      );
    });
    // The bins the chart draws, its overview and the Graph Diff's layers: each asked anew as the
    // view changes; a drag or resize asks once it settles.
    const ranged = (
      request: LatestRequest<PositionRangeResponse>,
      query: () => PositionRangeQuery | null,
      delay = 0,
    ) =>
      effect((onCleanup) => {
        const capture = this.state.captureId();
        const turn = this.turn();
        const stage = this.stage();
        const alignment = this.positionAlignment();
        const asked = this.request.data() ? query() : null;
        onCleanup(
          request.load(
            capture && asked
              ? (signal) =>
                  this.api.positionRange(
                    capture,
                    turn,
                    stage,
                    asked,
                    signal,
                    alignment,
                    untracked(this.capacity),
                  )
              : null,
            undefined,
            delay,
          ),
        );
      });
    ranged(
      this.rangeRequest,
      () => {
        const range = this.range();
        return range && this.selectedAnchors().length
          ? this.rangeQuery(range, rangeBinCount(range, this.capacity()))
          : null;
      },
      100,
    );
    ranged(
      this.overviewRequest,
      () => {
        const full = this.fullRange();
        return full && this.selectedAnchors().length
          ? this.rangeQuery(full, rangeBinCount(full, this.capacity()))
          : null;
      },
      100,
    );
    ranged(this.graphRequest, () => {
      const position = this.graphPosition();
      return this.view() === 'graph' && position !== null && this.anchors().includes('output')
        ? this.rangeQuery({start: position, end: position + 1}, 1, ['output'])
        : null;
    });
  }
}
