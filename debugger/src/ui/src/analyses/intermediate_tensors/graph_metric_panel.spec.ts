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

import {signal} from '@angular/core';
import {TestBed, type ComponentFixture} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {PositionAnalysisApi, type PositionRangeQuery} from '../../server/position_analysis_api';
import {CaptureStore} from '../../server/capture_store';
import {MappingsApi} from '../../server/mappings_api';
import type {
  PositionAnalysis,
  PositionRangeResponse,
  PositionTensorRow,
} from '../../server/contracts/position_analysis';
import type {KvRangeMetricSummary} from '../../server/contracts/kv';
import type {PlotHost, PlotlyApi} from '../../widgets/plotly_loader';
import {HeatmapControls} from '../../widgets/charts/heatmap_controls';
import type {layerColor as LayerColorFn} from '../../widgets/charts/layer_color';
import type {GraphMetricPanel as GraphMetricPanelType, GraphPanelState} from './graph_metric_panel';
import {
  IntermediateGraphArch,
  type GraphArchContext,
  type GraphArchSection,
  type GraphPairSave,
  type GraphRequest,
} from './intermediate_graph_arch';

const METRICS = ['cosine_similarity', 'cosine_distance', 'relative_l2', 'max_abs'];
function row(
  position: number,
  layer: number,
  anchor: string,
  values: Record<string, number | null>,
): PositionTensorRow {
  return {
    position,
    layer,
    anchor,
    status: 'ok',
    reasons: [],
    shape: [8],
    metrics: Object.fromEntries(
      Object.entries(values).map(([id, value]) => [id, {value, reasons: []}]),
    ),
  };
}
/** The server's analysis of these cells: their layers, sync points (layer outputs first) and
 *  columns. */
function analysisOf(rows: PositionTensorRow[]): PositionAnalysis {
  const positions = rows.map((r) => r.position);
  const anchors = [...new Set(rows.map((r) => r.anchor))].sort((a, b) =>
    a === 'output' ? -1 : b === 'output' ? 1 : a.localeCompare(b),
  );
  return {
    turn: 2,
    stage: 'decode',
    alignment: null,
    label: 'Intermediate Tensors',
    start: Math.min(...positions),
    end: Math.max(...positions) + 1,
    layers: [...new Set(rows.map((r) => r.layer))].sort((a, b) => a - b),
    anchors: anchors.map((id) => ({id, label: id === 'output' ? 'Layer output' : id})),
    points: [],
    pairing: {status: 'paired', reasons: []},
    rows,
    provenance: {kind: 'synthetic', label: 'Test data'},
    reasons: [],
  };
}
/** What the server's position-range answers of an analysis: as analysis/tensor_analysis does. */
function rangeOf(answer: PositionAnalysis, query: PositionRangeQuery): PositionRangeResponse {
  const count = Math.max(1, Math.min(query.bins, query.end - query.start));
  const edges = Array.from(
    {length: count + 1},
    (_, i) => query.start + Math.floor((i * (query.end - query.start)) / count),
  );
  const bins = edges.slice(0, -1).map((start, i) => ({start, end: edges[i + 1]}));
  const anchors = query.anchors ?? answer.anchors.map((a) => a.id);
  const statistic = (start: number, end: number, layer: number, metric: string) => {
    const values: {position: number; value: number; anchor: string | null}[] = [];
    for (let position = start; position < end; position++) {
      const cells = answer.rows
        .filter((r) => r.layer === layer && r.position === position && anchors.includes(r.anchor))
        .map((r) => ({value: r.metrics[metric]?.value ?? null, anchor: r.anchor}))
        .filter((c): c is {value: number; anchor: string} => c.value !== null);
      if (!cells.length) {
        continue;
      }
      if (query.reduction === 'mean' && cells.length > 1) {
        values.push({
          position,
          value: cells.reduce((sum, c) => sum + c.value, 0) / cells.length,
          anchor: null,
        });
      } else {
        const pick = cells.reduce((a, b) =>
          query.reduction === 'min' ? (b.value < a.value ? b : a) : b.value > a.value ? b : a,
        );
        values.push({position, ...pick});
      }
    }
    const min = values.reduce<(typeof values)[number] | null>(
      (a, b) => (!a || b.value < a.value ? b : a),
      null,
    );
    const max = values.reduce<(typeof values)[number] | null>(
      (a, b) => (!a || b.value > a.value ? b : a),
      null,
    );
    const summary: KvRangeMetricSummary & {min_anchor?: string | null; max_anchor?: string | null} =
      {
        min: min?.value ?? null,
        max: max?.value ?? null,
        mean: values.length ? values.reduce((sum, v) => sum + v.value, 0) / values.length : null,
        min_position: min?.position ?? null,
        max_position: max?.position ?? null,
        valid_count: values.length,
      };
    if (query.reduction !== 'mean') {
      summary.min_anchor = min?.anchor ?? null;
      summary.max_anchor = max?.anchor ?? null;
    }
    return summary;
  };
  const rowsOf = (spans: {start: number; end: number}[]) =>
    answer.layers.map((layer) => ({
      layer,
      kind: 'Intermediate Tensors',
      status: 'ok',
      reasons: [],
      bins: spans.map((bin) => ({
        ...bin,
        metrics: Object.fromEntries(
          METRICS.map((metric) => [metric, statistic(bin.start, bin.end, layer, metric)]),
        ) as PositionRangeResponse['rows'][number]['bins'][number]['metrics'],
      })),
    }));
  return {
    turn: 2,
    stage: 'decode',
    alignment: null,
    start: query.start,
    end: query.end,
    anchors: [...anchors],
    reduction: query.reduction,
    layers: answer.layers,
    bins,
    rows: rowsOf(bins),
    totals: rowsOf([{start: query.start, end: query.end}]),
    metadata: {
      ref: {dimensions: [], shape: null, dtype: null},
      target: {dimensions: [], shape: null, dtype: null},
      output: {
        dimensions: ['Token position', 'Layer'],
        shape: [query.end - query.start, answer.layers.length],
        dtype: 'float64',
      },
    },
  };
}

const data: PositionAnalysis = analysisOf(
  Array.from({length: 6}, (_, layer) =>
    Array.from({length: 320}, (_, position) =>
      row(position, layer, 'output', {
        cosine_similarity: 1 - layer / 10,
        relative_l2: layer / 100,
        max_abs: layer / 20,
      }),
    ),
  ).flat(),
);
/** Lets the panel's range requests, which wait for a drag to settle, be answered. */
async function settle(fixture: ComponentFixture<unknown>) {
  for (let i = 0; i < 2; i++) {
    fixture.detectChanges();
    await new Promise((resolve) => setTimeout(resolve, 130));
    await fixture.whenStable();
  }
  fixture.detectChanges();
}

describe('Activation comparison toolbar', () => {
  // The panel whose analysis the fake server bins.
  let current: GraphMetricPanelType | null = null;
  let GraphMetricPanel: typeof GraphMetricPanelType;
  let layerColor: typeof LayerColorFn;
  beforeAll(async () => {
    // Initialize the shared metric module before the builder's lazy component chunk.
    // Static cross-chunk imports otherwise capture its not-yet-initialized export in Vitest.
    ({GraphMetricPanel} = await import('./graph_metric_panel'));
    ({layerColor} = await import('../../widgets/charts/layer_color'));
  });
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
    }));
    TestBed.configureTestingModule({
      providers: [
        {
          provide: PositionAnalysisApi,
          useValue: {
            positionRange: async (
              _capture: string,
              _turn: number,
              _stage: string,
              query: PositionRangeQuery,
            ) => rangeOf(current!.request.data()!, query),
          },
        },
        {
          provide: CaptureStore,
          useValue: {captureId: signal('capture'), session: signal(null), revision: signal(0)},
        },
      ],
    });
    // Keep compiled signal-input metadata intact while the real toolbar is rendered.
    (window as Window & {Plotly?: PlotlyApi}).Plotly = {
      react: vi.fn(async (host: HTMLElement) => {
        (host as PlotHost).on = () => {};
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
  });
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('applies graph Display as a draft and restores persisted preferences', () => {
    const {panel} = create();
    panel.view.set('graph');
    panel.toggleDisplay();
    panel.toggleGraphDisplayOption('legend');
    expect(panel.graphDisplay().legend).toBe(true);
    panel.closeDisplay();
    panel.toggleDisplay();
    expect(panel.graphDisplayDraft().legend).toBe(true);
    panel.toggleGraphDisplayOption('legend');
    panel.applyDisplay();
    expect(panel.graphDisplay().legend).toBe(false);
    // Every value the Graph Diff shows is the server's: it is given no generated metrics.
    expect(panel.archContext()).not.toHaveProperty('simulatedMetrics');
    const saved = panel.captureView();
    expect(saved.graphDisplay).not.toHaveProperty('simulation');
    const restored = create({
      ...saved,
      graphDisplay: {...saved.graphDisplay!, tensorMode: 'name', simulation: false},
    }).panel;
    expect(restored.captureView().graphDisplay).not.toHaveProperty('tensorMode');
    expect(restored.captureView().graphDisplay).not.toHaveProperty('simulation');
    expect(restored.archContext()).not.toHaveProperty('simulatedMetrics');
    expect(restored.graphDisplay()).toEqual(panel.graphDisplay());
  });

  function create(initial?: GraphPanelState) {
    const fixture = TestBed.createComponent(GraphMetricPanel);
    // The metrics are the catalog's intermediate tensors analysis.
    expect(fixture.componentInstance.metrics().map((metric) => metric.id)).toEqual([
      'cosine_similarity',
      'relative_l2',
      'max_abs',
    ]);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('stage', 'decode');
    if (initial) {
      fixture.componentRef.setInput('initial', initial);
    }
    fixture.detectChanges();
    const panel = fixture.componentInstance;
    current = panel;
    panel.request.data.set(data);
    fixture.detectChanges();
    return {fixture, panel};
  }

  it('shares comparison state with Graph Diff and restores the saved view', async () => {
    const {fixture, panel} = create();
    panel.viewport.set({start: 10, end: 20});
    panel.chooseView('Tensor Comparison · Graph Diff');
    await settle(fixture);
    expect(panel.view()).toBe('graph');
    expect(panel.range()).toEqual({start: 10, end: 20});
    expect(fixture.nativeElement.querySelector('intermediate-graph-arch')).not.toBeNull();
    expect(fixture.nativeElement.querySelectorAll('.activation-controls').length).toBe(1);
    expect(panel.archContext().layers[0]).toBe(1);
    // As many layers as the server's rows have, without a model's count: no fixed 35.
    expect(panel.archContext().layers).toHaveLength(6);
    expect(panel.captureView().view).toBe('graph');
    panel.chooseView('Tensor Comparison · Heatmap');
    fixture.detectChanges();
    expect(panel.range()).toEqual({start: 10, end: 20});
    expect(fixture.nativeElement.querySelector('intermediate-graph-arch')).toBeNull();
    fixture.destroy();
    const restored = create({
      metric: 'cosine_similarity',
      anchor: 'min',
      view: 'graph',
    });
    expect(restored.panel.view()).toBe('graph');
  });

  it('keeps module summaries separate from tensor aggregation and restores the choice', async () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Graph Diff');
    await settle(fixture);
    expect(panel.archContext().summary).toBe('max');
    const layers = panel.archContext().layers;
    panel.setModuleSummary('Max');
    fixture.detectChanges();
    expect(panel.archContext().summary).toBe('max');
    expect(panel.archContext().layers).toEqual(layers);
    panel.setModuleSummary('Min');
    expect(panel.archContext().summary).toBe('min');
    panel.setModuleSummary('Max');
    expect(fixture.nativeElement.querySelector('.toolbar-summary')).not.toBeNull();
    const saved = panel.captureView();
    expect(saved.moduleSummary).toBe('max');
    panel.chooseView('Tensor Comparison · Heatmap');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.toolbar-summary')).toBeNull();
    fixture.destroy();
    const restored = create(saved);
    expect(restored.panel.archContext().summary).toBe('max');
  });

  it('lists the server’s reason for each point paired with nothing', () => {
    const {fixture, panel} = create();
    const text =
      'PyTorch has no counterpart for decode:239:0 (layer 3, FULLY_CONNECTED): it is inside ' +
      'Attention q_proj, not its output; only the anchors are captured there.';
    panel.request.data.set({
      ...data,
      unpaired: [{code: 'tensor_no_counterpart', side: null, text}],
    });
    fixture.detectChanges();
    const list = fixture.nativeElement.querySelector('ul.unpaired') as HTMLElement;
    expect(list.getAttribute('aria-label')).toBe('Points paired with nothing');
    expect(list.textContent?.trim()).toBe(text);
    panel.request.data.set(data);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('ul.unpaired')).toBeNull();
  });

  it('keeps Min/Max available without data', () => {
    const {fixture, panel} = create({
      metric: 'cosine_similarity',
      anchor: 'output',
      view: 'graph',
    });
    expect(panel.moduleSummary()).toBe('max');
    panel.request.data.set(null);
    fixture.detectChanges();
    const summary = fixture.debugElement
      .queryAll(By.directive(ConfigPicker))
      .find((item) => item.componentInstance.label() === 'Module summary')!.componentInstance;
    expect(summary.options()).toEqual(['Min', 'Max']);
    expect(summary.disabled()).toBe(false);
    panel.setModuleSummary('Min');
    expect(panel.moduleSummary()).toBe('min');
  });

  it('allows toolbar navigation after an external token selection without snapping back', () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Graph Diff');
    fixture.componentRef.setInput('chartSelection', {start: 10, end: 11, layer: null});
    fixture.detectChanges();
    expect(panel.graphPosition()).toBe(10);
    panel.setGraphPosition(11);
    fixture.detectChanges();
    expect(panel.graphPosition()).toBe(11);
    expect(panel.selected()).toEqual({start: 11, end: 12, layer: null});
    fixture.componentRef.setInput('chartSelection', {start: 13, end: 14, layer: null});
    fixture.detectChanges();
    expect(panel.graphPosition()).toBe(13);
  });

  it('uses a single token and fixed layer outputs without inheriting hidden aggregation', async () => {
    const {fixture, panel} = create();
    panel.request.data.set(
      analysisOf([
        row(10, 0, 'output', {cosine_similarity: 0.8}),
        row(11, 0, 'output', {cosine_similarity: 0.4}),
        row(12, 0, 'attn', {cosine_similarity: 0.1}),
      ]),
    );
    panel.viewport.set({start: 10, end: 13});
    panel.anchorScope.set('internal');
    panel.anchor.set('max');
    panel.chooseView('Tensor Comparison · Graph Diff');
    await settle(fixture);
    expect(panel.graphPosition()).toBe(10);
    expect(panel.archContext().layers[0]).toBe(0.8);
    expect(fixture.nativeElement.querySelector('position-range-control')).toBeNull();
    expect(fixture.nativeElement.querySelector('aggregation-control')).toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]').hidden).toBe(false);
    expect(
      fixture.nativeElement.querySelector('[aria-label="Previous token position"]').disabled,
    ).toBe(true);
    fixture.nativeElement.querySelector('[aria-label="Next token position"]').click();
    await settle(fixture);
    expect(panel.graphPosition()).toBe(11);
    expect(panel.archContext().layers[0]).toBe(0.4);
    fixture.componentRef.setInput('chartSelection', {start: 12, end: 13, layer: null});
    await settle(fixture);
    expect(panel.graphPosition()).toBe(12);
    expect(panel.archContext().layers[0]).toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Next token position"]').disabled).toBe(
      true,
    );
    expect(panel.captureView().selection).toEqual({start: 12, end: 13, layer: null});
    panel.chooseView('Tensor Comparison · Heatmap');
    fixture.detectChanges();
    expect(panel.range()).toEqual({start: 10, end: 13});
    expect(panel.anchorScope()).toBe('internal');
    expect(panel.anchor()).toBe('max');
    expect(fixture.nativeElement.querySelector('aggregation-control')).not.toBeNull();
  });

  it('offers comparison and inspection views, keeps comparison scope, and moves bins out of the toolbar', () => {
    const {fixture, panel} = create();
    const picker = fixture.debugElement.query(By.css('.view-picker'))
      .componentInstance as ConfigPicker;
    expect(picker.groups()).toEqual([
      {
        label: 'Tensor Comparison',
        options: [
          'Tensor Comparison · Heatmap',
          'Tensor Comparison · Line chart',
          'Tensor Comparison · Graph Diff',
        ],
      },
      {
        label: 'Tensor Inspection',
        options: ['Tensor Inspection · Line chart', 'Tensor Inspection · Scatter plot'],
      },
    ]);
    const controls = fixture.debugElement.query(By.directive(HeatmapControls))
      .componentInstance as HeatmapControls;
    expect(controls.showRangeControls()).toBe(false);
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]')).not.toBeNull();
    const selected = {start: 44, end: 46, layer: 2};
    panel.viewport.set({start: 32, end: 96});
    panel.select(selected);
    picker.valueChange.emit('Tensor Comparison · Line chart');
    fixture.detectChanges();
    expect(panel.view()).toBe('lineChart');
    expect(panel.range()).toEqual({start: 32, end: 96});
    expect(panel.selected()).toEqual(selected);
    expect(panel.series().map((series) => series.color)).toEqual(
      panel.selectedLayers().map(layerColor),
    );
    expect(
      fixture.nativeElement.querySelector('[aria-label="Layer legend"]').textContent,
    ).toContain('Layers 6/6');
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    picker.valueChange.emit('Tensor Comparison · Heatmap');
    expect(panel.view()).toBe('heatmap');
    expect(panel.selected()).toEqual(selected);
    fixture.destroy();
  });

  it('keeps the current layers when the Line chart opens with none chosen and more layers arrive', () => {
    const {fixture, panel} = create();
    expect(panel.lineChartLayers()).toBeNull();
    panel.chooseView('Tensor Comparison · Line chart');
    fixture.detectChanges();
    expect(panel.lineChartLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    const extra = data.rows.filter((row) => row.layer === 0).map((row) => ({...row, layer: 6}));
    panel.request.data.set(analysisOf([...data.rows, ...extra]));
    fixture.detectChanges();
    expect(panel.layers()).toContain(6);
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    expect(panel.captureView().lineChartLayers).toEqual([0, 1, 2, 3, 4, 5]);
    fixture.destroy();
  });

  const mixedRows = () =>
    data.rows.flatMap((row) => [
      row,
      {
        ...row,
        anchor: 'attention_output',
        metrics: {...row.metrics, cosine_similarity: {value: 0.2, reasons: []}},
      },
      {
        ...row,
        anchor: 'pre_attention_norm',
        metrics: {...row.metrics, cosine_similarity: {value: 0.9, reasons: []}},
      },
    ]);

  it("lists the architecture's anchors to aggregate, not an operator chosen inside the layers", () => {
    const {fixture, panel} = create();
    const answer = analysisOf(mixedRows());
    // The server marks output and pre_attention_norm as anchors; attention_output is an operator.
    answer.anchors = answer.anchors.map((a) => ({...a, default: a.id !== 'attention_output'}));
    panel.request.data.set(answer);
    fixture.detectChanges();
    expect(panel.aggregation().options.map((o) => o.id)).toEqual(['output', 'pre_attention_norm']);
    panel.setAggregation({selection: ['output', 'pre_attention_norm'], function: 'min'});
    expect(panel.selectedAnchors()).toEqual(['output', 'pre_attention_norm']);
    // An operator Inspection chose stays listed while it is chosen.
    panel.setAggregation({selection: ['attention_output'], function: 'min'});
    expect(panel.aggregation().options.map((o) => o.id)).toEqual([
      'output',
      'pre_attention_norm',
      'attention_output',
    ]);
    fixture.destroy();
  });

  it('defaults to output, exposes actual anchors, and persists selection through a loading state', () => {
    const {fixture, panel} = create();
    panel.request.data.set(analysisOf(mixedRows()));
    fixture.detectChanges();
    expect(panel.anchors()).toEqual(['output', 'attention_output', 'pre_attention_norm']);
    expect(panel.selectedAnchors()).toEqual(['output']);
    expect(panel.metricSource().every((row) => row.anchor === 'output')).toBe(true);
    expect(
      fixture.nativeElement.querySelector('[aria-label="Sync points aggregation"]').textContent,
    ).toContain('Layer output');
    panel.setAggregation({selection: ['attention_output'], function: 'max'});
    panel.lineChartLayers.set([1, 4]);
    const saved = panel.captureView();
    panel.request.data.set(null);
    panel.request.loading.set(true);
    expect(panel.captureView().anchorSelection).toEqual(['attention_output']);
    expect(panel.captureView().lineChartLayers).toEqual([1, 4]);
    fixture.destroy();
    const restored = create(saved);
    restored.panel.request.data.set(analysisOf(mixedRows()));
    restored.fixture.detectChanges();
    expect(restored.panel.selectedAnchors()).toEqual(['attention_output']);
    restored.fixture.destroy();
  });

  it('inspects the extremum anchor and syncs a changed inspection point back into comparison', async () => {
    const {fixture, panel} = create();
    fixture.componentRef.setInput('expanded', true);
    panel.request.data.set(analysisOf(mixedRows()));
    await settle(fixture);
    panel.select({start: 44, end: 45, layer: 2});
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'min'});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.anchor()).toBe('attention_output');
    panel.inspection()!.zoomElements({start: 2, end: 6});
    panel.inspection()!.selectedIndex.set(3);
    panel.inspection()!.setAnchor('pre_attention_norm');
    fixture.detectChanges();
    panel.showComparison('heatmap');
    fixture.detectChanges();
    expect(panel.selectedAnchors()).toEqual(['pre_attention_norm']);
    expect(panel.selected()).toEqual({start: 44, end: 46, layer: 2});
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'mean'});
    panel.chooseView('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    expect(panel.inspection()?.captureView()).toMatchObject({
      anchor: 'pre_attention_norm',
      position: 44,
      layer: 2,
      elementRange: {start: 2, end: 6},
      selection: null,
    });
    panel.showComparison('heatmap');
    fixture.detectChanges();
    expect(panel.selectedAnchors()).toEqual(['attention_output', 'pre_attention_norm']);
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'max'});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.anchor()).toBe('pre_attention_norm');
    fixture.destroy();
  });

  it('opens Inspection at the sync point the server names as the extremum', async () => {
    const {fixture, panel} = create();
    panel.request.data.set(analysisOf(mixedRows()));
    await settle(fixture);
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'min'});
    await settle(fixture);
    // The server names pre_attention_norm the minimum at position 44 of layer 2, whichever the
    // rows' own values would make it: the panel takes the server's word.
    const named = structuredClone(panel.rangeRequest.data()!);
    for (const row of named.rows) {
      for (const bin of row.bins) {
        if (row.layer === 2 && bin.start <= 44 && 44 < bin.end) {
          Object.assign(bin.metrics[panel.metric()]!, {
            min_position: 44,
            min_anchor: 'pre_attention_norm',
          });
        }
      }
    }
    panel.rangeRequest.data.set(named);
    panel.select({start: 44, end: 45, layer: 2});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.anchor()).toBe('pre_attention_norm');
    fixture.destroy();
  });

  it('keeps the statistics of the drawn bins while the next bins load', async () => {
    const {fixture, panel} = create();
    await settle(fixture);
    const drawn = panel.rangeRequest.data()!;
    expect(panel.totals().length).toBeGreaterThan(0);
    // A resize asks for other bins: the chart keeps its drawing, and the list the same view's.
    panel.rangeRequest.data.set(null);
    panel.rangeRequest.loading.set(true);
    expect(panel.rows()).toBe(drawn.rows);
    expect(panel.totals()).toBe(drawn.totals);
    expect(panel.chartPending()).toBe(false);
    expect(panel.waiting()).toBe(true);
    fixture.destroy();
  });

  it("asks the pairs made by hand of the Session, not of the chat's capture", async () => {
    const asked: string[] = [];
    const empty = {status: 'pairs', pairs: []};
    TestBed.overrideProvider(CaptureStore, {
      useValue: {captureId: signal('session-1#2'), session: signal(null), revision: signal(0)},
    });
    TestBed.overrideProvider(MappingsApi, {
      useValue: {
        pairs: async (id: string) => (asked.push('pairs ' + id), empty),
        save: async (id: string) => (asked.push('save ' + id), empty),
        delete: async (id: string) => (asked.push('delete ' + id), empty),
      },
    });
    const {fixture, panel} = create();
    panel.view.set('graph');
    await settle(fixture);
    const tensor = {signature: 'decode', node: '1', output: '0'};
    await panel.savePair({ref: tensor, target: tensor});
    await panel.deletePairs(['p1']);
    expect(asked).toContain('pairs session-1');
    expect(asked).toContain('save session-1');
    expect(asked).toContain('delete session-1');
    expect(asked.some((call) => call.includes('#'))).toBe(false);
    fixture.destroy();
  });

  it("shows the overview's and the Graph Diff's request errors, with Retry", async () => {
    const {fixture, panel} = create();
    await settle(fixture);
    const alert = () => fixture.nativeElement.querySelector('[role=alert]') as HTMLElement | null;
    expect(alert()).toBeNull();
    panel.overviewRequest.error.set('The overview could not be read.');
    fixture.detectChanges();
    expect(alert()?.textContent).toContain('The overview could not be read.');
    expect(alert()?.querySelector('button')?.textContent).toContain('Retry');
    panel.overviewRequest.error.set('');
    panel.view.set('graph');
    fixture.detectChanges();
    expect(alert()).toBeNull();
    panel.graphRequest.error.set('The layers could not be read.');
    fixture.detectChanges();
    expect(alert()?.textContent).toContain('The layers could not be read.');
    // The structure graph stays.
    expect(fixture.nativeElement.querySelector('intermediate-graph-arch')).not.toBeNull();
    fixture.destroy();
  });

  it('opens the selected layer and position in Inspection and retains both mode configurations', () => {
    const {fixture, panel} = create();
    fixture.componentRef.setInput('expanded', true);
    panel.viewport.set({start: 32, end: 96});
    panel.select({start: 44, end: 45, layer: 2});
    const comparison = panel.captureView();
    panel.chooseView('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    expect(panel.inspectionMode()).toBe(true);
    expect(panel.inspection()).toBeDefined();
    expect(panel.captureView()).toMatchObject({
      ...comparison,
      mode: 'inspection',
      inspection: {position: 44, layer: 2, plot: 'scatter'},
    });
    panel.select({start: 45, end: 46, layer: null});
    fixture.detectChanges();
    expect(panel.inspection()?.position()).toBe(45);
    expect(panel.selected()).toEqual({start: 45, end: 46, layer: 2});
    panel.request.loading.set(true);
    panel.request.data.set(null);
    fixture.detectChanges();
    expect(panel.inspection()).toBeUndefined();
    panel.request.data.set(data);
    panel.request.loading.set(false);
    fixture.detectChanges();
    expect(panel.inspection()?.captureView()).toMatchObject({
      position: 45,
      layer: 2,
      plot: 'scatter',
    });
    panel.showComparison('lineChart');
    fixture.detectChanges();
    expect(panel.inspectionMode()).toBe(false);
    expect(panel.range()).toEqual(comparison.range);
    expect(panel.selected()).toEqual({start: 45, end: 46, layer: 2});
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    const saved = panel.captureView();
    fixture.destroy();
    const restored = create({...saved, mode: 'inspection'});
    expect(restored.panel.inspectionMode()).toBe(true);
    expect(restored.panel.inspection()?.captureView()).toMatchObject({
      position: 45,
      layer: 2,
      plot: 'scatter',
    });
    restored.fixture.destroy();
  });

  it.each([false, true])(
    'enters a selected bin at its representative in expanded=%s',
    async (expanded) => {
      const {fixture, panel} = create();
      fixture.componentRef.setInput('expanded', expanded);
      panel.request.data.set(
        analysisOf(
          data.rows.map((row) => ({
            ...row,
            metrics: {
              ...row.metrics,
              cosine_similarity: {value: row.position % 4 === 2 ? 0.1 : 0.9, reasons: []},
            },
          })),
        ),
      );
      panel.viewport.set({start: 40, end: 48});
      panel.capacity.set(2);
      await settle(fixture);
      panel.select({start: 40, end: 44, layer: 2});
      panel.chooseView('Tensor Inspection · Line chart');
      fixture.detectChanges();
      expect(panel.inspection()?.position()).toBe(42);
      panel.inspection()!.setPosition(46);
      fixture.detectChanges();
      panel.showComparison('heatmap');
      expect(panel.selected()).toEqual(
        expanded ? {start: 44, end: 48, layer: 2} : {start: 46, end: 47, layer: 2},
      );
      expect(panel.range()).toEqual({start: 40, end: 48});
      fixture.destroy();
    },
  );

  it('syncs external token selection and passive inspection state without taking details focus', () => {
    const {fixture, panel} = create();
    panel.select({start: 44, end: 45, layer: 2});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    const inspect = vi.fn();
    panel.inspect.subscribe(inspect);
    fixture.componentRef.setInput('chartSelection', {start: 46, end: 47, layer: null});
    fixture.detectChanges();
    expect(panel.inspection()?.position()).toBe(46);
    expect(panel.captureView().inspection?.position).toBe(46);
    expect(inspect).not.toHaveBeenCalled();
    panel.inspection()!.setPosition(47);
    expect(inspect).toHaveBeenCalledOnce();
    fixture.destroy();
  });

  it('retains inspection zoom and runtime display across mode switches and clears old element selection for a new tensor', () => {
    const {fixture, panel} = create();
    panel.select({start: 44, end: 45, layer: 2});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    const inspection = panel.inspection()!;
    inspection.zoomElements({start: 2, end: 6});
    inspection.visibleRuntimes.set(['target']);
    inspection.selectedIndex.set(3);
    panel.showComparison('heatmap');
    fixture.detectChanges();
    panel.select({start: 48, end: 49, layer: 4});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.captureView()).toMatchObject({
      position: 48,
      layer: 4,
      elementRange: {start: 2, end: 6},
      visibleRuntimes: ['target'],
      selection: null,
    });
    fixture.destroy();
  });

  it('discards unapplied bins and layers on close, and only commits on Apply', async () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Line chart');
    await settle(fixture);
    panel.toggleDisplay();
    const before = panel.captureView();
    panel.draftReduction('Mean');
    panel.draftLayers([0, 2]);
    expect(panel.displayDirty()).toBe(true);
    expect(panel.captureView()).toEqual(before);
    panel.closeDisplay();
    panel.toggleDisplay();
    expect(panel.displayDraft()).toEqual({bins: null, layers: before.lineChartLayers});
    expect(panel.displayDirty()).toBe(false);
    panel.draftReduction('Mean');
    panel.draftLayers([0, 2]);
    panel.applyDisplay();
    expect(panel.captureView()).toEqual({
      ...before,
      positionReduction: 'mean',
      lineChartLayers: [0, 2],
    });
    expect(panel.displayDirty()).toBe(false);
    fixture.destroy();
  });

  it('restores metric-dependent defaults only after Apply without changing scope or selection', async () => {
    const {fixture, panel} = create();
    panel.viewport.set({start: 32, end: 280});
    await settle(fixture);
    panel.selected.set({start: 40, end: 43, layer: 1});
    panel.positionOverride.set('mean');
    panel.toggleDisplay();
    const before = panel.captureView();
    panel.resetDisplayReduction();
    expect(panel.positionReduction()).toBe('mean');
    panel.applyDisplay();
    expect(panel.captureView()).toEqual({...before, positionReduction: null});
    expect(panel.positionReduction()).toBe('min');
    panel.changeMetric('relative_l2');
    expect(panel.positionReduction()).toBe('max');
    expect(panel.range()).toEqual(before.range);
    expect(panel.selected()).toEqual(before.selection);
    fixture.destroy();
  });

  it('disables bins for one position while still applying layer visibility and preserving the selected cell', async () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Line chart');
    fixture.detectChanges();
    panel.viewport.set({start: 45, end: 46});
    await settle(fixture);
    panel.selected.set({start: 45, end: 46, layer: 5});
    panel.positionOverride.set('mean');
    panel.toggleDisplay();
    panel.draftReduction('Minimum');
    expect(panel.binsDisabled()).toBe(true);
    expect(panel.displayDirty()).toBe(false);
    panel.applyDisplay();
    expect(panel.positionOverride()).toBe('mean');
    panel.draftLayers([]);
    panel.applyDisplay();
    expect(panel.positionOverride()).toBe('mean');
    expect(panel.selectedLayers()).toEqual([]);
    expect(panel.detailsRows()).toEqual([]);
    expect(panel.selected()).toEqual({start: 45, end: 46, layer: 5});
    expect(panel.range()).toEqual({start: 45, end: 46});
    panel.draftLayers([5]);
    panel.applyDisplay();
    expect(panel.detailsRows().map((row) => [row.position, row.layer])).toEqual([[45, 5]]);
    fixture.destroy();
  });

  it('restores the saved comparison state and keeps it when switching visualizations', async () => {
    const initial: GraphPanelState = {
      metric: 'relative_l2',
      anchor: 'min',
      anchorSelection: null,
      view: 'lineChart',
      range: {start: 24, end: 260},
      positionReduction: 'mean',
      lineChartLayers: [0, 3],
      selection: {start: 30, end: 32, layer: 3},
    };
    const {fixture, panel} = create(initial);
    await settle(fixture);
    expect(panel.captureView()).toEqual({
      ...initial,
      anchorScope: 'output',
      moduleSummary: 'max',
    });
    panel.chooseView('Tensor Comparison · Heatmap');
    panel.chooseView('Tensor Comparison · Line chart');
    expect(panel.captureView()).toEqual({
      ...initial,
      anchorScope: 'output',
      moduleSummary: 'max',
    });
    panel.toggleDisplay();
    expect(panel.displayDraft()).toEqual({bins: 'mean', layers: [0, 3]});
    panel.draftLayers(panel.defaultLayers());
    expect(panel.selectedLayers()).toEqual([0, 3]);
    panel.applyDisplay();
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    fixture.destroy();
  });

  it('locates the View and Range without changing data, and only opens layer settings for a line chart', () => {
    const {fixture, panel} = create();
    panel.viewport.set({start: 32, end: 96});
    panel.selected.set({start: 40, end: 41, layer: 1});
    const before = panel.captureView();
    const picker = fixture.nativeElement.querySelector('.view-picker > button') as HTMLElement;
    const range = fixture.nativeElement.querySelector('.range-inputs') as HTMLElement;
    const frame = fixture.nativeElement.querySelector('.range-location-frame') as HTMLElement;
    picker.scrollIntoView = vi.fn();
    range.scrollIntoView = vi.fn();
    picker.animate = vi.fn().mockReturnValue({cancel: vi.fn()});
    frame.animate = vi.fn().mockReturnValue({cancel: vi.fn()});
    panel.locateView();
    panel.locateRange();
    expect(picker.scrollIntoView).toHaveBeenCalled();
    expect(frame.animate).toHaveBeenCalled();
    expect(panel.captureView()).toEqual(before);
    panel.locateLayers();
    expect(panel.displayOpen()).toBe(false);
    panel.chooseView('Tensor Comparison · Line chart');
    panel.locateLayers();
    expect(panel.displayOpen()).toBe(true);
    expect(panel.selected()).toEqual(before.selection);
    expect(panel.range()).toEqual(before.range);
    fixture.destroy();
  });

  it('formats all captured comparison metrics and missing values for the metric table', () => {
    const {fixture, panel} = create();
    expect(panel.metrics().map((metric) => metric.id)).toEqual([
      'cosine_similarity',
      'relative_l2',
      'max_abs',
    ]);
    // Relative L2 is a percent already.
    expect(panel.formatMetric('relative_l2', 25)).toBe('25.00%');
    expect(panel.formatMetric('max_abs', 0.00123)).toBe('0.00123');
    expect(panel.formatMetric('cosine_similarity', null)).toBe('-');
    expect(panel.formatMetric('max_abs', NaN)).toBe('-');
    panel.selected.set({start: 5, end: 7, layer: 2});
    expect(panel.detailsRows().map((row) => [row.position, row.layer])).toEqual([
      [5, 2],
      [6, 2],
    ]);
    panel.request.loading.set(true);
    expect(panel.detailsRows()).toEqual([]);
    panel.request.loading.set(false);
    panel.request.error.set('Request failed');
    expect(panel.detailsRows()).toEqual([]);
    panel.request.error.set('');
    expect(panel.detailsRows()).toHaveLength(2);
    fixture.destroy();
  });
});

describe('IntermediateGraphArch', () => {
  const baseContext: GraphArchContext = {
    stage: 'decode',
    metricId: 'cosine_similarity',
    metricName: 'Cosine Similarity',
    position: 4,
    legend: true,
    summary: 'max',
    layers: [0.95, 0.85],
    colors: {'0.95': {bgColor: 'rgb(0, 128, 0)', textColor: 'var(--text)'}},
    formatted: {'0.95': '0.95'},
  };

  function dispatchFrameMessage(
    source: unknown,
    data: unknown,
    origin = location.origin,
  ): void {
    const event = Object.assign(new Event('message'), {origin, source, data});
    window.dispatchEvent(event);
  }

  afterEach(() => {
    delete document.documentElement.dataset['theme'];
    vi.restoreAllMocks();
  });

  it('guards sync before view initialization, bridges iframe IPC messages, and cleans up listeners on destroy', async () => {
    TestBed.configureTestingModule({});
    const fixture = TestBed.createComponent(IntermediateGraphArch);
    fixture.componentRef.setInput('context', baseContext);
    const arch = fixture.componentInstance;

    // Calling sync() before view initialization (when viewChild frame is undefined) must not throw.
    expect(() => arch.sync()).not.toThrow();

    fixture.detectChanges();
    const iframe = fixture.nativeElement.querySelector('iframe') as HTMLIFrameElement;
    const postMessage = vi.fn();
    const frameWindow = {postMessage} as unknown as Window;
    Object.defineProperty(iframe, 'contentWindow', {
      value: frameWindow,
      configurable: true,
    });

    const graphRequests: GraphRequest[] = [];
    const pairSaves: GraphPairSave[] = [];
    const pairDeletes: string[][] = [];
    const selections: GraphArchSection[][] = [];
    arch.graphRequest.subscribe((req) => graphRequests.push(req));
    arch.pairSave.subscribe((pair) => pairSaves.push(pair));
    arch.pairDelete.subscribe((ids) => pairDeletes.push(ids));
    arch.selection.subscribe((sec) => selections.push(sec));

    // Ignore messages from untrusted origins, other windows, or other channels.
    dispatchFrameMessage(
      frameWindow,
      {channel: 'intermediate-graph-arch', type: 'graph-request', side: 'ref', signature: 'decode'},
      'https://untrusted.example',
    );
    dispatchFrameMessage(
      {} as Window,
      {channel: 'intermediate-graph-arch', type: 'graph-request', side: 'ref', signature: 'decode'},
    );
    dispatchFrameMessage(frameWindow, {
      channel: 'other-channel',
      type: 'graph-request',
      side: 'ref',
      signature: 'decode',
    });
    expect(graphRequests).toEqual([]);

    // Handle 'ready': resets mapping state and posts current context to the iframe.
    arch.mapping.set({
      active: true,
      editing: true,
      title: 'Old',
      anchor: 'output',
      outputs: [{id: 'o1', label: 'Out'}],
    });
    postMessage.mockClear();
    dispatchFrameMessage(frameWindow, {channel: 'intermediate-graph-arch', type: 'ready'});
    expect(arch.mapping()).toEqual({
      active: false,
      editing: false,
      title: '',
      anchor: '',
      outputs: [],
    });
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        channel: 'intermediate-graph-arch',
        type: 'context',
        stage: 'decode',
        position: 4,
      }),
      location.origin,
    );

    // Handle 'graph-request' and ignore invalid sides.
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'graph-request',
      side: 'invalid',
      signature: 'decode',
    });
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'graph-request',
      side: 'target',
      signature: 'decode',
    });
    expect(graphRequests).toEqual([{side: 'target', signature: 'decode'}]);

    // Handle 'pair-save' and 'pair-delete' with payload validation.
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'pair-save',
      ref: {signature: 'decode', node: '10', output: '0'},
      target: {signature: 'decode', node: 99, output: '0'},
    });
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'pair-save',
      ref: {signature: 'decode', node: '10', output: '0'},
      target: {signature: 'decode', node: '12', output: '0'},
    });
    expect(pairSaves).toEqual([
      {
        ref: {signature: 'decode', node: '10', output: '0'},
        target: {signature: 'decode', node: '12', output: '0'},
      },
    ]);

    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'pair-delete',
      ids: ['p1', 42, 'p2'],
    });
    expect(pairDeletes).toEqual([['p1', 'p2']]);

    // Handle 'mapping-view' and 'selection'.
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'mapping-view',
      active: true,
      editing: false,
      canResetDefaults: true,
      title: 'Layer 0 · Attention',
      anchor: 'attn',
      pairs: [
        {
          referenceId: 'r1',
          targetId: 't1',
          reference: 'q_proj',
          target: 'q_proj',
          referenceName: 'q',
          targetName: 'q',
          selected: true,
        },
      ],
      outputs: [{id: 'out0', label: 'Output 0'}],
    });
    expect(arch.mapping()).toMatchObject({
      active: true,
      editing: false,
      canResetDefaults: true,
      title: 'Layer 0 · Attention',
      anchor: 'attn',
      outputs: [{id: 'out0', label: 'Output 0'}],
    });
    expect(arch.mapping().pairs).toHaveLength(1);

    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'selection',
      sections: [
        {
          label: 'Analysis Info',
          items: [{label: 'Name', value: 'Layer 0', selected: true}],
        },
      ],
    });
    expect(selections).toEqual([
      [
        {
          label: 'Analysis Info',
          tensorItems: undefined,
          items: [
            {
              label: 'Name',
              value: 'Layer 0',
              shape: undefined,
              metricBg: undefined,
              selected: true,
            },
          ],
        },
      ],
    ]);

    // Outbound helper methods forward structured messages to the iframe.
    postMessage.mockClear();
    arch.mappingAction('edit', 'attn');
    arch.deliverGraph({side: 'ref', signature: 'decode', artifact: 'model-a', error: 'none'});
    arch.resetView();
    expect(postMessage.mock.calls.map((call) => call[0].type)).toEqual([
      'mapping-action',
      'execution-graph',
      'reset-view',
    ]);

    // Theme attribute changes re-sync context, and fixture.destroy() disconnects all listeners.
    postMessage.mockClear();
    document.documentElement.dataset['theme'] = 'dark';
    await Promise.resolve();
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({type: 'context', theme: 'dark'}),
      location.origin,
    );

    fixture.destroy();
    postMessage.mockClear();
    dispatchFrameMessage(frameWindow, {
      channel: 'intermediate-graph-arch',
      type: 'graph-request',
      side: 'ref',
      signature: 'prefill',
    });
    document.documentElement.dataset['theme'] = 'light';
    await Promise.resolve();
    expect(graphRequests).toHaveLength(1);
    expect(postMessage).not.toHaveBeenCalled();
  });
});
