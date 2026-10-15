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
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import type {PositionAnalysis} from '../../data/contracts/position_analysis';
import type {PlotHost, PlotlyApi} from '../../shared/plotly_loader';
import {HeatmapControls} from '../analysis/heatmap_controls';
import type {StageAnalysisPanel} from './stage_analysis';
import {TENSOR_METRICS} from '../analysis/position_tensor_model';

const data: PositionAnalysis = {
  provenance: {kind: 'synthetic', label: 'Test data'},
  rows: Array.from({length: 6}, (_, layer) =>
    Array.from({length: 320}, (_, position) => ({
      position,
      layer,
      anchor: 'output',
      status: 'ok',
      shape: [8],
      metrics: {
        CosSim: {value: 1 - layer / 10, status: 'ok'},
        'Relative L2': {value: layer / 100, status: 'ok'},
        'Max abs error': {value: layer / 20, status: 'ok'},
      },
    })),
  ).flat(),
};

describe('Activation comparison toolbar', () => {
  let GraphMetricPanel: typeof import('./graph_metric_panel').GraphMetricPanel;
  let layerColor: typeof import('../analysis/layer_legend').layerColor;
  beforeAll(async () => {
    // Initialize the shared metric module before the builder's lazy component chunk.
    // Static cross-chunk imports otherwise capture its not-yet-initialized export in Vitest.
    ({GraphMetricPanel} = await import('./graph_metric_panel'));
    ({layerColor} = await import('../analysis/layer_legend'));
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
        {provide: ReportApiService, useValue: {}},
        {provide: ReportStateService, useValue: {captureId: signal(''), revision: signal(0)}},
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
    const {panel} = create('intermediate');
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
    expect(panel.archContext().simulatedMetrics).toHaveLength(101);
    const saved = panel.captureView();
    expect(saved.graphDisplay).not.toHaveProperty('simulation');
    const restored = create('intermediate', {
      ...saved,
      graphDisplay: {...saved.graphDisplay!, tensorMode: 'name', simulation: false},
    }).panel;
    expect(restored.captureView().graphDisplay).not.toHaveProperty('tensorMode');
    expect(restored.captureView().graphDisplay).not.toHaveProperty('simulation');
    expect(restored.archContext().simulatedMetrics).toHaveLength(101);
    expect(restored.graphDisplay()).toEqual(panel.graphDisplay());
  });

  function create(
    category: 'activations' | 'intermediate' = 'activations',
    initial?: StageAnalysisPanel['graph'],
  ) {
    const fixture = TestBed.createComponent(GraphMetricPanel);
    expect(TENSOR_METRICS).toBeDefined();
    expect(fixture.componentInstance.metrics).toBe(TENSOR_METRICS);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('stage', 'decode');
    fixture.componentRef.setInput('category', category);
    if (initial) fixture.componentRef.setInput('initial', initial);
    fixture.detectChanges();
    const panel = fixture.componentInstance;
    panel.request.data.set(
      category === 'activations'
        ? data
        : {
            ...data,
            rows: data.rows.map((row) => ({...row, anchor: 'attn'})),
          },
    );
    fixture.detectChanges();
    return {fixture, panel};
  }

  it('shares comparison state with Graph Diff and restores the saved view', () => {
    const {fixture, panel} = create();
    panel.viewport.set({start: 10, end: 20});
    panel.chooseView('Tensor Comparison · Graph Diff');
    fixture.detectChanges();
    expect(panel.view()).toBe('graph');
    expect(panel.range()).toEqual({start: 10, end: 20});
    expect(fixture.nativeElement.querySelector('intermediate-graph-arch')).not.toBeNull();
    expect(fixture.nativeElement.querySelectorAll('.activation-controls').length).toBe(1);
    expect(panel.archContext().layers[0]).toBe(1);
    expect(panel.archContext().layers[6]).toBeNull();
    expect(panel.captureView().view).toBe('graph');
    panel.chooseView('Tensor Comparison · Heatmap');
    fixture.detectChanges();
    expect(panel.range()).toEqual({start: 10, end: 20});
    expect(fixture.nativeElement.querySelector('intermediate-graph-arch')).toBeNull();
    fixture.destroy();
    const restored = create('activations', {
      batchId: null,
      metric: 'cosine_similarity',
      anchor: 'min',
      view: 'graph',
    });
    expect(restored.panel.view()).toBe('graph');
  });

  it('keeps module summaries separate from tensor aggregation and restores the choice', () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Graph Diff');
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
    const restored = create('activations', saved);
    expect(restored.panel.archContext().summary).toBe('max');
  });

  it('migrates legacy Output summary to Max and keeps Min/Max available without data', () => {
    const {fixture, panel} = create('intermediate', {
      batchId: null,
      metric: 'cosine_similarity',
      anchor: 'output',
      view: 'graph',
      moduleSummary: 'output',
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

  it('uses a single token and fixed layer outputs without inheriting hidden aggregation', () => {
    const {fixture, panel} = create();
    panel.request.data.set({
      ...data,
      rows: [
        {...data.rows[0], position: 10, metrics: {CosSim: {value: 0.8, status: 'ok'}}},
        {...data.rows[0], position: 11, metrics: {CosSim: {value: 0.4, status: 'ok'}}},
        {
          ...data.rows[0],
          position: 12,
          anchor: 'attn',
          metrics: {CosSim: {value: 0.1, status: 'ok'}},
        },
      ],
    });
    panel.viewport.set({start: 10, end: 13});
    panel.anchorScope.set('internal');
    panel.anchor.set('max');
    panel.chooseView('Tensor Comparison · Graph Diff');
    fixture.detectChanges();
    expect(panel.graphPosition()).toBe(10);
    expect(panel.archContext().layers[0]).toBe(0.8);
    expect(fixture.nativeElement.querySelector('analysis-position-range')).toBeNull();
    expect(fixture.nativeElement.querySelector('analysis-aggregation')).toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]').hidden).toBe(false);
    expect(
      fixture.nativeElement.querySelector('[aria-label="Previous token position"]').disabled,
    ).toBe(true);
    fixture.nativeElement.querySelector('[aria-label="Next token position"]').click();
    fixture.detectChanges();
    expect(panel.graphPosition()).toBe(11);
    expect(panel.archContext().layers[0]).toBe(0.4);
    fixture.componentRef.setInput('chartSelection', {start: 12, end: 13, layer: null});
    fixture.detectChanges();
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
    expect(fixture.nativeElement.querySelector('analysis-aggregation')).not.toBeNull();
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
    expect(panel.view()).toBe('trends');
    expect(panel.range()).toEqual({start: 32, end: 96});
    expect(panel.selected()).toEqual(selected);
    expect(panel.series().map((series) => series.color)).toEqual(
      panel.selectedLayers().map(layerColor),
    );
    expect(fixture.nativeElement.querySelector('analysis-layer-legend')).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[aria-label="Layer legend"]').textContent,
    ).toContain('Layers 6/6');
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    picker.valueChange.emit('Tensor Comparison · Heatmap');
    expect(panel.view()).toBe('heatmap');
    expect(panel.selected()).toEqual(selected);
    fixture.destroy();
  });

  it('upgrades legacy Intermediate panels to the shared views and retains their internal scope', () => {
    const {fixture, panel} = create('intermediate');
    expect(fixture.nativeElement.querySelector('.view-switch')).toBeNull();
    expect(fixture.nativeElement.querySelector('.view-picker')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]')).not.toBeNull();
    expect(panel.label()).toBe('Intermediate Tensors');
    expect(panel.selectedAnchors()).toEqual(['attn']);
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.anchor()).toBe('attn');
    panel.showComparison('heatmap');
    fixture.detectChanges();
    const controls = fixture.debugElement.query(By.directive(HeatmapControls))
      .componentInstance as HeatmapControls;
    expect(controls.showRangeControls()).toBe(false);
    expect(controls.aggregation()?.dimension).toBe('syncPoint');
    fixture.destroy();
  });

  const mixedRows = () =>
    data.rows.flatMap((row) => [
      row,
      {
        ...row,
        anchor: 'attention_output',
        metrics: {...row.metrics, CosSim: {value: 0.2, status: 'ok'}},
      },
      {
        ...row,
        anchor: 'pre_attention_norm',
        metrics: {...row.metrics, CosSim: {value: 0.9, status: 'ok'}},
      },
    ]);

  it('defaults to output, exposes actual anchors, and persists selection through a loading state', () => {
    const {fixture, panel} = create();
    panel.request.data.set({...data, rows: mixedRows()});
    fixture.detectChanges();
    expect(panel.anchors()).toEqual(['output', 'attention_output', 'pre_attention_norm']);
    expect(panel.selectedAnchors()).toEqual(['output']);
    expect(panel.metricSource().every((row) => row.anchor === 'output')).toBe(true);
    expect(
      fixture.nativeElement.querySelector('[aria-label="Sync points aggregation"]').textContent,
    ).toContain('Layer output');
    panel.setAggregation({selection: ['attention_output'], function: 'max'});
    panel.trendLayers.set([1, 4]);
    const saved = panel.captureView();
    panel.request.data.set(null);
    panel.request.loading.set(true);
    expect(panel.captureView().anchorSelection).toEqual(['attention_output']);
    expect(panel.captureView().trendLayers).toEqual([1, 4]);
    fixture.destroy();
    const restored = create('activations', saved);
    restored.panel.request.data.set({...data, rows: mixedRows()});
    restored.fixture.detectChanges();
    expect(restored.panel.selectedAnchors()).toEqual(['attention_output']);
    restored.fixture.destroy();
  });

  it('preserves old internal-only aggregations when output becomes available', () => {
    const {fixture, panel} = create('intermediate', {
      batchId: null,
      metric: 'cosine_similarity',
      anchor: 'min',
      anchorSelection: null,
    });
    panel.request.data.set({...data, rows: mixedRows()});
    fixture.detectChanges();
    expect(panel.selectedAnchors()).toEqual(['attention_output', 'pre_attention_norm']);
    const saved = panel.captureView();
    expect(saved.anchorScope).toBe('internal');
    fixture.destroy();
    const restored = create('intermediate', saved);
    restored.panel.request.data.set({...data, rows: mixedRows()});
    restored.fixture.detectChanges();
    expect(restored.panel.selectedAnchors()).toEqual(['attention_output', 'pre_attention_norm']);
    restored.fixture.destroy();
  });

  it('resolves extrema to a real anchor and syncs a changed inspection point back into comparison', () => {
    const {fixture, panel} = create();
    fixture.componentRef.setInput('expanded', true);
    panel.request.data.set({...data, rows: mixedRows()});
    panel.select({start: 44, end: 45, layer: 2});
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'min'});
    expect(panel.layerGraphContext()).toMatchObject({
      layer: 2,
      position: 44,
      anchor: 'attention_output',
    });
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'mean'});
    expect(panel.layerGraphContext()).toMatchObject({layer: 2, anchor: undefined});
    panel.setAggregation({selection: ['attention_output', 'pre_attention_norm'], function: 'min'});
    panel.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(panel.inspection()?.anchor()).toBe('attention_output');
    expect(panel.layerGraphContext()).toEqual({layer: 2, position: 44, anchor: 'attention_output'});
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
    panel.showComparison('trends');
    fixture.detectChanges();
    expect(panel.inspectionMode()).toBe(false);
    expect(panel.range()).toEqual(comparison.range);
    expect(panel.selected()).toEqual({start: 45, end: 46, layer: 2});
    expect(panel.selectedLayers()).toEqual([0, 1, 2, 3, 4, 5]);
    const saved = panel.captureView();
    fixture.destroy();
    const restored = create('activations', {...saved, mode: 'inspection'});
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
    (expanded) => {
      const {fixture, panel} = create();
      fixture.componentRef.setInput('expanded', expanded);
      panel.request.data.set({
        ...data,
        rows: data.rows.map((row) => ({
          ...row,
          metrics: {
            ...row.metrics,
            CosSim: {value: row.position % 4 === 2 ? 0.1 : 0.9, status: 'ok'},
          },
        })),
      });
      panel.viewport.set({start: 40, end: 48});
      panel.capacity.set(2);
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

  it('discards unapplied bins and layers on close, and only commits on Apply', () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Line chart');
    fixture.detectChanges();
    panel.toggleDisplay();
    const before = panel.captureView();
    panel.draftReduction('Mean');
    panel.draftLayers([0, 2]);
    expect(panel.displayDirty()).toBe(true);
    expect(panel.captureView()).toEqual(before);
    panel.closeDisplay();
    panel.toggleDisplay();
    expect(panel.displayDraft()).toEqual({bins: null, layers: before.trendLayers});
    expect(panel.displayDirty()).toBe(false);
    panel.draftReduction('Mean');
    panel.draftLayers([0, 2]);
    panel.applyDisplay();
    expect(panel.captureView()).toEqual({
      ...before,
      positionReduction: 'mean',
      trendLayers: [0, 2],
    });
    expect(panel.displayDirty()).toBe(false);
    fixture.destroy();
  });

  it('restores metric-dependent defaults only after Apply without changing scope or selection', () => {
    const {fixture, panel} = create();
    panel.viewport.set({start: 32, end: 280});
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

  it('disables bins for one position while still applying layer visibility and preserving the selected cell', () => {
    const {fixture, panel} = create();
    panel.chooseView('Tensor Comparison · Line chart');
    fixture.detectChanges();
    panel.viewport.set({start: 45, end: 46});
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

  it('restores the saved comparison state and keeps it when switching visualizations', () => {
    const initial: NonNullable<StageAnalysisPanel['graph']> = {
      batchId: null,
      metric: 'relative_l2',
      anchor: 'min',
      anchorSelection: null,
      view: 'trends',
      range: {start: 24, end: 260},
      positionReduction: 'mean',
      layerChoice: 'All layers',
      trendLayers: [0, 3],
      selection: {start: 30, end: 32, layer: 3},
    };
    const {fixture, panel} = create('activations', initial);
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
    expect(panel.metrics.map((metric) => metric.source)).toEqual([
      'CosSim',
      'Relative L2',
      'Max abs error',
    ]);
    expect(panel.formatMetric('relative_l2', 0.25)).toBe('25.00%');
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

  it('supports layer indices beyond 35 in archContext and handles 200k rows without RangeError', () => {
    const {fixture, panel} = create();
    const largeRows = Array.from({length: 200_000}, (_, index) => ({
      position: index,
      layer: index === 199_999 ? 39 : 0,
      anchor: 'output',
      status: 'ok' as const,
      shape: [8],
      metrics: {
        CosSim: {value: 0.875, status: 'ok' as const},
        'Relative L2': {value: 0.01, status: 'ok' as const},
        'Max abs error': {value: 0.05, status: 'ok' as const},
      },
    }));
    panel.request.data.set({
      provenance: {kind: 'synthetic', label: 'Large test data'},
      rows: largeRows,
    });
    expect(panel.fullRange()).toEqual({start: 0, end: 200_000});
    panel.setGraphPosition(199_999, false);
    expect(panel.archContext().layers[39]).toBe(0.875);
    fixture.destroy();
  });
});
