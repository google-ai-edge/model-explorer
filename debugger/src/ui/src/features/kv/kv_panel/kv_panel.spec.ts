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

import type {KvHeadEvidence} from '../../../data/contracts/kv';
import {signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {KvPanel} from './kv_panel';
import {KvViewStore} from './kv_view_store';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import {defaultKvView, parseKvView} from './kv_view_state';

const summary = (value: number) => ({
  min: value,
  max: value,
  min_position: 0,
  max_position: 0,
  valid_count: 1,
});
const rows = Array.from({length: 80}, (_, layer) => ({
  layer,
  bins: [
    {
      start: 0,
      end: 200,
      metrics: {
        relative_l2: summary(layer),
        max_abs: summary(layer),
        cosine_distance: summary(layer / 80),
        cosine_similarity: summary(1 - layer / 80),
      },
    },
  ],
}));
const tokenStructure = {
  status: 'ok',
  shape: [2, 8],
  dtype: 'float32',
  head_count: 2,
  channel_count: 8,
  batch_count: 1,
  position_start: 0,
  position_count: 200,
};
const layers = Array.from({length: 80}, (_, layer) => ({
  layer,
  kind: 'key',
  token_structure: {ref: tokenStructure, target: tokenStructure},
  status: 'ok',
  head_count: 2,
  position_start: 0,
  position_count: 200,
  cells: [],
}));

describe('embedded KV analysis', () => {
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
  });
  afterEach(() => vi.unstubAllGlobals());
  async function render(saved = false) {
    const api = {
      kvMetadata: vi.fn(async () => ({
        contexts: [
          {
            id: 'end',
            turn: 1,
            phase: 'decode',
            moment: 'terminal',
            label: 'Generation end',
            layers,
          },
        ],
      })),
      kvRange: vi.fn(async (_capture, _turn, context_id, start, end) => ({
        context_id,
        start,
        end,
        bins: [{start, end}],
        rows,
      })),
      kvSelection: vi.fn(async (_capture, _turn, context_id, start, end) => ({
        context_id,
        start,
        end,
        rows: [],
      })),
      kvHead: vi.fn(async (_capture, turn, context_id, layer, kind, position, head) => ({
        selection: {turn, context_id, layer, kind, position, head},
        channels: [],
      })),
      kvFind: vi.fn(),
    };
    const store = {read: vi.fn(), save: vi.fn()};
    TestBed.configureTestingModule({
      providers: [
        {provide: ReportApiService, useValue: api},
        {
          provide: ReportStateService,
          useValue: {
            captureId: signal('capture'),
            revision: signal(0),
            batchId: signal(99),
            session: signal({batches: [{batch: 99, turn: 2}], turns: [{n: 1}]}),
            kvEntry: signal(null),
          },
        },
        {provide: KvViewStore, useValue: store},
      ],
    });
    const fixture = TestBed.createComponent(KvPanel);
    fixture.componentRef.setInput('embedded', true);
    fixture.componentRef.setInput('stageTurn', 1);
    fixture.componentRef.setInput('stageMoment', 'terminal');
    if (saved)
      fixture.componentRef.setInput('initialView', {
        ...defaultKvView(false),
        contextId: 'end',
        trendLayers: [],
        view: 'trends',
        formula: 'relative_l2 > 1',
        showResults: true,
      });
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    await new Promise((resolve) => setTimeout(resolve, 160));
    fixture.detectChanges();
    return {fixture, panel: fixture.componentInstance, api, store};
  }
  it('locates an exact tensor element across both caches and reveals it outside the current zoom', async () => {
    const {fixture, panel} = await render();
    vi.spyOn(panel, 'tensorEvidence').mockReturnValue({
      evidence: {
        channels: [
          {channel: 3, ref: -2, target: 4},
          {channel: 250, ref: 8, target: 9},
        ],
      },
    } as ReturnType<KvPanel['tensorEvidence']>);
    vi.spyOn(panel, 'tensorChannelFull').mockReturnValue({start: 0, end: 4096});
    panel.tensorChannelCapacity.set(128);
    panel.tensorChannelZoom.set({start: 0, end: 32});
    panel.tensorChannelSelection.set({start: 0, end: 16, plane: 0});
    const locate = vi.spyOn(panel, 'locateTensor');
    panel.locateTensorElement('key', 250);
    expect(panel.tensorChannels()).toEqual({key: 250, value: 250});
    expect(panel.tensorSelectedRange()).toEqual({start: 250, end: 251});
    expect(panel.tensorChannelRange()).toEqual({start: 234, end: 266});
    expect(locate).toHaveBeenCalledWith('key');
    panel.tensorChannelZoom.set(null);
    panel.locateTensorElement('value', 250);
    expect(panel.tensorChannelBinSize()).toBe(1);
    panel.tensorPlot.set('scatter');
    panel.tensorValueZoom.set({start: -1, end: 1});
    panel.locateTensorElement('key', 3);
    expect(panel.tensorValueZoom()).toBeNull();
    expect(panel.tensorChannels()).toEqual({key: 3, value: 3});
    panel.tensorValueZoom.set({start: -5, end: 5});
    panel.locateTensorElement('value', 3);
    expect(panel.tensorValueZoom()).toEqual({start: -5, end: 5});
    panel.locateTensorElement('key', 999);
    expect(panel.tensorChannels()).toEqual({key: 3, value: 3});
    fixture.destroy();
  });
  it('follows an external token selection during inspection without changing its tensor index or inspector ownership', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', {start: 8, end: 9, layer: null});
    panel.chooseVisualization('Tensor Inspection · Line chart');
    panel.setTensorLayer('3');
    panel.setTensorHead('1');
    fixture.detectChanges();
    const inspect = vi.fn();
    panel.inspect.subscribe(inspect);
    fixture.componentRef.setInput('chartSelection', {start: 12, end: 13, layer: null});
    fixture.detectChanges();
    expect(panel.tensorPosition()).toBe(12);
    expect(panel.tensorLayer()).toBe(3);
    expect(panel.tensorHead()).toBe(1);
    expect(inspect).not.toHaveBeenCalled();
    fixture.componentRef.setInput('chartSelection', {start: 20, end: 30, layer: null});
    fixture.detectChanges();
    expect(panel.tensorPosition()).toBe(12);
    fixture.componentRef.setInput('chartSelection', null);
    fixture.detectChanges();
    expect(panel.tensorPosition()).toBe(12);
    fixture.destroy();
  });
  it('switches range controls with the chart and keeps element and value zoom independent', async () => {
    const {fixture, panel} = await render();
    const rangeControl = () => fixture.nativeElement.querySelector('.toolbar-range') as HTMLElement;
    for (const view of ['Tensor Comparison · Heatmap', 'Tensor Comparison · Line chart']) {
      panel.chooseVisualization(view);
      fixture.detectChanges();
      expect(rangeControl().textContent).toContain('Token Range');
      expect(rangeControl().querySelector('analysis-value-range')).toBeNull();
    }
    panel.chooseVisualization('Tensor Inspection · Line chart');
    fixture.detectChanges();
    panel.tensorChannelZoom.set({start: 2, end: 6});
    fixture.detectChanges();
    const location = [panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()];
    expect(rangeControl().textContent).toContain('Element Range');
    expect(rangeControl().querySelector('analysis-value-range')).toBeNull();
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    panel.tensorValueZoom.set({start: -2, end: 2});
    fixture.detectChanges();
    expect(rangeControl().textContent).toContain('Value Range');
    expect(rangeControl().querySelector('analysis-position-range')).toBeNull();
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual(location);
    panel.chooseVisualization('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(rangeControl().textContent).toContain('Element Range');
    expect(panel.tensorChannelZoom()).toEqual({start: 2, end: 6});
    expect(panel.tensorValueZoom()).toEqual({start: -2, end: 2});
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual(location);
  });
  it('keeps selection fields and K/V rows stable for raw values, bins and missing sides', async () => {
    const {fixture, panel} = await render();
    expect(panel.tensorSelectionProperties().map((p) => p.value)).toEqual([
      '-',
      'Head dimension',
      '-',
      '-',
    ]);
    expect(panel.tensorSelectionValues().map((r) => r.values)).toEqual([
      [null, null, null, null],
      [null, null, null, null],
    ]);
    panel.tensorHead.set(0);
    const channels = [
      {channel: 0, ref: 3, target: 1, delta: -2, abs_delta: 2, status: 'ok'},
      {channel: 1, ref: 1, target: 4, delta: 3, abs_delta: 3, status: 'ok'},
    ];
    for (const request of [panel.tensorKey, panel.tensorValue]) {
      request.data.set([
        {head: 0, label: 'Head 0', error: '', evidence: {channels} as KvHeadEvidence},
      ]);
    }
    panel.selectTensorChannel('value', 0);
    expect(panel.tensorChannels()).toEqual({key: 0, value: 0});
    expect(panel.tensorSelectionValues().map((r) => r.values)).toEqual([
      [3, 1, -2, 2],
      [3, 1, -2, 2],
    ]);
    panel.tensorChannelSelection.set({start: 0, end: 2, plane: null});
    panel.tensorChannelReduction.set('min');
    expect(panel.tensorSelectionProperties().map((p) => p.value)).toEqual([
      '0',
      'Head dimension',
      '0–1',
      'Minimum',
    ]);
    expect(panel.tensorSelectionValues().map((r) => r.values)).toEqual([
      [1, 1, null, null],
      [1, 1, null, null],
    ]);
    panel.tensorChannelReduction.set('mean');
    expect(panel.tensorSelectionValues()[0].values).toEqual([2, 2.5, null, null]);
    panel.tensorKey.data.set([
      {
        head: 0,
        label: 'Head 0',
        error: '',
        evidence: {channels: [{...channels[0], target: null}]} as KvHeadEvidence,
      },
    ]);
    panel.selectTensorChannel('key', 0);
    expect(panel.tensorSelectionValues()[0].values).toEqual([3, null, null, null]);
    panel.tensorKey.loading.set(true);
    expect(panel.tensorSelectionValues()[0].values).toEqual([null, null, null, null]);
    fixture.destroy();
  });
  it('uses the stage identity, shows all layers in both chart views, and keeps an explicitly empty selection', async () => {
    const {fixture, panel, api, store} = await render();
    expect(api.kvMetadata.mock.calls[0].slice(0, 2)).toEqual(['capture', 1]);
    expect(panel.layerIds()).toHaveLength(80);
    const details = panel.detailsTemplate()!.createEmbeddedView({});
    details.detectChanges();
    const text = details.rootNodes.map((n) => n.textContent).join(' ');
    expect(text).toContain('Key cache');
    expect(text).toContain('Value cache');
    expect(text).not.toContain('Add bookmark');
    details.destroy();
    panel.chooseView('trends');
    fixture.detectChanges();
    expect(panel.metric()).toBe('cosine_similarity');
    expect(panel.head()).toBe('min');
    expect(panel.bothTensors()).toBe(true);
    expect(panel.selectedTrendLayers()).toEqual(panel.layerIds());
    expect(panel.chartSeries()).toHaveLength(160);
    panel.trendLayers.set([]);
    panel.chooseView('heatmap');
    panel.chooseView('trends');
    expect(panel.chartSeries()).toEqual([]);
    expect(panel.layerIds()).toHaveLength(80);
    fixture.destroy();
    expect(store.read).not.toHaveBeenCalled();
    expect(store.save).not.toHaveBeenCalled();
  });
  it('retains explicit Positions reduction through zoom and records it for bookmarks', async () => {
    const {fixture, panel} = await render();
    expect(panel.heatmapDimensions().map((control) => control.id)).toEqual(['position', 'head']);
    panel.setHeatmapDimension({id: 'position', value: 'mean'});
    panel.setHeatmapMetric('relative_l2');
    expect(panel.positionReduction()).toBe('mean');
    expect(panel.captureView().positionReduction).toBe('mean');
    panel.rangeData.set({
      context_id: 'end',
      start: 0,
      end: 1,
      bins: [{start: 0, end: 1}],
      rows: [],
    });
    fixture.detectChanges();
    expect(panel.heatmapDimensions()[0].disabled).toBe(true);
    fixture.nativeElement.querySelector('button[aria-label="Display"]').click();
    fixture.detectChanges();
    expect(
      (document.querySelector('button[aria-label="Bins"]') as HTMLButtonElement).disabled,
    ).toBe(true);
    panel.closePopup();
    panel.setHeatmapDimension({id: 'position', value: 'max'});
    expect(panel.positionReduction()).toBe('mean');
    fixture.destroy();
  });
  it('stages Display changes until Apply, discards on close, and restores series defaults', async () => {
    const {fixture, panel} = await render();
    panel.chooseView('trends');
    const original = panel.selectedTrendLayers();
    panel.toggleChartDisplay();
    panel.draftDisplayVisible([]);
    expect(panel.displayDirty()).toBe(true);
    expect(panel.selectedTrendLayers()).toEqual(original);
    panel.closePopup();
    panel.toggleChartDisplay();
    expect(panel.displayDraft().visible).toEqual(original.map(String));
    panel.draftDisplayVisible([]);
    panel.applyChartDisplay();
    expect(panel.selectedTrendLayers()).toEqual([]);
    expect(panel.displayDirty()).toBe(false);
    panel.draftDisplayVisible(panel.displayDefaultVisible());
    panel.applyChartDisplay();
    expect(panel.selectedTrendLayers()).toEqual(panel.layerIds());
    panel.closePopup();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    panel.toggleChartDisplay();
    panel.toggleDisplaySeries('ref');
    expect(panel.tensorVisibleRuntimes()).toEqual(['ref', 'target']);
    panel.applyChartDisplay();
    expect(panel.tensorVisibleRuntimes()).toEqual(['target']);
    expect(panel.displayDirty()).toBe(false);
    panel.draftDisplayVisible(panel.displayDefaultVisible());
    panel.applyChartDisplay();
    expect(panel.tensorVisibleRuntimes()).toEqual(['ref', 'target']);
    fixture.destroy();
  });
  it('filters Heatmap layers, retains their selection across views, and saves empty selections', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', null);
    const inspect = vi.fn();
    panel.inspect.subscribe(inspect);
    panel.chooseView('heatmap');
    expect(panel.visibleMetricLayers()).toEqual(panel.layerIds());
    panel.toggleChartDisplay();
    panel.draftDisplayVisible(['1', '4']);
    expect(panel.visibleMetricLayers()).toHaveLength(80);
    panel.applyChartDisplay();
    expect(panel.visibleMetricLayers()).toEqual([1, 4]);
    expect(inspect).not.toHaveBeenCalled();
    expect(parseKvView(panel.captureView())?.heatmapLayers).toEqual([1, 4]);
    panel.chooseView('trends');
    expect(panel.visibleMetricLayers()).toEqual(panel.selectedTrendLayers());
    panel.chooseView('heatmap');
    expect(panel.visibleMetricLayers()).toEqual([1, 4]);
    panel.draftDisplayVisible([]);
    panel.applyChartDisplay();
    fixture.detectChanges();
    expect(panel.visibleMetricLayers()).toEqual([]);
    expect(parseKvView(panel.captureView())?.heatmapLayers).toEqual([]);
    expect(fixture.nativeElement.textContent).toContain('No layers selected');
    expect(parseKvView({...panel.captureView(), heatmapLayers: [-1]})).toBeNull();
    expect(parseKvView({...panel.captureView(), heatmapLayers: [1, 1]})).toBeNull();
    panel.draftDisplayVisible(panel.displayDefaultVisible());
    panel.applyChartDisplay();
    expect(panel.visibleMetricLayers()).toEqual(panel.layerIds());
    fixture.destroy();
  });
  it('restores explicit empty Line chart from a bookmark without repopulating layers', async () => {
    const {fixture, panel} = await render(true);
    expect(panel.view()).toBe('trends');
    expect(panel.selectedTrendLayers()).toEqual([]);
    expect(panel.captureView().trendLayers).toEqual([]);
    expect(panel.formula()).toBe('');
    expect(panel.showResults()).toBe(false);
    expect(fixture.nativeElement.querySelector('[aria-label="Find by metric"]')).toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Choose KV observation"]')).toBeNull();
    fixture.destroy();
  });
  it('uses inclusive displayed endpoints and clamps them to captured Positions', async () => {
    const {fixture, panel} = await render();
    expect(fixture.nativeElement.querySelector('input[type="range"]')).toBeNull();
    panel.setRangeEdge('start', 180);
    panel.setRangeEdge('end', 185);
    expect(panel.visibleRange()).toEqual({start: 180, end: 186});
    panel.setRangeEdge('start', 999);
    expect(panel.visibleRange()).toEqual({start: 185, end: 186});
    panel.zoomTo({start: 20, end: 180});
    expect(panel.visibleRange()).toEqual({start: 20, end: 180});
    panel.zoomTo({start: 0, end: 200});
    expect(panel.visibleRange()).toEqual({start: 0, end: 200});
    fixture.destroy();
  });
  it('persists runtime visibility without changing tensor selection or metrics', async () => {
    const {fixture, panel} = await render();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    const coordinates = [panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()];
    panel.toggleTensorRuntime('ref');
    expect(panel.captureView().tensor?.visibleRuntimes).toEqual(['target']);
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    panel.chooseVisualization('Tensor Comparison · Heatmap');
    panel.chooseVisualization('Tensor Inspection · Line chart');
    expect(panel.tensorVisibleRuntimes()).toEqual(['target']);
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual(coordinates);
    expect(panel.head()).toBe('min');
    panel.toggleTensorRuntime('target');
    expect(panel.captureView().tensor?.visibleRuntimes).toEqual([]);
    expect(panel.tensorSelectedRange()).toBeNull();
    fixture.destroy();
  });
  it('defaults an unselected Tensor view inside the panel range and preserves it on return', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', null);
    panel.zoomTo({start: 20, end: 40});
    panel.aggregationHeads.set([1]);
    fixture.detectChanges();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual([20, 0, 1]);
    expect(panel.tensorSelectedRange()).toBeNull();
    panel.setTensorPosition('27');
    panel.setTensorLayer('3');
    panel.chooseView('heatmap');
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual([27, 3, 1]);
    expect(panel.tensorSelectedRange()).toBeNull();
    fixture.destroy();
  });
  it('uses the bin representative in fullscreen and restores its containing metric bin', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('expanded', true);
    fixture.componentRef.setInput('chartSelection', {start: 4, end: 12, layer: 3});
    panel.positionReductionOverride.set('mean');
    fixture.detectChanges();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual([4, 3, 0]);
    expect(panel.tensorSelectedRange()).toBeNull();
    panel.chooseView('heatmap');
    expect(panel.inspectedRange()).toEqual({start: 0, end: 200, layer: 3});
    fixture.destroy();
  });
  it('initializes a directly opened Tensor view after metadata is ready', async () => {
    const {fixture, panel} = await render();
    panel.tensorView.set(true);
    fixture.detectChanges();
    expect([panel.tensorPosition(), panel.tensorLayer(), panel.tensorHead()]).toEqual([0, 0, 0]);
    expect(panel.tensorSelectedRange()).toBeNull();
    fixture.destroy();
  });
  it('fills missing coordinates without selecting a Head dimension or using a bin as an exact position', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', {start: 4, end: 12, layer: 3});
    fixture.detectChanges();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    expect(panel.tensorPosition()).toBe(0);
    expect(panel.tensorLayer()).toBe(3);
    expect(panel.tensorHead()).toBe(0);
    expect(panel.tensorReady()).toBe(true);
    expect(panel.tensorSelectedRange()).toBeNull();
    panel.chooseView('heatmap');
    fixture.componentRef.setInput('chartSelection', {start: 8, end: 9, layer: null});
    fixture.detectChanges();
    panel.chooseVisualization('Tensor Inspection · Line chart');
    expect(panel.tensorPosition()).toBe(8);
    expect(panel.tensorLayer()).toBe(3);
    fixture.destroy();
  });
  it('activates this tensor inspector when a Head dimension is clicked', async () => {
    const {fixture, panel} = await render();
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    const inspected: KvPanel[] = [];
    panel.inspect.subscribe((value) => inspected.push(value));
    panel.selectTensorChannel('value', 3);
    expect(panel.tensorChannels()).toEqual({key: 3, value: 3});
    expect(panel.tensorSelectedRange()).toEqual({start: 3, end: 4});
    expect(inspected).toHaveLength(1);
    expect(inspected[0]).toBe(panel);
    fixture.destroy();
  });
  it('shares tensor coordinates with the chart without rewriting aggregation or zoom', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', {start: 8, end: 9, layer: 3});
    fixture.detectChanges();
    panel.zoomTo({start: 5, end: 40});
    panel.chooseVisualization('Tensor Inspection · Line chart');
    panel.setTensorPosition('12');
    panel.setTensorLayer('7');
    expect(panel.inspectedRange()).toEqual({start: 12, end: 13, layer: 7});
    expect(panel.visibleRange()).toEqual({start: 5, end: 40});
    expect(panel.head()).toBe('min');
    expect(panel.tensorHead()).toBe(0);
    panel.setTensorPosition('999');
    expect(panel.tensorPosition()).toBe(12);
    const snapshot = panel.captureView();
    expect(snapshot.tensor).toEqual({
      active: true,
      plot: 'index',
      position: 12,
      layer: 7,
      head: 0,
      channelRange: null,
      valueRange: null,
      channelReduction: 'mean',
      visibleRuntimes: ['ref', 'target'],
    });
    panel.chooseView('trends');
    expect(panel.tensorView()).toBe(false);
    expect(panel.head()).toBe('min');
    expect(panel.inspectedRange()).toEqual({start: 12, end: 13, layer: 7});
    fixture.destroy();
  });
  it('selects a data/chart pair without changing tensor coordinates or chart aggregation', async () => {
    const {fixture, panel} = await render();
    fixture.componentRef.setInput('chartSelection', {start: 8, end: 9, layer: 3});
    fixture.detectChanges();
    expect(panel.visualizationLabel()).toBe('Tensor Comparison · Heatmap');
    expect(panel.visualizationGroups.map((group) => group.options.length)).toEqual([2, 2]);
    panel.chooseVisualization('Tensor Comparison · Line chart');
    panel.chooseVisualization('Tensor Inspection · Line chart');
    panel.setTensorHead('1');
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    expect(panel.tensorPosition()).toBe(8);
    expect(panel.tensorLayer()).toBe(3);
    expect(panel.tensorHead()).toBe(1);
    panel.chooseVisualization('Tensor Inspection · Heatmap');
    expect(panel.visualizationLabel()).toBe('Tensor Inspection · Scatter plot');
    panel.chooseVisualization('Tensor Comparison · Line chart');
    expect(panel.tensorView()).toBe(false);
    expect(panel.view()).toBe('trends');
    expect(panel.head()).toBe('min');
    panel.chooseVisualization('Tensor Inspection · Scatter plot');
    expect(panel.tensorHead()).toBe(1);
    panel.chooseVisualization('Tensor Comparison · Heatmap');
    expect(panel.visualizationLabel()).toBe('Tensor Comparison · Heatmap');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelectorAll('button[aria-label="Visualization"]'),
    ).toHaveLength(1);
    fixture.destroy();
  });
});
