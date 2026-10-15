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
import type {
  PositionTensorRequest,
  PositionTensorResponse,
} from '../../data/contracts/position_analysis';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import type {PlotHost, PlotlyApi} from '../../shared/plotly_loader';
import {KvChannelChart} from '../kv/kv_panel/kv_channel_chart';
import {KvScatterChart} from '../kv/kv_panel/kv_scatter_chart';
import {TensorIndexControl} from './tensor_index_control';
import {
  LayerActivationInspection,
  type LayerActivationInspectionState,
} from './layer_activation_inspection';
const vector = (query: PositionTensorRequest, total = 320): PositionTensorResponse => ({
  position: query.position,
  layer: query.layer,
  anchor: query.anchor,
  reference_position: query.position - 1,
  target_position: query.position,
  reference: {
    values: Array.from({length: total}, (_, i) => i / 10),
    shape: [total],
    dtype: 'float16',
  },
  target: {values: Array.from({length: total}, (_, i) => i / 10 + 1), shape: [total], dtype: null},
  metrics: {},
  status: 'ok',
  provenance: {kind: 'synthetic', label: 'Test data'},
});
type Read = (
  capture: string | null,
  turn: number,
  phase: string,
  query: PositionTensorRequest,
  signal: AbortSignal,
) => Promise<PositionTensorResponse>;
describe('LayerActivationInspection', () => {
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
    vi.spyOn(document, 'createRange').mockImplementation(
      () =>
        ({
          getBoundingClientRect: () => ({width: 0}),
          selectNodeContents() {},
          detach() {},
        }) as unknown as Range,
    );
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
  function create(read?: Read, initial: LayerActivationInspectionState | null = null) {
    const api = {
      positionTensor: vi.fn(read ?? (async (_capture, _turn, _phase, query) => vector(query))),
    };
    const captureId = signal<string | null>('capture'),
      revision = signal(0);
    TestBed.configureTestingModule({
      providers: [
        {provide: ReportApiService, useValue: api},
        {provide: ReportStateService, useValue: {captureId, revision}},
      ],
    });
    const fixture = TestBed.createComponent(LayerActivationInspection);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('stage', 'decode');
    fixture.componentRef.setInput('fullRange', {start: 32, end: 67});
    fixture.componentRef.setInput('layers', [0, 2, 7]);
    fixture.componentRef.setInput('anchors', ['output', 'pre_attention_norm', 'attention_output']);
    fixture.componentRef.setInput('initial', initial);
    fixture.componentRef.setInput('alignment', {
      start: 32,
      pairs: [
        [31, 32],
        [null, 33],
      ],
    });
    fixture.detectChanges();
    return {fixture, view: fixture.componentInstance, api};
  }
  async function settle(fixture: ComponentFixture<LayerActivationInspection>) {
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    TestBed.tick();
  }
  it('requests the exact aligned tensor, renders complete vectors, and switches inspection plots', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    expect(api.positionTensor).toHaveBeenCalledWith(
      'capture',
      2,
      'decode',
      {
        position: 32,
        layer: 0,
        anchor: 'output',
        alignment: {
          start: 32,
          pairs: [
            [31, 32],
            [null, 33],
          ],
        },
      },
      expect.any(AbortSignal),
    );
    expect(view.channels()).toHaveLength(320);
    expect(view.channels()[319].ref).toBe(31.9);
    expect(view.data()?.target?.dtype).toBeNull();
    expect(view.notices()).toEqual([]);
    expect(fixture.nativeElement.textContent).not.toContain('Test data');
    expect(fixture.debugElement.query(By.directive(KvChannelChart))).not.toBeNull();
    const compare = vi.fn();
    view.compare.subscribe(compare);
    view.chooseView('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    expect(fixture.debugElement.query(By.directive(KvScatterChart))).not.toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]').disabled).toBe(true);
    view.chooseView('Tensor Comparison · Heatmap');
    expect(compare).toHaveBeenCalledWith('heatmap');
    fixture.destroy();
  });
  it('restores valid view state, resolves stale coordinates, and preserves zoom when changing tensors', async () => {
    const {fixture, view} = create(undefined, {
      plot: 'index',
      position: 999,
      layer: 3,
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: {channel: 15},
      reduction: 'max',
    });
    await settle(fixture);
    expect(view.position()).toBe(66);
    expect(view.layer()).toBe(0);
    expect(view.anchor()).toBe('output');
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.selectedChannel()?.channel).toBe(15);
    expect(view.reduction()).toBe('max');
    view.setLayer(7);
    await settle(fixture);
    expect(view.layer()).toBe(7);
    expect(view.selectedIndex()).toBeNull();
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.captureView().valueRange).toEqual({start: -2, end: 2});
    view.setPosition(-100);
    await settle(fixture);
    expect(view.position()).toBe(66);
    fixture.destroy();
  });
  it('keeps display changes in drafts and disables bins at individual-element scale', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    view.toggleDisplay();
    view.setDisplayDraft('Maximum');
    expect(view.displayDirty()).toBe(true);
    expect(view.reduction()).toBe('mean');
    view.displayOpen.set(false);
    view.toggleDisplay();
    expect(view.displayDraft()).toBe('mean');
    view.setDisplayDraft('Minimum');
    view.applyDisplay();
    expect(view.reduction()).toBe('min');
    view.zoomElements({start: 1, end: 2});
    expect(view.displayBinsDisabled()).toBe(true);
    view.setDisplayDraft('Mean');
    expect(view.reduction()).toBe('min');
    view.chooseView('Tensor Inspection · Scatter plot');
    view.toggleDisplay();
    expect(view.displayOpen()).toBe(false);
    fixture.destroy();
  });
  it('restores a sync point and changes tensors without losing position, layer, or zoom', async () => {
    const {fixture, view, api} = create(undefined, {
      plot: 'index',
      position: 40,
      layer: 7,
      anchor: 'attention_output',
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: {channel: 15},
    });
    await settle(fixture);
    expect(api.positionTensor.mock.calls.at(-1)?.[3]).toMatchObject({
      position: 40,
      layer: 7,
      anchor: 'attention_output',
    });
    expect(view.selectedIndex()).toBe(15);
    view.channelSelection.set({start: 10, end: 20, plane: null});
    view.setAnchor('pre_attention_norm');
    expect(view.selectedIndex()).toBeNull();
    expect(view.channelSelection()).toBeNull();
    await settle(fixture);
    expect(view.captureView()).toMatchObject({
      position: 40,
      layer: 7,
      anchor: 'pre_attention_norm',
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: null,
    });
    expect(api.positionTensor.mock.calls.at(-1)?.[3].anchor).toBe('pre_attention_norm');
    const requestCount = api.positionTensor.mock.calls.length;
    view.setAnchor('unknown');
    await settle(fixture);
    expect(view.anchor()).toBe('pre_attention_norm');
    expect(api.positionTensor).toHaveBeenCalledTimes(requestCount);
    fixture.destroy();
  });
  it('keeps an unavailable saved sync point instead of substituting a different tensor', async () => {
    const {fixture, view, api} = create(
      async (_capture, _turn, _phase, query) => ({
        ...vector(query),
        reference: null,
        target: null,
        status: 'unavailable',
      }),
      {plot: 'index', position: 32, layer: 0, anchor: 'missing_capture_point'},
    );
    await settle(fixture);
    expect(api.positionTensor.mock.calls.at(-1)?.[3].anchor).toBe('missing_capture_point');
    expect(view.anchor()).toBe('missing_capture_point');
    expect(view.tensorIndexDimensions()[1].value).toBe('missing_capture_point');
    expect(view.channels()).toEqual([]);
    expect(view.notices()).toHaveLength(2);
    view.setLayer(7);
    await settle(fixture);
    expect(view.captureView().anchor).toBe('missing_capture_point');
    fixture.destroy();
  });
  it('renders a missing side independently and suppresses paired differences and scatter', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) => ({
      ...vector(query, 3),
      reference: null,
      reference_position: null,
    }));
    await settle(fixture);
    expect(view.channels().map((row) => row.ref)).toEqual([null, null, null]);
    expect(view.channels().map((row) => row.target)).toEqual([1, 1.1, 1.2]);
    expect(view.channels().every((row) => row.delta === null)).toBe(true);
    expect(view.compatible()).toBe(false);
    expect(view.scatterChannels()).toEqual([]);
    expect(view.notices()).toContain('Reference tensor is unavailable at this aligned position.');
    fixture.destroy();
  });
  it('rejects mismatched source shapes and sanitizes unavailable elements', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) => ({
      ...vector(query, 3),
      reference: {values: [1, null, Infinity], shape: [1, 3], dtype: null},
    }));
    await settle(fixture);
    expect(view.channels().map((row) => row.ref)).toEqual([1, null, null]);
    expect(view.compatible()).toBe(false);
    expect(view.channels().every((row) => row.delta === null)).toBe(true);
    fixture.destroy();
  });
  it('cancels outdated requests, clears stale values immediately, and retries errors', async () => {
    const requests: {
      query: PositionTensorRequest;
      signal: AbortSignal;
      resolve: (value: PositionTensorResponse) => void;
      reject: (error: Error) => void;
    }[] = [];
    const {fixture, view} = create(
      async (_capture, _turn, _phase, query, signal) =>
        new Promise<PositionTensorResponse>((resolve, reject) =>
          requests.push({query, signal, resolve, reject}),
        ),
    );
    expect(view.loading()).toBe(true);
    requests[0].resolve(vector(requests[0].query, 3));
    await settle(fixture);
    expect(view.total()).toBe(3);
    view.setPosition(33);
    fixture.detectChanges();
    expect(requests[0].signal.aborted).toBe(true);
    expect(view.data()).toBeNull();
    expect(view.channels()).toEqual([]);
    view.setPosition(34);
    fixture.detectChanges();
    expect(requests[1].signal.aborted).toBe(true);
    requests[1].resolve(vector(requests[1].query, 20));
    requests[2].reject(new Error('Tensor missing'));
    await settle(fixture);
    expect(view.error()).toBe('Tensor missing');
    expect(view.total()).toBe(0);
    view.retry();
    fixture.detectChanges();
    expect(view.error()).toBe('');
    expect(view.loading()).toBe(true);
    requests[3].resolve(vector(requests[3].query, 4));
    await settle(fixture);
    expect(view.total()).toBe(4);
    fixture.destroy();
    expect(requests[3].signal.aborted).toBe(true);
  });
  it('rejects a response for another tensor identity', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector({...query, layer: 99}),
    );
    await settle(fixture);
    expect(view.error()).toContain('did not match');
    expect(view.data()).toBeNull();
    fixture.destroy();
  });
  it('rejects a response from another sync point', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector({...query, anchor: 'attention_output'}),
    );
    await settle(fixture);
    expect(view.error()).toContain('sync point');
    expect(view.data()).toBeNull();
    fixture.destroy();
  });
  it('ignores a late response for the previous sync point without taking details focus', async () => {
    const requests: {
      query: PositionTensorRequest;
      signal: AbortSignal;
      resolve: (value: PositionTensorResponse) => void;
    }[] = [];
    const {fixture, view} = create(
      async (_capture, _turn, _phase, query, signal) =>
        new Promise<PositionTensorResponse>((resolve) => requests.push({query, signal, resolve})),
    );
    const inspect = vi.fn();
    view.inspect.subscribe(inspect);
    view.setAnchor('attention_output', false);
    fixture.detectChanges();
    expect(requests[0].signal.aborted).toBe(true);
    expect(requests[1].query.anchor).toBe('attention_output');
    requests[1].resolve(vector(requests[1].query, 4));
    await settle(fixture);
    requests[0].resolve(vector(requests[0].query, 20));
    await settle(fixture);
    expect(view.data()?.anchor).toBe('attention_output');
    expect(view.total()).toBe(4);
    expect(inspect).not.toHaveBeenCalled();
    fixture.destroy();
  });
  it('uses the shared Tensor Index picker and rejects invalid position edits like KV', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    const picker = fixture.debugElement.query(By.directive(TensorIndexControl))
      .componentInstance as TensorIndexControl;
    expect(picker.summary()).toBe('[Layer: 0, Sync point: Layer output]');
    expect(picker.dimensions().map((dimension) => dimension.id)).toEqual(['layer', 'anchor']);
    picker.choose('layer', '7');
    await settle(fixture);
    expect(view.layer()).toBe(7);
    expect(picker.summary()).toBe('[Layer: 7, Sync point: Layer output]');
    picker.choose('anchor', 'attention_output');
    await settle(fixture);
    expect(view.anchor()).toBe('attention_output');
    expect(picker.summary()).toBe('[Layer: 7, Sync point: attention_output]');
    const input = fixture.nativeElement.querySelector(
      '[aria-label="Token Position"]',
    ) as HTMLInputElement;
    const requestCount = api.positionTensor.mock.calls.length;
    for (const value of ['', '31', '67', '32.5']) {
      input.value = value;
      input.dispatchEvent(new Event('change'));
      await settle(fixture);
      expect(input.value).toBe('32');
      expect(view.position()).toBe(32);
    }
    expect(api.positionTensor).toHaveBeenCalledTimes(requestCount);
    input.value = '40';
    input.dispatchEvent(new Event('change'));
    await settle(fixture);
    expect(view.position()).toBe(40);
    (
      fixture.nativeElement.querySelector(
        '[aria-label="Next tensor position"]',
      ) as HTMLButtonElement
    ).click();
    await settle(fixture);
    expect(view.position()).toBe(41);
    fixture.destroy();
  });
  it('applies runtime visibility only on Apply and retains it across plots and saved view restoration', async () => {
    const {fixture, view} = create(undefined, {
      plot: 'index',
      position: 32,
      layer: 0,
      visibleRuntimes: ['target'],
    });
    await settle(fixture);
    view.showElement(15);
    view.zoomElements({start: 10, end: 30});
    view.toggleDisplay();
    expect(view.displayBinsDisabled()).toBe(true);
    view.toggleDisplayRuntime('ref');
    expect(view.displayDirty()).toBe(true);
    expect(view.visibleRuntimes()).toEqual(['target']);
    view.displayOpen.set(false);
    view.toggleDisplay();
    expect(view.displayRuntimes()).toEqual(['target']);
    view.displayRuntimes.set([]);
    view.applyDisplay();
    fixture.detectChanges();
    expect(view.visibleRuntimes()).toEqual([]);
    expect(view.displayDirty()).toBe(false);
    expect(view.captureView().visibleRuntimes).toEqual([]);
    const chart = fixture.debugElement.query(By.directive(KvChannelChart))
      .componentInstance as KvChannelChart;
    expect(chart.visibleRuntimes()).toEqual([]);
    expect(fixture.nativeElement.querySelector('analysis-runtime-legend').textContent.trim()).toBe(
      '',
    );
    view.chooseView('Tensor Inspection · Scatter plot');
    view.chooseView('Tensor Inspection · Line chart');
    expect(view.visibleRuntimes()).toEqual([]);
    expect(view.selectedIndex()).toBe(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    view.toggleDisplay();
    view.displayRuntimes.set(['ref', 'target']);
    view.applyDisplay();
    expect(view.captureView().visibleRuntimes).toEqual(['ref', 'target']);
    fixture.destroy();
  });
  it('drops only multi-element bin selection on scatter and keeps the representative element and both zoom ranges', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    view.showElement(12);
    view.channelSelection.set({start: 10, end: 14, plane: null});
    view.zoomElements({start: 10, end: 100});
    view.zoomValues({start: -1, end: 2});
    view.chooseView('Tensor Inspection · Scatter plot');
    expect(view.channelSelection()).toBeNull();
    expect(view.selectedIndex()).toBe(12);
    expect(view.valueRange()).toEqual({start: -1, end: 2});
    view.chooseView('Tensor Inspection · Line chart');
    expect(view.elementRange()).toEqual({start: 10, end: 100});
    view.channelSelection.set({start: 12, end: 13, plane: null});
    view.chooseView('Tensor Inspection · Scatter plot');
    expect(view.channelSelection()).toEqual({start: 12, end: 13, plane: null});
    fixture.destroy();
  });
  it('preserves visible individual-element zoom during location and emits background inspection to its parent', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    const chart = fixture.debugElement.query(By.directive(KvChannelChart));
    const locate = vi
      .spyOn(chart.componentInstance as KvChannelChart, 'locateKind')
      .mockImplementation(() => {});
    view.zoomElements({start: 10, end: 30});
    view.locateElement(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.selectedIndex()).toBe(15);
    view.locateTensor();
    expect(view.selectedIndex()).toBe(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    view.locateElement(100);
    expect(view.elementRange()).toEqual({start: 90, end: 110});
    expect(locate).toHaveBeenCalledWith('key');
    const background = vi.fn();
    view.inspectStage.subscribe(background);
    const event = new MouseEvent('click');
    chart.triggerEventHandler('backgroundClick', event);
    expect(background).toHaveBeenCalledWith(event);
    expect(view.selectedIndex()).toBe(100);
    fixture.destroy();
  });
  it('reports asynchronous data changes without taking inspection focus from the active panel', async () => {
    let resolve!: (data: PositionTensorResponse) => void;
    let query!: PositionTensorRequest;
    const {fixture, view} = create(async (_capture, _turn, _phase, request) => {
      query = request;
      return new Promise((done) => {
        resolve = done;
      });
    });
    const inspect = vi.fn(),
      stateChange = vi.fn();
    view.inspect.subscribe(inspect);
    view.stateChange.subscribe(stateChange);
    resolve(vector(query));
    await settle(fixture);
    expect(stateChange).toHaveBeenCalledWith(view);
    expect(inspect).not.toHaveBeenCalled();
    view.showElement(4);
    expect(inspect).toHaveBeenCalledOnce();
    inspect.mockClear();
    view.zoomElements({start: 0, end: 20});
    view.toggleDisplay();
    view.toggleDisplayRuntime('ref');
    view.applyDisplay();
    await settle(fixture);
    expect(inspect).not.toHaveBeenCalled();
    view.setPosition(33, false);
    expect(inspect).not.toHaveBeenCalled();
    view.setPosition(34);
    expect(inspect).toHaveBeenCalledOnce();
    fixture.destroy();
  });
  it('applies initialPosition and initialLayer updates without clobbering plot or zoom state', async () => {
    const {fixture, view} = create(undefined, {
      plot: 'scatter',
      position: 35,
      layer: 0,
      elementRange: {start: 10, end: 30},
      valueRange: {start: -1, end: 1},
      reduction: 'max',
      visibleRuntimes: ['target'],
    });
    await settle(fixture);
    expect(view.plot()).toBe('scatter');
    expect(view.position()).toBe(35);
    expect(view.layer()).toBe(0);
    fixture.componentRef.setInput('initialPosition', 42);
    fixture.componentRef.setInput('initialLayer', 7);
    await settle(fixture);
    expect(view.position()).toBe(42);
    expect(view.layer()).toBe(7);
    expect(view.plot()).toBe('scatter');
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.valueRange()).toEqual({start: -1, end: 1});
    expect(view.reduction()).toBe('max');
    expect(view.visibleRuntimes()).toEqual(['target']);
    fixture.destroy();
  });
});
