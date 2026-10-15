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
import {MatTooltip} from '@angular/material/tooltip';
import type {
  FinalLogitsRequest,
  FinalLogitsResponse,
  FinalLogitsVectorRequest,
  FinalLogitsVectorResponse,
} from '../../data/contracts/final_logits';
import type {TokenAnalysisPair} from '../../data/contracts/token_analysis';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import type {TokenTrendState} from '../conversation/token_trends';
import {FinalLogitsInspection} from './final_logits_inspection';

describe('FinalLogitsInspection', () => {
  const savedDescriptors = [
    [Range.prototype, 'getBoundingClientRect'],
    [HTMLElement.prototype, 'scrollIntoView'],
    [HTMLElement.prototype, 'animate'],
  ].map(([prototype, key]) => ({
    prototype: prototype as object,
    key: key as string,
    descriptor: Object.getOwnPropertyDescriptor(prototype, key as string),
  }));
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
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(HTMLElement.prototype, 'animate', {
      configurable: true,
      value: vi.fn(() => ({cancel() {}})),
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    for (const saved of savedDescriptors) {
      if (saved.descriptor) Object.defineProperty(saved.prototype, saved.key, saved.descriptor);
      else Reflect.deleteProperty(saved.prototype, saved.key);
    }
  });
  const initial: TokenTrendState = {
    open: true,
    metric: 'logit_rmse',
    start: 0,
    end: 3,
    logitView: 'inspection',
    logitPlot: 'index',
    logitTransform: 'raw',
    logitOrder: 'index',
    logitOffset: 0,
  };
  const analysis: TokenAnalysisPair = {
    ref_step: 0,
    target_step: 0,
    metrics: Object.fromEntries(
      ['kl', 'js', 'relative_l2', 'cosine_distance', 'norm_ratio', 'max_abs_error'].map((key) => [
        key,
        {value: null, reason: null},
      ]),
    ) as TokenAnalysisPair['metrics'],
    distribution: {compatible: true, reason: null, ref: {size: 70}, target: {size: 70}, rows: []},
  };
  function response(query: FinalLogitsRequest, total = 70): FinalLogitsResponse {
    return {
      ...query,
      compatible: true,
      reason: null,
      total,
      ref: {size: total},
      target: {size: total},
      metrics: {
        rmse: {value: 1, reason: null},
        max_abs: {value: 1, reason: null},
        max_abs_index: 65,
      },
      rows: Array.from(
        {length: Math.max(0, Math.min(query.limit, total - query.offset))},
        (_, i) => ({
          index: query.offset + i,
          label: 'duplicate token label',
          ref: query.offset + i,
          target: query.offset + i + 1,
          delta: 1,
          absolute_delta: 1,
        }),
      ),
    };
  }
  function vectorResponse(query: FinalLogitsVectorRequest, total = 70): FinalLogitsVectorResponse {
    const {
      offset: _offset,
      limit: _limit,
      rows: _rows,
      ...metadata
    } = response({...query, offset: 0, limit: 16, order: 'index'}, total);
    return {
      ...metadata,
      complete: true,
      values: {
        ref: Array.from({length: total}, (_, index) => index),
        target: Array.from({length: total}, (_, index) => index + 1),
      },
    };
  }
  type Read = (
    capture: string | null,
    query: FinalLogitsRequest,
    signal: AbortSignal,
  ) => Promise<FinalLogitsResponse>;
  type ReadVector = (
    capture: string | null,
    query: FinalLogitsVectorRequest,
    signal: AbortSignal,
  ) => Promise<FinalLogitsVectorResponse>;
  function create(
    options: {
      read?: Read;
      readVector?: ReadVector;
      settings?: TokenTrendState;
      pair?: TokenAnalysisPair | null;
    } = {},
  ) {
    const api = {
      finalLogits: vi.fn(options.read ?? (async (_capture, query) => response(query))),
      finalLogitsVector: vi.fn(
        options.readVector ?? (async (_capture, query) => vectorResponse(query)),
      ),
    };
    const captureId = signal<string | null>('capture');
    TestBed.configureTestingModule({
      providers: [
        {provide: ReportApiService, useValue: api},
        {provide: ReportStateService, useValue: {captureId}},
      ],
    });
    const fixture = TestBed.createComponent(FinalLogitsInspection);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('position', 32);
    fixture.componentRef.setInput('fullRange', {start: 32, end: 35});
    fixture.componentRef.setInput(
      'analysis',
      options.pair === null ? undefined : (options.pair ?? analysis),
    );
    fixture.componentRef.setInput('settings', options.settings ?? initial);
    const changes: TokenTrendState[] = [];
    fixture.componentInstance.settingsChange.subscribe((next) => {
      changes.push(next);
      fixture.componentRef.setInput('settings', next);
    });
    const positions: number[] = [];
    fixture.componentInstance.positionChange.subscribe((next) => {
      positions.push(next);
      fixture.componentRef.setInput('position', next);
    });
    fixture.detectChanges();
    return {fixture, view: fixture.componentInstance, api, changes, positions, captureId};
  }
  async function settle(fixture: ComponentFixture<FinalLogitsInspection>) {
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    TestBed.tick();
  }

  it('migrates a saved Centered viewport to raw once without losing sorting or element range', async () => {
    const {fixture, view, api, changes} = create({
      settings: {
        ...initial,
        logitPlot: 'scatter',
        logitTransform: 'centered',
        logitValueRange: {start: -2, end: 2},
        logitElementRange: {start: 16, end: 48},
        logitOffset: 32,
        logitOrder: 'target',
        logitSortDirection: 'descending',
      },
    });
    await settle(fixture);
    expect(view.transform()).toBe('raw');
    expect(api.finalLogitsVector.mock.calls.every((call) => call[1].transform === 'raw')).toBe(
      true,
    );
    expect(
      api.finalLogits.mock.calls.every(
        (call) => call[1].transform === 'raw' && call[1].offset === 0,
      ),
    ).toBe(true);
    expect(view.valueRange()).toEqual(view.valueFull());
    expect(view.settings().logitTransform).toBe('raw');
    expect(view.elementRange()).toEqual({start: 16, end: 48});
    expect(view.order()).toBe('target');
    expect(view.sortDirection()).toBe('descending');
    expect(changes).toHaveLength(1);
    expect(fixture.nativeElement.querySelector('[label="Logits transform"]')).toBeNull();
    expect(fixture.nativeElement.textContent).not.toContain('Transform');
    view.zoomValues({start: 20, end: 40});
    await settle(fixture);
    expect(view.valueRange()).toEqual({start: 20, end: 40});
    view.changePage(1);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      transform: 'raw',
      offset: 16,
      order: 'target',
      direction: 'descending',
    });
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('keeps the legend and Display action outside scrolling controls in both views and removes the expanded slot', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    const element = fixture.nativeElement as HTMLElement;
    const display = element.querySelector<HTMLButtonElement>('[aria-label="Display"]')!;
    expect(element.querySelector('.toolbar-actions')?.contains(display)).toBe(true);
    expect(element.querySelector('.toolbar-scroll')?.contains(display)).toBe(false);
    expect(display.disabled).toBe(false);
    expect(element.querySelector('.toolbar-legend .ref')).not.toBeNull();
    expect(element.querySelector('.toolbar-legend .target')).not.toBeNull();
    expect(element.querySelector('[aria-label="Tensor runtime legend"]')?.textContent).toContain(
      'Ref',
    );
    expect(element.querySelector('.panel-expand')).not.toBeNull();
    view.chooseView('Tensor Inspection · Scatter plot');
    await settle(fixture);
    expect(element.querySelector('[aria-label="Display"]')).toBe(display);
    expect(display.disabled).toBe(true);
    expect(element.querySelector('[aria-label="Tensor scatter axes"]')?.textContent?.trim()).toBe(
      'X: Ref · Y: Target',
    );
    expect(
      fixture.debugElement.query(By.css('[aria-label="Display"]')).injector.get(MatTooltip).message,
    ).toBe('No display settings for Scatter plot');
    display.click();
    expect(view.displayOpen()).toBe(false);
    fixture.componentRef.setInput('expanded', true);
    await settle(fixture);
    expect(element.querySelector('.panel-expand')).toBeNull();
    view.chooseView('Tensor Inspection · Line chart');
    await settle(fixture);
    expect(display.disabled).toBe(false);
    expect(element.querySelector('[aria-label="Tensor runtime legend"]')).not.toBeNull();
    fixture.destroy();
  });

  it('stages Reset to mean, discards a closed draft, and applies bins without changing sorting or requests', async () => {
    const {fixture, view, api, changes} = create({
      settings: {
        ...initial,
        logitElementReduction: 'max',
        logitOrder: 'delta',
        logitSortDirection: 'descending',
      },
    });
    await settle(fixture);
    view.capacity.set(16);
    const display = fixture.nativeElement.querySelector(
      '[aria-label="Display"]',
    ) as HTMLButtonElement;
    display.click();
    await settle(fixture);
    let panel = document.querySelector('[aria-label="Final logits display settings"]')!;
    expect(panel.querySelector('header [aria-label="Close chart display"]')).not.toBeNull();
    (
      panel.querySelector('[aria-label="Reset Element bins to default"]') as HTMLButtonElement
    ).click();
    expect(view.displayDraft()).toBe('mean');
    expect(view.reduction()).toBe('max');
    expect(changes).toEqual([]);
    (panel.querySelector('[aria-label="Close chart display"]') as HTMLButtonElement).click();
    await settle(fixture);
    display.click();
    await settle(fixture);
    expect(view.displayDraft()).toBe('max');
    panel = document.querySelector('[aria-label="Final logits display settings"]')!;
    (
      panel.querySelector('[aria-label="Reset Element bins to default"]') as HTMLButtonElement
    ).click();
    await settle(fixture);
    (panel.querySelector('footer button') as HTMLButtonElement).click();
    await settle(fixture);
    expect(view.reduction()).toBe('mean');
    expect(view.displayOpen()).toBe(true);
    expect(panel.querySelector('[role="status"]')?.textContent?.trim()).toBe(
      'Display settings applied',
    );
    expect((panel.querySelector('footer button') as HTMLButtonElement).disabled).toBe(true);
    expect(changes).toHaveLength(1);
    expect(view.order()).toBe('delta');
    expect(view.sortDirection()).toBe('descending');
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    expect(api.finalLogits).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('disables Element bins when every bin is one element without losing the saved reduction', async () => {
    const {fixture, view, changes} = create({settings: {...initial, logitElementReduction: 'max'}});
    await settle(fixture);
    (fixture.nativeElement.querySelector('[aria-label="Display"]') as HTMLButtonElement).click();
    await settle(fixture);
    const panel = document.querySelector('[aria-label="Final logits display settings"]')!;
    const picker = panel.querySelector('config-picker')!;
    expect(picker.querySelector('.value')?.textContent?.trim()).toBe('-');
    expect(picker.querySelector('button')?.disabled).toBe(true);
    expect((panel.querySelector('footer button') as HTMLButtonElement).disabled).toBe(true);
    view.setDisplayDraft('Mean');
    view.applyDisplay();
    expect(view.reduction()).toBe('max');
    expect(changes).toEqual([]);
    fixture.destroy();
  });

  it.each(['index', 'ref', 'target', 'delta', 'absolute_delta'] as const)(
    'sorts the %s column in both directions, resets paging, and leaves the complete chart and metrics alone',
    async (column) => {
      const settings: TokenTrendState = {
        ...initial,
        logitOrder: column === 'index' ? 'target' : 'index',
        logitSortDirection: 'descending',
        logitOffset: 32,
        logitElementRange: {start: 16, end: 48},
      };
      const {fixture, view, api} = create({settings});
      await settle(fixture);
      const channels = view.channels(),
        metrics = view.data()!.metrics;
      view.sortBy(column);
      await settle(fixture);
      expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
        order: column,
        direction: 'ascending',
        offset: 0,
      });
      expect(view.order()).toBe(column);
      expect(view.sortDirection()).toBe('ascending');
      expect(view.columnSortDirection(column)).toBe('ascending');
      expect(view.columnSortDirection(column === 'index' ? 'target' : 'index')).toBe('none');
      view.changePage(1);
      await settle(fixture);
      expect(view.offset()).toBe(16);
      view.sortBy(column);
      await settle(fixture);
      expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
        order: column,
        direction: 'descending',
        offset: 0,
      });
      expect(view.columnSortDirection(column)).toBe('descending');
      expect(view.channels()).toBe(channels);
      expect(view.lineChart()?.rows()).toBe(channels);
      expect(view.data()?.metrics).toBe(metrics);
      expect(view.elementRange()).toEqual(settings.logitElementRange);
      expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
      fixture.destroy();
    },
  );

  it.each([
    ['absolute_delta', 'descending'],
    ['index', 'ascending'],
  ] as const)(
    'restores legacy %s ordering with an explicit %s request',
    async (order, direction) => {
      const {fixture, view, api, changes} = create({settings: {...initial, logitOrder: order}});
      await settle(fixture);
      expect(view.sortDirection()).toBe(direction);
      expect(view.columnSortDirection(order)).toBe(direction);
      expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({order, direction});
      expect(changes).toEqual([]);
      fixture.destroy();
    },
  );

  it('keeps the selected sort column and direction when a new source resets the table page', async () => {
    const {fixture, view, api} = create({
      settings: {
        ...initial,
        logitOrder: 'target',
        logitSortDirection: 'descending',
        logitOffset: 32,
      },
    });
    await settle(fixture);
    fixture.componentRef.setInput('position', 33);
    fixture.componentRef.setInput('analysis', {...analysis, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      order: 'target',
      direction: 'descending',
      offset: 0,
      ref_step: 1,
      target_step: 1,
    });
    expect(view.order()).toBe('target');
    expect(view.sortDirection()).toBe('descending');
    fixture.destroy();
  });

  it('keeps direct table selection in descending Index pages and locates an element in ascending Index order', async () => {
    const {fixture, view, api, positions} = create({
      settings: {...initial, logitOrder: 'index', logitSortDirection: 'descending'},
      read: async (_capture, query) => {
        const page = response(query);
        if (query.order === 'index' && query.direction === 'descending')
          page.rows = page.rows.map((row) => ({...row, index: page.total - 1 - row.index}));
        return page;
      },
    });
    await settle(fixture);
    expect(view.rows().map((row) => row.index)).toContain(65);
    const reads = api.finalLogits.mock.calls.length;
    view.selectElement(65);
    await settle(fixture);
    expect(api.finalLogits).toHaveBeenCalledTimes(reads);
    view.selectElement(1);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      order: 'index',
      direction: 'descending',
      offset: 64,
    });
    expect(view.rows().map((row) => row.index)).toContain(1);
    view.locateElement(65);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      order: 'index',
      direction: 'ascending',
      offset: 64,
    });
    expect(view.rows().map((row) => row.index)).toContain(65);
    expect(view.selectedIndex()).toBe(65);
    expect(view.position()).toBe(32);
    expect(positions).toEqual([]);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it.each(['target', 'absolute_delta', 'index'] as const)(
    'reveals a chart bin representative from %s sorting without zooming or losing the bin highlight',
    async (order) => {
      const {fixture, view, api, positions} = create({
        settings: {
          ...initial,
          logitOrder: order,
          logitSortDirection: 'descending',
          logitOffset: 16,
          logitElementRange: {start: 32, end: 70},
        },
      });
      await settle(fixture);
      const channels = view.channels();
      const reads = api.finalLogits.mock.calls.length;
      const chart = view.lineChart()!;
      // The shared chart emits its representative before restoring the selected bin.
      chart.select.emit(65);
      chart.channelSelection.set({start: 64, end: 70, plane: 0});
      await settle(fixture);
      expect(api.finalLogits).toHaveBeenCalledTimes(reads + 1);
      expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
        order: 'index',
        direction: 'ascending',
        offset: 64,
      });
      expect(view.rows().map((row) => row.index)).toContain(65);
      expect(view.selectedIndex()).toBe(65);
      expect(view.channelSelection()).toEqual({start: 64, end: 70, plane: 0});
      expect(chart.channelSelection()).toEqual({start: 64, end: 70, plane: 0});
      expect(view.elementRange()).toEqual({start: 32, end: 70});
      expect(view.channels()).toBe(channels);
      expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
      expect(view.position()).toBe(32);
      expect(positions).toEqual([]);
      fixture.destroy();
    },
  );

  it('reveals a Scatter element on its table page while direct table selection preserves value sorting', async () => {
    const {fixture, view, api} = create({
      settings: {
        ...initial,
        logitPlot: 'scatter',
        logitOrder: 'target',
        logitSortDirection: 'descending',
        logitValueRange: {start: 1, end: 8},
      },
    });
    await settle(fixture);
    view.selectElement(1);
    await settle(fixture);
    expect(view.order()).toBe('target');
    expect(view.sortDirection()).toBe('descending');
    expect(api.finalLogits).toHaveBeenCalledTimes(1);
    view.scatterChart()!.select.emit(65);
    await settle(fixture);
    expect(view.rows().map((row) => row.index)).toContain(65);
    expect(view.selectedIndex()).toBe(65);
    expect(view.order()).toBe('index');
    expect(view.sortDirection()).toBe('ascending');
    expect(view.valueRange()).toEqual({start: 1, end: 8});
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('keeps the complete chart vector independent of the 16-row auxiliary table pages', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    const channels = view.channels();
    expect(view.selectedIndex()).toBeNull();
    expect(view.lineChart()?.rows()).toBe(channels);
    expect(channels).toHaveLength(70);
    expect(channels.at(-1)).toMatchObject({channel: 69, ref: 69, target: 70});
    expect(view.rows()).toHaveLength(16);
    expect(view.pages()).toBe(5);
    expect(fixture.nativeElement.querySelector('kv-channel-chart')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('table')).toBeNull();
    expect(fixture.nativeElement.textContent).not.toMatch(/probability|rank/i);

    view.changePage(1);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      offset: 16,
      limit: 16,
      order: 'index',
    });
    expect(view.rows()[0].index).toBe(16);
    expect(view.channels()).toBe(channels);
    expect(view.lineChart()?.rows()).toBe(channels);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);

    view.changePage(99);
    await settle(fixture);
    expect(view.rows().map((row) => row.index)).toEqual([64, 65, 66, 67, 68, 69]);
    expect(view.pageIndex()).toBe(4);
    const pageReads = api.finalLogits.mock.calls.length;
    view.changePage(1);
    await settle(fixture);
    expect(api.finalLogits).toHaveBeenCalledTimes(pageReads);
    expect(view.channels()).toBe(channels);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('routes Line and Scatter through View without requesting the vector or page again', async () => {
    const {fixture, view, api, changes} = create();
    await settle(fixture);
    const channels = view.channels();
    view.chooseView('Tensor Inspection · Scatter plot');
    await settle(fixture);
    expect(view.scatterChart()?.rows()).toBe(channels);
    expect(view.scatterChart()?.singlePlane()).toBe(true);
    expect(view.scatterChart()?.dimensionLabel()).toBe('Vocabulary index');
    expect(fixture.nativeElement.querySelector('kv-channel-chart')).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[aria-label="Value Range controls"]'),
    ).not.toBeNull();
    view.chooseView('Tensor Inspection · Line chart');
    await settle(fixture);
    expect(view.lineChart()?.rows()).toBe(channels);
    expect(fixture.nativeElement.querySelector('kv-scatter-chart')).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[aria-label="Element Range controls"]'),
    ).not.toBeNull();
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    expect(api.finalLogits).toHaveBeenCalledTimes(1);
    view.chooseView('Tensor Comparison · Line chart');
    expect(changes.at(-1)?.logitView).toBe('comparison');
    fixture.destroy();
  });

  it('keeps page ordering notices separate from complete-vector metrics and chart notices', async () => {
    const fallback = 'Absolute delta ordering unavailable; showing vocabulary index order';
    const {fixture, view, api} = create({
      read: async (_capture, query) => ({
        ...response(query),
        reason: query.order === 'absolute_delta' ? fallback : null,
      }),
    });
    await settle(fixture);
    const metrics = view.data()!.metrics;
    view.sortBy('absolute_delta');
    await settle(fixture);
    expect(view.pageValuesNotices()).toEqual([fallback]);
    expect(view.notices()).toEqual([]);
    expect(view.data()?.reason).toBeNull();
    expect(view.data()?.metrics).toBe(metrics);
    expect(fixture.nativeElement.textContent).not.toContain(fallback);
    view.sortBy('index');
    await settle(fixture);
    expect(view.pageValuesNotices()).toEqual([]);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('preserves saved page and plot ranges when the selected analysis initially resolves', async () => {
    const settings: TokenTrendState = {
      ...initial,
      logitOffset: 32,
      logitPlot: 'scatter',
      logitElementRange: {start: 16, end: 48},
      logitValueRange: {start: -2, end: 12},
    };
    const {fixture, view, api, changes} = create({pair: null, settings});
    await settle(fixture);
    expect(api.finalLogitsVector).not.toHaveBeenCalled();
    expect(api.finalLogits).not.toHaveBeenCalled();
    fixture.componentRef.setInput('analysis', analysis);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1].offset).toBe(32);
    expect(view.elementRange()).toEqual(settings.logitElementRange);
    expect(view.valueRange()).toEqual(settings.logitValueRange);
    expect(view.plot()).toBe('scatter');
    expect(changes).toEqual([]);

    fixture.componentRef.setInput('analysis', undefined);
    await settle(fixture);
    fixture.componentRef.setInput('analysis', {...analysis});
    await settle(fixture);
    expect(view.settings()).toEqual(settings);
    expect(changes).toEqual([]);
    fixture.destroy();
  });

  it.each(['position', 'turn', 'capture'] as const)(
    'resets the saved viewport when %s changes while analysis is pending',
    async (source) => {
      const {fixture, view, captureId} = create({
        pair: null,
        settings: {
          ...initial,
          logitOffset: 32,
          logitElementRange: {start: 16, end: 48},
          logitValueRange: {start: -2, end: 12},
        },
      });
      await settle(fixture);
      if (source === 'capture') captureId.set('another-capture');
      else fixture.componentRef.setInput(source, source === 'turn' ? 3 : 33);
      await settle(fixture);
      expect(view.settings().logitOffset).toBe(0);
      expect(view.settings().logitElementRange).toBeUndefined();
      expect(view.settings().logitValueRange).toBeUndefined();
      fixture.componentRef.setInput('analysis', {...analysis, ref_step: 1, target_step: 1});
      await settle(fixture);
      expect(view.offset()).toBe(0);
      expect(view.elementRange()).toEqual(view.elementFull());
      fixture.destroy();
    },
  );

  it('resets the viewport when a resolved pair changes at the same token position', async () => {
    const {fixture, view, api} = create({
      settings: {
        ...initial,
        logitOffset: 32,
        logitElementRange: {start: 16, end: 48},
        logitValueRange: {start: -2, end: 12},
      },
    });
    await settle(fixture);
    fixture.componentRef.setInput('analysis', undefined);
    await settle(fixture);
    fixture.componentRef.setInput('analysis', {...analysis, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogitsVector.mock.lastCall?.[1]).toMatchObject({ref_step: 1, target_step: 1});
    expect(view.settings().logitOffset).toBe(0);
    expect(view.settings().logitElementRange).toBeUndefined();
    expect(view.settings().logitValueRange).toBeUndefined();
    expect(view.position()).toBe(32);
    fixture.destroy();
  });

  it('locates index 65 on its auxiliary page and narrows the chart without changing token position', async () => {
    const {fixture, view, api, positions} = create();
    await settle(fixture);
    view.capacity.set(16);
    view.sortBy('absolute_delta');
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1].order).toBe('absolute_delta');
    view.locateElement(65);
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      offset: 64,
      order: 'index',
      ref_step: 0,
      target_step: 0,
    });
    expect(view.rows().some((row) => row.index === 65)).toBe(true);
    expect(view.selectedIndex()).toBe(65);
    expect(view.lineChart()?.selected()).toBe(65);
    expect(view.elementRange().start).toBeLessThanOrEqual(65);
    expect(view.elementRange().end).toBeGreaterThan(65);
    expect(view.elementRange().end - view.elementRange().start).toBeLessThan(view.total());
    expect(view.channelSelection()).toMatchObject({start: 65, end: 66});
    expect(view.position()).toBe(32);
    expect(positions).toEqual([]);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    view.locateIndex();
    const rangeControl = fixture.nativeElement.querySelector('.toolbar-range') as HTMLElement;
    expect(rangeControl.scrollIntoView).toHaveBeenCalled();
    view.clearElementSelection();
    expect(view.selectedIndex()).toBeNull();
    expect(view.channelSelection()).toBeNull();
    expect(view.offset()).toBe(64);
    fixture.destroy();
  });

  it('reloads the complete raw vector for source changes while resetting page and ranges', async () => {
    const {fixture, view, api, captureId} = create();
    await settle(fixture);
    view.capacity.set(16);
    view.locateElement(65);
    view.zoomValues({start: 10, end: 20});
    await settle(fixture);
    fixture.componentRef.setInput('position', 33);
    fixture.componentRef.setInput('analysis', {...analysis, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogitsVector.mock.lastCall?.[1]).toMatchObject({
      ref_step: 1,
      target_step: 1,
      transform: 'raw',
    });
    expect(api.finalLogits.mock.lastCall?.[1].offset).toBe(0);
    expect(view.selectedIndex()).toBeNull();
    expect(view.elementRange()).toEqual(view.elementFull());
    view.locateElement(65);
    view.zoomValues({start: 10, end: 20});
    await settle(fixture);
    captureId.set('another-capture');
    await settle(fixture);
    expect(api.finalLogitsVector.mock.lastCall?.[0]).toBe('another-capture');
    expect(view.selectedIndex()).toBeNull();
    expect(view.settings().logitOffset).toBe(0);
    expect(view.settings().logitElementRange).toBeUndefined();
    expect(view.settings().logitValueRange).toBeUndefined();
    fixture.destroy();
  });

  it('aborts obsolete vector requests and ignores their late result after the new source is visible', async () => {
    const pending: {
      query: FinalLogitsVectorRequest;
      signal: AbortSignal;
      resolve: (value: FinalLogitsVectorResponse) => void;
    }[] = [];
    const {fixture, view} = create({
      readVector: (_capture, query, signal) =>
        new Promise((resolve) => pending.push({query, signal, resolve})),
    });
    expect(view.loading()).toBe(true);
    fixture.componentRef.setInput('position', 33);
    fixture.componentRef.setInput('analysis', {...analysis, ref_step: 1, target_step: 1});
    fixture.detectChanges();
    expect(pending).toHaveLength(2);
    expect(pending[0].signal.aborted).toBe(true);
    const current = vectorResponse(pending[1].query);
    current.values.target![0] = 1234;
    pending[1].resolve(current);
    await settle(fixture);
    expect(view.channels()[0].target).toBe(1234);
    pending[0].resolve(vectorResponse(pending[0].query));
    await settle(fixture);
    expect(view.data()?.ref_step).toBe(1);
    expect(view.channels()[0].target).toBe(1234);
    expect(view.loading()).toBe(false);
    fixture.destroy();
    expect(pending[1].signal.aborted).toBe(true);
  });

  it('retains one-sided values in Line while Scatter explains that a paired plot is unavailable', async () => {
    const reason = 'Reference logits unavailable.';
    const {fixture, view, api} = create({
      pair: {...analysis, ref_step: null},
      readVector: async (_capture, query) => ({
        ...vectorResponse(query, 3),
        compatible: false,
        reason,
        ref: {size: 0, reason},
        values: {ref: null, target: [2, 'Infinity', -3]},
      }),
      read: async (_capture, query) => ({
        ...response(query, 3),
        compatible: false,
        reason,
        rows: [2, 'Infinity', -3].map((target, index) => ({
          index,
          label: null,
          ref: null,
          target,
          delta: null,
          absolute_delta: null,
        })),
      }),
    });
    await settle(fixture);
    expect(
      view
        .lineChart()
        ?.rows()
        .map((row) => row.target),
    ).toEqual([2, 'Infinity', -3]);
    expect(view.channels().every((row) => row.ref === null && row.delta === null)).toBe(true);
    expect(view.lineChart()?.comparisonAllowed()).toBe(false);
    expect(view.formatValue(view.rows()[1].target)).toBe('Infinity');
    expect(view.pages()).toBe(1);
    expect(view.pageIndex()).toBe(0);
    view.chooseView('Tensor Inspection · Scatter plot');
    await settle(fixture);
    expect(view.scatterChart()?.rows()).toEqual([]);
    expect(view.scatterChart()?.messages()).toContain(reason);
    expect(fixture.nativeElement.textContent).toContain(reason);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fixture.componentRef.setInput('analysis', undefined);
    await settle(fixture);
    expect(view.data()).toBeNull();
    expect(view.channels()).toEqual([]);
    expect(fixture.nativeElement.textContent).toContain('No captured logits selection');
    fixture.destroy();
  });

  it('keeps the complete chart available when a table page fails and recovers the page on retry', async () => {
    let fail = true;
    const {fixture, view, api} = create({
      read: async (_capture, query) => {
        if (fail && query.offset === 16) throw new Error('Values page could not be read.');
        return response(query);
      },
    });
    await settle(fixture);
    const channels = view.channels();
    view.changePage(1);
    await settle(fixture);
    expect(view.valuesError()).toContain('Values page could not be read');
    expect(view.valuesLoading()).toBe(false);
    expect(view.loading()).toBe(false);
    expect(view.error()).toBe('');
    expect(view.channels()).toBe(channels);
    expect(view.lineChart()?.rows()).toBe(channels);
    expect(api.finalLogitsVector).toHaveBeenCalledTimes(1);
    fail = false;
    view.retry();
    await settle(fixture);
    expect(view.valuesError()).toBe('');
    expect(view.rows()[0].index).toBe(16);
    expect(view.channels()).toHaveLength(70);
    fixture.destroy();
  });
});
