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
  FinalLogitsAddress,
  FinalLogitsBins,
  FinalLogitsBinsRequest,
  FinalLogitsBinsResponse,
  FinalLogitsPointsRequest,
  FinalLogitsPointsResponse,
  FinalLogitsRequest,
  FinalLogitsResponse,
  FinalLogitsSummaryResponse,
  FinalLogitsZoomedPointsResponse,
} from '../../server/contracts/final_logits';
import type {KvHeadChannel, KvRangeBin} from '../../server/contracts/kv';
import type {Reason} from '../../server/contracts/metrics';
import {
  channelBins,
  channelRepresentative,
  type ChannelReduction,
} from '../../widgets/charts/channel_bins';
import {
  pairedChannels,
  scatterExtent,
  scatterRepresentatives,
  SCATTER_POINT_LIMIT,
} from '../../widgets/charts/scatter_model';
import type {TokenAnalysisPair} from '../../server/contracts/token_analysis';
import {FinalLogitsApi} from '../../server/final_logits_api';
import {CaptureStore} from '../../server/capture_store';
import type {LogitPanelState} from '../shared/logit_panel_state';
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
      if (saved.descriptor) {
        Object.defineProperty(saved.prototype, saved.key, saved.descriptor);
      } else {
        Reflect.deleteProperty(saved.prototype, saved.key);
      }
    }
  });
  const initial: LogitPanelState = {
    open: true,
    metric: 'logit_rmse',
    logitView: 'inspection',
    logitPlot: 'index',
    logitTransform: 'raw',
    logitOrder: 'index',
    logitOffset: 0,
  };
  const emptyMetric = (): {value: number | null; reasons: Reason[]} => ({
    value: null,
    reasons: [],
  });
  const analysis: TokenAnalysisPair = {
    stage: 'decode',
    alignment: 'content',
    row: 0,
    first_divergence: false,
    ref_step: 0,
    target_step: 0,
    metrics: {
      kl: emptyMetric(),
      js: emptyMetric(),
      relative_l2: emptyMetric(),
      cosine_similarity: emptyMetric(),
      cosine_distance: emptyMetric(),
      norm_ratio: emptyMetric(),
      max_abs: emptyMetric(),
    },
    distribution: {compatible: true, reasons: [], ref: {size: 70}, target: {size: 70}, rows: []},
  };
  function response(query: FinalLogitsRequest, total = 70): FinalLogitsResponse {
    return {
      ...query,
      first_divergence: !!query.first_divergence,
      // The fake comparison: row k pairs steps (k, k).
      ref_step: query.row,
      target_step: query.row,
      compatible: true,
      reasons: [],
      total,
      ref: {size: total},
      target: {size: total},
      metrics: {
        rmse: {value: 1, reasons: []},
        max_abs: {value: 1, reasons: []},
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
  /** The fake server's whole vectors of a row: what it is asked about and answers from. */
  interface Vectors extends FinalLogitsSummaryResponse {
    values: {
      ref: Array<number | string | null> | null;
      target: Array<number | string | null> | null;
    };
  }
  /** Ref i and target i + 1 at vocabulary index i, `total` long. */
  function vectors(query: FinalLogitsAddress, total = 70): Vectors {
    const {
      offset: _offset,
      limit: _limit,
      rows: _rows,
      ...metadata
    } = response({...query, offset: 0, limit: 16, order: 'index'}, total);
    const stat = (shift: number) => ({
      min: shift,
      max: total - 1 + shift,
      mean: (total - 1) / 2 + shift,
      min_position: 0,
      max_position: total - 1,
      valid_count: total,
    });
    const one = {...stat(0), min: 1, max: 1, mean: 1};
    return {
      ...metadata,
      values: {
        ref: Array.from({length: total}, (_, index) => index),
        target: Array.from({length: total}, (_, index) => index + 1),
      },
      statistics: {ref: stat(0), target: stat(1), delta: one, abs_delta: one},
    };
  }
  /** The elements of both vectors, as the charts read them. */
  function elements(source: Vectors): KvHeadChannel[] {
    return Array.from({length: source.total}, (_, channel) => ({
      channel,
      ref: source.values.ref?.[channel] ?? null,
      target: source.values.target?.[channel] ?? null,
      delta: null,
      abs_delta: null,
      status: 'available',
    }));
  }
  /** What the server answers, made with the UI's own reductions as the server makes them. */
  function summaryOf(source: Vectors): FinalLogitsSummaryResponse {
    const {values: _values, ...summary} = source;
    return summary;
  }
  function binsOf(source: Vectors, payload: FinalLogitsBinsRequest): FinalLogitsBinsResponse {
    const rows = elements(source);
    const length = Math.max(1, source.total);
    const start =
      payload.start === undefined ? 0 : Math.max(0, Math.min(length - 1, payload.start));
    const end =
      payload.end === undefined ? length : Math.max(start + 1, Math.min(length, payload.end));
    const listed = (range: KvRangeBin): FinalLogitsBins => {
      const bins = channelBins(rows, range, payload.capacity, payload.reduction);
      const represented = (side: 'ref' | 'target') =>
        bins.map((bin) =>
          channelRepresentative(
            rows.filter((row) => row.channel >= bin.start && row.channel < bin.end),
            bin,
            payload.reduction,
            side,
          ),
        );
      return {
        start: range.start,
        end: range.end,
        size: bins[0].end - bins[0].start,
        ref: bins.map((bin) => bin.ref),
        target: bins.map((bin) => bin.target),
        ref_index: represented('ref'),
        target_index: represented('target'),
      };
    };
    return {
      ...summaryOf(source),
      range: {start, end},
      capacity: payload.capacity,
      reduction: payload.reduction,
      overview: listed({start: 0, end: length}),
      visible: listed({start, end}),
    };
  }
  function pointsOf(
    source: Vectors,
    payload: FinalLogitsAddress | FinalLogitsPointsRequest,
  ): FinalLogitsPointsResponse {
    const paired = pairedChannels(source.compatible ? elements(source) : []);
    const extent = scatterExtent([paired]);
    const range = 'start' in payload ? {start: payload.start, end: payload.end} : extent;
    const drawn = scatterRepresentatives(paired, range, null, SCATTER_POINT_LIMIT, true);
    const runs: Array<[number, number]> = [];
    for (const row of paired) {
      if (runs.at(-1)?.[1] === row.channel) {
        runs.at(-1)![1]++;
      } else {
        runs.push([row.channel, row.channel + 1]);
      }
    }
    return {
      ...summaryOf(source),
      extent,
      range,
      paired_runs: runs,
      paired: drawn.total,
      visible: drawn.visible,
      grouped: drawn.grouped,
      points: {
        index: drawn.rows.map((row) => row.channel),
        ref: drawn.rows.map((row) => row.ref),
        target: drawn.rows.map((row) => row.target),
      },
    };
  }
  type Read = (
    capture: string | null,
    query: FinalLogitsRequest,
    signal: AbortSignal,
  ) => Promise<FinalLogitsResponse>;
  type ReadVectors = (
    capture: string | null,
    query: FinalLogitsAddress,
    signal: AbortSignal,
  ) => Promise<Vectors>;
  function create(
    options: {
      read?: Read;
      readVectors?: ReadVectors;
      readSummary?: (
        capture: string | null,
        query: FinalLogitsAddress,
        signal: AbortSignal,
      ) => Promise<FinalLogitsSummaryResponse>;
      readBins?: (
        capture: string | null,
        query: FinalLogitsBinsRequest,
        signal: AbortSignal,
      ) => Promise<FinalLogitsBinsResponse>;
      settings?: LogitPanelState;
      pair?: TokenAnalysisPair | null;
      measured?: boolean;
    } = {},
  ) {
    const readVectors: ReadVectors =
      options.readVectors ?? (async (_capture, query) => vectors(query));
    const api = {
      finalLogits: vi.fn(options.read ?? (async (_capture, query) => response(query))),
      finalLogitsSummary: vi.fn(
        options.readSummary ??
          (async (capture: string | null, query: FinalLogitsAddress, signal: AbortSignal) =>
            summaryOf(await readVectors(capture, query, signal))),
      ),
      finalLogitsBins: vi.fn(
        options.readBins ??
          (async (capture: string | null, query: FinalLogitsBinsRequest, signal: AbortSignal) =>
            binsOf(await readVectors(capture, query, signal), query)),
      ),
      finalLogitsPoints: vi.fn(
        async (capture: string | null, query: FinalLogitsAddress, signal: AbortSignal) =>
          pointsOf(await readVectors(capture, query, signal), query),
      ),
      // As the server: what no range changes comes with the whole extent's points alone.
      finalLogitsZoomedPoints: vi.fn(
        async (
          capture: string | null,
          query: FinalLogitsPointsRequest,
          signal: AbortSignal,
        ): Promise<FinalLogitsZoomedPointsResponse> => {
          const {
            extent: _extent,
            paired_runs: _runs,
            ...zoomed
          } = pointsOf(await readVectors(capture, query, signal), query);
          return zoomed;
        },
      ),
    };
    const captureId = signal<string | null>('capture');
    TestBed.configureTestingModule({
      providers: [
        {provide: FinalLogitsApi, useValue: api},
        {provide: CaptureStore, useValue: {captureId}},
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
    // The line chart measures its width (ResizeObserver is a stub here).
    if (options.measured !== false) {
      fixture.componentInstance.capacityMeasured.set(true);
    }
    const changes: LogitPanelState[] = [];
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

  it('passes on the server statistics of the vectors it reads', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    expect(view.statistics()).toEqual(view.summaryRequest.data()?.statistics);
    expect(view.statistics()?.target.max).toBe(70);
    fixture.destroy();
  });

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
    expect(
      [
        ...api.finalLogitsSummary.mock.calls,
        ...api.finalLogitsPoints.mock.calls,
        ...api.finalLogitsZoomedPoints.mock.calls,
        ...api.finalLogitsBins.mock.calls,
      ].every((call) => call[1].transform === 'raw'),
    ).toBe(true);
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
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
      const settings: LogitPanelState = {
        ...initial,
        logitOrder: column === 'index' ? 'target' : 'index',
        logitSortDirection: 'descending',
        logitOffset: 32,
        logitElementRange: {start: 16, end: 48},
      };
      const {fixture, view, api} = create({settings});
      await settle(fixture);
      const bins = view.lineBins();
      const metrics = view.data()!.metrics;
      const binReads = api.finalLogitsBins.mock.calls.length;
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
      expect(view.lineBins()).toBe(bins);
      expect(view.lineChart()?.bins()).toBe(bins);
      expect(api.finalLogitsBins).toHaveBeenCalledTimes(binReads);
      expect(view.data()?.metrics).toBe(metrics);
      expect(view.elementRange()).toEqual(settings.logitElementRange);
      expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
    fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogits.mock.lastCall?.[1]).toMatchObject({
      order: 'target',
      direction: 'descending',
      offset: 0,
      row: 1,
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
        if (query.order === 'index' && query.direction === 'descending') {
          page.rows = page.rows.map((row) => ({...row, index: page.total - 1 - row.index}));
        }
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
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
      const bins = view.lineBins();
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
      expect(view.lineBins()).toBe(bins);
      expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('keeps the complete chart vector independent of the 16-row auxiliary table pages', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    const bins = view.lineBins();
    expect(view.selectedIndex()).toBeNull();
    expect(view.lineChart()?.bins()).toBe(bins);
    // 70 elements in 128 bins: one element a bin, each its own representative.
    expect(bins?.visible).toHaveLength(70);
    expect(bins?.visible.at(-1)).toEqual({
      start: 69,
      end: 70,
      ref: 69,
      target: 70,
      refIndex: 69,
      targetIndex: 69,
    });
    expect(api.finalLogitsBins.mock.lastCall?.[1]).toEqual({
      turn: 2,
      stage: 'decode',
      alignment: 'content',
      row: 0,
      transform: 'raw',
      capacity: 128,
      reduction: 'mean',
    });
    expect(view.rows()).toHaveLength(16);
    expect(view.pages()).toBe(5);
    expect(fixture.nativeElement.querySelector('channel-chart')).not.toBeNull();
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
    expect(view.lineBins()).toBe(bins);
    expect(view.lineChart()?.bins()).toBe(bins);
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);

    view.changePage(99);
    await settle(fixture);
    expect(view.rows().map((row) => row.index)).toEqual([64, 65, 66, 67, 68, 69]);
    expect(view.pageIndex()).toBe(4);
    const pageReads = api.finalLogits.mock.calls.length;
    view.changePage(1);
    await settle(fixture);
    expect(api.finalLogits).toHaveBeenCalledTimes(pageReads);
    expect(view.lineBins()).toBe(bins);
    expect(api.finalLogitsBins).toHaveBeenCalledTimes(1);
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('routes Line and Scatter through View, asking each chart once and the page and summary never again', async () => {
    const {fixture, view, api, changes} = create();
    await settle(fixture);
    const bins = view.lineBins();
    expect(api.finalLogitsPoints).not.toHaveBeenCalled();
    view.chooseView('Tensor Inspection · Scatter plot');
    await settle(fixture);
    expect(api.finalLogitsPoints).toHaveBeenCalledTimes(1);
    const scatter = view.scatterChart()!.view()!;
    expect(scatter.full).toEqual(
      scatterExtent([pairedChannels(elements(vectors({...analysis, turn: 2, transform: 'raw'})))]),
    );
    expect(scatter.visible.rows.map((row) => row.channel)).toEqual(
      Array.from({length: 70}, (_, index) => index),
    );
    expect(scatter.overview).toEqual(scatter.visible.rows);
    expect(scatter.paired).toEqual([{start: 0, end: 70}]);
    expect(view.scatterChart()?.waiting()).toBe(false);
    expect(view.scatterChart()?.range()).toEqual(scatter.full);
    expect(view.scatterChart()?.singlePlane()).toBe(true);
    expect(view.scatterChart()?.dimensionLabel()).toBe('Vocabulary index');
    expect(fixture.nativeElement.querySelector('channel-chart')).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[aria-label="Value Range controls"]'),
    ).not.toBeNull();
    view.chooseView('Tensor Inspection · Line chart');
    await settle(fixture);
    expect(view.lineChart()?.bins()).toBe(bins);
    expect(api.finalLogitsBins).toHaveBeenCalledTimes(1);
    expect(fixture.nativeElement.querySelector('scatter-chart')).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[aria-label="Element Range controls"]'),
    ).not.toBeNull();
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
        reasons:
          query.order === 'absolute_delta'
            ? [{code: 'unavailable', side: null, text: fallback}]
            : [],
      }),
    });
    await settle(fixture);
    const metrics = view.data()!.metrics;
    view.sortBy('absolute_delta');
    await settle(fixture);
    expect(view.pageValuesNotices()).toEqual([fallback]);
    expect(view.notices()).toEqual([]);
    expect(view.data()?.reasons).toEqual([]);
    expect(view.data()?.metrics).toBe(metrics);
    expect(fixture.nativeElement.textContent).not.toContain(fallback);
    view.sortBy('index');
    await settle(fixture);
    expect(view.pageValuesNotices()).toEqual([]);
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });

  it('preserves saved page and plot ranges when the selected analysis initially resolves', async () => {
    const settings: LogitPanelState = {
      ...initial,
      logitOffset: 32,
      logitPlot: 'scatter',
      logitElementRange: {start: 16, end: 48},
      logitValueRange: {start: -2, end: 12},
    };
    const {fixture, view, api, changes} = create({pair: null, settings});
    await settle(fixture);
    expect(api.finalLogitsSummary).not.toHaveBeenCalled();
    expect(api.finalLogitsPoints).not.toHaveBeenCalled();
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
      if (source === 'capture') {
        captureId.set('another-capture');
      } else {
        fixture.componentRef.setInput(source, source === 'turn' ? 3 : 33);
      }
      await settle(fixture);
      expect(view.settings().logitOffset).toBe(0);
      expect(view.settings().logitElementRange).toBeUndefined();
      expect(view.settings().logitValueRange).toBeUndefined();
      fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
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
    fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogitsSummary.mock.lastCall?.[1]).toMatchObject({row: 1});
    expect(api.finalLogitsBins.mock.lastCall?.[1]).toMatchObject({row: 1});
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
      row: 0,
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
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
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
    fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
    await settle(fixture);
    expect(api.finalLogitsSummary.mock.lastCall?.[1]).toMatchObject({
      row: 1,
      transform: 'raw',
    });
    expect(api.finalLogitsBins.mock.lastCall?.[1]).toEqual({
      ...api.finalLogitsSummary.mock.lastCall?.[1],
      capacity: 16,
      reduction: 'mean',
    });
    expect(api.finalLogits.mock.lastCall?.[1].offset).toBe(0);
    expect(view.selectedIndex()).toBeNull();
    expect(view.elementRange()).toEqual(view.elementFull());
    view.locateElement(65);
    view.zoomValues({start: 10, end: 20});
    await settle(fixture);
    captureId.set('another-capture');
    await settle(fixture);
    expect(api.finalLogitsSummary.mock.lastCall?.[0]).toBe('another-capture');
    expect(api.finalLogitsBins.mock.lastCall?.[0]).toBe('another-capture');
    expect(view.selectedIndex()).toBeNull();
    expect(view.settings().logitOffset).toBe(0);
    expect(view.settings().logitElementRange).toBeUndefined();
    expect(view.settings().logitValueRange).toBeUndefined();
    fixture.destroy();
  });

  it('aborts obsolete summary requests and ignores their late result after the new source is visible', async () => {
    const pending: {
      query: FinalLogitsAddress;
      signal: AbortSignal;
      resolve: (value: FinalLogitsSummaryResponse) => void;
    }[] = [];
    const {fixture, view} = create({
      readSummary: (_capture, query, signal) =>
        new Promise((resolve) => pending.push({query, signal, resolve})),
    });
    expect(view.loading()).toBe(true);
    fixture.componentRef.setInput('position', 33);
    fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
    fixture.detectChanges();
    expect(pending).toHaveLength(2);
    expect(pending[0].signal.aborted).toBe(true);
    const current = summaryOf(vectors(pending[1].query));
    current.statistics.target.max = 1234;
    pending[1].resolve(current);
    await settle(fixture);
    expect(view.statistics()?.target.max).toBe(1234);
    pending[0].resolve(summaryOf(vectors(pending[0].query)));
    await settle(fixture);
    expect(view.data()?.ref_step).toBe(1);
    expect(view.statistics()?.target.max).toBe(1234);
    expect(view.loading()).toBe(false);
    fixture.destroy();
    expect(pending[1].signal.aborted).toBe(true);
  });

  it('asks for the line bins once the chart knows its width, and keeps the drawing while a zoom loads', async () => {
    const pending: {query: FinalLogitsBinsRequest; resolve: () => void}[] = [];
    const {fixture, view, api} = create({
      measured: false,
      readBins: (_capture, query) =>
        new Promise((resolve) =>
          pending.push({query, resolve: () => resolve(binsOf(vectors(query), query))}),
        ),
    });
    await settle(fixture);
    expect(api.finalLogitsBins).not.toHaveBeenCalled();
    expect(view.lineChart()?.waiting()).toBe(true);
    view.capacity.set(16);
    view.capacityMeasured.set(true);
    await settle(fixture);
    expect(pending.map((call) => call.query.capacity)).toEqual([16]);
    pending[0].resolve();
    await settle(fixture);
    const bins = view.lineBins();
    expect(bins?.visible).toHaveLength(14);
    expect(view.lineChart()?.waiting()).toBe(false);
    view.zoomElements({start: 10, end: 20});
    await settle(fixture);
    expect(pending[1].query).toMatchObject({start: 10, end: 20, capacity: 16});
    // The previous drawing stays, with its range, until the answer arrives.
    expect(view.lineChart()?.waiting()).toBe(true);
    expect(view.lineBins()).toBe(bins);
    expect(view.lineRange()).toEqual({start: 0, end: 70});
    pending[1].resolve();
    await settle(fixture);
    expect(view.lineChart()?.waiting()).toBe(false);
    expect(view.lineRange()).toEqual({start: 10, end: 20});
    expect(view.lineBins()?.visible.map((bin) => [bin.start, bin.end])).toEqual(
      Array.from({length: 10}, (_, index) => [10 + index, 11 + index]),
    );
    fixture.destroy();
  });

  it('reads a drawn Scatter element from its point, and looks up one no point holds', async () => {
    const {fixture, view, api} = create({
      settings: {...initial, logitPlot: 'scatter', logitValueRange: {start: 0, end: 5}},
    });
    await settle(fixture);
    expect(api.finalLogitsPoints.mock.calls.map((call) => call[1])).toEqual([
      expect.not.objectContaining({start: expect.anything()}),
    ]);
    expect(api.finalLogitsZoomedPoints.mock.calls.map((call) => call[1])).toEqual([
      expect.objectContaining({start: 0, end: 5}),
    ]);
    expect(view.scatterChart()?.range()).toEqual({start: 0, end: 5});
    const reads = api.finalLogits.mock.calls.length;
    // 70 points are all drawn: each one's values are known.
    view.scatterChart()!.select.emit(3);
    view.locateElement(40);
    await settle(fixture);
    expect(api.finalLogits.mock.calls.slice(reads).map((call) => call[1].limit)).toEqual([16]);
    expect(view.scatterChart()?.view()?.elements.get(40)).toMatchObject({ref: 40, target: 41});
    // Located outside the value range: the range is reset to show it.
    expect(view.settings().logitValueRange).toBeUndefined();
    fixture.destroy();
  });

  it('says why vectors read after the divergence still compare', async () => {
    const note = {
      code: 'context_different',
      side: null,
      text: 'After the divergence at position 17 the two sides read different tokens; the logits still compare.',
    };
    const {fixture, view} = create({
      readVectors: async (_capture, query) => ({...vectors(query), notes: [note]}),
    });
    await settle(fixture);
    expect(view.notices()).toEqual([note.text]);
    fixture.destroy();
  });

  it('looks up the values of a located element when no point holds them', async () => {
    const reason = 'Vocabularies differ.';
    const different = {code: 'vocabulary_mismatch', side: null, text: reason};
    const {fixture, view, api} = create({
      settings: {...initial, logitPlot: 'scatter', logitValueRange: {start: 0, end: 5}},
      readVectors: async (_capture, query) => ({
        ...vectors(query),
        compatible: false,
        reasons: [different],
      }),
    });
    await settle(fixture);
    expect(view.scatterChart()?.view()?.visible.rows).toEqual([]);
    const reads = api.finalLogits.mock.calls.length;
    view.locateElement(40);
    await settle(fixture);
    expect(api.finalLogits.mock.calls.slice(reads).map((call) => call[1])).toContainEqual(
      expect.objectContaining({offset: 40, limit: 1, order: 'index', direction: 'ascending'}),
    );
    // Not paired, so not drawn; still outside the range, which is reset as before.
    expect(view.scatterChart()?.view()?.elements.has(40)).toBe(false);
    expect(view.settings().logitValueRange).toBeUndefined();
    fixture.destroy();
  });

  it('retains one-sided values in Line while Scatter explains that a paired plot is unavailable', async () => {
    const reason = 'Reference logits unavailable.';
    const missing = {code: 'token_not_captured', side: null, text: reason};
    const {fixture, view, api} = create({
      pair: {...analysis, ref_step: null},
      readVectors: async (_capture, query) => ({
        ...vectors(query, 3),
        compatible: false,
        reasons: [missing],
        ref: {size: 0, reasons: [missing]},
        values: {ref: null, target: [2, 'Infinity', -3]},
      }),
      read: async (_capture, query) => ({
        ...response(query, 3),
        compatible: false,
        reasons: [missing],
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
    // A value that is not finite is a gap; the side without logits has only gaps.
    expect(
      view
        .lineChart()
        ?.bins()
        ?.visible.map((bin) => bin.target),
    ).toEqual([2, null, -3]);
    expect(view.lineBins()?.visible.every((bin) => bin.ref === null && bin.refIndex === null)).toBe(
      true,
    );
    expect(view.lineChart()?.comparisonAllowed()).toBe(false);
    expect(view.formatValue(view.rows()[1].target)).toBe('Infinity');
    expect(view.pages()).toBe(1);
    expect(view.pageIndex()).toBe(0);
    view.chooseView('Tensor Inspection · Scatter plot');
    await settle(fixture);
    expect(view.scatterChart()?.view()?.visible.rows).toEqual([]);
    expect(view.scatterChart()?.view()?.full).toEqual({start: -1, end: 1});
    expect(view.scatterChart()?.messages()).toContain(reason);
    expect(fixture.nativeElement.textContent).toContain(reason);
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
    fixture.componentRef.setInput('analysis', undefined);
    await settle(fixture);
    expect(view.data()).toBeNull();
    expect(view.lineBins()).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('No captured logits selection');
    fixture.destroy();
  });

  it('keeps the complete chart available when a table page fails and recovers the page on retry', async () => {
    let fail = true;
    const {fixture, view, api} = create({
      read: async (_capture, query) => {
        if (fail && query.offset === 16) {
          throw new Error('Values page could not be read.');
        }
        return response(query);
      },
    });
    await settle(fixture);
    const bins = view.lineBins();
    view.changePage(1);
    await settle(fixture);
    expect(view.valuesError()).toContain('Values page could not be read');
    expect(view.valuesLoading()).toBe(false);
    expect(view.loading()).toBe(false);
    expect(view.error()).toBe('');
    expect(view.lineBins()).toBe(bins);
    expect(view.lineChart()?.bins()).toBe(bins);
    expect(api.finalLogitsSummary).toHaveBeenCalledTimes(1);
    fail = false;
    view.retry();
    await settle(fixture);
    expect(view.valuesError()).toBe('');
    expect(view.rows()[0].index).toBe(16);
    expect(view.lineBins()?.visible).toHaveLength(70);
    fixture.destroy();
  });

  it('shows that the first drawing is loading instead of empty axes, and that the pair is', async () => {
    const pending: (() => void)[] = [];
    const {fixture, view} = create({
      pair: null,
      readBins: (_capture, query) =>
        new Promise((resolve) => pending.push(() => resolve(binsOf(vectors(query), query)))),
    });
    fixture.componentRef.setInput('analysisLoading', true);
    await settle(fixture);
    const text = () => fixture.nativeElement.querySelector('.results').textContent;
    expect(text()).toContain('Loading captured logits…');
    expect(text()).not.toContain('No captured logits selection');
    fixture.componentRef.setInput('analysisLoading', false);
    fixture.componentRef.setInput('analysis', analysis);
    await settle(fixture);
    // The chart is laid out, unseen, to measure its width; the note stands in its place.
    const chart = fixture.nativeElement.querySelector('channel-chart') as HTMLElement;
    expect(chart.classList).toContain('chart-pending');
    expect(fixture.nativeElement.querySelector('.chart-state')?.textContent).toContain(
      'Loading captured logits…',
    );
    pending[0]();
    await settle(fixture);
    expect(chart.classList).not.toContain('chart-pending');
    expect(fixture.nativeElement.querySelector('.chart-state')).toBeNull();
    // A later view keeps the drawing while it loads, without the note.
    view.zoomElements({start: 10, end: 20});
    await settle(fixture);
    expect(view.lineChart()?.waiting()).toBe(true);
    expect(fixture.nativeElement.querySelector('.chart-state')).toBeNull();
    fixture.destroy();
  });

  it('clears the error of a failed view when the view goes back to an answered one, and retries', async () => {
    let fail = true;
    const {fixture, view, api} = create({
      readBins: async (_capture, query) => {
        if (fail && query.start === 10) {
          throw new Error('Bins could not be read.');
        }
        return binsOf(vectors(query), query);
      },
    });
    await settle(fixture);
    const bins = view.lineBins();
    view.zoomElements({start: 10, end: 20});
    await settle(fixture);
    expect(view.drawingError()).toBe('Bins could not be read.');
    view.resetRange();
    await settle(fixture);
    expect(view.drawingError()).toBe('');
    expect(view.lineChart()?.waiting()).toBe(false);
    expect(view.lineBins()).toBe(bins);
    view.zoomElements({start: 10, end: 20});
    await settle(fixture);
    expect(view.drawingError()).toBe('Bins could not be read.');
    fail = false;
    view.retry();
    await settle(fixture);
    expect(view.drawingError()).toBe('');
    expect(view.lineRange()).toEqual({start: 10, end: 20});
    expect(api.finalLogitsBins.mock.calls.map((call) => call[1].start)).toEqual([
      undefined,
      10,
      10,
      10,
    ]);
    fixture.destroy();
  });

  it('asks once where a resize, a drag of the value range or held arrow keys come to rest', async () => {
    const {fixture, view, api} = create({
      settings: {...initial, logitPlot: 'scatter'},
      readVectors: async (_capture, query) => ({
        ...vectors(query),
        // Two points paired: the others are known from the table page alone.
        values: {
          ref: Array.from({length: 70}, (_, index) => (index < 2 ? index : null)),
          target: Array.from({length: 70}, (_, index) => index + 1),
        },
      }),
    });
    await settle(fixture);
    for (const start of [1, 2, 3, 4]) {
      view.zoomValues({start, end: start + 10});
    }
    await settle(fixture);
    expect(api.finalLogitsZoomedPoints.mock.calls.map((call) => call[1].start)).toEqual([4]);
    const reads = api.finalLogits.mock.calls.length;
    view.changePage(4);
    await settle(fixture);
    for (const index of [64, 65, 66, 67]) {
      view.showElement(index);
    }
    await settle(fixture);
    // Each is on the table page: its values are known, nothing is looked up.
    expect(api.finalLogits.mock.calls.slice(reads).map((call) => call[1].limit)).toEqual([16]);
    expect(
      view
        .scatterChart()
        ?.view()
        ?.visible.rows.map((row) => row.channel),
    ).toEqual([0, 1]);
    expect(view.scatterChart()?.view()?.elements.get(67)).toMatchObject({ref: 67, target: 68});
    view.chooseView('Tensor Inspection · Line chart');
    await settle(fixture);
    const asked = api.finalLogitsBins.mock.calls.length;
    for (const capacity of [100, 110, 120]) {
      view.capacity.set(capacity);
      fixture.detectChanges();
    }
    await settle(fixture);
    expect(api.finalLogitsBins.mock.calls.slice(asked).map((call) => call[1].capacity)).toEqual([
      120,
    ]);
    fixture.destroy();
  });

  it('asks a new token position for its whole vocabulary once, not first for the old range', async () => {
    const {fixture, view, api} = create({
      settings: {...initial, logitElementRange: {start: 10, end: 20}},
    });
    await settle(fixture);
    const asked = api.finalLogitsBins.mock.calls.length;
    view.setPosition(33);
    fixture.componentRef.setInput('analysis', {...analysis, row: 1, ref_step: 1, target_step: 1});
    await settle(fixture);
    const calls = api.finalLogitsBins.mock.calls.slice(asked).map((call) => call[1]);
    expect(calls).toEqual([expect.objectContaining({row: 1})]);
    expect(calls[0]).not.toHaveProperty('start');
    fixture.destroy();
  });
});
