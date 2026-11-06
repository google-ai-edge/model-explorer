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
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {FinalLogitsLineChart, type LogitPositionSeries} from './final_logits_line_chart';
import {RangeChart} from '../../widgets/charts/range_chart';
import type {PlotHost, PlotlyApi} from '../../widgets/plotly_loader';
import type {LogitPanelState} from '../shared/logit_panel_state';
import type {RangeSelection} from '../../widgets/charts/range_selection';
import {CaptureStore} from '../../server/capture_store';
import {TokenAnalysisApi} from '../../server/token_analysis_api';
import type {
  TokenRangeCell,
  TokenRangeRequest,
  TokenRangeResponse,
} from '../../server/contracts/token_analysis';

/** A cell of no value. */
const EMPTY = {min: null, max: null, mean: null, min_position: null, max_position: null};

/** The server's answer of a request: its bins as the server splits the range, each series'
 *  cells from `cells` (per series, by bin start; empty elsewhere), `all` from `all`. */
function answer(
  request: TokenRangeRequest,
  cells: Record<number, Partial<TokenRangeCell>>[],
  all: Record<number, Partial<TokenRangeCell>> = {},
): TokenRangeResponse {
  const count = Math.max(1, Math.min(request.bins, request.end - request.start));
  const bins = Array.from({length: count}, (_, i) => ({
    start: request.start + Math.floor((i * (request.end - request.start)) / count),
    end: request.start + Math.floor(((i + 1) * (request.end - request.start)) / count),
  }));
  const cell = (given: Record<number, Partial<TokenRangeCell>>, bin: (typeof bins)[number]) => ({
    ...bin,
    ...EMPTY,
    valid_count: 0,
    ...given[bin.start],
  });
  return {
    turn: request.turn,
    stage: request.stage,
    alignment: request.alignment,
    analysis: request.analysis,
    metric: request.metric,
    start: request.start,
    end: request.end,
    bins,
    rows: cells.map((given, index) => ({
      side: cells.length === 2 ? (['ref', 'target'] as const)[index] : null,
      bins: bins.map((bin) => cell(given, bin)),
    })),
    all: bins.map((bin) => cell(all, bin)),
    metadata: {
      ref: {vocabulary_sizes: []},
      target: {vocabulary_sizes: []},
      output: {
        dimensions: ['Position'],
        shape: [request.end - request.start],
        dtype: 'float64',
      },
    },
    peak_kl: null,
  };
}

describe('FinalLogitsLineChart display settings', () => {
  // What the fake server answers each request with: by default, every bin empty.
  let serve: (request: TokenRangeRequest) => TokenRangeResponse;
  let requests: TokenRangeRequest[];
  beforeEach(() => {
    requests = [];
    serve = (request) => answer(request, [{}]);
    TestBed.configureTestingModule({
      providers: [
        {provide: CaptureStore, useValue: {captureId: signal('capture')}},
        {
          provide: TokenAnalysisApi,
          useValue: {
            tokenRange: async (_capture: string, request: TokenRangeRequest) => {
              requests.push(request);
              return serve(request);
            },
          },
        },
      ],
    });
    // Match the browser API stubs used by the KV chart tests; keep compiled inputs intact.
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
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  const denseSeries: LogitPositionSeries[] = [
    {
      name: 'Logits',
      color: '#4285f4',
      points: Array.from({length: 320}, (_, index) => ({position: index, index})),
    },
  ];
  const settings: LogitPanelState = {
    open: true,
    metric: 'kl',
    positionRange: {start: 0, end: 320},
  };

  function create(initial = settings) {
    // Keep compiled signal-input metadata; no detectChanges is needed for these state tests.
    const fixture = TestBed.createComponent(FinalLogitsLineChart);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('alignment', 'content');
    fixture.componentRef.setInput('analysis', 'final_logits');
    fixture.componentRef.setInput('analysisLabel', 'Logits comparison');
    fixture.componentRef.setInput('metrics', [{id: 'kl', label: 'KL'}]);
    fixture.componentRef.setInput('metricLabel', 'KL (nats)');
    fixture.componentRef.setInput('settings', initial);
    fixture.componentRef.setInput('positionSeries', denseSeries);
    const changes: LogitPanelState[] = [];
    fixture.componentInstance.settingsChange.subscribe((change) => changes.push(change));
    return {fixture, chart: fixture.componentInstance, changes};
  }
  /** Runs the chart's effects and waits for the server's answers of its range and overview. */
  async function loaded(fixture: ReturnType<typeof create>['fixture']) {
    fixture.detectChanges();
    const chart = fixture.componentInstance;
    await vi.waitFor(() => {
      expect(chart.rangeData.loading() || chart.overviewData.loading()).toBe(false);
      expect(chart.overviewData.data()).not.toBeNull();
    });
  }

  it('asks the server for the whole stage once, and then for each zoomed range alone', async () => {
    const {fixture, chart} = create({...settings, metric: 'logit_rmse'});
    await loaded(fixture);
    expect(requests).toEqual([
      {
        turn: 2,
        stage: 'decode',
        alignment: 'content',
        analysis: 'final_logits',
        metric: 'rmse',
        start: 0,
        end: 320,
        bins: 160,
      },
    ]);
    expect(chart.bins()).toHaveLength(160);
    expect(chart.overview().bins).toEqual(chart.bins());
    // A new series of the same rows asks nothing anew.
    fixture.componentRef.setInput('positionSeries', [
      {...denseSeries[0], points: denseSeries[0].points.map((point) => ({...point}))},
    ]);
    fixture.detectChanges();
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(requests).toHaveLength(1);
    const rmse = {...settings, metric: 'logit_rmse' as const};
    fixture.componentRef.setInput('settings', {...rmse, positionRange: {start: 10, end: 20}});
    await loaded(fixture);
    await vi.waitFor(() => expect(chart.rangeData.data()).not.toBeNull());
    // The overview's answer cannot change with a zoom: it is not asked again.
    expect(requests.slice(1).map(({start, end, bins}) => [start, end, bins])).toEqual([
      [10, 20, 10],
    ]);
    expect(chart.overview().bins).toHaveLength(160);
    expect(chart.bins()).toHaveLength(10);
    // Back to the whole stage: the overview's answer is drawn, asking nothing.
    fixture.componentRef.setInput('settings', {...rmse, positionRange: {start: 0, end: 320}});
    fixture.detectChanges();
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(requests).toHaveLength(2);
    expect(chart.bins()).toHaveLength(160);
    fixture.destroy();
  });

  it('keeps drawing the last answer until the next one arrives', async () => {
    const {fixture, chart} = create();
    await loaded(fixture);
    const drawn = chart.chartRows();
    expect(chart.bins()).toHaveLength(160);
    let resolve!: () => void;
    const held = new Promise<void>((done) => (resolve = done));
    const served = serve;
    serve = (request) => answer(request, [{}]);
    const api = TestBed.inject(TokenAnalysisApi) as unknown as {
      tokenRange: (capture: string, request: TokenRangeRequest) => Promise<TokenRangeResponse>;
    };
    const tokenRange = api.tokenRange;
    api.tokenRange = async (capture, request) => {
      await held;
      return tokenRange(capture, request);
    };
    // A zoom, a resize and a metric change: each keeps the chart and its overview drawn.
    fixture.componentRef.setInput('settings', {...settings, positionRange: {start: 10, end: 50}});
    fixture.detectChanges();
    await new Promise((done) => setTimeout(done, 150));
    expect(chart.rangeData.loading()).toBe(true);
    expect(chart.chartRows()).toBe(drawn);
    expect(chart.overview().bins).toHaveLength(160);
    chart.capacity.set(20);
    fixture.componentRef.setInput('settings', {
      ...settings,
      metric: 'js',
      positionRange: {start: 10, end: 50},
    });
    fixture.detectChanges();
    await new Promise((done) => setTimeout(done, 150));
    expect(chart.chartRows()).toBe(drawn);
    expect(chart.overview().bins).toHaveLength(160);
    resolve();
    await vi.waitFor(() => expect(chart.rangeData.data()?.metric).toBe('js'));
    expect(chart.bins()).toHaveLength(20);
    serve = served;
    fixture.destroy();
  });

  it('says the server is computing until its first answer, instead of drawing empty axes', async () => {
    let resolve!: () => void;
    const held = new Promise<void>((done) => (resolve = done));
    const served = serve;
    serve = (request) => answer(request, [{}]);
    const api = TestBed.inject(TokenAnalysisApi) as unknown as {
      tokenRange: (capture: string, request: TokenRangeRequest) => Promise<TokenRangeResponse>;
    };
    const tokenRange = api.tokenRange;
    api.tokenRange = async (capture, request) => {
      await held;
      return tokenRange(capture, request);
    };
    const {fixture} = create();
    fixture.detectChanges();
    await new Promise((done) => setTimeout(done, 150));
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('.chart-state[role="status"]')?.textContent).toBe(
      'Loading Logits comparison…',
    );
    // The chart is laid out, unseen, so that it measures its width meanwhile.
    expect(element.querySelector('range-chart')?.classList).toContain('chart-pending');
    resolve();
    await loaded(fixture);
    fixture.detectChanges();
    expect(element.querySelector('.chart-state')).toBeNull();
    expect(element.querySelector('range-chart')?.classList).not.toContain('chart-pending');
    serve = served;
    fixture.destroy();
  });

  it('shows why the server could not answer, and asks again on Retry', async () => {
    let fail = true;
    serve = (request) => {
      if (fail) {
        throw new Error('The column range must lie in the stage.');
      }
      return answer(request, [{}]);
    };
    const {fixture, chart} = create();
    fixture.detectChanges();
    await vi.waitFor(() => expect(chart.failure()).toBe('The column range must lie in the stage.'));
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;
    const alert = element.querySelector('.chart-state.error[role="alert"]')!;
    expect(alert.textContent).toContain('The column range must lie in the stage.');
    fail = false;
    alert.querySelector('button')!.click();
    await loaded(fixture);
    fixture.detectChanges();
    expect(chart.failure()).toBe('');
    expect(element.querySelector('[role="alert"]')).toBeNull();
    expect(chart.bins()).toHaveLength(160);
    fixture.destroy();
  });

  it('asks nothing anew for a request equal to the last', async () => {
    const {fixture, chart} = create();
    await loaded(fixture);
    const asked = requests.length;
    // A capacity of the same bin count, a reduction, a selection: the answer cannot change.
    chart.capacity.set(161);
    fixture.componentRef.setInput('settings', {...settings, positionReduction: 'mean'});
    fixture.componentRef.setInput('selectedIndex', 3);
    fixture.detectChanges();
    await new Promise((done) => setTimeout(done, 150));
    expect(requests).toHaveLength(asked);
    expect(chart.rangeData.loading() || chart.overviewData.loading()).toBe(false);
    fixture.destroy();
  });

  it('routes both Inspection choices from the View picker while preserving comparison settings', () => {
    const {fixture, chart, changes} = create();
    fixture.componentRef.setInput('inspectionEnabled', true);
    fixture.detectChanges();
    const picker = fixture.debugElement.query(By.directive(ConfigPicker))
      .componentInstance as ConfigPicker;
    const inspect: ('index' | 'scatter')[] = [];
    chart.inspect.subscribe((plot) => inspect.push(plot));
    expect(picker.options()).toEqual([
      'Tensor Comparison · Line chart',
      'Tensor Inspection · Line chart',
      'Tensor Inspection · Scatter plot',
    ]);
    picker.valueChange.emit('Tensor Inspection · Line chart');
    picker.valueChange.emit('Tensor Inspection · Scatter plot');
    picker.valueChange.emit('Tensor Comparison · Line chart');
    expect(inspect).toEqual(['index', 'scatter']);
    expect(changes).toEqual([]);
    expect(chart.settings()).toEqual(settings);
    fixture.destroy();
  });

  it('keeps distribution views limited to their own Line chart when Inspection is not enabled', () => {
    const {fixture, chart} = create();
    fixture.componentRef.setInput('modeLabel', 'Tensor Distribution');
    fixture.detectChanges();
    const picker = fixture.debugElement.query(By.directive(ConfigPicker))
      .componentInstance as ConfigPicker;
    expect(picker.options()).toEqual(['Tensor Distribution · Line chart']);
    expect(chart.visualizationLabel()).toBe('Tensor Distribution · Line chart');
    fixture.destroy();
  });

  it('adapts margin summaries to percentage points without changing source values or selected positions', async () => {
    const {fixture, chart} = create({...settings, metric: 'margin'});
    const points = [0, 1, 2, 3].map((position) => ({position, index: position + 10}));
    fixture.componentRef.setInput('positionSeries', [{...denseSeries[0], points}]);
    chart.capacity.set(1);
    const cell = {min: 0, max: 0.5, mean: 0.25, valid_count: 3, min_position: 0, max_position: 2};
    serve = (request) => answer(request, [{0: cell}]);
    await loaded(fixture);
    const summary = {
      min: 0,
      max: 50,
      mean: 25,
      valid_count: 3,
      min_position: 0,
      max_position: 2,
    };
    expect(chart.chartRows()[0].bins[0].metrics.max_abs).toEqual(summary);
    expect(chart.overview().rows[0].bins[0].metrics.max_abs).toEqual(summary);
    expect(chart.representativeIndex({start: 0, end: 4, layer: 0})).toBe(12);
    expect(chart.format(25)).toBe('25.00 pp');
    expect(chart.format(0)).toBe('0.00 pp');
    expect(chart.format(null)).toBe('Unavailable');
    // A probability is drawn in percent, as the catalog scales it.
    fixture.componentRef.setInput('settings', {...settings, metric: 'selected_probability'});
    await loaded(fixture);
    expect(chart.chartRows()[0].bins[0].metrics.max_abs).toMatchObject({
      min: 0,
      max: 50,
      mean: 25,
    });
    expect(chart.format(25)).toBe('25.00%');
    fixture.destroy();
  });

  it('uses the same percentage-point value for rendered axes, hover text, and the hover marker', async () => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(private callback: (entries: unknown[]) => void) {}
        observe() {
          this.callback([{contentRect: {width: 600, height: 400}}]);
        }
        disconnect() {}
      },
    );
    const events: Record<string, (event: Record<string, unknown>) => void> = {};
    const yPixel = vi.fn((value: number) => value);
    const api: PlotlyApi = {
      react: vi.fn(async (host: HTMLElement) => {
        const plot = host as PlotHost;
        plot.on = (name, fn) => {
          events[name] = fn;
        };
        plot._fullLayout = {
          xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
          yaxis: {_offset: 20, _length: 100, d2p: yPixel},
        };
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
    (window as Window & {Plotly?: PlotlyApi}).Plotly = api;
    const {fixture} = create({...settings, metric: 'margin'});
    fixture.componentRef.setInput('metricLabel', 'Top-2 margin (pp)');
    fixture.componentRef.setInput('positionSeries', [
      {
        ...denseSeries[0],
        points: [{position: 0, index: 0}],
      },
    ]);
    const cell = {min: 0.25, max: 0.25, mean: 0.25, valid_count: 1, min_position: 0};
    serve = (request) => answer(request, [{0: {...cell, max_position: 0}}]);
    await loaded(fixture);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    await vi.waitFor(() => expect(events['plotly_hover']).toBeTypeOf('function'));
    // The chart's size sets its capacity, which asks again: draw once that answer is in.
    await vi.waitFor(() => {
      fixture.detectChanges();
      expect(vi.mocked(api.react).mock.calls.at(-1)![1][0]).toMatchObject({y: [25]});
    });
    const [_, traces, layout] = vi.mocked(api.react).mock.calls.at(-1)!;
    expect(traces[0]).toMatchObject({y: [25]});
    expect(layout['yaxis']).toMatchObject({
      title: {text: 'Top-2 margin (pp)'},
      tickformat: '.3~g',
    });
    const rendered = fixture.debugElement.query(By.directive(RangeChart))
      .componentInstance as RangeChart;
    events['plotly_hover']({points: [{customdata: {start: 0, end: 1, layer: 0, kind: 'logits'}}]});
    expect(rendered.tooltip()?.content.rows?.[0].values).toEqual(['25.00 pp']);
    expect(yPixel).toHaveBeenCalledWith(25);
    expect(rendered.hoveredPoints()).toEqual([{left: 10, top: 45}]);
    fixture.destroy();
  });

  it('renders the shared runtime legend and keeps theme colors and line styles in the plotted traces', async () => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(private callback: (entries: unknown[]) => void) {}
        observe() {
          this.callback([{contentRect: {width: 600, height: 400}}]);
        }
        disconnect() {}
      },
    );
    // jsdom does not resolve CSS custom properties; emulate the browser's computed color
    // for the chart's existing color probes while keeping all other computed styles real.
    const getStyle = window.getComputedStyle.bind(window);
    const colors = {ref: 'rgb(10, 20, 30)', target: 'rgb(80, 90, 100)'};
    vi.spyOn(window, 'getComputedStyle').mockImplementation((element, pseudo) => {
      const token = (element as HTMLElement).style?.color;
      if (token === 'var(--me-primary-color)') {
        return {color: colors.ref} as CSSStyleDeclaration;
      }
      if (token === 'var(--me-on-surface-variant-color)') {
        return {color: colors.target} as CSSStyleDeclaration;
      }
      return getStyle(element, pseudo);
    });
    const api: PlotlyApi = {
      react: vi.fn(async (host: HTMLElement) => {
        (host as PlotHost).on = () => {};
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
    (window as Window & {Plotly?: PlotlyApi}).Plotly = api;
    const {fixture} = create({...settings, metric: 'entropy'});
    fixture.componentRef.setInput('modeLabel', 'Distribution Comparison');
    fixture.componentRef.setInput('positionSeries', [
      {
        name: 'Reference',
        runtime: 'ref',
        color: 'var(--me-primary-color)',
        points: [{position: 0, index: 0}],
      },
      {
        name: 'Target',
        runtime: 'target',
        color: 'var(--me-on-surface-variant-color)',
        points: [{position: 0, index: 0}],
      },
    ]);
    fixture.componentRef.setInput('analysis', 'sampling');
    const at = (value: number) => ({
      0: {min: value, max: value, mean: value, valid_count: 1, min_position: 0, max_position: 0},
    });
    serve = (request) => answer(request, [at(0.25), at(0.5)]);
    await loaded(fixture);
    expect(requests.at(-1)).toMatchObject({analysis: 'sampling', metric: 'entropy'});
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    await vi.waitFor(() => expect(api.react).toHaveBeenCalled());
    const legend: HTMLElement = fixture.nativeElement.querySelector('runtime-legend');
    expect(legend).not.toBeNull();
    expect(legend.textContent).toContain('Ref');
    expect(legend.textContent).toContain('Target');
    const traces = vi.mocked(api.react).mock.calls.at(-1)![1];
    expect(traces[0]).toMatchObject({name: 'Reference', line: {color: colors.ref, dash: 'solid'}});
    expect(traces[1]).toMatchObject({name: 'Target', line: {color: colors.target, dash: 'dash'}});

    const theme = document.documentElement.getAttribute('data-theme');
    try {
      const calls = vi.mocked(api.react).mock.calls.length;
      colors.ref = 'rgb(130, 140, 150)';
      colors.target = 'rgb(190, 200, 210)';
      document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'light' : 'dark');
      await vi.waitFor(() => expect(vi.mocked(api.react).mock.calls.length).toBeGreaterThan(calls));
      const updated = vi.mocked(api.react).mock.calls.at(-1)![1];
      expect(updated[0]).toMatchObject({line: {color: colors.ref, dash: 'solid'}});
      expect(updated[1]).toMatchObject({line: {color: colors.target, dash: 'dash'}});
    } finally {
      fixture.destroy();
      if (theme === null) {
        document.documentElement.removeAttribute('data-theme');
      } else {
        document.documentElement.setAttribute('data-theme', theme);
      }
    }
  });

  it('shows four comparison metrics without a Transform control or changing saved inspection settings', () => {
    const {fixture, chart, changes} = create({...settings, logitTransform: 'centered'});
    const metrics = [
      {id: 'logit_rmse', label: 'Root mean square error'},
      {id: 'logit_max_abs', label: 'Max absolute error'},
      {id: 'kl', label: 'KL'},
      {id: 'js', label: 'JS'},
    ];
    fixture.componentRef.setInput('inspectionEnabled', true);
    fixture.componentRef.setInput('metrics', metrics);
    for (const metric of metrics) {
      fixture.componentRef.setInput('settings', {
        ...settings,
        metric: metric.id,
        logitTransform: 'centered',
      });
      fixture.detectChanges();
      const pickers = fixture.debugElement
        .queryAll(By.directive(ConfigPicker))
        .map((picker) => picker.componentInstance as ConfigPicker);
      expect(pickers.find((picker) => picker.label() === 'Analysis metric')?.options()).toEqual(
        metrics.map((value) => value.label),
      );
      expect(pickers.some((picker) => picker.label() === 'Logits transform')).toBe(false);
      expect(chart.settings().logitTransform).toBe('centered');
    }
    expect(changes).toEqual([]);
    fixture.destroy();
  });

  it('explains full-vocabulary softmax only while a divergence metric is selected', () => {
    const {fixture} = create();
    for (const [metric, label] of [
      ['kl', 'Kullback–Leibler divergence'],
      ['logit_rmse', null],
      ['js', 'Jensen–Shannon divergence'],
      ['logit_max_abs', null],
    ]) {
      fixture.componentRef.setInput('settings', {...settings, metric});
      fixture.detectChanges();
      const note = fixture.nativeElement.querySelector(
        '[role="note"][aria-label="Metric inputs"]',
      ) as HTMLElement | null;
      // The catalog names what a divergence compares; the whole-vector errors name nothing.
      if (label) {
        expect(note?.textContent).toContain(label);
        expect(note?.textContent).toContain('softmax (T=1)');
        expect(note?.textContent).toContain('Full captured vocabulary logits');
      } else {
        expect(note).toBeNull();
      }
    }
    fixture.destroy();
  });

  it('selects the server bin extrema and keeps the full selected bin', async () => {
    const {fixture, chart} = create({...settings, positionRange: {start: 32, end: 40}});
    const points = [37, 36, 39, 33, 35, 32, 38, 34].map((position) => ({
      position,
      index: position + 100,
    }));
    fixture.componentRef.setInput('positionSeries', [{...denseSeries[0], points}]);
    chart.capacity.set(1);
    const cell = {min: -2, max: 9, mean: 3.4, valid_count: 5, min_position: 34, max_position: 35};
    serve = (request) => answer(request, [{32: cell}]);
    await loaded(fixture);
    const range: RangeSelection = {...chart.bins()[0], layer: 0};
    const positions: number[] = [];
    const selections: {index: number; range: RangeSelection}[] = [];
    chart.selectPosition.subscribe((index) => {
      positions.push(index);
      fixture.componentRef.setInput('selectedIndex', index);
    });
    chart.selectRange.subscribe((selection) => selections.push(selection));
    expect(chart.chartRows()[0].bins[0].metrics.max_abs).toEqual({
      min: -2,
      max: 9,
      mean: 3.4,
      valid_count: 5,
      min_position: 34,
      max_position: 35,
    });
    for (const [positionReduction, index] of [
      ['min', 134],
      ['max', 135],
      ['mean', 132],
    ] as const) {
      fixture.componentRef.setInput('settings', {...settings, positionReduction});
      chart.select(range);
      expect(positions.at(-1)).toBe(index);
      expect(selections.at(-1)).toEqual({index, range});
      expect(chart.selection()).toEqual(range);
    }
    fixture.destroy();
  });

  it('uses the selected series for an extremum and the earliest captured position for missing bins', async () => {
    const {fixture, chart} = create({...settings, positionRange: {start: 32, end: 35}});
    fixture.componentRef.setInput('positionSeries', [
      {
        ...denseSeries[0],
        points: [
          {position: 34, index: 134},
          {position: 32, index: 132},
        ],
      },
      {...denseSeries[0], points: [{position: 32, index: 132}]},
    ]);
    chart.capacity.set(1);
    const at = (position: number, value: number) => ({
      32: {
        min: value,
        max: value,
        mean: value,
        valid_count: 1,
        min_position: position,
        max_position: position,
      },
    });
    serve = (request) => answer(request, [at(34, 3), at(32, 100)], at(32, 100));
    await loaded(fixture);
    expect(chart.bins()).toEqual([{start: 32, end: 35}]);
    const selections: {index: number; range: RangeSelection}[] = [];
    chart.selectRange.subscribe((selection) => selections.push(selection));
    chart.select({start: 32, end: 35, layer: 0});
    chart.select({start: 32, end: 35, layer: 1});
    chart.select({start: 32, end: 35, layer: null});
    chart.select({start: 31, end: 33, layer: 0});
    expect(selections.map(({index}) => index)).toEqual([134, 132, 132, 132]);
    expect(selections.at(-1)?.range).toEqual({start: 31, end: 33, layer: 0});
    fixture.destroy();
  });

  it('asks for a selected range that outlived its bins and finds its extremum there', async () => {
    const {fixture, chart} = create({...settings, positionRange: {start: 32, end: 36}});
    fixture.componentRef.setInput('positionSeries', [
      {
        ...denseSeries[0],
        points: [32, 33, 34, 35].map((position) => ({position, index: position})),
      },
    ]);
    chart.capacity.set(2);
    const cell = {min: 1, max: 5, mean: 3, valid_count: 3, min_position: 33, max_position: 34};
    serve = (request) =>
      request.bins === 1 ? answer(request, [{33: cell}], {33: cell}) : answer(request, [{}]);
    await loaded(fixture);
    expect(chart.bins()).toEqual([
      {start: 32, end: 34},
      {start: 34, end: 36},
    ]);
    const range = {start: 33, end: 36, layer: null};
    fixture.componentRef.setInput('selectedRange', range);
    fixture.detectChanges();
    await vi.waitFor(() => expect(chart.selectionData.data()).not.toBeNull());
    expect(requests.at(-1)).toMatchObject({start: 33, end: 36, bins: 1});
    expect(chart.representativeIndex(range)).toBe(34);
    expect(chart.representativeIndex({...range, layer: 0})).toBe(34);
    fixture.destroy();
  });

  it('discards an unapplied draft on close and emits a changed reduction only when applied', async () => {
    const {fixture, chart, changes} = create();
    await loaded(fixture);
    expect(chart.binsDisabled()).toBe(false);
    chart.toggleDisplay();
    chart.draftReduction('Minimum');
    expect(chart.displayDirty()).toBe(true);
    expect(changes).toEqual([]);
    expect(chart.reduction()).toBe('max');

    chart.toggleDisplay();
    chart.toggleDisplay();
    expect(chart.displayDraft()).toBe('max');
    expect(chart.displayDirty()).toBe(false);
    expect(changes).toEqual([]);

    chart.draftReduction('Mean');
    chart.applyDisplay();
    expect(changes).toEqual([{...settings, positionReduction: 'mean'}]);
    // The owning panel stores settingsChange and feeds its new settings back to the chart.
    fixture.componentRef.setInput('settings', changes[0]);
    expect(chart.displayDirty()).toBe(false);
    chart.applyDisplay();
    expect(changes).toHaveLength(1);
    fixture.destroy();
  });

  it('requires Apply before restoring the default reduction and preserves the metric and range', async () => {
    const initial: LogitPanelState = {...settings, metric: 'js', positionReduction: 'mean'};
    const {fixture, chart, changes} = create(initial);
    await loaded(fixture);
    chart.toggleDisplay();
    // Reset to default stages Maximum through the same draft state as the picker.
    chart.draftReduction('Maximum');
    expect(chart.displayDirty()).toBe(true);
    expect(chart.reduction()).toBe('mean');
    expect(changes).toEqual([]);

    chart.applyDisplay();
    expect(changes).toEqual([{...initial, positionReduction: 'max'}]);
    fixture.componentRef.setInput('settings', changes[0]);
    chart.toggleDisplay();
    chart.toggleDisplay();
    expect(chart.displayDraft()).toBe('max');
    expect(chart.displayDirty()).toBe(false);
    fixture.destroy();
  });

  it('disables bin changes for a single position without losing the stored reduction', async () => {
    const initial: LogitPanelState = {...settings, positionReduction: 'mean'};
    const {fixture, chart, changes} = create(initial);
    await loaded(fixture);
    expect(chart.binsDisabled()).toBe(false);
    fixture.componentRef.setInput('settings', {
      ...initial,
      positionRange: {start: 45, end: 46},
    });
    await loaded(fixture);
    chart.toggleDisplay();
    expect(chart.binsDisabled()).toBe(true);
    expect(chart.reduction()).toBe('mean');
    chart.draftReduction('Minimum');
    expect(chart.displayDirty()).toBe(false);
    chart.applyDisplay();
    expect(changes).toEqual([]);

    fixture.componentRef.setInput('settings', initial);
    await loaded(fixture);
    chart.toggleDisplay();
    chart.toggleDisplay();
    expect(chart.binsDisabled()).toBe(false);
    expect(chart.displayDraft()).toBe('mean');
    expect(chart.displayDirty()).toBe(false);
    fixture.destroy();
  });
});
