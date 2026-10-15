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

import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {ConfigPicker} from '../../shared/config_picker/config_picker';
import {FinalLogitsTrends, type LogitPositionSeries} from './final_logits_trends';
import {KvRangeChart} from '../kv/kv_panel/kv_range_chart';
import type {PlotHost, PlotlyApi} from '../../shared/plotly_loader';
import type {TokenTrendState} from '../conversation/token_trends';
import type {KvRangeSelection} from '../kv/kv_panel/kv_range_types';

describe('FinalLogitsTrends display settings', () => {
  beforeEach(() => {
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
      points: Array.from({length: 320}, (_, index) => ({
        position: index,
        index,
        value: index / 100,
      })),
    },
  ];
  const settings: TokenTrendState = {
    open: true,
    metric: 'kl',
    start: 0,
    end: 319,
    positionRange: {start: 0, end: 320},
  };

  function create(initial = settings) {
    // Keep compiled signal-input metadata; no detectChanges is needed for these state tests.
    const fixture = TestBed.createComponent(FinalLogitsTrends);
    fixture.componentRef.setInput('analysisLabel', 'Logits comparison');
    fixture.componentRef.setInput('metrics', [{id: 'kl', label: 'KL'}]);
    fixture.componentRef.setInput('metricLabel', 'KL (nats)');
    fixture.componentRef.setInput('settings', initial);
    fixture.componentRef.setInput('positionSeries', denseSeries);
    const changes: TokenTrendState[] = [];
    fixture.componentInstance.settingsChange.subscribe((change) => changes.push(change));
    return {fixture, chart: fixture.componentInstance, changes};
  }

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

  it('adapts margin summaries to percentage points without changing source values or selected positions', () => {
    const {fixture, chart} = create({...settings, metric: 'margin'});
    const points = [0, 0.25, 0.5, null].map((value, position) => ({
      value,
      position,
      index: position + 10,
    }));
    fixture.componentRef.setInput('positionSeries', [{...denseSeries[0], points}]);
    chart.capacity.set(1);
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
    expect(chart.positionSeries()[0].points.map((point) => point.value)).toEqual([
      0,
      0.25,
      0.5,
      null,
    ]);
    expect(chart.representativeIndex({start: 0, end: 4, layer: 0})).toBe(12);
    expect(chart.percent()).toBe(false);
    expect(chart.format(25)).toBe('25.00 pp');
    expect(chart.format(0)).toBe('0.00 pp');
    expect(chart.format(null)).toBe('Unavailable');
    fixture.componentRef.setInput('settings', {...settings, metric: 'selected_probability'});
    expect(chart.percent()).toBe(true);
    expect(chart.chartRows()[0].bins[0].metrics.max_abs).toMatchObject({
      min: 0,
      max: 0.5,
      mean: 0.25,
    });
    expect(chart.format(0.25)).toBe('25.00%');
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
        points: [{position: 0, index: 0, value: 0.25}],
      },
    ]);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    await vi.waitFor(() => expect(events['plotly_hover']).toBeTypeOf('function'));
    const [_, traces, layout] = vi.mocked(api.react).mock.calls.at(-1)!;
    expect(traces[0]).toMatchObject({y: [25]});
    expect(layout['yaxis']).toMatchObject({
      title: {text: 'Top-2 margin (pp)'},
      tickformat: '.3~g',
    });
    const rendered = fixture.debugElement.query(By.directive(KvRangeChart))
      .componentInstance as KvRangeChart;
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
      if (token === 'var(--me-primary-color)') return {color: colors.ref} as CSSStyleDeclaration;
      if (token === 'var(--me-on-surface-variant-color)')
        return {color: colors.target} as CSSStyleDeclaration;
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
        points: [{position: 0, index: 0, value: 0.25}],
      },
      {
        name: 'Target',
        runtime: 'target',
        color: 'var(--me-on-surface-variant-color)',
        points: [{position: 0, index: 0, value: 0.5}],
      },
    ]);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    await vi.waitFor(() => expect(api.react).toHaveBeenCalled());
    const legend: HTMLElement = fixture.nativeElement.querySelector('analysis-runtime-legend');
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
      if (theme === null) document.documentElement.removeAttribute('data-theme');
      else document.documentElement.setAttribute('data-theme', theme);
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
      ['kl', 'KL divergence'],
      ['logit_rmse', null],
      ['js', 'JS divergence'],
      ['logit_max_abs', null],
    ]) {
      fixture.componentRef.setInput('settings', {...settings, metric});
      fixture.detectChanges();
      const note = fixture.nativeElement.querySelector(
        '[role="note"][aria-label="Softmax before divergence"]',
      ) as HTMLElement | null;
      if (label) {
        expect(note?.textContent).toContain(label);
        expect(note?.textContent).toContain('softmax (T = 1)');
        expect(note?.textContent).toContain('full Ref and Target logits');
      } else expect(note).toBeNull();
    }
    fixture.destroy();
  });

  it('selects actual bin extrema with earliest-position ties and keeps the full selected bin', () => {
    const {fixture, chart} = create({...settings, positionRange: {start: 32, end: 40}});
    const values = [null, NaN, -2, 9, -2, 9, Infinity, 3];
    const points = [37, 36, 39, 33, 35, 32, 38, 34].map((position) => ({
      position,
      index: position + 100,
      value: values[position - 32],
    }));
    fixture.componentRef.setInput('positionSeries', [{...denseSeries[0], points}]);
    chart.capacity.set(1);
    const range: KvRangeSelection = {...chart.bins()[0], layer: 0};
    const positions: number[] = [];
    const selections: {index: number; range: KvRangeSelection}[] = [];
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

  it('uses the selected series for an extremum and the earliest captured position for missing bins', () => {
    const {fixture, chart} = create();
    fixture.componentRef.setInput('positionSeries', [
      {
        ...denseSeries[0],
        points: [
          {position: 34, index: 134, value: 3},
          {position: 32, index: 132, value: null},
        ],
      },
      {...denseSeries[0], points: [{position: 32, index: 132, value: 100}]},
    ]);
    const selections: {index: number; range: KvRangeSelection}[] = [];
    chart.selectRange.subscribe((selection) => selections.push(selection));
    chart.select({start: 32, end: 35, layer: 0});
    chart.select({start: 32, end: 35, layer: 1});
    chart.select({start: 32, end: 35, layer: null});
    chart.select({start: 31, end: 33, layer: 0});
    expect(selections.map(({index}) => index)).toEqual([134, 132, 132, 132]);
    expect(selections.at(-1)?.range).toEqual({start: 31, end: 33, layer: 0});
    fixture.destroy();
  });

  it('discards an unapplied draft on close and emits a changed reduction only when applied', () => {
    const {fixture, chart, changes} = create();
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

  it('requires Apply before restoring the default reduction and preserves the metric and range', () => {
    const initial: TokenTrendState = {...settings, metric: 'js', positionReduction: 'mean'};
    const {fixture, chart, changes} = create(initial);
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

  it('disables bin changes for a single position without losing the stored reduction', () => {
    const initial: TokenTrendState = {...settings, positionReduction: 'mean'};
    const {fixture, chart, changes} = create(initial);
    expect(chart.binsDisabled()).toBe(false);
    fixture.componentRef.setInput('settings', {
      ...initial,
      positionRange: {start: 45, end: 46},
    });
    chart.toggleDisplay();
    expect(chart.binsDisabled()).toBe(true);
    expect(chart.reduction()).toBe('mean');
    chart.draftReduction('Minimum');
    expect(chart.displayDirty()).toBe(false);
    chart.applyDisplay();
    expect(changes).toEqual([]);

    fixture.componentRef.setInput('settings', initial);
    chart.toggleDisplay();
    chart.toggleDisplay();
    expect(chart.binsDisabled()).toBe(false);
    expect(chart.displayDraft()).toBe('mean');
    expect(chart.displayDirty()).toBe(false);
    fixture.destroy();
  });
});
