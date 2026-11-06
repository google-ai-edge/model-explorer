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
import {ChannelChart} from './channel_chart';
import type {PlotHost, PlotlyApi} from '../plotly_loader';

function plotlyFixture() {
  const events: Record<string, (event: Record<string, unknown>) => void> = {};
  const react = vi.fn(
    async (host: PlotHost, _traces: unknown[], layout: Record<string, unknown>) => {
      host.on = (name, listener) => {
        events[name] = listener;
      };
      const yaxis = {_offset: 20, _length: 100, d2p: (n: number) => n * 10};
      host._fullLayout = {
        xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
        yaxis,
        ...(layout['yaxis2'] ? {yaxis2: {...yaxis, _offset: 140}} : {}),
      };
    },
  );
  (window as Window & {Plotly?: PlotlyApi}).Plotly = {react, purge: vi.fn(), restyle: vi.fn()};
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
  vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
  return {
    click(plane: number, side: 'ref' | 'target') {
      const traces = react.mock.calls.at(-1)![1] as {customdata: {plane: number; side: string}[]}[];
      const trace = traces.find(
        (trace) => trace.customdata[0]?.plane === plane && trace.customdata[0]?.side === side,
      )!;
      events['plotly_click']({points: [{customdata: trace.customdata[0]}]});
    },
  };
}

describe('channel chart keyboard navigation', () => {
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  it('uses vocabulary labels and preserves side values without deriving unavailable differences', async () => {
    const events: Record<string, (event: Record<string, unknown>) => void> = {};
    const react = vi.fn(
      async (host: PlotHost, _traces: unknown[], _layout: Record<string, unknown>) => {
        host.on = (name, listener) => {
          events[name] = listener;
        };
        host._fullLayout = {
          xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
          yaxis: {_offset: 20, _length: 100, d2p: (n) => n * 10},
        };
      },
    );
    (window as Window & {Plotly?: PlotlyApi}).Plotly = {react, purge: vi.fn(), restyle: vi.fn()};
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
    vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
    const fixture = TestBed.createComponent(ChannelChart);
    fixture.componentRef.setInput('range', {start: 0, end: 2});
    fixture.componentRef.setInput('full', {start: 0, end: 2});
    fixture.componentRef.setInput('rows', [{channel: 0, ref: 2, target: 5}]);
    fixture.componentRef.setInput('dimensionLabel', 'Vocabulary index');
    fixture.componentRef.setInput('groupLabels', ['Final logits']);
    fixture.componentRef.setInput('comparisonAllowed', false);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const layout = react.mock.calls.at(-1)![2];
    expect(layout['xaxis']).toMatchObject({title: {text: 'Vocabulary index'}});
    expect(
      fixture.nativeElement.querySelector('.channel-chart').getAttribute('aria-label'),
    ).toContain('Vocabulary index');
    events['plotly_hover']({points: [{customdata: {start: 0, end: 1, plane: 0}}]});
    expect(fixture.componentInstance.tooltip()?.content.rows?.[0]).toMatchObject({
      label: 'Final logits',
      values: ['2', '5', 'Unavailable', 'Unavailable'],
    });
    fixture.destroy();
  });
  it('synchronizes cache selection within the visible range without zooming', () => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
    const fixture = TestBed.createComponent(ChannelChart);
    fixture.componentRef.setInput('range', {start: 5, end: 10});
    fixture.componentRef.setInput('full', {start: 0, end: 10});
    fixture.componentRef.setInput('selected', 7);
    fixture.componentRef.setInput(
      'rows',
      Array.from({length: 10}, (_, channel) => ({
        channel,
        ref: channel,
        target: channel,
        delta: 0,
        abs_delta: 0,
        status: 'ok',
      })),
    );
    fixture.componentRef.setInput('otherRows', fixture.componentInstance.rows());
    const otherSelections: number[] = [];
    fixture.componentInstance.otherSelect.subscribe((value) => otherSelections.push(value));
    const selections: number[] = [],
      zooms: unknown[] = [];
    fixture.componentInstance.select.subscribe((value) => selections.push(value));
    fixture.componentInstance.zoom.subscribe((value) => zooms.push(value));
    fixture.detectChanges();
    fixture.nativeElement.dispatchEvent(new KeyboardEvent('keydown', {key: 'Home', bubbles: true}));
    expect(selections).toEqual([5]);
    expect(otherSelections).toEqual([5]);
    expect(fixture.componentInstance.channelSelection()).toEqual({start: 5, end: 6, plane: null});
    expect(zooms).toEqual([]);
    fixture.nativeElement.dispatchEvent(new MouseEvent('dblclick', {bubbles: true}));
    expect(zooms).toEqual([]);
    fixture.destroy();
  });
  it.each([
    ['min', [1, 2], [2, 0]],
    ['max', [0, 1], [1, 2]],
    ['mean', [0, 0], [0, 0]],
  ] as const)(
    'selects %s representatives only from the clicked cache and runtime while keeping the bin',
    async (reduction, key, value) => {
      const plot = plotlyFixture();
      const fixture = TestBed.createComponent(ChannelChart);
      fixture.componentRef.setInput('range', {start: 0, end: 3});
      fixture.componentRef.setInput('full', {start: 0, end: 3});
      fixture.componentRef.setInput('capacity', 1);
      fixture.componentRef.setInput('reduction', reduction);
      fixture.componentRef.setInput('rows', [
        {channel: 0, ref: 8, target: 1},
        {channel: 1, ref: -5, target: 9},
        {channel: 2, ref: 3, target: -2},
      ]);
      fixture.componentRef.setInput('otherRows', [
        {channel: 0, ref: 3, target: -3},
        {channel: 1, ref: 7, target: 0},
        {channel: 2, ref: -2, target: 8},
      ]);
      const chart = fixture.componentInstance;
      const selections: number[] = [],
        otherSelections: number[] = [],
        zooms: unknown[] = [];
      chart.select.subscribe((index) => {
        selections.push(index);
        chart.channelSelection.set(null);
      });
      chart.otherSelect.subscribe((index) => {
        otherSelections.push(index);
        chart.channelSelection.set(null);
      });
      chart.zoom.subscribe((range) => zooms.push(range));
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
      plot.click(0, 'ref');
      plot.click(0, 'target');
      expect(selections).toEqual(key);
      expect(otherSelections).toEqual([]);
      plot.click(1, 'ref');
      plot.click(1, 'target');
      expect(selections).toEqual(key);
      expect(otherSelections).toEqual(value);
      expect(chart.channelSelection()).toEqual({start: 0, end: 3, plane: null});
      expect(zooms).toEqual([]);
      fixture.destroy();
    },
  );
  it('uses Ref then Target in the first valid cache for a side-free multi-element axis selection', async () => {
    plotlyFixture();
    const fixture = TestBed.createComponent(ChannelChart);
    fixture.componentRef.setInput('range', {start: 0, end: 3});
    fixture.componentRef.setInput('full', {start: 0, end: 3});
    fixture.componentRef.setInput('capacity', 1);
    fixture.componentRef.setInput('reduction', 'max');
    const rows = [
      {channel: 0, ref: 8, target: 1},
      {channel: 1, ref: 0, target: 9},
      {channel: 2, ref: 3, target: -2},
    ];
    fixture.componentRef.setInput('rows', rows);
    fixture.componentRef.setInput(
      'otherRows',
      rows.map((row) => ({...row, ref: row.channel})),
    );
    const chart = fixture.componentInstance;
    const selections: number[] = [],
      otherSelections: number[] = [];
    chart.select.subscribe((index) => selections.push(index));
    chart.otherSelect.subscribe((index) => otherSelections.push(index));
    async function choose() {
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
      chart.columnKey(new KeyboardEvent('keydown', {key: 'Home'}));
    }
    await choose();
    fixture.componentRef.setInput(
      'rows',
      rows.map((row) => ({...row, ref: null})),
    );
    await choose();
    fixture.componentRef.setInput(
      'rows',
      rows.map((row) => ({...row, ref: null, target: null})),
    );
    await choose();
    expect(selections).toEqual([0, 1]);
    expect(otherSelections).toEqual([2]);
    expect(chart.channelSelection()).toEqual({start: 0, end: 3, plane: null});
    fixture.destroy();
  });
  it('draws reduced bins as given, reads their hover and click, and keeps its drawing while waiting', async () => {
    const events: Record<string, (event: Record<string, unknown>) => void> = {};
    const react = vi.fn(
      async (host: PlotHost, _traces: unknown[], _layout: Record<string, unknown>) => {
        host.on = (name, listener) => {
          events[name] = listener;
        };
        host._fullLayout = {
          xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
          yaxis: {_offset: 20, _length: 100, d2p: (n) => n * 10},
        };
      },
    );
    (window as Window & {Plotly?: PlotlyApi}).Plotly = {react, purge: vi.fn(), restyle: vi.fn()};
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
    vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
    const bin = (
      start: number,
      end: number,
      ref: number | null,
      target: number | null,
      refIndex: number | null,
      targetIndex: number | null,
    ) => ({start, end, ref, target, refIndex, targetIndex});
    const overview = [bin(0, 4, 1, 2, 1, 3), bin(4, 8, 5, 6, 4, 7), bin(8, 10, 9, null, 8, null)];
    const visible = [bin(4, 6, 5, 4, 5, 4), bin(6, 8, 3, 6, 6, 7)];
    const fixture = TestBed.createComponent(ChannelChart);
    const render = async () => {
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
    };
    fixture.componentRef.setInput('range', {start: 4, end: 8});
    fixture.componentRef.setInput('full', {start: 0, end: 10});
    fixture.componentRef.setInput('reduction', 'max');
    fixture.componentRef.setInput('bins', {overview, visible});
    fixture.componentRef.setInput('groupLabels', ['Final logits']);
    const selected: number[] = [];
    fixture.componentInstance.select.subscribe((index) => selected.push(index));
    await render();
    const traces = react.mock.calls.at(-1)![1] as {x: number[]; y: (number | null)[]}[];
    // The overview's bins outside the range, then the range's own.
    expect(traces.map((trace) => trace.y)).toEqual([
      [1, 5, 3, 9],
      [2, 4, 6, null],
    ]);
    expect(traces[0].x).toEqual([1.5, 4.5, 6.5, 8.5]);
    events['plotly_hover']({points: [{customdata: {...visible[1], plane: 0, side: 'ref'}}]});
    expect(fixture.componentInstance.tooltip()?.content.rows?.[0]).toMatchObject({
      label: 'Final logits',
      values: ['3', '6'],
    });
    events['plotly_click']({points: [{customdata: {...visible[1], plane: 0, side: 'target'}}]});
    events['plotly_click']({points: [{customdata: {...visible[0], plane: 0, side: 'ref'}}]});
    expect(selected).toEqual([7, 5]);
    // Every index of the range is an element: the keys walk them one by one.
    fixture.componentRef.setInput('selected', 6);
    await render();
    (fixture.nativeElement as HTMLElement).dispatchEvent(
      new KeyboardEvent('keydown', {key: 'ArrowRight'}),
    );
    expect(selected.at(-1)).toBe(7);
    (fixture.nativeElement as HTMLElement).dispatchEvent(
      new KeyboardEvent('keydown', {key: 'Home'}),
    );
    expect(selected.at(-1)).toBe(4);
    await render();
    const drawn = react.mock.calls.at(-1)!;
    fixture.componentRef.setInput('waiting', true);
    fixture.componentRef.setInput('range', {start: 0, end: 10});
    await render();
    // While waiting, a paint draws the last data again: its traces and its range.
    const kept = react.mock.calls.at(-1)!;
    expect(kept[1]).toEqual(drawn[1]);
    expect((kept[2] as {uirevision: string}).uirevision).toBe('4:8:max');
    // A new selection is painted meanwhile, over the drawing kept.
    const box = () =>
      (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.cell-selected')?.style
        .left;
    expect(box()).toBe('45px');
    fixture.componentRef.setInput('channelSelection', {start: 6, end: 7, plane: null});
    await render();
    expect(box()).toBe('65px');
    const paints = react.mock.calls.length;
    fixture.componentRef.setInput('bins', {overview, visible: overview});
    fixture.componentRef.setInput('waiting', false);
    await render();
    expect(react.mock.calls.length).toBeGreaterThan(paints);
    expect((react.mock.calls.at(-1)![1] as {y: unknown[]}[])[0].y).toEqual([1, 5, 9]);
    fixture.destroy();
  });
});
