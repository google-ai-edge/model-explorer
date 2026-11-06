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
import {ScatterChart} from './scatter_chart';
import * as plotly from '../plotly_loader';

describe('scatter hover and committed selection', () => {
  afterEach(() => {
    delete (window as Window & {Plotly?: plotly.PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  function setup() {
    const events: Record<string, (e: Record<string, unknown>) => void> = {};
    const api = {
      react: vi.fn(
        async (host: plotly.PlotHost, _traces: unknown[], layout: Record<string, unknown>) => {
          host.on = (name, fn) => {
            events[name] = fn;
          };
          host._fullLayout = {
            xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
            yaxis: {_offset: 20, _length: 100, d2p: (n) => n * 10},
            ...(layout['yaxis2']
              ? {yaxis2: {_offset: 150, _length: 100, d2p: (n: number) => n * 10}}
              : {}),
          };
        },
      ),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
    (window as Window & {Plotly?: plotly.PlotlyApi}).Plotly = api;
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(private cb: (e: unknown[]) => void) {}
        observe() {
          this.cb([{contentRect: {width: 600, height: 400}}]);
        }
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
    const f = TestBed.createComponent(ScatterChart);
    return {f, api, events};
  }
  it('keeps selection visible during hover and forwards clicks without redrawing traces', async () => {
    const {f, api, events} = setup();
    const rows = [
      {channel: 0, ref: 1, target: 2},
      {channel: 1, ref: 2, target: 3},
    ];
    f.componentRef.setInput('rows', rows);
    f.componentRef.setInput('otherRows', rows);
    f.componentRef.setInput('range', {start: 0, end: 5});
    f.componentRef.setInput('selected', 0);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    const changes: number[] = [];
    f.componentInstance.select.subscribe((c) => {
      changes.push(c);
      f.componentRef.setInput('selected', c);
    });
    const paints = api.react.mock.calls.length;
    events['plotly_hover']({points: [{customdata: 1}]});
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    expect(f.nativeElement.querySelectorAll('.scatter-selected')).toHaveLength(2);
    expect(f.nativeElement.querySelectorAll('.scatter-hover')).toHaveLength(2);
    expect(api.restyle).not.toHaveBeenCalled();
    events['plotly_click']({points: [{customdata: 1}]});
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    expect(changes).toEqual([1]);
    expect(f.nativeElement.querySelector('.scatter-selected').getAttribute('data-channel')).toBe(
      '1',
    );
    events['plotly_unhover']({});
    f.detectChanges();
    expect(f.nativeElement.querySelectorAll('.scatter-hover')).toHaveLength(0);
    expect(f.nativeElement.querySelectorAll('.scatter-selected')).toHaveLength(2);
    expect(api.react.mock.calls.length).toBe(paints);
    f.destroy();
  });
  it('renders one vocabulary plane with generic labels and gates unavailable differences', async () => {
    const {f, api, events} = setup();
    f.componentRef.setInput('rows', [{channel: 123, ref: 1, target: 2}]);
    f.componentRef.setInput('otherRows', [{channel: 1, ref: -1000, target: 1000}]);
    f.componentRef.setInput('range', {start: 0, end: 5});
    f.componentRef.setInput('singlePlane', true);
    f.componentRef.setInput('dimensionLabel', 'Vocabulary index');
    f.componentRef.setInput('groupLabels', ['Final logits']);
    f.componentRef.setInput('comparisonAllowed', false);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    const [, traces, layout] = api.react.mock.calls.at(-1)!;
    expect(traces).toHaveLength(1);
    expect(layout['yaxis2']).toBeUndefined();
    expect(layout['yaxis']).toMatchObject({domain: [0, 1]});
    expect(layout['annotations']).toContainEqual(expect.objectContaining({text: 'Final logits'}));
    expect(f.nativeElement.querySelectorAll('scatter-overview')).toHaveLength(1);
    expect(f.nativeElement.querySelector('.scatter-chart').getAttribute('aria-label')).toContain(
      'Vocabulary index',
    );
    expect(f.componentInstance.fullRange().end).toBeLessThan(3);
    events['plotly_hover']({points: [{customdata: 123}]});
    expect(f.componentInstance.tooltip()?.content.rows?.[0]).toMatchObject({
      label: 'Final logits',
      values: ['1', '2', 'Unavailable', 'Unavailable'],
    });
    const selections: number[] = [];
    f.componentInstance.select.subscribe((index) => selections.push(index));
    events['plotly_click']({points: [{customdata: 123}]});
    expect(selections).toEqual([123]);
    f.destroy();
  });
  it('plots bounded exact representatives and restores a selected point omitted by its cell', async () => {
    const {f, api} = setup();
    const rows = Array.from({length: 5000}, (_, channel) => ({
      channel,
      ref: 1,
      target: channel === 4999 ? 1.001 : 1,
    }));
    f.componentRef.setInput('rows', rows);
    f.componentRef.setInput('range', {start: 0, end: 5});
    f.componentRef.setInput('singlePlane', true);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    expect(f.componentInstance.representation()).toEqual({
      total: 5000,
      visible: 5000,
      rendered: 1,
      grouped: true,
    });
    let trace = api.react.mock.calls.at(-1)![1][0] as {customdata: number[]};
    expect(trace.customdata).toEqual([4999]);
    f.componentRef.setInput('selected', 10);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    trace = api.react.mock.calls.at(-1)![1][0] as {customdata: number[]};
    expect(trace.customdata).toEqual([10, 4999]);
    expect(f.nativeElement.querySelector('.scatter-selected').getAttribute('data-channel')).toBe(
      '10',
    );
    f.destroy();
  });
  it('draws the points of a view, adds the selected element, and reads hover and keys from it', async () => {
    const {f, api, events} = setup();
    const point = (channel: number, ref: number, target: number) => ({
      channel,
      ref,
      target,
      delta: target - ref,
      abs_delta: Math.abs(target - ref),
      status: 'available',
    });
    const drawn = [point(2, 1, 1.5), point(9, 3, 2)];
    const selected = point(5, 2, 2.5);
    const view = {
      full: {start: 0, end: 4},
      visible: {rows: drawn, total: 5000, visible: 4500, rendered: 2, grouped: true},
      overview: drawn,
      paired: [
        {start: 2, end: 6},
        {start: 9, end: 10},
      ],
      elements: new Map([...drawn, selected].map((row) => [row.channel, row])),
    };
    f.componentRef.setInput('view', view);
    f.componentRef.setInput('singlePlane', true);
    f.componentRef.setInput('showGroupingNotice', true);
    f.componentRef.setInput('range', {start: 0, end: 4});
    f.componentRef.setInput('selected', 5);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    const trace = (api.react.mock.calls.at(-1)![1] as {customdata: number[]; x: number[]}[])[0];
    expect(trace.customdata).toEqual([2, 5, 9]);
    expect(trace.x).toEqual([1, 2, 3]);
    expect(f.componentInstance.fullRange()).toEqual({start: 0, end: 4});
    expect(f.nativeElement.textContent).toContain('Points grouped by value');
    events['plotly_hover']({points: [{customdata: 9}]});
    expect(f.componentInstance.tooltip()?.content.rows?.[0].values).toEqual(['3', '2', '-1', '1']);
    const picks: number[] = [];
    f.componentInstance.select.subscribe((channel) => picks.push(channel));
    const key = (name: string) =>
      f.nativeElement
        .querySelector('.scatter-chart')
        .dispatchEvent(new KeyboardEvent('keydown', {key: name, bubbles: true}));
    key('ArrowRight');
    key('End');
    expect(picks).toEqual([9, 9]);
    f.componentRef.setInput('waiting', true);
    f.componentRef.setInput('range', {start: 1, end: 2});
    // While waiting, a new selection is drawn over the points and range kept.
    f.componentRef.setInput('selected', 9);
    f.detectChanges();
    await f.whenStable();
    const [, traces, layout] = api.react.mock.calls.at(-1)! as [
      unknown,
      {customdata: number[]}[],
      {xaxis: {range: number[]}},
    ];
    expect(traces[0].customdata).toEqual([2, 9]);
    expect(layout.xaxis.range).toEqual([0, 4]);
  });
  it('keeps the last highlight while the values of the next selection are looked up', async () => {
    const {f} = setup();
    const point = (channel: number, ref: number, target: number) => ({
      channel,
      ref,
      target,
      delta: target - ref,
      abs_delta: Math.abs(target - ref),
      status: 'available',
    });
    const drawn = [point(2, 1, 1.5), point(9, 3, 2)];
    const view = (elements: ReturnType<typeof point>[]) => ({
      full: {start: 0, end: 4},
      visible: {rows: drawn, total: 3, visible: 3, rendered: 2, grouped: false},
      overview: drawn,
      paired: [{start: 2, end: 10}],
      elements: new Map(elements.map((row) => [row.channel, row])),
    });
    f.componentRef.setInput('view', view(drawn));
    f.componentRef.setInput('singlePlane', true);
    f.componentRef.setInput('range', {start: 0, end: 4});
    f.componentRef.setInput('selected', 2);
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    const highlight = f.componentInstance.selectedPoints();
    expect(highlight).toHaveLength(1);
    // Element 5's values are not known yet: the highlight stays where it was.
    f.componentRef.setInput('selected', 5);
    f.detectChanges();
    expect(f.componentInstance.selectedPoints()).toEqual(highlight);
    f.componentRef.setInput('view', view([...drawn, point(5, 2, 2.5)]));
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    expect(f.componentInstance.selectedPoints()).not.toEqual(highlight);
    expect(f.componentInstance.selectedPoints()).toHaveLength(1);
  });
});
