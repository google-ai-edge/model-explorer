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
import {KvScatterChart} from './kv_scatter_chart';
import * as plotly from '../../../shared/plotly_loader';

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
    const f = TestBed.createComponent(KvScatterChart);
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
    expect(f.nativeElement.querySelectorAll('kv-scatter-overview')).toHaveLength(1);
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
});
