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
import {RangeChart} from './range_chart';
import type {PlotHost, PlotlyApi} from '../plotly_loader';

describe('metric hover readout', () => {
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.unstubAllGlobals();
  });
  it('uses the rendered reduction, preserves alignment and never commits hover as selection', async () => {
    const events: Record<string, (event: Record<string, unknown>) => void> = {};
    const api = {
      react: vi.fn(async (host: PlotHost) => {
        host.on = (name, fn) => {
          events[name] = fn;
        };
        host._fullLayout = {
          xaxis: {_offset: 10, _length: 200, d2p: (n) => n * 10, p2d: (n) => n / 10},
          yaxis: {_offset: 20, _length: 100, d2p: (n) => n * 10},
          yaxis2: {_offset: 150, _length: 100, d2p: (n) => n * 10},
        };
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
    (window as Window & {Plotly?: PlotlyApi}).Plotly = api;
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
    const f = TestBed.createComponent(RangeChart);
    const row = (min: number, max: number) => [
      {
        layer: 0,
        bins: [
          {
            start: 0,
            end: 1,
            metrics: {cosine_similarity: {min, max, mean: (min + max) / 2, valid_count: 1}},
          },
          {
            start: 1,
            end: 3,
            metrics: {cosine_similarity: {min, max, mean: (min + max) / 2, valid_count: 2}},
          },
        ],
      },
    ];
    f.componentRef.setInput('label', 'KV');
    f.componentRef.setInput('layers', [0]);
    f.componentRef.setInput('bins', [
      {start: 0, end: 1},
      {start: 1, end: 3},
    ]);
    f.componentRef.setInput('rows', row(0.2, 0.8));
    f.componentRef.setInput('otherRows', row(0.3, 0.9));
    f.componentRef.setInput('visibleRange', {start: 0, end: 3});
    f.componentRef.setInput('fullRange', {start: 0, end: 3});
    f.componentRef.setInput('metric', 'cosine_similarity');
    f.componentRef.setInput('positionReduction', 'max');
    f.componentRef.setInput('format', (v: number | null) => String(v));
    f.componentRef.setInput('positionAlignment', {
      start: 0,
      pairs: [
        [10, 11],
        [12, 13],
        [14, 15],
      ],
    });
    f.componentRef.setInput('columnSteps', (column: number) =>
      column === 0
        ? {
            ref: 'Written at decode step 11, when the model reads "blue"',
            target: 'Written at decode step 12, when the model reads "blue"',
          }
        : {ref: 'Decode step 13 · reads "," · produces "blue"', target: null},
    );
    const selections: unknown[] = [];
    f.componentInstance.select.subscribe((v) => selections.push(v));
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
    await vi.waitFor(() => expect(events['plotly_hover']).toBeTypeOf('function'));
    events['plotly_hover']({points: [{customdata: {start: 0, end: 1, layer: 0, kind: 'value'}}]});
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Bin',
      value: '-',
    });
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Reference',
      value: '10',
    });
    expect(f.componentInstance.tooltip()?.content.activeColumn).toBe(1);
    // One column says which step it is on each side, in the server's words.
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Reference step',
      value: 'Written at decode step 11, when the model reads "blue"',
    });
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Target step',
      value: 'Written at decode step 12, when the model reads "blue"',
    });
    events['plotly_hover']({points: [{customdata: {start: 1, end: 3, layer: 0, kind: 'key'}}]});
    const content = f.componentInstance.tooltip()?.content;
    expect(content?.properties).toContainEqual({label: 'Bin', value: 'Maximum'});
    // A bin of several columns is of several steps: it names none.
    expect(content?.properties.map((p) => p.label)).not.toContain('Reference step');
    expect(content?.rows?.[0].values).toEqual(['0.8', '0.9']);
    expect(selections).toEqual([]);
    events['plotly_unhover']({});
    expect(f.componentInstance.tooltip()).toBeNull();
    f.destroy();
  });

  it("says why a bin without a value has none, in the server's words", async () => {
    const events: Record<string, (event: Record<string, unknown>) => void> = {};
    (window as Window & {Plotly?: PlotlyApi}).Plotly = {
      react: vi.fn(async (host: PlotHost) => {
        host.on = (name, fn) => {
          events[name] = fn;
        };
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
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
    const f = TestBed.createComponent(RangeChart);
    const why = {
      code: 'tensor_other_graph',
      side: null,
      text: 'prefill_128 step 13 produced position 1; it holds no tensor of this sync point.',
    };
    const empty = {min: null, max: null, mean: null, min_position: null, max_position: null};
    f.componentRef.setInput('label', 'Intermediate Tensors');
    f.componentRef.setInput('layers', [17]);
    f.componentRef.setInput('bins', [
      {start: 0, end: 1},
      {start: 1, end: 3},
    ]);
    f.componentRef.setInput('rows', [
      {
        layer: 17,
        bins: [
          {start: 0, end: 1, metrics: {max_abs: {...empty, valid_count: 0, reasons: [why]}}},
          {
            start: 1,
            end: 3,
            metrics: {max_abs: {...empty, max: 0.5, valid_count: 1, reasons: [why]}},
          },
        ],
      },
    ]);
    f.componentRef.setInput('visibleRange', {start: 0, end: 3});
    f.componentRef.setInput('fullRange', {start: 0, end: 3});
    f.componentRef.setInput('metric', 'max_abs');
    f.componentRef.setInput('positionReduction', 'max');
    f.componentRef.setInput('format', (v: number | null) => String(v));
    f.detectChanges();
    await f.whenStable();
    await vi.waitFor(() => expect(events['plotly_hover']).toBeTypeOf('function'));
    events['plotly_hover']({points: [{customdata: {start: 0, end: 1, layer: 17}}]});
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Unavailable',
      value: why.text,
    });
    events['plotly_hover']({points: [{customdata: {start: 1, end: 3, layer: 17}}]});
    expect(f.componentInstance.tooltip()?.content.properties).toContainEqual({
      label: 'Partly unavailable',
      value: why.text,
    });
    f.destroy();
  });
});
