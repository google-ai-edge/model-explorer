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
import {flatLineRange} from './range_chart_style';
import type {PlotHost, PlotlyApi} from '../plotly_loader';

describe('range chart paint', () => {
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  it('draws the ranges it had when Plotly started loading, then the ones set later', async () => {
    // Hold the Plotly script so the inputs can change while it loads.
    let script: HTMLScriptElement | undefined;
    const append = document.head.append.bind(document.head);
    vi.spyOn(document.head, 'append').mockImplementation((...nodes) => {
      const plotly = nodes.find(
        (node): node is HTMLScriptElement =>
          node instanceof HTMLScriptElement && node.src.includes('plotly'),
      );
      if (plotly) script = plotly;
      else append(...nodes);
    });
    const api: PlotlyApi = {
      react: vi.fn(async (host: HTMLElement) => {
        (host as PlotHost).on = () => {};
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
    const metrics = {cosine_similarity: {min: 0.5, max: 0.5, mean: 0.5, valid_count: 1}};
    f.componentRef.setInput('label', 'Test');
    f.componentRef.setInput('metric', 'cosine_similarity');
    f.componentRef.setInput('layers', [0]);
    f.componentRef.setInput('bins', [
      {start: 0, end: 1},
      {start: 1, end: 2},
    ]);
    f.componentRef.setInput('rows', [
      {
        layer: 0,
        bins: [
          {start: 0, end: 1, metrics},
          {start: 1, end: 2, metrics},
        ],
      },
    ]);
    f.componentRef.setInput('visibleRange', {start: 0, end: 2});
    f.componentRef.setInput('fullRange', {start: 0, end: 2});
    f.detectChanges();
    await vi.waitFor(() => expect(script).toBeDefined());

    // Both ranges go away before Plotly arrives, as when a caller reloads.
    f.componentRef.setInput('visibleRange', null);
    f.componentRef.setInput('fullRange', null);
    f.detectChanges();
    (window as Window & {Plotly?: PlotlyApi}).Plotly = api;
    script!.onload!(new Event('load'));
    await vi.waitFor(() => expect(vi.mocked(api.react)).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.react).mock.calls[0][2]).toMatchObject({
      uirevision: 'heatmap:cosine_similarity:0:2',
    });
    expect(f.componentInstance.error()).toBe('');

    f.componentRef.setInput('visibleRange', {start: 1, end: 2});
    f.componentRef.setInput('fullRange', {start: 0, end: 2});
    f.detectChanges();
    await vi.waitFor(() => expect(vi.mocked(api.react)).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.react).mock.calls[1][2]).toMatchObject({
      uirevision: 'heatmap:cosine_similarity:1:2',
    });
    expect(f.componentInstance.error()).toBe('');
    f.destroy();
  });
});

describe('flatLineRange', () => {
  it('draws one value everywhere clear of the axis edge, and leaves varied lines to Plotly', () => {
    // A greedy sampler: entropy 0 bits, selected probability 1, at every step.
    const zero = flatLineRange([0, 0, null, 0])!;
    expect(zero[0]).toBeLessThan(0);
    expect(zero[1]).toBeGreaterThan(0);
    const one = flatLineRange([1, 1])!;
    expect(one[0]).toBeLessThan(0);
    expect(one[1]).toBeGreaterThan(1);
    expect(flatLineRange([0, 0.5])).toBeNull();
    expect(flatLineRange([null, Number.NaN])).toBeNull();
  });
});
