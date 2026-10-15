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

export interface PlotHost extends HTMLElement {
  on(name: string, listener: (event: Record<string, unknown>) => void): void;
  _fullLayout?: {
    xaxis: {
      _offset: number;
      _length: number;
      p2d: (value: number) => number;
      d2p: (value: number) => number;
    };
    yaxis: {_offset: number; _length: number; d2p: (value: number) => number};
    yaxis2?: {_offset: number; _length: number; d2p: (value: number) => number};
  };
}
export interface PlotlyApi {
  react(
    host: HTMLElement,
    traces: unknown[],
    layout: Record<string, unknown>,
    config: Record<string, unknown>,
  ): Promise<void>;
  restyle(host: HTMLElement, update: Record<string, unknown>, indices?: number[]): Promise<void>;
  purge(host: HTMLElement): void;
}
let pending: Promise<PlotlyApi> | null = null;
export function loadPlotly(): Promise<PlotlyApi> {
  const windowWithPlotly = window as Window & {Plotly?: PlotlyApi};
  if (windowWithPlotly.Plotly) return Promise.resolve(windowWithPlotly.Plotly);
  if (!pending)
    pending = new Promise<PlotlyApi>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'assets/plotly.min.js';
      script.async = true;
      script.onload = () =>
        windowWithPlotly.Plotly
          ? resolve(windowWithPlotly.Plotly)
          : reject(new Error('Chart library unavailable'));
      script.onerror = () => {
        script.remove();
        pending = null;
        reject(new Error('Chart library could not load'));
      };
      document.head.append(script);
    });
  return pending;
}

/** Resolve theme colors to Plotly-compatible RGB (including CSS color(srgb ...)). */
export function plotlyColor(host: HTMLElement, value: string): string {
  const probe = document.createElement('span');
  probe.style.color = value;
  host.append(probe);
  const result = getComputedStyle(probe).color;
  probe.remove();
  const srgb = result.match(/^color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)/);
  return srgb
    ? `rgb(${srgb
        .slice(1, 4)
        .map((v) => Math.round(Number(v) * 255))
        .join(',')})`
    : result;
}
