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

import {DestroyRef, NgZone, effect, inject, signal} from '@angular/core';
import {loadPlotly, type PlotHost, type PlotlyApi} from '../plotly_loader';

/** Plotly's own mode bar, resizing, wheel zoom and double click stay off in every chart. */
const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: false,
  scrollZoom: false,
  doubleClick: false,
};

/**
 * What a chart gives its PlotlySurface. `Frame` is what a paint reads before
 * Plotly loads, so a later change cannot reach a paint that already started.
 */
export interface PlotlySurfaceOptions<Frame> {
  /** The element Plotly draws into. */
  plot: () => PlotHost | undefined;
  /** The element whose content box sizes the chart; the plot element if absent. */
  measured?: () => HTMLElement | undefined;
  /** Called with each new content box of the measured element. */
  resized?: (box: DOMRectReadOnly) => void;
  /** What a paint starting now draws, read before Plotly loads; null when it cannot start. */
  ready: () => Frame | null;
  /** Draws `frame` through `react`. Never runs twice at once. */
  draw: (host: PlotHost, frame: Frame) => Promise<void>;
  /** Adds the chart's Plotly listeners, after the first `react` on a fresh plot. */
  bind: (host: PlotHost) => void;
  /** Called when the latest paint succeeded and no other paint is waiting. */
  settled?: () => void;
  /** The message shown when a paint fails. */
  failure: () => string;
}

/**
 * One Plotly chart's lifecycle, shared by the range, channel and scatter charts.
 *
 * It loads Plotly, follows the chart's size and the page theme, runs one paint
 * at a time (a paint asked for meanwhile runs once the current one ends), binds
 * Plotly's listeners once per plot and purges the plot when the component goes.
 * Create it while the component is being constructed.
 */
export class PlotlySurface<Frame> {
  /** The measured element's content box. */
  readonly size = signal({width: 0, height: 0});
  /** Changes whenever the page theme may have changed the chart's colors. */
  readonly theme = signal(0);
  /** Why the last paint failed, or ''. */
  readonly error = signal('');
  private plotly: PlotlyApi | null = null;
  private isPainting = false;
  private pending = false;
  private bound = false;
  private isDestroyed = false;
  private readonly zone = inject(NgZone);

  constructor(private readonly options: PlotlySurfaceOptions<Frame>) {
    effect((onCleanup) => {
      const element = (options.measured ?? options.plot)();
      if (!element) {
        return;
      }
      const observer = new ResizeObserver(([entry]) =>
        this.zone.run(() => {
          this.size.set({width: entry.contentRect.width, height: entry.contentRect.height});
          options.resized?.(entry.contentRect);
        }),
      );
      observer.observe(element);
      onCleanup(() => observer.disconnect());
    });
    const themeChanged = () => this.zone.run(() => this.theme.update((n) => n + 1));
    const theme = new MutationObserver(themeChanged);
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'style', 'data-theme'],
    });
    const media = matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', themeChanged);
    inject(DestroyRef).onDestroy(() => {
      this.isDestroyed = true;
      theme.disconnect();
      media.removeEventListener('change', themeChanged);
      // A paint in progress purges the plot when it ends.
      const host = options.plot();
      if (host && this.plotly && !this.isPainting) {
        this.plotly.purge(host);
      }
    });
  }

  /** Plotly, once a paint has loaded it. */
  get api(): PlotlyApi | null {
    return this.plotly;
  }
  get painting(): boolean {
    return this.isPainting;
  }
  get destroyed(): boolean {
    return this.isDestroyed;
  }

  /** Paints now, or right after the paint in progress. */
  async paint(): Promise<void> {
    if (this.isPainting) {
      this.pending = true;
      return;
    }
    const host = this.options.plot();
    if (!host || this.isDestroyed) {
      return;
    }
    const frame = this.options.ready();
    if (frame === null) {
      return;
    }
    this.isPainting = true;
    try {
      this.plotly = await loadPlotly();
      if (this.isDestroyed) {
        return;
      }
      await this.options.draw(host, frame);
      this.error.set('');
    } catch {
      if (!this.isDestroyed) {
        this.error.set(this.options.failure());
      }
    } finally {
      this.isPainting = false;
      if (this.isDestroyed) {
        this.plotly?.purge(host);
      } else if (this.pending) {
        this.pending = false;
        void this.paint();
      } else if (!this.error()) {
        this.options.settled?.();
      }
    }
  }

  /** Draws with Plotly.react, then binds the chart's listeners if this plot has none. */
  async react(host: PlotHost, traces: unknown[], layout: Record<string, unknown>): Promise<void> {
    const plotly = this.plotly ?? (await loadPlotly());
    this.plotly = plotly;
    await plotly.react(host, traces, layout, PLOTLY_CONFIG);
    if (!this.bound && !this.isDestroyed) {
      this.bound = true;
      this.options.bind(host);
    }
  }

  /** Empties the plot, so the next `react` builds it afresh and binds again. */
  purge(host: PlotHost): void {
    this.plotly?.purge(host);
    this.bound = false;
  }
}
