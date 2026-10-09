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

import {
  afterNextRender,
  Component,
  computed,
  effect,
  ElementRef,
  inject,
  input,
  NgZone,
  OnDestroy,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  TokenAnalysisPair,
  TokenMetricKey,
} from '../../../data/contracts/token_analysis';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {formatFixed} from '../../../shared/format/format';
import {
  loadTokenPlotly,
  PlotHost,
  PlotlyApi,
} from '../../../shared/plotly_loader';
import {AlignmentMode, ForkPair, TokenPair} from '../token_alignment';
import {pairKey, presentValue, TOKEN_COLOR_MAX} from '../token_analysis';
import {TOKEN_METRICS} from '../token_metric_metadata';
import {
  isSideTrendMetric,
  normalizeTrendRange,
  sampleTokenTrend,
  SIDE_TREND_METRICS,
  TokenTrendState,
  trendTickStep,
  trendTokenLine,
} from '../token_trends';

/**
 * Custom properties keep `light-dark(...)` unresolved in getComputedStyle; a probe
 * element that uses the token as its color yields the concrete value Plotly needs.
 */
function resolvedColor(host: HTMLElement, token: string): string {
  let probe = host.querySelector<HTMLElement>(':scope > .color-probe');
  if (!probe) {
    probe = host.ownerDocument.createElement('span');
    probe.className = 'color-probe';
    probe.setAttribute('aria-hidden', 'true');
    probe.style.cssText =
      'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none';
    host.appendChild(probe);
  }
  probe.style.color = `var(${token})`;
  return getComputedStyle(probe).color;
}
/** Candidates' Plotly layout, range/reset and picking policy in an Angular lifecycle. */
@Component({
  selector: 'token-trends',
  imports: [ConfigPicker, MatIconModule, MatTooltipModule],
  template: ` <section
    class="turnTrendPanel"
    [attr.aria-label]="'Turn ' + turn() + ' comparison trends'"
  >
    <div
      class="trendGraphTools"
      role="group"
      [attr.aria-label]="'Chart controls for turn ' + turn()"
    >
      <span>Turn {{ turn() }}</span>
      <div class="trendMetricChoice">
        <span>Metric</span
        ><config-picker
          [compact]="true"
          [label]="'Metric for turn ' + turn()"
          plural="metrics"
          [popupWidth]="360"
          [value]="metric().label"
          [options]="options"
          (valueChange)="changeMetric($event)"
        />
      </div>
      <label
        >Positions
        <input
          type="number"
          min="0"
          [max]="maximum() - 1"
          step="1"
          [value]="range()[0]"
          [attr.aria-label]="'Start position for turn ' + turn()"
          (change)="commit($event, false)"
          (keydown)="inputKey($event, false)" /></label
      ><span>–</span>
      <input
        type="number"
        min="1"
        [max]="maximum()"
        step="1"
        [value]="range()[1]"
        [attr.aria-label]="'End position for turn ' + turn()"
        (change)="commit($event, true)"
        (keydown)="inputKey($event, true)"
      />
      <button
        type="button"
        [disabled]="!customRange()"
        [attr.aria-label]="'Reset chart view for turn ' + turn()"
        matTooltip="Reset view"
        (click)="reset()"
      >
        <mat-icon>center_focus_strong</mat-icon>
      </button>
    </div>
    <div
      #plot
      class="turnPlotly"
      tabindex="0"
      role="group"
      [attr.aria-label]="
        'Turn ' +
        turn() +
        ' comparison chart. Drag to zoom. Use left and right arrows to select positions.'
      "
    ></div>
    @if (error()) {
      <p role="alert">{{ error() }} <button type="button" (click)="retry()">Retry chart</button></p>
    }
    @if (loading()) {
      <p role="status">Loading captured metrics…</p>
    }
    @if (outside()) {
      <div class="trendSelectionNotice">
        <span>Selected position is outside the visible range.</span
        ><button type="button" (click)="reveal()">Show selection</button>
      </div>
    }
  </section>`,
  styleUrl: './token_trends.scss',
})
export class TokenTrends implements OnDestroy {
  readonly turn = input.required<number>();
  readonly rows = input.required<TokenPair[]>();
  readonly analysis = input.required<Map<string, TokenAnalysisPair>>();
  readonly settings = input.required<TokenTrendState>();
  readonly selected = input<number | null>(null);
  readonly loading = input(false);
  /** The turn's first differing generation step, paired by step (see `forkPair`). */
  readonly fork = input<ForkPair | null>(null);
  readonly alignment = input<AlignmentMode>('content');
  readonly settingsChange = output<TokenTrendState>();
  readonly selectPosition = output<number>();
  readonly options = [...TOKEN_METRICS, ...SIDE_TREND_METRICS].map(
    (m) => m.label,
  );
  readonly maximum = computed(() => Math.max(1, this.rows().length - 1));
  readonly metric = computed(
    () =>
      [...TOKEN_METRICS, ...SIDE_TREND_METRICS].find(
        (m) => m.key === this.settings().metric,
      ) ?? TOKEN_METRICS[0],
  );
  readonly range = computed(() =>
    normalizeTrendRange(this.rows().length, [
      this.settings().start,
      this.settings().end,
    ]),
  );
  readonly customRange = computed(
    () => this.range()[0] !== 0 || this.range()[1] !== this.maximum(),
  );
  readonly outside = computed(
    () =>
      this.selected() !== null &&
      (this.selected()! < this.range()[0] ||
        this.selected()! > this.range()[1]),
  );
  readonly series = computed(() => {
    const data = this.analysis(),
      key = this.metric().key,
      turn = this.turn();
    if (isSideTrendMetric(key)) return [];
    return this.rows().map((row) => {
      const value = data.get(pairKey(turn, this.pairAt(row)))?.metrics[key]
        ?.value;
      return value == null ? null : presentValue(key, value);
    });
  });
  /** The pair a position compares: its row, except that the first differing row compares the
   *  two tokens of its generation step when the text alignment left one side empty. Paired
   *  metrics exist for that pair; per-runtime lines keep every token at its own row. */
  private pairAt(row: TokenPair): TokenPair {
    const fork = this.fork();
    return fork?.filled && fork.index === row.index ? fork.pair : row;
  }
  /** One series per runtime for a per-runtime metric: each token's own value, null at a gap. */
  readonly sideSeries = computed(() => {
    const key = this.metric().key;
    if (!isSideTrendMetric(key)) return null;
    const data = this.analysis(),
      turn = this.turn();
    const pick = (side: 'ref' | 'target') =>
      this.rows().map((pair) =>
        pair[side]
          ? (data.get(pairKey(turn, pair))?.distribution?.[side]?.[key] ?? null)
          : null,
      );
    return {ref: pick('ref'), target: pick('target')};
  });
  /** The first aligned position where the two runtimes produced different tokens. */
  readonly firstDivergence = computed(() =>
    this.rows().findIndex((row) => row.match === false),
  );
  readonly width = signal(800);
  readonly sampled = computed(() =>
    sampleTokenTrend(this.series(), this.range(), this.width()),
  );
  readonly error = signal('');
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  private readonly zone = inject(NgZone);
  private api: PlotlyApi | null = null;
  private resizeObserver?: ResizeObserver;
  private themeObserver?: MutationObserver;
  private controller = new AbortController();
  private destroyed = false;
  private frame = 0;
  private timer = 0;
  private painting = false;
  private queued = false;
  private bound = false;
  constructor() {
    effect(() => {
      this.rows();
      this.analysis();
      this.settings();
      this.selected();
      this.loading();
      this.fork();
      this.alignment();
      this.schedule();
    });
    afterNextRender(() => {
      const host = this.plot()?.nativeElement;
      if (!host) return;
      this.zone.runOutsideAngular(() => {
        this.resizeObserver = new ResizeObserver(() => {
          this.width.set(Math.max(200, Math.floor(host.clientWidth)));
          this.schedule();
        });
        this.resizeObserver.observe(host);
        this.themeObserver = new MutationObserver(() => this.schedule());
        this.themeObserver.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['data-theme', 'style'],
        });
        matchMedia('(prefers-color-scheme: dark)').addEventListener(
          'change',
          () => this.schedule(),
          {signal: this.controller.signal},
        );
        this.schedule();
      });
    });
  }
  private schedule() {
    if (this.destroyed) return;
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => void this.paint());
  }
  private setRange(range: number[]) {
    const [start, end] = normalizeTrendRange(this.rows().length, range);
    this.settingsChange.emit({...this.settings(), start, end});
  }
  changeMetric(label: string) {
    const metric = [...TOKEN_METRICS, ...SIDE_TREND_METRICS].find(
      (m) => m.label === label,
    )?.key;
    if (metric) this.settingsChange.emit({...this.settings(), metric});
  }
  commit(event: Event, end: boolean) {
    const input = event.target as HTMLInputElement,
      value = input.value.trim() === '' ? NaN : Number(input.value),
      range = this.range();
    if (Number.isFinite(value))
      this.setRange(
        end
          ? [range[0], Math.max(range[0] + 1, value)]
          : [Math.min(value, range[1] - 1), range[1]],
      );
    input.value = String(this.range()[end ? 1 : 0]);
  }
  inputKey(event: KeyboardEvent, end: boolean) {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.commit(event, end);
      (event.target as HTMLInputElement).blur();
    }
    if (event.key === 'Escape') {
      event.stopPropagation();
      (event.target as HTMLInputElement).value = String(
        this.range()[end ? 1 : 0],
      );
    }
  }
  reset() {
    this.setRange([0, this.maximum()]);
  }
  reveal() {
    const idx = this.selected();
    if (idx === null) return;
    const span = this.range()[1] - this.range()[0],
      lo = Math.max(
        0,
        Math.min(this.maximum() - span, idx - Math.floor(span / 2)),
      );
    this.setRange([lo, lo + span]);
  }
  private pick(index: number) {
    if (!this.rows().length) return;
    this.zone.run(() =>
      this.selectPosition.emit(
        Math.max(0, Math.min(this.rows().length - 1, Math.round(index))),
      ),
    );
    this.plot()?.nativeElement.focus({preventScroll: true});
  }
  plotKey(event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    this.pick(
      event.key === 'Home'
        ? this.range()[0]
        : event.key === 'End'
          ? this.range()[1]
          : (this.selected() ?? this.range()[0]) +
            (event.key === 'ArrowLeft' ? -1 : 1),
    );
  }
  retry() {
    this.error.set('');
    this.schedule();
  }
  private bind(host: PlotHost) {
    this.bound = true;
    host.addEventListener('keydown', (event) => this.plotKey(event), {
      capture: true,
      signal: this.controller.signal,
    });
    host.on('plotly_relayout', (event) => {
      if (this.painting || this.destroyed) return;
      let range = event['xaxis.range'];
      if (event['xaxis.range[0]'] != null)
        range = [event['xaxis.range[0]'], event['xaxis.range[1]']];
      if (event['xaxis.autorange']) this.zone.run(() => this.reset());
      else if (Array.isArray(range) && range.every(Number.isFinite))
        this.zone.run(() => this.setRange(range as number[]));
    });
    let press: number[] | null = null;
    const options = {capture: true, signal: this.controller.signal};
    host.addEventListener(
      'pointerdown',
      (e) => {
        if (e.button === 0) {
          clearTimeout(this.timer);
          press = [e.clientX, e.clientY];
        }
      },
      options,
    );
    window.addEventListener(
      'pointerup',
      (e) => {
        if (!press) return;
        const distance = Math.hypot(e.clientX - press[0], e.clientY - press[1]);
        press = null;
        if (distance > 4) return;
        const f = host._fullLayout;
        if (!f) return;
        const rect = host.getBoundingClientRect(),
          x = e.clientX - rect.left,
          y = e.clientY - rect.top,
          axis = f.xaxis;
        if (
          y >= f.yaxis._offset &&
          y <= f.yaxis._offset + f.yaxis._length &&
          x >= axis._offset &&
          x <= axis._offset + axis._length
        ) {
          const index = axis.p2d(x - axis._offset);
          this.timer = window.setTimeout(() => {
            if (!this.destroyed) this.pick(index);
          }, 320);
        }
      },
      options,
    );
    window.addEventListener(
      'dblclick',
      () => clearTimeout(this.timer),
      options,
    );
    host.on('plotly_doubleclick', () => clearTimeout(this.timer));
    host.addEventListener(
      'pointercancel',
      () => {
        press = null;
        clearTimeout(this.timer);
      },
      options,
    );
  }
  private async paint() {
    const host = this.plot()?.nativeElement;
    if (!host || !host.isConnected || this.destroyed) return;
    if (this.painting) {
      this.queued = true;
      return;
    }
    this.painting = true;
    try {
      this.api = await loadTokenPlotly();
      if (this.destroyed) return;
      const style = getComputedStyle(host),
        css = (name: string) => style.getPropertyValue(name).trim(),
        color = (name: string) => resolvedColor(host, name),
        mode = document.documentElement.dataset['theme'],
        dark =
          mode === 'dark' ||
          (mode !== 'light' &&
            matchMedia('(prefers-color-scheme: dark)').matches);
      const metric = this.metric(),
        key = metric.key,
        rows = this.rows(),
        axisTitle =
          'Aligned position · ' +
          (this.alignment() === 'steps' ? 'Steps' : 'Context') +
          ' alignment',
        position = (x: number) => 'Aligned position ' + x,
        range = this.range(),
        source = this.series(),
        sample = this.sampled(),
        idx = this.selected(),
        sides = this.sideSeries(),
        palette = key === 'kl' || key === 'js';
      const family = palette ? 'semantic' : 'activation';
      const colors = [
        color(`--heat-${family}-low`),
        color(`--heat-${family}-high`),
      ];
      const pointOutline = color(`--heat-${family}-point`),
        muted = color('--me-on-surface-variant-color'),
        rule = color('--me-outline-variant-color'),
        accent = color('--me-primary-color'),
        surface = color('--trend-surface'),
        unit = isSideTrendMetric(key)
          ? this.metric().unit
          : key === 'kl'
            ? 'nats'
            : key === 'js'
              ? 'bits'
              : key === 'relative_l2'
                ? '%'
                : key === 'norm_ratio'
                  ? '×'
                  : '';
      const pointScale = [
          [0, colors[0]],
          [1, colors[1]],
        ],
        pointMax = isSideTrendMetric(key) ? 1 : TOKEN_COLOR_MAX[key],
        shapes: unknown[] = [],
        annotations: unknown[] = [];
      const axis = {
        tickfont: {family: css('--me-font-family'), size: 12},
        showline: true,
        linecolor: muted,
        linewidth: 1,
        ticks: 'outside',
        ticklen: 4,
        tickcolor: muted,
        zeroline: false,
      };
      const sideSamples = sides
        ? {
            ref: sampleTokenTrend(sides.ref, range, this.width()),
            target: sampleTokenTrend(sides.target, range, this.width()),
          }
        : null;
      const available = sideSamples
          ? sideSamples.ref.available + sideSamples.target.available
          : sample.available,
        peak = sideSamples
          ? Math.max(sideSamples.ref.max, sideSamples.target.max)
          : sample.max;
      if (!available)
        annotations.push({
          xref: 'paper',
          yref: 'paper',
          x: 0.5,
          y: 0.5,
          text: this.loading()
            ? 'Loading captured metrics…'
            : key === 'kl' || key === 'js' || sides
              ? 'Full-vocabulary logits not captured' +
                (sides ? '' : ' or not comparable')
              : 'Activation metric not captured or not comparable',
          showarrow: false,
        });
      if (unit || metric.reading)
        annotations.push({
          xref: 'paper',
          yref: 'paper',
          x: 0,
          y: 1,
          xanchor: 'left',
          yanchor: 'bottom',
          yshift: 8,
          text: [unit, metric.reading].filter(Boolean).join(' · '),
          showarrow: false,
        });
      for (const side of ['ref', 'target'] as const) {
        const boundary = this.rows().findIndex(
          (r) => r[side] && (r[side]!.phase ?? 'response') === 'response',
        );
        if (boundary > 0)
          shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: boundary - 0.5,
            x1: boundary - 0.5,
            y0: 0,
            y1: 1,
            line: {color: rule, width: 1, dash: 'dot'},
          });
      }
      // From the first diverging token on the two runtimes read different histories: paired
      // metrics end there, per-runtime lines go their own ways.
      const diverged = this.firstDivergence();
      if (diverged >= 0) {
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: diverged,
          x1: diverged,
          y0: 0,
          y1: 1,
          line: {color: muted, width: 1, dash: 'dash'},
        });
        annotations.push({
          xref: 'x',
          yref: 'paper',
          x: diverged,
          y: 1,
          xanchor: 'left',
          yanchor: 'bottom',
          xshift: 4,
          text: 'First divergence',
          showarrow: false,
          font: {size: 11, color: muted},
        });
      }
      if (idx !== null)
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: idx,
          x1: idx,
          y0: 0,
          y1: 1,
          line: {color: accent, width: 1},
        });
      const selected = idx === null ? null : source[idx],
        marker = {
          colorscale: pointScale,
          cmin: 0,
          cmax: pointMax,
          showscale: false,
          symbol: 'circle',
        };
      const percent = sides && key !== 'entropy';
      const sideTraces = sideSamples
        ? (['ref', 'target'] as const).flatMap((side) => {
            const data = sideSamples[side],
              line = color(side === 'ref' ? '--trend-ref' : '--trend-target'),
              name = side === 'ref' ? 'Reference' : 'Target',
              chosen = idx === null ? null : sides![side][idx];
            const text = (v: number | null | undefined) =>
              v == null
                ? '—'
                : percent
                  ? (v * 100).toFixed(1) + '%'
                  : formatFixed(v, 3) + ' bits';
            return [
              {
                type: 'scatter',
                mode: data.x.length > 1200 ? 'lines' : 'lines+markers',
                x: data.x,
                y: data.y,
                connectgaps: false,
                line: {color: line, width: 1.5},
                marker: {size: 5, color: line},
                customdata: data.x.map((x) =>
                  [
                    position(x),
                    trendTokenLine(
                      'Reference',
                      rows[x]?.ref,
                      text(sides!.ref[x]),
                    ),
                    trendTokenLine(
                      'Target',
                      rows[x]?.target,
                      text(sides!.target[x]),
                    ),
                  ].join('<br>'),
                ),
                hovertemplate: '%{customdata}<extra></extra>',
                name,
              },
              {
                type: 'scatter',
                mode: 'markers',
                x: idx === null || chosen == null ? [] : [idx],
                y: chosen == null ? [] : [chosen],
                marker: {
                  size: 10,
                  color: line,
                  line: {width: 2, color: accent},
                },
                hoverinfo: 'skip',
                showlegend: false,
              },
            ];
          })
        : null;
      const traces = sideTraces ?? [
        {
          type: 'scatter',
          mode: sample.x.length > 1200 ? 'lines' : 'lines+markers',
          x: sample.x,
          y: sample.y,
          connectgaps: false,
          line: {color: muted, width: 1},
          marker: {
            ...marker,
            size: 6,
            color: sample.y,
            line: {width: dark ? 1.2 : 0.6, color: pointOutline},
          },
          customdata: sample.x.map((x, i) => {
            const v = sample.y[i],
              row = rows[x],
              pair = row ? this.pairAt(row) : undefined;
            return [
              position(x),
              trendTokenLine('Reference', pair?.ref),
              trendTokenLine('Target', pair?.target),
              ...(pair && pair !== row ? ['Paired by generation step'] : []),
              metric.label +
                ': <b>' +
                (v === null
                  ? '—'
                  : formatFixed(
                      v,
                      key === 'relative_l2' || key === 'norm_ratio' ? 2 : 4,
                    ) + (unit ? ' ' + unit : '')) +
                '</b>',
            ].join('<br>');
          }),
          hovertemplate: '%{customdata}<extra></extra>',
          name: metric.label,
        },
        {
          type: 'scatter',
          mode: 'markers',
          x: idx === null || selected == null ? [] : [idx],
          y: selected == null ? [] : [selected],
          marker: {
            ...marker,
            size: 10,
            color: selected == null ? [] : [selected],
            line: {width: 2, color: accent},
          },
          hoverinfo: 'skip',
          showlegend: false,
        },
      ];
      await this.api.react(
        host,
        traces,
        {
          width: Math.max(200, host.clientWidth),
          height: 360,
          paper_bgcolor: surface,
          plot_bgcolor: surface,
          font: {family: css('--me-font-family'), size: 12, color: muted},
          margin: {l: 76, r: 24, t: 30, b: 50},
          showlegend: !!sides,
          legend: {
            orientation: 'h',
            x: 1,
            xanchor: 'right',
            y: 1,
            yanchor: 'bottom',
          },
          hovermode: 'closest',
          hoverdistance: 14,
          hoverlabel: {
            // Resolved colors: an unresolved theme token makes Plotly fall back to the trace color.
            bgcolor: color('--me-surface-color'),
            bordercolor: rule,
            font: {
              family: css('--me-font-family'),
              size: 12,
              color: color('--me-on-surface-color'),
            },
            align: 'left',
          },
          dragmode: 'zoom',
          uirevision: range.join(':') + key,
          shapes,
          annotations,
          xaxis: {
            ...axis,
            range,
            dtick: trendTickStep(range),
            tickformat: 'd',
            showgrid: false,
            title: {text: axisTitle, standoff: 6},
            automargin: true,
            rangeslider: {
              visible: true,
              thickness: 0.12,
              range: [0, this.maximum()],
              bgcolor: surface,
              bordercolor: rule,
              borderwidth: 1,
              yaxis: {rangemode: 'auto'},
            },
          },
          yaxis: {
            ...axis,
            range: [
              0,
              percent ? 1.05 : available ? Math.max(1e-8, peak) * 1.1 : 1,
            ],
            fixedrange: true,
            gridcolor: rule,
            tickformat: percent ? '.0%' : peak < 0.01 ? '.2e' : '.3~g',
          },
        },
        {
          displayModeBar: false,
          responsive: true,
          scrollZoom: false,
          doubleClick: 'reset',
        },
      );
      if (!this.destroyed && !this.bound) this.bind(host);
      this.error.set('');
    } catch {
      if (!this.destroyed) this.error.set('Chart could not render.');
    } finally {
      this.painting = false;
      if (this.queued) {
        this.queued = false;
        this.schedule();
      }
    }
  }
  ngOnDestroy() {
    this.destroyed = true;
    cancelAnimationFrame(this.frame);
    clearTimeout(this.timer);
    this.controller.abort();
    this.resizeObserver?.disconnect();
    this.themeObserver?.disconnect();
    const host = this.plot()?.nativeElement;
    if (host) this.api?.purge(host);
  }
}
