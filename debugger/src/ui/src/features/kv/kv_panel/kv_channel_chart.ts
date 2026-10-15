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

import {ChartBackground} from '../../analysis/chart_background';
import {formatMetric} from '../../../shared/format/format';
import {tensorHover} from './kv_tensor_hover';
import {ChartHover, type ChartHoverPosition} from '../../analysis/chart_hover';
import {RangeAxisSelection} from '../../analysis/range_axis_selection';
import {KV_RANGE_MIN_COLUMN_WIDTH, rangeKeyboardSelection} from './kv_range_geometry';
import {stackRangeOverview} from '../../analysis/range_chart_overview';
import {rangeChartStyle, runtimeLineStyle} from '../../analysis/range_chart_style';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  ElementRef,
  NgZone,
  afterRenderEffect,
  effect,
  inject,
  input,
  model,
  output,
  signal,
  viewChild,
} from '@angular/core';
import type {KvHeadChannel, KvRangeBin} from '../../../data/contracts/kv';
import {loadPlotly, type PlotHost, type PlotlyApi} from '../../../shared/plotly_loader';
import {
  channelBins,
  channelRepresentative,
  type ChannelBin,
  type ChannelReduction,
} from './kv_channel_bins';
import {
  plotlyPositionRange,
  plotlyColumnSelection,
  heatmapHighlightBoxes,
  type HeatmapHighlightBox,
} from './kv_plotly_range';

/** Same Plotly range-slider/half-open zoom contract as the metric line chart. */
@Component({
  selector: 'kv-channel-chart',
  imports: [RangeAxisSelection, ChartHover],
  // Plotly owns the inner element and may replace its listeners during react().
  host: {'(keydown)': 'channelKey($event)'},
  hostDirectives: [{directive: ChartBackground, outputs: ['backgroundClick']}],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div
      #plot
      class="channel-chart"
      role="group"
      tabindex="0"
      [attr.aria-label]="
        dimensionLabel() +
        ' values. Hover or click to inspect. Select the range in the overview below.'
      "
    ></div>
    @if (columnAxis(); as axis) {
      <analysis-range-axis-selection
        [axis]="axis"
        [cells]="axisCells()"
        [hover]="hoveredCells()[0] ?? null"
        [selected]="axisSelection()"
        [label]="'Select a ' + dimensionLabel() + ' column'"
        (pointerHover)="hoverColumn($event)"
        (pointerLeave)="clearColumnHover()"
        (choose)="selectColumn($event)"
        (navigate)="columnKey($event)"
      />
    }
    @for (box of selectedCells(); track $index) {
      <div
        class="cell-selected"
        aria-hidden="true"
        [style.left.px]="box.left"
        [style.top.px]="box.top"
        [style.width.px]="box.width"
        [style.height.px]="box.height"
      ></div>
    }
    @for (box of hoveredCells(); track $index) {
      <div
        class="column-preview"
        aria-hidden="true"
        [style.left.px]="box.left"
        [style.top.px]="box.top"
        [style.width.px]="box.width"
        [style.height.px]="box.height"
      ></div>
    }
    @for (point of hoveredPoints(); track $index) {
      <div
        class="cell-hover point-hover"
        aria-hidden="true"
        [style.left.px]="point.left"
        [style.top.px]="point.top"
      ></div>
    }
    @if (tooltip(); as tip) {
      <analysis-chart-hover [info]="tip" />
    }
    @if (located(); as box) {
      <div
        class="cache-location"
        aria-hidden="true"
        [style.left.px]="box.left"
        [style.top.px]="box.top"
        [style.width.px]="box.width"
        [style.height.px]="box.height"
      ></div>
    }
    @if (error()) {
      <p role="alert">{{ error() }}</p>
    }`,
  styles: [
    `
      @use '../../analysis/range_chart_style' as chart;
      @include chart.overlays;
      @include chart.selection;
      :host {
        display: block;
        min-width: 0;
        height: 100%;
        position: relative;
      }
      .channel-chart {
        height: 100%;
        min-height: 260px;
        width: 100%;
      }
      p {
        font: 12px var(--me-font-family);
        color: var(--me-on-surface-color);
      }
    `,
  ],
})
export class KvChannelChart {
  private readonly background = inject(ChartBackground);
  readonly otherRows = input<readonly KvHeadChannel[] | null>(null);
  readonly messages = input<readonly string[]>([]);
  readonly dimensionLabel = input('Head dimension');
  readonly groupLabels = input<readonly string[]>(['Key cache', 'Value cache']);
  readonly comparisonAllowed = input(true);
  readonly otherSelected = input<number | null>(null);
  readonly otherSelect = output<number>();
  readonly rows = input<readonly KvHeadChannel[]>([]);
  readonly range = input.required<KvRangeBin>();
  readonly full = input.required<KvRangeBin>();
  readonly reduction = input<ChannelReduction>('mean');
  readonly capacity = input(128);
  readonly visibleRuntimes = input<readonly string[]>(['ref', 'target']);
  readonly selected = input<number | null>(null);
  readonly zoom = output<KvRangeBin>();
  readonly select = output<number>();
  readonly capacityChange = output<number>();
  readonly tooltip = signal<ChartHoverPosition | null>(null);
  readonly located = signal<{left: number; top: number; width: number; height: number} | null>(
    null,
  );
  private pendingLocation: string | null = null;
  private activePlane = 0;
  readonly channelSelection = model<(KvRangeBin & {plane: number | null}) | null>(null);
  readonly columnAxis = signal<{left: number; top: number; width: number} | null>(null);
  readonly axisCells = signal<{left: number; width: number}[]>([]);
  readonly selectedCells = signal<HeatmapHighlightBox[]>([]);
  readonly hoveredCells = signal<HeatmapHighlightBox[]>([]);
  readonly hoveredPoints = signal<{left: number; top: number}[]>([]);
  private readonly groups = computed(() => [
    this.rows(),
    ...(this.otherRows() !== null ? [this.otherRows()!] : []),
  ]);
  private readonly rowMaps = computed(() =>
    this.groups().map((rows) => new Map(rows.map((row) => [row.channel, row]))),
  );
  private readonly binnedGroups = computed(() =>
    this.groups().map((rows) => ({
      overview: channelBins(rows, this.full(), this.capacity(), this.reduction()),
      visible: channelBins(rows, this.range(), this.capacity(), this.reduction()),
    })),
  );
  private rowsInBin(bin: KvRangeBin) {
    return this.rowMaps().map((rows) => {
      const values: KvHeadChannel[] = [];
      for (let index = bin.start; index < bin.end; index++) {
        const row = rows.get(index);
        if (row) values.push(row);
      }
      return values;
    });
  }
  private overviewLabels() {
    return this.otherRows() !== null
      ? this.groupLabels().map((label) => label.replace(/ cache$/, ''))
      : [];
  }
  private highlightPoints(bin: KvRangeBin) {
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!layout) return;
    this.hoveredPoints.set(
      this.rowsInBin(bin).flatMap((rows, index) => {
        const axis = index ? layout.yaxis2 : layout.yaxis;
        const value = channelBins(rows, bin, 1, this.reduction())[0];
        if (!axis || !value) return [];
        const x = layout.xaxis.d2p((bin.start + bin.end - 1) / 2);
        return (['ref', 'target'] as const).flatMap((side) => {
          const v = value[side];
          if (!this.visibleRuntimes().includes(side) || v == null) return [];
          const y = axis.d2p(v);
          return x >= 0 && x <= layout.xaxis._length && y >= 0 && y <= axis._length
            ? [{left: layout.xaxis._offset + x, top: axis._offset + y}]
            : [];
        });
      }),
    );
  }
  readonly axisSelection = computed(() =>
    this.channelSelection()?.plane === null ? (this.selectedCells()[0] ?? null) : null,
  );
  private visibleBins: KvRangeBin[] = [];
  private boxes(selection: KvRangeBin, plane: number | null) {
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!layout) return [];
    const axes = [layout.yaxis, ...(layout.yaxis2 ? [layout.yaxis2] : [])];
    return heatmapHighlightBoxes(
      layout.xaxis,
      plane === null ? axes : axes.slice(plane, plane + 1),
      [],
      {...selection, layer: null},
    );
  }
  private columnHit(clientX: number) {
    const host = this.plot()?.nativeElement,
      axis = host?._fullLayout?.xaxis;
    return host && axis
      ? plotlyColumnSelection(
          this.visibleBins,
          axis.p2d(clientX - host.getBoundingClientRect().left - axis._offset),
        )
      : null;
  }
  hoverColumn(event: MouseEvent) {
    const hit = this.columnHit(event.clientX);
    this.hoveredCells.set(hit ? this.boxes(hit, null) : []);
    const host = this.plot()?.nativeElement;
    if (hit && host) {
      this.highlightPoints(hit);
      this.tooltip.set({
        left: event.clientX - host.getBoundingClientRect().left,
        top: this.columnAxis()!.top,
        content: this.hoverContent(hit),
      });
    } else this.clearColumnHover();
  }
  clearColumnHover() {
    this.hoveredCells.set([]);
    this.hoveredPoints.set([]);
    this.tooltip.set(null);
  }
  selectColumn(event: MouseEvent) {
    const hit = this.columnHit(event.clientX);
    if (hit) this.commitSelection(hit, null);
  }
  columnKey(event: KeyboardEvent) {
    const current = this.channelSelection();
    const next = rangeKeyboardSelection(
      this.visibleBins,
      [],
      current ? {...current, layer: null} : null,
      event.key === ' ' ? 'Enter' : event.key,
    );
    if (!next) return;
    event.preventDefault();
    event.stopPropagation();
    this.commitSelection(next, null);
  }
  private hoverContent(bin: KvRangeBin) {
    return tensorHover(this.rowsInBin(bin), bin, this.reduction(), {
      dimensionLabel: this.dimensionLabel(),
      groupLabels: this.groupLabels(),
      comparisonAllowed: this.comparisonAllowed(),
    });
  }
  private commitSelection(bin: KvRangeBin, plane: number | null, side?: 'ref' | 'target') {
    this.background.handled();
    // The highlighted interval is shared, but an extremum belongs to the clicked cache/runtime.
    const highlightPlane = this.otherRows() !== null ? null : plane;
    if (bin.end - bin.start === 1) {
      if (highlightPlane !== 1) this.select.emit(bin.start);
      if (highlightPlane !== 0 && this.otherRows() !== null) this.otherSelect.emit(bin.start);
    } else {
      const groups = this.rowsInBin(bin);
      // Axis selection has no side: use the first valid cache, preferring Ref over Target.
      for (const cache of plane === null ? groups.map((_, index) => index) : [plane]) {
        const index = channelRepresentative(groups[cache] ?? [], bin, this.reduction(), side);
        if (index === null) continue;
        (cache === 0 ? this.select : this.otherSelect).emit(index);
        break;
      }
    }
    // Parent selection handlers may clear the interval; restore it after emitting the element.
    this.channelSelection.set({start: bin.start, end: bin.end, plane: highlightPlane});
  }
  private updateSelection() {
    const selection = this.channelSelection();
    this.selectedCells.set(
      selection
        ? this.boxes(selection, selection.plane)
        : [this.selected(), this.otherSelected()].flatMap((channel, plane) =>
            channel === null ? [] : this.boxes({start: channel, end: channel + 1}, plane),
          ),
    );
  }
  private locationTimer?: ReturnType<typeof setTimeout>;
  locateKind(kind: string) {
    this.pendingLocation = kind;
    this.resolveLocation();
  }
  private resolveLocation() {
    const kind = this.pendingLocation;
    if (!kind) return;
    const host = this.plot()?.nativeElement,
      layout = host?._fullLayout;
    const axis = kind === 'value' ? layout?.yaxis2 : layout?.yaxis;
    if (!host || !layout || !axis) return;
    this.pendingLocation = null;
    host.scrollIntoView({block: 'nearest', behavior: 'smooth'});
    this.located.set({
      left: layout.xaxis._offset - 4,
      top: axis._offset - 22,
      width: layout.xaxis._length + 8,
      height: axis._length + 26,
    });
    clearTimeout(this.locationTimer);
    this.locationTimer = setTimeout(() => this.zone.run(() => this.located.set(null)), 900);
  }
  channelKey(event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const rows = (this.activePlane ? (this.otherRows() ?? []) : this.rows()).filter(
      (row) => row.channel >= this.range().start && row.channel < this.range().end,
    );
    if (!rows.length) return;
    const current = this.activePlane ? this.otherSelected() : this.selected();
    const index = Math.max(
      0,
      rows.findIndex((r) => r.channel === current),
    );
    const next =
      rows[
        event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? rows.length - 1
            : Math.max(0, Math.min(rows.length - 1, index + (event.key === 'ArrowLeft' ? -1 : 1)))
      ].channel;
    this.commitSelection({start: next, end: next + 1}, this.activePlane);
  }
  readonly error = signal('');
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  private readonly width = signal(0);
  private readonly height = signal(280);
  private readonly theme = signal(0);
  private readonly zone = inject(NgZone);
  private api?: PlotlyApi;
  private bound = false;
  private painting = false;
  private pending = false;
  private destroyed = false;
  private timer?: ReturnType<typeof setTimeout>;

  constructor() {
    effect((onCleanup) => {
      const host = this.plot()?.nativeElement;
      if (!host) return;
      const resize = new ResizeObserver(([entry]) => {
        const width = entry.contentRect.width;
        this.zone.run(() => {
          this.width.set(width);
          this.height.set(Math.max(260, entry.contentRect.height));
          this.capacityChange.emit(Math.max(16, Math.floor((width - 76) / 6)));
        });
      });
      resize.observe(host);
      const theme = new MutationObserver(() =>
        this.zone.run(() => this.theme.update((n) => n + 1)),
      );
      const media = matchMedia('(prefers-color-scheme: dark)');
      const themeChanged = () => this.zone.run(() => this.theme.update((n) => n + 1));
      media.addEventListener('change', themeChanged);
      theme.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class', 'style', 'data-theme'],
      });
      onCleanup(() => {
        resize.disconnect();
        theme.disconnect();
        media.removeEventListener('change', themeChanged);
      });
    });
    afterRenderEffect(() => {
      this.rows();
      this.otherRows();
      this.visibleRuntimes();
      this.otherSelected();
      this.messages();
      this.dimensionLabel();
      this.groupLabels();
      this.comparisonAllowed();
      this.height();
      this.range();
      this.full();
      this.reduction();
      this.capacity();
      this.selected();
      this.channelSelection();
      this.width();
      this.theme();
      this.plot();
      void this.paint();
    });
    inject(DestroyRef).onDestroy(() => {
      this.destroyed = true;
      clearTimeout(this.timer);
      clearTimeout(this.locationTimer);
      const host = this.plot()?.nativeElement;
      if (host) this.api?.purge(host);
    });
  }
  private async paint() {
    if (this.painting) {
      this.pending = true;
      return;
    }
    const host = this.plot()?.nativeElement;
    if (!host || this.width() <= 0 || this.destroyed) return;
    this.painting = true;
    try {
      this.api = await loadPlotly();
      if (this.destroyed) return;
      const range = this.range(),
        full = this.full(),
        reduction = this.reduction();
      const dual = this.otherRows() !== null;
      const style = rangeChartStyle(host, this.height(), dual);
      const traces = this.binnedGroups().flatMap(({overview, visible}, plane) => {
        const bins = [
          ...overview.filter((b) => b.end <= range.start),
          ...visible,
          ...overview.filter((b) => b.start >= range.end),
        ];
        return (['ref', 'target'] as const).map((side) => ({
          type: 'scatter',
          mode: 'lines+markers',
          yaxis: plane ? 'y2' : 'y',
          name: side === 'ref' ? 'Ref' : 'Target',
          legendgroup: side,
          visible: this.visibleRuntimes().includes(side),
          showlegend: plane === 0,
          x: bins.map((b) => (b.start + b.end - 1) / 2),
          y: bins.map((b) => b[side]),
          customdata: bins.map((b) => ({...b, plane, side})),
          connectgaps: false,
          ...runtimeLineStyle(host, side),
          text: bins.map(
            (b) =>
              `${this.groupLabels()[plane] ?? ''}<br>${this.dimensionLabel()} ${b.start}${b.end - b.start > 1 ? `–${b.end - 1} · ${reduction === 'mean' ? 'Mean' : reduction === 'min' ? 'Minimum' : 'Maximum'}` : ''}<br>${side === 'ref' ? 'Ref' : 'Target'} ${formatMetric(b[side])}`,
          ),
          hoverinfo: 'none',
        }));
      });
      await this.api.react(
        host,
        traces,
        {
          width: this.width(),
          height: this.height(),
          ...style.layout,
          dragmode: false,
          uirevision: `${range.start}:${range.end}:${reduction}`,
          xaxis: {
            ...style.xaxis(range, full, this.dimensionLabel()),
            fixedrange: true,
            ticklabelstandoff: KV_RANGE_MIN_COLUMN_WIDTH + 2,
            dtick: range.end - range.start <= 10 ? 1 : undefined,
          },
          yaxis: style.yaxis(dual ? [1 - style.fraction, 1] : [0, 1], 'Tensor value'),
          ...(dual
            ? {yaxis2: {...style.yaxis([0, style.fraction], 'Tensor value'), matches: 'y'}}
            : {}),
          annotations: [
            ...(dual ? style.labels(this.groupLabels()) : []),
            ...this.messages().flatMap((text, i) =>
              text
                ? [
                    {
                      text,
                      x: 0.5,
                      y: dual ? (i ? style.fraction / 2 : 1 - style.fraction / 2) : 0.5,
                      xref: 'paper',
                      yref: 'paper',
                      showarrow: false,
                    },
                  ]
                : [],
            ),
          ],
        },
        {displayModeBar: false, responsive: false, scrollZoom: false, doubleClick: false},
      );
      stackRangeOverview(host, this.overviewLabels());
      this.visibleBins = this.binnedGroups()[0].visible;
      const layout = host._fullLayout,
        bottom = layout?.yaxis2 ?? layout?.yaxis;
      this.columnAxis.set(
        layout && bottom
          ? {
              left: layout.xaxis._offset,
              width: layout.xaxis._length,
              top: bottom._offset + bottom._length + 2,
            }
          : null,
      );
      this.axisCells.set(
        layout
          ? this.visibleBins.map((bin) => ({
              left: layout.xaxis.d2p(bin.start - 0.5),
              width: layout.xaxis.d2p(bin.end - 0.5) - layout.xaxis.d2p(bin.start - 0.5),
            }))
          : [],
      );
      this.updateSelection();
      if (!this.bound) {
        host.on('plotly_afterplot', () => stackRangeOverview(host, this.overviewLabels()));
        this.bound = true;
        host.on('plotly_hover', (event) =>
          this.zone.run(() => {
            const point = (
              event['points'] as
                | {customdata?: ChannelBin & {plane: number; side: 'ref' | 'target'}}[]
                | undefined
            )?.[0]?.customdata;
            if (!point) return;
            this.highlightPoints(point);
            this.hoveredCells.set(
              this.boxes(point, this.otherRows() !== null ? null : point.plane),
            );
            const mouse = event['event'] as MouseEvent | undefined,
              rect = host.getBoundingClientRect();
            this.tooltip.set({
              left: (mouse?.clientX ?? rect.left + 60) - rect.left,
              top: Math.max(8, (mouse?.clientY ?? rect.top + 30) - rect.top),
              content: this.hoverContent(point),
            });
          }),
        );
        host.on('plotly_unhover', () => this.zone.run(() => this.clearColumnHover()));
        host.on('plotly_relayout', (event) => {
          if (this.painting || this.destroyed) return;
          const next = plotlyPositionRange(event, this.full());
          if (!next || (next.start === this.range().start && next.end === this.range().end)) return;
          clearTimeout(this.timer);
          this.timer = setTimeout(() => this.zone.run(() => this.zoom.emit(next)), 100);
        });
        host.on('plotly_click', (event) =>
          this.zone.run(() => {
            const bin = (
              event['points'] as
                | {customdata?: ChannelBin & {plane: number; side: 'ref' | 'target'}}[]
                | undefined
            )?.[0]?.customdata;
            if (!bin) return;
            this.activePlane = bin.plane;
            this.commitSelection(bin, bin.plane, bin.side);
          }),
        );
      }
      this.resolveLocation();
      this.error.set('');
    } catch {
      this.error.set(
        `${this.dimensionLabel()} chart unavailable. Switch the visualization to retry.`,
      );
    } finally {
      this.painting = false;
      if (this.pending && !this.destroyed) {
        this.pending = false;
        void this.paint();
      }
    }
  }
}
