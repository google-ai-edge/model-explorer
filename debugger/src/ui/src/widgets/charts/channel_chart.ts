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

import {ChartBackground} from './chart_background';
import {formatMetric} from '../format/format';
import {binHover} from './tensor_hover';
import {ChartHover, type ChartHoverContent, type ChartHoverPosition} from './chart_hover';
import {RangeAxisSelection} from './range_axis_selection';
import {RANGE_MIN_COLUMN_WIDTH, rangeKeyboardSelection} from './range_geometry';
import {stackRangeOverview} from './range_chart_overview';
import {rangeChartStyle, runtimeLineStyle} from './range_chart_style';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  ElementRef,
  NgZone,
  afterRenderEffect,
  inject,
  input,
  model,
  output,
  signal,
  viewChild,
} from '@angular/core';
import type {KvHeadChannel, KvRangeBin} from '../../server/contracts/kv';
import type {PlotHost} from '../plotly_loader';
import {PlotlySurface} from './plotly_surface';
import {
  channelBins,
  channelRepresentative,
  reducedRepresentative,
  type ChannelBin,
  type ChannelReduction,
  type ReducedChannelBins,
} from './channel_bins';
import {
  plotlyPositionRange,
  plotlyColumnSelection,
  heatmapHighlightBoxes,
  type HeatmapHighlightBox,
} from './plotly_range';

/** What a paint draws of the chart's data: its bins, of every group, over its range. */
interface ChannelDrawing {
  groups: readonly {overview: readonly ChannelBin[]; visible: readonly ChannelBin[]}[];
  range: KvRangeBin;
  full: KvRangeBin;
  reduction: ChannelReduction;
}

/** Same Plotly range-slider/half-open zoom contract as the metric line chart. */
@Component({
  selector: 'channel-chart',
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
      <range-axis-selection
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
      <chart-hover [info]="tip" />
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
      @use './range_chart_style' as chart;
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
export class ChannelChart {
  private readonly background = inject(ChartBackground);
  readonly otherRows = input<readonly KvHeadChannel[] | null>(null);
  readonly messages = input<readonly string[]>([]);
  readonly dimensionLabel = input('Head dimension');
  readonly groupLabels = input<readonly string[]>(['Key cache', 'Value cache']);
  readonly comparisonAllowed = input(true);
  readonly otherSelected = input<number | null>(null);
  readonly otherSelect = output<number>();
  readonly rows = input<readonly KvHeadChannel[]>([]);
  /** Bins another reduced, drawn instead of `rows`' own: one group, whose elements are every index
   *  of `full`. `range` and `reduction` must be those the bins were reduced for. */
  readonly bins = input<ReducedChannelBins | null>(null);
  /** While true the chart keeps what it last drew: the bins of its new view are on their way. */
  readonly waiting = input(false);
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
  private readonly groups = computed(() => {
    const other = this.otherRows();
    return [this.rows(), ...(other !== null ? [other] : [])];
  });
  private readonly rowMaps = computed(() =>
    this.groups().map((rows) => new Map(rows.map((row) => [row.channel, row]))),
  );
  private readonly binnedGroups = computed(() => {
    const reduced = this.bins();
    return reduced
      ? [reduced]
      : this.groups().map((rows) => ({
          overview: channelBins(rows, this.full(), this.capacity(), this.reduction()),
          visible: channelBins(rows, this.range(), this.capacity(), this.reduction()),
        }));
  });
  /** The reduced bins by their interval. */
  private readonly reducedBins = computed(() => {
    const reduced = this.bins();
    return reduced
      ? new Map(
          [...reduced.overview, ...reduced.visible].map((bin) => [`${bin.start}:${bin.end}`, bin]),
        )
      : null;
  });
  /** Each group's values of a bin, as its own reduction of the bin gives them. */
  private binValues(bin: KvRangeBin): (ChannelBin | undefined)[] {
    const reduced = this.reducedBins();
    return reduced
      ? [reduced.get(`${bin.start}:${bin.end}`)]
      : this.rowsInBin(bin).map((rows) => channelBins(rows, bin, 1, this.reduction())[0]);
  }
  private rowsInBin(bin: KvRangeBin): KvHeadChannel[][] {
    return this.rowMaps().map((rows) => {
      const values: KvHeadChannel[] = [];
      for (let index = bin.start; index < bin.end; index++) {
        const row = rows.get(index);
        if (row) {
          values.push(row);
        }
      }
      return values;
    });
  }
  private overviewLabels(): string[] {
    return this.otherRows() !== null
      ? this.groupLabels().map((label) => label.replace(/ cache$/, ''))
      : [];
  }
  private highlightPoints(bin: KvRangeBin): void {
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!layout) {
      return;
    }
    this.hoveredPoints.set(
      this.binValues(bin).flatMap((value, index) => {
        const axis = index ? layout.yaxis2 : layout.yaxis;
        if (!axis || !value) {
          return [];
        }
        const x = layout.xaxis.d2p((bin.start + bin.end - 1) / 2);
        return (['ref', 'target'] as const).flatMap((side) => {
          const v = value[side];
          if (!this.visibleRuntimes().includes(side) || v == null) {
            return [];
          }
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
  private visibleBins: readonly KvRangeBin[] = [];
  private boxes(selection: KvRangeBin, plane: number | null): HeatmapHighlightBox[] {
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!layout) {
      return [];
    }
    const axes = [layout.yaxis, ...(layout.yaxis2 ? [layout.yaxis2] : [])];
    return heatmapHighlightBoxes(
      layout.xaxis,
      plane === null ? axes : axes.slice(plane, plane + 1),
      [],
      {...selection, layer: null},
    );
  }
  private columnHit(clientX: number): KvRangeBin | null {
    const host = this.plot()?.nativeElement;
    const axis = host?._fullLayout?.xaxis;
    return host && axis
      ? plotlyColumnSelection(
          this.visibleBins,
          axis.p2d(clientX - host.getBoundingClientRect().left - axis._offset),
        )
      : null;
  }
  hoverColumn(event: MouseEvent): void {
    const axis = this.columnAxis();
    if (!axis) {
      return;
    }
    const hit = this.columnHit(event.clientX);
    this.hoveredCells.set(hit ? this.boxes(hit, null) : []);
    const host = this.plot()?.nativeElement;
    if (hit && host) {
      this.highlightPoints(hit);
      this.tooltip.set({
        left: event.clientX - host.getBoundingClientRect().left,
        top: axis.top,
        content: this.hoverContent(hit),
      });
    } else {
      this.clearColumnHover();
    }
  }
  clearColumnHover(): void {
    this.hoveredCells.set([]);
    this.hoveredPoints.set([]);
    this.tooltip.set(null);
  }
  selectColumn(event: MouseEvent): void {
    const hit = this.columnHit(event.clientX);
    if (hit) {
      this.commitSelection(hit, null);
    }
  }
  columnKey(event: KeyboardEvent): void {
    const current = this.channelSelection();
    const next = rangeKeyboardSelection(
      this.visibleBins,
      [],
      current ? {...current, layer: null} : null,
      event.key === ' ' ? 'Enter' : event.key,
    );
    if (!next) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.commitSelection(next, null);
  }
  private hoverContent(bin: KvRangeBin): ChartHoverContent {
    return binHover(this.binValues(bin), bin, this.reduction(), {
      dimensionLabel: this.dimensionLabel(),
      groupLabels: this.groupLabels(),
      comparisonAllowed: this.comparisonAllowed(),
    });
  }
  private commitSelection(bin: KvRangeBin, plane: number | null, side?: 'ref' | 'target'): void {
    this.background.handled();
    // The highlighted interval is shared, but an extremum belongs to the clicked cache/runtime.
    const highlightPlane = this.otherRows() !== null ? null : plane;
    const reducedBins = this.reducedBins();
    if (bin.end - bin.start === 1) {
      if (highlightPlane !== 1) {
        this.select.emit(bin.start);
      }
      if (highlightPlane !== 0 && this.otherRows() !== null) {
        this.otherSelect.emit(bin.start);
      }
    } else if (reducedBins) {
      const reduced = reducedBins.get(`${bin.start}:${bin.end}`);
      const index = reduced ? reducedRepresentative(reduced, side) : null;
      if (index !== null) {
        this.select.emit(index);
      }
    } else {
      const groups = this.rowsInBin(bin);
      // Axis selection has no side: use the first valid cache, preferring Ref over Target.
      for (const cache of plane === null ? groups.map((_, index) => index) : [plane]) {
        const index = channelRepresentative(groups[cache] ?? [], bin, this.reduction(), side);
        if (index === null) {
          continue;
        }
        (cache === 0 ? this.select : this.otherSelect).emit(index);
        break;
      }
    }
    // Parent selection handlers may clear the interval; restore it after emitting the element.
    this.channelSelection.set({start: bin.start, end: bin.end, plane: highlightPlane});
  }
  private updateSelection(): void {
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
  locateKind(kind: string): void {
    this.pendingLocation = kind;
    this.resolveLocation();
  }
  private resolveLocation(): void {
    const kind = this.pendingLocation;
    if (!kind) {
      return;
    }
    const host = this.plot()?.nativeElement;
    const layout = host?._fullLayout;
    const axis = kind === 'value' ? layout?.yaxis2 : layout?.yaxis;
    if (!host || !layout || !axis) {
      return;
    }
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
  channelKey(event: KeyboardEvent): void {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const range = this.range();
    // Reduced bins stand for every index of the range.
    const channels = this.bins()
      ? {
          length: range.end - range.start,
          at: (index: number) => range.start + index,
          indexOf: (channel: number | null) =>
            channel !== null && channel >= range.start && channel < range.end
              ? channel - range.start
              : -1,
        }
      : this.channelsIn(range);
    if (!channels.length) {
      return;
    }
    const current = this.activePlane ? this.otherSelected() : this.selected();
    const index = Math.max(0, channels.indexOf(current));
    const next = channels.at(
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? channels.length - 1
          : Math.max(
              0,
              Math.min(channels.length - 1, index + (event.key === 'ArrowLeft' ? -1 : 1)),
            ),
    );
    this.commitSelection({start: next, end: next + 1}, this.activePlane);
  }
  /** The active plane's channels in a range, in order. */
  private channelsIn(range: KvRangeBin): {
    length: number;
    at: (index: number) => number;
    indexOf: (channel: number | null) => number;
  } {
    const channels = (this.activePlane ? (this.otherRows() ?? []) : this.rows())
      .filter((row) => row.channel >= range.start && row.channel < range.end)
      .map((row) => row.channel);
    return {
      length: channels.length,
      at: (index: number) => channels[index],
      indexOf: (channel: number | null) => channels.findIndex((c) => c === channel),
    };
  }
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  /** The data the chart last drew: kept while it is waiting, when a paint (a new selection, size or
   *  theme) draws it again. */
  private drawing: ChannelDrawing | null = null;
  private readonly surface: PlotlySurface<ChannelDrawing> = new PlotlySurface({
    plot: () => this.plot()?.nativeElement,
    resized: ({width}) => this.capacityChange.emit(Math.max(16, Math.floor((width - 76) / 6))),
    ready: () => {
      if (this.width() <= 0) {
        return null;
      }
      if (!this.waiting()) {
        this.drawing = {
          groups: this.binnedGroups(),
          range: this.range(),
          full: this.full(),
          reduction: this.reduction(),
        };
      }
      return this.drawing;
    },
    draw: (host, drawing) => this.draw(host, drawing),
    bind: (host) => this.bind(host),
    settled: () => this.resolveLocation(),
    failure: () => `${this.dimensionLabel()} chart unavailable. Switch the visualization to retry.`,
  });
  readonly error = this.surface.error;
  private readonly width = computed(() => this.surface.size().width);
  private readonly height = computed(() => Math.max(260, this.surface.size().height));
  private readonly zone = inject(NgZone);
  private timer?: ReturnType<typeof setTimeout>;

  constructor() {
    afterRenderEffect(() => {
      this.rows();
      this.bins();
      this.waiting();
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
      this.surface.theme();
      this.plot();
      void this.surface.paint();
    });
    inject(DestroyRef).onDestroy(() => {
      clearTimeout(this.timer);
      clearTimeout(this.locationTimer);
    });
  }
  private async draw(
    host: PlotHost,
    {groups, range, full, reduction}: ChannelDrawing,
  ): Promise<void> {
    const dual = this.otherRows() !== null;
    const style = rangeChartStyle(host, this.height(), dual);
    const traces = groups.flatMap(({overview, visible}, plane) => {
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
    await this.surface.react(host, traces, {
      width: this.width(),
      height: this.height(),
      ...style.layout,
      dragmode: false,
      uirevision: `${range.start}:${range.end}:${reduction}`,
      xaxis: {
        ...style.xaxis(range, full, this.dimensionLabel()),
        fixedrange: true,
        ticklabelstandoff: RANGE_MIN_COLUMN_WIDTH + 2,
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
    });
    stackRangeOverview(host, this.overviewLabels());
    this.visibleBins = groups[0].visible;
    const layout = host._fullLayout;
    const bottom = layout?.yaxis2 ?? layout?.yaxis;
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
  }
  private bind(host: PlotHost): void {
    host.on('plotly_afterplot', () => stackRangeOverview(host, this.overviewLabels()));
    host.on('plotly_hover', (event) =>
      this.zone.run(() => {
        const point = (
          event['points'] as
            | {customdata?: ChannelBin & {plane: number; side: 'ref' | 'target'}}[]
            | undefined
        )?.[0]?.customdata;
        if (!point) {
          return;
        }
        this.highlightPoints(point);
        this.hoveredCells.set(this.boxes(point, this.otherRows() !== null ? null : point.plane));
        const mouse = event['event'] as MouseEvent | undefined;
        const rect = host.getBoundingClientRect();
        this.tooltip.set({
          left: (mouse?.clientX ?? rect.left + 60) - rect.left,
          top: Math.max(8, (mouse?.clientY ?? rect.top + 30) - rect.top),
          content: this.hoverContent(point),
        });
      }),
    );
    host.on('plotly_unhover', () => this.zone.run(() => this.clearColumnHover()));
    host.on('plotly_relayout', (event) => {
      if (this.surface.painting || this.surface.destroyed) {
        return;
      }
      const next = plotlyPositionRange(event, this.full());
      if (!next || (next.start === this.range().start && next.end === this.range().end)) {
        return;
      }
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
        if (!bin) {
          return;
        }
        this.activePlane = bin.plane;
        this.commitSelection(bin, bin.plane, bin.side);
      }),
    );
  }
}
