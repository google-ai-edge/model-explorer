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
import {tensorHover} from './tensor_hover';
import {ChartHover, type ChartHoverPosition} from './chart_hover';
import {
  Component,
  computed,
  ChangeDetectionStrategy,
  DestroyRef,
  ElementRef,
  NgZone,
  afterRenderEffect,
  inject,
  input,
  linkedSignal,
  output,
  signal,
  viewChild,
} from '@angular/core';
import type {KvHeadChannel} from '../../server/contracts/kv';
import {plotlyColor, type PlotHost} from '../plotly_loader';
import {PlotlySurface} from './plotly_surface';
import {rangeChartStyle} from './range_chart_style';
import {ScatterOverview} from './scatter_overview';
import {
  idList,
  idRuns,
  pairedChannels,
  type PairedChannel,
  isPairedChannel,
  scatterExtent,
  scatterRelayout,
  scatterRepresentatives,
  type ScatterRepresentation,
  withSelected,
  SCATTER_POINT_LIMIT,
  type ScatterView,
  type ValueRange,
} from './scatter_model';
/** What a paint draws of the chart's data: the view or the paired rows, over its range. */
interface ScatterData {
  view: ScatterView | null;
  range: ValueRange;
  paired: readonly (readonly PairedChannel[])[];
}

@Component({
  selector: 'scatter-chart',
  imports: [ScatterOverview, ChartHover],
  // Keys are read on the host, as range-chart and channel-chart read them: the plot element is
  // Plotly's, and a key listener on it never ran in the app.
  host: {'(keydown)': 'navigate($event)'},
  hostDirectives: [{directive: ChartBackground, outputs: ['backgroundClick']}],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div
      #plot
      class="scatter-chart"
      [class.point-hovered]="hover() !== null"
      role="group"
      tabindex="0"
      [attr.aria-label]="
        'Reference versus Target ' +
        dimensionLabel() +
        ' values. Hover or click to inspect; drag to zoom values.'
      "
    ></div>
    @for (point of selectedPoints(); track point.plane) {
      <div
        class="cell-selected scatter-selected"
        aria-hidden="true"
        [attr.data-channel]="selected()"
        [style.left.px]="point.left"
        [style.top.px]="point.top"
      ></div>
    }
    @for (point of hoveredPoints(); track point.plane) {
      <div
        class="cell-hover scatter-hover"
        aria-hidden="true"
        [attr.data-channel]="hover()"
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
    @for (overview of overviews(); track overview.plane) {
      <scatter-overview
        [rows]="view()?.overview ?? (overview.plane ? otherRows() : rows())"
        [label]="groupLabels()[overview.plane]"
        [comparisonAllowed]="comparisonAllowed()"
        [full]="fullRange()"
        [range]="range()"
        (zoom)="zoom.emit($event)"
        [style.left.px]="overview.left"
        [style.top.px]="overview.top"
        [style.width.px]="overview.size"
        [style.height.px]="overview.size"
      />
    }
    @if (error()) {
      <p role="alert">{{ error() }}</p>
    }
    @if (showGroupingNotice() && representation().grouped) {
      <p class="plot-caption">Points grouped by value; zoom in to inspect individual elements.</p>
    }`,
  styles: `
    @use './range_chart_style' as chart;
    @include chart.overlays;
    @include chart.selection;
    .scatter-selected,
    .scatter-hover {
      border-radius: 50%;
      transform: translate(-50%, -50%);
      z-index: 3;
    }
    .scatter-selected {
      width: 12px;
      height: 12px;
    }
    .scatter-hover {
      width: 18px;
      height: 18px;
      z-index: 2;
    }
    :host {
      display: block;
      height: 100%;
      min-width: 0;
      position: relative;
    }
    .scatter-chart {
      width: 100%;
      height: 100%;
      min-height: 260px;
    }
    :host ::ng-deep .scatter-chart .nsewdrag {
      cursor: crosshair !important;
    }
    :host ::ng-deep .scatter-chart.point-hovered .nsewdrag:not(:active) {
      cursor: pointer !important;
    }
    scatter-overview {
      position: absolute;
      z-index: 4;
    }
    p {
      font: 12px var(--me-font-family);
      color: var(--me-on-surface-color);
    }
    .plot-caption {
      margin: 8px 12px;
      color: var(--me-on-surface-variant-color);
      font: 12px/20px var(--me-font-family);
      text-align: center;
    }
  `,
})
export class ScatterChart {
  private readonly background = inject(ChartBackground);
  readonly rows = input<readonly KvHeadChannel[]>([]);
  readonly otherRows = input<readonly KvHeadChannel[]>([]);
  /** Points another chose, drawn instead of `rows`' own, on a single plane; `range` must be the
   *  one `view.visible` was chosen for. */
  readonly view = input<ScatterView | null>(null);
  /** While true the chart keeps what it last drew: the points of its new view are on their way. */
  readonly waiting = input(false);
  readonly messages = input<readonly string[]>([]);
  readonly dimensionLabel = input('Head dimension');
  readonly showGroupingNotice = input(false);
  readonly groupLabels = input<readonly string[]>(['Key cache', 'Value cache']);
  readonly singlePlane = input(false);
  readonly comparisonAllowed = input(true);
  readonly range = input.required<ValueRange>();
  readonly selected = input<number | null>(null);
  readonly select = output<number>();
  readonly zoom = output<ValueRange>();
  readonly tooltip = signal<ChartHoverPosition | null>(null);
  readonly located = signal<{left: number; top: number; width: number; height: number} | null>(
    null,
  );
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  /** The data the chart last drew: kept while it is waiting, when a paint (a new selection, size or
   *  theme) draws it again. */
  private drawing: ScatterData | null = null;
  private readonly surface: PlotlySurface<ScatterData & {size: {width: number; height: number}}> =
    new PlotlySurface({
      plot: () => this.plot()?.nativeElement,
      ready: () => {
        const size = this.surface.size();
        if (size.width <= 0) {
          return null;
        }
        if (!this.waiting()) {
          this.drawing = {view: this.view(), range: this.range(), paired: this.pairedGroups()};
        }
        return this.drawing && {...this.drawing, size};
      },
      draw: (host, frame) => this.draw(host, frame),
      bind: (host) => this.bind(host),
      settled: () => this.resolveLocation(),
      failure: () => 'Scatter chart unavailable. Switch the visualization to retry.',
    });
  readonly error = this.surface.error;
  private readonly zone = inject(NgZone);
  readonly hover = signal<number | null>(null);
  private readonly geometryRevision = signal(0);
  /** Where the selected element is drawn. One whose values the view does not hold yet (they are
   *  being looked up) keeps the previous highlight until they arrive. */
  readonly selectedPoints = linkedSignal<
    {pending: boolean; points: {plane: number; left: number; top: number}[]},
    {plane: number; left: number; top: number}[]
  >({
    source: () => {
      const selected = this.selected();
      const view = this.view();
      return {
        pending: selected !== null && !!view && !view.elements.has(selected),
        points: this.pointLocations(selected),
      };
    },
    computation: (now, previous) => (now.pending && previous ? previous.value : now.points),
  });
  readonly hoveredPoints = computed(() => this.pointLocations(this.hover()));
  private readonly groups = computed(() => [
    this.rows(),
    ...(this.singlePlane() ? [] : [this.otherRows()]),
  ]);
  private readonly pairedGroups = computed(() => this.groups().map(pairedChannels));
  /** How many elements of each plane are paired. */
  private pairedCounts({view, paired}: ScatterData): number[] {
    return view ? [view.visible.total] : paired.map((rows) => rows.length);
  }
  private readonly rowMaps = computed(() => {
    const view = this.view();
    return view
      ? [view.elements]
      : this.groups().map((rows) => new Map(rows.map((row) => [row.channel, row])));
  });
  private readonly channelIds = computed(() => {
    const view = this.view();
    return view
      ? idRuns(view.paired)
      : idList(
          [...new Set(this.pairedGroups().flatMap((rows) => rows.map((row) => row.channel)))].sort(
            (a, b) => a - b,
          ),
        );
  });
  /** The points drawn of each plane, with the selected element: of the view, whose elements
   *  are `elements`, or of the paired rows. */
  private rendered(
    {view, range, paired}: ScatterData,
    elements = view?.elements,
  ): ScatterRepresentation[] {
    if (view) {
      const selected = this.selected();
      const row = selected === null ? undefined : elements?.get(selected);
      return [withSelected(view.visible, row && isPairedChannel(row) ? row : undefined, range)];
    }
    return paired.map((rows) =>
      scatterRepresentatives(
        rows,
        range,
        rows.length > SCATTER_POINT_LIMIT ? this.selected() : null,
        SCATTER_POINT_LIMIT,
        this.comparisonAllowed(),
      ),
    );
  }
  private readonly renderedGroups = computed(() =>
    this.rendered({view: this.view(), range: this.range(), paired: this.pairedGroups()}),
  );
  readonly representation = computed(() =>
    this.renderedGroups().reduce(
      (result, group) => ({
        total: result.total + group.total,
        visible: result.visible + group.visible,
        rendered: result.rendered + group.rendered,
        grouped: result.grouped || group.grouped,
      }),
      {total: 0, visible: 0, rendered: 0, grouped: false},
    ),
  );
  readonly fullRange = computed(() => this.view()?.full ?? scatterExtent(this.groups()));
  readonly overviews = computed(() => {
    this.geometryRevision();
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!layout) {
      return [];
    }
    return [layout.yaxis, ...(this.singlePlane() ? [] : [layout.yaxis2])].flatMap((axis, plane) => {
      if (!axis) {
        return [];
      }
      const size = Math.max(48, Math.min(120, axis._length * 0.42, layout.xaxis._length * 0.25));
      return [
        {
          plane,
          size,
          left: layout.xaxis._offset + layout.xaxis._length - size - 8,
          top: axis._offset + axis._length - size - 8,
        },
      ];
    });
  });
  private pointLocations(
    channel: number | null,
  ): {plane: number; left: number; top: number}[] {
    this.geometryRevision();
    const range = this.range();
    const layout = this.plot()?.nativeElement._fullLayout;
    if (channel === null || !layout) {
      return [];
    }
    return this.rowMaps().flatMap((rows, plane) => {
      const row = rows.get(channel);
      const axis = plane ? layout.yaxis2 : layout.yaxis;
      if (
        !row ||
        !isPairedChannel(row) ||
        !axis ||
        row.ref < range.start ||
        row.ref > range.end ||
        row.target < range.start ||
        row.target > range.end
      ) {
        return [];
      }
      return [
        {
          plane,
          left: layout.xaxis._offset + layout.xaxis.d2p(row.ref),
          top: axis._offset + axis.d2p(row.target),
        },
      ];
    });
  }
  private locationTimer?: ReturnType<typeof setTimeout>;
  private pendingLocation: string | null = null;
  constructor() {
    afterRenderEffect(() => {
      this.rows();
      this.view();
      this.waiting();
      this.otherRows();
      this.messages();
      this.dimensionLabel();
      this.groupLabels();
      this.singlePlane();
      this.comparisonAllowed();
      this.renderedGroups();
      this.range();
      this.surface.size();
      this.surface.theme();
      void this.surface.paint();
    });
    inject(DestroyRef).onDestroy(() => clearTimeout(this.locationTimer));
  }
  navigate(event: KeyboardEvent): void {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      return;
    }
    const ids = this.channelIds();
    if (!ids.length) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const i = ids.indexOf(this.selected() ?? -1);
    const index =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? ids.length - 1
          : Math.max(0, Math.min(ids.length - 1, i + (event.key === 'ArrowLeft' ? -1 : 1)));
    this.select.emit(ids.at(index));
  }
  locateKind(kind: string): void {
    this.pendingLocation = kind;
    this.resolveLocation();
  }
  private resolveLocation(): void {
    const host = this.plot()?.nativeElement;
    const layout = host?._fullLayout;
    if (!this.pendingLocation || !host || !layout) {
      return;
    }
    const axis = this.pendingLocation === 'value' ? layout.yaxis2 : layout.yaxis;
    if (!axis) {
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
  private async draw(
    host: PlotHost,
    frame: ScatterData & {size: {width: number; height: number}},
  ): Promise<void> {
    const {width} = frame.size;
    const height = Math.max(260, frame.size.height);
    const range = frame.range;
    const dual = !this.singlePlane();
    const style = rangeChartStyle(host, height, dual);
    const color = plotlyColor(host, 'var(--me-primary-color)');
    const fraction = dual ? (1 - 64 / Math.max(1, height - 80)) / 2 : 1;
    // The selected element's values may have arrived since the view did.
    const traces = this.rendered(frame, this.view()?.elements).flatMap((group, plane) => {
      const paired = group.rows;
      const axes = {xaxis: plane ? 'x2' : 'x', yaxis: plane ? 'y2' : 'y'};
      return [
        {
          ...axes,
          type: 'scatter',
          mode: 'markers',
          x: paired.map((r) => r.ref),
          y: paired.map((r) => r.target),
          customdata: paired.map((r) => r.channel),
          marker: {size: paired.length > 1000 ? 4 : 6, color, opacity: 0.65},
          hoverinfo: 'none',
        },
      ];
    });
    const axis = {
      range: [range.start, range.end],
      autorange: false,
      fixedrange: false,
      tickformat: '.4~g',
      gridcolor: style.rule,
      zeroline: false,
    };
    await this.surface.react(host, traces, {
      ...style.layout,
      width,
      height,
      margin: {l: 60, r: 16, t: 30, b: 50, autoexpand: false},
      dragmode: 'zoom',
      uirevision: `${range.start}:${range.end}`,
      xaxis: {...axis, anchor: 'y', title: {text: 'Reference', standoff: 6}},
      yaxis: {
        ...axis,
        anchor: 'x',
        title: {text: 'Target', standoff: 6},
        domain: [1 - fraction, 1],
      },
      ...(dual
        ? {
            xaxis2: {
              ...axis,
              anchor: 'y2',
              matches: 'x',
              title: {text: 'Reference', standoff: 6},
            },
            yaxis2: {
              ...axis,
              anchor: 'x2',
              title: {text: 'Target', standoff: 6},
              matches: 'y',
              domain: [0, fraction],
            },
          }
        : {}),
      shapes: this.groups().map((_, plane) => ({
        type: 'line',
        xref: plane ? 'x2' : 'x',
        yref: plane ? 'y2' : 'y',
        x0: range.start,
        x1: range.end,
        y0: range.start,
        y1: range.end,
        line: {color: style.muted, width: 1, dash: 'dot'},
        layer: 'below',
      })),
      annotations: this.groups().flatMap((_, plane) => [
        {
          text: this.groupLabels()[plane] ?? '',
          x: 0,
          y: plane ? fraction : 1,
          xref: 'paper',
          yref: 'paper',
          xanchor: 'left',
          yanchor: 'bottom',
          yshift: 6,
          showarrow: false,
        },
        {
          text: 'Ref = Target',
          x: 1,
          y: plane ? fraction : 1,
          xref: 'paper',
          yref: 'paper',
          xanchor: 'right',
          yanchor: 'bottom',
          yshift: 6,
          showarrow: false,
        },
        ...(this.messages()[plane] || !this.pairedCounts(frame)[plane]
          ? [
              {
                text:
                  this.messages()[plane] ||
                  `No paired ${this.dimensionLabel().toLowerCase()} values`,
                x: 0.5,
                y: plane ? fraction / 2 : 1 - fraction / 2,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
              },
            ]
          : []),
      ]),
    });
    this.geometryRevision.update((n) => n + 1);
  }
  private bind(host: PlotHost): void {
    host.on('plotly_relayout', (event) => {
      if (this.surface.painting || this.surface.destroyed) {
        return;
      }
      const next = scatterRelayout(event);
      if (next && (next.start !== this.range().start || next.end !== this.range().end)) {
        this.zone.run(() => this.zoom.emit(next));
      }
    });
    host.on('plotly_click', (event) => {
      const channel = (event['points'] as {customdata?: number}[])?.[0]?.customdata;
      if (typeof channel === 'number') {
        this.background.handled();
        this.zone.run(() => this.select.emit(channel));
      }
    });
    host.on('plotly_hover', (event) =>
      this.zone.run(() => {
        const channel = (event['points'] as {customdata?: number}[])?.[0]?.customdata;
        if (typeof channel !== 'number') {
          return;
        }
        this.hover.set(channel);
        const mouse = event['event'] as MouseEvent | undefined;
        const rect = host.getBoundingClientRect();
        this.tooltip.set({
          left: (mouse?.clientX ?? rect.left + 60) - rect.left,
          top: Math.max(8, (mouse?.clientY ?? rect.top + 30) - rect.top),
          content: tensorHover(
            this.rowMaps().map((rows) => {
              const row = rows.get(channel);
              return row ? [row] : [];
            }),
            {start: channel, end: channel + 1},
            'mean',
            {
              dimensionLabel: this.dimensionLabel(),
              groupLabels: this.groupLabels(),
              comparisonAllowed: this.comparisonAllowed(),
            },
          ),
        });
      }),
    );
    host.on('plotly_unhover', () =>
      this.zone.run(() => {
        this.hover.set(null);
        this.tooltip.set(null);
      }),
    );
  }
}
