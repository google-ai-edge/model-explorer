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

import {animateAnalysisLocationFeedback} from '../locate_feedback';
import {ChartHover, type ChartHoverPosition, hoverBin, hoverInterval} from './chart_hover';
import {RangeAxisSelection} from './range_axis_selection';
import {stackRangeOverview} from './range_chart_overview';
import {
  flatLineRange,
  rangeChartStyle,
  rangeLineStyle,
  runtimeLineStyle,
} from './range_chart_style';
import type {Runtime} from './runtime_series';
import {lineChartPoints} from './line_chart_points';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  NgZone,
  afterRenderEffect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import type {StagePositions} from '../../server/contracts/capture';
import type {KvMetric, KvRangeBin, KvRangeResponse, KvRangeRow} from '../../server/contracts/kv';
import type {MetricDescription} from '../../server/contracts/metrics';
import {reasonText} from '../../server/reasons';
import {metricHeat, type MetricScaleMode} from '../metric_scale/metric_scale';
import {plotlyColor, PlotHost} from '../plotly_loader';
import {PlotlySurface} from './plotly_surface';
import {heatmapAxes, heatmapTraces, positionAxisLabels} from './heatmap_plot';
import {heatmapMissingRanges, type HeatmapData, type HeatmapReduction} from './heatmap_model';
import {rangeHeatmapData} from './range_heatmap_data';
import {RangeSelection} from './range_selection';
import {stepRows, type ColumnSteps} from '../format/step_labels';
import {RANGE_MIN_COLUMN_WIDTH, rangeCapacity, rangeKeyboardSelection} from './range_geometry';
import {
  plotlyCellIndex,
  plotlyColumns,
  plotlyPositionRange,
  plotlyColumnSelection,
  heatmapHighlightBoxes,
  type HeatmapHighlightBox,
} from './plotly_range';

/** What the overview draws of a range answer: its range, bins and rows. */
export type RangeOverview = Pick<KvRangeResponse, 'start' | 'end' | 'bins' | 'rows'>;

/** The size and ranges a paint reads before Plotly loads; inputs may change meanwhile. */
interface RangeFrame {
  size: {width: number; height: number};
  range: KvRangeBin;
  full: KvRangeBin;
}

@Component({
  selector: 'range-chart',
  // Keep keyboard handling outside the Plotly-owned element across purge/react.
  host: {
    '(keydown)': 'keyDown($event)',
    '(pointerdown)': 'startPointer()',
    '(click)': 'clickBackground($event)',
  },
  standalone: true,
  imports: [RangeAxisSelection, ChartHover],
  templateUrl: './range_chart.ng.html',
  styleUrl: './range_chart.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RangeChart {
  readonly positionAlignment = input<StagePositions | null>(null);
  /** Which step a column is on each side, in the server's words; the tooltip of one column says
   *  it. */
  readonly columnSteps = input<((column: number) => ColumnSteps | null) | null>(null);
  readonly columnStripHeight = RANGE_MIN_COLUMN_WIDTH;
  readonly metricLabel = input<string | null>(null);
  /** The metric as the server's catalog describes it: its label, unit and color scale. */
  readonly description = input<MetricDescription | undefined>(undefined);
  readonly showLayers = input(true);
  /** Names what the chart shows, as in "KV Heatmap with position range slider". */
  readonly label = input.required<string>();
  readonly view = input<'heatmap' | 'lineChart'>('heatmap');
  readonly layers = input<readonly number[]>([]);
  readonly bins = input<readonly KvRangeBin[]>([]);
  readonly visibleRange = input<KvRangeBin | null>(null);
  readonly fullRange = input<KvRangeBin | null>(null);
  /** The whole range's bins and rows, which the overview draws. */
  readonly overview = input<RangeOverview | null>(null);
  readonly otherOverview = input<RangeOverview | null>(null);
  readonly kind = input('key');
  readonly head = input('min');
  readonly positionReduction = input<HeatmapReduction>('min');
  readonly rows = input<readonly KvRangeRow[]>([]);
  readonly otherRows = input<readonly KvRangeRow[] | null>(null);
  readonly selectedKind = input<string | null>(null);
  readonly metric = input<KvMetric>('relative_l2');
  readonly selected = input<RangeSelection | null>(null);
  readonly highlightedLayer = input<number | null>(null);
  readonly lineChartLayer = input(0);
  readonly series = input<
    | readonly {
        layer: number;
        kind: string;
        rows: readonly KvRangeRow[];
        color: string;
        name?: string;
        runtime?: Runtime;
      }[]
    | null
  >(null);
  readonly disabled = input(false);
  readonly scaleMaximum = input(1);
  readonly scaleMode = input<MetricScaleMode>('auto');
  readonly format = input<(value: number | null) => string>((v) =>
    v === null ? 'Unavailable' : String(v),
  );
  readonly select = output<RangeSelection>();
  readonly backgroundClick = output<MouseEvent>();
  readonly seriesSelect = output<string>();
  readonly zoom = output<KvRangeBin>();
  readonly capacity = output<number>();
  readonly positionHover = output<RangeSelection | null>();
  readonly locatedRegion = signal<
    | (HeatmapHighlightBox & {
        kind: string;
        position?: number;
        layer?: number;
      })
    | null
  >(null);
  private readonly locationTarget = viewChild<ElementRef<HTMLElement>>('locationTarget');
  private pendingLocation: {kind: string; position?: number; layer?: number} | null = null;
  private locationTimer: ReturnType<typeof setTimeout> | undefined;
  locationName(kind: string, layer?: number): string {
    if (kind === 'key') {
      return 'Key cache';
    }
    if (kind === 'value') {
      return 'Value cache';
    }
    if (!this.showLayers()) {
      return this.series()?.find((s) => s.layer === layer)?.name ?? this.label();
    }
    return kind;
  }
  locateKind(kind: string, position?: number, layer?: number): void {
    this.pendingLocation = {kind, position, layer};
    this.resolveLocation();
  }
  private resolveLocation(): void {
    const location = this.pendingLocation;
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!location || !layout || this.renderedView !== this.view() || this.surface.painting) {
      return;
    }
    const {kind, position, layer} = location;
    const axis = kind === this.kind() ? layout.yaxis : layout.yaxis2;
    if (!axis) {
      return;
    }
    if (position !== undefined && layer !== undefined) {
      const range = this.visibleRange();
      if (!range) {
        return;
      }
      const column = plotlyColumns(this.bins(), this.overview()?.bins ?? [], range).find(
        (bin) => bin.start <= position && position < bin.end,
      );
      if (!column) {
        return;
      }
      let box: HeatmapHighlightBox | undefined;
      if (this.view() === 'heatmap') {
        box = heatmapHighlightBoxes(layout.xaxis, [axis], this.layers(), {...column, layer})[0];
      } else {
        const series = this.series()?.find((s) => s.kind === kind && s.layer === layer);
        const rows = column.overview
          ? (kind === this.kind() ? this.overview() : this.otherOverview())?.rows
          : (series?.rows ?? this.rows());
        const summary = rows
          ?.find((r) => r.layer === layer)
          ?.bins.find((b) => b.start === column.source.start && b.end === column.source.end)
          ?.metrics[this.metric()];
        const value = summary?.[this.positionReduction()];
        if (value == null || !Number.isFinite(value)) {
          return;
        }
        box = {
          left: layout.xaxis._offset + layout.xaxis.d2p((column.start + column.end - 1) / 2) - 6,
          top: axis._offset + axis.d2p(value) - 6,
          width: 12,
          height: 12,
        };
      }
      if (!box) {
        return;
      }
      this.pendingLocation = null;
      this.locatedRegion.set({...box, kind, position, layer});
      this.announcement.set(
        `Located ${this.locationName(kind, layer)} · ${this.showLayers() ? `Layer ${layer}` : 'Final'} · Position ${position}`,
      );
      return;
    }
    this.pendingLocation = null;
    this.locatedRegion.set({
      kind,
      left: layout.xaxis._offset - 4,
      top: Math.max(0, axis._offset - 22),
      width: layout.xaxis._length + 8,
      height: axis._length + 26,
    });
    this.announcement.set(
      `Located ${this.locationName(kind)} ${this.view() === 'heatmap' ? 'Heatmap' : 'Line chart'}`,
    );
  }
  private readonly surface: PlotlySurface<RangeFrame> = new PlotlySurface({
    plot: () => this.plot()?.nativeElement,
    measured: () => this.host.nativeElement,
    resized: (box) => this.resize(box),
    ready: () => {
      const size = this.size();
      const range = this.visibleRange();
      const full = this.fullRange();
      return range && full && size.width > 0 ? {size, range, full} : null;
    },
    draw: (host, frame) => this.draw(host, frame),
    bind: (host) => this.bind(host),
    settled: () => this.resolveLocation(),
    failure: () => 'Chart could not render.',
  });
  readonly error = this.surface.error;
  readonly announcement = signal('');
  readonly missingCells = signal<HeatmapHighlightBox[]>([]);
  readonly hoveredCells = signal<HeatmapHighlightBox[]>([]);
  readonly hoveredBands = signal<HeatmapHighlightBox[]>([]);
  readonly hoveredPoints = signal<{left: number; top: number}[]>([]);
  readonly selectedCells = signal<HeatmapHighlightBox[]>([]);
  readonly selectedAxisRange = signal<{left: number; width: number} | null>(null);
  readonly tooltip = signal<ChartHoverPosition | null>(null);
  private pointer = {x: 0, y: 0};
  trackPointer(event: MouseEvent): void {
    const rect = this.host.nativeElement.getBoundingClientRect();
    this.pointer = {x: event.clientX - rect.left, y: event.clientY - rect.top};
    if (this.hoverSelection) {
      this.showTooltip(this.hoverSelection);
    }
  }
  private showTooltip(selection: RangeSelection & {kind?: string}): void {
    const alignment = this.positionAlignment();
    const properties = [
      {
        label: alignment ? 'Aligned position' : 'Position',
        value: hoverInterval(selection.start, selection.end),
      },
    ];
    if (alignment && selection.end - selection.start === 1) {
      const pair = alignment.pairs[selection.start - alignment.start];
      properties.push(
        {label: 'Reference', value: String(pair?.[0] ?? '-')},
        {label: 'Target', value: String(pair?.[1] ?? '-')},
      );
    }
    if (selection.end - selection.start === 1) {
      properties.push(...stepRows(this.columnSteps()?.(selection.start)));
    }
    properties.push(
      {
        label: selection.layer === null ? 'Layers' : 'Layer',
        value:
          selection.layer === null
            ? 'All displayed'
            : this.showLayers()
              ? String(selection.layer)
              : 'Final',
      },
      {label: 'Bin', value: hoverBin(selection.start, selection.end, this.positionReduction())},
    );
    const range = this.visibleRange();
    const column =
      range &&
      plotlyColumns(this.bins(), this.overview()?.bins ?? [], range).find(
        (c) => c.source.start === selection.start && c.source.end === selection.end,
      );
    const kinds = [
      this.kind(),
      ...(this.otherRows() !== null ? [this.kind() === 'key' ? 'value' : 'key'] : []),
    ];
    const summaries = kinds.map((kind) => {
      const overview = kind === this.kind() ? this.overview() : this.otherOverview();
      const rows = column?.overview
        ? overview?.rows
        : (this.series()?.find((s) => s.kind === kind && s.layer === selection.layer)?.rows ??
          (kind === this.kind() ? this.rows() : this.otherRows()));
      return rows
        ?.find((r) => r.layer === selection.layer)
        ?.bins.find((b) => b.start === selection.start && b.end === selection.end)?.metrics[
        this.metric()
      ];
    });
    const values = summaries.map((summary) =>
      summary?.valid_count ? (summary[this.positionReduction()] ?? null) : null,
    );
    // Why the bin's positions without a value have none, in the server's words.
    const why = reasonText(summaries.flatMap((summary) => summary?.reasons ?? []));
    if (why) {
      properties.push({
        label: values.some((value) => value !== null) ? 'Partly unavailable' : 'Unavailable',
        value: why,
      });
    }
    const layout = this.plot()?.nativeElement._fullLayout;
    this.hoveredBands.set(
      selection.layer !== null ? this.highlightBoxes({...selection, layer: null}) : [],
    );
    this.hoveredPoints.set(
      this.view() === 'lineChart' && selection.layer !== null && layout
        ? values.flatMap((value, index) => {
            const axis = index ? layout.yaxis2 : layout.yaxis;
            if (value == null || !Number.isFinite(value) || !axis) {
              return [];
            }
            const x = layout.xaxis.d2p(
              ((column?.start ?? selection.start) + (column?.end ?? selection.end) - 1) / 2,
            );
            const y = axis.d2p(value);
            return x >= 0 && x <= layout.xaxis._length && y >= 0 && y <= axis._length
              ? [{left: layout.xaxis._offset + x, top: axis._offset + y}]
              : [];
          })
        : [],
    );
    this.tooltip.set({
      left: this.pointer.x,
      top: this.pointer.y,
      content: {
        properties,
        ...(selection.layer === null
          ? {}
          : {
              title: this.metricLabel() ?? this.description()?.label ?? this.metric(),
              columns: kinds.map((kind) => this.locationName(kind, selection.layer ?? undefined)),
              activeColumn: kinds.indexOf(selection.kind ?? ''),
              rows: [{label: 'Value', values: values.map((value) => this.format()(value))}],
            }),
      },
    });
  }
  private hoverSelection: RangeSelection | null = null;
  protected pointClicked = false;
  protected startPointer(): void {
    this.pointClicked = false;
  }
  protected clickBackground(event: MouseEvent): void {
    if (
      event.button !== 0 ||
      (event.target as Element).closest(
        'button, a, input, select, textarea, [role="button"], .rangeslider-container, .modebar, .legend',
      )
    ) {
      return;
    }
    // Let Plotly's document-level click handlers finish before deciding whether
    // this was background. Stopping propagation here would swallow data clicks.
    setTimeout(() => {
      if (this.surface.destroyed || this.pointClicked) {
        return;
      }
      const plot = this.plot()?.nativeElement;
      const layout = plot?._fullLayout;
      if (this.view() === 'heatmap' && plot && layout) {
        const rect = plot.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const axis = layout.xaxis;
        const inColumn =
          x >= axis._offset &&
          x < axis._offset + axis._length &&
          plotlyColumnSelection(this.bins(), axis.p2d(x - axis._offset));
        // Missing cells still belong to the data grid, not the background.
        if (
          inColumn &&
          [layout.yaxis, layout.yaxis2].some(
            (yAxis) => yAxis && y >= yAxis._offset && y < yAxis._offset + yAxis._length,
          )
        ) {
          return;
        }
      }
      this.clearCellHover();
      this.clearColumnHover();
      this.backgroundClick.emit(event);
    });
  }
  private highlightBoxes(selection: RangeSelection | null): HeatmapHighlightBox[] {
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!selection || !layout) {
      return [];
    }
    return heatmapHighlightBoxes(
      layout.xaxis,
      layout.yaxis2 ? [layout.yaxis, layout.yaxis2] : [layout.yaxis],
      this.layers(),
      this.view() === 'heatmap' ? selection : {...selection, layer: null},
    );
  }
  clearCellHover(): void {
    this.tooltip.set(null);
    this.hoverSelection = null;
    this.hoveredCells.set([]);
    this.hoveredBands.set([]);
    this.hoveredPoints.set([]);
    this.positionHover.emit(null);
    this.emphasizeLayer(this.highlightedLayer());
  }

  readonly axisCells = signal<{left: number; width: number}[]>([]);
  readonly columnAxis = signal<{left: number; top: number; width: number} | null>(null);
  readonly hoveredColumn = signal<{
    left: number;
    width: number;
    boxes: {left: number; top: number; width: number; height: number}[];
    label: string;
  } | null>(null);
  private columnPointer: number | null = null;
  private columnHit(clientX: number): KvRangeBin | null {
    const host = this.plot()?.nativeElement;
    const axis = host?._fullLayout?.xaxis;
    if (!host || !axis || this.disabled()) {
      return null;
    }
    return plotlyColumnSelection(
      this.bins(),
      axis.p2d(clientX - host.getBoundingClientRect().left - axis._offset),
    );
  }
  hoverColumn(event: MouseEvent): void {
    this.clearCellHover();
    this.trackPointer(event);
    this.columnPointer = event.clientX;
    this.updateColumnHover(event.clientX);
  }
  clearColumnHover(): void {
    if (!this.hoverSelection) {
      this.tooltip.set(null);
    }
    this.columnPointer = null;
    this.hoveredColumn.set(null);
    this.positionHover.emit(null);
  }
  private updateColumnHover(clientX: number): void {
    const hit = this.columnHit(clientX);
    const layout = this.plot()?.nativeElement._fullLayout;
    if (!hit || !layout) {
      this.hoveredColumn.set(null);
      return;
    }
    this.positionHover.emit({...hit, layer: null});
    this.showTooltip({...hit, layer: null});
    const x = layout.xaxis;
    this.hoveredColumn.set({
      left: x._offset + x.d2p(hit.start - 0.5),
      width: x.d2p(hit.end - 0.5) - x.d2p(hit.start - 0.5),
      boxes: this.highlightBoxes({...hit, layer: null}),
      label: `Global Positions ${hit.start}–${hit.end - 1} · All layers${this.otherRows() !== null ? ' · K/V' : ''}`,
    });
  }
  selectColumn(event: MouseEvent): void {
    const hit = this.columnHit(event.clientX);
    if (hit) {
      this.commitColumn(hit);
    }
  }
  private commitColumn(selection: RangeSelection): void {
    this.select.emit({...selection, layer: null});
    this.announcement.set(
      `${this.positionAlignment() ? 'Aligned positions' : 'Global Positions'} ${selection.start}–${selection.end - 1}, all layers selected.`,
    );
  }
  columnKeyDown(event: KeyboardEvent): void {
    if (this.disabled()) {
      return;
    }
    const current = this.selected();
    const selection = rangeKeyboardSelection(
      this.bins(),
      this.layers(),
      current ? {...current, layer: null} : null,
      event.key === ' ' ? 'Enter' : event.key,
    );
    if (selection) {
      event.preventDefault();
      event.stopPropagation();
      this.commitColumn(selection);
    }
  }

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly zone = inject(NgZone);
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  private readonly size = signal({width: 0, height: 0});
  private renderedView: 'heatmap' | 'lineChart' | null = null;
  private frame = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private activeSeries = 0;

  constructor() {
    afterRenderEffect(() => {
      const region = this.locatedRegion();
      const target = this.locationTarget();
      if (region && target) {
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        target.nativeElement.scrollIntoView({
          block: 'center',
          inline: 'nearest',
          behavior: reducedMotion ? 'instant' : 'smooth',
        });
        // Locate and outline the region without moving chart content.
        // Animate only the locator so plotted values and inspection stay fixed.
        for (const animation of target.nativeElement.getAnimations()) {
          animation.cancel();
        }
        animateAnalysisLocationFeedback(target.nativeElement, reducedMotion);
        clearTimeout(this.locationTimer);
        this.locationTimer = setTimeout(() => this.locatedRegion.set(null), 3500);
      }
    });
    afterRenderEffect(() => {
      this.plot();
      this.size();
      this.surface.theme();
      this.view();
      this.layers();
      this.bins();
      this.rows();
      this.otherRows();
      this.selectedKind();
      this.visibleRange();
      this.fullRange();
      this.overview();
      this.otherOverview();
      this.kind();
      this.head();
      this.positionReduction();
      this.metric();
      this.metricLabel();
      this.description();
      this.selected();
      this.series();
      this.lineChartLayer();
      this.disabled();
      this.scaleMaximum();
      this.scaleMode();
      this.schedule();
    });
    afterRenderEffect(() => {
      this.emphasizeLayer(this.highlightedLayer());
    });
    inject(DestroyRef).onDestroy(() => {
      cancelAnimationFrame(this.frame);
      clearTimeout(this.timer);
      clearTimeout(this.locationTimer);
    });
  }
  /** Keeps whole-pixel sizes and reports a new column capacity when the width allows one. */
  private resize(box: DOMRectReadOnly): void {
    const width = Math.floor(box.width);
    const height = Math.floor(box.height);
    if (width <= 0 || height <= 0) {
      return;
    }
    const previous = this.size();
    if (previous.width === width && previous.height === height) {
      return;
    }
    this.size.set({width, height});
    if (rangeCapacity(previous.width) !== rangeCapacity(width)) {
      this.capacity.emit(rangeCapacity(width));
    }
  }
  /** Paints on the next frame; a paint asked for during one runs after it. */
  private schedule(): void {
    if (this.surface.destroyed) {
      return;
    }
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => void this.surface.paint());
  }
  private resolved(value: string): string {
    return plotlyColor(this.host.nativeElement, value);
  }

  private emphasizeLayer(layer: number | null): void {
    const host = this.plot()?.nativeElement;
    const api = this.surface.api;
    if (!host || !api || this.renderedView !== 'lineChart' || this.surface.painting) {
      return;
    }
    const series = this.series() ?? [];
    if (!series.length) {
      return;
    }
    void api
      .restyle(
        host,
        {
          opacity: series.map((s) => (layer === null || s.layer === layer ? 1 : 0.18)),
          'line.width': series.map((s) =>
            s.layer === layer || this.selected()?.layer === s.layer ? 3 : 1.5,
          ),
        },
        series.map((_, index) => index),
      )
      .catch(() => {});
  }
  private bind(host: PlotHost): void {
    host.on('plotly_afterplot', () =>
      stackRangeOverview(
        host,
        this.otherRows() !== null
          ? this.kind() === 'key'
            ? ['Key', 'Value']
            : ['Value', 'Key']
          : [],
      ),
    );
    host.on('plotly_hover', (event) =>
      this.zone.run(() => {
        if (this.disabled()) {
          return;
        }
        const point = (event['points'] as {customdata?: RangeSelection}[] | undefined)?.[0]
          ?.customdata;
        if (!point || !Number.isInteger(point.layer)) {
          return;
        }
        const mouse = event['event'] as MouseEvent | undefined;
        if (mouse) {
          const rect = this.host.nativeElement.getBoundingClientRect();
          this.pointer = {x: mouse.clientX - rect.left, y: mouse.clientY - rect.top};
        }
        this.positionHover.emit(point);
        this.hoverSelection = point;
        this.showTooltip(point);
        if (this.view() === 'lineChart') {
          this.emphasizeLayer(point.layer);
        }
        this.hoveredCells.set(this.view() === 'heatmap' ? this.highlightBoxes(point) : []);
      }),
    );
    host.on('plotly_unhover', () => this.zone.run(() => this.clearCellHover()));
    host.on('plotly_click', (event) =>
      this.zone.run(() => {
        this.pointClicked = true;
        if (this.disabled()) {
          return;
        }
        const points = event['points'] as
          | {customdata?: {start: number; end: number; layer: number; kind: string}}[]
          | undefined;
        const point = points?.[0]?.customdata;
        if (!point || !Number.isInteger(point.layer)) {
          return;
        }
        this.seriesSelect.emit(point.kind);
        this.select.emit({start: point.start, end: point.end, layer: point.layer});
        this.announcement.set(
          `Layer ${point.layer}, Positions ${point.start}–${point.end - 1} selected.`,
        );
      }),
    );
    host.on('plotly_relayout', (event) => {
      if (this.surface.painting || this.surface.destroyed) {
        return;
      }
      const full = this.fullRange();
      const current = this.visibleRange();
      const range = full && plotlyPositionRange(event, full);
      if (!range || (range.start === current?.start && range.end === current.end)) {
        return;
      }
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.zone.run(() => this.zoom.emit(range)), 100);
    });
  }
  private buildHeatmapTraces(
    columns: ReturnType<typeof plotlyColumns>,
    metric: KvMetric,
  ): {data: HeatmapData; traces: unknown[]} {
    const normalized = (v: number | null | undefined): number | null =>
      metricHeat(this.description(), v, this.scaleMode(), this.scaleMaximum());
    const tensors = [
      {kind: this.kind(), rows: this.rows(), overview: this.overview()?.rows ?? []},
    ];
    const otherRows = this.otherRows();
    if (otherRows !== null) {
      tensors.push({
        kind: this.kind() === 'key' ? 'value' : 'key',
        rows: otherRows,
        overview: this.otherOverview()?.rows ?? [],
      });
    }
    const data = rangeHeatmapData(
      this.layers(),
      columns,
      tensors,
      metric,
      this.head(),
      this.format(),
      this.positionReduction(),
    );
    return {
      data,
      traces: heatmapTraces(data, {
        normalize: normalized,
        // The host may choose the palette; without one it is the activation palette.
        low: this.resolved('var(--heat-low, var(--heat-activation-low))'),
        high: this.resolved('var(--heat-high, var(--heat-activation-high))'),
      }),
    };
  }
  private buildLineTraces(
    host: PlotHost,
    columns: ReturnType<typeof plotlyColumns>,
    metric: KvMetric,
    accent: string,
  ): unknown[] {
    const traces: unknown[] = [];
    const datum = (layer: number, bin: KvRangeBin, kind: string) => ({...bin, layer, kind});
    const label = (layer: number, bin: KvRangeBin, value: number | null): string =>
      `Layer ${layer} · Positions ${bin.start}–${bin.end - 1}<br>${this.description()?.label ?? metric}: ${this.format()(value)}${bin.end - bin.start > 1 ? `<br>Position bins: ${this.positionReduction()}` : ''}`;
    const series = this.series() ?? [
      {layer: this.lineChartLayer(), kind: this.kind(), rows: this.rows(), color: accent},
    ];
    for (const s of series) {
      const source = s.kind === this.kind() ? this.overview() : this.otherOverview();
      const contextRows = plotlyCellIndex(source?.rows ?? []);
      const detailRows = plotlyCellIndex(s.rows);
      const cells = columns.map((col) =>
        (col.overview ? contextRows : detailRows)
          .get(s.layer)
          ?.get(`${col.source.start}:${col.source.end}`),
      );
      const points = lineChartPoints(
        columns,
        cells.map((c) => c?.metrics[metric]),
        this.positionReduction(),
      );
      const emphasized =
        this.highlightedLayer() === s.layer || this.selected()?.layer === s.layer;
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: this.otherRows() !== null && s.kind !== this.kind() ? 'y2' : 'y',
        opacity:
          this.highlightedLayer() === null || this.highlightedLayer() === s.layer ? 1 : 0.18,
        name:
          s.name ??
          `L${s.layer} · ${s.kind === 'key' ? 'Key cache' : s.kind === 'value' ? 'Value cache' : s.kind}`,
        x: points.map((p) => p?.position ?? null),
        y: points.map((p) => p?.value ?? null),
        connectgaps: false,
        customdata: points.map((p) => (p ? datum(s.layer, columns[p.bin].source, s.kind) : null)),
        text: points.map((p) => (p ? label(s.layer, columns[p.bin].source, p.value) : '')),
        hovertemplate: '%{text}<extra>%{fullData.name}</extra>',
        ...(s.runtime
          ? runtimeLineStyle(host, s.runtime, emphasized)
          : rangeLineStyle(s.color, emphasized)),
      });
    }
    return traces;
  }
  private buildLayout(
    host: PlotHost,
    {size, range, full}: RangeFrame,
    traces: unknown[],
    heatmapData: HeatmapData | undefined,
  ): Record<string, unknown> {
    const view = this.view();
    const metric = this.metric();
    const surface = this.resolved('var(--chart-canvas)');
    const muted = this.resolved('var(--chart-muted)');
    const rule = this.resolved('var(--chart-line)');
    const dual = this.otherRows() !== null;
    const style = rangeChartStyle(host, size.height, dual);
    const fraction = style.fraction;
    const description = this.description();
    const lineChartAxis = (domain: number[]) =>
      style.yaxis(
        domain,
        this.metricLabel() ??
          (description
            ? `${description.label}${description.unit_label ? ` (${description.unit_label})` : ''}`
            : metric),
      );
    // A line chart of one value everywhere draws it clear of the axis' edge.
    const flat =
      view === 'heatmap'
        ? null
        : flatLineRange(
            traces.flatMap(
              (trace) => ((trace as {y?: (number | null)[]}).y ?? []) as (number | null)[],
            ),
          );
    const flatRange = flat ? {range: flat, autorange: false} : {};
    return {
      width: size.width,
      height: Math.max(260, size.height),
      ...style.layout,
      dragmode: false,
      uirevision: `${view}:${metric}:${range.start}:${range.end}`,
      shapes: [],
      xaxis: {
        ...style.xaxis(range, full, 'Position'),
        fixedrange: true,
        ticklabelstandoff: RANGE_MIN_COLUMN_WIDTH + 2,
        ...positionAxisLabels(this.bins(), range, size.width - 76),
      },
      ...(heatmapData
        ? heatmapAxes(heatmapData, size.height)
        : {
            yaxis: {
              ...lineChartAxis(dual ? [1 - fraction, 1] : [0, 1]),
              ...flatRange,
            },
            ...(dual ? {yaxis2: {...lineChartAxis([0, fraction]), matches: 'y'}} : {}),
            annotations: dual
              ? style.labels(
                  [this.kind(), this.kind() === 'key' ? 'value' : 'key'].map((kind) =>
                    kind === 'key' ? 'Key cache' : 'Value cache',
                  ),
                )
              : [],
          }),
      hoverlabel: {bgcolor: surface, bordercolor: rule, font: {color: muted}},
    };
  }
  private syncPostDrawOverlays(host: PlotHost, heatmapData: HeatmapData | undefined): void {
    const dual = this.otherRows() !== null;
    const heatmap = this.view() === 'heatmap';
    const metric = this.metric();
    stackRangeOverview(
      host,
      dual ? (this.kind() === 'key' ? ['Key', 'Value'] : ['Value', 'Key']) : [],
    );
    if (this.columnPointer !== null) {
      this.updateColumnHover(this.columnPointer);
    } else {
      this.clearColumnHover();
    }
    const layout = host._fullLayout;
    const bottom = layout?.yaxis2 ?? layout?.yaxis;
    this.missingCells.set(
      heatmapData && layout
        ? heatmapMissingRanges(heatmapData).flatMap((cell) => {
            const axis = cell.plane === 0 ? layout.yaxis : layout.yaxis2;
            return axis ? heatmapHighlightBoxes(layout.xaxis, [axis], this.layers(), cell) : [];
          })
        : layout && !heatmap
          ? [layout.yaxis, ...(layout.yaxis2 ? [layout.yaxis2] : [])].flatMap((axis, plane) => {
              const kind = plane ? (this.kind() === 'key' ? 'value' : 'key') : this.kind();
              const series = (this.series() ?? []).filter((s) => s.kind === kind);
              if (!series.length) {
                return [];
              }
              return this.bins()
                .filter((bin) =>
                  series.every((s) => {
                    const summary = s.rows
                      .find((r) => r.layer === s.layer)
                      ?.bins.find((b) => b.start === bin.start && b.end === bin.end)?.metrics[
                      metric
                    ];
                    return (
                      !summary ||
                      summary.valid_count <= 0 ||
                      summary[this.positionReduction()] == null
                    );
                  }),
                )
                .flatMap((bin) =>
                  heatmapHighlightBoxes(layout.xaxis, [axis], [], {...bin, layer: null}),
                );
            })
          : [],
    );
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
        ? this.bins().flatMap((bin) => {
            const left = Math.max(0, layout.xaxis.d2p(bin.start - 0.5));
            const right = Math.min(layout.xaxis._length, layout.xaxis.d2p(bin.end - 0.5));
            return right > left ? [{left, width: right - left}] : [];
          })
        : [],
    );
    const selected = this.selected();
    const boxes = this.highlightBoxes(selected);
    this.selectedCells.set(boxes);
    this.selectedAxisRange.set(
      selected?.layer === null && boxes.length
        ? {left: boxes[0].left, width: boxes[0].width}
        : null,
    );
    this.hoveredCells.set(this.highlightBoxes(this.hoverSelection));
  }
  private async draw(host: PlotHost, frame: RangeFrame): Promise<void> {
    const view = this.view();
    const columns = plotlyColumns(this.bins(), this.overview()?.bins ?? [], frame.range);
    const metric = this.metric();
    const accent = this.resolved('var(--chart-accent)');
    let heatmapData: HeatmapData | undefined;
    let traces: unknown[];
    if (view === 'heatmap') {
      const built = this.buildHeatmapTraces(columns, metric);
      heatmapData = built.data;
      traces = built.traces;
    } else {
      traces = this.buildLineTraces(host, columns, metric, accent);
    }
    if (!traces.length) {
      traces = [
        {
          type: 'scatter',
          x: [frame.full.start, frame.full.end - 1],
          y: [null, null],
          hoverinfo: 'skip',
        },
      ];
    }
    // Plotly retains range-slider subplot DOM across trace-type changes.
    // Rebuild only on view switches so Line chart cannot inherit Heatmap thumbnails.
    if (this.renderedView !== null && this.renderedView !== view) {
      this.surface.purge(host);
    }
    await this.surface.react(
      host,
      traces.map((trace) => ({
        ...(trace as object),
        hovertemplate: undefined,
        hoverinfo: 'none',
      })),
      this.buildLayout(host, frame, traces, heatmapData),
    );
    this.syncPostDrawOverlays(host, heatmapData);
    this.renderedView = view;
  }
  retry(): void {
    this.schedule();
  }
  keyDown(event: KeyboardEvent): void {
    if (event.target !== this.plot()?.nativeElement || this.disabled()) {
      return;
    }
    const series = this.series() ?? [];
    if (
      this.view() === 'lineChart' &&
      (event.key === 'ArrowUp' || event.key === 'ArrowDown') &&
      series.length
    ) {
      this.activeSeries = Math.max(
        0,
        Math.min(series.length - 1, this.activeSeries + (event.key === 'ArrowUp' ? -1 : 1)),
      );
    }
    const current = series[this.activeSeries];
    const selected = rangeKeyboardSelection(
      this.bins(),
      this.layers(),
      this.selected(),
      event.key,
      this.view() === 'lineChart' ? (current?.layer ?? this.lineChartLayer()) : null,
    );
    if (selected) {
      event.preventDefault();
      event.stopPropagation();
      if (current) {
        this.seriesSelect.emit(current.kind);
      }
      this.select.emit(selected);
    }
  }
}
