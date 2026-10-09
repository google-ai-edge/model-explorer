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
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterRenderEffect,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import type {
  KvMetric,
  KvRangeBin,
  KvRangeCell,
  KvRangeRow,
} from '../../../data/contracts/kv';
import {
  KV_RANGE_LEFT,
  KV_RANGE_RIGHT,
  KV_RANGE_ROW_HEIGHT,
  KV_RANGE_TOP,
  rangeBinAtX,
  rangeCapacity,
  rangeColumnWidth,
  rangeDragStarted,
  rangeFromBins,
  rangeFromBrush,
  rangeKeyboardBrush,
  rangeKeyboardSelection,
  rangeOverlappingBins,
  rangeSelectionFrame,
  rangeTrendSegments,
  rangeXAtPosition,
} from './kv_range_geometry';
import {KvRangeSelection} from './kv_range_types';

interface Gesture {
  pointer: number;
  target: SVGSVGElement;
  clientX: number;
  first: number;
  firstX: number;
  lastX: number;
  layer: number | null;
  dragging: boolean;
}

/** A bounded range chart. Fetching, range history and the shared controls belong to its parent. */
@Component({
  selector: 'kv-range-chart',
  standalone: true,
  templateUrl: './kv_range_chart.ng.html',
  styleUrl: './kv_range_chart.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvRangeChart {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly view = input<'heatmap' | 'trends'>('heatmap');
  readonly layers = input<readonly number[]>([]);
  readonly bins = input<readonly KvRangeBin[]>([]);
  readonly visibleRange = input<KvRangeBin | null>(null);
  readonly rows = input<readonly KvRangeRow[]>([]);
  readonly metric = input<KvMetric>('relative_l2');
  readonly selected = input<KvRangeSelection | null>(null);
  readonly trendLayer = input(0);
  readonly filtered = input(false);
  readonly disabled = input(false);
  readonly color = input<(value: number | null) => string>((value) =>
    value === null ? 'var(--kv-missing)' : 'var(--kv-accent)',
  );
  readonly format = input<(value: number | null) => string>((value) =>
    value === null ? 'Unavailable' : String(value),
  );
  readonly select = output<KvRangeSelection>();
  readonly zoom = output<KvRangeBin>();
  readonly capacity = output<number>();
  readonly rowHeight = KV_RANGE_ROW_HEIGHT;
  readonly left = KV_RANGE_LEFT;
  readonly top = KV_RANGE_TOP;
  private readonly viewport = viewChild<ElementRef<HTMLDivElement>>('viewport');
  private readonly size = signal({width: 0, height: 0});
  private gesture: Gesture | null = null;
  private keyboardGesture: {anchorX: number; focusX: number} | null = null;
  private lastCapacity = 0;
  readonly brush = signal<KvRangeBin | null>(null);
  readonly announcement = signal('');
  readonly columnWidth = computed(() =>
    rangeColumnWidth(this.size().width, this.bins(), this.visibleRange()),
  );
  readonly matchRadius = computed(() => Math.min(2.5, this.columnWidth() / 3));
  readonly width = computed(() =>
    Math.max(
      160,
      this.size().width,
      this.view() === 'heatmap'
        ? this.left + this.bins().length * this.columnWidth() + KV_RANGE_RIGHT
        : 0,
    ),
  );
  readonly dataHeight = computed(() => this.layers().length * this.rowHeight);
  readonly height = computed(() =>
    Math.max(
      this.size().height,
      this.view() === 'heatmap' ? this.top + this.dataHeight() + 8 : 240,
    ),
  );
  readonly plotBottom = computed(() =>
    this.view() === 'heatmap'
      ? this.top + this.dataHeight()
      : this.height() - 44,
  );
  readonly plotSpan = computed(() =>
    this.view() === 'heatmap'
      ? this.bins().length * this.columnWidth()
      : Math.max(1, this.width() - this.left - KV_RANGE_RIGHT),
  );
  readonly metricLabel = computed(
    () =>
      ({
        relative_l2: 'Relative L2',
        max_abs: 'Max |Δ|',
        cosine_distance: 'Cosine distance',
      })[this.metric()],
  );
  private readonly rowDetails = computed(
    () => new Map(this.rows().map((row) => [row.layer, row])),
  );
  private readonly rowIndex = computed(
    () =>
      new Map(
        this.rows().map((row) => [
          row.layer,
          new Map(row.bins.map((bin) => [`${bin.start}:${bin.end}`, bin])),
        ]),
      ),
  );
  readonly ticks = computed(() => {
    const result: {bin: KvRangeBin; index: number}[] = [];
    let previous = -Infinity,
      previousWidth = 0;
    for (const [index, bin] of this.bins().entries()) {
      const x = (this.binStartX(index) + this.binEndX(index)) / 2;
      const labelWidth = String(bin.start).length * 6;
      if (x - previous < Math.max(14, (previousWidth + labelWidth) / 2 + 8))
        continue;
      result.push({bin, index});
      previous = x;
      previousWidth = labelWidth;
    }
    return result;
  });
  readonly trendSegments = computed(() =>
    rangeTrendSegments(
      this.bins(),
      this.bins().map(
        (bin) =>
          this.cell(this.trendLayer(), bin)?.metrics[this.metric()] ?? null,
      ),
    ),
  );
  readonly trendPoints = computed(() => this.trendSegments().flat());
  readonly trendMaximum = computed(
    () =>
      this.trendPoints().reduce(
        (max, point) => Math.max(max, point.value),
        0,
      ) || 1,
  );
  readonly trendPaths = computed(() =>
    this.trendSegments().map((segment) =>
      segment
        .map(
          (point, i) =>
            `${i ? 'L' : 'M'}${this.trendX(point.position)},${this.trendY(point.value)}`,
        )
        .join(' '),
    ),
  );
  readonly selectionBox = computed(() => {
    const selected = this.selected(),
      indices = rangeOverlappingBins(this.bins(), selected);
    if (!selected || !indices) return null;
    const layerIndex = this.layers().indexOf(selected.layer ?? -1);
    if (
      selected.layer !== null &&
      (this.view() === 'heatmap'
        ? layerIndex < 0
        : selected.layer !== this.trendLayer())
    )
      return null;
    return {
      x: this.binStartX(indices[0]),
      width: this.binEndX(indices[1]) - this.binStartX(indices[0]),
      y:
        this.view() === 'heatmap' && selected.layer !== null
          ? this.top + layerIndex * this.rowHeight
          : this.top,
      height:
        this.view() === 'heatmap' && selected.layer !== null
          ? this.rowHeight
          : this.plotBottom() - this.top,
    };
  });
  readonly selectionFrame = computed(() => {
    const box = this.selectionBox();
    return box ? rangeSelectionFrame(box) : null;
  });
  readonly brushBox = computed(() => {
    const range = this.brush();
    if (!range) return null;
    const x = rangeXAtPosition(
      this.bins(),
      range.start,
      this.left,
      this.plotSpan(),
      this.view() === 'heatmap',
    );
    return {
      x,
      width:
        rangeXAtPosition(
          this.bins(),
          range.end,
          this.left,
          this.plotSpan(),
          this.view() === 'heatmap',
        ) - x,
    };
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.cancelGesture());
    effect((onCleanup) => {
      const observer = new ResizeObserver(([entry]) => {
        const width = Math.floor(entry.contentRect.width),
          height = Math.floor(entry.contentRect.height);
        if (width <= 0 || height <= 0) return;
        this.cancelGesture();
        this.size.set({width, height});
        const capacity = rangeCapacity(width);
        if (capacity !== this.lastCapacity) {
          this.lastCapacity = capacity;
          this.capacity.emit(capacity);
        }
      });
      // Observe the host, so a view's vertical scrollbar cannot change capacity.
      observer.observe(this.host.nativeElement);
      onCleanup(() => observer.disconnect());
    });
    effect(() => {
      this.view();
      this.bins();
      this.visibleRange();
      this.layers();
      this.trendLayer();
      this.disabled();
      this.selected();
      untracked(() => this.cancelGesture());
    });
    afterRenderEffect(() => {
      const box = this.selectionBox(),
        viewport = this.viewport()?.nativeElement;
      if (!box || !viewport || this.gesture) return;
      if (box.width <= viewport.clientWidth) {
        if (box.x < viewport.scrollLeft) viewport.scrollLeft = box.x;
        else if (box.x + box.width > viewport.scrollLeft + viewport.clientWidth)
          viewport.scrollLeft = box.x + box.width - viewport.clientWidth;
      }
      if (
        this.selected()?.layer !== null &&
        box.height <= viewport.clientHeight
      ) {
        if (box.y < viewport.scrollTop) viewport.scrollTop = box.y;
        else if (
          box.y + box.height >
          viewport.scrollTop + viewport.clientHeight
        )
          viewport.scrollTop = box.y + box.height - viewport.clientHeight;
      }
    });
  }

  cell(layer: number, bin: KvRangeBin): KvRangeCell | null {
    return this.rowIndex().get(layer)?.get(`${bin.start}:${bin.end}`) ?? null;
  }
  value(layer: number, bin: KvRangeBin) {
    const summary = this.cell(layer, bin)?.metrics[this.metric()];
    return summary &&
      summary.valid_count > 0 &&
      summary.max !== null &&
      Number.isFinite(summary.max)
      ? summary.max
      : null;
  }
  isMatch(layer: number, bin: KvRangeBin) {
    return (this.cell(layer, bin)?.match_count ?? 0) > 0;
  }
  rangeLabel(bin: KvRangeBin) {
    return bin.end - bin.start === 1
      ? `Position ${bin.start}`
      : `Positions ${bin.start}–${bin.end - 1}`;
  }
  tickLabel(bin: KvRangeBin) {
    return String(bin.start);
  }
  columnTitle(bin: KvRangeBin) {
    return `${this.rangeLabel(bin)} · All layers\n${bin.end - bin.start === 1 ? 'Select token position' : 'Select position range'} across layers. Drag horizontally to zoom.`;
  }
  cellTitle(layer: number, bin: KvRangeBin) {
    const summary = this.cell(layer, bin)?.metrics[this.metric()],
      format = this.format();
    const lines = [
      `Layer ${layer} · ${this.rangeLabel(bin)}`,
      `${bin.end - bin.start} position${bin.end - bin.start === 1 ? '' : 's'}`,
    ];
    if (!summary || summary.valid_count <= 0)
      lines.push(`${this.metricLabel()}: Unavailable`);
    else if (bin.end - bin.start === 1)
      lines.push(`${this.metricLabel()}: ${format(summary.max)}`);
    else
      lines.push(
        `${this.metricLabel()} min: ${format(summary.min)} at Position ${summary.min_position ?? 'Unavailable'}`,
        `${this.metricLabel()} max: ${format(summary.max)} at Position ${summary.max_position ?? 'Unavailable'}`,
      );
    lines.push(`${summary?.valid_count ?? 0} valid positions`);
    if (this.filtered())
      lines.push(
        `${this.cell(layer, bin)?.match_count ?? 0} matching positions`,
      );
    const row = this.rowDetails().get(layer);
    if (row?.reason) lines.push(row.reason);
    else if (
      (!summary || summary.valid_count <= 0) &&
      row?.status &&
      row.status !== 'ok'
    )
      lines.push(row.status.replaceAll('_', ' '));
    return lines.join('\n');
  }
  binStartX(index: number) {
    if (this.view() === 'heatmap')
      return this.left + index * this.columnWidth();
    const bins = this.bins(),
      start = bins[0]?.start ?? 0,
      end = bins[bins.length - 1]?.end ?? start + 1;
    return (
      this.left +
      ((bins[index].start - start) / Math.max(1, end - start)) * this.plotSpan()
    );
  }
  binEndX(index: number) {
    if (this.view() === 'heatmap')
      return this.left + (index + 1) * this.columnWidth();
    const bins = this.bins(),
      start = bins[0]?.start ?? 0,
      end = bins[bins.length - 1]?.end ?? start + 1;
    return (
      this.left +
      ((bins[index].end - start) / Math.max(1, end - start)) * this.plotSpan()
    );
  }
  trendX(position: number) {
    const bins = this.bins(),
      start = bins[0]?.start ?? 0,
      end = bins[bins.length - 1]?.end ?? start + 1;
    return (
      this.left +
      ((position - start + 0.5) / Math.max(1, end - start)) * this.plotSpan()
    );
  }
  trendY(value: number) {
    return (
      this.plotBottom() -
      (value / this.trendMaximum()) * Math.max(1, this.plotBottom() - this.top)
    );
  }
  pointSelected(position: number) {
    const selected = this.selected();
    return (
      !!selected &&
      (selected.layer === null || selected.layer === this.trendLayer()) &&
      position >= selected.start &&
      position < selected.end
    );
  }

  private eventPoint(event: PointerEvent, target: SVGSVGElement) {
    const rect = target.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) * this.width()) / Math.max(1, rect.width),
      y:
        ((event.clientY - rect.top) * this.height()) / Math.max(1, rect.height),
    };
  }
  pointerDown(event: PointerEvent) {
    if (
      this.disabled() ||
      event.button !== 0 ||
      !event.isPrimary ||
      this.gesture
    )
      return;
    const target = event.currentTarget as SVGSVGElement,
      point = this.eventPoint(event, target);
    const index = rangeBinAtX(
      this.bins(),
      point.x,
      this.left,
      this.plotSpan(),
      this.view() === 'heatmap',
    );
    if (index < 0 || point.y < 4 || point.y > this.plotBottom()) return;
    const axis = point.y < this.top;
    const layer = axis
      ? null
      : this.view() === 'trends'
        ? this.trendLayer()
        : this.layers()[Math.floor((point.y - this.top) / this.rowHeight)];
    if (layer === undefined) return;
    event.preventDefault();
    this.cancelGesture();
    target.focus({preventScroll: true});
    this.gesture = {
      pointer: event.pointerId,
      target,
      clientX: event.clientX,
      first: index,
      firstX: point.x,
      lastX: point.x,
      layer,
      dragging: false,
    };
    target.setPointerCapture(event.pointerId);
  }
  pointerMove(event: PointerEvent) {
    const gesture = this.gesture;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    const point = this.eventPoint(event, gesture.target);
    gesture.lastX = point.x;
    gesture.dragging ||= rangeDragStarted(gesture.clientX, event.clientX);
    if (gesture.dragging)
      this.brush.set(
        rangeFromBrush(
          this.bins(),
          gesture.firstX,
          gesture.lastX,
          this.left,
          this.plotSpan(),
          this.view() === 'heatmap',
        ),
      );
  }
  pointerUp(event: PointerEvent) {
    const gesture = this.gesture;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    this.pointerMove(event);
    const range = gesture.dragging
      ? rangeFromBrush(
          this.bins(),
          gesture.firstX,
          gesture.lastX,
          this.left,
          this.plotSpan(),
          this.view() === 'heatmap',
        )
      : rangeFromBins(this.bins(), gesture.first, gesture.first);
    this.cancelGesture();
    if (!range || this.disabled()) return;
    if (gesture.dragging) {
      this.announcement.set(`Zoom to ${this.rangeLabel(range)}.`);
      this.zoom.emit(range);
    } else {
      this.announcement.set(
        `${this.rangeLabel(range)}, ${gesture.layer === null ? 'all layers' : `Layer ${gesture.layer}`} selected.`,
      );
      this.select.emit({...range, layer: gesture.layer});
    }
  }
  pointerCancel(event: PointerEvent) {
    if (this.gesture?.pointer === event.pointerId) this.cancelGesture();
  }
  cancelBrush() {
    this.cancelGesture();
  }
  private cancelGesture() {
    const gesture = this.gesture;
    this.gesture = null;
    this.keyboardGesture = null;
    this.brush.set(null);
    if (gesture?.target.hasPointerCapture(gesture.pointer))
      gesture.target.releasePointerCapture(gesture.pointer);
  }
  keyDown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      if (this.gesture || this.keyboardGesture) {
        event.preventDefault();
        event.stopPropagation();
        this.cancelGesture();
        this.announcement.set('Range selection cancelled.');
      }
      return;
    }
    if (this.disabled()) return;
    if (
      event.shiftKey &&
      (event.key === 'ArrowLeft' || event.key === 'ArrowRight')
    ) {
      event.preventDefault();
      event.stopPropagation();
      if (this.gesture) this.cancelGesture();
      const next = rangeKeyboardBrush(
        this.bins(),
        this.selected(),
        this.keyboardGesture,
        event.key === 'ArrowLeft' ? -1 : 1,
        this.left,
        this.plotSpan(),
        this.view() === 'heatmap',
      );
      if (next) {
        this.keyboardGesture = next;
        this.brush.set(next.range);
        this.announcement.set(
          `Zoom range: ${this.rangeLabel(next.range)}. Enter to zoom, Escape to cancel.`,
        );
      }
      return;
    }
    if (event.key === 'Enter' && this.keyboardGesture) {
      event.preventDefault();
      event.stopPropagation();
      const range = this.brush();
      this.cancelGesture();
      if (range) {
        this.announcement.set(`Zoom to ${this.rangeLabel(range)}.`);
        this.zoom.emit(range);
      }
      return;
    }
    const selection = rangeKeyboardSelection(
      this.bins(),
      this.layers(),
      this.selected(),
      event.key,
      this.view() === 'trends' ? this.trendLayer() : null,
    );
    if (!selection) return;
    event.preventDefault();
    event.stopPropagation();
    this.cancelGesture();
    this.announcement.set(
      `${this.rangeLabel(selection)}, ${selection.layer === null ? 'all layers' : `Layer ${selection.layer}`} selected.`,
    );
    this.select.emit(selection);
  }
}
