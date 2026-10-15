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

import {ChangeDetectionStrategy, Component, computed, input, output, signal} from '@angular/core';
import type {KvHeadChannel} from '../../../data/contracts/kv';
import {scatterRepresentatives, type ValueRange} from './kv_scatter_model';

/** Numerical-domain navigator; gestures never select tensor channels. */
@Component({
  selector: 'kv-scatter-overview',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      viewBox="0 0 100 100"
      role="group"
      tabindex="0"
      [attr.aria-label]="
        label() + ' overview. Drag the window to pan; drag its corners to resize. Escape resets.'
      "
      (pointerdown)="begin($event)"
      (pointermove)="move($event)"
      (pointerup)="finish($event)"
      (pointercancel)="cancel($event)"
      (keydown)="navigate($event)"
      (dblclick)="reset($event)"
    >
      <path class="diagonal" d="M 5 95 L 95 5" />
      <path class="points" [attr.d]="points()" />
      <path class="shade" fill-rule="evenodd" [attr.d]="shade()" />
      <rect
        class="window"
        data-action="pan"
        [attr.x]="box().low"
        [attr.y]="100 - box().high"
        [attr.width]="box().high - box().low"
        [attr.height]="box().high - box().low"
      />
      <circle
        class="handle"
        data-action="start"
        [attr.cx]="box().low"
        [attr.cy]="100 - box().low"
        r="4"
      />
      <circle
        class="handle"
        data-action="end"
        [attr.cx]="box().high"
        [attr.cy]="100 - box().high"
        r="4"
      />
    </svg>
  `,
  styles: `
    :host {
      display: block;
      background: var(--me-surface-color, #fff);
      border: 1px solid var(--me-border-color, #dadce0);
      border-radius: 3px;
      overflow: hidden;
    }
    svg {
      display: block;
      width: 100%;
      height: 100%;
      touch-action: none;
      cursor: crosshair;
    }
    svg:focus-visible {
      outline: 2px solid var(--me-primary-color);
      outline-offset: -2px;
    }
    .diagonal {
      fill: none;
      stroke: var(--me-on-surface-color);
      stroke-opacity: 0.3;
      stroke-width: 0.5;
      stroke-dasharray: 2 2;
    }
    .points {
      fill: none;
      stroke: var(--me-primary-color);
      stroke-width: 1.6;
      stroke-linecap: round;
      stroke-opacity: 0.55;
      pointer-events: none;
    }
    .shade {
      fill: var(--me-on-surface-color);
      fill-opacity: 0.09;
      pointer-events: none;
    }
    .window {
      fill: var(--me-primary-color);
      fill-opacity: 0.06;
      stroke: var(--me-primary-color);
      stroke-width: 1.5;
      cursor: grab;
    }
    .window:active {
      cursor: grabbing;
    }
    .handle {
      fill: var(--me-surface-color, #fff);
      stroke: var(--me-primary-color);
      stroke-width: 1.5;
      cursor: nesw-resize;
    }
  `,
})
export class KvScatterOverview {
  readonly rows = input<readonly KvHeadChannel[]>([]);
  readonly full = input.required<ValueRange>();
  readonly range = input.required<ValueRange>();
  readonly label = input('Scatter');
  readonly comparisonAllowed = input(true);
  readonly zoom = output<ValueRange>();
  readonly draft = signal<ValueRange | null>(null);
  readonly points = computed(() => {
    const {start, end} = this.full();
    const p = (n: number) => 5 + ((n - start) / (end - start)) * 90;
    return scatterRepresentatives(this.rows(), this.full(), null, 4096, this.comparisonAllowed())
      .rows.map((r) => `M${p(r.ref).toFixed(2)},${(100 - p(r.target)).toFixed(2)}h.01`)
      .join('');
  });
  readonly box = computed(() => {
    const full = this.full(),
      r = this.draft() ?? this.range();
    const p = (n: number) =>
      5 + Math.max(0, Math.min(1, (n - full.start) / (full.end - full.start))) * 90;
    return {low: p(r.start), high: p(r.end)};
  });
  readonly shade = computed(() => {
    const {low, high} = this.box();
    return `M5 5H95V95H5Z M${low} ${100 - high}H${high}V${100 - low}H${low}Z`;
  });
  private gesture?: {
    id: number;
    x: number;
    y: number;
    width: number;
    range: ValueRange;
    action: string;
  };
  begin(event: PointerEvent) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const svg = event.currentTarget as SVGSVGElement;
    svg.focus();
    svg.setPointerCapture(event.pointerId);
    this.gesture = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      width: svg.getBoundingClientRect().width * 0.9,
      range: this.range(),
      action: (event.target as Element).getAttribute('data-action') ?? 'pan',
    };
  }
  move(event: PointerEvent) {
    const g = this.gesture;
    if (!g || g.id !== event.pointerId) return;
    event.stopPropagation();
    const full = this.full(),
      span = full.end - full.start;
    const delta = ((event.clientX - g.x - (event.clientY - g.y)) / 2 / g.width) * span;
    const min = span / 1000;
    let {start, end} = g.range;
    if (g.action === 'start') start = Math.max(full.start, Math.min(end - min, start + delta));
    else if (g.action === 'end') end = Math.min(full.end, Math.max(start + min, end + delta));
    else {
      const shift = Math.max(full.start - start, Math.min(full.end - end, delta));
      start += shift;
      end += shift;
    }
    const next = {start, end};
    this.draft.set(next);
    this.zoom.emit(next);
  }
  finish(event: PointerEvent) {
    if (this.gesture?.id !== event.pointerId) return;
    event.stopPropagation();
    this.gesture = undefined;
    this.draft.set(null);
    const svg = event.currentTarget as SVGSVGElement;
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  }
  cancel(event: PointerEvent) {
    if (this.gesture?.id !== event.pointerId) return;
    this.zoom.emit(this.gesture.range);
    this.finish(event);
  }
  reset(event: Event) {
    event.preventDefault();
    event.stopPropagation();
    this.zoom.emit(this.full());
  }
  navigate(event: KeyboardEvent) {
    if (event.key === 'Escape' || event.key === 'Home') return this.reset(event);
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const r = this.range(),
      full = this.full();
    const sign = ['ArrowRight', 'ArrowUp'].includes(event.key) ? 1 : -1;
    const shift = Math.max(
      full.start - r.start,
      Math.min(full.end - r.end, sign * (r.end - r.start) * 0.1),
    );
    this.zoom.emit({start: r.start + shift, end: r.end + shift});
  }
}
