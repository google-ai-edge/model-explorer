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
  ElementRef,
  computed,
  effect,
  input,
  model,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import type {KvHeadChannel, KvHeadEvidence} from '../../../data/contracts/kv';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {formatMetric} from '../../../shared/format/format';
import {KvMetricValues} from './kv_metric_values';

type Side = 'ref' | 'target';
interface Domain {
  low: number;
  high: number;
  scale: number;
}
const PAGE_SIZE = 256;

/** Exact head evidence only. Its wrapper owns fetching; its parent owns persistent UI choices. */
@Component({
  selector: 'kv-head-details',
  standalone: true,
  imports: [
    MatIconModule,
    ExplorerInfoSection,
    ExplorerInfoValue,
    KvMetricValues,
  ],
  templateUrl: './kv_head_details.ng.html',
  styleUrl: './kv_head_details.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvHeadDetails {
  readonly evidence = input<KvHeadEvidence | null>(null);
  readonly loading = input(false);
  readonly error = input('');
  readonly retry = output<void>();
  readonly view = model<'index' | 'scatter'>('index');
  readonly channel = signal(0);
  readonly windowStart = signal(0);
  readonly copyStatus = signal('');
  readonly numericalExpanded = model(true);
  readonly channelsExpanded = model(true);
  private readonly plotContainer =
    viewChild<ElementRef<HTMLElement>>('plotContainer');
  readonly plotWidth = signal(290);
  readonly plotHeight = 230;
  readonly plotViewBox = computed(
    () => `0 0 ${this.plotWidth()} ${this.plotHeight}`,
  );
  readonly sides: Side[] = ['ref', 'target'];
  readonly channels = computed(() =>
    [...(this.evidence()?.channels ?? [])].sort(
      (a, b) => a.channel - b.channel,
    ),
  );
  readonly selectedIndex = computed(() =>
    Math.max(
      0,
      this.channels().findIndex((c) => c.channel === this.channel()),
    ),
  );
  readonly selected = computed<KvHeadChannel | undefined>(
    () => this.channels()[this.selectedIndex()],
  );
  readonly plotChannels = computed(() =>
    this.channels().slice(this.windowStart(), this.windowStart() + PAGE_SIZE),
  );
  readonly hasPages = computed(() => this.channels().length > PAGE_SIZE);
  readonly canNextPage = computed(
    () => this.windowStart() + PAGE_SIZE < this.channels().length,
  );
  readonly lastChannel = computed(() => this.channels().at(-1)?.channel ?? 0);
  readonly firstChannel = computed(() => this.channels()[0]?.channel ?? 0);
  readonly largest = computed(() => {
    const value = this.evidence()?.largest_channel;
    return value != null &&
      this.channels().some(
        (c) => c.channel === value && this.finite(c.abs_delta),
      )
      ? value
      : null;
  });
  readonly scalarRows = computed(() => {
    const point = this.selected();
    return [
      {label: 'Ref', value: point?.ref ?? null, signed: false},
      {label: 'Target', value: point?.target ?? null, signed: false},
      {label: 'Signed Δ', value: point?.delta ?? null, signed: true},
      {label: '|Δ|', value: point?.abs_delta ?? null, signed: false},
    ];
  });
  readonly finitePairs = computed(() =>
    ['ok', 'partial'].includes(this.evidence()?.status ?? '')
      ? this.plotChannels().filter(
          (c) => this.finite(c.ref) && this.finite(c.target),
        )
      : [],
  );
  readonly hasValues = computed(() =>
    this.plotChannels().some(
      (c) => this.finite(c.ref) || this.finite(c.target),
    ),
  );
  readonly domain = computed<Domain>(() => {
    // A channel-page change must not change the head's value scale.
    const rows =
      this.view() === 'scatter'
        ? this.channels().filter(
            (c) => this.finite(c.ref) && this.finite(c.target),
          )
        : this.channels();
    return this.valueDomain(
      rows
        .flatMap((c) => [c.ref, c.target])
        .filter((v): v is number => this.finite(v)),
    );
  });
  readonly box = computed(() => {
    const left = this.view() === 'scatter' ? 64 : 52;
    return {
      left,
      right: Math.max(left + 1, this.plotWidth() - 12),
      top: 30,
      bottom: this.plotHeight - 42,
    };
  });
  readonly axisDomains = computed(() => {
    const base = this.domain(),
      box = this.box(),
      width = box.right - box.left,
      height = box.bottom - box.top;
    const expanded = (factor: number): Domain => {
      const center = (base.low + base.high) / 2,
        span = (base.high - base.low) * factor;
      return {
        low: center - span / 2,
        high: center + span / 2,
        scale: base.scale,
      };
    };
    // Scatter uses equal value units on both axes, including in tall/narrow panels.
    return this.view() === 'scatter'
      ? {
          x: expanded(Math.max(1, width / height)),
          y: expanded(Math.max(1, height / width)),
        }
      : {x: base, y: base};
  });
  readonly xTickCount = computed(() =>
    Math.max(
      2,
      Math.min(7, Math.floor((this.box().right - this.box().left) / 65) + 1),
    ),
  );
  readonly tickValues = computed(() =>
    this.domainTicks(this.axisDomains().y, 5),
  );
  readonly scatterTicks = computed(() =>
    this.domainTicks(this.axisDomains().x, this.xTickCount()),
  );
  readonly channelTicks = computed(() => {
    const rows = this.plotChannels();
    if (!rows.length) return [];
    return [
      ...new Set(
        Array.from(
          {length: this.xTickCount()},
          (_, i) =>
            rows[Math.round(((rows.length - 1) * i) / (this.xTickCount() - 1))]
              .channel,
        ),
      ),
    ];
  });
  readonly paths = computed(() =>
    this.sides.map((side) => ({side, d: this.channelPath(side)})),
  );

  constructor() {
    effect((onCleanup) => {
      const element = this.plotContainer()?.nativeElement;
      if (!element) return;
      const observer = new ResizeObserver(([entry]) => {
        if (entry.contentRect.width > 0)
          this.plotWidth.set(entry.contentRect.width);
      });
      observer.observe(element);
      onCleanup(() => observer.disconnect());
    });
    effect(() => {
      const data = this.evidence();
      // An evidence response is tied to one exact head/position/batch selection.
      const rows = this.channels(),
        largest = data?.largest_channel;
      const first =
        largest != null && rows.some((c) => c.channel === largest)
          ? largest
          : (rows[0]?.channel ?? 0);
      this.channel.set(first);
      const index = Math.max(
        0,
        rows.findIndex((c) => c.channel === first),
      );
      this.windowStart.set(Math.floor(index / PAGE_SIZE) * PAGE_SIZE);
      this.copyStatus.set('');
    });
  }

  finite(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
  }
  readonly shortNumber = formatMetric;
  valueText(value: number | string | null | undefined, signed = false) {
    if (value == null) return 'Unavailable';
    if (typeof value === 'string') return value;
    if (Object.is(value, -0)) return '-0';
    return (signed && value > 0 ? '+' : '') + String(value);
  }
  private valueDomain(values: number[]): Domain {
    if (!values.length) return {low: -1, high: 1, scale: 1};
    let min = values[0],
      max = values[0];
    for (const value of values) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const scale = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const low = min / scale,
      high = max / scale,
      pad = Math.max((high - low) * 0.08, 0.08);
    const limit = Number.MAX_VALUE / scale;
    return {
      low: Math.max(-limit, low - pad),
      high: Math.min(limit, high + pad),
      scale,
    };
  }
  private domainTicks(domain: Domain, count: number) {
    return Array.from(
      {length: count},
      (_, i) =>
        (domain.low + ((domain.high - domain.low) * i) / (count - 1)) *
        domain.scale,
    ).filter(Number.isFinite);
  }
  x(channel: number) {
    const rows = this.plotChannels(),
      first = rows[0]?.channel ?? 0,
      last = rows.at(-1)?.channel ?? first;
    const box = this.box();
    return (
      box.left +
      ((channel - first + 0.5) / Math.max(1, last - first + 1)) *
        (box.right - box.left)
    );
  }
  valueY(value: number) {
    const d = this.axisDomains().y,
      b = this.box();
    return (
      b.bottom -
      ((value / d.scale - d.low) / (d.high - d.low)) * (b.bottom - b.top)
    );
  }
  scatterX(value: number) {
    const d = this.axisDomains().x,
      b = this.box();
    return (
      b.left +
      ((value / d.scale - d.low) / (d.high - d.low)) * (b.right - b.left)
    );
  }
  private channelPath(side: Side) {
    let path = '',
      connected = false,
      previous = -2;
    for (const row of this.plotChannels()) {
      const value = row[side];
      if (!this.finite(value)) {
        connected = false;
        continue;
      }
      path +=
        (connected && row.channel === previous + 1 ? 'L' : 'M') +
        this.x(row.channel) +
        ',' +
        this.valueY(value) +
        ' ';
      connected = true;
      previous = row.channel;
    }
    return path;
  }
  pointTitle(row: KvHeadChannel) {
    return `Channel ${row.channel}\nRef ${this.valueText(row.ref)}\nTarget ${this.valueText(row.target)}\nSigned Δ ${this.valueText(row.delta, true)}\n|Δ| ${this.valueText(row.abs_delta)}`;
  }
  selectChannel(value: number) {
    const rows = this.channels();
    if (!rows.length || !Number.isFinite(value)) return;
    const nearest = rows.reduce((best, row) =>
      Math.abs(row.channel - value) < Math.abs(best.channel - value)
        ? row
        : best,
    );
    this.channel.set(nearest.channel);
    const index = rows.indexOf(nearest);
    this.windowStart.set(Math.floor(index / PAGE_SIZE) * PAGE_SIZE);
    this.copyStatus.set('');
  }
  moveChannel(delta: number) {
    const rows = this.channels(),
      next =
        rows[
          Math.max(0, Math.min(rows.length - 1, this.selectedIndex() + delta))
        ];
    if (next) this.selectChannel(next.channel);
  }
  movePage(delta: number) {
    const rows = this.channels(),
      index = Math.max(
        0,
        Math.min(rows.length - 1, this.windowStart() + delta * PAGE_SIZE),
      );
    if (rows[index]) this.selectChannel(rows[index].channel);
  }
  commitChannel(event: Event) {
    const field = event.target as HTMLInputElement;
    if (field.value.trim() !== '' && Number.isInteger(Number(field.value)))
      this.selectChannel(Number(field.value));
    field.value = String(this.selected()?.channel ?? 0);
  }
  channelInputKey(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.commitChannel(event);
    } else if (event.key === 'Escape') {
      event.stopPropagation();
      (event.target as HTMLInputElement).value = String(
        this.selected()?.channel ?? 0,
      );
    }
  }
  chartKey(event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === 'Home') this.selectChannel(this.firstChannel());
    else if (event.key === 'End') this.selectChannel(this.lastChannel());
    else this.moveChannel(event.key === 'ArrowLeft' ? -1 : 1);
  }
  async copyValue(
    label: string,
    value: number | string | null,
    signed = false,
  ) {
    if (value === null) return;
    try {
      await navigator.clipboard.writeText(this.valueText(value, signed));
      this.copyStatus.set(`Copied ${label}.`);
    } catch {
      this.copyStatus.set('Copy unavailable. Select the value to copy it.');
    }
  }
}
