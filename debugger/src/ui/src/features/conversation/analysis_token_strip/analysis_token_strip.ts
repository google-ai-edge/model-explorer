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
  afterNextRender,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {KvRangeBin} from '../../../data/contracts/kv';
import {tokenMarkup, capturedTokenLabel, isInvisibleChar} from '../conversation_content';
import {
  axisColumns,
  tokenPages,
  type AnalysisAxisBin,
  type AnalysisAxisToken,
} from './analysis_token_model';

@Component({
  selector: 'analysis-token-strip',
  imports: [MatIconModule, MatTooltipModule],
  templateUrl: './analysis_token_strip.ng.html',
  styleUrl: './analysis_token_strip.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AnalysisTokenStrip {
  readonly context = input.required<string>();
  readonly tokens = input<readonly AnalysisAxisToken[]>([]);
  readonly bins = input<readonly AnalysisAxisBin[]>([]);
  readonly selected = input<KvRangeBin | null>(null);
  readonly disabled = input(false);
  readonly selectRange = output<KvRangeBin>();
  private readonly viewport = viewChild<ElementRef<HTMLElement>>('viewport');
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  private readonly width = signal(600);
  private readonly font = signal('12px Arial', {equal: () => false});
  private readonly pageAnchor = signal<number | null>(null);
  private readonly measure = document.createElement('canvas').getContext('2d');
  readonly columns = computed(() => axisColumns(this.tokens(), this.bins()));
  readonly widths = computed(() => {
    const context = this.measure;
    if (context) context.font = this.font();
    return this.columns().map((column) => {
      const sides = ['ref', 'target'] as const;
      return (
        9 +
        Math.ceil(
          Math.max(
            ...sides.map(
              (side) =>
                column.tokens.reduce(
                  (sum, t) =>
                    sum + this.textWidth(t.pair[side] ? capturedTokenLabel(t.pair[side]) : '∅'),
                  0,
                ) +
                (column.before ? this.textWidth('…') : 0) +
                (column.after ? this.textWidth('…') : 0),
            ),
          ),
        )
      );
    });
  });
  readonly pages = computed(() => tokenPages(this.widths(), Math.max(1, this.width())));
  readonly page = computed(() => {
    const anchor = this.pageAnchor(),
      columns = this.columns();
    const index = columns.findIndex((c) => anchor !== null && c.start <= anchor && c.end > anchor);
    return Math.max(
      0,
      this.pages().findIndex((p) => index >= p.start && index < p.end),
    );
  });
  readonly visible = computed(() => {
    const page = this.pages()[this.page()];
    return page
      ? this.columns()
          .slice(page.start, page.end)
          .map((column, i) => ({
            ...column,
            width: Math.min(this.width(), this.widths()[page.start + i]),
          }))
      : [];
  });
  readonly pageLabel = computed(
    () => `Token positions, page ${this.page() + 1} of ${Math.max(1, this.pages().length)}`,
  );
  constructor() {
    effect(() => {
      this.context();
      const selection = this.selected();
      // Paging itself never changes selection or snaps back to its page.
      untracked(() => this.pageAnchor.set(selection?.start ?? this.bins()[0]?.start ?? null));
    });
    afterNextRender(() => {
      const viewport = this.viewport()?.nativeElement;
      if (!viewport) return;
      const update = () => this.width.set(Math.max(1, viewport.clientWidth));
      const style = getComputedStyle(this.host.nativeElement);
      const font = `12px ${style.getPropertyValue('--me-font-family') || style.fontFamily}`;
      this.font.set(font);
      // A fallback font can have different widths; repaginate once trace fonts finish loading.
      void document.fonts.ready.then(() => {
        if (!this.destroyRef.destroyed) this.font.set(font);
      });
      const observer = new ResizeObserver(update);
      observer.observe(viewport);
      update();
      this.destroyRef.onDestroy(() => observer.disconnect());
    });
  }
  move(delta: number) {
    const page = this.pages()[this.page() + delta];
    if (page) this.pageAnchor.set(this.columns()[page.start].start);
  }
  isSelected(column: KvRangeBin) {
    const selected = this.selected();
    return !!selected && selected.start < column.end && selected.end > column.start;
  }
  label(column: KvRangeBin) {
    return column.end - column.start === 1
      ? String(column.start)
      : `${column.start}–${column.end - 1}`;
  }
  private textWidth(text: string) {
    const context = this.measure;
    if (!context) return 24;
    // Match the shared trace's fixed whitespace slots and code-point badges.
    return Array.from(text).reduce((width, ch) => {
      if (ch === ' ' || ch === '\n' || ch === '\r') return width + 7.8;
      if (ch === '\t') return width + context.measureText(' ').width * 8;
      if (isInvisibleChar(ch))
        return width + 6 * ('U+' + ch.codePointAt(0)!.toString(16).padStart(4, '0')).length + 8;
      return width + context.measureText(ch).width;
    }, 0);
  }
  readonly tokenLabel = capturedTokenLabel;
  markup(text: string | null | undefined) {
    return tokenMarkup(text ?? '∅', false);
  }
}
