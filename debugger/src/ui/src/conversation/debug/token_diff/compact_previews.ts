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

import {Injectable, computed, inject, signal} from '@angular/core';

import {formatCount} from '../../../widgets/format/format';
import {capturedTokenLabel, tokenMarkup} from '../../conversation_content';
import {ConversationState} from '../../conversation_state';
import type {TokenPair} from '../../token_alignment';
import {TokenDiff} from './token_diff';
import {TokenDiffDom} from './token_diff_dom';

interface CompactNodeLayout {
  available: number;
  widths: Map<number, number>;
}

function sameCompactLayout(
  left: ReadonlyMap<string, CompactNodeLayout>,
  right: ReadonlyMap<string, CompactNodeLayout>,
): boolean {
  if (left.size !== right.size) {
    return false;
  }
  for (const [key, leftEntry] of left) {
    const rightEntry = right.get(key);
    if (
      !rightEntry ||
      leftEntry.available !== rightEntry.available ||
      leftEntry.widths.size !== rightEntry.widths.size
    ) {
      return false;
    }
    for (const [index, width] of leftEntry.widths) {
      if (rightEntry.widths.get(index) !== width) {
        return false;
      }
    }
  }
  return true;
}

/**
 * The previews of collapsed Token Diff nodes: as many of a stage's first and last aligned pairs
 * as fit in one line, by their measured widths, with the stage's counts.
 */
@Injectable()
export class CompactPreviews {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly dom = inject(TokenDiffDom);

  private readonly compactLayout = signal(new Map<string, CompactNodeLayout>());

  compactTokenHtml(text: string): string {
    return tokenMarkup(text, false);
  }

  /** Measures the previews' room and their tokens' widths. Run after rendering. */
  syncLayout(): void {
    this.tokens.boundaries();
    this.tokens.color();
    this.tokens.collapsed();
    this.tokens.panelWidth();
    this.tokens.details();
    this.dom.layoutTick();
    this.tokens.aligned();
    this.conversation.turns();
    const host = this.conversation.element.nativeElement as HTMLElement;
    const next = new Map<string, CompactNodeLayout>();
    for (const container of Array.from(host.querySelectorAll<HTMLElement>('.nodeContextText'))) {
      const node = container.closest<HTMLElement>('[data-node-stage]');
      const sample = container.querySelector<HTMLElement>('.nodePreviewToken');
      const stage = node?.dataset['nodeStage'];
      if (!node || !sample || !stage) {
        continue;
      }
      const turn = Number(node.dataset['nodeTurn']);
      const key = `${turn}:${stage}`;
      const available = Math.floor(container.getBoundingClientRect().width);
      const previous = next.get(key);
      if (previous) {
        previous.available = Math.min(previous.available, available);
        continue;
      }
      const probe = sample.cloneNode(false) as HTMLElement;
      probe.removeAttribute('id');
      probe.style.cssText =
        'position:fixed;visibility:hidden;pointer-events:none;width:max-content;max-width:none;min-width:0;overflow:visible;';
      container.appendChild(probe);
      const pairs = this.tokens.compactPairs(turn, stage);
      const widths = new Map<number, number>();
      for (const pair of [...pairs.slice(0, 6), ...pairs.slice(-6)]) {
        let width = 20;
        for (const token of [pair.ref, pair.target]) {
          probe.innerHTML = tokenMarkup(token ? capturedTokenLabel(token) : '∅', false);
          width = Math.max(width, Math.ceil(probe.getBoundingClientRect().width));
        }
        widths.set(pair.index, width);
      }
      probe.remove();
      next.set(key, {available, widths});
    }
    this.compactLayout.update((old) => (sameCompactLayout(old, next) ? old : next));
  }
  readonly compactTokens = computed(() => {
    const result = new Map<
      string,
      {
        text: string;
        pair: TokenPair;
        index: number;
        step: number | null;
        ellipsis: boolean;
        match: boolean | null;
        missing: boolean;
        width: number;
      }[]
    >();
    for (const turn of this.conversation.turns()) {
      for (const stage of ['prefill', 'decode'] as const) {
        const pairs = this.tokens.compactPairs(turn.n, stage);
        const layout = this.compactLayout().get(`${turn.n}:${stage}`);
        const width = (pair: TokenPair) => layout?.widths.get(pair.index) ?? 120;
        let count = Math.min(6, Math.ceil(pairs.length / 2));
        let preview = pairs;
        while (count > 0) {
          preview =
            pairs.length > count * 2 ? [...pairs.slice(0, count), ...pairs.slice(-count)] : pairs;
          const omitted = preview.length < pairs.length;
          const needed =
            preview.reduce((sum, pair) => sum + width(pair), 0) +
            Math.max(0, preview.length - 1) * 4 +
            (omitted ? 19 : 0);
          if (!layout || needed <= layout.available || count === 1) {
            break;
          }
          count--;
        }
        for (const side of ['ref', 'target'] as const) {
          result.set(
            `${turn.n}:${side}:${stage}`,
            preview.map((pair, i) => ({
              text: pair[side] ? capturedTokenLabel(pair[side]) : '∅',
              pair,
              index: pair.index,
              step: pair[side]?.step ?? null,
              ellipsis: preview.length < pairs.length && i === count,
              match: pair.match,
              missing: !pair[side],
              width: width(pair),
            })),
          );
        }
      }
    }
    return result;
  });
  compactColumns(turn: number, stage: string): string {
    return (this.compactTokens().get(`${turn}:ref:${stage}`) ?? [])
      .flatMap((t) => [...(t.ellipsis ? ['16px'] : []), `${t.width}px`])
      .join(' ');
  }
  compactInput(turn: number, side: string): string {
    const captured = this.conversation.captured(turn, side);
    const text = captured?.input ?? 'Input text not captured';
    return text.replace(/\s+/g, ' ').slice(0, 200);
  }
  compactOutput(turn: number, side: string): string {
    return (this.conversation.captured(turn, side)?.output ?? 'Response text not captured')
      .replace(/\s+/g, ' ')
      .slice(0, 200);
  }
  compactCount(turn: number, side: string, stage: string): string {
    const count = this.tokens.stageCount(turn, side, stage);
    if (typeof count !== 'number') {
      return count;
    }
    const pairs =
      stage === 'prefill'
        ? (this.tokens.inputPairRows(turn) ?? []).flatMap((row) => [
            ...row.leading,
            ...row.text,
            ...row.trailing,
          ])
        : this.tokens.rows(turn);
    const gaps = pairs.filter((pair) => !this.tokens.pairToken(pair, side)).length;
    return `${formatCount(count)} tokens · ${formatCount(gaps)} gaps`;
  }
}
