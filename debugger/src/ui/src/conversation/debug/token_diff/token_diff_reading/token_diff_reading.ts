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

import {NgTemplateOutlet} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterNextRender,
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import type {CapturedToken} from '../../../../server/contracts/capture';
import {
  UNKNOWN_STOP_HINT,
  capturedInputSegments,
  capturedPhaseText,
  capturedTokenLabel,
  isTemplateToken,
  isUnknownStop,
  pairLineBreaks,
  spokenTokenText,
  tokenMarkup,
  type CapturedInputSegment,
  type TokenPhase,
} from '../../../conversation_content';
import {ConversationState} from '../../../conversation_state';
import {RejectedTurn} from '../../../rejected_turn/rejected_turn';
import {messageRows, type InputRowsOf, type TokenPair} from '../../../token_alignment';
import {StageAnalysisRail} from '../../stage_analysis_rail/stage_analysis_rail';
import {CompactPreviews} from '../compact_previews';
import {TokenDiff} from '../token_diff';
import {TokenDiffDom} from '../token_diff_dom';

const NO_BREAKS: readonly number[] = [];

/**
 * The Debug reading: both runtimes' Token Diff, turn by turn, each stage a node of aligned
 * tokens that collapses to a one-line preview, the stage analysis rails between the nodes and
 * the boundaries between turns.
 */
@Component({
  selector: 'token-diff-reading',
  imports: [
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    NgTemplateOutlet,
    RejectedTurn,
    StageAnalysisRail,
  ],
  templateUrl: './token_diff_reading.ng.html',
  styleUrls: ['./token_diff_reading.scss', './token_diff_reading_nodes.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TokenDiffReading {
  protected readonly conversation = inject(ConversationState);
  protected readonly tokens = inject(TokenDiff);
  protected readonly compact = inject(CompactPreviews);
  protected readonly dom = inject(TokenDiffDom);
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  readonly stages = ['prefill', 'decode'] as const;

  // The reading's width changes its pairs' shared line widths and the previews' room.
  constructor() {
    afterNextRender(() => {
      const reading = this.element.nativeElement.querySelector('.reading');
      if (!reading || typeof ResizeObserver === 'undefined') {
        return;
      }
      const observer = new ResizeObserver(() => this.dom.layoutTick.update((n) => n + 1));
      observer.observe(reading);
      this.destroyRef.onDestroy(() => observer.disconnect());
      document.fonts?.ready.then(() => this.dom.layoutTick.update((n) => n + 1));
    });
  }
  /** Prefill text follows the same whitespace rule as Decode tokens. */
  inputTokenHtml(pair: TokenPair, side: string): string {
    const token = this.tokens.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token)) : '∅';
  }
  /** A token's text for labels: an invisible-only token is named ("line feed × 2"), a gap is ∅. */
  spokenToken(token: CapturedToken | undefined): string {
    return token ? spokenTokenText(capturedTokenLabel(token)) : '∅';
  }
  nodeIndex(turn: number, stage: string): number {
    return (
      this.conversation.turns().findIndex((t) => t.n === turn) * 2 + (stage === 'decode' ? 1 : 0)
    );
  }
  inputSegments(turn: number, run: string): CapturedInputSegment[] {
    return capturedInputSegments(this.conversation.captured(turn, run));
  }
  /** Admitted input as the runtime tokenized it, grouped by message by the server; null without. */
  inputRows(turn: number, run: string): InputRowsOf<CapturedToken>[] | null {
    const capture = this.conversation.captured(turn, run);
    return messageRows(capture?.input_messages, capture?.input_tokens ?? []);
  }
  roleIcon(role: string): string {
    return role === 'system'
      ? 'settings'
      : role === 'user'
        ? 'person'
        : role === 'model'
          ? 'smart_toy'
          : 'subject';
  }
  roleLabel(role: string): string {
    return role === 'system'
      ? 'System'
      : role === 'user'
        ? 'User'
        : role === 'model'
          ? 'Model'
          : 'Input';
  }
  readonly tokenPhases: TokenPhase[] = ['thinking', 'response'];
  rowsForPhase(turn: number, side: string, phase: TokenPhase): TokenPair[] {
    return this.tokens.phasePairs().get(turn)?.[side === 'ref' ? 'ref' : 'target'][phase] ?? [];
  }
  phaseText(turn: number, side: string, phase: TokenPhase): string {
    return capturedPhaseText(this.conversation.captured(turn, side), phase);
  }
  readonly isTemplateToken = isTemplateToken;
  readonly isUnknownStop = isUnknownStop;
  readonly unknownStopHint = UNKNOWN_STOP_HINT;
  /** Plain token text for labels; a missing token reads as ∅. */
  tokenText(token: CapturedToken | undefined): string {
    return token ? capturedTokenLabel(token) : '∅';
  }
  /** Debug token markup: real characters, with whitespace glyphs drawn in place when shown. */
  tokenHtml(token: CapturedToken | undefined): string {
    return tokenMarkup(this.tokenText(token), false);
  }
  /** One entry per line break that follows this row, for both runtimes alike. */
  rowBreaks(pair: TokenPair): readonly number[] {
    const count = pairLineBreaks(pair);
    return count ? Array.from({length: count}, (_, i) => i) : NO_BREAKS;
  }
}
