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
  Injectable,
  computed,
  inject,
  linkedSignal,
  type EffectCleanupRegisterFn,
} from '@angular/core';
import type {StageSpan} from '../../../server/contracts/capture';
import type {MatchRateRequest, MatchRateResponse} from '../../../server/contracts/match_rate';
import {LatestRequest} from '../../../server/latest_request';
import {MatchRateApi} from '../../../server/match_rate_api';
import {formatCount, formatPercent} from '../../../widgets/format/format';
import {capturedTokenLabel, tokenMarkup} from '../../conversation_content';
import {ConversationState} from '../../conversation_state';
import type {TokenPair} from '../../token_alignment';
import {TokenDiff, stageStart} from '../token_diff/token_diff';

function sameSpan(a: StageSpan, b: StageSpan): boolean {
  return a.start === b.start && a.end === b.end;
}
function spanText(span: StageSpan): string {
  return span.end - span.start === 1 ? String(span.start) : `${span.start}–${span.end - 1}`;
}

/**
 * What the details panel shows while no analysis view has the focus: Conversation Info with each
 * stage's comparison, or the selected stage, range or token with its token pairs, counts and the
 * server's match rate of them.
 */
@Injectable()
export class DetailsInfo {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly matchRateApi = inject(MatchRateApi);
  readonly matchRate = new LatestRequest<MatchRateResponse>();
  readonly tokenInfoPageSize = 10;
  readonly tokenInfoPageIndex = linkedSignal({
    source: () => JSON.stringify([this.selectionInfo(), this.conversation.alignment()]),
    computation: () => 0,
  });
  readonly conversationStages = computed(() =>
    this.tokens.stageTokenSummaries().map((summary) => ({
      turn: summary.turn,
      stage: summary.stage,
      title: `Turn ${summary.turn} · ${summary.stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      properties: [
        {
          label: 'Token Range',
          value: this.stagePositions(summary.turn, summary.stage, true),
          description: 'Full stage range in the current global alignment.',
        },
        {
          label: '#Tokens (Ref / Target)',
          value: ['ref', 'target']
            .map((side) => {
              const capture = this.conversation.captured(summary.turn, side);
              const tokens = summary.stage === 'prefill' ? capture?.input_tokens : capture?.tokens;
              const count =
                tokens?.length ?? this.tokens.stageCount(summary.turn, side, summary.stage);
              return typeof count === 'number' ? formatCount(count) : count;
            })
            .join(' / '),
          description:
            'Actual tokens in the full stage, including template and special tokens; alignment gaps are excluded.',
        },
        {
          label: 'History Match',
          value: this.historyLabel(summary.history),
          description:
            'Whether both runtimes fed the same tokens before this stage. Excludes this stage; - means no history or tokens the runner did not name.',
        },
        {
          label: 'Token Match Rate',
          value: formatPercent(summary.match_rate, {placeholder: '-'}),
          description:
            'Matched aligned positions / all aligned positions in this stage. Includes template and special tokens; one-sided gaps are mismatches.',
        },
        {
          label: 'Cumulative Match Rate',
          value: formatPercent(summary.cumulative_match_rate, {placeholder: '-'}),
          description:
            'Matched aligned positions / all aligned positions from the first stage through this stage, weighted by position count. Incomplete data is shown as -.',
        },
        {
          label: 'First Divergence',
          value: summary.first_mismatch === null ? '-' : summary.first_mismatch ? 'Yes' : 'No',
          description:
            'Whether this stage contains the first observed token mismatch in the conversation.',
        },
      ],
    })),
  );
  readonly infoTokenPairs = computed(() => {
    const selection = this.selectionInfo();
    return selection ? this.stageInfoPairs(selection.turn, selection.stage) : [];
  });
  readonly infoTokenIndexDigits = computed(() =>
    this.infoTokenPairs().reduce(
      (digits, pair) => Math.max(digits, String(this.infoPairPosition(pair)).length),
      1,
    ),
  );
  readonly infoTokenPage = computed(() => {
    const pairs = this.infoTokenPairs();
    const pages = Math.max(1, Math.ceil(pairs.length / this.tokenInfoPageSize));
    const index = Math.min(Math.max(0, this.tokenInfoPageIndex()), pages - 1);
    const start = index * this.tokenInfoPageSize;
    return {index, pages, start, pairs: pairs.slice(start, start + this.tokenInfoPageSize)};
  });
  readonly stageAnalysisProperties = computed(() => {
    const selected = this.selectionInfo();
    if (!selected) {
      return [];
    }
    const {turn, stage} = selected;
    const countLabel = (side: string) => {
      const count = this.stageInfoCount(turn, side, stage);
      return typeof count === 'number' ? formatCount(count) : count;
    };
    return [
      {label: 'Token Range', value: this.stagePositions(turn, stage)},
      {label: 'Match Rate', value: this.stageTokenMatchRate(turn, stage)},
      {label: '#Tokens (Ref / Target)', value: `${countLabel('ref')} / ${countLabel('target')}`},
    ];
  });
  private readonly matchRateQuery = computed(() => this.matchRateRequest(), {
    equal: (a, b) => JSON.stringify(a) === JSON.stringify(b),
  });

  infoTokenHtml(pair: TokenPair, side: string): string {
    const token = this.tokens.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token), false) : '∅';
  }
  infoPairText(pair: TokenPair): string {
    return ['target', 'ref']
      .map((side) => {
        const token = this.tokens.pairToken(pair, side);
        return token ? capturedTokenLabel(token) : '∅';
      })
      .join(' / ');
  }
  infoTokenTitle(pair: TokenPair, side: string): string {
    const token = this.tokens.pairToken(pair, side);
    return token
      ? JSON.stringify(capturedTokenLabel(token)) + (token.id != null ? ` · ID ${token.id}` : '')
      : 'No token at this aligned position';
  }
  /** Sidebar context is the same for a directly selected pair and a Stage range. */
  selectionInfo(): {
    turn: number;
    stage: 'prefill' | 'decode';
    index?: number;
    range?: {start: number; end: number};
  } | null {
    const selected = this.tokens.selection();
    if (!selected) {
      return null;
    }
    if (selected.kind === 'stage') {
      return {...selected, index: undefined};
    }
    const stage = selected.kind === 'input' ? 'prefill' : 'decode';
    const pair = this.tokens
      .compactPairs(selected.turn, stage)
      .find((p) => p.index === selected.index);
    const start = stageStart(
      this.conversation.turns().find((t) => t.n === selected.turn),
      stage,
    );
    const token = pair?.target ?? pair?.ref;
    const position =
      start !== null && token
        ? start + (this.conversation.alignment() === 'content' ? selected.index : token.step)
        : null;
    return {
      turn: selected.turn,
      stage,
      index: selected.index,
      range: position === null ? undefined : {start: position, end: position + 1},
    };
  }
  changeTokenInfoPage(delta: number): void {
    const {index, pages} = this.infoTokenPage();
    this.tokenInfoPageIndex.set(Math.min(Math.max(0, index + delta), pages - 1));
  }
  infoPairPosition(pair: TokenPair): string {
    const selection = this.selectionInfo();
    const start = selection
      ? stageStart(
          this.conversation.turns().find((t) => t.n === selection.turn),
          selection.stage,
        )
      : null;
    if (start === null) {
      return '-';
    }
    return String(
      start +
        (this.conversation.alignment() === 'content'
          ? pair.index
          : ((pair.target ?? pair.ref)?.step ?? pair.index)),
    );
  }
  historyLabel(history: string | null): string {
    return history === 'same' ? 'Yes' : history === 'different' ? 'No' : '-';
  }
  stagePositions(turn: number, stage: string, fullStage = false): string {
    const selected = fullStage ? null : this.selectionInfo();
    if (selected?.turn === turn && selected.stage === stage && selected.range) {
      return selected.range.end - selected.range.start === 1
        ? String(selected.range.start)
        : selected.range.start + '–' + (selected.range.end - 1);
    }
    const info = this.conversation.turns().find((t) => t.n === turn);
    const start = stageStart(info, stage);
    if (start === null) {
      return 'Not captured';
    }
    // Where the two sides' stages sit apart, each side's own positions, as the server gives them.
    const sides = info?.stages?.[stage as 'prefill' | 'decode'].positions;
    if (sides?.ref && sides.target && !sameSpan(sides.ref, sides.target)) {
      return `Reference: ${spanText(sides.ref)} · Target: ${spanText(sides.target)}`;
    }
    const pairs = this.tokens.compactPairs(turn, stage);
    if (pairs.length) {
      const last = Math.max(
        ...pairs.map((pair) =>
          this.conversation.alignment() === 'content'
            ? pair.index
            : ((pair.ref ?? pair.target)?.step ?? pair.index),
        ),
      );
      return last === 0 ? String(start) : start + '–' + (start + last);
    }
    const counts = ['ref', 'target'].map((side) => this.tokens.stageCount(turn, side, stage));
    const count = Math.max(0, ...counts.filter((n): n is number => typeof n === 'number'));
    return count === 1 ? String(start) : count ? start + '–' + (start + count - 1) : '-';
  }

  /** Use the same aligned positions as the charts; gaps are not captured tokens. */
  private stageInfoPairs(turn: number, stage: string): TokenPair[] {
    const pairs = this.tokens.compactPairs(turn, stage);
    const selected = this.selectionInfo();
    if (selected?.turn === turn && selected.stage === stage && selected.index !== undefined) {
      return pairs.filter((pair) => pair.index === selected.index);
    }
    const range = selected?.range;
    if (selected?.turn !== turn || selected.stage !== stage || !range) {
      return pairs;
    }
    const start = stageStart(
      this.conversation.turns().find((t) => t.n === turn),
      stage,
    );
    if (start === null) {
      return [];
    }
    return pairs.filter((pair) => {
      const position =
        start +
        (this.conversation.alignment() === 'content'
          ? pair.index
          : ((pair.ref ?? pair.target)?.step ?? pair.index));
      return position >= range.start && position < range.end;
    });
  }
  stageInfoCount(turn: number, run: string, stage: string): number | string {
    const capture = this.conversation.captured(turn, run);
    const tokens = stage === 'prefill' ? capture?.input_tokens : capture?.tokens;
    const selected = this.selectionInfo();
    if (
      selected?.turn === turn &&
      selected.stage === stage &&
      (selected.range || selected.index !== undefined)
    ) {
      if (
        !tokens ||
        stageStart(
          this.conversation.turns().find((t) => t.n === turn),
          stage,
        ) === null
      ) {
        return 'Not captured';
      }
      return this.stageInfoPairs(turn, stage).filter((pair) => this.tokens.pairToken(pair, run))
        .length;
    }
    return tokens?.length ?? this.tokens.stageCount(turn, run, stage);
  }
  /**
   * The Match Rate the sidebar asks the server for: the selected stage's rows (a Prefill's are
   * by content), all of them or those the selection names, by index.
   */
  matchRateRequest(): MatchRateRequest | null {
    const selected = this.selectionInfo();
    if (!selected) {
      return null;
    }
    const {turn, stage} = selected;
    const address = {
      turn,
      stage,
      alignment: stage === 'prefill' ? 'content' : this.conversation.alignment(),
    } as const;
    if (selected.index === undefined && !selected.range) {
      return address;
    }
    return {...address, rows: this.stageInfoPairs(turn, stage).map((pair) => pair.index)};
  }
  /** Asks the server for the match rate of the selection. Run in an effect. */
  loadMatchRate(onCleanup: EffectCleanupRegisterFn): void {
    const capture = this.conversation.state.captureId();
    const query = this.matchRateQuery();
    onCleanup(
      this.matchRate.load(
        capture && query ? (signal) => this.matchRateApi.matchRate(capture, query, signal) : null,
      ),
    );
  }
  /**
   * The server's match rate of the selected rows; a dash until it answers, or when it cannot
   * tell.
   */
  stageTokenMatchRate(turn: number, stage: string): string {
    const query = this.matchRateQuery();
    const data = this.matchRate.data();
    if (
      !query ||
      !data ||
      query.turn !== turn ||
      query.stage !== stage ||
      data.turn !== turn ||
      data.stage !== stage ||
      data.alignment !== query.alignment ||
      data.match_rate === null
    ) {
      return '-';
    }
    return formatPercent(data.match_rate);
  }
}
