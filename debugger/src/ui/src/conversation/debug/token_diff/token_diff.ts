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
  afterNextRender,
  computed,
  inject,
  signal,
  untracked,
  type EffectCleanupRegisterFn,
} from '@angular/core';

import type {GraphMetricPanel, KvPanel} from '../../../analyses/registry';
import type {
  AnalysisColumns,
  CapturedToken,
  CapturedTurn,
  StageAnalysisId,
  TurnStage,
} from '../../../server/contracts/capture';
import type {KvPositionAlignment} from '../../../server/contracts/kv';
import type {TokenMetricKey} from '../../../server/contracts/token_analysis';
import {MetricCatalogStore} from '../../../server/metric_catalog';
import {TokenAnalysisApi} from '../../../server/token_analysis_api';
import {splitPhasePairs, type ConversationSelection} from '../../conversation_content';
import {ConversationState} from '../../conversation_state';
import {PANEL_WIDTH} from '../../conversation_view_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {
  comparisonPairs,
  forkPair,
  messageRows,
  type InputRowsOf,
  type TokenPair,
} from '../../token_alignment';
import {
  TokenColor,
  colorLabel,
  heatLevel,
  heatPalette,
  pairKey,
  queryRecord,
  tokenMetrics,
} from '../../token_analysis';
import {compileQueryOrNothing} from '../../token_query';
import {AnalysisRequest, TokenAnalysisController} from './token_analysis_controller';

/** Where a stage's columns start, as the server gives it: the Reference side's first position of
 *  the stage, where its rows' columns (`positions.start`) start too; null for a turn the index did
 *  not read. */
export function stageStart(turn: CapturedTurn | undefined, stage: string): number | null {
  return turn?.stages?.[stage as 'prefill' | 'decode'].positions.ref?.start ?? null;
}

/**
 * The Token Diff of the Debug reading: its display settings, its rows in the current alignment
 * and the metrics of their tokens, the find formula and its results, and what the details
 * describe, which is either a selection in the reading or an analysis view of the stage rail (its
 * overview, its chart's inspection or its token metric).
 */
@Injectable()
export class TokenDiff {
  private readonly conversation = inject(ConversationState);
  private readonly metricCatalog = inject(MetricCatalogStore);

  // The analysis view the details describe instead of the selection, if any: its overview, its
  // token metric, or (metricInspector) the live panel whose chart point is inspected.
  readonly analysisOverview = signal<StageAnalysisPanel | null>(null);
  readonly tokenMetricInspector = signal<StageAnalysisPanel | null>(null);
  stageGlobalStart(turn: number): number {
    return (
      stageStart(
        this.conversation.turns().find((t) => t.n === turn),
        'decode',
      ) ?? 0
    );
  }
  /** The server's stage of a turn: what it offers, its KV snapshot and each side's positions. */
  turnStage(turn: number, stage: string): TurnStage | null {
    return (
      this.conversation.turns().find((t) => t.n === turn)?.stages?.[
        stage as 'prefill' | 'decode'
      ] ?? null
    );
  }
  /** Which of a stage's columns the server gives an analysis, under the reading's alignment (a
   *  Prefill's is by content); null when the server says nothing of them. */
  analysisColumns(
    turn: number,
    stage: 'prefill' | 'decode',
    analysis: StageAnalysisId,
  ): AnalysisColumns | null {
    const comparison = this.conversation.comparison(turn);
    const positions =
      stage === 'prefill'
        ? comparison?.prefill.positions
        : comparison?.alignments[this.conversation.alignment()].positions;
    return positions?.columns?.[analysis] ?? null;
  }
  /** What the server says an analysis of a stage leaves out, what its values cannot show, and
   *  why its columns after the divergence still compare. */
  analysisNotes(turn: number, stage: 'prefill' | 'decode', analysis: StageAnalysisId): string[] {
    const columns = this.analysisColumns(turn, stage, analysis);
    return columns
      ? [
          ...columns.omitted,
          ...columns.notes,
          ...(columns.context ? [columns.context.reason] : []),
        ].map((reason) => reason.text)
      : [];
  }
  readonly metricInspector = signal<KvPanel | GraphMetricPanel | null>(null);
  private readonly analysis = new TokenAnalysisController(inject(TokenAnalysisApi));
  readonly tokenAnalysis = this.analysis.results;
  readonly analysisLoading = this.analysis.loading;
  readonly analysisError = this.analysis.error;
  readonly analysisRetry = signal(0);
  readonly numericFindPending = computed(
    () => this.predicate().requiresMetrics && (this.analysisLoading() || !!this.analysisError()),
  );
  readonly findResultLabel = computed(() =>
    this.numericFindPending()
      ? this.analysisError()
        ? 'Find results unavailable; retry token metrics'
        : 'Find results loading'
      : 'Find results: ' + this.mismatches().length + ' tokens',
  );
  /** Prefill rows over the server's aligned input pairs, grouped by its messages; null without. */
  inputPairRows(turn: number): InputRowsOf<TokenPair>[] | null {
    const prefill = this.conversation.comparison(turn)?.prefill;
    return messageRows(
      prefill?.messages,
      comparisonPairs(
        prefill?.rows,
        this.conversation.captured(turn, 'ref')?.input_tokens,
        this.conversation.captured(turn, 'target')?.input_tokens,
      ),
    );
  }
  pairToken(pair: TokenPair, side: string): CapturedToken | undefined {
    return side === 'ref' ? pair.ref : pair.target;
  }
  readonly selectionSide = signal('target');
  /** Each stage's summary in the current alignment, as the server gives it. */
  readonly stageTokenSummaries = computed(() =>
    [...this.conversation.turns()]
      .sort((a, b) => a.n - b.n)
      .flatMap((turn) =>
        (['prefill', 'decode'] as const).map((stage) => ({
          turn: turn.n,
          stage,
          history: turn.comparison?.history[stage] ?? null,
          ...(turn.comparison?.alignments[this.conversation.alignment()].stages[stage] ?? {
            match_rate: null,
            cumulative_match_rate: null,
            first_mismatch: null,
          }),
        })),
      ),
  );
  readonly firstMismatchStage = computed(() => {
    const first = this.stageTokenSummaries().find((summary) => summary.first_mismatch);
    return first ? {turn: first.turn, stage: first.stage} : null;
  });
  isFirstMismatchStage(turn: number, stage: string): boolean {
    const first = this.firstMismatchStage();
    return first?.turn === turn && first.stage === stage;
  }
  readonly formula = signal('NOT token_match');
  /** Finds nothing, rather than throwing, while the formula needs a catalog not loaded yet. */
  readonly predicate = computed(() =>
    compileQueryOrNothing(this.formula(), this.metricCatalog.lookup()),
  );

  readonly selection = signal<ConversationSelection | null>(null);
  readonly tokenSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'token' ? selected : null;
  });
  readonly selectedStage = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'stage' ? selected : null;
  });
  readonly inputSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'input' ? selected : null;
  });
  /** The selected admitted Prefill pair, found across its message rows. */
  readonly selectedInputPair = computed(() => {
    const s = this.inputSelection();
    if (!s) {
      return null;
    }
    for (const row of this.inputPairRows(s.turn) ?? []) {
      for (const pair of [...row.leading, ...row.text, ...row.trailing]) {
        if (pair.index === s.index) {
          return {...pair, turn: s.turn};
        }
      }
    }
    return null;
  });
  isInputSelected(turn: number, index: number): boolean {
    const s = this.inputSelection();
    return !!s && s.turn === turn && s.index === index;
  }
  selectInput(turn: number, index: number, event?: Event): void {
    if (this.isInputSelected(turn, index)) {
      this.closeDetails();
      return;
    }
    const trigger = event?.currentTarget ?? document.activeElement;
    if (trigger instanceof HTMLElement) {
      this.returnFocus = trigger;
      this.selectionSide.set(trigger.dataset['side'] ?? 'target');
    }
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set({kind: 'input', turn, index});
    this.details.set(true);
  }
  readonly details = signal(true);
  readonly boundaries = signal(true);
  readonly color = signal<TokenColor>('token_match');
  colorLabel(color: TokenColor): string {
    return colorLabel(this.metricCatalog.lookup(), color);
  }
  heatPalette(color: TokenColor): 'logits' | 'activation' {
    return heatPalette(this.metricCatalog.lookup(), color);
  }
  readonly metricAvailability = computed(() =>
    Object.fromEntries(
      tokenMetrics(this.metricCatalog.lookup()).map((m) => [
        m.id,
        [...this.tokenAnalysis().values()].some(
          (p) => p.metrics[m.id as TokenMetricKey]?.value != null,
        ),
      ]),
    ),
  );
  readonly numericColor = computed(() => this.color() !== 'none' && this.color() !== 'token_match');
  /** The colored metric as the catalog describes it; its heat is full at `heat_full`. */
  private readonly colorDescription = computed(() =>
    this.numericColor() ? this.metricCatalog.lookup().metric(this.color()) : undefined,
  );
  /** What the legend writes at the ends of the heat scale, as the catalog words them. */
  readonly colorLegend = computed(() => this.colorDescription()?.color.heat_labels ?? null);
  tokenHeat(turn: number, pair: TokenPair | undefined | null, side: string): string | null {
    return side === 'target' && pair && this.numericColor()
      ? heatLevel(
          this.metricCatalog.lookup(),
          this.color(),
          this.tokenAnalysis().get(pairKey(turn, pair)),
        )
      : null;
  }
  readonly collapsed = signal(new Set<string>());
  readonly groups = signal<Record<string, boolean>>({});
  groupOpen(key: string): boolean {
    return this.groups()[key] ?? true;
  }
  setGroup(key: string, value: boolean): void {
    this.groups.update((groups) => ({...groups, [key]: value}));
  }
  readonly infoExpanded = signal(true);
  readonly panelBounds = PANEL_WIDTH;
  readonly panelWidth = signal(350);
  returnFocus: HTMLElement | null = null;
  /** Each turn's rows: the server's alignment in the current mode. */
  readonly aligned = computed(() =>
    this.conversation.turns().map((turn) => {
      if (this.conversation.virtualOutput(turn.n) && this.conversation.workspace.mode() === 'Debug') {
        return {turn: turn.n, rows: this.conversation.virtualModels().get(turn.n)?.pairs ?? []};
      }
      return {
        turn: turn.n,
        rows: comparisonPairs(
          turn.comparison?.alignments[this.conversation.alignment()].rows,
          this.conversation.captured(turn.n, 'ref')?.tokens,
          this.conversation.captured(turn.n, 'target')?.tokens,
        ),
      };
    }),
  );
  /** Each stage's rows are the KV and activation columns, in either alignment: the server's
   *  positions of them, and all a request names, the stage and the alignment. A Prefill is aligned
   *  by content alone; a Decode by the current alignment. */
  readonly kvAlignments = computed(() => {
    const result = new Map<string, KvPositionAlignment>();
    const alignment = this.conversation.alignment();
    for (const turn of this.conversation.turns()) {
      const comparison = turn.comparison;
      if (!comparison) {
        continue;
      }
      result.set(turn.n + ':prefill', {
        stage: 'prefill',
        alignment: 'content',
        ...comparison.prefill.positions,
      });
      result.set(turn.n + ':decode', {
        stage: 'decode',
        alignment,
        ...comparison.alignments[alignment].positions,
      });
    }
    return result;
  });
  rows(turn: number): TokenPair[] {
    return this.aligned().find((t) => t.turn === turn)?.rows ?? [];
  }
  readonly phasePairs = computed(
    () =>
      new Map(
        this.aligned()
          .filter((t) => !this.conversation.virtualOutput(t.turn))
          .map((t) => [t.turn, splitPhasePairs(t.rows)]),
      ),
  );
  readonly selectedPair = computed(() => {
    const s = this.tokenSelection();
    const row = s ? this.rows(s.turn)[s.index] : null;
    return s && row ? {...row, turn: s.turn} : null;
  });
  /** Each turn's first differing generation step, as the server pairs it by step. */
  readonly forks = computed(
    () =>
      new Map(
        this.aligned().map((t) => [
          t.turn,
          forkPair(
            this.conversation.comparison(t.turn)?.alignments[this.conversation.alignment()]
              .first_divergence,
            t.rows,
            this.conversation.captured(t.turn, 'ref')?.tokens,
            this.conversation.captured(t.turn, 'target')?.tokens,
          ),
        ]),
      ),
  );
  /** The token drawn elsewhere that the details pair with the selected first differing row. */
  isStepPartner(turn: number, index: number | undefined, side: string): boolean {
    const fork = this.forks().get(turn);
    const selected = this.tokenSelection();
    return (
      !!fork?.filled &&
      fork.filled === side &&
      fork.partner === index &&
      this.selection()?.turn === turn &&
      selected?.index === fork.index
    );
  }
  /** The first turn whose two runtimes sampled different released tokens at one step: the one
   *  pair that both differs and can share its context, so every metric may be defined for it. */
  readonly previewStepPair = computed(() => {
    for (const [turn, fork] of this.forks()) {
      if (fork && fork.pair.ref?.released !== false && fork.pair.target?.released !== false) {
        return {turn, pair: fork.pair};
      }
    }
    return null;
  });
  // The preview is a fixed pair: the first sampled difference, whose metrics exist; identical
  // runs fall back to the first short differing pair, then the selection or the first row.
  readonly displayPreview = computed(() => {
    const fixed = this.previewStepPair();
    if (fixed) {
      return {
        ref: fixed.pair.ref?.text ?? 'Not captured',
        target: fixed.pair.target?.text ?? 'Not captured',
        match: false,
        analysis: this.tokenAnalysis().get(pairKey(fixed.turn, fixed.pair)),
      };
    }
    const short = (token: CapturedToken | undefined) => /^\s*\S{1,10}$/.test(token?.text ?? '');
    let changed: {turn: number; pair: TokenPair} | undefined;
    for (const turn of this.aligned()) {
      const row = turn.rows.find((r) => r.match === false && short(r.ref) && short(r.target));
      if (row) {
        changed = {turn: turn.turn, pair: row};
        break;
      }
    }
    const turn =
      changed?.turn ??
      this.selection()?.turn ??
      this.conversation.state.batch()?.turn ??
      this.conversation.turns()[0]?.n;
    const rows = this.rows(turn);
    const pair =
      changed?.pair ?? this.selectedPair() ?? rows.find((r) => r.match === false) ?? rows[0];
    return pair
      ? {
          ref: pair.ref?.text ?? 'Not captured',
          target: pair.target?.text ?? 'Not captured',
          match: pair.match,
          analysis: this.tokenAnalysis().get(pairKey(turn, pair)),
        }
      : null;
  });
  // The demo's first-divergence glyph is hidden; preserve its accessible identity.
  readonly firstDivergence = computed(() => {
    for (const turn of this.conversation.turns()) {
      const index = turn.comparison?.alignments[this.conversation.alignment()].first_mismatch;
      if (index != null) {
        return {turn: turn.n, index};
      }
    }
    return null;
  });
  isFirstDivergence(turn: number, index: number | undefined, side: string): boolean {
    const first = this.firstDivergence();
    return side === 'target' && first?.turn === turn && first.index === index;
  }
  readonly mismatches = computed(() => {
    const predicate = this.predicate();
    const records = predicate.requiresMetrics ? this.tokenAnalysis() : null;
    return this.aligned().flatMap((turn) =>
      turn.rows
        .filter(
          (row) =>
            predicate(
              records
                ? queryRecord(row, records.get(pairKey(turn.turn, row)))
                : {token_match: row.match},
            ) === true,
        )
        .map((row) => ({turn: turn.turn, index: row.index})),
    );
  });
  readonly resultIndex = computed(() =>
    this.mismatches().findIndex(
      (r) => r.turn === this.selection()?.turn && r.index === this.tokenSelection()?.index,
    ),
  );
  compactPairs(turn: number, stage: string): TokenPair[] {
    return stage === 'prefill'
      ? (this.inputPairRows(turn) ?? []).flatMap((r) => [...r.leading, ...r.text, ...r.trailing])
      : this.rows(turn);
  }
  nodeCollapsed(turn: number, run: string, phase: string): boolean {
    return this.collapsed().has(`${turn}:${run}:${phase}`);
  }
  stageSelected(turn: number, stage: string): boolean {
    const selected = this.selectedStage();
    return selected?.turn === turn && selected.stage === stage;
  }
  stageCount(turn: number, run: string, stage: string): number | string {
    const capture = this.conversation.captured(turn, run);
    return (
      (stage === 'prefill' ? capture?.input_token_count : capture?.tokens?.length) ?? 'Not captured'
    );
  }
  selectStagePointer(
    event: MouseEvent,
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
  ): void {
    if (
      (event.target as Element).closest(
        'button,summary,a,input,select,[data-v-gap],[role="button"]',
      ) ||
      window.getSelection()?.toString()
    ) {
      return;
    }
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  selectStageKey(
    event: KeyboardEvent,
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
  ): void {
    if (event.target !== event.currentTarget || !['Enter', ' '].includes(event.key)) {
      return;
    }
    event.preventDefault();
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  selectStage(
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
    trigger: HTMLElement | null,
  ): void {
    if (this.conversation.workspace.mode() !== 'Debug') {
      return;
    }
    const anchor = this.conversation.virtualizer?.capture();
    this.returnFocus = trigger;
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set({kind: 'stage', turn, side, stage});
    this.details.set(true);
    this.conversation.state.selectTurn(turn);
    afterNextRender(
      () => {
        this.conversation.virtualizer?.mount();
        if (anchor) {
          this.conversation.virtualizer?.restore(anchor);
        }
      },
      {injector: this.conversation.injector},
    );
  }
  select(turn: number, index: number, event?: Event): void {
    const current = this.selection();
    if (current?.kind === 'token' && current.turn === turn && current.index === index) {
      this.closeDetails();
      return;
    }
    const trigger = event?.currentTarget ?? document.activeElement;
    if (trigger instanceof HTMLElement) {
      this.returnFocus = trigger;
      this.selectionSide.set(trigger.dataset['side'] ?? 'target');
    }
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set({kind: 'token', turn, index});
    this.details.set(true);
    this.conversation.state.selectTurn(turn);
  }
  closeDetails(): void {
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set(null);
    this.returnFocus?.focus({preventScroll: true});
  }
  zeroWidthText(text: string): boolean {
    return /^[\r\n\u200b-\u200d\ufeff]*$/.test(text);
  }
  /** Loads the token metrics of the rows in view, and of each turn's first divergence where the
   *  server fills it in. Run in an effect. */
  loadTokenAnalysis(onCleanup: EffectCleanupRegisterFn): void {
    this.analysisRetry();
    const captureId = this.conversation.state.captureId();
    const aligned = this.aligned();
    const debug = this.conversation.workspace.mode() === 'Debug';
    const controller = new AbortController();
    onCleanup(() => controller.abort());
    // Skip uncaptured links entirely, including synthetic 128k text-only sessions.
    const linked = (p: TokenPair) =>
      p.ref?.source_forward_id !== undefined ||
      p.target?.source_forward_id !== undefined ||
      p.ref?.batch !== undefined ||
      p.target?.batch !== undefined;
    const alignment = this.conversation.alignment();
    const forks = this.forks();
    // The server's rows by index, and the first divergence where it fills a row's empty side.
    const pending: AnalysisRequest[] = debug
      ? aligned
          .map((t) => {
            const fork = forks.get(t.turn);
            return {
              turn: t.turn,
              alignment,
              rows: t.rows.filter(linked),
              firstDivergence: fork?.filled && linked(fork.pair) ? fork.pair : null,
            };
          })
          .filter((t) => t.rows.length || t.firstDivergence)
      : [];
    // Then the other alignment's rows, so that switching to it finds them read.
    const other = alignment === 'content' ? 'steps' : 'content';
    if (debug) {
      for (const turn of this.conversation.turns()) {
        if (!turn.comparison || this.conversation.virtualOutput(turn.n)) {
          continue;
        }
        const rows = comparisonPairs(
          turn.comparison.alignments[other].rows,
          this.conversation.captured(turn.n, 'ref')?.tokens,
          this.conversation.captured(turn.n, 'target')?.tokens,
        ).filter(linked);
        if (rows.length) {
          pending.push({turn: turn.n, alignment: other, rows, firstDivergence: null});
        }
      }
    }
    const preview = debug ? this.previewStepPair() : null;
    if (preview) {
      const filled = !!forks.get(preview.turn)?.filled;
      pending.push({
        turn: preview.turn,
        alignment,
        rows: filled ? [] : [preview.pair],
        firstDivergence: filled ? preview.pair : null,
      });
    }
    untracked(() => this.analysis.load(captureId, pending, controller.signal));
  }
}
