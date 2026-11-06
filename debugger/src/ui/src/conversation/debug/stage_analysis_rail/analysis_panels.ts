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
  type Signal,
  afterRenderEffect,
  computed,
  inject,
  signal,
  untracked,
} from '@angular/core';

import {
  type FinalLogitsInspection,
  GraphMetricPanel,
  KvPanel,
  type LogitPanelState,
  analysisOf,
} from '../../../analyses/registry';
import {reasonText} from '../../../server/reasons';
import type {RangeSelection} from '../../../widgets/charts/range_selection';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {TokenDiff, stageStart} from '../token_diff/token_diff';
import type {LogitPanel} from './logit_panel';

/** A rail's live analysis panels, as its view queries them. */
export interface AnalysisRail {
  kvPanels: Signal<readonly KvPanel[]>;
  graphPanels: Signal<readonly GraphMetricPanel[]>;
  tokenPanels: Signal<readonly LogitPanel[]>;
}

/**
 * The analysis views of every stage and where each one looks: the views a reader added, their
 * live panels in the stage rails, the expanded one, and the ranges and selections the panels,
 * the reading and the details share.
 */
@Injectable()
export class AnalysisPanels {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);

  readonly tokenMetricRange = signal<{
    panelId: string;
    index: number;
    start: number;
    end: number;
  } | null>(null);
  tokenPanelSelectedIndex(panel: StageAnalysisPanel): number | null {
    const bin = this.tokenMetricRange();
    if (this.tokens.tokenMetricInspector()?.id === panel.id && bin?.panelId === panel.id) {
      return bin.index;
    }
    const selected = this.tokens.tokenSelection();
    return selected?.turn === panel.turn ? selected.index : null;
  }
  readonly expandedAnalysis = signal<StageAnalysisPanel | null>(null);
  expandTrigger: HTMLElement | null = null;
  isExpandedStage(turn: number, stage: string): boolean {
    const p = this.expandedAnalysis();
    return p?.turn === turn && p?.stage === stage;
  }
  exitExpandedAnalysis(): void {
    this.expandedAnalysis.set(null);
    requestAnimationFrame(() => this.expandTrigger?.focus({preventScroll: true}));
  }
  readonly analysisPanels = signal<StageAnalysisPanel[]>([]);
  /** The rails on the page; each adds its live panels while it is there. */
  private readonly rails = signal<readonly AnalysisRail[]>([]);
  /** Adds a rail's live panels; returns what removes them. */
  addRail(rail: AnalysisRail): () => void {
    this.rails.update((rails) => [...rails, rail]);
    return () => this.rails.update((rails) => rails.filter((item) => item !== rail));
  }
  readonly kvPanels = computed(() => this.rails().flatMap((rail) => rail.kvPanels()));
  readonly graphPanels = computed(() => this.rails().flatMap((rail) => rail.graphPanels()));
  readonly tokenPanels = computed(() => this.rails().flatMap((rail) => rail.tokenPanels()));
  readonly pendingInspector = signal<string | null>(null);
  /** The panels, by capture and id, whose first data has arrived or failed: each opens from then
   *  on, and what it asks later loads inside it. */
  private readonly loadedPanels = signal<ReadonlySet<string>>(new Set());
  private loadedKey(panel: StageAnalysisPanel): string {
    return `${this.conversation.state.captureId()}:${panel.id}`;
  }
  /** Whether a panel still loads its first data: it cannot open, and shows nothing, meanwhile. */
  panelLoading(panel: StageAnalysisPanel): boolean {
    return !this.loadedPanels().has(this.loadedKey(panel));
  }
  /** Whether a panel waits for its first data, as its live panel says; one not on the page yet
   *  waits too. */
  panelWaiting(panel: StageAnalysisPanel): boolean {
    if (panel.kind === 'kv') {
      return this.overviewKv(panel)?.waiting() ?? true;
    }
    if (panel.kind === 'activations') {
      return this.overviewGraph(panel)?.waiting() ?? true;
    }
    return (
      this.tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.waiting() ?? true
    );
  }
  constructor() {
    // Asked once the page is drawn, when every live panel has asked for what it shows first.
    afterRenderEffect(() => {
      const loaded = untracked(this.loadedPanels);
      const ready = this.analysisPanels()
        .map((panel) => [panel, this.loadedKey(panel)] as const)
        .filter(([panel, key]) => !loaded.has(key) && !this.panelWaiting(panel))
        .map(([, key]) => key);
      if (ready.length) {
        this.loadedPanels.set(new Set([...loaded, ...ready]));
      }
    });
  }
  analysisSerial = 0;
  panelsFor(turn: number, stage: string): StageAnalysisPanel[] {
    return this.analysisPanels().filter((p) => p.turn === turn && p.stage === stage);
  }
  /** Whether an analysis can be added to a stage, as the server says: '' if it can, else the
   *  server's reasons why not. */
  analysisUnavailable(turn: number, stage: string, kind: StageAnalysisPanel['kind']): string {
    const answer = this.conversation.state.session()?.turns.find((t) => t.n === turn)?.stages?.[
      stage as 'prefill' | 'decode'
    ]?.analyses[analysisOf(kind)];
    if (!answer) {
      return 'Not captured';
    }
    return answer.available ? '' : reasonText(answer.reasons);
  }
  analysisOverviewRange(panel: StageAnalysisPanel): {start: number; end: number} {
    const kv = panel.kind === 'kv' ? this.overviewKv(panel) : undefined;
    const graph = panel.kind === 'activations' ? this.overviewGraph(panel) : undefined;
    const position = kv?.tensorView()
      ? kv.tensorPosition()
      : graph?.inspectionMode()
        ? (graph.inspection()?.position() ?? null)
        : null;
    return position !== null
      ? {start: position, end: position + 1}
      : this.analysisViewportRange(panel);
  }
  analysisViewportRange(panel: StageAnalysisPanel): {start: number; end: number} {
    const current = this.analysisPanels().find((candidate) => candidate.id === panel.id) ?? panel;
    if (current.kind === 'kv') {
      return this.overviewKv(current)?.visibleRange() ?? this.entireStageRange(current);
    }
    if (current.kind === 'activations') {
      return this.overviewGraph(current)?.range() ?? this.entireStageRange(current);
    }
    const logits = this.overviewFinalLogits(current);
    if (logits) {
      return {start: logits.position(), end: logits.position() + 1};
    }
    return (
      this.tokenPanels()
        .find((candidate) => candidate.panelId() === current.id)
        ?.finalChart()
        ?.range() ?? this.tokenPanelRange(current)
    );
  }
  readonly analysisChartSelections = computed(
    () =>
      new Map(
        this.analysisPanels().map((panel) => {
          const range = this.tokens.analysisOverview() ? null : this.selectionRangeFor(panel);
          const selection: RangeSelection | null = range
            ? {
                ...range,
                layer:
                  this.tokens.metricInspector()?.panelId() === panel.id &&
                  'layer' in range &&
                  typeof range.layer === 'number'
                    ? range.layer
                    : null,
              }
            : null;
          return [panel.id, selection];
        }),
      ),
  );
  setPanelLogitState(panel: StageAnalysisPanel, logit: LogitPanelState): void {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, logit} : p)),
    );
  }
  capturePanels(): StageAnalysisPanel[] {
    return this.analysisPanels().map((p) => {
      const live = this.kvPanels().find((kv) => kv.panelId() === p.id);
      const graph = this.graphPanels().find((g) => g.panelId() === p.id);
      return live ? {...p, kv: live.captureView()} : graph ? {...p, graph: graph.captureView()} : p;
    });
  }
  readonly analysisHover = signal<{panelId: string; range: RangeSelection} | null>(null);
  hoverAnalysis(panelId: string, range: RangeSelection | null): void {
    if (range) {
      this.analysisHover.set({panelId, range});
    } else if (this.analysisHover()?.panelId === panelId) {
      this.analysisHover.set(null);
    }
  }
  selectionRangeFor(
    panel: Pick<StageAnalysisPanel, 'turn' | 'stage'> & Partial<Pick<StageAnalysisPanel, 'kind'>>,
  ): {start: number; end: number; layer?: number | null} | null {
    const inspector = this.tokens.metricInspector();
    const owner = this.analysisPanels().find((p) => p.id === inspector?.panelId());
    if (inspector && owner?.turn === panel.turn && owner.stage === panel.stage) {
      if (inspector instanceof KvPanel && inspector.tensorView()) {
        const position = inspector.tensorPosition();
        return position === null
          ? null
          : {start: position, end: position + 1, layer: inspector.tensorLayer()};
      }
      if (inspector instanceof GraphMetricPanel && inspector.inspectionMode()) {
        const tensor = inspector.inspection();
        return tensor
          ? {start: tensor.position(), end: tensor.position() + 1, layer: tensor.layer()}
          : null;
      }
      return inspector instanceof KvPanel ? inspector.inspectedRange() : inspector.selected();
    }
    const selected = this.tokens.selection();
    if (!selected || selected.turn !== panel.turn) {
      return null;
    }
    if (selected.kind === 'stage') {
      return selected.stage === panel.stage ? (selected.range ?? null) : null;
    }
    if ((selected.kind === 'input' ? 'prefill' : 'decode') !== panel.stage) {
      return null;
    }
    const pair = this.tokens.selectedInputPair() ?? this.tokens.selectedPair();
    const token = pair?.target ?? pair?.ref;
    if (!pair || !token) {
      return null;
    }
    const start =
      (stageStart(
        this.conversation.turns().find((t) => t.n === panel.turn),
        panel.stage,
      ) ?? 0) + (this.conversation.alignment() === 'content' ? pair.index : token.step);
    return {start, end: start + 1};
  }
  readonly kvDetailSelection = computed(() => {
    const inspector = this.tokens.metricInspector();
    if (!(inspector instanceof KvPanel)) {
      return null;
    }
    const range = inspector.inspectedRange();
    return range
      ? {
          ...range,
          kind: range.layer === null ? null : (inspector.selectedCurve() ?? inspector.kind()),
        }
      : null;
  });
  overviewPanel(): StageAnalysisPanel | null {
    const inspector = this.tokens.metricInspector();
    const p = inspector
      ? this.analysisPanels().find((panel) => panel.id === inspector.panelId())
      : (this.tokens.tokenMetricInspector() ?? this.tokens.analysisOverview());
    return p ? (this.analysisPanels().find((x) => x.id === p.id) ?? p) : null;
  }
  graphDetailSelection(panel: StageAnalysisPanel): RangeSelection | null {
    const graph = this.overviewGraph(panel);
    if (graph?.inspectionMode()) {
      const tensor = graph.inspection();
      return tensor
        ? {start: tensor.position(), end: tensor.position() + 1, layer: tensor.layer()}
        : null;
    }
    const graphPos = graph?.graphPosition() ?? null;
    if (graph?.view() === 'graph' && graphPos !== null) {
      return {start: graphPos, end: graphPos + 1, layer: null};
    }
    return graph && this.tokens.metricInspector() === graph ? graph.selected() : null;
  }
  /** The row a panel's selection stands for: the one its chart picked for the selected range, else
   *  the selected token of its turn; null with neither. */
  inspectedRow(panel: StageAnalysisPanel): number | null {
    const picked = this.tokenMetricRange();
    if (picked?.panelId === panel.id) {
      return picked.index;
    }
    const token = this.tokens.tokenSelection();
    return token?.turn === panel.turn ? token.index : null;
  }
  analysisRange(panel: StageAnalysisPanel): {start: number; end: number} {
    if (panel.kind === 'kv') {
      return (
        this.kvDetailSelection() ??
        this.overviewKv(panel)?.visibleRange() ??
        this.entireStageRange(panel)
      );
    }
    const graph = this.overviewGraph(panel);
    if (graph) {
      return this.graphDetailSelection(panel) ?? graph.range() ?? this.entireStageRange(panel);
    }
    const logits = this.overviewFinalLogits(panel);
    if (logits) {
      return {start: logits.position(), end: logits.position() + 1};
    }
    if (this.tokens.tokenMetricInspector()?.id === panel.id) {
      const selected = this.tokens.tokenSelection();
      const bin = this.tokenMetricRange();
      if (bin?.panelId === panel.id) {
        return bin;
      }
      const restored = this.tokens.selection();
      if (
        restored?.kind === 'stage' &&
        restored.turn === panel.turn &&
        restored.stage === panel.stage &&
        restored.range
      ) {
        return restored.range;
      }
      const pair = this.tokens.rows(panel.turn).find((row) => row.index === selected?.index);
      if (pair) {
        const token =
          panel.kind === 'tokens' ? (pair.ref ?? pair.target) : (pair.target ?? pair.ref);
        const start =
          this.tokens.stageGlobalStart(panel.turn) +
          (this.conversation.alignment() === 'content' ? pair.index : (token?.step ?? pair.index));
        return {start, end: start + 1};
      }
    }
    return this.tokenPanelRange(panel);
  }
  tokenPanelRange(panel: StageAnalysisPanel): {start: number; end: number} {
    const positions =
      this.conversation.alignment() === 'content'
        ? this.tokens
            .rows(panel.turn)
            .map((row) => this.tokens.stageGlobalStart(panel.turn) + row.index)
        : this.tokens
            .rows(panel.turn)
            .flatMap((row) =>
              panel.kind === 'tokens' ? [row.ref ?? row.target] : [row.ref, row.target],
            )
            .filter((token): token is NonNullable<typeof token> => !!token)
            .map((token) => this.tokens.stageGlobalStart(panel.turn) + token.step);
    const all = positions.length
      ? {
          start: positions.reduce((a, b) => Math.min(a, b)),
          end: positions.reduce((a, b) => Math.max(a, b)) + 1,
        }
      : this.entireStageRange(panel);
    // Only the columns the server gives the analysis, as its panel shows them.
    const columns = this.tokens.analysisColumns(
      panel.turn,
      panel.stage,
      panel.kind === 'distribution' ? 'sampling' : 'final_logits',
    );
    const lo = Math.max(all.start, columns?.start ?? all.start);
    const hi = Math.min(all.end, columns?.end ?? all.end);
    const full = lo < hi ? {start: lo, end: hi} : all;
    const saved = panel.logit?.positionRange;
    if (!saved) {
      return full;
    }
    const start = Math.max(full.start, saved.start);
    const end = Math.min(full.end, saved.end);
    return start < end ? {start, end} : full;
  }
  overviewFinalLogits(panel: StageAnalysisPanel): FinalLogitsInspection | undefined {
    return this.tokenPanels()
      .find((p) => p.panelId() === panel.id)
      ?.inspection();
  }
  overviewGraph(panel: StageAnalysisPanel): GraphMetricPanel | undefined {
    return this.graphPanels().find((p) => p.panelId() === panel.id);
  }
  overviewKv(panel: StageAnalysisPanel): KvPanel | undefined {
    return this.kvPanels().find((p) => p.panelId() === panel.id);
  }
  private readonly entireStageRanges = computed(() => {
    const ranges = new Map<string, {start: number; end: number}>();
    for (const turn of this.conversation.turns()) {
      for (const stage of ['prefill', 'decode'] as const) {
        const start = stageStart(turn, stage) ?? 0;
        const count = Math.max(
          0,
          ...['ref', 'target']
            .map((side) => this.tokens.stageCount(turn.n, side, stage))
            .filter((n): n is number => typeof n === 'number'),
        );
        ranges.set(`${turn.n}:${stage}`, {start, end: start + count});
      }
    }
    return ranges;
  });
  private readonly emptyStageRange = {start: 0, end: 0};
  entireStageRange(panel: StageAnalysisPanel): {start: number; end: number} {
    return this.entireStageRanges().get(`${panel.turn}:${panel.stage}`) ?? this.emptyStageRange;
  }
}
