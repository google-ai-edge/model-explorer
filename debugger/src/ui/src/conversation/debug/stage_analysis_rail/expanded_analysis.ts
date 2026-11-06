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

import {representativePosition} from '../../../widgets/charts/range_bins';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {TokenDiff} from '../token_diff/token_diff';
import {AnalysisPanels} from './analysis_panels';
import type {AnalysisAxisBin, AnalysisAxisToken} from './analysis_token_strip/analysis_token_model';
import {StageAnalyses} from './stage_analyses';

/**
 * The expanded analysis view: one stage's analyses as tabs over the page, with the strip of the
 * stage's tokens that selects positions in the shown one.
 */
@Injectable()
export class ExpandedAnalysis {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);
  private readonly analyses = inject(StageAnalyses);

  readonly expandedTokens = computed<AnalysisAxisToken[]>(() => {
    const panel = this.panels.expandedAnalysis();
    if (!panel) {
      return [];
    }
    const start = this.panels.entireStageRange(panel).start;
    const pairs =
      panel.stage === 'prefill'
        ? (this.tokens.inputPairRows(panel.turn) ?? []).flatMap((row) => [
            ...row.leading,
            ...row.text,
            ...row.trailing,
          ])
        : this.tokens.rows(panel.turn);
    return pairs
      .map((pair) => ({
        pair,
        position:
          start +
          (this.conversation.alignment() === 'content'
            ? pair.index
            : ((pair.ref ?? pair.target)?.step ?? pair.index)),
      }))
      .sort((a, b) => a.position - b.position);
  });
  readonly expandedTokenBins = computed<AnalysisAxisBin[]>(() => {
    const panel = this.panels.expandedAnalysis();
    if (!panel) {
      return [];
    }
    const kv = this.panels.overviewKv(panel);
    const graph = this.panels.overviewGraph(panel);
    if (kv) {
      if (kv.tensorView()) {
        return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
      }
      return kv.chartBins().map((bin) => ({...bin, representative: kv.tokenRepresentative(bin)}));
    }
    if (graph) {
      if (graph.inspectionMode() || graph.view() === 'graph') {
        return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
      }
      return graph.bins().map((bin) => ({
        ...bin,
        representative: graph.tokenRepresentative(bin),
      }));
    }
    if (this.panels.overviewFinalLogits(panel)) {
      return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
    }
    const chart = this.panels
      .tokenPanels()
      .find((p) => p.panelId() === panel.id)
      ?.finalChart();
    if (chart) {
      const row =
        chart.chartRows().find((r) => r.layer === chart.selection()?.layer) ?? chart.chartRows()[0];
      return chart.bins().map((bin) => ({
        ...bin,
        representative: representativePosition(
          bin,
          chart.reduction(),
          row?.bins.find((c) => c.start === bin.start && c.end === bin.end)?.metrics.max_abs,
        ),
      }));
    }
    return [];
  });
  readonly expandedTokenSelection = computed(() => {
    const panel = this.panels.expandedAnalysis();
    if (!panel) {
      return null;
    }
    const kv = this.panels.overviewKv(panel);
    if (kv?.tensorView()) {
      const position = kv.tensorPosition();
      return position === null ? null : {start: position, end: position + 1};
    }
    const graph = this.panels.overviewGraph(panel);
    if (graph?.inspectionMode()) {
      const tensor = graph.inspection();
      return tensor ? {start: tensor.position(), end: tensor.position() + 1} : null;
    }
    if (graph?.view() === 'graph') {
      const position = graph.graphPosition();
      return position === null ? null : {start: position, end: position + 1};
    }
    const logits = this.panels.overviewFinalLogits(panel);
    if (logits) {
      return {start: logits.position(), end: logits.position() + 1};
    }
    return (
      this.panels.analysisChartSelections().get(panel.id) ??
      this.panels.overviewGraph(panel)?.selected() ??
      null
    );
  });
  readonly expandedTokenContext = computed(() => {
    const panel = this.panels.expandedAnalysis();
    const bins = this.expandedTokenBins();
    return `${panel?.id}:${this.conversation.alignment()}:${(panel ? this.panels.overviewKv(panel)?.tensorView() : false) ?? false}:${bins[0]?.start}:${bins.at(-1)?.end}:${bins.length}`;
  });
  readonly expandedTokenDisabled = computed(() => {
    const panel = this.panels.expandedAnalysis();
    if (!panel) {
      return true;
    }
    const kv = this.panels.overviewKv(panel);
    return !!kv && !kv.tensorView() && (kv.chartLoading() || !!kv.chartError());
  });
  selectExpandedToken(range: {start: number; end: number}): void {
    const panel = this.panels.expandedAnalysis();
    if (!panel) {
      return;
    }
    const kv = this.panels.overviewKv(panel);
    const graph = this.panels.overviewGraph(panel);
    if (kv?.tensorView()) {
      kv.setTensorPosition(String(range.start));
    } else if (kv) {
      kv.selectRange({...range, layer: null});
    } else if (graph) {
      graph.select({...range, layer: null});
    } else if (this.panels.overviewFinalLogits(panel)) {
      this.panels
        .tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.inspectPosition(range.start);
    } else {
      this.panels
        .tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.finalChart()
        ?.select({...range, layer: null});
    }
  }
  readonly expandedTop = signal(44);
  readonly expandedPlaceholder = signal(0);
  toggleExpandedAnalysis(panel: StageAnalysisPanel): void {
    if (this.panels.expandedAnalysis()) {
      this.panels.exitExpandedAnalysis();
      return;
    }
    this.expandedTop.set(this.conversation.element.nativeElement.getBoundingClientRect().top);
    this.panels.expandTrigger = document.activeElement as HTMLElement;
    this.expandedPlaceholder.set(
      this.panels.expandTrigger?.closest('.stage-analysis-list')?.getBoundingClientRect().height ??
        0,
    );
    this.panels.analysisPanels.update((ps) =>
      ps.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    this.panels.expandedAnalysis.set(panel);
  }
  expandedTabKey(event: KeyboardEvent, panel: StageAnalysisPanel): void {
    const tabs = this.panels.panelsFor(panel.turn, panel.stage);
    const current = tabs.findIndex((p) => p.id === panel.id);
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? tabs.length - 1
          : event.key === 'ArrowRight'
            ? (current + 1) % tabs.length
            : event.key === 'ArrowLeft'
              ? (current + tabs.length - 1) % tabs.length
              : null;
    if (next === null) {
      return;
    }
    event.preventDefault();
    this.analyses.selectExpandedAnalysis(tabs[next]);
    const buttons = (
      event.currentTarget as HTMLElement
    ).parentElement?.querySelectorAll<HTMLElement>('[role="tab"]');
    buttons?.[next]?.focus();
  }
}
