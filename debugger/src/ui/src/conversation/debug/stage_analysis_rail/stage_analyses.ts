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

import {Injectable, afterNextRender, inject, untracked} from '@angular/core';

import {GraphMetricPanel, KvPanel, defaultKvView} from '../../../analyses/registry';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import type {AlignmentMode} from '../../token_alignment';
import {TokenDiff} from '../token_diff/token_diff';
import {AnalysisPanels} from './analysis_panels';
import {AnalysisPicker, type AnalysisPickerData} from './analysis_picker/analysis_picker';

/**
 * What a reader does with the analysis views: adds one to a stage, opens, closes or removes it,
 * inspects its overview, a chart's point or a token metric in the details, and keeps that focus
 * and the reading's selection in step while charts zoom and panels load.
 */
@Injectable()
export class StageAnalyses {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);

  inspectTokenMetric(
    panel: StageAnalysisPanel,
    index: number,
    range?: {start: number; end: number},
  ): void {
    this.tokens.analysisOverview.set(null);
    this.tokens.select(panel.turn, index);
    this.tokens.tokenMetricInspector.set(panel);
    this.panels.tokenMetricRange.set(range ? {panelId: panel.id, index, ...range} : null);
    if (range && range.end - range.start > 1) {
      this.tokens.selection.set({
        kind: 'stage',
        turn: panel.turn,
        stage: panel.stage,
        side: 'target',
        range,
      });
    }
  }
  selectExpandedAnalysis(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((ps) =>
      ps.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    const range = this.panels.selectionRangeFor(panel);
    this.panels.expandedAnalysis.set(panel);
    this.panels.analysisHover.set(null);
    const kv = this.panels.overviewKv(panel);
    const graph = this.panels.overviewGraph(panel);
    if (kv?.tensorView()) {
      if (range) {
        kv.setTensorPosition(String(kv.tokenRepresentative(range)));
      }
      this.inspectMetric(kv);
    } else if (kv && range) {
      kv.selectRange({
        ...range,
        layer: 'layer' in range && typeof range.layer === 'number' ? range.layer : null,
      });
      this.inspectMetric(kv);
    } else if (graph && range) {
      graph.select({
        ...range,
        layer: 'layer' in range && typeof range.layer === 'number' ? range.layer : null,
      });
    } else if (this.panels.overviewFinalLogits(panel)) {
      if (range) {
        this.panels
          .tokenPanels()
          .find((p) => p.panelId() === panel.id)
          ?.inspectPosition(range.start);
      }
      this.inspectAnalysisOverview(panel);
    } else if (range && (panel.kind === 'tokens' || panel.kind === 'distribution')) {
      this.panels
        .tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.finalChart()
        ?.select({...range, layer: null});
    } else {
      this.inspectAnalysisOverview(panel);
    }
  }
  openAnalysisPicker(turn: number, stage: 'prefill' | 'decode'): void {
    const dialog = this.conversation.dialog.open<
      AnalysisPicker,
      AnalysisPickerData,
      StageAnalysisPanel['kind']
    >(AnalysisPicker, {
      width: '466px',
      maxWidth: 'calc(100vw - 32px)',
      ariaLabel: 'Add analysis view',
      autoFocus: 'dialog',
      data: {turn, stage},
      // The picker reads this page's stages and analyses.
      injector: this.conversation.injector,
    });
    dialog.afterClosed().subscribe((kind) => {
      if (kind && !this.panels.analysisUnavailable(turn, stage, kind)) {
        this.addAnalysis(turn, stage, kind);
      }
    });
  }
  addAnalysis(
    turn: number,
    stage: 'prefill' | 'decode',
    kind: StageAnalysisPanel['kind'],
    initialKvView?: {
      tensor: boolean;
      metricChart: 'heatmap' | 'lineChart';
      tensorChart: 'index' | 'scatter';
    },
  ): void {
    const inheritedRange = this.panels.selectionRangeFor({turn, stage, kind});
    if (this.tokens.metricInspector() && inheritedRange) {
      this.tokens.selection.set({
        kind: 'stage',
        turn,
        stage,
        side: 'target',
        range: inheritedRange,
      });
    }
    const peers = this.panels.panelsFor(turn, stage).filter((p) => p.kind === kind);
    const number = peers.length
      ? Math.max(...peers.map((p) => p.instanceNumber ?? peers.indexOf(p) + 1)) + 1
      : 1;
    const panel: StageAnalysisPanel = {
      id: 'analysis-' + ++this.panels.analysisSerial,
      instanceNumber: number,
      turn,
      stage,
      kind,
      open: true,
      ...(kind === 'kv' && initialKvView
        ? {
            kv: {
              ...defaultKvView(),
              contextId: this.tokens.turnStage(turn, stage)?.kv.context_id ?? '',
              metric: 'cosine_similarity' as const,
              head: 'min',
              view: initialKvView.metricChart,
              tensor: {
                active: initialKvView.tensor,
                plot: initialKvView.tensorChart,
                position: null,
                layer: null,
                head: null,
              },
            },
          }
        : {}),
      ...(['tokens', 'distribution'].includes(kind)
        ? {
            logit: {
              metric: kind === 'distribution' ? 'entropy' : 'logit_rmse',
              open: true,
            },
          }
        : {}),
    };
    this.panels.analysisPanels.update((panels) => [
      ...panels.map((p) =>
        peers.includes(p) ? {...p, instanceNumber: p.instanceNumber ?? peers.indexOf(p) + 1} : p,
      ),
      panel,
    ]);
    const selected = this.tokens.selection();
    if (
      !selected ||
      selected.turn !== turn ||
      (selected.kind === 'stage'
        ? selected.stage !== stage
        : (selected.kind === 'input' ? 'prefill' : 'decode') !== stage)
    ) {
      this.tokens.selection.set({kind: 'stage', turn, stage, side: 'target'});
    }
    this.tokens.setGroup('analyses', true);
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.tokens.analysisOverview.set(panel);
    this.tokens.details.set(true);
    if (this.panels.isExpandedStage(turn, stage)) {
      this.panels.expandedAnalysis.set(panel);
      afterNextRender(() => this.selectExpandedAnalysis(panel), {
        injector: this.conversation.injector,
      });
    } else {
      this.locateAnalysis(panel);
    }
  }
  private leaveAnalysis(panel: StageAnalysisPanel): void {
    if (this.panels.analysisHover()?.panelId === panel.id) {
      this.panels.analysisHover.set(null);
    }
    const ownsMetric = this.tokens.metricInspector()?.panelId() === panel.id;
    const ownsTokenMetric = this.tokens.tokenMetricInspector()?.id === panel.id;
    const ownsOverview = this.tokens.analysisOverview()?.id === panel.id;
    if (!ownsMetric && !ownsTokenMetric && !ownsOverview) {
      return;
    }

    // Analysis focus is a stage range, even when it contains just one position.
    // Leaving the panel must not turn that focus into a direct token selection.
    const tokenRange = ownsTokenMetric ? this.panels.tokenMetricRange() : null;
    const range = ownsOverview
      ? this.panels.analysisOverviewRange(panel)
      : tokenRange?.panelId === panel.id
        ? tokenRange
        : this.panels.selectionRangeFor(panel);
    if (ownsMetric) {
      this.tokens.metricInspector.set(null);
    }
    if (ownsTokenMetric) {
      this.tokens.tokenMetricInspector.set(null);
      this.panels.tokenMetricRange.set(null);
    }
    if (ownsOverview) {
      this.tokens.analysisOverview.set(null);
    }
    this.tokens.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      ...(range ? {range: {start: range.start, end: range.end}} : {}),
    });
  }
  toggleAnalysis(panel: StageAnalysisPanel): void {
    if (!panel.open) {
      this.showAnalysis(panel);
      return;
    }
    this.leaveAnalysis(panel);
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: !p.open} : p)),
    );
  }
  /** Opens a panel, if closed, and gives the details its overview, as opening it in its rail
   *  does. */
  showAnalysis(panel: StageAnalysisPanel): void {
    const range = this.panels.selectionRangeFor(panel);
    if (range && this.tokens.metricInspector()) {
      this.tokens.selection.set({
        kind: 'stage',
        turn: panel.turn,
        stage: panel.stage,
        side: 'target',
        range,
      });
    }
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.tokens.analysisOverview.set(panel);
    if (!range) {
      this.tokens.selection.set({
        kind: 'stage',
        turn: panel.turn,
        stage: panel.stage,
        side: 'target',
      });
    }
    this.tokens.details.set(true);
    this.locateAnalysis(panel);
  }
  removeAnalysis(panel: StageAnalysisPanel): void {
    this.leaveAnalysis(panel);
    if (this.panels.expandedAnalysis()?.id === panel.id) {
      this.panels.exitExpandedAnalysis();
    }
    if (this.tokens.metricInspector()?.panelId() === panel.id) {
      this.tokens.metricInspector.set(null);
    }
    if (this.tokens.tokenMetricInspector()?.id === panel.id) {
      this.tokens.tokenMetricInspector.set(null);
    }
    this.panels.analysisPanels.update((panels) => panels.filter((p) => p.id !== panel.id));
  }
  inspectAnalysisOverview(panel: StageAnalysisPanel): void {
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.panels.analysisHover.set(null);
    this.tokens.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      range: this.panels.analysisOverviewRange(panel),
    });
    this.tokens.analysisOverview.set(panel);
    this.tokens.details.set(true);
  }
  inspectMetric(panel: KvPanel | GraphMetricPanel): void {
    const config = this.panels.analysisPanels().find((p) => p.id === panel.panelId());
    if (config?.open === false && panel instanceof GraphMetricPanel && panel.inspectionMode()) {
      return;
    }
    this.tokens.analysisOverview.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.tokens.metricInspector.set(panel);
    if (config) {
      this.tokens.selection.set({
        kind: 'stage',
        turn: config.turn,
        side: 'target',
        stage: config.stage,
      });
    }
    this.tokens.details.set(true);
  }
  private analysisPointer: {x: number; y: number; moved: boolean} | null = null;
  analysisPointerDown(event: PointerEvent, panel: StageAnalysisPanel): void {
    this.analysisPointer =
      event.button === 0 ? {x: event.clientX, y: event.clientY, moved: false} : null;
    if (event.button === 0 && this.panels.overviewPanel()?.id !== panel.id) {
      this.inspectAnalysisOverview(panel);
    }
  }
  analysisPointerMove(event: PointerEvent): void {
    const pointer = this.analysisPointer;
    if (
      pointer &&
      event.buttons !== 0 &&
      Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 4
    ) {
      pointer.moved = true;
    }
  }
  analysisPointerCancel(): void {
    if (this.analysisPointer) {
      this.analysisPointer.moved = true;
    }
  }
  analysisBackgroundClick(event: MouseEvent, panel: StageAnalysisPanel): void {
    if (
      (event.target as Element).closest(
        'range-chart, channel-chart, scatter-chart, .results, .details, heatmap-controls, button, a, input, select, textarea, label, [role="button"], [role="dialog"], [role="menu"], [role="listbox"], [role="option"], [contenteditable="true"]',
      )
    ) {
      return;
    }
    this.inspectAnalysisStage(panel, event);
  }
  inspectAnalysisStage(panel: StageAnalysisPanel, event: MouseEvent): void {
    event.stopPropagation();
    const pointer = this.analysisPointer;
    if (
      event.button !== 0 ||
      event.defaultPrevented ||
      (pointer &&
        (pointer.moved || Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 4)) ||
      window.getSelection()?.toString()
    ) {
      return;
    }
    this.analysisPointer = null;
    // Clear the plotted selection while retaining the active analysis and its viewport.
    this.panels.overviewKv(panel)?.clearElementSelection();
    this.panels.overviewFinalLogits(panel)?.clearElementSelection();
    const graph = this.panels.overviewGraph(panel);
    graph?.inspection()?.clearElementSelection();
    graph?.selected.set(null);
    this.inspectAnalysisOverview(panel);
  }
  private readonly analysisViewports = new Map<string, {start: number; end: number}>();
  syncAnalysisViewport(): void {
    const panel = this.panels.overviewPanel();
    if (!panel) {
      return;
    }
    const range = this.panels.analysisOverviewRange(panel);
    const previous = this.analysisViewports.get(panel.id);
    this.analysisViewports.set(panel.id, range);
    const changed = previous && (previous.start !== range.start || previous.end !== range.end);
    const overview = this.tokens.analysisOverview()?.id === panel.id;
    untracked(() => {
      // A new viewport replaces the old bin even when that bin still fits inside it.
      if (changed) {
        this.inspectAnalysisOverview(panel);
      } else if (overview) {
        this.syncAnalysisOverview(panel, range);
      }
    });
  }
  private syncAnalysisOverview(
    panel: StageAnalysisPanel,
    range: {start: number; end: number},
  ): void {
    const selected = this.tokens.selection();
    if (
      selected?.kind === 'stage' &&
      selected.turn === panel.turn &&
      selected.stage === panel.stage &&
      selected.range?.start === range.start &&
      selected.range?.end === range.end
    ) {
      return;
    }
    this.tokens.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      range: {start: range.start, end: range.end},
    });
  }
  syncMetricInspectorVisibility(): void {
    const inspector = this.tokens.metricInspector();
    if (!inspector) {
      return;
    }
    const graphInspection = inspector instanceof GraphMetricPanel && inspector.inspectionMode();
    // A linked token can select a graph while its initial tensors are still loading.
    if (
      inspector instanceof GraphMetricPanel &&
      (inspector.request.loading() ||
        !(graphInspection ? inspector.fullRange() : inspector.range()))
    ) {
      return;
    }
    const tensor = graphInspection ? inspector.inspection() : null;
    const selected = graphInspection
      ? tensor
        ? {start: tensor.position(), end: tensor.position() + 1, layer: tensor.layer()}
        : null
      : inspector instanceof KvPanel
        ? inspector.inspectedRange()
        : inspector.selected();
    if (!selected) {
      return;
    }
    const range = graphInspection
      ? inspector.fullRange()
      : inspector instanceof KvPanel
        ? inspector.visibleRange()
        : inspector.range();
    const {start, end, layer} = selected;
    const layers =
      inspector instanceof KvPanel
        ? inspector.view() === 'lineChart'
          ? inspector.selectedLineChartLayers()
          : inspector.layerIds()
        : !graphInspection && inspector.view() === 'lineChart'
          ? inspector.selectedLayers()
          : inspector.layers();
    const hiddenLayer = layers.length === 0 || (layer !== null && !layers.includes(layer));
    if (!range || start < range.start || end > range.end || hiddenLayer) {
      const panel = this.panels.analysisPanels().find((p) => p.id === inspector.panelId());
      if (panel) {
        untracked(() => {
          this.tokens.metricInspector.set(null);
          this.panels.analysisHover.set(null);
          this.tokens.selection.set({
            kind: 'stage',
            turn: panel.turn,
            stage: panel.stage,
            side: 'target',
          });
          this.tokens.analysisOverview.set(panel);
        });
      }
    }
  }
  setAlignment(mode: AlignmentMode): void {
    const active = this.panels.overviewPanel();
    this.conversation.alignment.set(mode);
    this.panels.analysisHover.set(null);
    this.tokens.analysisOverview.set(active ?? null);
    for (const graph of this.panels.graphPanels()) {
      graph.selected.set(null);
      graph.viewport.set(null);
    }
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.logit ? {...p, logit: {...p.logit, positionRange: undefined}} : p)),
    );
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.tokens.selection.set(null);
  }
  locateAnalysis(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const range = this.panels.selectionRangeFor(panel);
        if (range && this.tokens.metricInspector()?.panelId() !== panel.id) {
          this.panels
            .graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.selected.set({...range, layer: null});
          const kv = this.panels.kvPanels().find((p) => p.panelId() === panel.id);
          if (kv) {
            kv.position.set(range.start);
            kv.selectionEnd.set(range.end);
            kv.column.set(true);
          }
        }
        if (this.tokens.analysisOverview()?.id === panel.id && range) {
          const graph = this.panels.graphPanels().find((p) => p.panelId() === panel.id);
          const kv = this.panels.kvPanels().find((p) => p.panelId() === panel.id);
          const target = graph ?? kv;
          const tokenSelection = this.tokens.tokenSelection();
          if (target) {
            this.inspectMetric(target);
          } else if (tokenSelection) {
            this.inspectTokenMetric(panel, tokenSelection.index);
          }
        }
        this.conversation.element.nativeElement
          .querySelector('#' + panel.id)
          ?.scrollIntoView({block: 'start', behavior: 'smooth'});
      },
      {injector: this.conversation.injector},
    );
  }
  /** Gives up a token metric's focus once its range leaves the panel's view. Run in an effect. */
  syncTokenMetricInspector(): void {
    const owner = this.tokens.tokenMetricInspector();
    if (!owner) {
      return;
    }
    const panel = this.panels.analysisPanels().find((p) => p.id === owner.id);
    if (!panel) {
      return;
    }
    const range = this.panels.analysisRange(panel);
    const visible = this.panels.tokenPanelRange(panel);
    if (range.start < visible.start || range.end > visible.end) {
      untracked(() => {
        this.tokens.tokenMetricInspector.set(null);
        this.panels.tokenMetricRange.set(null);
        this.tokens.selection.set({
          kind: 'stage',
          turn: panel.turn,
          stage: panel.stage,
          side: 'target',
        });
        this.tokens.analysisOverview.set(panel);
      });
    }
  }
  /** Marks the reading's selection on every panel of its stage while no panel has the focus.
   *  Run in an effect. */
  followSelection(): void {
    if (this.tokens.metricInspector() || this.tokens.analysisOverview()) {
      return;
    }
    for (const panel of this.panels.analysisPanels()) {
      const range = this.panels.selectionRangeFor(panel);
      const graph = this.panels.graphPanels().find((p) => p.panelId() === panel.id);
      const kv = this.panels.kvPanels().find((p) => p.panelId() === panel.id);
      if (range) {
        untracked(() => {
          graph?.selected.set({...range, layer: null});
          if (kv) {
            kv.position.set(range.start);
            kv.selectionEnd.set(range.end);
            kv.column.set(true);
          }
        });
      } else if (graph) {
        untracked(() => graph.selected.set(null));
      }
    }
  }
  /** Gives a restored view's inspected panel the focus once its live panel is there. Run in an
   *  effect. */
  followPendingInspector(): void {
    const id = this.panels.pendingInspector();
    const panel = [...this.panels.kvPanels(), ...this.panels.graphPanels()].find(
      (p) => p.panelId() === id,
    );
    const tokenPanel = this.panels
      .analysisPanels()
      .find((p) => p.id === id && (p.kind === 'tokens' || p.kind === 'distribution'));
    if (tokenPanel) {
      this.tokens.tokenMetricInspector.set(tokenPanel);
      this.panels.pendingInspector.set(null);
    }
    if (id && panel) {
      this.tokens.tokenMetricInspector.set(null);
      this.tokens.metricInspector.set(panel);
      this.panels.pendingInspector.set(null);
    }
  }
}
