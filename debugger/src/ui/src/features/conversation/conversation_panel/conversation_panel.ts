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

import {AnalysisArchitecture} from '../analysis_architecture';
import {tensorPointLabel} from '../../analysis/tensor_labels';
import {ExplorerPagination} from '../../../shared/explorer_pagination/explorer_pagination';
import {
  animateLocationFeedback,
  animateAnalysisLocationFeedback,
} from '../../../shared/locate_feedback';
import {AnalysisTokenStrip} from '../analysis_token_strip/analysis_token_strip';
import {
  representativePosition,
  type AnalysisAxisToken,
  type AnalysisAxisBin,
} from '../analysis_token_strip/analysis_token_model';
import {defaultKvView} from '../../kv/kv_panel/kv_view_state';
import {StageStateInfo, type StateLocation} from '../stage_state_info';
import type {KvRangeSelection} from '../../kv/kv_panel/kv_range_types';
import type {PositionAnalysis} from '../../../data/contracts/position_analysis';
import {KvRequest} from '../../kv/kv_panel/kv_data';
import type {KvAnalysis} from '../../../data/contracts/kv';
import {GraphMetricPanel} from '../graph_metric_panel';
import {NgTemplateOutlet} from '@angular/common';
import {KvPanel, KV_METRICS} from '../../kv/kv_panel/kv_panel';
import {
  StageAnalysisPanel,
  restoreLogitPanel,
  numberStagePanels,
  canonicalAnalysisKind,
} from '../stage_analysis';
import {RunnerStatus} from '../../sessions/runner_status/runner_status';
import {
  TokenDiffViewState,
  pairIdentity,
  readViewState,
  prepareViewRestore,
  captureViewState,
  type ConversationObservation,
} from '../conversation_view_state';
import {ConversationReadingController} from '../conversation_reading_controller';
import {ConversationBookmark} from '../../../app/workspace_state_service';
import {ReportApiService} from '../../../data/report_api_service';
import {TOKEN_METRICS} from '../token_metric_metadata';
import {TokenTrends} from './token_trends';
import {TokenTrendState, SIDE_TREND_METRICS, FINAL_LOGIT_METRICS} from '../token_trends';
import {TENSOR_METRICS} from '../../analysis/position_tensor_model';
import {KvSnapshot} from '../../../data/contracts/telemetry';
import {
  TokenColor,
  pairKey,
  queryRecord,
  colorLabel,
  heatLevel,
  heatPalette,
  TOKEN_COLOR_MAX,
} from '../token_analysis';
import {createConversationScrollbar} from '../conversation_scrollbar';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {
  ExplorerInfoList,
  type ExplorerInfoItem,
} from '../../../shared/explorer_info_list/explorer_info_list';
import {ExplorerLocator} from '../../../shared/explorer_locator/explorer_locator';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {
  tokenContent,
  isTemplateToken,
  capturedTokenLabel,
  pairLineBreaks,
  tokenMarkup,
  capturedInputPairRows,
  spokenTokenText,
  capturedTokenContent,
} from '../conversation_content';
import {syncPairWidths} from '../conversation_line_sync';
import {batchForwardId, tokenMatchesBatch} from '../../graph/batch_forward';
import {
  ChangeDetectionStrategy,
  Component,
  inject,
  signal,
  computed,
  linkedSignal,
  effect,
  ElementRef,
  TemplateRef,
  HostListener,
  Injector,
  afterNextRender,
  OnDestroy,
  NgZone,
  untracked,
  afterRenderEffect,
  DestroyRef,
  viewChildren,
} from '@angular/core';
import {createFlowVirtualizer, FlowVirtualizer} from '../flow_virtualization';
import {virtualConversation, streamingConversation} from '../conversation_virtualization';
import {
  capturedInputRows,
  capturedInputSegments,
  historyStatus,
  capturedPhaseText,
  hasPhaseTokens,
  splitPhasePairs,
  tokenPhase,
  TokenPhase,
  ConversationSelection,
} from '../conversation_content';
import {createFlowNodeNavigator, FlowNodeNavigator} from '../flow_node_navigation';
import {EmptyChatDemo} from '../empty_chat_demo/empty_chat_demo';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {TokenDisplay} from '../token_display/token_display';
import {TokenQueryControl} from '../token_query_control/token_query_control';
import {compileQuery} from '../token_query';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatButtonModule} from '@angular/material/button';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {CapturedToken} from '../../../data/contracts/capture';
import {
  alignTokens,
  AlignmentMode,
  decodeOutcome,
  forkPair,
  positionPairs,
  tokenPosition,
  TokenPair,
} from '../token_alignment';
import {ReportStateService} from '../../../data/report_state_service';
import {WorkspaceStateService} from '../../../app/workspace_state_service';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';
import {PANEL_WIDTH} from '../conversation_view_state';
import {FindNavigator} from '../../../shared/find_navigator/find_navigator';
import {formatCount, formatPercent, formatSignificant} from '../../../shared/format/format';
import {summarizeStageTokens} from '../stage_token_summary';
import {TokenAnalysisController} from './token_analysis_controller';
import {ConversationComposerController} from './conversation_composer';
import {
  renderInputToken as renderInput,
  renderOutputToken as renderOutput,
} from '../conversation_virtualization';
import type {CapturedConversation} from '../../../data/contracts/capture';
/** Aligned tokens on either runtime: static and virtualized Decode rows share these attributes. */
const HOVER_TARGET = '[data-turn][data-side][data-row]';
const NO_BREAKS: readonly number[] = [];

@Component({
  selector: 'conversation-panel',
  standalone: true,
  imports: [
    AnalysisArchitecture,
    ExplorerPagination,
    NgTemplateOutlet,
    AnalysisTokenStrip,
    KvPanel,
    GraphMetricPanel,
    FindNavigator,
    ResizeHandle,
    RunnerStatus,
    TokenTrends,
    ExplorerInfoValue,
    ExplorerInfoSection,
    ExplorerInfoList,
    ExplorerLocator,
    EmptyChatDemo,
    ConfigPicker,
    TokenQueryControl,
    StageStateInfo,
    TokenDisplay,
    MatButtonModule,
    MatSlideToggleModule,
    MatIconModule,
    MatDialogModule,
    MatTooltipModule,
  ],
  templateUrl: './conversation_panel.ng.html',
  styleUrl: './conversation_panel.scss',
  host: {
    '(document:pointermove)': 'analysisPointerMove($event)',
    '(document:pointercancel)': 'analysisPointerCancel()',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConversationPanel implements OnDestroy {
  readonly analysisOverview = signal<StageAnalysisPanel | null>(null);
  readonly tokenMetricInspector = signal<StageAnalysisPanel | null>(null);
  readonly tokenMetricRange = signal<{
    panelId: string;
    index: number;
    start: number;
    end: number;
  } | null>(null);
  inspectTokenMetric(
    panel: StageAnalysisPanel,
    index: number,
    range?: {start: number; end: number},
  ) {
    this.analysisOverview.set(null);
    this.select(panel.turn, index, undefined, false);
    this.tokenMetricInspector.set(panel);
    this.tokenMetricRange.set(range ? {panelId: panel.id, index, ...range} : null);
    if (range && range.end - range.start > 1)
      this.selection.set({
        kind: 'stage',
        turn: panel.turn,
        stage: panel.stage,
        side: 'target',
        range,
      });
  }
  tokenPanelSelectedIndex(panel: StageAnalysisPanel): number | null {
    const bin = this.tokenMetricRange();
    if (this.tokenMetricInspector()?.id === panel.id && bin?.panelId === panel.id) return bin.index;
    const selected = this.tokenSelection();
    return selected?.turn === panel.turn ? selected.index : null;
  }
  readonly expandedAnalysis = signal<StageAnalysisPanel | null>(null);
  readonly expandedTokens = computed<AnalysisAxisToken[]>(() => {
    const panel = this.expandedAnalysis();
    if (!panel) return [];
    const start = this.entireStageRange(panel).start;
    const pairs =
      panel.stage === 'prefill'
        ? (this.inputPairRows(panel.turn) ?? []).flatMap((row) => [
            ...row.leading,
            ...row.text,
            ...row.trailing,
          ])
        : this.rows(panel.turn);
    return pairs
      .map((pair) => ({
        pair,
        position:
          start +
          (this.alignment() === 'content'
            ? pair.index
            : ((pair.ref ?? pair.target)?.step ?? pair.index)),
      }))
      .sort((a, b) => a.position - b.position);
  });
  readonly expandedTokenBins = computed<AnalysisAxisBin[]>(() => {
    const panel = this.expandedAnalysis();
    if (!panel) return [];
    const kv = this.overviewKv(panel),
      graph = this.overviewGraph(panel);
    if (kv) {
      if (kv.tensorView())
        return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
      return kv.chartBins().map((bin) => ({...bin, representative: kv.tokenRepresentative(bin)}));
    }
    if (graph) {
      if (graph.inspectionMode() || graph.view() === 'graph')
        return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
      return graph.bins().map((bin) => ({
        ...bin,
        representative: graph.tokenRepresentative(bin),
      }));
    }
    if (this.overviewFinalLogits(panel))
      return this.expandedTokens().map((t) => ({start: t.position, end: t.position + 1}));
    const chart = this.tokenPanels()
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
    const panel = this.expandedAnalysis();
    if (!panel) return null;
    const kv = this.overviewKv(panel);
    if (kv?.tensorView()) {
      const position = kv.tensorPosition();
      return position === null ? null : {start: position, end: position + 1};
    }
    const graph = this.overviewGraph(panel);
    if (graph?.inspectionMode()) {
      const tensor = graph.inspection();
      return tensor ? {start: tensor.position(), end: tensor.position() + 1} : null;
    }
    if (graph?.view() === 'graph') {
      const position = graph.graphPosition();
      return position === null ? null : {start: position, end: position + 1};
    }
    const logits = this.overviewFinalLogits(panel);
    if (logits) return {start: logits.position(), end: logits.position() + 1};
    return (
      this.analysisChartSelections().get(panel.id) ?? this.overviewGraph(panel)?.selected() ?? null
    );
  });
  readonly expandedTokenContext = computed(() => {
    const panel = this.expandedAnalysis(),
      bins = this.expandedTokenBins();
    return `${panel?.id}:${this.alignment()}:${(panel ? this.overviewKv(panel)?.tensorView() : false) ?? false}:${bins[0]?.start}:${bins.at(-1)?.end}:${bins.length}`;
  });
  readonly expandedTokenDisabled = computed(() => {
    const panel = this.expandedAnalysis();
    if (!panel) return true;
    const kv = this.overviewKv(panel);
    return !!kv && !kv.tensorView() && (kv.chartLoading() || !!kv.chartError());
  });
  selectExpandedToken(range: {start: number; end: number}) {
    const panel = this.expandedAnalysis();
    if (!panel) return;
    const kv = this.overviewKv(panel),
      graph = this.overviewGraph(panel);
    if (kv?.tensorView()) kv.setTensorPosition(String(range.start));
    else if (kv) kv.selectRange({...range, layer: null});
    else if (graph) graph.select({...range, layer: null});
    else if (this.overviewFinalLogits(panel))
      this.tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.inspectPosition(range.start);
    else
      this.tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.finalChart()
        ?.select({...range, layer: null});
  }
  readonly expandedTop = signal(44);
  readonly expandedPlaceholder = signal(0);
  private expandTrigger: HTMLElement | null = null;
  stageGlobalStart(turn: number) {
    return this.turns().find((t) => t.n === turn)?.step_start ?? 0;
  }
  isExpandedStage(turn: number, stage: string) {
    const p = this.expandedAnalysis();
    return p?.turn === turn && p?.stage === stage;
  }
  toggleExpandedAnalysis(panel: StageAnalysisPanel) {
    if (this.expandedAnalysis()) {
      this.exitExpandedAnalysis();
      return;
    }
    this.expandedTop.set(this.element.nativeElement.getBoundingClientRect().top);
    this.expandTrigger = document.activeElement as HTMLElement;
    this.expandedPlaceholder.set(
      this.expandTrigger?.closest('.stage-analysis-list')?.getBoundingClientRect().height ?? 0,
    );
    this.analysisPanels.update((ps) => ps.map((p) => (p.id === panel.id ? {...p, open: true} : p)));
    this.expandedAnalysis.set(panel);
  }
  selectExpandedAnalysis(panel: StageAnalysisPanel) {
    this.analysisPanels.update((ps) => ps.map((p) => (p.id === panel.id ? {...p, open: true} : p)));
    const range = this.selectionRangeFor(panel);
    this.expandedAnalysis.set(panel);
    this.analysisHover.set(null);
    const kv = this.overviewKv(panel),
      graph = this.overviewGraph(panel);
    if (kv?.tensorView()) {
      if (range) kv.setTensorPosition(String(kv.tokenRepresentative(range)));
      this.inspectMetric(kv);
    } else if (kv && range) {
      kv.selectRange({
        ...range,
        layer: 'layer' in range && typeof range.layer === 'number' ? range.layer : null,
      });
      this.inspectMetric(kv);
    } else if (graph && range)
      graph.select({
        ...range,
        layer: 'layer' in range && typeof range.layer === 'number' ? range.layer : null,
      });
    else if (this.overviewFinalLogits(panel)) {
      if (range)
        this.tokenPanels()
          .find((p) => p.panelId() === panel.id)
          ?.inspectPosition(range.start);
      this.inspectAnalysisOverview(panel);
    } else if (range && (panel.kind === 'tokens' || panel.kind === 'distribution'))
      this.tokenPanels()
        .find((p) => p.panelId() === panel.id)
        ?.finalChart()
        ?.select({...range, layer: null});
    else this.inspectAnalysisOverview(panel);
  }
  expandedTabKey(event: KeyboardEvent, panel: StageAnalysisPanel) {
    const tabs = this.panelsFor(panel.turn, panel.stage);
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
    if (next === null) return;
    event.preventDefault();
    this.selectExpandedAnalysis(tabs[next]);
    const buttons = (
      event.currentTarget as HTMLElement
    ).parentElement?.querySelectorAll<HTMLElement>('[role="tab"]');
    buttons?.[next]?.focus();
  }
  exitExpandedAnalysis() {
    this.expandedAnalysis.set(null);
    requestAnimationFrame(() => this.expandTrigger?.focus({preventScroll: true}));
  }
  readonly analysisPanels = signal<StageAnalysisPanel[]>([]);
  readonly kvPanels = viewChildren(KvPanel);
  readonly graphPanels = viewChildren(GraphMetricPanel);
  readonly tokenPanels = viewChildren(TokenTrends);
  readonly pendingInspector = signal<string | null>(null);
  readonly metricInspector = signal<KvPanel | GraphMetricPanel | null>(null);
  private analysisSerial = 0;
  panelsFor(turn: number, stage: string) {
    return this.analysisPanels().filter((p) => p.turn === turn && p.stage === stage);
  }
  readonly analysisGraph = new KvRequest<PositionAnalysis>();
  readonly analysisKv = new KvRequest<KvAnalysis>();
  readonly analysisOptions = [
    {
      kind: 'tokens',
      label: 'Final Logits',
      splitBy: '-',
      breakdownBy: '-',
      aggregation: '-',
      icon: 'show_chart',
    },
    {
      kind: 'distribution',
      label: 'Sampling Distribution',
      splitBy: '-',
      breakdownBy: 'Runtime',
      aggregation: '-',
      icon: 'show_chart',
    },
    {
      kind: 'activations',
      label: 'Intermediate Tensors',
      splitBy: '-',
      breakdownBy: 'Layer',
      aggregation: 'Sync points',
      icon: 'analytics',
    },
    {
      kind: 'kv',
      label: 'KV cache',
      splitBy: 'Cache type',
      breakdownBy: 'Layer',
      aggregation: 'Heads',
      icon: 'grid_on',
    },
  ] as const;
  readonly analysisPickerKind = signal<StageAnalysisPanel['kind'] | null>(null);
  readonly analysisPickerOption = computed(() =>
    this.analysisOptions.find((option) => option.kind === this.analysisPickerKind()),
  );
  analysisArchitectureUnavailable(turn: number, stage: string) {
    return Object.fromEntries(
      this.analysisOptions.map((option) => [
        option.kind,
        this.analysisUnavailable(turn, stage, option.kind),
      ]),
    );
  }
  openAnalysisPicker(template: TemplateRef<unknown>, turn: number, stage: 'prefill' | 'decode') {
    this.analysisPickerKind.set(null);
    this.loadAnalysisOptions(turn, stage);
    const dialog = this.dialog.open<unknown, unknown, StageAnalysisPanel['kind']>(template, {
      width: '466px',
      maxWidth: 'calc(100vw - 32px)',
      ariaLabel: 'Add analysis view',
      autoFocus: 'dialog',
      data: {turn, stage},
    });
    dialog.afterClosed().subscribe((kind) => {
      if (kind && !this.analysisUnavailable(turn, stage, kind)) {
        this.addAnalysis(turn, stage, kind);
      }
    });
  }
  loadAnalysisOptions(turn: number, stage: string) {
    const capture = this.state.captureId(),
      batch = this.state
        .session()
        ?.batches.filter((b) => b.turn === turn && b.phase === stage)
        .at(-1);
    this.analysisGraph.load(
      batch ? (signal) => this.api.positionAnalysis(capture, turn, stage, signal) : null,
    );
    this.analysisKv.load(
      this.state.session() ? (signal) => this.api.kvMetadata(capture, turn, signal) : null,
    );
  }
  analysisUnavailable(turn: number, stage: string, kind: StageAnalysisPanel['kind']) {
    if (kind === 'tokens' || kind === 'distribution') {
      if (stage !== 'decode') return 'Per-position logits not captured for Prefill';
      if (this.analysisLoading())
        return kind === 'distribution'
          ? 'Checking sampling probabilities…'
          : 'Checking captured logits…';
      if (kind === 'distribution')
        return this.rows(turn).some((pair) => {
          const distribution = this.tokenAnalysis().get(pairKey(turn, pair))?.sampling_distribution;
          return distribution?.ref?.entropy != null || distribution?.target?.entropy != null;
        })
          ? ''
          : 'Sampling probabilities unavailable';
      return this.rows(turn).some((pair) => {
        const captured = this.tokenAnalysis().get(pairKey(turn, pair));
        return !!(captured?.distribution?.ref?.size || captured?.distribution?.target?.size);
      })
        ? ''
        : 'Final logits not captured';
    }
    if (kind === 'kv') {
      if (this.analysisKv.loading()) return 'Checking captured KV…';
      if (this.analysisKv.error()) return 'Could not check KV capture; reopen to retry';
      const moment = stage === 'prefill' ? 'prefill_post' : 'terminal';
      return this.analysisKv
        .data()
        ?.contexts.some(
          (c) =>
            c.turn === turn &&
            c.moment === moment &&
            c.layers.some((l) => l.position_count && l.head_count),
        )
        ? ''
        : 'Comparable KV positions not captured';
    }
    if (this.analysisGraph.loading()) return 'Checking captured tensors…';
    if (this.analysisGraph.error()) return 'Could not check tensors; reopen to retry';
    return this.analysisGraph.data()?.rows.some((r) => !!r.shape?.length)
      ? ''
      : 'Comparable layer tensors not captured';
  }
  addAnalysis(
    turn: number,
    stage: 'prefill' | 'decode',
    kind: StageAnalysisPanel['kind'],
    initialKvView?: {
      tensor: boolean;
      metricChart: 'heatmap' | 'trends';
      tensorChart: 'index' | 'scatter';
    },
  ) {
    kind = canonicalAnalysisKind(kind);
    const inheritedRange = this.selectionRangeFor({turn, stage, kind});
    if (this.metricInspector() && inheritedRange)
      this.selection.set({kind: 'stage', turn, stage, side: 'target', range: inheritedRange});
    const peers = this.panelsFor(turn, stage).filter((p) => canonicalAnalysisKind(p.kind) === kind);
    const number = peers.length
      ? Math.max(...peers.map((p) => p.instanceNumber ?? peers.indexOf(p) + 1)) + 1
      : 1;
    this.analysisPanels.update((panels) => [
      ...panels.map((p) =>
        peers.includes(p) ? {...p, instanceNumber: p.instanceNumber ?? peers.indexOf(p) + 1} : p,
      ),
      {
        id: 'analysis-' + ++this.analysisSerial,
        instanceNumber: number,
        turn,
        stage,
        kind,
        open: true,
        ...(kind === 'kv' && initialKvView
          ? {
              kv: {
                ...defaultKvView(false),
                contextId:
                  this.analysisKv
                    .data()
                    ?.contexts.find(
                      (context) =>
                        context.turn === turn &&
                        context.moment === (stage === 'prefill' ? 'prefill_post' : 'terminal'),
                    )?.id ?? '',
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
              trend: {
                ...this.trendState(turn),
                metric: kind === 'distribution' ? 'entropy' : 'logit_rmse',
                open: true,
              },
            }
          : {}),
      },
    ]);
    const panel = this.analysisPanels().at(-1)!;
    const selected = this.selection();
    if (
      !selected ||
      selected.turn !== turn ||
      (selected.kind === 'stage'
        ? selected.stage !== stage
        : (selected.kind === 'input' ? 'prefill' : 'decode') !== stage)
    )
      this.selection.set({kind: 'stage', turn, stage, side: 'target'});
    this.setGroup('analyses', true);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.analysisOverview.set(panel);
    this.details.set(true);
    if (this.isExpandedStage(turn, stage)) {
      this.expandedAnalysis.set(panel);
      afterNextRender(() => this.selectExpandedAnalysis(panel), {injector: this.injector});
    } else this.locateAnalysis(panel);
  }
  private leaveAnalysis(panel: StageAnalysisPanel) {
    if (this.analysisHover()?.panelId === panel.id) this.analysisHover.set(null);
    const ownsMetric = this.metricInspector()?.panelId() === panel.id;
    const ownsTokenMetric = this.tokenMetricInspector()?.id === panel.id;
    const ownsOverview = this.analysisOverview()?.id === panel.id;
    if (!ownsMetric && !ownsTokenMetric && !ownsOverview) return;

    // Analysis focus is a stage range, even when it contains just one position.
    // Leaving the panel must not turn that focus into a direct token selection.
    const tokenRange = ownsTokenMetric ? this.tokenMetricRange() : null;
    const range = ownsOverview
      ? this.analysisOverviewRange(panel)
      : tokenRange?.panelId === panel.id
        ? tokenRange
        : this.selectionRangeFor(panel);
    if (ownsMetric) this.metricInspector.set(null);
    if (ownsTokenMetric) {
      this.tokenMetricInspector.set(null);
      this.tokenMetricRange.set(null);
    }
    if (ownsOverview) this.analysisOverview.set(null);
    this.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      ...(range ? {range: {start: range.start, end: range.end}} : {}),
    });
  }
  toggleAnalysis(panel: StageAnalysisPanel) {
    if (!panel.open) {
      const range = this.selectionRangeFor(panel);
      if (range && this.metricInspector())
        this.selection.set({
          kind: 'stage',
          turn: panel.turn,
          stage: panel.stage,
          side: 'target',
          range,
        });
      this.metricInspector.set(null);
      this.tokenMetricInspector.set(null);
      this.analysisOverview.set(panel);
      if (!range)
        this.selection.set({kind: 'stage', turn: panel.turn, stage: panel.stage, side: 'target'});
      this.details.set(true);
      this.locateAnalysis(panel);
      return;
    }
    if (panel.open) this.leaveAnalysis(panel);
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: !p.open} : p)),
    );
  }
  removeAnalysis(panel: StageAnalysisPanel) {
    this.leaveAnalysis(panel);
    if (this.expandedAnalysis()?.id === panel.id) this.exitExpandedAnalysis();
    if (this.metricInspector()?.panelId() === panel.id) this.metricInspector.set(null);
    if (this.tokenMetricInspector()?.id === panel.id) this.tokenMetricInspector.set(null);
    this.analysisPanels.update((panels) => panels.filter((p) => p.id !== panel.id));
  }
  inspectAnalysisOverview(panel: StageAnalysisPanel) {
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.analysisHover.set(null);
    this.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      range: this.analysisOverviewRange(panel),
    });
    this.analysisOverview.set(panel);
    this.details.set(true);
  }
  inspectMetric(panel: KvPanel | GraphMetricPanel) {
    const config = this.analysisPanels().find((p) => p.id === panel.panelId());
    if (config?.open === false && panel instanceof GraphMetricPanel && panel.inspectionMode())
      return;
    this.analysisOverview.set(null);
    this.tokenMetricInspector.set(null);
    this.metricInspector.set(panel);
    if (config)
      this.selection.set({kind: 'stage', turn: config.turn, side: 'target', stage: config.stage});
    this.details.set(true);
  }
  private analysisPointer: {x: number; y: number; moved: boolean} | null = null;
  analysisPointerDown(event: PointerEvent, panel: StageAnalysisPanel) {
    this.analysisPointer =
      event.button === 0 ? {x: event.clientX, y: event.clientY, moved: false} : null;
    if (event.button === 0 && this.overviewPanel()?.id !== panel.id)
      this.inspectAnalysisOverview(panel);
  }
  analysisPointerMove(event: PointerEvent) {
    const pointer = this.analysisPointer;
    if (
      pointer &&
      event.buttons !== 0 &&
      Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 4
    )
      pointer.moved = true;
  }
  analysisPointerCancel() {
    if (this.analysisPointer) this.analysisPointer.moved = true;
  }
  analysisBackgroundClick(event: MouseEvent, panel: StageAnalysisPanel) {
    if (
      (event.target as Element).closest(
        'kv-range-chart, kv-channel-chart, kv-scatter-chart, .results, .details, analysis-heatmap-controls, analysis-layer-legend, button, a, input, select, textarea, label, [role="button"], [role="dialog"], [role="menu"], [role="listbox"], [role="option"], [contenteditable="true"]',
      )
    )
      return;
    this.inspectAnalysisStage(panel, event);
  }
  inspectAnalysisStage(panel: StageAnalysisPanel, event: MouseEvent) {
    event.stopPropagation();
    const pointer = this.analysisPointer;
    if (
      event.button !== 0 ||
      event.defaultPrevented ||
      (pointer &&
        (pointer.moved || Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 4)) ||
      window.getSelection()?.toString()
    )
      return;
    this.analysisPointer = null;
    // Clear the plotted selection while retaining the active analysis and its viewport.
    this.overviewKv(panel)?.clearElementSelection();
    this.overviewFinalLogits(panel)?.clearElementSelection();
    const graph = this.overviewGraph(panel);
    graph?.inspection()?.clearElementSelection();
    graph?.selected.set(null);
    this.inspectAnalysisOverview(panel);
  }
  private analysisOverviewRange(panel: StageAnalysisPanel): {start: number; end: number} {
    const kv = panel.kind === 'kv' ? this.overviewKv(panel) : undefined;
    const graph =
      panel.kind === 'graph' || panel.kind === 'activations'
        ? this.overviewGraph(panel)
        : undefined;
    const position = kv?.tensorView()
      ? kv.tensorPosition()
      : graph?.inspectionMode()
        ? (graph.inspection()?.position() ?? null)
        : null;
    return position !== null
      ? {start: position, end: position + 1}
      : this.analysisViewportRange(panel);
  }
  private readonly analysisViewports = new Map<string, {start: number; end: number}>();
  private syncAnalysisViewport() {
    const panel = this.overviewPanel();
    if (!panel) return;
    const range = this.analysisOverviewRange(panel);
    const previous = this.analysisViewports.get(panel.id);
    this.analysisViewports.set(panel.id, range);
    const changed = previous && (previous.start !== range.start || previous.end !== range.end);
    const overview = this.analysisOverview()?.id === panel.id;
    untracked(() => {
      // A new viewport replaces the old bin even when that bin still fits inside it.
      if (changed) this.inspectAnalysisOverview(panel);
      else if (overview) this.syncAnalysisOverview(panel, range);
    });
  }
  private syncAnalysisOverview(panel: StageAnalysisPanel, range: {start: number; end: number}) {
    const selected = this.selection();
    if (
      selected?.kind === 'stage' &&
      selected.turn === panel.turn &&
      selected.stage === panel.stage &&
      selected.range?.start === range.start &&
      selected.range?.end === range.end
    )
      return;
    this.selection.set({
      kind: 'stage',
      turn: panel.turn,
      stage: panel.stage,
      side: 'target',
      range: {start: range.start, end: range.end},
    });
  }
  private analysisViewportRange(panel: StageAnalysisPanel): {start: number; end: number} {
    const current = this.analysisPanels().find((candidate) => candidate.id === panel.id) ?? panel;
    if (current.kind === 'kv')
      return this.overviewKv(current)?.visibleRange() ?? this.entireStageRange(current);
    if (current.kind === 'graph' || current.kind === 'activations')
      return this.overviewGraph(current)?.range() ?? this.entireStageRange(current);
    const logits = this.overviewFinalLogits(current);
    if (logits) return {start: logits.position(), end: logits.position() + 1};
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
          const range = this.analysisOverview() ? null : this.selectionRangeFor(panel);
          const selection: KvRangeSelection | null = range
            ? {
                ...range,
                layer:
                  this.metricInspector()?.panelId() === panel.id &&
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
  setPanelTrend(panel: StageAnalysisPanel, trend: TokenTrendState) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, trend} : p)),
    );
  }
  private capturePanels() {
    return this.analysisPanels().map((p) => {
      const live = this.kvPanels().find((kv) => kv.panelId() === p.id);
      const graph = this.graphPanels().find((g) => g.panelId() === p.id);
      return live ? {...p, kv: live.captureView()} : graph ? {...p, graph: graph.captureView()} : p;
    });
  }
  private readonly injector = inject(Injector);
  private readonly dialog = inject(MatDialog);
  async readFullText(turn: number, side: string, kind: 'input' | 'output' | 'thinking') {
    const {FullTextDialog} = await import('../../../shared/full_text_dialog/full_text_dialog');
    const capture = this.captured(turn, side),
      serialized =
        kind === 'input' && this.workspace.mode() === 'Debug'
          ? capture?.serialized_input
          : undefined;
    this.dialog.open(FullTextDialog, {
      width: '760px',
      maxWidth: 'calc(100vw - 24px)',
      data: {
        title:
          (side === 'ref' ? 'Reference' : 'Target') +
          ' · ' +
          (kind === 'input'
            ? serialized !== undefined
              ? 'Serialized input'
              : 'User message'
            : kind === 'thinking'
              ? 'Thinking'
              : 'Model response'),
        text:
          kind === 'thinking'
            ? capturedPhaseText(capture, 'thinking')
            : (serialized ??
              capture?.[kind] ??
              (kind === 'input'
                ? this.liveGeneration()?.prompt
                : this.liveGeneration()?.output[side]) ??
              ''),
      },
      ariaLabel: 'Read full captured text',
    });
  }
  private readonly contextKey =
    inject(WorkspaceStateService).selectedId() + ':' + inject(WorkspaceStateService).activeChat();
  private readonly element = inject(ElementRef<HTMLElement>);
  readonly state = inject(ReportStateService);
  private readonly api = inject(ReportApiService);
  /** Keyed by `${captureId}:${turn}` so a capture change cannot serve another capture's snapshots. */
  readonly kvSnapshots = signal(new Map<string, KvSnapshot[]>());
  readonly kvLoadErrors = signal(new Map<string, string>());
  private snapshotCapture: string | null = null;
  private snapshotKey(turn: number, capture = this.state.captureId()) {
    return (capture ?? '') + ':' + turn;
  }
  private readonly analysis = new TokenAnalysisController(this.api);
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
  readonly selectedAnalysis = computed(() => {
    const pair = this.detailPair();
    return pair ? this.tokenAnalysis().get(pairKey(pair.turn, pair)) : undefined;
  });
  readonly workspace = inject(WorkspaceStateService);
  private readonly zone = inject(NgZone);
  private virtualizer: FlowVirtualizer | null = null;
  private composerObserver: ResizeObserver | null = null;
  private followLive = true;
  private resumeAtTail = true;
  trackReadingIntent(event: WheelEvent | KeyboardEvent | PointerEvent) {
    if (
      (event instanceof WheelEvent && event.deltaY < 0) ||
      (event instanceof KeyboardEvent && ['ArrowUp', 'PageUp', 'Home'].includes(event.key))
    ) {
      this.followLive = false;
      this.resumeAtTail = false;
    }
    if (
      (event instanceof WheelEvent && event.deltaY > 0) ||
      (event instanceof KeyboardEvent && ['ArrowDown', 'PageDown', 'End'].includes(event.key))
    ) {
      this.resumeAtTail = true;
      this.readingScrolled();
    }
    if (event instanceof PointerEvent) {
      const r = this.element.nativeElement.querySelector('.reading')?.getBoundingClientRect();
      if (r && event.clientX >= r.right - 16) {
        this.followLive = false;
        this.resumeAtTail = true;
      }
    }
  }
  readingScrolled() {
    this.clearTokenFeedback();
    const r = this.element.nativeElement.querySelector('.reading');
    if (this.resumeAtTail && r && r.scrollHeight - r.clientHeight - r.scrollTop < 80)
      this.followLive = true;
  }
  // Shared line breaks: every aligned pair takes the width of its wider side on both runtimes.
  private readonly destroyRef = inject(DestroyRef);
  private readonly layoutTick = signal(0);
  private readonly pairWidthSync = afterRenderEffect(() => {
    // Everything that changes token widths or the column width re-runs the sync.
    this.aligned();
    this.boundaries();
    this.color();
    this.alignment();
    this.panelWidth();
    this.collapsed();
    this.turns();
    this.layoutTick();
    if (this.workspace.mode() === 'Debug') syncPairWidths(this.element.nativeElement);
  });
  private readonly readingResize = afterNextRender(() => {
    const reading = this.element.nativeElement.querySelector('.reading');
    if (!reading || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => this.layoutTick.update((n) => n + 1));
    observer.observe(reading);
    this.destroyRef.onDestroy(() => observer.disconnect());
    document.fonts?.ready.then(() => this.layoutTick.update((n) => n + 1));
  });
  infoTokenHtml(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token), false) : '∅';
  }
  infoPairText(pair: TokenPair) {
    return ['target', 'ref']
      .map((side) => {
        const token = this.pairToken(pair, side);
        return token ? capturedTokenLabel(token) : '∅';
      })
      .join(' / ');
  }
  activateTokenLocator(event: KeyboardEvent, pair: TokenPair) {
    if (event.target !== event.currentTarget || !['Enter', ' '].includes(event.key)) return;
    event.preventDefault();
    this.locateInfoToken(pair);
  }
  infoTokenTitle(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token
      ? JSON.stringify(capturedTokenLabel(token)) + (token.id != null ? ` · ID ${token.id}` : '')
      : 'No token at this aligned position';
  }
  /** Prefill rows over aligned pairs; null when either side lacks token kinds. */
  inputPairRows(turn: number) {
    return capturedInputPairRows(this.captured(turn, 'ref'), this.captured(turn, 'target'));
  }
  pairToken(pair: TokenPair, side: string) {
    return side === 'ref' ? pair.ref : pair.target;
  }
  inputTokenText(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? capturedTokenLabel(token) : '∅';
  }
  /** Prefill text follows the same whitespace rule as Decode tokens. */
  inputTokenHtml(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token)) : '∅';
  }
  /** A token's text for labels: an invisible-only token is named ("line feed × 2"), a gap is ∅. */
  spokenToken(token: CapturedToken | undefined) {
    return token ? spokenTokenText(capturedTokenLabel(token)) : '∅';
  }
  private readonly analysisHover = signal<{panelId: string; range: KvRangeSelection} | null>(null);
  private readonly selectionSide = signal('target');
  hoverAnalysis(panelId: string, range: KvRangeSelection | null) {
    if (range) this.analysisHover.set({panelId, range});
    else if (this.analysisHover()?.panelId === panelId) this.analysisHover.set(null);
  }
  private readonly tokenLinkSync = afterRenderEffect(() => this.refreshTokenLinks());
  private readonly tokenLinkMounts = afterNextRender(() => {
    const observer = new MutationObserver(() => this.refreshTokenLinks());
    observer.observe(this.element.nativeElement, {childList: true, subtree: true});
    this.destroyRef.onDestroy(() => observer.disconnect());
  });
  private refreshTokenLinks() {
    const panels = this.analysisPanels(),
      hovered = this.analysisHover(),
      inspector = this.metricInspector();
    const hoveredPanel = panels.find((p) => p.id === hovered?.panelId && p.open);
    const selectedPanel = panels.find((p) => p.id === inspector?.panelId() && p.open);
    const range = inspector instanceof KvPanel ? inspector.inspectedRange() : inspector?.selected();
    const selection = this.selection(),
      selectedSide = this.selectionSide();
    const inputPairs = new Map<number, Map<number, TokenPair>>();
    for (const node of Array.from(
      (this.element.nativeElement as HTMLElement).querySelectorAll<HTMLElement>(HOVER_TARGET),
    )) {
      const turn = Number(node.dataset['turn']),
        row = Number(node.dataset['row']),
        side = node.dataset['side']!;
      const input = node.dataset['kind'] === 'input',
        stage = input ? 'prefill' : 'decode';
      let token: CapturedToken | undefined;
      if (input) {
        if (!inputPairs.has(turn))
          inputPairs.set(
            turn,
            new Map(
              (this.inputPairRows(turn) ?? [])
                .flatMap((r) => [...r.leading, ...r.text, ...r.trailing])
                .map((p) => [p.index, p]),
            ),
          );
        const pair = inputPairs.get(turn)?.get(row);
        token = pair ? this.pairToken(pair, side) : undefined;
      } else {
        const pair = this.rows(turn)[row];
        token = pair ? this.pairToken(pair, side) : undefined;
      }
      const info = this.state.session()?.turns.find((t) => t.n === turn);
      const start =
        info?.step_start == null
          ? null
          : info.step_start - (input ? (info.prefill_tokens ?? 0) : 0);
      const position = token && start !== null ? start + token.step : null;
      const coordinate = this.alignment() === 'content' && start !== null ? start + row : position;
      const matches = (
        panel: StageAnalysisPanel | undefined,
        interval: KvRangeSelection | null | undefined,
      ) => {
        return (
          !!panel &&
          panel.turn === turn &&
          panel.stage === stage &&
          !!interval &&
          coordinate !== null &&
          coordinate >= interval.start &&
          coordinate < interval.end
        );
      };
      const selectedPair =
        !!selection &&
        selection.turn === turn &&
        selection.kind === (input ? 'input' : 'token') &&
        'index' in selection &&
        selection.index === row;
      const origin = selectedPair && side === selectedSide;
      node.classList.toggle('selection-origin', origin);
      node.classList.toggle('linked-token-selected', selectedPair && !origin);
      const inspectedStage = selectedPanel?.turn === turn && selectedPanel.stage === stage;
      const selectedStage =
        selection?.kind === 'stage' && selection.turn === turn && selection.stage === stage;
      const focusRange = inspectedStage && range ? range : selectedStage ? selection.range : null;
      node.classList.toggle(
        'linked-token-focus',
        !selectedPair &&
          (inspectedStage || selectedStage) &&
          (!focusRange ||
            (coordinate !== null && coordinate >= focusRange.start && coordinate < focusRange.end)),
      );
      node.classList.toggle('linked-token-hover', matches(hoveredPanel, hovered?.range));
    }
  }
  // Pointer feedback frames the hovered token and its aligned partner on the other runtime,
  // as in the design study. It never changes the selection or opens a detail surface.
  private tokenFeedback: Element[] = [];
  private hoverTarget(target: EventTarget | null) {
    return target instanceof Element ? target.closest<HTMLElement>(HOVER_TARGET) : null;
  }
  @HostListener('pointerover', ['$event']) tokenPointerOver(event: PointerEvent) {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null)) this.showTokenFeedback(token);
  }
  @HostListener('pointerout', ['$event']) tokenPointerOut(event: PointerEvent) {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null)) this.clearTokenFeedback();
  }
  @HostListener('focusin', ['$event']) tokenFocusIn(event: FocusEvent) {
    const token = this.hoverTarget(event.target);
    if (token) this.showTokenFeedback(token);
  }
  @HostListener('focusout', ['$event']) tokenFocusOut(event: FocusEvent) {
    if (this.hoverTarget(event.target)) this.clearTokenFeedback();
  }
  private clearTokenFeedback() {
    for (const token of this.tokenFeedback) token.classList.remove('pair-hover', 'hover-origin');
    this.tokenFeedback = [];
  }
  private showTokenFeedback(token: HTMLElement) {
    this.clearTokenFeedback();
    if (this.workspace.mode() !== 'Debug') return;
    const {turn, row, kind} = token.dataset;
    const partners = this.element.nativeElement.querySelectorAll(
      `[data-turn="${turn}"][data-row="${row}"]` +
        (kind ? `[data-kind="${kind}"]` : ':not([data-kind])'),
    );
    const fork = kind ? null : this.forks().get(Number(turn));
    const stepPartner =
      fork?.filled && fork.partner !== null && String(fork.index) === row
        ? this.element.nativeElement.querySelectorAll(
            `[data-turn="${turn}"][data-row="${fork.partner}"][data-side="${fork.filled}"]:not([data-kind])`,
          )
        : [];
    this.tokenFeedback = [token, ...partners, ...stepPartner].filter(
      (el, i, all) => all.indexOf(el) === i,
    );
    for (const el of this.tokenFeedback) el.classList.add('pair-hover');
    token.classList.add('hover-origin');
  }
  private nodeNavigator: FlowNodeNavigator | null = null;
  nodeIndex(turn: number, stage: string) {
    return this.turns().findIndex((t) => t.n === turn) * 2 + (stage === 'decode' ? 1 : 0);
  }
  readonly virtualModels = computed(() => {
    const models = new Map(
      this.turns().map((t) => [
        t.n,
        virtualConversation(
          this.captured(t.n, 'ref'),
          this.captured(t.n, 'target'),
          this.alignment(),
          this.workspace.mode() === 'Debug',
        ),
      ]),
    );
    const job = this.liveGeneration();
    if (
      job &&
      Math.max(job.prompt?.length ?? 0, ...Object.values(job.output).map((t) => t.length)) > 16000
    )
      models.set(job.turn, streamingConversation(job.prompt ?? '', job.output));
    return models;
  });
  virtualOutput(turn: number) {
    const model = this.virtualModels().get(turn);
    return (
      (model?.rows.length ?? 0) > 2048 &&
      (this.workspace.mode() === 'Debug' || !!model?.responseTextComplete)
    );
  }
  virtualPhase(turn: number, phase: TokenPhase) {
    return (
      this.virtualOutput(turn) &&
      ['ref', 'target'].some((side) => hasPhaseTokens(this.captured(turn, side), phase))
    );
  }
  virtualInput(turn: number) {
    const t = this.virtualModels().get(turn);
    return Math.max(t?.inputTokens.ref.length ?? 0, t?.inputTokens.target.length ?? 0) > 2048;
  }
  readonly usesVirtual = computed(() =>
    [...this.virtualModels().keys()].some((n) => this.virtualInput(n) || this.virtualOutput(n)),
  );
  readonly longAlignmentUnavailable = computed(
    () =>
      this.workspace.mode() === 'Chat' &&
      [...this.virtualModels().values()].some((t) => t.alignmentUnavailable),
  );
  readonly runs = computed(
    () =>
      this.state.session()?.runs ??
      this.workspace.runtime.selected()?.runs ??
      this.workspace.sessions.items().find((s) => s.id === this.workspace.selectedId())?.runs ??
      [],
  );
  readonly turns = computed(() => this.state.session()?.turns ?? []);
  private readonly stageTokenSummaries = computed(() =>
    summarizeStageTokens(
      [...this.turns()]
        .sort((a, b) => a.n - b.n)
        .flatMap((turn) =>
          (['prefill', 'decode'] as const).map((stage) => {
            const pairs = this.compactPairs(turn.n, stage);
            const complete = (['ref', 'target'] as const).every((side) => {
              const capture = this.captured(turn.n, side);
              const tokens = stage === 'prefill' ? capture?.input_tokens : capture?.tokens;
              return (
                tokens !== undefined && pairs.filter((pair) => pair[side]).length === tokens.length
              );
            });
            return {turn: turn.n, stage, pairs, complete};
          }),
        ),
    ),
  );
  readonly conversationStages = computed(() =>
    this.stageTokenSummaries().map((summary) => ({
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
              const capture = this.captured(summary.turn, side);
              const tokens = summary.stage === 'prefill' ? capture?.input_tokens : capture?.tokens;
              const count = tokens?.length ?? this.stageCount(summary.turn, side, summary.stage);
              return typeof count === 'number' ? formatCount(count) : count;
            })
            .join(' / '),
          description:
            'Actual tokens in the full stage, including template and special tokens; alignment gaps are excluded.',
        },
        {
          label: 'History Match',
          value: summary.historyMatch === null ? '-' : summary.historyMatch ? 'Yes' : 'No',
          description:
            'Whether all tokens in preceding stages match. Excludes this stage; - means no history or incomplete data.',
        },
        {
          label: 'Token Match Rate',
          value: formatPercent(summary.matchRate, {placeholder: '-'}),
          description:
            'Matched aligned positions / all aligned positions in this stage. Includes template and special tokens; one-sided gaps are mismatches.',
        },
        {
          label: 'Cumulative Match Rate',
          value: formatPercent(summary.cumulativeMatchRate, {placeholder: '-'}),
          description:
            'Matched aligned positions / all aligned positions from the first stage through this stage, weighted by position count. Incomplete data is shown as -.',
        },
        {
          label: 'First Divergence',
          value: summary.firstDivergence === null ? '-' : summary.firstDivergence ? 'Yes' : 'No',
          description:
            'Whether this stage contains the first observed token mismatch in the conversation.',
        },
      ],
    })),
  );
  readonly thinkingExpanded = signal(new Set<string>());
  readonly firstMismatchStage = computed(() => {
    const first = this.stageTokenSummaries().find((summary) => summary.firstDivergence);
    return first ? {turn: first.turn, stage: first.stage} : null;
  });
  isFirstMismatchStage(turn: number, stage: string) {
    const first = this.firstMismatchStage();
    return first?.turn === turn && first.stage === stage;
  }
  readonly liveGeneration = computed(() => {
    const job = this.workspace.runtime.job();
    if (
      this.workspace.mode() !== 'Chat' ||
      job?.operation !== 'generate' ||
      job.session_id !== this.workspace.context().captureRecordId
    )
      return null;
    const captured = this.turns().some((t) => t.n === job.turn);
    return job.status === 'completed' && captured ? null : job;
  });
  readonly composer = new ConversationComposerController(this.workspace);
  thinkingOpen(turn: number, run: string) {
    return this.thinkingExpanded().has(`${turn}:${run}`);
  }
  toggleThinking(turn: number, run: string) {
    const key = `${turn}:${run}`;
    this.thinkingExpanded.update((s) => {
      const n = new Set(s);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  }
  private readonly chatText = computed(() => {
    const result: Record<string, {text: string; mismatch: boolean}[]> = {};
    for (const turn of this.state.session()?.turns ?? []) {
      if (this.virtualOutput(turn.n)) continue;
      let rows: TokenPair[] = [];
      try {
        rows = alignTokens(
          this.captured(turn.n, 'ref')?.tokens,
          this.captured(turn.n, 'target')?.tokens,
          'content',
          this.captured(turn.n, 'ref')?.alignments?.content,
        );
      } catch {
        continue;
      }
      for (const run of ['ref', 'target']) {
        const parts = rows.flatMap((r) => {
          const t = run === 'ref' ? r.ref : r.target;
          return t?.text != null && (t.phase ?? 'response') === 'response'
            ? [{text: t.text, mismatch: r.match === false}]
            : [];
        });
        if (parts.map((t) => t.text).join('') === this.captured(turn.n, run)?.output)
          result[`${turn.n}:${run}`] = parts;
      }
    }
    return result;
  });
  chatTokens(turn: number, run: string) {
    return this.chatText()[`${turn}:${run}`] ?? [];
  }

  readonly formula = signal('NOT token_match');
  readonly predicate = computed(() => compileQuery(this.formula()));

  readonly alignment = signal<AlignmentMode>('content');
  readonly selection = signal<ConversationSelection | null>(null);
  readonly tokenSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'token' ? selected : null;
  });
  readonly selectedStage = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'stage' ? selected : null;
  });
  /** Sidebar context is the same for a directly selected pair and a Stage range. */
  selectionInfo(): {
    turn: number;
    stage: 'prefill' | 'decode';
    index?: number;
    range?: {start: number; end: number};
  } | null {
    const selected = this.selection();
    if (!selected) return null;
    if (selected.kind === 'stage') return {...selected, index: undefined};
    const stage = selected.kind === 'input' ? 'prefill' : 'decode';
    const pair = this.compactPairs(selected.turn, stage).find((p) => p.index === selected.index);
    const info = this.turns().find((t) => t.n === selected.turn);
    const token = pair?.target ?? pair?.ref;
    const position =
      info?.step_start != null && token
        ? info.step_start -
          (stage === 'prefill' ? (info.prefill_tokens ?? 0) : 0) +
          (this.alignment() === 'content' ? selected.index : token.step)
        : null;
    return {
      turn: selected.turn,
      stage,
      index: selected.index,
      range: position === null ? undefined : {start: position, end: position + 1},
    };
  }
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
  readonly tokenInfoPageSize = 10;
  readonly tokenInfoPageIndex = linkedSignal({
    source: () => JSON.stringify([this.selectionInfo(), this.alignment()]),
    computation: () => 0,
  });
  readonly infoTokenPage = computed(() => {
    const pairs = this.infoTokenPairs();
    const pages = Math.max(1, Math.ceil(pairs.length / this.tokenInfoPageSize));
    const index = Math.min(Math.max(0, this.tokenInfoPageIndex()), pages - 1);
    const start = index * this.tokenInfoPageSize;
    return {index, pages, start, pairs: pairs.slice(start, start + this.tokenInfoPageSize)};
  });
  changeTokenInfoPage(delta: number) {
    const {index, pages} = this.infoTokenPage();
    this.tokenInfoPageIndex.set(Math.min(Math.max(0, index + delta), pages - 1));
  }
  infoTokenPosition(pair: TokenPair, side: string) {
    const selected = this.selectionInfo();
    const token = this.pairToken(pair, side);
    if (!token) return '∅';
    const info = this.turns().find((t) => t.n === selected?.turn);
    if (!selected || info?.step_start == null) return '-';
    return String(
      tokenPosition(
        token,
        info.step_start - (selected.stage === 'prefill' ? (info.prefill_tokens ?? 0) : 0),
      ),
    );
  }
  infoPairPosition(pair: TokenPair) {
    const selection = this.selectionInfo();
    const info = this.turns().find((t) => t.n === selection?.turn);
    if (!selection || info?.step_start == null) return '-';
    const start =
      info.step_start - (selection.stage === 'prefill' ? (info.prefill_tokens ?? 0) : 0);
    return String(
      start +
        (this.alignment() === 'content'
          ? pair.index
          : ((pair.target ?? pair.ref)?.step ?? pair.index)),
    );
  }
  readonly inputSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'input' ? selected : null;
  });
  /** The selected admitted Prefill pair, found across its message rows. */
  readonly selectedInputPair = computed(() => {
    const s = this.inputSelection();
    if (!s) return null;
    for (const row of this.inputPairRows(s.turn) ?? [])
      for (const pair of [...row.leading, ...row.text, ...row.trailing])
        if (pair.index === s.index) return {...pair, turn: s.turn};
    return null;
  });
  /** A generated token's content kind, read against its own run's admitted template tokens. */
  tokenContentFor(turn: number, side: string, token: CapturedToken | undefined) {
    return capturedTokenContent(this.captured(turn, side), token);
  }
  /** What an admitted token is: message text, a chat-template marker or a special token. */
  inputTokenContent(token: CapturedToken | undefined) {
    return !token
      ? '∅'
      : token.kind === 'template'
        ? 'Chat template'
        : token.kind === 'special'
          ? 'Special token'
          : 'Input context';
  }
  isInputSelected(turn: number, index: number) {
    const s = this.inputSelection();
    return !!s && s.turn === turn && s.index === index;
  }
  selectInput(turn: number, index: number, event?: Event) {
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
  readonly colorLabel = colorLabel;
  readonly heatPalette = heatPalette;
  readonly colorMax = TOKEN_COLOR_MAX;
  readonly metricAvailability = computed(() =>
    Object.fromEntries(
      TOKEN_METRICS.map((m) => [
        m.key,
        [...this.tokenAnalysis().values()].some((p) => p.metrics[m.key]?.value != null),
      ]),
    ),
  );
  readonly numericColor = computed(() => this.color() !== 'none' && this.color() !== 'token_match');
  readonly colorMaximum = computed(() =>
    this.numericColor() ? this.colorMax[this.color() as keyof typeof this.colorMax] : null,
  );
  readonly colorUnit = computed(() =>
    this.color() === 'relative_l2'
      ? '%'
      : this.color() === 'js'
        ? 'bits'
        : this.color() === 'kl'
          ? 'nats'
          : this.color() === 'norm_ratio'
            ? '×'
            : '',
  );
  tokenHeat(turn: number, pair: TokenPair | undefined | null, side: string) {
    return side === 'target' && pair && this.numericColor()
      ? heatLevel(this.color(), this.tokenAnalysis().get(pairKey(turn, pair)))
      : null;
  }
  readonly collapsed = signal(new Set<string>());
  readonly bookmarks = this.workspace.bookmarks;
  readonly bookmarkNotice = signal('');
  readonly groups = signal<Record<string, boolean>>({});
  readonly trends = signal<TokenDiffViewState['trends']>({});
  trendState(turn: number): TokenTrendState {
    return (
      this.trends()[turn + ':' + this.alignment()] ?? {
        open: false,
        metric: 'kl',
        start: 0,
        end: Math.max(1, this.rows(turn).length - 1),
      }
    );
  }
  setTrend(turn: number, setting: TokenTrendState) {
    this.trends.update((s) => ({...s, [turn + ':' + this.alignment()]: setting}));
  }
  tokenPanelOpen(turn: number) {
    return this.panelsFor(turn, 'decode').some((p) => p.kind === 'tokens' && p.open);
  }
  toggleTrend(turn: number) {
    const existing = this.panelsFor(turn, 'decode').find((p) => p.kind === 'tokens');
    if (existing) this.toggleAnalysis(existing);
    else this.addAnalysis(turn, 'decode', 'tokens');
  }
  analysisLabel(panel: StageAnalysisPanel) {
    if (panel.kind === 'kv') return 'KV cache';
    if (canonicalAnalysisKind(panel.kind) === 'activations') return 'Intermediate Tensors';
    return panel.kind === 'distribution' ? 'Sampling Distribution' : 'Final Logits';
  }
  analysisInstanceLabel(panel: StageAnalysisPanel) {
    const current = this.analysisPanels().find((p) => p.id === panel.id) ?? panel;
    const prefix: Record<StageAnalysisPanel['kind'], string> = {
      kv: 'KV Cache Analysis',
      graph: 'Intermediate Tensor Analysis',
      activations: 'Intermediate Tensor Analysis',
      tokens: 'Final Logits Analysis',
      distribution: 'Sampling Distribution Analysis',
    };
    const peers = this.panelsFor(current.turn, current.stage).filter(
      (p) => canonicalAnalysisKind(p.kind) === canonicalAnalysisKind(current.kind),
    );
    const number =
      current.instanceNumber ?? Math.max(1, peers.findIndex((p) => p.id === current.id) + 1);
    return `${prefix[current.kind]} ${number - 1}`;
  }
  analysisViewProperties(panel: StageAnalysisPanel): ExplorerInfoItem[] {
    const current = this.analysisPanels().find((item) => item.id === panel.id) ?? panel;
    const kv = this.overviewKv(current);
    const graph = this.overviewGraph(current);
    const activationInspection =
      canonicalAnalysisKind(current.kind) === 'activations' &&
      (graph?.inspectionMode() ?? current.graph?.mode === 'inspection');
    const activationState = graph?.inspection()?.captureView() ?? current.graph?.inspection;
    const logitsInspection = current.kind === 'tokens' && current.trend?.logitView === 'inspection';
    const inspection =
      logitsInspection ||
      activationInspection ||
      (current.kind === 'kv' && (kv?.tensorView() ?? current.kv?.tensor?.active ?? false));
    const plot = activationInspection
      ? activationState?.plot
      : (kv?.tensorPlot() ?? current.kv?.tensor?.plot);
    const view = kv?.view() ?? graph?.view() ?? current.kv?.view ?? current.graph?.view;
    const properties: ExplorerInfoItem[] = [
      {
        label: 'Mode',
        value: inspection
          ? 'Tensor Inspection'
          : current.kind === 'distribution'
            ? 'Distribution Comparison'
            : 'Tensor Comparison',
      },
      {
        label: 'Visualization',
        value: logitsInspection
          ? current.trend?.logitPlot === 'scatter'
            ? 'Scatter plot'
            : 'Line chart'
          : inspection
            ? plot === 'scatter'
              ? 'Scatter plot'
              : 'Line chart'
            : view === 'graph'
              ? 'Graph Diff'
              : view === 'heatmap' ||
                  (!view && ['kv', 'graph', 'activations'].includes(current.kind))
                ? 'Heatmap'
                : 'Line chart',
      },
    ];
    if (logitsInspection) {
      properties.push({
        label: 'Token Position',
        value: String(
          this.overviewFinalLogits(current)?.position() ?? current.trend?.logitPosition ?? '-',
        ),
      });
    } else if (activationInspection) {
      properties.push(
        {label: 'Token Position', value: String(activationState?.position ?? '-')},
        {
          label: 'Tensor Index',
          value: `[Layer: ${activationState?.layer ?? '-'}, Sync point: ${tensorPointLabel(activationState?.anchor ?? 'output')}]`,
          expandable: true,
        },
      );
    } else if (inspection) {
      const tensor = current.kv?.tensor;
      properties.push(
        {label: 'Token Position', value: String(kv?.tensorPosition() ?? tensor?.position ?? '-')},
        {
          label: 'Tensor Index',
          value: `[Layer: ${kv?.tensorLayer() ?? tensor?.layer ?? '-'}, Head: ${kv?.tensorHead() ?? tensor?.head ?? '-'}]`,
          expandable: true,
        },
      );
    } else {
      const range =
        kv?.visibleRange() ??
        graph?.range() ??
        current.kv?.zoomRange ??
        current.graph?.range ??
        this.analysisViewportRange(current);
      const metric =
        kv?.metric() ??
        graph?.metric() ??
        current.kv?.metric ??
        current.graph?.metric ??
        current.trend?.metric ??
        (current.kind === 'distribution'
          ? 'entropy'
          : current.kind === 'tokens'
            ? 'logit_rmse'
            : 'cosine_similarity');
      const label =
        current.kind === 'kv'
          ? KV_METRICS.find((item) => item.id === metric)?.label
          : ['graph', 'activations'].includes(current.kind)
            ? TENSOR_METRICS.find((item) => item.id === metric || item.source === metric)?.label
            : [...TOKEN_METRICS, ...SIDE_TREND_METRICS, ...FINAL_LOGIT_METRICS].find(
                (item) => item.key === metric,
              )?.label;
      properties.push(
        {label: 'Token Range', value: `${range.start}–${range.end - 1}`},
        {
          label: 'Metric',
          value: label?.replace(/^(Logits|Per runtime) · /, '').replace(/ \(%\)$/, '') ?? metric,
          expandable: true,
        },
      );
    }
    return properties;
  }
  analysisConfig(panel: StageAnalysisPanel) {
    const kv = this.kvPanels().find((p) => p.panelId() === panel.id);
    if (kv)
      return `${kv.metricLabel()} · ${kv.view() === 'heatmap' ? 'Heatmap' : 'Line chart'} · Heads: ${kv.head()}`;
    const graph = this.graphPanels().find((p) => p.panelId() === panel.id);
    if (graph?.inspectionMode())
      return `Tensor Inspection · ${graph.inspection()?.plot() === 'scatter' ? 'Scatter plot' : 'Line chart'}`;
    if (graph)
      return `${graph.metrics.find((m) => m.id === graph.metric())?.label ?? graph.metric()} · ${graph.view() === 'graph' ? 'Graph Diff' : graph.view() === 'heatmap' ? 'Heatmap' : 'Line chart'} · Sync points: ${graph.anchor()}`;
    return `${panel.trend?.metric ?? ''} · Line chart`;
  }
  selectedPositionLabel() {
    const pair = this.selectedInputPair() ?? this.selectedPair();
    if (!pair) return '—';
    const info = this.turns().find((t) => t.n === pair.turn);
    const start =
      (info?.step_start ?? 0) - (this.selectedInputPair() ? (info?.prefill_tokens ?? 0) : 0);
    const ref = pair.ref ? tokenPosition(pair.ref, start) : null;
    const target = pair.target ? tokenPosition(pair.target, start) : null;
    return ref === target
      ? String(ref ?? '—')
      : `Reference: ${ref ?? '∅'} · Target: ${target ?? '∅'}`;
  }
  private selectionRangeFor(
    panel: Pick<StageAnalysisPanel, 'turn' | 'stage'> & Partial<Pick<StageAnalysisPanel, 'kind'>>,
  ) {
    const inspector = this.metricInspector();
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
    const selected = this.selection();
    if (!selected || selected.turn !== panel.turn) return null;
    if (selected.kind === 'stage')
      return selected.stage === panel.stage ? (selected.range ?? null) : null;
    if ((selected.kind === 'input' ? 'prefill' : 'decode') !== panel.stage) return null;
    const pair = this.selectedInputPair() ?? this.selectedPair();
    const token = pair?.target ?? pair?.ref;
    if (!token) return null;
    const info = this.turns().find((t) => t.n === panel.turn);
    const start =
      (info?.step_start ?? 0) -
      (panel.stage === 'prefill' ? (info?.prefill_tokens ?? 0) : 0) +
      (this.alignment() === 'content' ? pair!.index : token.step);
    return {start, end: start + 1};
  }
  locateInfoToken(pair: TokenPair) {
    const selected = this.selectionInfo();
    if (!selected) return;
    this.locateSelectedToken({
      kind: selected.stage === 'prefill' ? 'input' : 'token',
      turn: selected.turn,
      index: pair.index,
    });
  }
  private tokenLocationAnimations: Animation[] = [];
  locateSelectedToken(selected = this.selection()) {
    if (!selected || !('index' in selected)) return;
    const side = this.selectionSide() === 'ref' ? 'ref' : 'target';
    const stage = selected.kind === 'input' ? 'prefill' : 'decode';
    this.collapsed.update(
      (set) => new Set([...set].filter((k) => !k.startsWith(selected.turn + ':'))),
    );
    afterNextRender(
      () => {
        if (selected.kind === 'token' && this.virtualOutput(selected.turn)) {
          const row = this.virtualModels()
            .get(selected.turn)
            ?.rows.find((r) => r.index === selected.index);
          const runtime = row?.[side] != null ? side : side === 'ref' ? 'target' : 'ref';
          const step = row?.[runtime];
          if (step != null)
            this.zone.runOutsideAngular(() => {
              this.virtualizer?.mount();
              this.virtualizer?.jump(selected.turn, step, runtime, false);
            });
        }
        const root = this.element.nativeElement;
        const selector =
          '[data-turn="' +
          selected.turn +
          '"][data-row="' +
          selected.index +
          '"]' +
          (stage === 'prefill' ? '[data-kind="input"]' : ':not([data-kind])');
        const node =
          root.querySelector(selector + '[data-side="' + side + '"]') ??
          root.querySelector(selector);
        const reading = root.querySelector('.reading') as HTMLElement | null;
        const bounds = this.reading.visibleBounds();
        if (!node || !reading || !bounds) return;
        const rect = node.getBoundingClientRect();
        const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
        reading.scrollTo({
          top: reading.scrollTop + (rect.top + rect.bottom - bounds.top - bounds.bottom) / 2,
          behavior: reduced ? 'auto' : 'smooth',
        });
        this.tokenLocationAnimations?.forEach((animation) => animation.cancel());
        this.tokenLocationAnimations = Array.from(
          (root as HTMLElement).querySelectorAll<HTMLElement>(selector),
        ).map((token) => animateLocationFeedback(token, reduced));
      },
      {injector: this.injector},
    );
  }
  groupOpen(key: string) {
    return this.groups()[key] ?? true;
  }
  setGroup(key: string, value: boolean) {
    this.groups.update((groups) => ({...groups, [key]: value}));
  }
  readonly alignmentChoices = ['Context', 'Steps'];
  readonly alignmentDescriptions = {
    Context: 'Match output content',
    Steps: 'Match generation-step index',
  };
  readonly infoExpanded = signal(true);
  readonly panelBounds = PANEL_WIDTH;
  readonly panelWidth = signal(350);
  private scrollbar: ReturnType<typeof createConversationScrollbar> | null = null;
  private returnFocus: HTMLElement | null = null;
  private readonly reading = new ConversationReadingController({
    context: this.contextKey,
    positions: this.workspace.readingPositions,
    root: () => this.element.nativeElement,
    mode: () => this.workspace.mode(),
    virtualizer: () => this.virtualizer,
    afterRender: (work) =>
      afterNextRender(() => this.zone.runOutsideAngular(work), {injector: this.injector}),
    afterRestore: () => {
      this.nodeNavigator?.mount();
      this.scrollbar?.mount();
    },
  });
  private captureView(): TokenDiffViewState {
    return captureViewState({
      observation: this.currentObservation(),
      formula: this.formula(),
      alignment: this.alignment(),
      selection: this.selection(),
      identity: pairIdentity(this.selectedPair()),
      details: this.details(),
      boundaries: this.boundaries(),
      // Invisible characters are always drawn; the field keeps saved views in one shape.
      whitespace: true,
      color: this.color(),
      collapsed: [...this.collapsed()],
      thinking: [...this.thinkingExpanded()],
      infoExpanded: this.infoExpanded(),
      groups: {...this.groups()},
      panelWidth: this.panelWidth(),
      ...this.reading.capture(),
      trends: this.trends(),
      analysisPanels: this.capturePanels(),
      inspectedPanel: this.metricInspector()?.panelId() ?? this.tokenMetricInspector()?.id ?? null,
    });
  }
  private currentObservation(): ConversationObservation | null {
    const batch = this.state.session()?.batches.find((item) => item.batch === this.state.batchId());
    return batch ? {turn: batch.turn, phase: batch.phase} : null;
  }
  private saveView() {
    const view = this.layerGraphReturnView ?? this.captureView();
    this.workspace.conversationViews.set(
      this.contextKey,
      this.layerGraphReturnView
        ? view
        : {
            ...view,
            analysisPanels: [],
            inspectedPanel: null,
            analysisFindScope: '',
          },
    );
  }
  private rowsForRestore(turn: number, alignment: AlignmentMode): readonly TokenPair[] {
    const ref = this.captured(turn, 'ref'),
      target = this.captured(turn, 'target');
    if (Math.max(ref?.tokens?.length ?? 0, target?.tokens?.length ?? 0) > 2048) {
      const model = virtualConversation(ref, target, alignment, true);
      return model.alignmentUnavailable ? [] : model.pairs;
    }
    return alignTokens(ref?.tokens, target?.tokens, alignment, ref?.alignments?.content);
  }
  /** Both ordinary return and bookmarks cross the same validated restore path. */
  private applyView(
    value: unknown,
    source: 'return' | 'bookmark' | 'legacy',
    expected?: {turn: number; alignment: AlignmentMode},
  ): boolean {
    const prepared = prepareViewRestore(
      value,
      {
        hasTurn: (turn) => this.turns().some((item) => item.n === turn),
        rows: (turn, alignment) => this.rowsForRestore(turn, alignment),
      },
      source === 'return'
        ? {
            source,
            observation: this.currentObservation(),
          }
        : {source, expected: expected!},
    );
    if (prepared.status !== 'ready') {
      const message = {
        invalid: 'Saved settings are invalid or from an unsupported version.',
        'missing-turn': 'This saved turn is no longer available.',
        'missing-token': 'This saved token is no longer available.',
        'inconsistent-turn': 'Saved bookmark has inconsistent turn metadata.',
      }[prepared.status];
      this.bookmarkNotice.set(
        message +
          (source === 'return'
            ? ' Conversation defaults were retained.'
            : ' The bookmark was kept.'),
      );
      return false;
    }
    const view = prepared.state;
    this.formula.set(view.formula);
    this.alignment.set(view.alignment);
    this.selection.set(view.selection);
    this.details.set(view.details);
    this.boundaries.set(view.boundaries);
    this.color.set(view.color);
    this.collapsed.set(new Set(view.collapsed));
    this.thinkingExpanded.set(new Set(view.thinking));
    this.infoExpanded.set(view.infoExpanded);
    this.groups.set(view.groups);
    this.panelWidth.set(view.panelWidth);
    this.trends.set(view.trends);
    const panels =
      view.analysisPanels ??
      Object.entries(view.trends)
        .filter(([key, trend]) => key.endsWith(':' + view.alignment) && trend.open)
        .map(
          ([key, trend], index): StageAnalysisPanel => ({
            id: 'analysis-' + (index + 1),
            turn: Number(key.split(':')[0]),
            stage: 'decode',
            kind: 'tokens',
            open: true,
            trend,
          }),
        );
    this.analysisPanels.set(numberStagePanels(panels.map(restoreLogitPanel)));
    this.pendingInspector.set(view.inspectedPanel ?? null);
    this.tokenMetricInspector.set(null);
    this.tokenMetricRange.set(null);
    this.analysisSerial = Math.max(0, ...panels.map((p) => Number(p.id.slice(9))));
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    if (source !== 'return' && view.selection) this.state.selectTurn(view.selection.turn);
    if (source !== 'legacy') this.reading.apply(view, source !== 'return', prepared.observation);
    return true;
  }
  ngOnDestroy() {
    // Capture while the rendered virtual layout and semantic anchor still exist.
    this.saveView();
    this.tokenLocationAnimations.forEach((animation) => animation.cancel());
    this.analysisLocationAnimation?.cancel();
    this.analysisGraph.destroy();
    this.analysisKv.destroy();
    this.reading.destroy();
    this.scrollbar?.destroy();
    this.composerObserver?.disconnect();
    this.nodeNavigator?.destroy();
  }
  private syncMetricInspectorVisibility() {
    const inspector = this.metricInspector();
    if (!inspector) return;
    const graphInspection = inspector instanceof GraphMetricPanel && inspector.inspectionMode();
    // A linked token can select a graph while its initial tensors are still loading.
    if (
      inspector instanceof GraphMetricPanel &&
      (inspector.request.loading() ||
        !(graphInspection ? inspector.fullRange() : inspector.range()))
    )
      return;
    const tensor = graphInspection ? inspector.inspection() : null;
    const selected = graphInspection
      ? tensor
        ? {start: tensor.position(), end: tensor.position() + 1, layer: tensor.layer()}
        : null
      : inspector instanceof KvPanel
        ? inspector.inspectedRange()
        : inspector.selected();
    if (!selected) return;
    const range = graphInspection
      ? inspector.fullRange()
      : inspector instanceof KvPanel
        ? inspector.visibleRange()
        : inspector.range();
    const {start, end, layer} = selected;
    const layers =
      inspector instanceof KvPanel
        ? inspector.view() === 'trends'
          ? inspector.selectedTrendLayers()
          : inspector.layerIds()
        : !graphInspection && inspector.view() === 'trends'
          ? inspector.selectedLayers()
          : inspector.layers();
    const hiddenLayer = layers.length === 0 || (layer !== null && !layers.includes(layer));
    if (!range || start < range.start || end > range.end || hiddenLayer) {
      const panel = this.analysisPanels().find((p) => p.id === inspector.panelId());
      if (panel)
        untracked(() => {
          this.metricInspector.set(null);
          this.analysisHover.set(null);
          this.selection.set({
            kind: 'stage',
            turn: panel.turn,
            stage: panel.stage,
            side: 'target',
          });
          this.analysisOverview.set(panel);
        });
    }
  }
  constructor() {
    effect(() => this.syncAnalysisViewport());
    effect(() => this.syncMetricInspectorVisibility());
    effect(() => {
      const owner = this.tokenMetricInspector();
      if (!owner) return;
      const panel = this.analysisPanels().find((p) => p.id === owner.id);
      if (!panel) return;
      const range = this.analysisRange(panel),
        visible = this.tokenPanelRange(panel);
      if (range.start < visible.start || range.end > visible.end)
        untracked(() => {
          this.tokenMetricInspector.set(null);
          this.tokenMetricRange.set(null);
          this.selection.set({kind: 'stage', turn: panel.turn, stage: panel.stage, side: 'target'});
          this.analysisOverview.set(panel);
        });
    });
    effect(() => {
      if (this.metricInspector() || this.analysisOverview()) return;
      for (const panel of this.analysisPanels()) {
        const range = this.selectionRangeFor(panel);
        const graph = this.graphPanels().find((p) => p.panelId() === panel.id);
        const kv = this.kvPanels().find((p) => p.panelId() === panel.id);
        if (range)
          untracked(() => {
            graph?.selected.set({...range, layer: null});
            if (kv) {
              kv.position.set(range.start);
              kv.selectionEnd.set(range.end);
              kv.column.set(true);
            }
          });
        else if (graph) untracked(() => graph.selected.set(null));
      }
    });
    effect((onCleanup) => {
      this.analysisRetry();
      const captureId = this.state.captureId(),
        aligned = this.aligned(),
        debug = this.workspace.mode() === 'Debug';
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      // Skip uncaptured links entirely, including synthetic 128k text-only sessions.
      const pending = debug
        ? aligned
            .map((t) => ({
              turn: t.turn,
              pairs: [
                ...t.rows,
                ...(this.forks().get(t.turn)?.filled ? [this.forks().get(t.turn)!.pair] : []),
              ]
                .filter(
                  (p) =>
                    p.ref?.source_forward_id !== undefined ||
                    p.target?.source_forward_id !== undefined ||
                    p.ref?.batch !== undefined ||
                    p.target?.batch !== undefined,
                )
                .map((p) => ({ref: p.ref?.step ?? null, target: p.target?.step ?? null})),
            }))
            .filter((t) => t.pairs.length)
        : [];
      const preview = debug ? this.previewStepPair() : null;
      if (preview)
        pending.push({
          turn: preview.turn,
          pairs: [{ref: preview.pair.ref?.step ?? null, target: preview.pair.target?.step ?? null}],
        });
      untracked(() => this.analysis.load(captureId, pending, controller.signal));
    });
    effect((onCleanup) => {
      const turns = this.turns();
      const debug = this.workspace.mode() === 'Debug',
        session = this.state.captureId();
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      if (session !== this.snapshotCapture) {
        this.snapshotCapture = session;
        this.kvSnapshots.set(new Map());
        this.kvLoadErrors.set(new Map());
      }
      if (!debug || !session) return;
      for (const turn of turns) {
        const key = this.snapshotKey(turn.n, session);
        this.api
          .telemetry(session, turn.n, controller.signal)
          .then((data) => {
            if (!controller.signal.aborted)
              this.kvSnapshots.update((value) => new Map(value).set(key, data.kv_snapshots ?? []));
          })
          .catch(() => {
            if (!controller.signal.aborted)
              this.kvLoadErrors.update((value) =>
                new Map(value).set(key, 'Snapshot metadata could not be loaded'),
              );
          });
      }
    });
    afterNextRender(() =>
      this.zone.runOutsideAngular(() => {
        const host = this.element.nativeElement,
          composer = host.querySelector('.composer') as HTMLElement | null;
        if (!composer) return;
        this.scrollbar = createConversationScrollbar(host, () => this.workspace.mode() === 'Debug');
        this.scrollbar.mount();
        const measure = () =>
          host.style.setProperty(
            '--composer-clearance',
            `${composer.getBoundingClientRect().height + (parseFloat(getComputedStyle(composer).bottom) || 0) + 16}px`,
          );
        this.composerObserver = new ResizeObserver(measure);
        this.composerObserver.observe(composer);
        measure();
      }),
    );
    this.virtualizer = this.zone.runOutsideAngular(() =>
      createFlowVirtualizer({
        threshold: 2048,
        getTurn: (turn) => this.virtualModels().get(turn)!,
        getRows: (turn) => this.virtualModels().get(turn)?.rows ?? [],
        getAlignment: () => this.alignment(),
        isDebug: () => this.workspace.mode() === 'Debug',
        getScroller: () => this.element.nativeElement.querySelector('.reading'),
        phase: (turn, side, index) =>
          tokenPhase(this.captured(turn, side)?.tokens?.[index]) === 'thinking'
            ? 'Thinking'
            : 'Output',
        getVisibleBounds: () => this.reading.visibleBounds(),
        findRow: (turn, side, index) => {
          const model = this.virtualModels().get(turn);
          return (
            model?.bySide[side].get(index) ??
            (!model?.hasCapturedTokens ? model?.rows[index] : undefined)
          );
        },
        getSelectedRow: (turn) => {
          const model = this.virtualModels().get(turn),
            selection = this.tokenSelection();
          return selection?.turn === turn
            ? (model?.rows.find((r) => r.index === selection.index) ?? null)
            : null;
        },
        renderInputToken: (word, index, side) => renderInput(word, index, side),
        renderOutputToken: (word, index, side, turn) => {
          const model = this.virtualModels().get(turn)!,
            row = model.bySide[side].get(index),
            pair = row ? model.pairs[row.index] : null,
            debug = this.workspace.mode() === 'Debug',
            heat = this.tokenHeat(turn, pair, side);
          return renderOutput(
            word,
            {turn, side, step: index, row: row?.index},
            {
              interactive: debug && !model.alignmentUnavailable,
              debug,
              side,
              numeric:
                debug && side === 'target' && this.numericColor()
                  ? heat === null
                    ? 'missing'
                    : 'token'
                  : null,
              first: debug && this.isFirstDivergence(turn, row?.index, side),
              partner: debug && this.isStepPartner(turn, row?.index, side),
              template: debug && isTemplateToken(this.captured(turn, side)?.tokens?.[index]),
              mismatch:
                !model.alignmentUnavailable &&
                pair?.match === false &&
                (debug ? this.color() === 'token_match' : true),
              active:
                this.selection()?.turn === turn && this.tokenSelection()?.index === row?.index,
              boundaries: this.boundaries() && debug,
              palette: heatPalette(this.color()),
              heat,
              empty: this.zeroWidthText(word),
            },
          );
        },
        displayKey: () => String(this.boundaries()),
        styleKey: () =>
          [
            this.workspace.mode(),
            this.color(),
            this.boundaries(),
            this.tokenAnalysis().size,
            this.selection()?.turn,
            this.tokenSelection()?.index,
          ].join(':'),
        widthAdjustment: () => (this.boundaries() && this.workspace.mode() === 'Debug' ? 10 : 0),
        lineHeight: () => (this.boundaries() && this.workspace.mode() === 'Debug' ? 30 : 26),
        onSelect: (turn, index, side) => {
          const model = this.virtualModels().get(turn),
            row = model?.bySide[side].get(index);
          if (row && !model?.alignmentUnavailable && this.workspace.mode() === 'Debug')
            this.zone.run(() => this.select(turn, row.index));
        },
      }),
    );
    this.nodeNavigator = this.zone.runOutsideAngular(() =>
      createFlowNodeNavigator({
        getRoot: () => this.element.nativeElement,
        getScroller: () => this.element.nativeElement.querySelector('.reading'),
        isEnabled: () => this.workspace.mode() === 'Debug',
        isCollapsed: (turn, stage) => this.nodeCollapsed(turn, 'target', stage),
        toggleCollapsed: (turn, stage) => this.zone.run(() => this.togglePairedNode(turn, stage)),
        afterJump: () => this.virtualizer?.paint(true),
      }),
    );
    let lastSession: unknown = null;
    // Structure: rows, the session, collapsed stages and the details pane change the layout.
    effect(() => {
      this.virtualModels();
      const session = this.state.session();
      this.details();
      this.collapsed();
      this.thinkingExpanded();
      afterNextRender(
        () =>
          this.zone.runOutsideAngular(() => {
            if (lastSession !== session) {
              this.virtualizer?.clearLayout();
              lastSession = session;
            }
            this.virtualizer?.mount();
            this.nodeNavigator?.mount();
            this.scrollbar?.mount();
          }),
        {injector: this.injector},
      );
    });
    // Style: selection, colors and metrics only repaint the rendered tokens.
    effect(() => {
      this.selection();
      this.color();
      this.tokenAnalysis();
      this.boundaries();
      afterNextRender(() => this.zone.runOutsideAngular(() => this.virtualizer?.paint(true)), {
        injector: this.injector,
      });
    });
    let lastJob = '';
    effect(() => {
      const job = this.liveGeneration();
      if (!job) return;
      if (job.id !== lastJob) this.followLive = true;
      lastJob = job.id;
      // Lay out virtual text before following its tail. Native scroll intent is
      // separate from content growth, which can change scrollHeight by many pages.
      afterNextRender(
        () =>
          this.zone.runOutsideAngular(() => {
            const current = this.element.nativeElement.querySelector('.reading');
            if (current) {
              this.virtualizer?.mount();
              if (this.followLive) {
                current.scrollTop = current.scrollHeight;
                this.virtualizer?.paint(true);
              }
            }
          }),
        {injector: this.injector},
      );
    });
    const saved = this.workspace.conversationViews.get(this.contextKey);
    if (saved) this.applyView(saved, 'return');
    effect(() => {
      this.workspace.mode();
      untracked(() => this.reading.syncMode());
    });
    effect(() => {
      const id = this.pendingInspector();
      const panel = [...this.kvPanels(), ...this.graphPanels()].find((p) => p.panelId() === id);
      const tokenPanel = this.analysisPanels().find(
        (p) => p.id === id && (p.kind === 'tokens' || p.kind === 'distribution'),
      );
      if (tokenPanel) {
        this.tokenMetricInspector.set(tokenPanel);
        this.pendingInspector.set(null);
      }
      if (id && panel) {
        this.tokenMetricInspector.set(null);
        this.metricInspector.set(panel);
        this.pendingInspector.set(null);
      }
    });
    effect(() => {
      const target = this.workspace.bookmarkTarget();
      if (target) {
        this.openBookmark(target);
        this.workspace.bookmarkTarget.set(null);
      }
    });
  }

  turnEnding(turn: number) {
    const job = this.workspace.runtime.job();
    if (
      job?.operation === 'generate' &&
      job.session_id === this.workspace.context().captureRecordId &&
      job.turn === turn
    ) {
      if (!['completed', 'failed', 'cancelled'].includes(job.status))
        return 'Generation in progress';
      if (job.status === 'cancelled') return 'Generation stopped';
      if (job.status === 'failed') return 'Generation failed';
    }
    const reasons = this.runs().map((run) => this.captured(turn, run.id)?.stop_reason);
    if (reasons.some((r) => r === 'cancelled')) return 'Generation stopped';
    if (reasons.some((r) => r === 'error')) return 'Generation failed';
    return reasons.length && reasons.every((r) => r === 'eos' || r === 'max_output_tokens')
      ? 'Conversation complete'
      : 'End of captured conversation';
  }
  /** Captured conversation items by `turn:run`; the template asks for them on every check. */
  private readonly capturedByKey = computed(() => {
    const items = new Map<string, CapturedConversation>();
    for (const item of this.state.session()?.conversation ?? [])
      items.set(`${item.turn}:${item.run}`, item);
    return items;
  });
  captured(turn: number, run: string) {
    return this.capturedByKey().get(`${turn}:${run}`);
  }
  hasTokenCapture(turn: number, run: string) {
    return this.captured(turn, run)?.tokens !== undefined;
  }
  inputSegments(turn: number, run: string) {
    return capturedInputSegments(this.captured(turn, run));
  }
  readonly historyByTurn = computed(() => {
    const conversation = this.state.session()?.conversation;
    return new Map(
      (this.state.session()?.turns ?? []).map((turn) => [
        turn.n,
        historyStatus(conversation, turn.n),
      ]),
    );
  });
  historyStatus(turn: number) {
    return this.historyByTurn().get(turn) ?? null;
  }
  historyReason(status: string) {
    return status === 'same'
      ? 'Previous turns have identical inputs and consumed outputs in both runtimes. This does not compare the current prompt.'
      : 'Previous turns differ in their inputs or consumed outputs between runtimes.';
  }
  /** Admitted input as the runtime tokenized it, grouped by message; null without token kinds. */
  inputRows(turn: number, run: string) {
    return capturedInputRows(this.captured(turn, run));
  }
  roleIcon(role: string) {
    return role === 'system'
      ? 'settings'
      : role === 'user'
        ? 'person'
        : role === 'model'
          ? 'smart_toy'
          : 'subject';
  }
  roleLabel(role: string) {
    return role === 'system'
      ? 'System'
      : role === 'user'
        ? 'User'
        : role === 'model'
          ? 'Model'
          : 'Input';
  }
  readonly aligned = computed(() =>
    (this.state.session()?.turns ?? []).map((turn) => {
      const virtual = this.virtualModels().get(turn.n);
      if (this.virtualOutput(turn.n) && this.workspace.mode() === 'Debug')
        return {
          turn: turn.n,
          rows: virtual?.alignmentUnavailable ? [] : (virtual?.pairs ?? []),
          error: virtual?.alignmentUnavailable
            ? 'Context alignment exceeds the supported size. Select Steps alignment.'
            : '',
        };
      try {
        return {
          turn: turn.n,
          rows: alignTokens(
            this.captured(turn.n, 'ref')?.tokens,
            this.captured(turn.n, 'target')?.tokens,
            this.alignment(),
            this.captured(turn.n, 'ref')?.alignments?.content,
          ),
          error: '',
        };
      } catch (error) {
        return {turn: turn.n, rows: [] as TokenPair[], error: String((error as Error).message)};
      }
    }),
  );
  readonly kvAlignments = computed(() => {
    const result = new Map<string, import('../../../data/contracts/kv').KvPositionAlignment>();
    if (this.alignment() !== 'content') return result;
    for (const turn of this.turns()) {
      for (const stage of ['prefill', 'decode']) {
        const start =
          (turn.step_start ?? 0) - (stage === 'prefill' ? (turn.prefill_tokens ?? 0) : 0);
        const pairs =
          stage === 'prefill'
            ? (this.inputPairRows(turn.n) ?? []).flatMap((row) => [
                ...row.leading,
                ...row.text,
                ...row.trailing,
              ])
            : this.rows(turn.n);
        result.set(turn.n + ':' + stage, {start, pairs: positionPairs(pairs, start)});
      }
    }
    return result;
  });
  rows(turn: number) {
    return this.aligned().find((t) => t.turn === turn)?.rows ?? [];
  }
  readonly tokenPhases: TokenPhase[] = ['thinking', 'response'];
  readonly phasePairs = computed(
    () =>
      new Map(
        this.aligned()
          .filter((t) => !this.virtualOutput(t.turn))
          .map((t) => [t.turn, splitPhasePairs(t.rows)]),
      ),
  );
  rowsForPhase(turn: number, side: string, phase: TokenPhase) {
    return this.phasePairs().get(turn)?.[side === 'ref' ? 'ref' : 'target'][phase] ?? [];
  }
  phaseText(turn: number, side: string, phase: TokenPhase) {
    return capturedPhaseText(this.captured(turn, side), phase);
  }
  alignmentError(turn: number) {
    return this.aligned().find((t) => t.turn === turn)?.error;
  }
  readonly selectedPair = computed(() => {
    const s = this.tokenSelection();
    const row = s ? this.rows(s.turn)[s.index] : null;
    return s && row ? {...row, turn: s.turn} : null;
  });
  /** Each turn's first differing generation step, paired by step (see `forkPair`). */
  readonly forks = computed(
    () =>
      new Map(
        this.aligned().map((t) => [
          t.turn,
          forkPair(
            t.rows,
            this.captured(t.turn, 'ref')?.tokens,
            this.captured(t.turn, 'target')?.tokens,
          ),
        ]),
      ),
  );
  /** Each turn's Decode outcome: identical, diverged at a step, unknown or not captured. */
  readonly outcomes = computed(
    () =>
      new Map(
        this.aligned().map((t) => [
          t.turn,
          decodeOutcome(
            t.rows,
            this.forks().get(t.turn) ?? null,
            this.captured(t.turn, 'ref')?.tokens,
            this.captured(t.turn, 'target')?.tokens,
          ),
        ]),
      ),
  );
  outcomeText(turn: number) {
    const outcome = this.outcomes().get(turn);
    return !outcome || outcome.kind === 'uncaptured'
      ? 'Not captured'
      : outcome.kind === 'identical'
        ? 'Identical'
        : outcome.kind === 'diverged'
          ? 'Diverged at step ' + outcome.step
          : 'Not comparable';
  }
  /** The selected Decode stage's summary. Every value comes from the captures or from analysis
   *  the Server returned; a value that is not loaded or not provable reads as a dash. */
  readonly stageDecode = computed(() => {
    const stage = this.selectedStage();
    if (!stage || stage.stage !== 'decode') return null;
    const turn = stage.turn,
      outcome = this.outcomes().get(turn),
      fork = this.forks().get(turn) ?? null,
      rows = this.rows(turn),
      data = this.tokenAnalysis();
    // The Server gives paired metrics only where both runtimes read the same context, so the
    // largest loaded value is the peak of the comparable steps.
    let peak: {value: number; step: number} | null = null;
    for (const row of rows) {
      const pair = fork?.filled && fork.index === row.index ? fork.pair : row,
        value = data.get(pairKey(turn, pair))?.metrics['kl']?.value;
      if (value != null && (!peak || value > peak.value))
        peak = {value, step: (pair.target ?? pair.ref)!.step};
    }
    const distribution = fork ? data.get(pairKey(turn, fork.pair))?.distribution : undefined,
      margin = (side: 'ref' | 'target') => {
        const value = distribution?.[side]?.margin;
        return value == null || !Number.isFinite(value) ? '—' : (value * 100).toFixed(1) + ' pp';
      };
    return {
      turn,
      outcome: this.outcomeText(turn),
      prefix:
        !outcome || outcome.kind === 'uncaptured'
          ? 'Not captured'
          : formatCount(outcome.prefix) + (outcome.prefix === 1 ? ' token' : ' tokens'),
      peak: peak ? `${formatSignificant(peak.value, 5)} nats · step ${peak.step}` : '—',
      fork,
      margin: {ref: margin('ref'), target: margin('target')},
    };
  });
  /** One turn's Prefill: whether both runtimes tokenized the input alike and entered the turn
   *  with the same history. */
  private prefillOutcome(turn: number) {
    const rows = this.inputPairRows(turn),
      pairs = rows?.flatMap((row) => [...row.leading, ...row.text, ...row.trailing]) ?? [],
      first = pairs.find((pair) => pair.match !== true),
      history = this.historyStatus(turn);
    const tokenized = !rows
      ? 'Not captured'
      : !first
        ? 'Same'
        : first.match === null
          ? 'Unknown'
          : 'Different · first at token ' + first.index;
    return {
      tokenized,
      history: history === 'same' ? 'Same' : history === 'differs' ? 'Differs' : 'Not proven',
      historyReason:
        history === 'same' || history === 'differs'
          ? this.historyReason(history)
          : 'The captures do not prove whether both runtimes entered this turn with the same history.',
      // The index states what differs first: the input's tokens, then the history behind them.
      index: tokenized.startsWith('Different')
        ? 'Input differs'
        : tokenized !== 'Same'
          ? tokenized
          : history === 'differs'
            ? 'History differs'
            : 'Same',
    };
  }
  /** Conversation Info's index: one line per turn, each cell opening its stage. */
  readonly turnIndex = computed(() =>
    this.turns().map((turn) => {
      const outcome = this.outcomes().get(turn.n);
      return {
        turn: turn.n,
        prefill: this.prefillOutcome(turn.n).index,
        decode:
          outcome?.kind === 'diverged'
            ? 'Differs · step ' + outcome.step
            : this.outcomeText(turn.n),
      };
    }),
  );
  /** Open a stage from the index: select it on the Target side and bring its card into view. */
  openStage(turn: number, stage: 'prefill' | 'decode', event: Event) {
    const card = this.element.nativeElement.querySelector(
      `[data-node-turn="${turn}"][data-node-stage="${stage}"][data-node-side="target"]`,
    ) as HTMLElement | null;
    this.selectStage(turn, 'target', stage, card ?? (event.currentTarget as HTMLElement));
    // After the selection's own render work, which restores the previous scroll anchor.
    afterNextRender(
      () =>
        card?.scrollIntoView({
          block: 'center',
          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        }),
      {injector: this.injector},
    );
  }
  /** Why one runtime stopped: its recorded stop reason, else the stop token it sampled last. */
  stageEnding(turn: number, side: string) {
    const capture = this.captured(turn, side),
      reason = capture?.stop_reason;
    return reason === 'eos' || (!reason && capture?.tokens?.at(-1)?.stop === true)
      ? 'Stop token'
      : reason === 'max_output_tokens'
        ? 'Output limit'
        : reason === 'cancelled'
          ? 'Stopped'
          : reason === 'error'
            ? 'Failed'
            : 'Not captured';
  }
  /** Select a turn's first differing row from its stage summary. */
  selectFork(turn: number) {
    const fork = this.forks().get(turn);
    if (fork) {
      this.select(turn, fork.index);
      this.locateSelectedToken();
    }
  }
  /** What the details describe: the selected row, except that the first differing row shows the
   *  two tokens of its generation step when the text alignment left one side empty. */
  readonly detailPair = computed(() => {
    const pair = this.selectedPair();
    if (!pair) return null;
    const fork = this.forks().get(pair.turn);
    return fork?.filled && fork.index === pair.index
      ? {...fork.pair, turn: pair.turn, filled: fork.filled}
      : {...pair, filled: null};
  });
  pairingTitle(filled: 'ref' | 'target') {
    return (
      'The first generation step where the two runtimes produced different tokens. Context ' +
      `alignment matches text and leaves the ${filled === 'ref' ? 'Reference' : 'Target'} side of ` +
      'this row empty, so the details pair the two tokens of this step instead. Metrics appear ' +
      'only where the captures prove that both runtimes read the same context.'
    );
  }
  /** The token drawn elsewhere that the details pair with the selected first differing row. */
  isStepPartner(turn: number, index: number | undefined, side: string) {
    const fork = this.forks().get(turn),
      selected = this.tokenSelection();
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
    for (const [turn, fork] of this.forks())
      if (fork && fork.pair.ref?.released !== false && fork.pair.target?.released !== false)
        return {turn, pair: fork.pair};
    return null;
  });
  // The preview is a fixed pair: the first sampled difference, whose metrics exist; identical
  // runs fall back to the first short differing pair, then the selection or the first row.
  readonly displayPreview = computed(() => {
    const fixed = this.previewStepPair();
    if (fixed)
      return {
        ref: fixed.pair.ref?.text ?? 'Not captured',
        target: fixed.pair.target?.text ?? 'Not captured',
        match: false,
        analysis: this.tokenAnalysis().get(pairKey(fixed.turn, fixed.pair)),
      };
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
      changed?.turn ?? this.selection()?.turn ?? this.state.batch()?.turn ?? this.turns()[0]?.n;
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
    for (const turn of this.aligned()) {
      const row = turn.rows.find((r) => r.match === false);
      if (row) return {turn: turn.turn, index: row.index};
    }
    return null;
  });
  isFirstDivergence(turn: number, index: number | undefined, side: string) {
    const first = this.firstDivergence();
    return side === 'target' && first?.turn === turn && first.index === index;
  }
  readonly mismatches = computed(() => {
    const predicate = this.predicate(),
      records = predicate.requiresMetrics ? this.tokenAnalysis() : null;
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
  private bookmarkMatches(b: ConversationBookmark) {
    const selected = this.selection(),
      savedView = readViewState(b.view),
      saved = savedView?.selection;
    const activePanel = this.metricInspector()?.panelId() ?? this.tokenMetricInspector()?.id;
    if (activePanel)
      return (
        savedView?.inspectedPanel === activePanel &&
        JSON.stringify(savedView.analysisPanels) === JSON.stringify(this.capturePanels())
      );
    if (selected?.kind === 'input')
      return (
        saved?.kind === 'input' && saved.turn === selected.turn && saved.index === selected.index
      );
    return selected?.kind === 'stage'
      ? saved?.kind === 'stage' && saved.turn === selected.turn && saved.stage === selected.stage
      : selected?.kind === 'token' &&
          b.turn === selected.turn &&
          b.index === selected.index &&
          b.alignment === this.alignment();
  }
  readonly isBookmarked = computed(() => this.bookmarks().some((b) => this.bookmarkMatches(b)));
  setAlignment(mode: AlignmentMode) {
    const active = this.overviewPanel();
    this.alignment.set(mode);
    this.analysisHover.set(null);
    this.analysisOverview.set(active ?? null);
    for (const graph of this.graphPanels()) {
      graph.selected.set(null);
      graph.viewport.set(null);
    }
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.trend ? {...p, trend: {...p.trend, positionRange: undefined}} : p)),
    );
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set(null);
  }
  readonly compactTokenHtml = (text: string) => tokenMarkup(text, false);
  private readonly compactLayout = signal(
    new Map<string, {available: number; widths: Map<number, number>}>(),
  );
  private compactPairs(turn: number, stage: string) {
    return stage === 'prefill'
      ? (this.inputPairRows(turn) ?? []).flatMap((r) => [...r.leading, ...r.text, ...r.trailing])
      : this.rows(turn);
  }
  private readonly compactLayoutSync = afterRenderEffect(() => {
    this.boundaries();
    this.color();
    this.collapsed();
    this.panelWidth();
    this.details();
    this.layoutTick();
    this.aligned();
    this.turns();
    const host = this.element.nativeElement as HTMLElement;
    const next = new Map<string, {available: number; widths: Map<number, number>}>();
    for (const container of Array.from(host.querySelectorAll<HTMLElement>('.nodeContextText'))) {
      const node = container.closest<HTMLElement>('[data-node-stage]');
      const sample = container.querySelector<HTMLElement>('.nodePreviewToken');
      if (!node || !sample) continue;
      const turn = Number(node.dataset['nodeTurn']),
        stage = node.dataset['nodeStage']!;
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
      const pairs = this.compactPairs(turn, stage);
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
    this.compactLayout.update((old) => {
      const serialize = (map: typeof next) =>
        JSON.stringify([...map].map(([key, value]) => [key, value.available, [...value.widths]]));
      return serialize(old) === serialize(next) ? old : next;
    });
  });
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
    for (const turn of this.turns())
      for (const stage of ['prefill', 'decode'] as const) {
        const pairs = this.compactPairs(turn.n, stage);
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
          if (!layout || needed <= layout.available || count === 1) break;
          count--;
        }
        for (const side of ['ref', 'target'] as const)
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
    return result;
  });
  compactColumns(turn: number, stage: string) {
    return (this.compactTokens().get(`${turn}:ref:${stage}`) ?? [])
      .flatMap((t) => [...(t.ellipsis ? ['16px'] : []), `${t.width}px`])
      .join(' ');
  }
  compactInput(turn: number, side: string) {
    const c = this.captured(turn, side),
      text = c?.serialized_input ?? c?.input ?? 'Input text not captured';
    return text.replace(/\s+/g, ' ').slice(0, 200);
  }
  compactOutput(turn: number, side: string) {
    return (this.captured(turn, side)?.output ?? 'Response text not captured')
      .replace(/\s+/g, ' ')
      .slice(0, 200);
  }
  compactCount(turn: number, side: string, stage: string) {
    const count = this.stageCount(turn, side, stage);
    if (typeof count !== 'number') return count;
    const pairs =
      stage === 'prefill'
        ? (this.inputPairRows(turn) ?? []).flatMap((row) => [
            ...row.leading,
            ...row.text,
            ...row.trailing,
          ])
        : this.rows(turn);
    const gaps = pairs.filter((pair) => !this.pairToken(pair, side)).length;
    return `${formatCount(count)} tokens · ${formatCount(gaps)} gaps`;
  }
  togglePairedNode(turn: number, stage: string) {
    const reading = this.element.nativeElement.querySelector('.reading'),
      node = reading?.querySelector(`[data-node-turn="${turn}"][data-node-stage="${stage}"]`);
    const viewport = reading?.getBoundingClientRect(),
      before = node?.getBoundingClientRect(),
      collapsed = this.nodeCollapsed(turn, 'target', stage);
    const anchor =
      before && viewport
        ? Math.max(viewport.top + 72, Math.min(before.top, viewport.bottom - 112))
        : null;
    this.collapsed.update((set) => {
      const next = new Set(set);
      for (const side of ['ref', 'target']) {
        const key = `${turn}:${side}:${stage}`;
        collapsed ? next.delete(key) : next.add(key);
      }
      return next;
    });
    afterNextRender(
      () => {
        this.nodeNavigator?.mount();
        if (reading && node && anchor !== null)
          reading.scrollTop += node.getBoundingClientRect().top - anchor;
        this.virtualizer?.mount();
      },
      {injector: this.injector},
    );
  }
  nodeCollapsed(turn: number, run: string, phase: string) {
    return this.collapsed().has(`${turn}:${run}:${phase}`);
  }
  stageSelected(turn: number, stage: string) {
    const selected = this.selectedStage();
    return selected?.turn === turn && selected.stage === stage;
  }
  generatedCount(run: string) {
    const captures = this.turns().map((t) => this.captured(t.n, run));
    return captures.every((c) => c?.tokens !== undefined)
      ? captures.reduce((n, c) => n + c!.tokens!.length, 0)
      : 'Not captured';
  }
  turnPrompt(turn: number) {
    const capture = this.captured(turn, 'ref') ?? this.captured(turn, 'target');
    return capture?.input?.replace(/\s+/g, ' ').trim() || 'Input not captured';
  }
  matchLabel(value: string) {
    return value === 'Same'
      ? 'Yes'
      : value.startsWith('Different') || value === 'Differs'
        ? 'No'
        : 'Unknown';
  }
  stageHistory(turn: number, stage: string) {
    const input = this.prefillOutcome(turn);
    if (stage === 'prefill') return this.matchLabel(input.history);
    if (input.history === 'Differs' || input.tokenized.startsWith('Different')) return 'No';
    return input.history === 'Same' && input.tokenized === 'Same' ? 'Yes' : 'Unknown';
  }
  readonly kvDetailSelection = computed(() => {
    const inspector = this.metricInspector();
    if (!(inspector instanceof KvPanel)) return null;
    const range = inspector.inspectedRange();
    return range
      ? {
          ...range,
          kind: range.layer === null ? null : (inspector.selectedCurve() ?? inspector.kind()),
        }
      : null;
  });
  overviewPanel() {
    const inspector = this.metricInspector();
    const p = inspector
      ? this.analysisPanels().find((panel) => panel.id === inspector.panelId())
      : (this.tokenMetricInspector() ?? this.analysisOverview());
    return p ? (this.analysisPanels().find((x) => x.id === p.id) ?? p) : null;
  }
  graphDetailSelection(panel: StageAnalysisPanel) {
    const graph = this.overviewGraph(panel);
    if (graph?.inspectionMode()) {
      const tensor = graph.inspection();
      return tensor
        ? {start: tensor.position(), end: tensor.position() + 1, layer: tensor.layer()}
        : null;
    }
    if (graph?.view() === 'graph' && graph.graphPosition() !== null) {
      const position = graph.graphPosition()!;
      return {start: position, end: position + 1, layer: null};
    }
    return graph && this.metricInspector() === graph ? graph.selected() : null;
  }
  analysisRange(panel: StageAnalysisPanel): {start: number; end: number} {
    if (panel.kind === 'kv')
      return (
        this.kvDetailSelection() ??
        this.overviewKv(panel)?.visibleRange() ??
        this.entireStageRange(panel)
      );
    const graph = this.overviewGraph(panel);
    if (graph)
      return this.graphDetailSelection(panel) ?? graph.range() ?? this.entireStageRange(panel);
    const logits = this.overviewFinalLogits(panel);
    if (logits) return {start: logits.position(), end: logits.position() + 1};
    if (this.tokenMetricInspector()?.id === panel.id) {
      const selected = this.tokenSelection();
      const bin = this.tokenMetricRange();
      if (bin?.panelId === panel.id) return bin;
      const restored = this.selection();
      if (
        restored?.kind === 'stage' &&
        restored.turn === panel.turn &&
        restored.stage === panel.stage &&
        restored.range
      )
        return restored.range;
      const pair = this.rows(panel.turn).find((row) => row.index === selected?.index);
      if (pair) {
        const token =
          panel.kind === 'tokens' ? (pair.ref ?? pair.target) : (pair.target ?? pair.ref);
        const start =
          this.stageGlobalStart(panel.turn) +
          (this.alignment() === 'content' ? pair.index : (token?.step ?? pair.index));
        return {start, end: start + 1};
      }
    }
    return this.tokenPanelRange(panel);
  }
  private tokenPanelRange(panel: StageAnalysisPanel) {
    const positions =
      this.alignment() === 'content'
        ? this.rows(panel.turn).map((row) => this.stageGlobalStart(panel.turn) + row.index)
        : this.rows(panel.turn)
            .flatMap((row) =>
              panel.kind === 'tokens' ? [row.ref ?? row.target] : [row.ref, row.target],
            )
            .filter((token) => !!token)
            .map((token) => this.stageGlobalStart(panel.turn) + token!.step);
    const full = positions.length
      ? {
          start: positions.reduce((a, b) => Math.min(a, b)),
          end: positions.reduce((a, b) => Math.max(a, b)) + 1,
        }
      : this.entireStageRange(panel);
    const saved = panel.trend?.positionRange;
    if (!saved) return full;
    const start = Math.max(full.start, saved.start),
      end = Math.min(full.end, saved.end);
    return start < end ? {start, end} : full;
  }
  overviewFinalLogits(panel: StageAnalysisPanel) {
    return this.tokenPanels()
      .find((p) => p.panelId() === panel.id)
      ?.inspection();
  }
  overviewGraph(panel: StageAnalysisPanel) {
    return this.graphPanels().find((p) => p.panelId() === panel.id);
  }
  overviewKv(panel: StageAnalysisPanel) {
    return this.kvPanels().find((p) => p.panelId() === panel.id);
  }
  private readonly entireStageRanges = computed(() => {
    const ranges = new Map<string, {start: number; end: number}>();
    for (const turn of this.turns())
      for (const stage of ['prefill', 'decode'] as const) {
        const start =
          (turn.step_start ?? 0) - (stage === 'prefill' ? (turn.prefill_tokens ?? 0) : 0);
        const count = Math.max(
          0,
          ...['ref', 'target']
            .map((side) => this.stageCount(turn.n, side, stage))
            .filter((n): n is number => typeof n === 'number'),
        );
        ranges.set(`${turn.n}:${stage}`, {start, end: start + count});
      }
    return ranges;
  });
  private readonly emptyStageRange = {start: 0, end: 0};
  entireStageRange(panel: StageAnalysisPanel) {
    return this.entireStageRanges().get(`${panel.turn}:${panel.stage}`) ?? this.emptyStageRange;
  }
  stagePoints(panel: StageAnalysisPanel) {
    const start = this.stageGlobalStart(panel.turn);
    const range = this.analysisRange(panel);
    return this.rows(panel.turn)
      .map((pair) => ({
        position:
          start +
          (this.alignment() === 'content'
            ? pair.index
            : ((panel.kind === 'tokens' ? (pair.ref ?? pair.target) : (pair.target ?? pair.ref))
                ?.step ?? pair.index)),
        refPosition: pair.ref
          ? start + (this.alignment() === 'content' ? pair.index : pair.ref.step)
          : null,
        targetPosition: pair.target
          ? start + (this.alignment() === 'content' ? pair.index : pair.target.step)
          : null,
        index: pair.index,
        hasRef: !!pair.ref,
        hasTarget: !!pair.target,
        analysis: this.tokenAnalysis().get(pairKey(panel.turn, pair)),
      }))
      .filter((point) =>
        (panel.kind === 'distribution'
          ? [point.refPosition, point.targetPosition]
          : [point.position]
        ).some((position) => position !== null && position >= range.start && position < range.end),
      );
  }
  locateKvCache(
    panel: StageAnalysisPanel,
    kind: 'key' | 'value',
    position?: number,
    layer?: number,
  ) {
    // Location reveals the exact analysis instance without changing its inspection.
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        this.kvPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateCache(kind, position, layer);
      },
      {injector: this.injector},
    );
  }
  locateMetric(panel: StageAnalysisPanel) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (panel.kind === 'kv')
          this.kvPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
        else if (panel.kind === 'graph' || panel.kind === 'activations')
          this.graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
        else
          this.tokenPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
      },
      {injector: this.injector},
    );
  }
  locateScopeControl(panel: StageAnalysisPanel, control: 'positions' | 'layers') {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const kv = this.kvPanels().find((p) => p.panelId() === panel.id);
        const graph = ['graph', 'activations'].includes(panel.kind)
          ? this.graphPanels().find((p) => p.panelId() === panel.id)
          : undefined;
        if (control === 'positions') {
          if (kv) kv.locatePositions();
          else if (graph) graph.locateRange();
          else
            this.tokenPanels()
              .find((p) => p.panelId() === panel.id)
              ?.locateRange();
        } else if (kv) kv.locateLayers();
        else graph?.locateLayers();
      },
      {injector: this.injector},
    );
  }
  locateTokenView(panel: StageAnalysisPanel) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (canonicalAnalysisKind(panel.kind) === 'activations')
          this.graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateView();
        else
          this.tokenPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateView();
      },
      {injector: this.injector},
    );
  }
  locateAggregation(panel: StageAnalysisPanel) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (panel.kind === 'kv')
          this.kvPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateHeads();
        else
          this.graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateAggregation();
      },
      {injector: this.injector},
    );
  }
  locateGraphChart(panel: StageAnalysisPanel, position?: number, layer?: number) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () =>
        this.graphPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateChart(position, layer),
      {injector: this.injector},
    );
  }
  locateFinalChart(panel: StageAnalysisPanel, side?: string, position?: number) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () =>
        this.tokenPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateFinal(side, position),
      {injector: this.injector},
    );
  }
  locateStatePosition(panel: StageAnalysisPanel, value: StateLocation) {
    if ((panel.kind === 'tokens' || panel.kind === 'distribution') && !value.inspect) {
      this.locateFinalChart(panel, value.kind, value.position);
      return;
    }
    if ((panel.kind === 'graph' || panel.kind === 'activations') && !value.inspect) {
      this.locateGraphChart(panel, value.position, value.layer ?? undefined);
      return;
    }
    if (
      panel.kind === 'kv' &&
      !value.inspect &&
      value.layer !== null &&
      (value.kind === 'key' || value.kind === 'value')
    ) {
      this.locateKvCache(panel, value.kind, value.position, value.layer);
      return;
    }
    const graph = this.overviewGraph(panel),
      kv = this.overviewKv(panel);
    const range = {start: value.position, end: value.position + 1, layer: value.layer};
    if (value.inspect) {
      if (graph) {
        if (value.anchor) graph.anchor.set(value.anchor);
        graph.select(range);
        this.inspectMetric(graph);
      } else if (kv) {
        if (value.kind) kv.selectedCurve.set(value.kind);
        if (value.head !== undefined) kv.head.set(String(value.head));
        kv.position.set(value.position);
        kv.selectionEnd.set(value.position + 1);
        kv.column.set(value.layer === null);
        if (value.layer !== null) kv.layer.set(value.layer);
        this.inspectMetric(kv);
      } else {
        const point = this.stagePoints(panel).find((p) =>
          value.index !== undefined ? p.index === value.index : p.position === value.position,
        );
        if (point) this.inspectTokenMetric(panel, point.index);
      }
    } else {
      this.analysisHover.set({panelId: panel.id, range});
    }
    const start = this.entireStageRange(panel).start;
    const pairs =
      panel.stage === 'prefill'
        ? (this.inputPairRows(panel.turn) ?? []).flatMap((r) => [
            ...r.leading,
            ...r.text,
            ...r.trailing,
          ])
        : this.rows(panel.turn);
    const pair = pairs.find((p) =>
      this.alignment() === 'content'
        ? start + p.index === value.position
        : [p.ref, p.target].some((t) => t && t.step + start === value.position),
    );
    if (pair)
      this.element.nativeElement
        .querySelector(
          `[data-turn="${panel.turn}"][data-row="${pair.index}"]${panel.stage === 'prefill' ? '[data-kind="input"]' : ':not([data-kind])'}`,
        )
        ?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }
  stagePositions(turn: number, stage: string, fullStage = false) {
    const selected = fullStage ? null : this.selectionInfo();
    if (selected?.turn === turn && selected.stage === stage && selected.range)
      return selected.range.end - selected.range.start === 1
        ? String(selected.range.start)
        : selected.range.start + '–' + (selected.range.end - 1);
    const info = this.turns().find((t) => t.n === turn);
    if (info?.step_start == null) return 'Not captured';
    const start = info.step_start - (stage === 'prefill' ? (info.prefill_tokens ?? 0) : 0);
    const pairs = this.compactPairs(turn, stage);
    if (pairs.length) {
      const last = pairs.reduce(
        (max, pair) =>
          Math.max(
            max,
            this.alignment() === 'content'
              ? pair.index
              : ((pair.ref ?? pair.target)?.step ?? pair.index),
          ),
        0,
      );
      return last === 0 ? String(start) : start + '–' + (start + last);
    }
    const counts = ['ref', 'target'].map((side) => this.stageCount(turn, side, stage));
    const count = Math.max(0, ...counts.filter((n): n is number => typeof n === 'number'));
    return count === 1 ? String(start) : count ? start + '–' + (start + count - 1) : '-';
  }

  readingBackgroundClick(event: MouseEvent) {
    if (this.workspace.mode() !== 'Debug' || window.getSelection()?.toString()) return;
    const target = event.target as Element;
    if (
      target.closest(
        '[data-node-stage], .analysis-content, .expanded-analysis-header, button, a, input, select, textarea, label, [role="button"], [role="tab"], [role="dialog"], [contenteditable="true"]',
      )
    )
      return;
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set(null);
    this.returnFocus = null;
    this.details.set(true);
  }
  locateConversationStage(turn: number, stage: 'prefill' | 'decode') {
    const reading = this.element.nativeElement.querySelector('.reading') as HTMLElement | null;
    const cards = reading?.querySelectorAll<HTMLElement>(
      `[data-node-turn="${turn}"][data-node-stage="${stage}"]`,
    );
    const node = cards?.[0];
    if (!reading || !node) return;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const heading = reading.querySelector('.column-headings')?.getBoundingClientRect().height ?? 0;
    reading.scrollTo({
      top:
        reading.scrollTop +
        node.getBoundingClientRect().top -
        reading.getBoundingClientRect().top -
        heading -
        8,
      behavior: reduced ? 'auto' : 'smooth',
    });
    if (!reduced) {
      for (const card of Array.from(cards ?? [])) {
        card.getAnimations().forEach((animation) => animation.cancel());
        card.animate(
          [
            {outline: '2px solid transparent', outlineOffset: '2px'},
            {
              outline: '2px solid var(--me-color-primary, #1a73e8)',
              outlineOffset: '2px',
              offset: 0.4,
            },
            {outline: '2px solid transparent', outlineOffset: '2px'},
          ],
          {duration: 1000, easing: 'ease-out'},
        );
      }
    }
  }
  locateStage(turn: number, stage: string) {
    const selected = this.selectedStage();
    if (selected?.turn === turn && selected.stage === stage && selected.range) {
      const info = this.turns().find((t) => t.n === turn);
      const start =
        (info?.step_start ?? 0) - (stage === 'prefill' ? (info?.prefill_tokens ?? 0) : 0);
      const pairs =
        stage === 'prefill'
          ? (this.inputPairRows(turn) ?? []).flatMap((r) => [
              ...r.leading,
              ...r.text,
              ...r.trailing,
            ])
          : this.rows(turn);
      const pair = pairs.find((p) =>
        [p.ref, p.target].some(
          (t) =>
            t && start + t.step >= selected.range!.start && start + t.step < selected.range!.end,
        ),
      );
      const node =
        pair &&
        this.element.nativeElement.querySelector(
          '[data-turn="' +
            turn +
            '"][data-row="' +
            pair.index +
            '"]' +
            (stage === 'prefill' ? '[data-kind="input"]' : ':not([data-kind])'),
        );
      if (node) {
        node.scrollIntoView({block: 'center', behavior: 'smooth'});
        return;
      }
    }
    this.element.nativeElement
      .querySelector(
        '[data-node-turn="' + turn + '"][data-node-stage="' + stage + '"][data-node-side="target"]',
      )
      ?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }
  locateDivergence(turn: number) {
    const fork = this.forks().get(turn);
    if (!fork) return;
    this.element.nativeElement
      .querySelector('[data-turn="' + turn + '"][data-row="' + fork.index + '"]:not([data-kind])')
      ?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }
  private analysisLocationAnimation?: Animation;
  /** Name's locator reveals the whole view without reinterpreting its selection. */
  locateAnalysisArea(panel: StageAnalysisPanel) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const area = (this.element.nativeElement as HTMLElement)
          .querySelector('#' + panel.id)
          ?.closest<HTMLElement>('.stage-analysis');
        const frame = area?.querySelector<HTMLElement>('.analysis-location-frame');
        if (!area || !frame) return;
        const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
        area.scrollIntoView({
          block: 'start',
          inline: 'nearest',
          behavior: reduced ? 'instant' : 'smooth',
        });
        this.analysisLocationAnimation?.cancel();
        this.analysisLocationAnimation = animateAnalysisLocationFeedback(frame, reduced);
      },
      {injector: this.injector},
    );
  }
  locateAnalysis(panel: StageAnalysisPanel) {
    this.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const range = this.selectionRangeFor(panel);
        if (range && this.metricInspector()?.panelId() !== panel.id) {
          this.graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.selected.set({...range, layer: null});
          const kv = this.kvPanels().find((p) => p.panelId() === panel.id);
          if (kv) {
            kv.position.set(range.start);
            kv.selectionEnd.set(range.end);
            kv.column.set(true);
          }
        }
        if (this.analysisOverview()?.id === panel.id && range) {
          const graph = this.graphPanels().find((p) => p.panelId() === panel.id);
          const kv = this.kvPanels().find((p) => p.panelId() === panel.id);
          if (graph || kv) this.inspectMetric((graph ?? kv)!);
          else if (this.tokenSelection())
            this.inspectTokenMetric(panel, this.tokenSelection()!.index);
        }
        this.element.nativeElement
          .querySelector('#' + panel.id)
          ?.scrollIntoView({block: 'start', behavior: 'smooth'});
      },
      {injector: this.injector},
    );
  }
  /** Use the same aligned positions as the charts; gaps are not captured tokens. */
  private stageInfoPairs(turn: number, stage: string) {
    const pairs = this.compactPairs(turn, stage);
    const selected = this.selectionInfo();
    if (selected?.turn === turn && selected.stage === stage && selected.index !== undefined)
      return pairs.filter((pair) => pair.index === selected.index);
    if (selected?.turn !== turn || selected.stage !== stage || !selected.range) return pairs;
    const info = this.turns().find((t) => t.n === turn);
    if (info?.step_start == null) return [];
    const start = info.step_start - (stage === 'prefill' ? (info.prefill_tokens ?? 0) : 0);
    return pairs.filter((pair) => {
      const position =
        start +
        (this.alignment() === 'content'
          ? pair.index
          : ((pair.ref ?? pair.target)?.step ?? pair.index));
      return position >= selected.range!.start && position < selected.range!.end;
    });
  }
  stageInfoCount(turn: number, run: string, stage: string) {
    const capture = this.captured(turn, run);
    const tokens = stage === 'prefill' ? capture?.input_tokens : capture?.tokens;
    const selected = this.selectionInfo();
    if (
      selected?.turn === turn &&
      selected.stage === stage &&
      (selected.range || selected.index !== undefined)
    ) {
      if (!tokens || this.turns().find((t) => t.n === turn)?.step_start == null)
        return 'Not captured';
      return this.stageInfoPairs(turn, stage).filter((pair) => this.pairToken(pair, run)).length;
    }
    return tokens?.length ?? this.stageCount(turn, run, stage);
  }
  stageTokenMatchRate(turn: number, stage: string) {
    const captures = ['ref', 'target'].map((side) => this.captured(turn, side));
    if (
      captures.some(
        (capture) => (stage === 'prefill' ? capture?.input_tokens : capture?.tokens) === undefined,
      )
    )
      return '-';
    const pairs = this.stageInfoPairs(turn, stage);
    if (!pairs.length || pairs.some((pair) => pair.match === null)) return '-';
    return formatPercent(pairs.filter((pair) => pair.match === true).length / pairs.length);
  }
  readonly stageAnalysisProperties = computed(() => {
    const selected = this.selectionInfo();
    if (!selected) return [];
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
  stageCount(turn: number, run: string, stage: string) {
    const capture = this.captured(turn, run);
    return (
      (stage === 'prefill' ? capture?.input_token_count : capture?.tokens?.length) ?? 'Not captured'
    );
  }
  selectStagePointer(event: MouseEvent, turn: number, side: string, stage: 'prefill' | 'decode') {
    if (
      (event.target as Element).closest(
        'button,summary,a,input,select,[data-v-gap],[role="button"]',
      ) ||
      window.getSelection()?.toString()
    )
      return;
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  selectStageKey(event: KeyboardEvent, turn: number, side: string, stage: 'prefill' | 'decode') {
    if (event.target !== event.currentTarget || !['Enter', ' '].includes(event.key)) return;
    event.preventDefault();
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  private selectStage(
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
    trigger: HTMLElement,
  ) {
    if (this.workspace.mode() !== 'Debug') return;
    const anchor = this.virtualizer?.capture();
    this.returnFocus = trigger;
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set({kind: 'stage', turn, side, stage});
    this.details.set(true);
    this.state.selectTurn(turn);
    afterNextRender(
      () => {
        this.virtualizer?.mount();
        if (anchor) this.virtualizer?.restore(anchor);
      },
      {injector: this.injector},
    );
  }
  select(turn: number, index: number, event?: Event, toggle = true) {
    const current = this.selection();
    if (toggle && current?.kind === 'token' && current.turn === turn && current.index === index) {
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
    this.state.selectTurn(turn);
  }
  closeDetails() {
    this.analysisOverview.set(null);
    this.metricInspector.set(null);
    this.tokenMetricInspector.set(null);
    this.selection.set(null);
    this.returnFocus?.focus({preventScroll: true});
  }
  @HostListener('keydown.escape', ['$event']) escape(event: KeyboardEvent) {
    if (this.expandedAnalysis()) {
      event.stopPropagation();
      this.exitExpandedAnalysis();
      return;
    }
    if (this.selection()) {
      event.stopPropagation();
      this.closeDetails();
    }
  }
  find(direction: number) {
    const results = this.mismatches();
    const i = direction === 0 ? 0 : this.resultIndex() + direction;
    if (i >= 0 && i < results.length) {
      this.select(results[i].turn, results[i].index);
      this.locateSelectedToken();
    }
  }
  bookmark(addOnly = false) {
    const selected = this.selection();
    if (!selected) return;
    if (this.isBookmarked()) {
      if (!addOnly)
        this.workspace.setBookmarks(this.bookmarks().filter((b) => !this.bookmarkMatches(b)));
      this.bookmarkNotice.set(addOnly ? 'Bookmark already saved' : 'Bookmark removed');
      return;
    }
    const pair = selected.kind === 'input' ? this.selectedInputPair() : this.selectedPair(),
      token = pair?.target ?? pair?.ref;
    const title = this.metricInspector()
      ? `Turn ${selected.turn} · ${this.analysisLabel(this.analysisPanels().find((p) => p.id === this.metricInspector()!.panelId())!)}${this.metricInspector() instanceof KvPanel ? ' · Position ' + (this.metricInspector() as KvPanel).position() : ''}`
      : selected.kind === 'stage'
        ? `Turn ${selected.turn} · ${selected.stage === 'prefill' ? 'Prefill' : 'Decode'}`
        : selected.kind === 'input'
          ? `Turn ${selected.turn} · Prefill · Token ${selected.index}${token ? ' “' + capturedTokenLabel(token).trim().slice(0, 80) + '”' : ''}`
          : `Turn ${selected.turn} · Decode · Token ${token?.step ?? selected.index}${token ? ' “' + capturedTokenLabel(token).trim().slice(0, 80) + '”' : ''}`;
    this.workspace.setBookmarks([
      ...this.bookmarks(),
      {
        id: crypto.randomUUID(),
        title,
        turn: selected.turn,
        index: selected.kind === 'token' ? selected.index : -1,
        alignment: this.alignment(),
        view: this.captureView(),
      },
    ]);
    this.bookmarkNotice.set('Bookmark added');
  }
  openBookmark(bookmark: ConversationBookmark) {
    const expected = {turn: bookmark.turn, alignment: bookmark.alignment};
    if (bookmark.view === undefined) {
      const view = {
        ...this.captureView(),
        mode: 'Debug' as const,
        alignment: bookmark.alignment,
        selection: {kind: 'token' as const, turn: bookmark.turn, index: bookmark.index},
        identity: undefined,
      };
      if (this.applyView(view, 'legacy', expected)) {
        this.locateSelectedToken();
        this.bookmarkNotice.set('Opened legacy bookmark; no saved display settings.');
      }
      return;
    }
    if (this.applyView(bookmark.view, 'bookmark', expected))
      this.bookmarkNotice.set('Bookmark restored');
  }
  readonly tokenContent = tokenContent;
  readonly isTemplateToken = isTemplateToken;
  /** Plain token text for labels; a missing token reads as ∅. */
  tokenText(token: CapturedToken | undefined) {
    return token ? capturedTokenLabel(token) : '∅';
  }
  /** Debug token markup: real characters, with whitespace glyphs drawn in place when shown. */
  tokenHtml(token: CapturedToken | undefined) {
    return tokenMarkup(this.tokenText(token), false);
  }
  /** One entry per line break that follows this row, for both runtimes alike. */
  rowBreaks(pair: TokenPair) {
    const count = pairLineBreaks(pair);
    return count ? Array.from({length: count}, (_, i) => i) : NO_BREAKS;
  }
  zeroWidthText(text: string) {
    return /^[\r\n\u200b-\u200d\ufeff]*$/.test(text);
  }
  stageBatch(turn: number, stage: string) {
    return this.state.session()?.batches.find((b) => b.turn === turn && b.phase === stage);
  }
  layerGraphUnavailable(turn: number, stage: string) {
    return this.stageBatch(turn, stage) ? null : 'No graph observation captured for this stage';
  }
  private layerGraphReturnView: TokenDiffViewState | null = null;
  openAnalysisLayerGraph(panel: StageAnalysisPanel) {
    if (this.layerGraphUnavailable(panel.turn, panel.stage)) return;
    const graph = this.overviewGraph(panel);
    const saved = panel.graph;
    const context =
      graph?.layerGraphContext() ??
      (saved?.mode === 'inspection' && saved.inspection
        ? {
            layer: saved.inspection.layer ?? undefined,
            anchor: saved.inspection.anchor ?? 'output',
            position: saved.inspection.position,
          }
        : {layer: saved?.selection?.layer ?? undefined, anchor: undefined, position: undefined});
    const batch =
      context.position === undefined
        ? undefined
        : this.recordedPositionBatch(panel, context.position);
    // Child view queries are already cleared when the parent is destroyed.
    this.layerGraphReturnView = this.captureView();
    this.workspace.graph(panel.turn, panel.stage, {
      layer: context.layer,
      anchor: context.anchor,
      batch: batch?.batch,
    });
  }
  /** Only recorded token/forward links can select an execution observation. */
  recordedPositionBatch(panel: StageAnalysisPanel, position: number) {
    const start =
      this.stageGlobalStart(panel.turn) -
      (panel.stage === 'prefill'
        ? (this.turns().find((turn) => turn.n === panel.turn)?.prefill_tokens ?? 0)
        : 0);
    const pairs =
      panel.stage === 'prefill'
        ? (this.inputPairRows(panel.turn) ?? []).flatMap((row) => [
            ...row.leading,
            ...row.text,
            ...row.trailing,
          ])
        : this.rows(panel.turn);
    const pair = pairs.find(
      (pair) =>
        start +
          (this.alignment() === 'content'
            ? pair.index
            : ((pair.target ?? pair.ref)?.step ?? pair.index)) ===
        position,
    );
    if (!pair) return;
    const linked = (['ref', 'target'] as const).filter((side) => pair[side]?.batch !== undefined);
    if (!linked.length) return;
    return this.state
      .session()
      ?.batches.find(
        (batch) =>
          batch.turn === panel.turn &&
          batch.phase === panel.stage &&
          linked.every((side) => tokenMatchesBatch(pair[side]!, batch, side)),
      );
  }
  openLayerGraph() {
    const batches = this.state.session()?.batches;
    if (this.state.loading() || !batches?.length) return;
    this.layerGraphReturnView = this.captureView();
    if (!batches.some((batch) => batch.batch === this.state.batchId())) {
      this.state.selectBatch(batches[0].batch);
    }
    // Open the shared workspace without a drill-down target so its saved view is restored.
    this.workspace.graphTarget.set(null);
    this.workspace.view.set('Graph Diff');
  }
  kvSnapshot(turn: number, stage: string, token?: CapturedToken) {
    const moment =
      stage === 'start' ? 'prefill_pre' : stage === 'prefill' ? 'prefill_post' : 'terminal';
    let candidates = (this.kvSnapshots().get(this.snapshotKey(turn)) ?? []).filter(
      (s) => s.run === 'target' && s.moment === moment,
    );
    if (token) {
      const batch = this.linkedBatch(token);
      if (!batch) return undefined;
      candidates = (this.kvSnapshots().get(this.snapshotKey(turn)) ?? []).filter(
        (s) =>
          s.run === 'target' &&
          s.forward_id === batchForwardId(batch, 'target') &&
          s.phase === batch.phase &&
          s.runtime === batch.runtime &&
          ['prefill_post', 'decode_post', 'terminal'].includes(s.moment),
      );
    }
    candidates = candidates.slice().sort((a, b) => a.forward_id - b.forward_id);
    return stage === 'start' ? candidates[0] : candidates.at(-1);
  }
  kvReason(turn: number, stage: string, token?: CapturedToken) {
    if (this.kvSnapshot(turn, stage, token)) return 'Open the recorded cache observation';
    const error = this.kvLoadErrors().get(this.snapshotKey(turn));
    if (error) return error;
    const inferred = (this.kvSnapshots().get(this.snapshotKey(turn)) ?? []).some(
      (s) => s.run === 'target' && s.basis,
    );
    return inferred && stage === 'start'
      ? 'No cache snapshot before Prefill: the runtime dump records the cache only after Prefill and at generation end'
      : 'No cache snapshot captured at this boundary';
  }
  openKv(turn: number, stage: string, token?: CapturedToken) {
    const snapshot = this.kvSnapshot(turn, stage, token);
    if (!snapshot) return;
    const entry = {
      sessionId: this.state.captureId(),
      turn,
      moment: snapshot.moment,
      phase: snapshot.phase,
      step: snapshot.step,
      forward_id: snapshot.forward_id,
      runtime: snapshot.runtime,
    };
    const parentStage = stage === 'start' || stage === 'prefill' ? 'prefill' : 'decode';
    this.analysisPanels.update((panels) =>
      numberStagePanels([
        ...panels,
        {
          id: 'analysis-' + ++this.analysisSerial,
          turn,
          stage: parentStage,
          kind: 'kv',
          open: true,
          entry,
        },
      ]),
    );
    this.workspace.mode.set('Debug');
  }
  linkedBatch(token: CapturedToken | undefined) {
    return token?.batch == null
      ? undefined
      : this.state.preview.batches.find(
          (b) => b.turn === this.selection()?.turn && tokenMatchesBatch(token, b, 'target'),
        );
  }
  canInspect(token: CapturedToken | undefined) {
    return !!this.linkedBatch(token);
  }
  inspectToken(token: CapturedToken | undefined) {
    const batch = this.linkedBatch(token);
    if (!batch) return;
    this.workspace.graph(batch.turn, batch.phase);
    this.state.selectBatch(batch.batch);
  }
}
