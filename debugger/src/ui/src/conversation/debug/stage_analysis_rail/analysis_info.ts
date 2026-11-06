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

import {Injectable, afterNextRender, inject} from '@angular/core';

import {
  type StagePoint,
  type StateLocation,
  catalogMetricId,
  tensorMetricId,
  tensorPointLabel,
} from '../../../analyses/registry';
import {MetricCatalogStore} from '../../../server/metric_catalog';
import type {ExplorerInfoItem} from '../../../widgets/explorer_info_list/explorer_info_list';
import {animateAnalysisLocationFeedback} from '../../../widgets/locate_feedback';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {pairKey} from '../../token_analysis';
import {TokenDiff} from '../token_diff/token_diff';
import {AnalysisPanels} from './analysis_panels';
import {StageAnalyses} from './stage_analyses';

/**
 * What the details say about an analysis view and how they reveal its parts: its name and
 * properties, the token points of its range, and the locators that open and show a chart, a
 * control or a position of it in its rail.
 */
@Injectable()
export class AnalysisInfo {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);
  private readonly analyses = inject(StageAnalyses);
  private readonly metricCatalog = inject(MetricCatalogStore);

  analysisLabel(panel: StageAnalysisPanel): string {
    if (panel.kind === 'kv') {
      return 'KV cache';
    }
    if (panel.kind === 'activations') {
      return 'Intermediate Tensors';
    }
    return panel.kind === 'distribution' ? 'Sampling Distribution' : 'Final Logits';
  }
  analysisInstanceLabel(panel: StageAnalysisPanel): string {
    const current = this.panels.analysisPanels().find((p) => p.id === panel.id) ?? panel;
    const prefix: Record<StageAnalysisPanel['kind'], string> = {
      kv: 'KV Cache Analysis',
      activations: 'Intermediate Tensor Analysis',
      tokens: 'Final Logits Analysis',
      distribution: 'Sampling Distribution Analysis',
    };
    const peers = this.panels
      .panelsFor(current.turn, current.stage)
      .filter((p) => p.kind === current.kind);
    const number =
      current.instanceNumber ?? Math.max(1, peers.findIndex((p) => p.id === current.id) + 1);
    return `${prefix[current.kind]} ${number - 1}`;
  }
  analysisViewProperties(panel: StageAnalysisPanel): ExplorerInfoItem[] {
    const current = this.panels.analysisPanels().find((item) => item.id === panel.id) ?? panel;
    const kv = this.panels.overviewKv(current);
    const graph = this.panels.overviewGraph(current);
    const activationInspection =
      current.kind === 'activations' &&
      (graph?.inspectionMode() ?? current.graph?.mode === 'inspection');
    const activationState = graph?.inspection()?.captureView() ?? current.graph?.inspection;
    const logitsInspection = current.kind === 'tokens' && current.logit?.logitView === 'inspection';
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
          ? current.logit?.logitPlot === 'scatter'
            ? 'Scatter plot'
            : 'Line chart'
          : inspection
            ? plot === 'scatter'
              ? 'Scatter plot'
              : 'Line chart'
            : view === 'graph'
              ? 'Graph Diff'
              : view === 'heatmap' || (!view && ['kv', 'activations'].includes(current.kind))
                ? 'Heatmap'
                : 'Line chart',
      },
    ];
    if (logitsInspection) {
      properties.push({
        label: 'Token Position',
        value: String(
          this.panels.overviewFinalLogits(current)?.position() ??
            current.logit?.logitPosition ??
            '-',
        ),
      });
    } else if (activationInspection) {
      properties.push(
        {label: 'Token Position', value: String(activationState?.position ?? '-')},
        {
          label: 'Tensor Index',
          value: `[Layer: ${activationState?.layer ?? '-'}, Sync point: ${graph?.anchorLabel(activationState?.anchor ?? 'output') ?? tensorPointLabel(activationState?.anchor ?? 'output')}]`,
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
        this.panels.analysisViewportRange(current);
      const metric =
        kv?.metric() ??
        graph?.metric() ??
        current.kv?.metric ??
        current.graph?.metric ??
        current.logit?.metric ??
        (current.kind === 'distribution'
          ? 'entropy'
          : current.kind === 'tokens'
            ? 'logit_rmse'
            : 'cosine_similarity');
      const id = current.kind === 'activations' ? tensorMetricId(metric) : catalogMetricId(metric);
      properties.push(
        {label: 'Token Range', value: `${range.start}–${range.end - 1}`},
        {label: 'Metric', value: this.metricCatalog.lookup().label(id), expandable: true},
      );
    }
    return properties;
  }
  analysisConfig(panel: StageAnalysisPanel): string {
    const kv = this.panels.kvPanels().find((p) => p.panelId() === panel.id);
    if (kv) {
      return `${kv.metricLabel()} · ${kv.view() === 'heatmap' ? 'Heatmap' : 'Line chart'} · Heads: ${kv.head()}`;
    }
    const graph = this.panels.graphPanels().find((p) => p.panelId() === panel.id);
    if (graph?.inspectionMode()) {
      return `Tensor Inspection · ${graph.inspection()?.plot() === 'scatter' ? 'Scatter plot' : 'Line chart'}`;
    }
    if (graph) {
      return `${graph.metricName()} · ${graph.view() === 'graph' ? 'Graph Diff' : graph.view() === 'heatmap' ? 'Heatmap' : 'Line chart'} · Sync points: ${graph.anchor()}`;
    }
    return `${panel.logit?.metric ?? ''} · Line chart`;
  }
  stagePoints(panel: StageAnalysisPanel): StagePoint[] {
    const start = this.tokens.stageGlobalStart(panel.turn);
    const range = this.panels.analysisRange(panel);
    return this.tokens
      .rows(panel.turn)
      .map((pair) => ({
        position:
          start +
          (this.conversation.alignment() === 'content'
            ? pair.index
            : ((panel.kind === 'tokens' ? (pair.ref ?? pair.target) : (pair.target ?? pair.ref))
                ?.step ?? pair.index)),
        refPosition: pair.ref
          ? start + (this.conversation.alignment() === 'content' ? pair.index : pair.ref.step)
          : null,
        targetPosition: pair.target
          ? start + (this.conversation.alignment() === 'content' ? pair.index : pair.target.step)
          : null,
        index: pair.index,
        hasRef: !!pair.ref,
        hasTarget: !!pair.target,
        analysis: this.tokens.tokenAnalysis().get(pairKey(panel.turn, pair)),
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
  ): void {
    // Location reveals the exact analysis instance without changing its inspection.
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        this.panels
          .kvPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateCache(kind, position, layer);
      },
      {injector: this.conversation.injector},
    );
  }
  locateMetric(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (panel.kind === 'kv') {
          this.panels
            .kvPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
        } else if (panel.kind === 'activations') {
          this.panels
            .graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
        } else {
          this.panels
            .tokenPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateMetric();
        }
      },
      {injector: this.conversation.injector},
    );
  }
  locateScopeControl(panel: StageAnalysisPanel, control: 'positions' | 'layers'): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const kv = this.panels.kvPanels().find((p) => p.panelId() === panel.id);
        const graph =
          panel.kind === 'activations'
            ? this.panels.graphPanels().find((p) => p.panelId() === panel.id)
            : undefined;
        if (control === 'positions') {
          if (kv) {
            kv.locatePositions();
          } else if (graph) {
            graph.locateRange();
          } else {
            this.panels
              .tokenPanels()
              .find((p) => p.panelId() === panel.id)
              ?.locateRange();
          }
        } else if (kv) {
          kv.locateLayers();
        } else {
          graph?.locateLayers();
        }
      },
      {injector: this.conversation.injector},
    );
  }
  locateTokenView(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (panel.kind === 'activations') {
          this.panels
            .graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateView();
        } else {
          this.panels
            .tokenPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateView();
        }
      },
      {injector: this.conversation.injector},
    );
  }
  locateAggregation(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        if (panel.kind === 'kv') {
          this.panels
            .kvPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateHeads();
        } else {
          this.panels
            .graphPanels()
            .find((p) => p.panelId() === panel.id)
            ?.locateAggregation();
        }
      },
      {injector: this.conversation.injector},
    );
  }
  locateGraphChart(panel: StageAnalysisPanel, position?: number, layer?: number): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () =>
        this.panels
          .graphPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateChart(position, layer),
      {injector: this.conversation.injector},
    );
  }
  locateFinalChart(panel: StageAnalysisPanel, side?: string, position?: number): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id ? {...p, open: true} : p)),
    );
    afterNextRender(
      () =>
        this.panels
          .tokenPanels()
          .find((p) => p.panelId() === panel.id)
          ?.locateFinal(side, position),
      {injector: this.conversation.injector},
    );
  }
  locateStatePosition(panel: StageAnalysisPanel, value: StateLocation): void {
    if ((panel.kind === 'tokens' || panel.kind === 'distribution') && !value.inspect) {
      this.locateFinalChart(panel, value.kind, value.position);
      return;
    }
    if (panel.kind === 'activations' && !value.inspect) {
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
    const graph = this.panels.overviewGraph(panel);
    const kv = this.panels.overviewKv(panel);
    const range = {start: value.position, end: value.position + 1, layer: value.layer};
    if (value.inspect) {
      if (graph) {
        if (value.anchor) {
          graph.anchor.set(value.anchor);
        }
        graph.select(range);
        this.analyses.inspectMetric(graph);
      } else if (kv) {
        if (value.kind) {
          kv.selectedCurve.set(value.kind);
        }
        if (value.head !== undefined) {
          kv.head.set(String(value.head));
        }
        kv.position.set(value.position);
        kv.selectionEnd.set(value.position + 1);
        kv.column.set(value.layer === null);
        if (value.layer !== null) {
          kv.layer.set(value.layer);
        }
        this.analyses.inspectMetric(kv);
      } else {
        const point = this.stagePoints(panel).find((p) =>
          value.index !== undefined ? p.index === value.index : p.position === value.position,
        );
        if (point) {
          this.analyses.inspectTokenMetric(panel, point.index);
        }
      }
    } else {
      this.panels.analysisHover.set({panelId: panel.id, range});
    }
    const start = this.panels.entireStageRange(panel).start;
    const pairs =
      panel.stage === 'prefill'
        ? (this.tokens.inputPairRows(panel.turn) ?? []).flatMap((r) => [
            ...r.leading,
            ...r.text,
            ...r.trailing,
          ])
        : this.tokens.rows(panel.turn);
    const pair = pairs.find((p) =>
      this.conversation.alignment() === 'content'
        ? start + p.index === value.position
        : [p.ref, p.target].some((t) => t && t.step + start === value.position),
    );
    if (pair) {
      this.conversation.element.nativeElement
        .querySelector(
          `[data-turn="${panel.turn}"][data-row="${pair.index}"]${panel.stage === 'prefill' ? '[data-kind="input"]' : ':not([data-kind])'}`,
        )
        ?.scrollIntoView({block: 'center', behavior: 'smooth'});
    }
  }
  analysisLocationAnimation?: Animation;
  /** Name's locator reveals the whole view without reinterpreting its selection. */
  locateAnalysisArea(panel: StageAnalysisPanel): void {
    this.panels.analysisPanels.update((panels) =>
      panels.map((p) => (p.id === panel.id && !p.open ? {...p, open: true} : p)),
    );
    afterNextRender(
      () => {
        const area = (this.conversation.element.nativeElement as HTMLElement)
          .querySelector('#' + panel.id)
          ?.closest<HTMLElement>('.stage-analysis');
        const frame = area?.querySelector<HTMLElement>('.analysis-location-frame');
        if (!area || !frame) {
          return;
        }
        const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
        area.scrollIntoView({
          block: 'start',
          inline: 'nearest',
          behavior: reduced ? 'instant' : 'smooth',
        });
        this.analysisLocationAnimation?.cancel();
        this.analysisLocationAnimation = animateAnalysisLocationFeedback(frame, reduced);
      },
      {injector: this.conversation.injector},
    );
  }
}
