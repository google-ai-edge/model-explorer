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

import {validValueRange} from '../kv/kv_panel/kv_scatter_model';
import {TOKEN_METRICS} from './token_metric_metadata';
import {SIDE_TREND_METRICS, FINAL_LOGIT_METRICS, isFinalLogitMetric} from './token_trends';
import {parseKvView, type KvExplicitEntry, type KvViewSnapshot} from '../kv/kv_panel/kv_view_state';
import type {TokenTrendState} from './token_trends';
import type {LayerActivationInspectionState} from '../analysis/layer_activation_inspection';

/** Panel identity is independent of token selection and of the currently selected batch. */
export interface StageAnalysisPanel {
  id: string;
  turn: number;
  stage: 'prefill' | 'decode';
  kind: 'kv' | 'tokens' | 'distribution' | 'graph' | 'activations';
  open: boolean;
  /** Stable one-based identity; UI names display a zero-based suffix. */
  instanceNumber?: number;
  entry?: KvExplicitEntry;
  graph?: {
    mode?: 'comparison' | 'inspection';
    inspection?: LayerActivationInspectionState;
    view?: 'heatmap' | 'trends' | 'graph';
    range?: {start: number; end: number} | null;
    positionReduction?: 'min' | 'max' | 'mean' | null;
    moduleSummary?: 'min' | 'max' | 'output';
    graphDisplay?: {tensorMode?: string; simulation?: boolean; legend: boolean};
    layerChoice?: string;
    trendLayers?: number[];
    selection?: {start: number; end: number; layer: number | null} | null;
    batchId: number | null;
    metric: string;
    anchor: string;
    anchorSelection?: string[] | null;
    /** Missing in legacy state: activations meant output; graph meant internal anchors. */
    anchorScope?: 'output' | 'internal' | 'all';
    selected?: {layer: number; anchor: string} | null;
  };
  kv?: KvViewSnapshot;
  trend?: TokenTrendState;
}
export function validStagePanels(value: unknown): value is StageAnalysisPanel[] {
  if (!Array.isArray(value) || value.length > 100) return false;
  const ids = new Set<string>();
  return value.every((p) => {
    if (
      !p ||
      typeof p !== 'object' ||
      typeof p.id !== 'string' ||
      !/^analysis-\d+$/.test(p.id) ||
      ids.has(p.id)
    )
      return false;
    ids.add(p.id);
    if (
      p.instanceNumber !== undefined &&
      (!Number.isInteger(p.instanceNumber) || p.instanceNumber < 1)
    )
      return false;
    if ((p.kind === 'tokens' || p.kind === 'distribution') && p.stage !== 'decode') return false;
    if (
      p.entry !== undefined &&
      (!p.entry ||
        p.entry.turn !== p.turn ||
        typeof p.entry.moment !== 'string' ||
        typeof p.entry.phase !== 'string' ||
        !Number.isInteger(p.entry.forward_id) ||
        typeof p.entry.runtime !== 'string')
    )
      return false;
    if (
      p.graph !== undefined &&
      (!p.graph ||
        !(p.graph.batchId === null || Number.isInteger(p.graph.batchId)) ||
        typeof p.graph.metric !== 'string' ||
        typeof p.graph.anchor !== 'string' ||
        (p.graph.anchorScope !== undefined &&
          !['output', 'internal', 'all'].includes(p.graph.anchorScope)) ||
        (p.graph.anchorSelection != null &&
          (!Array.isArray(p.graph.anchorSelection) ||
            !p.graph.anchorSelection.length ||
            !p.graph.anchorSelection.every((value: unknown) => typeof value === 'string') ||
            new Set(p.graph.anchorSelection).size !== p.graph.anchorSelection.length)) ||
        (p.graph.view !== undefined && !['heatmap', 'trends', 'graph'].includes(p.graph.view)) ||
        (p.graph.graphDisplay !== undefined &&
          (!p.graph.graphDisplay ||
            (p.graph.graphDisplay.tensorMode !== undefined &&
              !['name', 'metric', 'name-metric'].includes(p.graph.graphDisplay.tensorMode)) ||
            (p.graph.graphDisplay.simulation !== undefined &&
              typeof p.graph.graphDisplay.simulation !== 'boolean') ||
            typeof p.graph.graphDisplay.legend !== 'boolean')) ||
        (p.graph.moduleSummary !== undefined &&
          !['min', 'max', 'output'].includes(p.graph.moduleSummary)) ||
        (p.graph.mode !== undefined && !['comparison', 'inspection'].includes(p.graph.mode)) ||
        (p.graph.inspection !== undefined && !validActivationInspection(p.graph.inspection)) ||
        (p.graph.positionReduction != null &&
          !['min', 'max', 'mean'].includes(p.graph.positionReduction)) ||
        (p.graph.range != null && !validRange(p.graph.range)) ||
        (p.graph.selection != null &&
          (!validRange(p.graph.selection) ||
            !(
              p.graph.selection.layer === null ||
              (Number.isInteger(p.graph.selection.layer) && p.graph.selection.layer >= 0)
            ))) ||
        (p.graph.trendLayers !== undefined &&
          (!Array.isArray(p.graph.trendLayers) ||
            p.graph.trendLayers.some((id: unknown) => !Number.isInteger(id) || Number(id) < 0))) ||
        (p.graph.layerChoice !== undefined && typeof p.graph.layerChoice !== 'string') ||
        (p.graph.selected != null &&
          (!Number.isInteger(p.graph.selected.layer) ||
            p.graph.selected.layer < 0 ||
            typeof p.graph.selected.anchor !== 'string')))
    )
      return false;
    return (
      Number.isInteger(p.turn) &&
      p.turn > 0 &&
      ['prefill', 'decode'].includes(p.stage) &&
      ['kv', 'tokens', 'distribution', 'graph', 'activations'].includes(p.kind) &&
      typeof p.open === 'boolean' &&
      (p.kv === undefined || !!parseKvView(p.kv)) &&
      (p.trend === undefined ||
        (typeof p.trend.open === 'boolean' &&
          (p.trend.samplingView === undefined ||
            ['comparison', 'inspection'].includes(p.trend.samplingView)) &&
          (p.trend.samplingPosition === undefined ||
            (Number.isInteger(p.trend.samplingPosition) && p.trend.samplingPosition >= 0)) &&
          (p.trend.samplingTopK === undefined ||
            (Number.isInteger(p.trend.samplingTopK) &&
              p.trend.samplingTopK >= 1 &&
              p.trend.samplingTopK <= 1000)) &&
          (p.trend.positionRange === undefined || validRange(p.trend.positionRange)) &&
          (p.trend.positionReduction === undefined ||
            ['min', 'max', 'mean'].includes(p.trend.positionReduction)) &&
          (p.trend.logitView === undefined ||
            ['comparison', 'inspection'].includes(p.trend.logitView)) &&
          (p.trend.logitPlot === undefined || ['index', 'scatter'].includes(p.trend.logitPlot)) &&
          (p.trend.logitElementRange === undefined || validRange(p.trend.logitElementRange)) &&
          (p.trend.logitValueRange === undefined || validValueRange(p.trend.logitValueRange)) &&
          (p.trend.logitElementReduction === undefined ||
            ['min', 'max', 'mean'].includes(p.trend.logitElementReduction)) &&
          (p.trend.logitTransform === undefined ||
            ['raw', 'centered'].includes(p.trend.logitTransform)) &&
          (p.trend.logitOrder === undefined ||
            ['index', 'ref', 'target', 'delta', 'absolute_delta'].includes(p.trend.logitOrder)) &&
          (p.trend.logitSortDirection === undefined ||
            ['ascending', 'descending'].includes(p.trend.logitSortDirection)) &&
          [p.trend.logitPosition, p.trend.logitOffset].every(
            (value) => value === undefined || (Number.isInteger(value) && value >= 0),
          ) &&
          [...TOKEN_METRICS, ...SIDE_TREND_METRICS, ...FINAL_LOGIT_METRICS].some(
            (m) => m.key === p.trend.metric,
          ) &&
          Number.isInteger(p.trend.start) &&
          Number.isInteger(p.trend.end) &&
          p.trend.end > p.trend.start))
    );
  });
}

function validRange(value: {start: number; end: number}) {
  return (
    value !== null &&
    typeof value === 'object' &&
    Number.isInteger(value.start) &&
    value.start >= 0 &&
    Number.isInteger(value.end) &&
    value.end > value.start
  );
}

function validActivationInspection(value: LayerActivationInspectionState) {
  return (
    value !== null &&
    typeof value === 'object' &&
    ['index', 'scatter'].includes(value.plot) &&
    Number.isInteger(value.position) &&
    value.position >= 0 &&
    (value.layer === null || (Number.isInteger(value.layer) && value.layer >= 0)) &&
    (value.anchor === undefined ||
      (typeof value.anchor === 'string' &&
        value.anchor.trim().length > 0 &&
        value.anchor.length <= 128)) &&
    (value.reduction === undefined || ['min', 'max', 'mean'].includes(value.reduction)) &&
    (value.visibleRuntimes === undefined ||
      (Array.isArray(value.visibleRuntimes) &&
        value.visibleRuntimes.every((runtime) => runtime === 'ref' || runtime === 'target') &&
        new Set(value.visibleRuntimes).size === value.visibleRuntimes.length)) &&
    (value.elementRange == null || validRange(value.elementRange)) &&
    (value.valueRange == null || validValueRange(value.valueRange)) &&
    (value.selection == null ||
      (Number.isInteger(value.selection.channel) && value.selection.channel >= 0))
  );
}

/** Older bookmarks used one panel kind for both logits metric families. */
export function restoreLogitPanel(panel: StageAnalysisPanel): StageAnalysisPanel {
  if (panel.kind !== 'tokens' && panel.kind !== 'distribution') return panel;
  const distribution =
    (panel.kind === 'distribution' && !['kl', 'js'].includes(panel.trend?.metric ?? '')) ||
    SIDE_TREND_METRICS.some((m) => m.key === panel.trend?.metric);
  const metric = distribution
    ? (SIDE_TREND_METRICS.find((m) => m.key === panel.trend?.metric)?.key ?? 'entropy')
    : isFinalLogitMetric(panel.trend?.metric ?? '') ||
        ['kl', 'js'].includes(panel.trend?.metric ?? '')
      ? panel.trend!.metric
      : 'logit_rmse';
  return {
    ...panel,
    kind: distribution ? 'distribution' : 'tokens',
    ...(distribution !== (panel.kind === 'distribution') ? {instanceNumber: undefined} : {}),
    trend: {...(panel.trend ?? {open: true, start: 0, end: 1}), metric},
  };
}

/** Legacy intermediate panels share the Layer Activation product and numbering. */
export function canonicalAnalysisKind(kind: StageAnalysisPanel['kind']) {
  return kind === 'graph' ? 'activations' : kind;
}

/** Preserve identities, assigning new numbers only to missing or colliding legacy views. */
export function numberStagePanels(panels: StageAnalysisPanel[]): StageAnalysisPanel[] {
  const key = (p: StageAnalysisPanel) => `${p.turn}:${p.stage}:${canonicalAnalysisKind(p.kind)}`;
  const counters = new Map<string, number>();
  for (const panel of panels)
    counters.set(key(panel), Math.max(counters.get(key(panel)) ?? 0, panel.instanceNumber ?? 0));
  const used = new Map<string, Set<number>>();
  return panels.map((panel) => {
    const numbers = used.get(key(panel)) ?? new Set<number>();
    used.set(key(panel), numbers);
    if (panel.instanceNumber !== undefined && !numbers.has(panel.instanceNumber)) {
      numbers.add(panel.instanceNumber);
      return panel;
    }
    const number = (counters.get(key(panel)) ?? 0) + 1;
    counters.set(key(panel), number);
    numbers.add(number);
    return {...panel, instanceNumber: number};
  });
}
