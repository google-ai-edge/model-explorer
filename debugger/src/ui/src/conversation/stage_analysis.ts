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

import {validValueRange} from '../widgets/charts/scatter_model';
import {TOKEN_METRIC_KEYS} from '../server/contracts/token_analysis';
import {
  PER_RUNTIME_METRICS,
  FINAL_LOGIT_METRICS,
  isFinalLogitMetric,
  parseKvView,
  type GraphPanelState,
  type KvViewSnapshot,
  type LayerActivationInspectionState,
  type LogitPanelState,
} from '../analyses/panel_state';

const VALID_LOGIT_METRICS: ReadonlySet<string> = new Set<string>([
  ...TOKEN_METRIC_KEYS,
  ...PER_RUNTIME_METRICS,
  ...FINAL_LOGIT_METRICS,
]);

/** Panel identity is independent of token selection and of the currently selected batch. */
export interface StageAnalysisPanel {
  id: string;
  turn: number;
  stage: 'prefill' | 'decode';
  kind: 'kv' | 'tokens' | 'distribution' | 'activations';
  open: boolean;
  /** Stable one-based identity; UI names display a zero-based suffix. */
  instanceNumber?: number;
  graph?: GraphPanelState;
  kv?: KvViewSnapshot;
  logit?: LogitPanelState;
}
/** Whether a saved list holds only valid panels, each id once. */
export function validStagePanels(value: unknown): value is StageAnalysisPanel[] {
  return Array.isArray(value) && keptStagePanels(value)?.length === value.length;
}
/**
 * The valid panels of a saved list, in order, each id once (its first panel); null when it is no
 * list of at most 100. A panel that is not valid is dropped alone, the rest kept.
 */
export function keptStagePanels(value: unknown): StageAnalysisPanel[] | null {
  if (!Array.isArray(value) || value.length > 100) {
    return null;
  }
  const ids = new Set<string>();
  return value.filter((p) => {
    if (
      !p ||
      typeof p !== 'object' ||
      typeof p.id !== 'string' ||
      !/^analysis-\d+$/.test(p.id) ||
      ids.has(p.id)
    ) {
      return false;
    }
    if (
      p.instanceNumber !== undefined &&
      (!Number.isInteger(p.instanceNumber) || p.instanceNumber < 1)
    ) {
      return false;
    }
    // No such panel can be added: the server offers token analyses of a Decode only.
    if ((p.kind === 'tokens' || p.kind === 'distribution') && p.stage !== 'decode') {
      return false;
    }
    if (
      p.graph !== undefined &&
      (!p.graph ||
        typeof p.graph.metric !== 'string' ||
        typeof p.graph.anchor !== 'string' ||
        (p.graph.anchorScope !== undefined &&
          !['output', 'internal', 'all'].includes(p.graph.anchorScope)) ||
        (p.graph.anchorSelection != null &&
          (!Array.isArray(p.graph.anchorSelection) ||
            !p.graph.anchorSelection.length ||
            !p.graph.anchorSelection.every((value: unknown) => typeof value === 'string') ||
            new Set(p.graph.anchorSelection).size !== p.graph.anchorSelection.length)) ||
        (p.graph.view !== undefined && !['heatmap', 'lineChart', 'graph'].includes(p.graph.view)) ||
        (p.graph.graphDisplay !== undefined &&
          (!p.graph.graphDisplay ||
            (p.graph.graphDisplay.tensorMode !== undefined &&
              !['name', 'metric', 'name-metric'].includes(p.graph.graphDisplay.tensorMode)) ||
            (p.graph.graphDisplay.simulation !== undefined &&
              typeof p.graph.graphDisplay.simulation !== 'boolean') ||
            typeof p.graph.graphDisplay.legend !== 'boolean')) ||
        (p.graph.moduleSummary !== undefined && !['min', 'max'].includes(p.graph.moduleSummary)) ||
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
        (p.graph.lineChartLayers !== undefined &&
          (!Array.isArray(p.graph.lineChartLayers) ||
            p.graph.lineChartLayers.some(
              (id: unknown) => !Number.isInteger(id) || Number(id) < 0,
            ))) ||
        (p.graph.selected != null &&
          (!Number.isInteger(p.graph.selected.layer) ||
            p.graph.selected.layer < 0 ||
            typeof p.graph.selected.anchor !== 'string')))
    ) {
      return false;
    }
    const valid =
      Number.isInteger(p.turn) &&
      p.turn > 0 &&
      ['prefill', 'decode'].includes(p.stage) &&
      ['kv', 'tokens', 'distribution', 'activations'].includes(p.kind) &&
      typeof p.open === 'boolean' &&
      (p.kv === undefined || !!parseKvView(p.kv)) &&
      (p.logit === undefined ||
        (typeof p.logit.open === 'boolean' &&
          (p.logit.positionRange === undefined || validRange(p.logit.positionRange)) &&
          (p.logit.positionReduction === undefined ||
            ['min', 'max', 'mean'].includes(p.logit.positionReduction)) &&
          (p.logit.logitView === undefined ||
            ['comparison', 'inspection'].includes(p.logit.logitView)) &&
          (p.logit.logitPlot === undefined || ['index', 'scatter'].includes(p.logit.logitPlot)) &&
          (p.logit.logitElementRange === undefined || validRange(p.logit.logitElementRange)) &&
          (p.logit.logitValueRange === undefined || validValueRange(p.logit.logitValueRange)) &&
          (p.logit.logitElementReduction === undefined ||
            ['min', 'max', 'mean'].includes(p.logit.logitElementReduction)) &&
          (p.logit.logitTransform === undefined ||
            ['raw', 'centered'].includes(p.logit.logitTransform)) &&
          (p.logit.logitOrder === undefined ||
            ['index', 'ref', 'target', 'delta', 'absolute_delta'].includes(p.logit.logitOrder)) &&
          (p.logit.logitSortDirection === undefined ||
            ['ascending', 'descending'].includes(p.logit.logitSortDirection)) &&
          [p.logit.logitPosition, p.logit.logitOffset].every(
            (value) => value === undefined || (Number.isInteger(value) && value >= 0),
          ) &&
          VALID_LOGIT_METRICS.has(p.logit.metric)));
    if (valid) {
      ids.add(p.id);
    }
    return valid;
  });
}

function validRange(value: {start: number; end: number}): boolean {
  return (
    value !== null &&
    typeof value === 'object' &&
    Number.isInteger(value.start) &&
    value.start >= 0 &&
    Number.isInteger(value.end) &&
    value.end > value.start
  );
}

function validActivationInspection(value: LayerActivationInspectionState): boolean {
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

/** Older saved panels used one panel kind for both logits metric families. */
export function restoreLogitPanel(panel: StageAnalysisPanel): StageAnalysisPanel {
  if (panel.kind !== 'tokens' && panel.kind !== 'distribution') {
    return panel;
  }
  const logitMetric = panel.logit?.metric ?? '';
  const distribution =
    (panel.kind === 'distribution' && !['kl', 'js'].includes(logitMetric)) ||
    (PER_RUNTIME_METRICS as readonly string[]).includes(logitMetric);
  const metric = distribution
    ? (PER_RUNTIME_METRICS.find((key) => key === logitMetric) ?? 'entropy')
    : isFinalLogitMetric(logitMetric) || logitMetric === 'kl' || logitMetric === 'js'
      ? logitMetric
      : 'logit_rmse';
  return {
    ...panel,
    kind: distribution ? 'distribution' : 'tokens',
    ...(distribution !== (panel.kind === 'distribution') ? {instanceNumber: undefined} : {}),
    logit: {...(panel.logit ?? {open: true}), metric},
  };
}
