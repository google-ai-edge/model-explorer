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

import type {StageAnalysisId} from '../server/contracts/capture';
import {FinalLogitsDetails} from './final_logits/final_logits_details';
import {FinalLogitsLineChart} from './final_logits/final_logits_line_chart';
import {GraphMetricPanel} from './intermediate_tensors/graph_metric_panel';
import {IntermediateTensorsDetails} from './intermediate_tensors/intermediate_tensors_details';
import {KvDetails} from './kv/kv_details';
import {KvPanel} from './kv/kv_panel';
import {SamplingDetails} from './sampling/sampling_details';

/**
 * The analyses a stage of the conversation can show, by the server's analysis id: the kind of
 * analysis panel the conversation keeps for it, the panel that draws it in the stage's analysis
 * column and the body of the details panel that describes it. Final Logits and Sampling
 * Distribution share one Line chart; the conversation's LogitPanel feeds it either one.
 *
 * The rest of the UI reaches analyses/ only through this file, or through panel_state.ts where it
 * must stay free of components.
 */
export const ANALYSES = {
  kv: {kind: 'kv', panel: KvPanel, details: KvDetails},
  final_logits: {kind: 'tokens', panel: FinalLogitsLineChart, details: FinalLogitsDetails},
  sampling: {kind: 'distribution', details: SamplingDetails},
  intermediate_tensors: {
    kind: 'activations',
    panel: GraphMetricPanel,
    details: IntermediateTensorsDetails,
  },
} as const;
/** A kind of analysis panel, as the conversation saves it. */
type AnalysisKind = (typeof ANALYSES)[StageAnalysisId]['kind'];

const ANALYSIS_BY_KIND: Record<AnalysisKind, StageAnalysisId> = {
  kv: 'kv',
  tokens: 'final_logits',
  distribution: 'sampling',
  activations: 'intermediate_tensors',
};

/** The server's id of the analysis a panel kind shows. */
export function analysisOf(kind: AnalysisKind): StageAnalysisId {
  const id = ANALYSIS_BY_KIND[kind];
  if (!id) {
    throw new Error(`Unknown analysis kind: ${String(kind)}`);
  }
  return id;
}

export {
  FINAL_LOGIT_METRICS,
  PER_RUNTIME_METRICS,
  isFinalLogitMetric,
  parseKvView,
  type GraphPanelState,
  type KvViewSnapshot,
  type LayerActivationInspectionState,
  type LogitPanelState,
} from './panel_state';
export {FinalLogitsInspection} from './final_logits/final_logits_inspection';
export {
  FinalLogitsLineChart,
  type LogitPositionSeries,
} from './final_logits/final_logits_line_chart';
export {GraphMetricPanel} from './intermediate_tensors/graph_metric_panel';
export {tensorMetricId} from './intermediate_tensors/position_tensor_model';
export {tensorPointLabel} from './intermediate_tensors/tensor_labels';
export {KvPanel} from './kv/kv_panel';
export {defaultKvView} from './kv/kv_view_state';
export type {StagePoint, StateLocation} from './shared/analysis_details';
export {columnContext} from './shared/column_steps';
export {catalogMetricId, finalLogitKey, isPerRuntimeMetric} from './shared/logit_panel_state';
