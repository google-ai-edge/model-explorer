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

/**
 * Each analysis' saved panel settings and the metric keys they name, without the analyses'
 * components: what the conversation's saved views and panels keep and check. Its models import
 * it so they stay free of components; the registry gives it to the conversation's components too.
 */
export type {GraphPanelState} from './intermediate_tensors/graph_metric_panel';
export type {LayerActivationInspectionState} from './intermediate_tensors/layer_activation_inspection';
export {parseKvView, type KvViewSnapshot} from './kv/kv_view_state';
export {
  FINAL_LOGIT_METRICS,
  PER_RUNTIME_METRICS,
  isFinalLogitMetric,
  type LogitPanelState,
} from './shared/logit_panel_state';
