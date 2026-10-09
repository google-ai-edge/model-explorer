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

import {Metric} from './types';

export interface KvTensor {
  kind: string;
  shape: number[];
  dtype: string;
  resource_id?: string;
  storage_status?: string;
}
export interface KvLayer {
  layer: number;
  state: string;
  layer_type?: string;
  reason?: string;
  capacity: number | null;
  valid_length: number | null;
  logical_start: number | null;
  logical_end: number | null;
  tensors: KvTensor[];
}
export interface KvSnapshot {
  run: string;
  runtime: string;
  turn: number;
  snapshot_id: number | string;
  moment: string;
  phase: string;
  step: number | null;
  forward_id: number;
  state: string;
  reason?: string;
  preparation_status?: string;
  terminal_status?: string;
  storage_complete?: boolean;
  processed_token_count: number | null;
  /** Set on snapshots whose logical context was derived from a runtime dump, never observed. */
  basis?: string;
  context_basis?: string;
  logical_context_identity?: string;
  layers: KvLayer[];
}
export interface GenerationRecord {
  run: string;
  turn: number;
  status: string;
  stop_reason: string | null;
  generated_token_count: number;
  processed_token_count: number | null;
  pending_token_ids: number[];
}
export interface Telemetry {
  format_version: number;
  forwards: unknown[];
  kv_snapshots: KvSnapshot[];
  token_records: unknown[];
  generations: GenerationRecord[];
  resources: unknown[];
}
export interface ResourceComparison {
  reference: string;
  target: string;
  status: string;
  shape?: number[];
  comparison_basis?: string;
  metrics: Partial<Record<string, Metric>>;
}
export interface ResourcePreview {
  resource_id: string;
  shape: number[];
  dtype: string;
  offset: number;
  total: number;
  values: (number | string | boolean)[];
}
export interface ExplicitObservation {
  turn: number;
  phase: string;
  step?: number;
  forward_id?: number;
  signature?: string;
  pos_offset?: number;
  compared_position_range?: number[];
}
export interface ExplicitPair {
  pair_id: string;
  scope?: 'module' | 'kv';
  owner_layer?: number;
  kind?: string;
  status: string;
  observation: ExplicitObservation;
  compared_position_range?: number[];
  semantic: string;
  comparison_basis: string;
  weight_equivalence: string;
  quantization_profile: unknown;
  shape: number[];
  accepted_token_count: number;
  processed_token_count: number;
  pending_token_id: number;
  metrics: Partial<Record<string, Metric>>;
  runs: Record<
    string,
    {
      runtime: string;
      observation?: ExplicitObservation;
      model: Record<string, unknown>;
      original_tensor: {
        shape: number[];
        dtype: string;
        path: string;
        key: string;
        sha256: string;
      };
      layout: string[];
      view: {start: number; stop: number; step: number}[];
      axis_order?: number[];
      dequantization?: {scale: number; zero_point: number; formula: string};
    }
  >;
}
