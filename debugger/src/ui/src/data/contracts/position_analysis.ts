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

import type {KvPositionAlignment} from './kv';
import type {Metric} from './types';
export interface PositionTensorRow {
  position: number;
  layer: number;
  anchor: string;
  status: string;
  shape: number[] | null;
  metrics: Partial<Record<string, Metric>>;
}
export interface PositionAnalysis {
  provenance: {kind: 'captured' | 'synthetic'; label: string} | null;
  rows: PositionTensorRow[];
}

export interface PositionTensorRequest {
  position: number;
  layer: number;
  anchor: string;
  alignment?: KvPositionAlignment | null;
}
/** Source dtype is recorded only when supplied; never inferred from comparison precision. */
export interface PositionTensorSource {
  values: (number | null)[];
  shape: number[];
  dtype: string | null;
}
export interface PositionTensorResponse {
  position: number;
  layer: number;
  anchor: string;
  reference_position: number | null;
  target_position: number | null;
  reference: PositionTensorSource | null;
  target: PositionTensorSource | null;
  metrics: Partial<Record<string, Metric>>;
  status: string;
  provenance: PositionAnalysis['provenance'];
}
