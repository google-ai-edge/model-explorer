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

export interface Edge {
  sourceNodeId: string;
  sourceNodeOutputId: string;
  targetNodeInputId: string;
}
export interface Node {
  id: string;
  label: string;
  namespace: string;
  incomingEdges: Edge[];
  attrs?: {key: string; value: string}[];
  outputsMetadata?: {id: string; attrs: {key: string; value: string}[]}[];
}
export interface Anchor {
  id: string;
  of: string;
  semantic: string;
  label?: string;
  edge?: 'in' | 'out';
  block?: string;
}
export interface Definition {
  kind: string;
  inputs: {id: string; shape: (string | number)[]}[];
  nodes: Node[];
  anchors: Anchor[];
}
export interface Layer {
  def: number;
  attrs: Record<string, {ops: Record<string, unknown>}>;
  from?: Record<string, {layer: number; anchor: string}>;
}
export interface Semantic {
  semantic_graph: Definition[];
  layers: Layer[];
}
export interface Metric {
  value: number | null;
  status: string;
}
export interface ComparisonRow {
  layer: number;
  anchor: string;
  shape: number[] | null;
  reference: string | null;
  target: string | null;
  status: string;
  metrics: Partial<Record<string, Metric>>;
}
export interface SummaryMetric {
  value: number | null;
  valid: number;
  total: number;
}
export interface Comparison {
  batch: number;
  rows: ComparisonRow[];
  layers: {layer: number; metrics: Record<string, SummaryMetric>}[];
}

export interface Overview {
  batches: {batch: number; metrics: Record<string, SummaryMetric>}[];
}
export interface Selection {
  layer: number;
  batch: number;
  semantic: string;
  reference: string;
  target: string;
}
export interface TensorRecord {
  id: string;
  run: string;
  graph: string;
  node: string;
  output: string;
  layer: number;
  batch: number;
  sample: string | null;
  shape: number[];
  dtype: string;
}
export interface SavedMapping {
  status: string;
  record?: Selection;
  comparison?: SelectionResult;
}
export interface Execution {
  id: string;
  runtime?: string;
  graphs: {id: string; nodes: Node[]}[];
}
export interface NodeDetails {
  node: Node;
  anchor: Anchor | null;
  parameters: Record<string, unknown>;
  binding?: unknown;
  sources: {
    file: string;
    start: number;
    end: number;
    evidence_start?: number;
    evidence_end?: number;
    code: string;
    reason: string;
    status: string;
  }[];
  tensors: TensorRecord[];
  executions: Execution[];
  saved: SavedMapping;
}
export interface SelectionResult {
  status: string;
  reference: string;
  target: string;
  shape: number[];
  metrics: Record<string, Metric>;
}
