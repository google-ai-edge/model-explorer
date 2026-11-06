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

/** GET /api/position-analysis, /api/position-range and /api/position-tensor: a stage's
 *  intermediate tensors, the points the Session observes as the server reads them. */

import type {ComparisonAlignment, ComparisonStage} from './capture';
import type {ElementStatistics, KvHeadChannel, KvRangeBin, KvRangeRow, LogicalTensor} from './kv';
import type {Reason} from './metrics';

/** A metric's value, or null with the reasons why. */
export interface Metric {
  value: number | null;
  reasons: Reason[];
}

/** What every answer repeats of its request: the turn, the stage, and the alignment whose rows
 *  are the columns, or null when a column is the position on both sides. */
export interface TensorStage {
  turn: number;
  stage: ComparisonStage;
  alignment: ComparisonAlignment | null;
}
/** One place at one column: both sides' tensors compared. A metric is by its catalog id, in its
 *  unit in GET /api/metrics: `relative_l2` is a percent (1 is 1%), not a ratio. */
export interface PositionTensorRow {
  position: number;
  layer: number;
  /** `output` for the layer's output, else the point's id. */
  anchor: string;
  status: string;
  reasons: Reason[];
  /** The compared values' shape, where both sides have one. */
  shape: number[] | null;
  metrics: Partial<Record<string, Metric>>;
  /** Per side that has a value, the id of the point it is of. */
  points?: Partial<Record<'ref' | 'target', string>>;
}
/** The forward pass a side reads a column from: its signature and step. */
export interface ColumnStep {
  signature: string;
  step: number;
}
/** Per column, the forward pass each side reads it from; null where none produced it. */
export interface PositionColumn {
  position: number;
  ref: ColumnStep | null;
  target: ColumnStep | null;
}
/** A point the Session observes, as the server read it from the model. */
export interface ObservedPoint {
  id: string;
  layer: number;
  anchor: string;
  signature: string;
  tensor_name: string;
  shape: number[];
  context_axes: number[];
  dtype: string;
  /** Its node in the model's graph: the signature's graph, the operator, its output; null for a
   *  point of no graph (a PyTorch runner's anchor). */
  graph_node?: {graph: string; node: string; output: string} | null;
  /**
   * Every place (layer, anchor) it is of: two runtimes are paired by the architecture's anchors,
   * and a LiteRT-LM point may be one layer's output and the next one's input.
   */
  places?: Array<{layer: number; anchor: string}>;
  /** How many of the stage's cells hold it. */
  cells: number;
  /** Why no cell holds it; empty when one does. */
  reasons: Reason[];
}
/** Intermediate tensor comparison response from GET /api/position-analysis. */
export interface PositionAnalysis extends TensorStage {
  label: string;
  /** The stage's columns, [start, end). */
  start: number;
  end: number;
  layers: number[];
  /**
   * The sync points; `default` for an architecture's anchor (a layer's input, each block's
   * output, its output), not an operator chosen inside the layers or a pair made by hand.
   */
  anchors: Array<{id: string; label: string; default?: boolean}>;
  points: ObservedPoint[];
  pairing: {status: 'paired' | 'not_paired' | 'none'; reasons: Reason[]};
  rows: PositionTensorRow[];
  columns?: PositionColumn[];
  /** Why a point of the model that `points` does not list has no cell. */
  not_observed?: Reason;
  /** Why each point chosen of a LiteRT-LM side beside a PyTorch one is paired with nothing: PyTorch
   *  has no counterpart for it, and only the anchors are captured there. */
  unpaired?: Reason[];
  provenance: {kind: 'captured' | 'synthetic'; label: string} | null;
  /** Why the stage has no tensors; empty when it has. */
  reasons: Reason[];
}

/** One layer's statistics over a bin; for a min or max over sync points, the anchor of the
 *  extremum at its position; and why the bin's positions without a value have none. */
export type TensorRangeRow = KvRangeRow;
export interface PositionRangeResponse extends TensorStage, KvRangeBin {
  anchors: string[];
  reduction: 'min' | 'max' | 'mean';
  layers: number[];
  bins: KvRangeBin[];
  rows: TensorRangeRow[];
  /** Each layer over the whole range, one bin. */
  totals: TensorRangeRow[];
  metadata: {ref: LogicalTensor; target: LogicalTensor; output: LogicalTensor};
}

export interface PositionTensorRequest {
  position: number;
  layer: number;
  anchor: string;
}
/** A side's tensor of an inspected place: where it is and what it was stored as. */
export interface PositionTensorSource {
  point: string;
  signature: string;
  /** The step that produced the column's token, and its row of it (0 for a decode step). */
  step: number;
  row: number;
  /** The position that row fed: the one before the column's. */
  reads?: number;
  tensor_name: string;
  /** The compared values' shape; `source_shape` the point's in the model. */
  shape: number[];
  source_shape: number[];
  dtype: string;
  comparison_dtype: string;
}
export interface PositionTensorResponse extends TensorStage, PositionTensorRequest {
  reference_position: number | null;
  target_position: number | null;
  reference: PositionTensorSource | null;
  target: PositionTensorSource | null;
  /** Whether both sides have values of one shape: then each element has a delta. */
  comparable: boolean;
  channels: KvHeadChannel[];
  statistics: ElementStatistics;
  /** As PositionTensorRow's: `relative_l2` is a percent. */
  metrics: Partial<Record<string, Metric>>;
  status: string;
  reasons: Reason[];
  provenance: PositionAnalysis['provenance'];
}
