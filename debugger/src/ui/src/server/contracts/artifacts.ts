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

/** POST /api/artifacts/upload, and GET or POST /api/artifacts/{id}/scan. */
export interface TapPoint {
  id: string;
  signature: string;
  op: number;
  output: number;
  tensor: number;
  tensor_name: string;
  shape: number[];
  dtype: string;
  operator: string;
  selectable: boolean;
  reason: string;
  /**
   * What to know before choosing it, in the server's words: its signature may not run in a
   * chat, its shape holds the KV length, its signature outputs it already; '' for nothing.
   */
  note: string;
}
/** One signature of a scanned model artifact and when LiteRT-LM runs it. */
export interface TapSignature {
  name: string;
  /** When LiteRT-LM runs it: 'always', 'long_prompts', 'speculative_decoding' or 'unknown'. */
  runs: string;
  note: string;
}
/** Result of scanning a model artifact's tap points. */
export interface TapScan {
  status: string;
  error?: string;
  /**
   * What observing this model's tensors costs on disk, in the server's words: the copy turning
   * the Model Server on writes, how long it lives, and the size of the Session's own copy when
   * the scan names the Session.
   */
  copy_note?: string;
  /** The most points of one model a Session observes, its anchors among them. */
  max_points?: number;
  points?: TapPoint[];
  recommended?: string[];
  signatures?: TapSignature[];
}

/**
 * GET /api/artifacts/{id}/anchors: the anchors a Session of a LiteRT-LM model captures by
 * default, every decoder layer's that its graph binds, in the server's words.
 */
export interface TapAnchors {
  /**
   * 'pending' until the model's graph is converted, as a Session of it first turns on; 'none'
   * for a model of no architecture the server has; `text` says why.
   */
  status: 'found' | 'pending' | 'none';
  text: string;
  /** How many tensor outputs are captured, each once. */
  points: number;
  copy_note: string;
  signatures: Array<{
    name: string;
    points: number;
    layers: number[];
    text: string;
    /** Each anchor left out, with the reason: one line per reason. */
    skipped: string[];
  }>;
}

/**
 * GET /api/artifacts/{id}/counterparts?pytorch=&points=: how a Session of the LiteRT-LM model
 * and the PyTorch one pairs each point chosen, in the server's words.
 */
export interface TapCounterparts {
  /**
   * 'pending' until the model's graph is converted; 'none' for a model of no architecture the
   * server has; `text` says why.
   */
  status: 'found' | 'pending' | 'none';
  text: string;
  points: Array<{
    id: string;
    /**
     * 'anchor', 'module' (the PyTorch module of its semantic operator), 'manual' (a pair made by
     * hand of the two models names it), or '' (not paired).
     */
    paired: 'anchor' | 'module' | 'manual' | '';
    layer: number | null;
    place: string;
    label: string;
    module: string;
    edge: string;
    /** How it pairs, or why it does not, as the user reads it. */
    text: string;
  }>;
  /**
   * Once found: what the PyTorch side captures for the points beside its anchors, as its box in
   * the editor says it.
   */
  pytorch_text?: string;
}

/**
 * GET /api/models/compare?ref=&target=: the Session editor's hint of its two models, by the
 * analyses' own rules.
 */
export interface ModelComparison {
  /** 'same_file', 'same_architecture' (two runtimes of one architecture) or 'different'. */
  code: 'same_file' | 'same_architecture' | 'different';
  /** What of the two compares, as the user reads it. */
  text: string;
  /** What the listing calls the two models; the editor suggests a name by it. */
  label: string;
  /**
   * What the editor says of two LiteRT-LM sides of one file, which share one tensor capture; ''
   * when the sides do not share one.
   */
  capture_shared: string;
}

/** One operator inside a decoder layer, as the server lists it (GET /api/models/ops). */
export interface LayerOp {
  /** The id a Session's `tap_ops` names it by, e.g. 'attn/q_proj' or 'norm.ffn.in'. */
  id: string;
  /** Its name in its block, e.g. 'q_proj' or 'Pre-RMSNorm'. */
  name: string;
  /** What it computes, e.g. 'Linear', 'RMSNorm', 'GELU' or 'Mul'. */
  kind: string;
  /** Its place's label in Intermediate Tensors. */
  label: string;
  /** Whether both sides have it in some layer; one that is not cannot be ticked. */
  available: boolean;
  /** Where a side lacks it and why; '' if every layer of both sides has it. */
  reason: string;
  /** What it adds to the capture per generated token, both sides together. */
  size: string;
  /** The same per side. */
  size_detail: string;
}

/** The operators inside the layers of a Session's two models (GET /api/models/ops). */
export interface LayerOps {
  /** 'found', 'pending' (a LiteRT-LM model's graph is not converted yet) or 'none'. */
  status: 'found' | 'pending' | 'none';
  /** Why the list is not known; '' once found. */
  text: string;
  title: string;
  description: string;
  /** By block, in the skeleton's order. */
  blocks: {name: string; ops: LayerOp[]}[];
  /** What holds of every operator. */
  notes: string[];
  /** What those ticked add, as the user reads it. */
  total: {bytes: number; text: string};
  /** Why those ticked cannot be captured; '' if they can. */
  refusal: string;
}

/** One signature of a model's converted graph (GET /api/artifacts/{id}/graph). */
export interface GraphSignature {
  name: string;
  /** How many layers its points are in. */
  layers: number;
  /** The layers it has operators of, ascending: the ones it computes (a prefill only those whose
   *  KV cache it fills). */
  layer_ids: number[];
}
/** GET /api/artifacts/{id}/graph: where a model's graph is, converted as a Session of it starts. */
export interface GraphStatus {
  /** 'unavailable': the server draws no graph of this model, e.g. a PyTorch one; `text` says
   *  why. */
  status: 'idle' | 'converting' | 'completed' | 'failed' | 'unavailable';
  /** Of 'unavailable': why, as the server words it. */
  text?: string;
  error?: string;
  signatures?: GraphSignature[];
  /** Per signature, the stem its layers' scopes are named by (`layer` names `layer_3`; a PyTorch
   *  model's module paths, `layers/`, name `layers/3`). */
  layer_stems?: Record<string, string>;
  architecture_text?: string;
}
/** A semantic skeleton's node (model_arch_explore's schema): one stage of a layer. */
export interface SkeletonNode {
  id: string;
  label: string;
  namespace: string;
  incomingEdges: {sourceNodeId: string; sourceNodeOutputId: string; targetNodeInputId: string}[];
  outputsMetadata: {id: string; attrs: {key: string; value: string}[]}[];
  attrs: {key: string; value: string}[];
  ops: {id: string; op: string}[];
}
/** A skeleton's anchor: a tensor of the layer the execution graph is aligned to it by. */
export interface SkeletonAnchor {
  id: string;
  semantic: string;
  of: string;
  block?: string;
}
/** Semantic block graph definition for one layer archetype in the skeleton. */
export interface SkeletonDefinition {
  kind: string;
  inputs: Array<{id: string; shape: Array<string | number>; source: string; label?: string}>;
  outputs: Array<{id: string; shape: Array<string | number>}>;
  nodes: SkeletonNode[];
  anchors: SkeletonAnchor[];
}
/** Per-layer instantiation of a `SkeletonDefinition` in the model's skeleton. */
export interface SkeletonLayer {
  def: number;
  attrs: Record<string, {ops: Record<string, Record<string, unknown>>}>;
  from?: {input?: {layer: number; anchor: string}; kv?: {layer: number}};
}
/** Full semantic architecture skeleton for a model family. */
export interface Skeleton {
  semantic_ops: Record<string, string>;
  semantic_graph: SkeletonDefinition[];
  layers: SkeletonLayer[];
}
/** Where one anchor of one layer is in a signature's graph, or why it is nowhere. */
export interface AnchorBinding {
  bound: boolean;
  /** Why it is unbound, or why its tensor cannot be observed; '' otherwise. */
  reason?: string;
  checked?: boolean;
  node?: string;
  output?: string;
  tensor_name?: string;
  tensor_shape?: string;
  /** Its tap point's id; null for a signature input, no operator's output. */
  point?: string | null;
  observable?: boolean;
  layer_output?: boolean;
}
/** A block of one layer in a signature's graph: the operators between its anchors, or, when an
 *  anchor is not bound, the layer's operators with the reason. */
export interface BlockRegion {
  block: string;
  /** The anchor whose tensor it reads; null if the skeleton gives it no one anchor. */
  from: string | null;
  /** The anchor it ends at. */
  to: string;
  /** 'anchors': the operators on a path between them; 'layer': the layer's, by namespace. */
  by: 'anchors' | 'layer';
  /** Node ids, in graph order. */
  nodes: string[];
  /** Why it is the layer's operators; '' when it is the anchors'. */
  reason: string;
}
/** A module of the model that is not in its skeleton, why it is not drawn, and the stages it runs
 *  in ('prefill', 'decode'), with why. */
export interface OutsideModule {
  id: string;
  name: string;
  text: string;
  stages: string[];
  stages_evidence: string;
}
/** GET /api/artifacts/{id}/architecture: the model's skeleton and its anchors in its graphs. */
export interface ModelArchitecture {
  status: string;
  architecture: {
    id: string;
    name: string;
    skeleton: string;
    /** Why each input port no anchor is of is bound to nothing. */
    ports: Record<string, string>;
    outside?: OutsideModule[];
  } | null;
  family: string | null;
  /** Why the model has no skeleton, e.g. 'No architecture template exists for this model…'. */
  text: string;
  /** Why its skeleton could not be bound. */
  error: string;
  skeleton: Skeleton | null;
  /** Per signature, per layer of the skeleton: its scope in the graph, per anchor its binding and
   *  per block its region. */
  bindings: Record<
    string,
    {
      layer: number;
      /** The layer's scope component in the signature's graph (`layer_3`, `layers/3`); null when
       *  the graph names no layers. */
      scope?: string | null;
      anchors: Record<string, AnchorBinding>;
      blocks?: BlockRegion[];
    }[]
  >;
}
/** A Model Explorer graph node, as GET /api/artifacts/{id}/graph/{signature} serves it: each
 *  operator output's attrs carry its point's facts (`point_id`, `observable`, `reason`). */
export interface ExecutionGraphNode {
  id: string;
  label: string;
  namespace?: string;
  attrs?: {key: string; value: string}[];
  incomingEdges?: {sourceNodeId: string; sourceNodeOutputId: string; targetNodeInputId: string}[];
  outputsMetadata?: {id: string; attrs?: {key: string; value: string}[]}[];
  [key: string]: unknown;
}
/** GET /api/artifacts/{id}/graph/{signature}: one signature's graph collection. */
export interface ExecutionGraphCollection {
  label: string;
  graphs: {id: string; nodes: ExecutionGraphNode[]; [key: string]: unknown}[];
}
