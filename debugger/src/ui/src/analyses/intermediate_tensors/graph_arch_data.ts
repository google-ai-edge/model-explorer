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

import type {
  AnchorBinding,
  BlockRegion,
  GraphStatus,
  ModelArchitecture,
  OutsideModule,
  Skeleton,
} from '../../server/contracts/artifacts';
import type {Reason} from '../../server/contracts/metrics';
import type {ManualPairs, PairTensor} from '../../server/contracts/mappings';
import type {PositionAnalysis} from '../../server/contracts/position_analysis';
import {reasonText} from '../../server/reasons';

/** The two sides of a Session. */
export type GraphSide = 'ref' | 'target';
/** What the server says of one side's model: where its graph is, and its architecture. */
export interface ModelGraphFacts {
  artifact: string;
  status: GraphStatus;
  architecture: ModelArchitecture | null;
}
/** A tensor's tag at the selected column: the server's value of its point, or why there is none. */
export interface GraphArchTag {
  value: number | null;
  text: string;
  bg: string;
  fg: string;
  /** What the tag says on hover: the point and its value, or the server's reason. */
  title: string;
  /** Whether the Session observes its point: a sync point is set. */
  observed: boolean;
  /** Its tensor's shape in the graph, as the converter writes it ('' if unbound). */
  shape: string;
  /** Where it is in the signature's graph ('' if unbound). */
  source: string;
}
/** An anchor's place in one side's graph, for the execution graph to open at: its tensor's node
 *  and output there, and its tap point (null for a signature input). */
export interface GraphArchBinding {
  node: string;
  output: string;
  point: string | null;
}
/** A pair made by hand of the two models, and its row's tag at the column: the server's value of
 *  its place, or why it has none. */
export interface GraphManualPair {
  id: string;
  label: string;
  ref: PairTensor;
  target: PairTensor;
  tag: GraphArchTag;
}
/** Everything the Graph Diff draws of the model at the selected column, all the server's. */
export interface GraphArchModel {
  state: 'loading' | 'ready' | 'unavailable';
  /** Why the model cannot be drawn, while loading or unavailable. */
  text: string;
  /** The skeleton, when the server has a template of the model, and the modules not in it that
   *  run in the stage. */
  architecture: {
    name: string;
    skeleton: Skeleton;
    ports: Record<string, string>;
    outside: OutsideModule[];
  } | null;
  /** Why there is none: the server's text. */
  noTemplate: string;
  layerCount: number;
  /** The layers the column's signature computes, as the server lists them (its `layer_ids`); with
   *  no signature, those any signature computes. The Graph Diff lists these alone. */
  layerIds: number[];
  /** The signature each side ran for the column; null with `signatureText` saying why. */
  signatures: Record<GraphSide, string | null>;
  signatureText: Record<GraphSide, string>;
  /** Per side, per layer: its scope in the column's signature's graph, as the server names it. */
  scopes: Record<GraphSide, Record<number, string>>;
  /** Per layer, per anchor: the ref side's tag. */
  anchors: Record<number, Record<string, GraphArchTag>>;
  /** Per layer, per semantic operator the server pairs across two runtimes (a place that is no
   *  anchor, such as `norm.attn.in` or `attn/q_proj`): its row's tag at the column. */
  ops: Record<number, Record<string, GraphArchTag>>;
  /** Per side, per layer, per anchor: where it is in the column's signature's graph. */
  bindings: Record<GraphSide, Record<number, Record<string, GraphArchBinding>>>;
  /** Per side, per layer, per block: its region of the column's signature's graph. */
  blocks: Record<GraphSide, Record<number, Record<string, BlockRegion>>>;
  /** Per side, the tag of each point the Session observes, at the column. */
  points: Record<GraphSide, Record<string, GraphArchTag>>;
  /** Per ref point at the column, the target point the server compares it with (its row's): the
   *  same point, or, across two runtimes, the other side's point of the same anchor. */
  pointPairs: Record<string, string>;
  /** Why a point the Session does not observe has no value. */
  notObserved: string;
  /** Where the values come from, in the server's words. */
  provenance: string;
  /** The pairs made by hand of the two models, each with its tag at the column. */
  manualPairs: GraphManualPair[];
}

const SIDES: GraphSide[] = ['ref', 'target'];
const EMPTY_BG = 'var(--metric-empty-bg)';
const EMPTY_FG = 'var(--muted)';

/** What the Graph Diff needs to draw a model, from the server's answers: its graph's status and
 *  architecture, each side's, and the stage's intermediate tensors at the selected column. It
 *  decides nothing: every value is a server cell's, every reason the server's text. */
export function graphArchModel(input: {
  facts: Partial<Record<GraphSide, ModelGraphFacts>> | null;
  loading: boolean;
  error: string;
  analysis: PositionAnalysis | null;
  /** The pairs made by hand of the Session's two models, as the server lists them. */
  manual?: ManualPairs | null;
  /** The stage the column is of: 'prefill' or 'decode'. */
  stage: string;
  position: number | null;
  metric: string;
  /** The catalog's words for a reason code. */
  reason: (code: string) => string;
  format: (value: number) => string;
  color: (value: number) => string;
}): GraphArchModel {
  const {facts, analysis, stage, position, metric} = input;
  const empty: GraphArchModel = {
    state: 'loading',
    text: '',
    architecture: null,
    noTemplate: '',
    layerCount: 0,
    layerIds: [],
    signatures: {ref: null, target: null},
    signatureText: {ref: '', target: ''},
    scopes: {ref: {}, target: {}},
    anchors: {},
    ops: {},
    bindings: {ref: {}, target: {}},
    blocks: {ref: {}, target: {}},
    points: {ref: {}, target: {}},
    pointPairs: {},
    notObserved: '',
    provenance: analysis?.provenance?.label ?? '',
    manualPairs: [],
  };
  if (input.error) {
    return {...empty, state: 'unavailable', text: input.error};
  }
  const ref = facts?.ref;
  if (!ref || input.loading) {
    return {...empty, text: 'Loading the model’s graph…'};
  }
  if (ref.status.status !== 'completed') {
    const text =
      ref.status.status === 'unavailable'
        ? (ref.status.text ?? '')
        : ref.status.status === 'failed'
          ? `The model’s graph could not be converted: ${ref.status.error ?? ''}`
          : ref.status.status === 'converting'
            ? 'The model’s graph is being converted.'
            : 'The model’s graph is converted when a Session of it starts.';
    return {...empty, state: 'unavailable', text};
  }
  const column = analysis?.columns?.find((c) => c.position === position) ?? null;
  const notProduced = input.reason('tensor_position_not_produced');
  const signatures = {
    ref: column?.ref?.signature ?? null,
    target: column?.target?.signature ?? null,
  };
  const signatureText = {
    ref: signatures.ref ? '' : notProduced,
    target: signatures.target ? '' : notProduced,
  };
  const architecture = ref.architecture;
  const skeleton = architecture?.skeleton ?? null;
  const layerCount = skeleton
    ? skeleton.layers.length
    : Math.max(
        0,
        ...(ref.status.signatures ?? [])
          .filter((s) => !signatures.ref || s.name === signatures.ref)
          .map((s) => s.layers),
      );
  const layerIds = [
    ...new Set(
      (ref.status.signatures ?? [])
        .filter((s) => !signatures.ref || s.name === signatures.ref)
        .flatMap((s) => s.layer_ids ?? []),
    ),
  ].sort((a, b) => a - b);
  const notObserved =
    reasonText(analysis?.reasons) ||
    (analysis?.not_observed ? reasonText([analysis.not_observed]) : '') ||
    input.reason('tensor_point_not_observed');
  // The tag of each observed point at the column, per side: its cell's value or reasons.
  const points: Record<GraphSide, Record<string, GraphArchTag>> = {ref: {}, target: {}};
  if (analysis) {
    for (const point of analysis.points) {
      const row = analysis.rows.find(
        (r) => r.position === position && r.layer === point.layer && r.anchor === point.anchor,
      );
      for (const side of SIDES) {
        if (row?.points && row.points[side] !== undefined && row.points[side] !== point.id) {
          continue;
        }
        const cell = row?.metrics[metric];
        const value = cell?.value ?? null;
        const why =
          reasonText(cell?.reasons) ||
          reasonText(row?.reasons) ||
          reasonText(point.reasons as Reason[]) ||
          (row ? '' : 'The stage has no cell of this point at the column.');
        points[side][point.id] = tag(point.id, value, why, true, '', '', input);
      }
    }
  }
  // A pair made by hand is no automatic pair: its rows are its own (manualPairs).
  const manualAnchors = new Set(
    (input.manual?.pairs ?? []).flatMap((pair) => (pair.place ? [pair.place.anchor] : [])),
  );
  const pointPairs: Record<string, string> = {};
  for (const row of analysis?.rows ?? []) {
    if (
      row.position === position &&
      row.points?.ref &&
      row.points.target &&
      !manualAnchors.has(row.anchor)
    ) {
      pointPairs[row.points.ref] = row.points.target;
    }
  }
  const manualPairs: GraphManualPair[] = (input.manual?.pairs ?? []).map((pair) => {
    const place = pair.place;
    const row = place
      ? analysis?.rows.find(
          (r) => r.position === position && r.layer === place.layer && r.anchor === place.anchor,
        )
      : undefined;
    const cell = row?.metrics[metric];
    const why =
      reasonText(cell?.reasons) ||
      reasonText(row?.reasons) ||
      pair.text ||
      (place ? 'The stage has no cell of this pair at the column.' : '');
    return {
      id: pair.id,
      label: pair.label,
      ref: pair.ref,
      target: pair.target,
      tag: tag(pair.label, cell?.value ?? null, why, !!place, '', '', input),
    };
  });
  const bindings: GraphArchModel['bindings'] = {ref: {}, target: {}};
  const scopes: GraphArchModel['scopes'] = {ref: {}, target: {}};
  const blocks: GraphArchModel['blocks'] = {ref: {}, target: {}};
  const anchors: GraphArchModel['anchors'] = {};
  for (const side of SIDES) {
    const signature = signatures[side];
    const layers = signature ? facts?.[side]?.architecture?.bindings?.[signature] : undefined;
    for (const entry of layers ?? []) {
      const placed: Record<string, GraphArchBinding> = {};
      for (const [id, binding] of Object.entries(entry.anchors)) {
        if (binding.bound && binding.node !== undefined) {
          placed[id] = {
            node: binding.node,
            output: binding.output ?? '0',
            point: binding.point ?? null,
          };
        }
      }
      bindings[side][entry.layer] = placed;
      if (entry.scope) {
        scopes[side][entry.layer] = entry.scope;
      }
      blocks[side][entry.layer] = Object.fromEntries((entry.blocks ?? []).map((b) => [b.block, b]));
    }
  }
  const ops: GraphArchModel['ops'] = {};
  const anchorIds = new Set(
    skeleton
      ? skeleton.layers.flatMap((l) => skeleton.semantic_graph[l.def].anchors.map((a) => a.id))
      : [],
  );
  for (const row of analysis?.rows ?? []) {
    if (row.position !== position || anchorIds.has(row.anchor) || manualAnchors.has(row.anchor)) {
      continue;
    }
    const cell = row.metrics[metric];
    (ops[row.layer] ??= {})[row.anchor] = tag(
      row.points?.ref ?? '',
      cell?.value ?? null,
      reasonText(cell?.reasons) || reasonText(row.reasons),
      true,
      '',
      '',
      input,
    );
  }
  if (skeleton) {
    const definition = (layer: number) => skeleton.semantic_graph[skeleton.layers[layer].def];
    const layers = signatures.ref ? architecture?.bindings?.[signatures.ref] : undefined;
    for (let layer = 0; layer < skeleton.layers.length; layer++) {
      anchors[layer] = {};
      const layerBinding = layers?.find((l) => l.layer === layer);
      for (const anchor of definition(layer).anchors) {
        const binding = layerBinding?.anchors[anchor.id];
        anchors[layer][anchor.id] = anchorTag(
          binding,
          signatures.ref,
          signatureText.ref,
          points.ref,
          notObserved,
          input,
        );
      }
    }
  }
  return {
    ...empty,
    state: 'ready',
    architecture:
      architecture?.architecture && skeleton
        ? {
            name: architecture.architecture.name,
            skeleton,
            ports: architecture.architecture.ports ?? {},
            outside: (architecture.architecture.outside ?? []).filter((m) =>
              m.stages.includes(stage),
            ),
          }
        : null,
    noTemplate: architecture?.text || architecture?.error || ref.status.architecture_text || '',
    layerCount,
    layerIds,
    signatures,
    signatureText,
    scopes,
    anchors,
    ops,
    bindings,
    blocks,
    points,
    pointPairs,
    notObserved,
    manualPairs,
  };
}

/** An anchor's tag: its bound point's value, or why it has none. */
function anchorTag(
  binding: AnchorBinding | undefined,
  signature: string | null,
  noSignature: string,
  points: Record<string, GraphArchTag>,
  notObserved: string,
  input: {format: (value: number) => string; color: (value: number) => string},
): GraphArchTag {
  if (!signature) {
    return tag('', null, noSignature, false, '', '', input);
  }
  if (!binding) {
    return tag('', null, `${signature} has no binding of it.`, false, '', '', input);
  }
  if (!binding.bound) {
    return tag('', null, binding.reason ?? '', false, '', '', input);
  }
  const shape = binding.tensor_shape ?? '';
  const source = `${signature} · node ${binding.node}:${binding.output} · ${binding.tensor_name ?? ''}`;
  if (!binding.point || !binding.observable) {
    return tag(binding.point ?? '', null, binding.reason ?? '', false, shape, source, input);
  }
  const observed = points[binding.point];
  if (!observed) {
    return tag(binding.point, null, notObserved, false, shape, source, input);
  }
  return {...observed, shape, source};
}

function tag(
  point: string,
  value: number | null,
  why: string,
  observed: boolean,
  shape: string,
  source: string,
  input: {format: (value: number) => string; color: (value: number) => string},
): GraphArchTag {
  if (value === null || !Number.isFinite(value)) {
    return {
      value: null,
      text: '—',
      bg: EMPTY_BG,
      fg: EMPTY_FG,
      title: [point, why || 'No value'].filter(Boolean).join(' · '),
      observed,
      shape,
      source,
    };
  }
  const text = input.format(value);
  return {
    value,
    text,
    bg: input.color(value),
    fg: 'var(--text)',
    title: [point, text].filter(Boolean).join(' · '),
    observed,
    shape,
    source,
  };
}
