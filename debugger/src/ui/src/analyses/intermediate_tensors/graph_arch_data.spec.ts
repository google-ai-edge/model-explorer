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
  GraphStatus,
  ModelArchitecture,
  Skeleton,
} from '../../server/contracts/artifacts';
import type {PositionAnalysis, PositionTensorRow} from '../../server/contracts/position_analysis';
import {graphArchModel, type ModelGraphFacts} from './graph_arch_data';

const REASONS: Record<string, string> = {
  tensor_position_not_produced: 'No forward pass of this turn produced the token at the position.',
  tensor_point_not_observed: 'This Session does not observe the point.',
};
/** One layer definition with an input anchor, a residual anchor and an output anchor. */
const skeleton: Skeleton = {
  semantic_ops: {},
  semantic_graph: [
    {
      kind: 'decoder',
      inputs: [{id: 'input', shape: ['B', 'S', 8], source: 'residual'}],
      outputs: [{id: 'output', shape: ['B', 'S', 8]}],
      nodes: [],
      anchors: [
        {id: 'input', semantic: 'layer input', of: 'input:0'},
        {id: 'intermediate.1', semantic: 'residual after attention', of: 'add:0'},
        {id: 'output', semantic: 'layer output', of: 'scale:0'},
      ],
    },
  ],
  layers: [
    {def: 0, attrs: {}},
    {def: 0, attrs: {}, from: {input: {layer: 0, anchor: 'output'}}},
  ],
};
function bound(point: string | null, extra: Partial<AnchorBinding> = {}): AnchorBinding {
  return {
    bound: true,
    checked: true,
    node: point?.split(':')[1] ?? '40',
    output: '0',
    tensor_name: 'layer/x',
    tensor_shape: 'float32[1,1,8]',
    point,
    observable: point !== null,
    reason: point === null ? 'It is the input embeddings of decode; it cannot be observed.' : '',
    ...extra,
  };
}
const EMBEDDING = {
  id: 'embedding',
  name: 'Embedding',
  text: 'Not in the decoder skeleton.',
  stages: ['prefill', 'decode'],
  stages_evidence: 'Every text signature reads it.',
};
const OUTPUT_HEAD = {
  id: 'output_head',
  name: 'Output Head',
  text: 'Not in the decoder skeleton.',
  stages: ['decode'],
  stages_evidence: 'The prefill signatures produce no logits.',
};
const architecture: ModelArchitecture = {
  status: 'completed',
  architecture: {
    id: 'gemma4',
    name: 'Gemma 4',
    skeleton: 'tiny',
    ports: {},
    outside: [EMBEDDING, OUTPUT_HEAD],
  },
  family: 'Gemma 4',
  text: '',
  error: '',
  skeleton,
  bindings: {
    decode: [
      {
        layer: 0,
        anchors: {
          input: bound(null),
          'intermediate.1': bound('decode:5:0'),
          output: bound('decode:9:0', {layer_output: true}),
        },
      },
      {
        layer: 1,
        scope: 'layer_1',
        anchors: {
          input: bound('decode:9:0', {layer_output: true}),
          'intermediate.1': bound('decode:15:0'),
          output: bound('decode:19:0', {layer_output: true}),
        },
        blocks: [
          {
            block: 'Attention',
            from: 'input',
            to: 'intermediate.1',
            by: 'anchors',
            nodes: ['10', '15'],
            reason: '',
          },
        ],
      },
    ],
    prefill: [
      {layer: 0, anchors: {}},
      {
        layer: 1,
        anchors: {
          input: {bound: false, reason: 'prefill has no operator of layer 1.'},
          'intermediate.1': {bound: false, reason: 'prefill has no operator of layer 1.'},
          output: {bound: false, reason: 'prefill has no operator of layer 1.'},
        },
      },
    ],
  },
};
const status: GraphStatus = {
  status: 'completed',
  signatures: [
    {
      name: 'decode',
      layers: 2,
      layer_ids: [0, 1],
    },
    {
      name: 'prefill',
      layers: 1,
      layer_ids: [0],
    },
  ],
  layer_stems: {decode: 'layer', prefill: 'layer'},
};
const facts: ModelGraphFacts = {artifact: 'model-a', status, architecture};
function row(
  position: number,
  layer: number,
  anchor: string,
  value: number | null,
  point?: string,
): PositionTensorRow {
  return {
    position,
    layer,
    anchor,
    status: value === null ? 'unavailable' : 'ok',
    reasons:
      value === null ? [{code: 'tensor_not_captured', side: 'ref', text: 'Not captured.'}] : [],
    shape: [8],
    metrics: {cosine_similarity: {value, reasons: []}},
    points: point ? {ref: point, target: point} : {},
  };
}
const analysis: PositionAnalysis = {
  turn: 1,
  stage: 'decode',
  alignment: null,
  label: 'Intermediate Tensors',
  start: 4,
  end: 6,
  layers: [0],
  anchors: [{id: 'output', label: 'Layer output'}],
  points: [
    {
      id: 'decode:9:0',
      layer: 0,
      anchor: 'output',
      signature: 'decode',
      tensor_name: 'x',
      shape: [8],
      context_axes: [],
      dtype: 'float32',
      cells: 2,
      reasons: [],
    },
    {
      id: 'decode:5:0',
      layer: 0,
      anchor: 'decode:5:0',
      signature: 'decode',
      tensor_name: 'x',
      shape: [8],
      context_axes: [],
      dtype: 'float32',
      cells: 2,
      reasons: [],
    },
  ],
  pairing: {status: 'paired', reasons: []},
  rows: [
    row(4, 0, 'output', 0.5, 'decode:9:0'),
    row(4, 0, 'decode:5:0', null),
    row(5, 0, 'output', 0.25, 'decode:9:0'),
  ],
  columns: [
    {position: 4, ref: {signature: 'decode', step: 1}, target: {signature: 'decode', step: 1}},
    {position: 5, ref: null, target: null},
  ],
  not_observed: {
    code: 'tensor_point_not_observed',
    side: null,
    text: REASONS['tensor_point_not_observed'],
  },
  provenance: {kind: 'captured', label: 'Compared by the server'},
  reasons: [],
};
function model(position: number, overrides: Partial<Parameters<typeof graphArchModel>[0]> = {}) {
  return graphArchModel({
    facts: {ref: facts, target: facts},
    loading: false,
    error: '',
    analysis,
    stage: 'decode',
    position,
    metric: 'cosine_similarity',
    reason: (code) => REASONS[code] ?? '',
    format: (value) => value.toFixed(2),
    color: () => 'heat',
    ...overrides,
  });
}

describe('graphArchModel', () => {
  it('tags each anchor with its point’s cell, or the server’s reason', () => {
    const found = model(4);
    expect(found.state).toBe('ready');
    expect(found.layerCount).toBe(2);
    // The layers the decode computes, the server's: all of them.
    expect(found.layerIds).toEqual([0, 1]);
    expect(found.signatures).toEqual({ref: 'decode', target: 'decode'});
    const layer0 = found.anchors[0];
    // A signature input: no operator's output, the server says why.
    expect(layer0.input).toEqual(
      expect.objectContaining({value: null, text: '—', observed: false}),
    );
    expect(layer0.input.title).toContain('cannot be observed');
    // Observed, no value at the column: its cell's reasons.
    expect(layer0['intermediate.1'].value).toBeNull();
    expect(layer0['intermediate.1'].observed).toBe(true);
    expect(layer0['intermediate.1'].title).toContain('Not captured.');
    // Observed, with a value.
    expect(layer0.output).toEqual(
      expect.objectContaining({value: 0.5, text: '0.50', bg: 'heat', observed: true}),
    );
    expect(layer0.output.shape).toBe('float32[1,1,8]');
    // Layer 1's input is layer 0's output: the same point, the same cell.
    expect(found.anchors[1].input.value).toBe(0.5);
    // Not observed: the server's words.
    expect(found.anchors[1]['intermediate.1'].title).toBe(
      'decode:15:0 · This Session does not observe the point.',
    );
    // Where each anchor is, for the execution graph to open at.
    expect(found.bindings.ref[1].output).toEqual({
      node: '19',
      output: '0',
      point: 'decode:19:0',
    });
    expect(found.points.target['decode:9:0'].value).toBe(0.5);
    // Each block's region, the server's, per side.
    expect(found.blocks.ref[1].Attention.nodes).toEqual(['10', '15']);
    expect(found.blocks.target[1].Attention.by).toBe('anchors');
    expect(found.blocks.ref[0]).toEqual({});
    // Each layer's scope in the graph, as the server names it; none where it names none.
    expect(found.scopes.ref).toEqual({1: 'layer_1'});
    // The points the server compares at the column, its rows'.
    expect(found.pointPairs).toEqual({'decode:9:0': 'decode:9:0'});
    // The modules not in the skeleton that run in the stage: all of them in decode.
    expect(found.architecture?.outside).toEqual([EMBEDDING, OUTPUT_HEAD]);
  });

  it('tags each semantic operator two runtimes pair, by its row', () => {
    const paired: PositionAnalysis = {
      ...analysis,
      rows: [
        ...analysis.rows,
        {
          ...row(4, 0, 'norm.attn.in', 0.75),
          points: {ref: 'decode:x.input_layernorm:out', target: 'decode:3:0'},
        },
        row(4, 1, 'attn/q_proj', null),
      ],
    };
    const found = model(4, {analysis: paired});
    expect(found.ops[0]['norm.attn.in']).toEqual(
      expect.objectContaining({value: 0.75, text: '0.75', observed: true}),
    );
    expect(found.ops[0]['norm.attn.in'].title).toBe('decode:x.input_layernorm:out · 0.75');
    expect(found.ops[1]['attn/q_proj'].title).toContain('Not captured.');
    // Anchors are the anchors' tags, not operators'.
    expect(found.ops[0].output).toBeUndefined();
  });

  it('tags each pair made by hand by its own row, or the server’s words', () => {
    const tensor = (point: string) => ({
      signature: 'decode',
      node: point.split(':')[1],
      output: '0',
      point,
    });
    const manual = {
      status: 'pairs' as const,
      text: '',
      pairs: [
        {
          id: 'p1',
          label: 'o_proj input ↔ attn out',
          created: '2026-10-20T00:00:00+00:00',
          ref: tensor('decode:5:0'),
          target: tensor('decode:7:0'),
          place: {layer: 0, anchor: 'manual:p1'},
          text: '',
        },
        {
          id: 'p2',
          label: 'not observed',
          created: '2026-10-20T00:00:00+00:00',
          ref: tensor('decode:6:0'),
          target: tensor('decode:8:0'),
          place: null,
          text: 'not observed: Target decode:8:0 is not observed in this Session.',
        },
      ],
    };
    const paired: PositionAnalysis = {
      ...analysis,
      rows: [
        ...analysis.rows,
        {
          ...row(4, 0, 'manual:p1', 0.9),
          points: {ref: 'decode:5:0', target: 'decode:7:0'},
        },
      ],
    };
    const found = model(4, {analysis: paired, manual});
    const [placed, unplaced] = found.manualPairs;
    expect(placed.tag).toEqual(expect.objectContaining({value: 0.9, text: '0.90', observed: true}));
    expect(placed.ref.node).toBe('5');
    expect(unplaced.tag.value).toBeNull();
    expect(unplaced.tag.title).toContain('is not observed in this Session');
    // Its row is no automatic pair, nor a semantic operator's.
    expect(found.pointPairs['decode:5:0']).toBeUndefined();
    expect(found.ops[0]?.['manual:p1']).toBeUndefined();
  });

  it('says why where no forward pass produced the column', () => {
    const found = model(5);
    expect(found.signatures.ref).toBeNull();
    expect(found.anchors[0].output.title).toBe(REASONS['tensor_position_not_produced']);
    expect(found.bindings.ref).toEqual({});
    // No signature ran: the layers any signature computes.
    expect(found.layerIds).toEqual([0, 1]);
  });

  it('passes on a graph that did not run a layer as unbound', () => {
    const prefill: PositionAnalysis = {
      ...analysis,
      columns: [
        {
          position: 4,
          ref: {signature: 'prefill', step: 0},
          target: {signature: 'prefill', step: 0},
        },
      ],
    };
    const found = model(4, {analysis: prefill, stage: 'prefill'});
    expect(found.anchors[1].output.title).toBe('prefill has no operator of layer 1.');
    // The prefill computes layer 0 alone, as the server says: layer 1 is not listed.
    expect(found.layerIds).toEqual([0]);
    expect(found.layerCount).toBe(2);
    // Only the modules the server says run in prefill: no output head.
    expect(found.architecture?.outside).toEqual([EMBEDDING]);
  });

  it('gives the server’s text for a model without a template', () => {
    const none: ModelArchitecture = {
      ...architecture,
      architecture: null,
      skeleton: null,
      bindings: {},
      text: 'No architecture template exists for this model: its graph is of none of them.',
    };
    const found = model(4, {
      facts: {ref: {...facts, architecture: none}, target: {...facts, architecture: none}},
    });
    expect(found.architecture).toBeNull();
    expect(found.noTemplate).toContain('No architecture template exists for this model');
    expect(found.layerCount).toBe(2);
    expect(found.anchors).toEqual({});
  });

  it('is loading, or unavailable with the server’s words', () => {
    expect(model(4, {facts: null}).state).toBe('loading');
    const failed = model(4, {
      facts: {ref: {...facts, status: {status: 'failed', error: 'The converter crashed.'}}},
    });
    expect(failed.state).toBe('unavailable');
    expect(failed.text).toContain('The converter crashed.');
    const text = 'The execution graph of a PyTorch model is not drawn yet.';
    const undrawn = model(4, {facts: {ref: {...facts, status: {status: 'unavailable', text}}}});
    expect(undrawn.state).toBe('unavailable');
    expect(undrawn.text, 'the server’s text, as it is').toBe(text);
  });
});
