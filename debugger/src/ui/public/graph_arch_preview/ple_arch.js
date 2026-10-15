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

// PLE spans an independent lookup signature and a region inside Prefill/Decode.
// Inputs is an input collection: token IDs and main embeddings are distinct ports.
(() => {
  const page = window.SEMANTIC_MODEL.arch_graphs.ple_preparation;
  const lookupBlock = {
    id: 'ple_lookup_block', label: 'Per-Layer Embedding Lookup',
    nodes: ['lookup'], fixed: true,
    summaryMembers: [], mappingAnchor: 'lookup',
  };
  const projectionBlock = {
    id: 'ple_projection_block', label: 'Per-Layer Input Preparation',
    nodes: ['project', 'combine'], fixed: true,
    summaryMembers: ['project'], mappingAnchor: 'combine',
  };
  page.blocks = [lookupBlock, projectionBlock];
  page.fixed = true;
  page.attrs.scope = 'Lookup runs in per_layer_embedder. Projection and Combine run inside the selected Prefill/Decode signature. Runtime fills the lookup results into per_layer_embeddings.';
  const input = page.nodes.find(n => n.id === 'input');
  const output = page.nodes.find(n => n.id === 'output');
  const lookup = page.nodes.find(n => n.id === 'lookup');
  const combine = page.nodes.find(n => n.id === 'combine');
  input.label = 'Inputs';
  const inputTensors = [
    {id: 'input_ids', label: 'Input IDs', kind: 'boundary', inputs: [], mappingRegion: 'lookup', runtimeInput: 'token_ids', tensor: {name: 'Input IDs'}},
    {id: 'main_embeddings', label: 'Main Embeddings', kind: 'boundary', inputs: [], mappingRegion: 'projection', runtimeInput: 'embeddings', tensor: {name: 'Main Embeddings'}},
  ];
  page.nodes.push(...inputTensors);
  lookup.tensor.name = 'Embedding Lookup output';
  lookup.attrs.computation = 'Token ID handling → 35 embedding lookups → concatenate → reshape. Output per token: [1,1,35,256].';
  lookup.attrs.signature = 'per_layer_embedder';
  lookup.attrs.transfer = 'Runtime fills per-token lookup results into the selected Prefill/Decode per_layer_embeddings input.';
  const edge = (tensor, from, to, points, arrow, semanticOnly = false, port, targetPort) => ({tensor, from, to, points: points.map(([x,y]) => ({x,y})), arrow, semanticOnly, port, targetPort, cornerRadius: 8});
  window.ARCH_LAYOUTS.ple_preparation = {collapsed: {
    positions: {
      input: {x: 280, y: 24, w: 80, h: 26},
      [lookupBlock.id]: {x: 210, y: 150, w: 220, h: 58},
      [projectionBlock.id]: {x: 200, y: 310, w: 240, h: 58},
      output: {x: 280, y: 460, w: 80, h: 26},
      'tensor:input_ids': {x: 232, y: 88, w: 176, h: 24},
      'tensor:main_embeddings': {x: 452, y: 88, w: 176, h: 24},
      'tensor:lookup': {x: 232, y: 239, w: 176, h: 24},
      'tensor:combine': {x: 232, y: 402, w: 176, h: 24},
    },
    nodes: [input, {...lookupBlock, kind: 'block', inputs: ['input']}, {...projectionBlock, kind: 'block', inputs: ['input', lookupBlock.id]}, output],
    tensors: [
      ...inputTensors.map(n => ({id: 'tensor:'+n.id, source: n.id, tensor: n.tensor})),
      {id: 'tensor:lookup', source: 'lookup', tensor: lookup.tensor},
      {id: 'tensor:combine', source: 'combine', tensor: combine.tensor},
    ],
    blocks: [],
    edges: [
      edge('input_ids', 'input', lookupBlock.id, [[320,50],[320,150]], true, false, 'ids'),
      edge('main_embeddings', 'input', projectionBlock.id, [[344,50],[344,62],[540,62],[540,339],[440,339]], true, false, 'embeddings', 'right'),
      edge('lookup', lookupBlock.id, 'tensor:lookup', [[320,208],[320,251]], false),
      edge('lookup', 'tensor:lookup', projectionBlock.id, [[320,251],[320,310]], true),
      edge('combine', projectionBlock.id, 'tensor:combine', [[320,368],[320,414]], false),
      edge('combine', 'tensor:combine', 'output', [[320,414],[320,460]], true),
    ],
  }};
})();
