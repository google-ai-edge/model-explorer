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

// Runtime node/port identities are the shared source for PLE Arch and Mapping.
window.pleMapping = (() => {
  const attrs = port => Object.fromEntries((port.attrs || []).map(a => [a.key, a.value]));
  const tensorId = (graph, node, port) => `${graph.id}::${node}::${port}`;
  function prepare(graph, region) {
    const presentation = {...graph.mappingPresentation, pairInputs: true};
    presentation.anchors = {...presentation.anchors};
    const inputs = graph.nodes.find(n => n.id === 'boundary:inputs');
    for (const port of inputs?.outputsMetadata || []) {
      const name = attrs(port).signature_name;
      const anchor = name === 'token_ids' ? 'input_ids' : name === 'embeddings' ? 'main_embeddings' : 'lookup_input';
      presentation.anchors[anchor] = {outputNode: inputs.id, outputPort: port.id};
    }
    // Restore source namespaces; Model Explorer owns grouping and expansion.
    presentation.nativeStyle = true;
    delete presentation.expandedGroups;
    presentation.anchors = Object.fromEntries(Object.entries(presentation.anchors).map(([key, value]) => {
      const {expandedGroups, ...binding} = value;
      return [key, binding];
    }));
    const nodes = graph.nodes.map(node => {
      const source = node.attrs?.find(a => a.key === 'source_namespace')?.value;
      return source == null ? node : {...node, namespace: source};
    });
    return {...graph, nodes, mappingPresentation: presentation};
  }
  function ports(graph, anchor, internal = false) {
    if (!graph) return [];
    const output = graph.mappingPresentation.anchors[anchor];
    return graph.nodes.flatMap(node => (node.outputsMetadata || []).filter(port => {
      const isOutput = output?.outputNode === node.id && output.outputPort === port.id;
      return internal ? !node.id.startsWith('boundary:') && !isOutput : isOutput;
    }).map(port => ({node, port, id: tensorId(graph, node.id, port.id)})));
  }
  function aggregate(metrics, summary) {
    const mode = summary === 'min' ? 'min' : 'max';
    const ready = metrics.filter(m => m?.state === 'ready' && Number.isFinite(m.value));
    const chosen = ready.reduce((a, b) => !a || (mode === 'min' ? b.value < a.value : b.value > a.value) ? b : a, null);
    const pending = metrics.some(m => m?.state === 'pending');
    // Don't show a partial aggregate as if the complete region has finished.
    const result = pending ? {state: 'pending', value: null, text: '…'} : chosen || {state: 'missing', value: null, text: '—'};
    return {...result, mode, available: ready.length, total: metrics.length};
  }
  return {prepare, ports, aggregate};
})();
