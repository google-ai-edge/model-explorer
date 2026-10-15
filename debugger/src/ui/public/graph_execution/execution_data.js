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

/* Pure adapter: topology comes from execution graphs; selectable tensors require records. */
(() => {
  'use strict';
  const attributes = values => Object.fromEntries((values || []).map(item => [item.key, item.value]));
  const attr = (key, value) => ({key, value: String(value)});
  const key = (...parts) => JSON.stringify(parts);
  const id = (...parts) => parts.map(value => encodeURIComponent(String(value))).join(':');
  const colors = {
    operation: '#edf0f4', operationBorder: '#c5ccd6', tensor: '#ffffff', tensorBorder: '#b7c9e2',
    hover: '#6089bd', missing: '#f3f6fa', text: '#27364b', operationText: '#4b5563', missingText: '#607087',
  };
  const darkColors = {
    ...colors, operation: '#303134', tensor: '#292a2d', missing: '#282a2d',
    text: '#e8eaed', operationText: '#c4c7c5', missingText: '#9aa0a6',
  };
  const style = kind => ({backgroundColor: kind === 'operation' ? colors.operation : colors.tensor,
    borderColor: kind === 'operation' ? colors.operationBorder : colors.tensorBorder, hoveredBorderColor: colors.hover});
  const severity = (value, metric) => {
    const name = String(metric ?? '').toLowerCase();
    return name === 'cossim' || name === 'cosine_similarity' ? Math.max(0, 1 - value) : Math.abs(value);
  };

  function executionSides(payload) {
    const records = payload.details?.tensors || [], runs = payload.details?.executions || [];
    const refRecord = records.find(record => record.id === payload.reference);
    const targetRecord = records.find(record => record.id === payload.target);
    const selectedRef = runs.find(run => run.id === refRecord?.run);
    const selectedTarget = runs.find(run => run.id === targetRecord?.run);
    const ref = selectedRef || runs.find(run => run.id !== selectedTarget?.id);
    const target = selectedTarget || runs.find(run => run.id !== ref?.id);
    return [{side: 'ref', run: ref, selected: refRecord}, {side: 'target', run: target, selected: targetRecord}];
  }

  function buildExecutionGraphs(payload, choices = {}) {
    const records = payload.details?.tensors || [], lookup = new Map(), recordNodes = new Map();
    const panes = executionSides(payload).map(({side, run, selected}, paneIndex) => {
      const original = run?.graphs.find(graph => graph.id === choices[side])
        || run?.graphs.find(graph => graph.id === selected?.graph) || run?.graphs[0];
      const graphId = id(side, run?.id || '', original?.id || 'empty');
      const nodes = [], operationIds = new Set((original?.nodes || []).map(node => node.id));
      const portIds = new Map(), recordsByPort = new Map(), portsByNode = new Map();
      for (const record of records) {
        if (record.run !== run?.id || record.graph !== original?.id || !operationIds.has(record.node)) continue;
        const port = key(record.node, String(record.output));
        if (!recordsByPort.has(port)) recordsByPort.set(port, []);
        recordsByPort.get(port).push(record);
      }
      for (const node of original?.nodes || []) {
        const ports = new Map((node.outputsMetadata || []).map(output => [String(output.id), output]));
        // A real captured record is also evidence for an output omitted from graph metadata.
        for (const record of records) {
          if (record.run === run?.id && record.graph === original.id && record.node === node.id && !ports.has(String(record.output)))
            ports.set(String(record.output), {id: String(record.output), attrs: [attr('shape', JSON.stringify(record.shape)), attr('dtype', record.dtype)]});
        }
        portsByNode.set(node.id, ports);
        for (const outputId of ports.keys()) portIds.set(key(node.id, outputId), id(side, original.id, 'tensor', node.id, outputId));
      }
      for (const node of original?.nodes || []) {
        const operationId = id(side, original.id, 'op', node.id);
        const hit = {kind: 'operation', side, paneIndex, graphId: original.id, nodeId: node.id,
          rendererGraphId: graphId, rendererNodeId: operationId, label: node.label, node};
        lookup.set(operationId, hit);
        nodes.push({id: operationId, label: node.label, namespace: '', style: style('operation'),
          attrs: [...(node.attrs || []), attr('kind', 'operation'), attr('original_node_id', node.id)],
          outputsMetadata: [...(portsByNode.get(node.id)?.values() || [])],
          incomingEdges: (node.incomingEdges || []).filter(edge => operationIds.has(edge.sourceNodeId)).map(edge => ({
            ...edge, sourceNodeId: portIds.get(key(edge.sourceNodeId, String(edge.sourceNodeOutputId)))
              || id(side, original.id, 'op', edge.sourceNodeId),
            sourceNodeOutputId: String(edge.sourceNodeOutputId), targetNodeInputId: String(edge.targetNodeInputId),
          }))});
        for (const [outputId, output] of portsByNode.get(node.id) || []) {
          const tensorId = portIds.get(key(node.id, outputId)), captured = recordsByPort.get(key(node.id, outputId)) || [];
          const record = captured.find(item => item.id === selected?.id) || (captured.length === 1 ? captured[0] : null);
          const metadata = attributes(output.attrs), label = metadata.tensor_name || metadata.name
            || (captured.length === 1 ? captured[0].id : `${node.label} · ${outputId}`);
          const tensorHit = {...hit, kind: 'tensor', rendererNodeId: tensorId, outputId, label,
            recordId: record?.id || null, recordIds: captured.map(item => item.id), captured: !!record};
          lookup.set(tensorId, tensorHit);
          for (const item of captured) recordNodes.set(item.id, {...tensorHit, recordId: item.id});
          nodes.push({id: tensorId, label: label.replace(/^layer_\d+_/, ''), namespace: '', style: style('tensor'),
            attrs: [attr('kind', 'tensor'), attr('producer', node.id), attr('output_id', outputId),
              attr('capture', captured.length ? `${captured.length} tensor record${captured.length === 1 ? '' : 's'}` : 'Not captured'), ...(output.attrs || [])],
            incomingEdges: [{sourceNodeId: operationId, sourceNodeOutputId: outputId, targetNodeInputId: '0'}],
            inputsMetadata: [{id: '0', attrs: output.attrs || []}], outputsMetadata: [{id: outputId, attrs: output.attrs || []}]});
        }
      }
      return {side, paneIndex, run, original, graph: {id: graphId, nodes}};
    });
    const label = 'Captured execution';
    const graphCollections = [{label, graphs: panes.map(pane => pane.graph)}];
    return {panes, lookup, recordNodes, graphCollections,
      topologyKey: JSON.stringify(graphCollections),
      initialUiState: {paneStates: panes.map((pane, index) => ({deepestExpandedGroupNodeIds: [], selectedNodeId: '',
        selectedGraphId: pane.graph.id, selectedCollectionLabel: label, widthFraction: .5,
        selected: index === 0, flattenLayers: true}))}};
  }

  function buildNodeData(model, payload) {
    const palette = payload.theme === 'dark' ? darkColors : colors;
    const number = payload.metrics?.[payload.metric];
    const valid = typeof number === 'number' && Number.isFinite(number) && !!payload.reference && !!payload.target;
    const amount = valid ? severity(number, payload.metric) : 0;
    const selected = new Set([payload.reference, payload.target].filter(Boolean));
    return Object.fromEntries(model.panes.map(pane => [pane.graph.id, {
      name: payload.metric || 'Metric', hideInAggregatedStatsTable: true, hideInChildrenStatsTable: true,
      results: Object.fromEntries(pane.graph.nodes.map(node => {
        const hit = model.lookup.get(node.id), operation = hit.kind === 'operation';
        const hasMetric = !operation && valid && hit.recordIds.some(recordId => selected.has(recordId));
        return [node.id, {value: operation ? '' : hasMetric ? number : '—',
          bgColor: operation ? palette.operation : hasMetric ? amount > 0 ? (payload.theme === 'dark' ? '#4a2020' : '#f7cfcd') : palette.tensor : palette.missing,
          textColor: operation ? palette.operationText : hasMetric ? palette.text : palette.missingText}];
      })),
    }]));
  }
  globalThis.ExecutionGraphData = {buildExecutionGraphs, buildNodeData, executionSides};
})();
