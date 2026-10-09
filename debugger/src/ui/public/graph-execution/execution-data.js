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

/* Pure adapter: topology comes from execution graphs; selectable tensors
 * require records. */

(() => {
  'use strict';
  const LIGHT_COLORS = {
    operation: '#edf0f4',
    operationBorder: '#c5ccd6',
    tensor: '#ffffff',
    tensorBorder: '#b7c9e2',
    hover: '#6089bd',
    missing: '#f3f6fa',
    text: '#27364b',
    operationText: '#4b5563',
    missingText: '#607087',
  };
  const DARK_COLORS = {
    ...LIGHT_COLORS,
    operation: '#303134',
    tensor: '#292a2d',
    missing: '#282a2d',
    text: '#e8eaed',
    operationText: '#c4c7c5',
    missingText: '#9aa0a6',
  };
  function attributes(values) {
    return Object.fromEntries(
        (values || []).map((item) => [item.key, item.value]));
  }
  function attr(key, value) {
    return {key, value: String(value)};
  }
  function compositeKey(...parts) {
    return JSON.stringify(parts);
  }
  function encodeId(...parts) {
    return parts.map((value) => encodeURIComponent(String(value))).join(':');
  }
  function nodeStyle(kind) {
    return {
      backgroundColor: kind === 'operation' ? LIGHT_COLORS.operation :
                                              LIGHT_COLORS.tensor,
      borderColor: kind === 'operation' ? LIGHT_COLORS.operationBorder :
                                          LIGHT_COLORS.tensorBorder,
      hoveredBorderColor: LIGHT_COLORS.hover,
    };
  }
  function severity(value, metric) {
    const name = String(metric ?? '').toLowerCase();
    return name === 'cossim' || name === 'cosine_similarity' ?
        Math.max(0, 1 - value) :
        Math.abs(value);
  }
  /**
   * Resolves reference and target run records from the active selection
   * payload.
   */
  function executionSides(payload) {
    const records = payload.details?.tensors || [];
    const runs = payload.details?.executions || [];
    const refRecord = records.find((record) => record.id === payload.reference);
    const targetRecord = records.find((record) => record.id === payload.target);
    const selectedRef = runs.find((run) => run.id === refRecord?.run);
    const selectedTarget = runs.find((run) => run.id === targetRecord?.run);
    const ref =
        selectedRef || runs.find((run) => run.id !== selectedTarget?.id);
    const target = selectedTarget || runs.find((run) => run.id !== ref?.id);
    return [
      {side: 'ref', run: ref, selected: refRecord},
      {side: 'target', run: target, selected: targetRecord},
    ];
  }
  /**
   * Builds dual-pane execution graphs and hit-testing indices from capture
   * records.
   */
  function buildExecutionGraphs(payload, choices = {}) {
    const records = payload.details?.tensors || [];
    const lookup = new Map();
    const recordNodes = new Map();
    const panes =
        executionSides(payload).map(({side, run, selected}, paneIndex) => {
          const original =
              run?.graphs.find((graph) => graph.id === choices[side]) ||
              run?.graphs.find((graph) => graph.id === selected?.graph) ||
              run?.graphs[0];
          const originalGraphId = original?.id || 'empty';
          const graphId = encodeId(side, run?.id || '', originalGraphId);
          const nodes = [];
          if (!original) {
            return {
              side,
              paneIndex,
              run,
              original,
              graph: {id: graphId, nodes}
            };
          }
          const operationIds = new Set(original.nodes.map((node) => node.id));
          const portIds = new Map();
          const recordsByPort = new Map();
          const portsByNode = new Map();
          for (const record of records) {
            if (record.run !== run?.id || record.graph !== original.id ||
                !operationIds.has(record.node)) {
              continue;
            }
            const port = compositeKey(record.node, String(record.output));
            const portRecords = recordsByPort.get(port);
            if (portRecords) {
              portRecords.push(record);
            } else {
              recordsByPort.set(port, [record]);
            }
          }
          for (const node of original.nodes) {
            const ports = new Map((node.outputsMetadata || [])
                                      .map(
                                          (output) =>
                                              [String(output.id),
                                               output,
            ]));
            // A real captured record is also evidence for an output omitted
            // from graph metadata.
            for (const record of records) {
              if (record.run === run?.id && record.graph === original.id &&
                  record.node === node.id &&
                  !ports.has(String(record.output))) {
                ports.set(String(record.output), {
                  id: String(record.output),
                  attrs: [
                    attr('shape', JSON.stringify(record.shape)),
                    attr('dtype', record.dtype),
                  ],
                });
              }
            }
            portsByNode.set(node.id, ports);
            for (const outputId of ports.keys()) {
              portIds.set(
                  compositeKey(node.id, outputId),
                  encodeId(side, original.id, 'tensor', node.id, outputId));
            }
          }
          for (const node of original.nodes) {
            const operationId = encodeId(side, original.id, 'op', node.id);
            const hit = {
              kind: 'operation',
              side,
              paneIndex,
              graphId: original.id,
              nodeId: node.id,
              rendererGraphId: graphId,
              rendererNodeId: operationId,
              label: node.label,
              node,
            };
            lookup.set(operationId, hit);
            nodes.push({
              id: operationId,
              label: node.label,
              namespace: '',
              style: nodeStyle('operation'),
              attrs: [
                ...(node.attrs || []),
                attr('kind', 'operation'),
                attr('original_node_id', node.id),
              ],
              outputsMetadata: [...(portsByNode.get(node.id)?.values() || [])],
              incomingEdges:
                  (node.incomingEdges || [])
                      .filter((edge) => operationIds.has(edge.sourceNodeId))
                      .map(
                          (edge) => ({
                            ...edge,
                            sourceNodeId:
                                portIds.get(compositeKey(
                                    edge.sourceNodeId,
                                    String(edge.sourceNodeOutputId))) ||
                                encodeId(
                                    side, original.id, 'op', edge.sourceNodeId),
                            sourceNodeOutputId: String(edge.sourceNodeOutputId),
                            targetNodeInputId: String(edge.targetNodeInputId),
                          })),
            });
            for (const [outputId, output] of portsByNode.get(node.id) || []) {
              const tensorId = portIds.get(compositeKey(node.id, outputId)) ||
                  encodeId(side, original.id, 'tensor', node.id, outputId);
              const captured =
                  recordsByPort.get(compositeKey(node.id, outputId)) || [];
              const record =
                  captured.find((item) => item.id === selected?.id) ||
                  (captured.length === 1 ? captured[0] : null);
              const metadata = attributes(output.attrs);
              const label = metadata['tensor_name'] || metadata['name'] ||
                  (captured.length === 1 ? captured[0].id :
                                           `${node.label} · ${outputId}`);
              const tensorHit = {
                ...hit,
                kind: 'tensor',
                rendererNodeId: tensorId,
                outputId,
                label,
                recordId: record?.id || null,
                recordIds: captured.map((item) => item.id),
                captured: !!record,
              };
              lookup.set(tensorId, tensorHit);
              for (const item of captured) {
                recordNodes.set(item.id, {...tensorHit, recordId: item.id});
              }
              nodes.push({
                id: tensorId,
                label: label.replace(/^layer_\d+_/, ''),
                namespace: '',
                style: nodeStyle('tensor'),
                attrs: [
                  attr('kind', 'tensor'),
                  attr('producer', node.id),
                  attr('output_id', outputId),
                  attr(
                      'capture',
                      captured.length ? `${captured.length} tensor record${
                                            captured.length === 1 ? '' : 's'}` :
                                        'Not captured'),
                  ...(output.attrs || []),
                ],
                incomingEdges: [
                  {
                    sourceNodeId: operationId,
                    sourceNodeOutputId: outputId,
                    targetNodeInputId: '0',
                  },
                ],
                inputsMetadata: [{id: '0', attrs: output.attrs || []}],
                outputsMetadata: [{id: outputId, attrs: output.attrs || []}],
              });
            }
          }
          return {side, paneIndex, run, original, graph: {id: graphId, nodes}};
        });
    const label = 'Captured execution';
    const graphCollections = [
      {label, graphs: panes.map((pane) => pane.graph)},
    ];
    return {
      panes,
      lookup,
      recordNodes,
      graphCollections,
      topologyKey: JSON.stringify(graphCollections),
      initialUiState: {
        paneStates: panes.map((pane, index) => ({
                                deepestExpandedGroupNodeIds: [],
                                selectedNodeId: '',
                                selectedGraphId: pane.graph.id,
                                selectedCollectionLabel: label,
                                widthFraction: 0.5,
                                selected: index === 0,
                                flattenLayers: true,
                              })),
      },
    };
  }
  /**
   * Builds node color and metric provider overlays for the active pair
   * selection.
   */
  function buildNodeData(model, payload) {
    const palette = payload.theme === 'dark' ? DARK_COLORS : LIGHT_COLORS;
    const metricName = payload.metric || '';
    const metricValue = payload.metrics?.[metricName];
    const valid = typeof metricValue === 'number' &&
        Number.isFinite(metricValue) && Boolean(payload.reference) &&
        Boolean(payload.target);
    const amount = valid ? severity(metricValue, payload.metric) : 0;
    const alertBgColor = payload.theme === 'dark' ? '#4a2020' : '#f7cfcd';
    const metricBgColor = amount > 0 ? alertBgColor : palette.tensor;
    const selected = new Set(
        [payload.reference, payload.target].filter((id) => Boolean(id)));
    return Object.fromEntries(model.panes.map(
        (pane) =>
            [pane.graph.id,
             {
               name: payload.metric || 'Metric',
               hideInAggregatedStatsTable: true,
               hideInChildrenStatsTable: true,
               results: Object.fromEntries(pane.graph.nodes.map((node) => {
                 const hit = model.lookup.get(node.id);
                 if (hit?.kind === 'operation') {
                   return [
                     node.id,
                     {
                       value: '',
                       bgColor: palette.operation,
                       textColor: palette.operationText,
                     },
                   ];
                 }
                 const hasMetric = valid &&
                     (hit?.recordIds ||
                      []).some((recordId) => selected.has(recordId));
                 return [
                   node.id,
                   {
                     value: hasMetric ? metricValue : '—',
                     bgColor: hasMetric ? metricBgColor : palette.missing,
                     textColor: hasMetric ? palette.text : palette.missingText,
                   },
                 ];
               })),
             },
    ]));
  }
  const executionGraphDataAdapter = {
    buildExecutionGraphs,
    buildNodeData,
    executionSides,
  };
  if (typeof window !== 'undefined') {
    window.ExecutionGraphData = executionGraphDataAdapter;
  }
})();
