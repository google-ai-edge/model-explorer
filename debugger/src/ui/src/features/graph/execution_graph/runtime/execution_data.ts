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

/** Identifies the reference or target side of a dual-pane execution comparison. */
export type ExecutionSide = 'ref' | 'target';

/** Key-value metadata attribute attached to an execution graph node or port. */
export declare interface ExecutionAttribute {
  key: string;
  value: string;
}

/** Input or output tensor port metadata on an execution graph operation node. */
export declare interface ExecutionPortMetadata {
  id: string;
  attrs?: ExecutionAttribute[];
}

/** Directed edge connecting a producer node output port to a consumer input port. */
export declare interface ExecutionIncomingEdge {
  sourceNodeId: string;
  sourceNodeOutputId: string;
  targetNodeInputId: string;
}

/** Operation or synthetic tensor node rendered inside an execution graph pane. */
export declare interface ExecutionGraphNode {
  id: string;
  label: string;
  namespace?: string;
  style?: {
    backgroundColor: string;
    borderColor: string;
    hoveredBorderColor: string;
  };
  attrs?: ExecutionAttribute[];
  inputsMetadata?: ExecutionPortMetadata[];
  outputsMetadata?: ExecutionPortMetadata[];
  incomingEdges: ExecutionIncomingEdge[];
}

/** Subgraph topology specification containing nodes for a single execution graph. */
export declare interface ExecutionGraphSpec {
  id: string;
  nodes: ExecutionGraphNode[];
}

/** Captured execution run metadata and its associated subgraph specifications. */
export declare interface ExecutionRunRecord {
  id: string;
  runtime?: string;
  graphs: ExecutionGraphSpec[];
}

/** Captured tensor artifact metadata linking a tensor record to its producer node and port. */
export declare interface ExecutionTensorRecord {
  id: string;
  run: string;
  graph: string;
  node: string;
  output: string | number;
  shape?: number[];
  dtype?: string;
}

/** Identifies a selected graph operation on the Reference or Target side. */
export declare interface ExecutionOperation {
  side: ExecutionSide;
  nodeId: string;
  graphId: string;
}

/** Identifies a selected tensor record on the Reference or Target side. */
export declare interface ExecutionNodeSelection {
  side: ExecutionSide;
  id: string;
}

/** Outbound message emitted when the execution graph iframe runtime finishes initializing. */
export declare interface ExecutionReadyMessage {
  type: 'ready';
}

/** Outbound message emitted when a tensor node is selected inside the execution graph iframe. */
export declare interface ExecutionTensorSelectedMessage {
  type: 'tensorSelected';
  hit: ExecutionNodeSelection;
}

/** Outbound message emitted when an operation node is selected inside the execution graph iframe. */
export declare interface ExecutionOperationSelectedMessage {
  type: 'operationSelected';
  hit: ExecutionOperation;
}

/** Union of all outbound messages posted from the execution graph iframe to the host. */
export type ExecutionOutboundMessage =
  | ExecutionReadyMessage
  | ExecutionTensorSelectedMessage
  | ExecutionOperationSelectedMessage;

/** Complete execution comparison state sent from the host UI to the execution graph view. */
export declare interface ExecutionPayload {
  details?: {
    tensors?: ExecutionTensorRecord[];
    executions?: ExecutionRunRecord[];
  } | null;
  reference?: string;
  target?: string;
  metric?: string;
  metrics?: Record<string, number | null>;
  mapping?: boolean;
  operation?: ExecutionOperation | null;
  theme?: 'light' | 'dark';
}

/** Resolved hit-testing record mapping a rendered node ID back to its operation or tensor. */
export declare interface ExecutionHit {
  kind: 'operation' | 'tensor';
  side: ExecutionSide;
  paneIndex: number;
  graphId: string;
  nodeId: string;
  rendererGraphId: string;
  rendererNodeId: string;
  label: string;
  node: ExecutionGraphNode;
  outputId?: string;
  recordId?: string | null;
  recordIds?: string[];
  captured?: boolean;
}

/** Renderable graph model and source run metadata for one side of the split pane. */
export declare interface ExecutionPaneModel {
  side: ExecutionSide;
  paneIndex: number;
  run?: ExecutionRunRecord;
  original?: ExecutionGraphSpec;
  graph: ExecutionGraphSpec;
}

/** Dual-pane execution graph collection, lookup tables, and initial visualizer state. */
export declare interface BuiltExecutionGraphs {
  panes: ExecutionPaneModel[];
  lookup: Map<string, ExecutionHit>;
  recordNodes: Map<string, ExecutionHit>;
  graphCollections: Array<{label: string; graphs: ExecutionGraphSpec[]}>;
  topologyKey: string;
  initialUiState: {
    paneStates: Array<{
      deepestExpandedGroupNodeIds: string[];
      selectedNodeId: string;
      selectedGraphId: string;
      selectedCollectionLabel: string;
      widthFraction: number;
      selected: boolean;
      flattenLayers: boolean;
    }>;
  };
}

/** Camera center coordinates and zoom scale for an individual execution graph pane. */
export declare interface ExecutionPaneViewport {
  graphId: string;
  centerX: number;
  centerY: number;
  pixelsPerUnit: number;
}

/** Viewport alignment and animation parameters when focusing a node in a pane. */
export declare interface FocusNodeViewportOptions {
  labelSize?: number;
  align?: 'top' | 'center';
  paddingTop?: number;
  duration?: number;
}

/** Custom DOM events dispatched by the `<model-explorer-visualizer>` custom element. */
export declare interface ExecutionVisualizerEventMap
  extends HTMLElementEventMap {
  modelGraphProcessed: CustomEvent<{paneIndex: number}>;
  selectedNodeChanged: CustomEvent<{nodeId?: string}>;
  uiStateChanged: CustomEvent<{paneStates?: Array<{widthFraction?: number}>}>;
  hoveredNodeChanged: CustomEvent<{nodeId?: string}>;
}

/** Display and chrome configuration flags passed to `<model-explorer-visualizer>`. */
export declare interface ExecutionVisualizerConfig {
  hideTitleBar?: boolean;
  hideToolBar?: boolean;
  hideInfoPanel?: boolean;
  hideLegends?: boolean;
  hideEmptyNodeDataEntries?: boolean;
  edgeColor?: string;
}

/**
 * Verified 7-method Execution Viewport Bridge contract implemented by
 * `<model-explorer-visualizer>` inside the execution iframe.
 */
export declare interface ExecutionViewportBridge extends HTMLElement {
  workerScriptPath?: string;
  graphCollections: BuiltExecutionGraphs['graphCollections'];
  initialUiState: BuiltExecutionGraphs['initialUiState'];
  config: ExecutionVisualizerConfig;
  readPaneViewport(paneIndex: number): ExecutionPaneViewport | null;
  setPaneViewport(
    viewport: ExecutionPaneViewport,
    paneIndex: number,
    duration?: number,
  ): boolean;
  fitPaneGraph(paneIndex: number): boolean;
  focusNodeViewport(
    nodeId: string,
    paneIndex: number,
    options?: FocusNodeViewportOptions,
  ): boolean;
  setPaneSelection(nodeId: string, paneIndex: number): boolean;
  setPaneBackground?(background: string, paneIndex: number): boolean;
  addNodeDataProviderDataWithGraphIndex(
    name: string,
    data: Record<string, unknown>,
    paneIndex: number,
    clearExisting: boolean,
  ): void;
  addEventListener<K extends keyof ExecutionVisualizerEventMap>(
    type: K,
    listener: (
      this: ExecutionViewportBridge,
      ev: ExecutionVisualizerEventMap[K],
    ) => unknown,
    options?: boolean | AddEventListenerOptions,
  ): void;
}

/** Active run and tensor selection state resolved for one execution comparison side. */
export declare interface ExecutionSideSelection {
  side: ExecutionSide;
  run: ExecutionRunRecord | undefined;
  selected: ExecutionTensorRecord | undefined;
}

/** Adapter functions that transform raw capture payloads into visualizer graph models. */
export declare interface ExecutionDataAdapter {
  buildExecutionGraphs: (
    payload: ExecutionPayload,
    choices?: Partial<Record<ExecutionSide, string>>,
  ) => BuiltExecutionGraphs;
  buildNodeData: (
    model: BuiltExecutionGraphs,
    payload: ExecutionPayload,
  ) => Record<string, unknown>;
  executionSides: (payload: ExecutionPayload) => ExecutionSideSelection[];
}

declare global {
  interface Window {
    ExecutionGraphData: ExecutionDataAdapter;
    modelExplorer: {assetFilesBaseUrl: string; workerScriptPath: string};
  }
  interface HTMLElementTagNameMap {
    'model-explorer-visualizer': ExecutionViewportBridge;
  }
}

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

function attributes(
  values: ExecutionAttribute[] | undefined,
): Record<string, string> {
  return Object.fromEntries(
    (values || []).map((item) => [item.key, item.value]),
  );
}

function attr(key: string, value: unknown): ExecutionAttribute {
  return {key, value: String(value)};
}

function compositeKey(...parts: string[]): string {
  return JSON.stringify(parts);
}

function encodeId(...parts: Array<string | number>): string {
  return parts.map((value) => encodeURIComponent(String(value))).join(':');
}

function nodeStyle(kind: 'operation' | 'tensor') {
  return {
    backgroundColor:
      kind === 'operation' ? LIGHT_COLORS.operation : LIGHT_COLORS.tensor,
    borderColor:
      kind === 'operation'
        ? LIGHT_COLORS.operationBorder
        : LIGHT_COLORS.tensorBorder,
    hoveredBorderColor: LIGHT_COLORS.hover,
  };
}

function severity(value: number, metric: string | undefined): number {
  const name = String(metric ?? '').toLowerCase();
  return name === 'cossim' || name === 'cosine_similarity'
    ? Math.max(0, 1 - value)
    : Math.abs(value);
}

/**
 * Resolves reference and target run records from the active selection payload.
 */
export function executionSides(
  payload: ExecutionPayload,
): ExecutionSideSelection[] {
  const records = payload.details?.tensors || [];
  const runs = payload.details?.executions || [];
  const refRecord = records.find((record) => record.id === payload.reference);
  const targetRecord = records.find((record) => record.id === payload.target);
  const selectedRef = runs.find((run) => run.id === refRecord?.run);
  const selectedTarget = runs.find((run) => run.id === targetRecord?.run);
  const ref = selectedRef || runs.find((run) => run.id !== selectedTarget?.id);
  const target = selectedTarget || runs.find((run) => run.id !== ref?.id);
  return [
    {side: 'ref', run: ref, selected: refRecord},
    {side: 'target', run: target, selected: targetRecord},
  ];
}

/**
 * Builds dual-pane execution graphs and hit-testing indices from capture records.
 */
export function buildExecutionGraphs(
  payload: ExecutionPayload,
  choices: Partial<Record<ExecutionSide, string>> = {},
): BuiltExecutionGraphs {
  const records = payload.details?.tensors || [];
  const lookup = new Map<string, ExecutionHit>();
  const recordNodes = new Map<string, ExecutionHit>();
  const panes = executionSides(payload).map(
    ({side, run, selected}, paneIndex) => {
      const original =
        run?.graphs.find((graph) => graph.id === choices[side]) ||
        run?.graphs.find((graph) => graph.id === selected?.graph) ||
        run?.graphs[0];
      const originalGraphId = original?.id || 'empty';
      const graphId = encodeId(side, run?.id || '', originalGraphId);
      const nodes: ExecutionGraphNode[] = [];
      if (!original) {
        return {side, paneIndex, run, original, graph: {id: graphId, nodes}};
      }
      const operationIds = new Set(original.nodes.map((node) => node.id));
      const portIds = new Map<string, string>();
      const recordsByPort = new Map<string, ExecutionTensorRecord[]>();
      const portsByNode = new Map<string, Map<string, ExecutionPortMetadata>>();

      for (const record of records) {
        if (
          record.run !== run?.id ||
          record.graph !== original.id ||
          !operationIds.has(record.node)
        ) {
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
        const ports = new Map<string, ExecutionPortMetadata>(
          (node.outputsMetadata || []).map((output) => [
            String(output.id),
            output,
          ]),
        );
        // A real captured record is also evidence for an output omitted from graph metadata.
        for (const record of records) {
          if (
            record.run === run?.id &&
            record.graph === original.id &&
            record.node === node.id &&
            !ports.has(String(record.output))
          ) {
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
            encodeId(side, original.id, 'tensor', node.id, outputId),
          );
        }
      }

      for (const node of original.nodes) {
        const operationId = encodeId(side, original.id, 'op', node.id);
        const hit: ExecutionHit = {
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
          incomingEdges: (node.incomingEdges || [])
            .filter((edge) => operationIds.has(edge.sourceNodeId))
            .map((edge) => ({
              ...edge,
              sourceNodeId:
                portIds.get(
                  compositeKey(
                    edge.sourceNodeId,
                    String(edge.sourceNodeOutputId),
                  ),
                ) || encodeId(side, original.id, 'op', edge.sourceNodeId),
              sourceNodeOutputId: String(edge.sourceNodeOutputId),
              targetNodeInputId: String(edge.targetNodeInputId),
            })),
        });

        for (const [outputId, output] of portsByNode.get(node.id) || []) {
          const tensorId =
            portIds.get(compositeKey(node.id, outputId)) ||
            encodeId(side, original.id, 'tensor', node.id, outputId);
          const captured =
            recordsByPort.get(compositeKey(node.id, outputId)) || [];
          const record =
            captured.find((item) => item.id === selected?.id) ||
            (captured.length === 1 ? captured[0] : null);
          const metadata = attributes(output.attrs);
          const label =
            metadata['tensor_name'] ||
            metadata['name'] ||
            (captured.length === 1
              ? captured[0].id
              : `${node.label} · ${outputId}`);
          const tensorHit: ExecutionHit = {
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
                captured.length
                  ? `${captured.length} tensor record${
                      captured.length === 1 ? '' : 's'
                    }`
                  : 'Not captured',
              ),
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
    },
  );
  const label = 'Captured execution';
  const graphCollections = [{label, graphs: panes.map((pane) => pane.graph)}];
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
 * Builds node color and metric provider overlays for the active pair selection.
 */
export function buildNodeData(
  model: BuiltExecutionGraphs,
  payload: ExecutionPayload,
): Record<string, unknown> {
  const palette = payload.theme === 'dark' ? DARK_COLORS : LIGHT_COLORS;
  const metricName = payload.metric || '';
  const metricValue = payload.metrics?.[metricName];
  const valid =
    typeof metricValue === 'number' &&
    Number.isFinite(metricValue) &&
    Boolean(payload.reference) &&
    Boolean(payload.target);
  const amount = valid ? severity(metricValue, payload.metric) : 0;
  const alertBgColor = payload.theme === 'dark' ? '#4a2020' : '#f7cfcd';
  const metricBgColor = amount > 0 ? alertBgColor : palette.tensor;
  const selected = new Set(
    [payload.reference, payload.target].filter((id): id is string =>
      Boolean(id),
    ),
  );
  return Object.fromEntries(
    model.panes.map((pane) => [
      pane.graph.id,
      {
        name: payload.metric || 'Metric',
        hideInAggregatedStatsTable: true,
        hideInChildrenStatsTable: true,
        results: Object.fromEntries(
          pane.graph.nodes.map((node) => {
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
            const hasMetric =
              valid &&
              (hit?.recordIds || []).some((recordId) => selected.has(recordId));
            return [
              node.id,
              {
                value: hasMetric ? metricValue : '—',
                bgColor: hasMetric ? metricBgColor : palette.missing,
                textColor: hasMetric ? palette.text : palette.missingText,
              },
            ];
          }),
        ),
      },
    ]),
  );
}

/** Default singleton execution data adapter attached to `window.ExecutionGraphData`. */
export const executionGraphDataAdapter: ExecutionDataAdapter = {
  buildExecutionGraphs,
  buildNodeData,
  executionSides,
};

if (typeof window !== 'undefined') {
  window.ExecutionGraphData = executionGraphDataAdapter;
}
