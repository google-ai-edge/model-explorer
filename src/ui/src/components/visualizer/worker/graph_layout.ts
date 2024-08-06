/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import {
  LAYOUT_MARGIN_X,
  MAX_IO_ROWS_IN_ATTRS_TABLE,
  NODE_ATTRS_TABLE_FONT_SIZE,
  NODE_ATTRS_TABLE_LABEL_VALUE_PADDING,
  NODE_ATTRS_TABLE_LEFT_RIGHT_PADDING,
  NODE_ATTRS_TABLE_MARGIN_TOP,
  NODE_ATTRS_TABLE_ROW_HEIGHT,
  NODE_ATTRS_TABLE_VALUE_MAX_WIDTH,
  NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
} from '../common/consts';
import {
  GroupNode,
  ModelEdge,
  ModelGraph,
  ModelNode,
  OpNode,
} from '../common/model_graph';
import {
  GraphNodeConfig,
  KeyValueList,
  NodeDataProviderRunData,
  Point,
  Rect,
  ShowOnNodeItemData,
  ShowOnNodeItemType,
} from '../common/types';
import {
  generateCurvePoints,
  getGroupNodeFieldLabelsFromShowOnNodeItemTypes,
  getLabelWidth,
  getNodeInfoFieldValue,
  getOpNodeAttrsKeyValuePairsForAttrsTable,
  getOpNodeDataProviderKeyValuePairsForAttrsTable,
  getOpNodeFieldLabelsFromShowOnNodeItemTypes,
  getOpNodeInputsKeyValuePairsForAttrsTable,
  getOpNodeOutputsKeyValuePairsForAttrsTable,
  isGroupNode,
  isOpNode,
} from '../common/utils';

import {Dagre, DagreGraphInstance} from './dagre_types';

/** The margin for the top and bottom side of the layout. */
export const LAYOUT_MARGIN_TOP = 36;

/** The margin for the bottom side of the layout */
export const LAYOUT_MARGIN_BOTTOM = 16;

/** The default height of a node. */
export const DEFAULT_NODE_HEIGHT = 26;

/** Node width for test cases. */
export const NODE_WIDTH_FOR_TEST = 50;

const LABEL_PADDING = 24;

const MIN_NODE_WIDTH = 80;

const ATTRS_TABLE_MARGIN_X = 8;

/** A node in dagre. */
export declare interface DagreNode {
  id: string;
  width: number;
  height: number;
  x?: number;
  y?: number;
  config?: GraphNodeConfig;
}

interface LayoutGraph {
  nodes: {[id: string]: DagreNode};
  incomingEdges: {[fromId: string]: string[]};
  outgoingEdges: {[fromId: string]: string[]};
}

/**
 * To manage graph layout related tasks.
 *
 * TODO: distribute this task to multiple workers to improvement performance.
 */
export class GraphLayout {
  dagreGraph!: DagreGraphInstance;

  constructor(
    private readonly modelGraph: ModelGraph,
    private readonly dagre: Dagre,
    private readonly showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
    private readonly nodeDataProviderRuns: Record<
      string,
      NodeDataProviderRunData
    >,
    private readonly testMode = false,
  ) {
    this.dagreGraph = new this.dagre.graphlib.Graph();
  }

  /** Lays out the model graph rooted from the given root node.  */
  layout(rootNodeId?: string): Rect {
    // Get the children nodes of the given root node.
    let rootNode: GroupNode | undefined = undefined;
    let nodes: ModelNode[] = [];
    if (rootNodeId == null) {
      nodes = this.modelGraph.rootNodes;
    } else {
      rootNode = this.modelGraph.nodesById[rootNodeId] as GroupNode;
      nodes = (rootNode.nsChildrenIds || []).map(
        (nodeId) => this.modelGraph.nodesById[nodeId],
      );
    }

    // Init.
    this.configLayout(this.dagreGraph);

    // Get layout graph.
    const layoutGraph = getLayoutGraph(
      rootNode?.id || '',
      nodes,
      this.modelGraph,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.testMode,
    );

    // Set nodes/edges to dagre.
    for (const id of Object.keys(layoutGraph.nodes)) {
      const dagreNode = layoutGraph.nodes[id];
      if (dagreNode.config?.pinToGroupTop) {
        continue;
      }
      this.dagreGraph.setNode(id, dagreNode);
    }
    for (const fromNodeId of Object.keys(layoutGraph.outgoingEdges)) {
      for (const toNodeId of layoutGraph.outgoingEdges[fromNodeId]) {
        this.dagreGraph.setEdge(fromNodeId, toNodeId);
      }
    }

    // Run the layout algorithm.
    this.dagre.layout(this.dagreGraph);

    // Set the results back to the original model nodes and calculate the bound
    // that contains all the nodes.
    let minX = Number.MAX_VALUE;
    let minY = Number.MAX_VALUE;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const node of nodes) {
      const dagreNode = layoutGraph.nodes[node.id];
      if (!dagreNode) {
        console.warn(`Node "${node.id}" is not in the dagre layout result`);
        continue;
      }
      node.x = (dagreNode.x || 0) - dagreNode.width / 2;
      node.y = (dagreNode.y || 0) - dagreNode.height / 2;
      node.width = dagreNode.width;
      node.height = dagreNode.height;
      node.localOffsetX = 0;
      node.localOffsetY = 0;

      // Don't consider the bound of the node if it's pinned to the top of the
      // group.
      if (!dagreNode.config?.pinToGroupTop) {
        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x + node.width);
        maxY = Math.max(maxY, node.y + node.height);
      }
    }

    // Expand the bound to include all the edges.
    let minEdgeX = Number.MAX_VALUE;
    let minEdgeY = Number.MAX_VALUE;
    let maxEdgeX = Number.NEGATIVE_INFINITY;
    let maxEdgeY = Number.NEGATIVE_INFINITY;
    const dagreEdgeRefs = this.dagreGraph.edges();
    const edges: ModelEdge[] = [];
    for (const dagreEdge of dagreEdgeRefs) {
      const points = this.dagreGraph.edge(dagreEdge).points as Point[];
      // tslint:disable-next-line:no-any Allow arbitrary types.
      const d3 = (globalThis as any)['d3'];
      // tslint:disable-next-line:no-any Allow arbitrary types.
      const three = (globalThis as any)['THREE'];
      const curvePoints =
        typeof three === 'undefined'
          ? []
          : generateCurvePoints(
              points,
              d3['line'],
              d3['curveMonotoneY'],
              three,
            );
      const fromNode = this.modelGraph.nodesById[dagreEdge.v];
      const toNode = this.modelGraph.nodesById[dagreEdge.w];
      if (fromNode == null) {
        console.warn(`Edge from node not found: "${dagreEdge.v}"`);
        continue;
      }
      if (toNode == null) {
        console.warn(`Edge to node not found: "${dagreEdge.w}"`);
        continue;
      }
      const edgeId = `${fromNode.id}|${toNode.id}`;
      edges.push({
        id: edgeId,
        fromNodeId: fromNode.id,
        toNodeId: toNode.id,
        points,
        curvePoints,
      });
      for (const point of points) {
        minEdgeX = Math.min(minEdgeX, point.x);
        minEdgeY = Math.min(minEdgeY, point.y);
        maxEdgeX = Math.max(maxEdgeX, point.x);
        maxEdgeY = Math.max(maxEdgeY, point.y);
      }
    }
    this.modelGraph.edgesByGroupNodeIds[rootNodeId || ''] = edges;

    // Offset nodes to take into account of edges going out of the bound of all
    // the nodes.
    if (minEdgeX < minX) {
      for (const node of nodes) {
        node.localOffsetX = Math.max(0, minX - minEdgeX);
      }
    }

    minX = Math.min(minEdgeX, minX);
    maxX = Math.max(maxEdgeX, maxX);

    // Make sure the subgraph width is at least the width of the root node.
    let subgraphFullWidth = maxX - minX + LAYOUT_MARGIN_X * 2;
    if (rootNode) {
      const parentNodeWidth = getNodeWidth(
        rootNode,
        this.modelGraph,
        this.showOnNodeItemTypes,
        this.nodeDataProviderRuns,
      );
      if (subgraphFullWidth < parentNodeWidth) {
        const extraOffsetX = (parentNodeWidth - subgraphFullWidth) / 2;
        for (const node of nodes) {
          if (!node.localOffsetX) {
            node.localOffsetX = 0;
          }
          node.localOffsetX += extraOffsetX;
        }
        subgraphFullWidth = parentNodeWidth;
      }
    }

    // Offset downwards if the root node has attrs table shown.
    if (rootNode && isGroupNode(rootNode)) {
      const attrsRowCount = getGroupNodeAttrsTableRowCount(
        this.showOnNodeItemTypes,
      );
      if (attrsRowCount > 0) {
        const localOffsetY = attrsRowCount * NODE_ATTRS_TABLE_ROW_HEIGHT + 16;
        for (const node of nodes) {
          node.localOffsetY = localOffsetY;
        }
        maxY += localOffsetY;
      }
    }

    return {
      x: minX,
      y: minY,
      width: subgraphFullWidth - LAYOUT_MARGIN_X * 2,
      height: maxY - minY,
    };
  }

  private configLayout(dagreGraph: DagreGraphInstance) {
    // See available configs here:
    // https://github.com/dagrejs/dagre/wiki#configuring-the-layout.
    dagreGraph.setGraph({
      nodesep: 20,
      ranksep: 50,
      edgesep: 20,
      marginx: LAYOUT_MARGIN_X,
      marginy: LAYOUT_MARGIN_TOP,
    });
    // No edge labels.
    dagreGraph.setDefaultEdgeLabel(() => ({}));
  }
}

/** An utility function to get the node width using an offscreen canvas. */
export function getNodeWidth(
  node: ModelNode,
  modelGraph: ModelGraph,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  testMode = false,
) {
  // Always return 32 in test mode.
  if (testMode) {
    return NODE_WIDTH_FOR_TEST;
  }

  const label = node.label;
  let labelWidth = getLabelWidth(label, 11, isGroupNode(node)) + LABEL_PADDING;
  // Add space to label width for the "expand/collapse icon" at the left and the
  // "more" icon at the right.
  if (isGroupNode(node)) {
    labelWidth += 28;
  }

  // Calculate the width of attrs table.
  //
  // Figure out the max width of all the labels and values respectively.
  let maxAttrLabelWidth = 0;
  let maxAttrValueWidth = 0;
  if (isOpNode(node)) {
    // Basic info.
    //
    // Gather field ids for the selected show-on-node items.
    const fieldIds: string[] =
      getOpNodeFieldLabelsFromShowOnNodeItemTypes(showOnNodeItemTypes);
    // Calculate width.
    for (const fieldId of fieldIds) {
      const attrLabelWidth = getLabelWidth(
        `${fieldId}:`,
        NODE_ATTRS_TABLE_FONT_SIZE,
        true,
      );
      const value = getNodeInfoFieldValue(node, fieldId);
      const attrValueWidth = getLabelWidth(
        value,
        NODE_ATTRS_TABLE_FONT_SIZE,
        false,
      );
      maxAttrLabelWidth = Math.max(maxAttrLabelWidth, attrLabelWidth);
      maxAttrValueWidth = Math.max(maxAttrValueWidth, attrValueWidth);
    }

    // Attrs.
    if (showOnNodeItemTypes[ShowOnNodeItemType.OP_ATTRS]?.selected) {
      const keyValuePairs = getOpNodeAttrsKeyValuePairsForAttrsTable(
        node,
        showOnNodeItemTypes[ShowOnNodeItemType.OP_ATTRS]?.filterRegex || '',
      );
      const widths = getMaxAttrLabelAndValueWidth(keyValuePairs);
      maxAttrLabelWidth = Math.max(maxAttrLabelWidth, widths.maxAttrLabelWidth);
      maxAttrValueWidth = Math.max(maxAttrValueWidth, widths.maxAttrValueWidth);
    }

    // Inputs.
    if (showOnNodeItemTypes[ShowOnNodeItemType.OP_INPUTS]?.selected) {
      const keyValuePairs = getOpNodeInputsKeyValuePairsForAttrsTable(
        node,
        modelGraph,
      );
      const widths = getMaxAttrLabelAndValueWidth(keyValuePairs);
      maxAttrLabelWidth = Math.max(maxAttrLabelWidth, widths.maxAttrLabelWidth);
      maxAttrValueWidth = Math.max(maxAttrValueWidth, widths.maxAttrValueWidth);
    }

    // Outputs
    if (showOnNodeItemTypes[ShowOnNodeItemType.OP_OUTPUTS]?.selected) {
      const keyValuePairs = getOpNodeOutputsKeyValuePairsForAttrsTable(node);
      const widths = getMaxAttrLabelAndValueWidth(keyValuePairs);
      maxAttrLabelWidth = Math.max(maxAttrLabelWidth, widths.maxAttrLabelWidth);
      maxAttrValueWidth = Math.max(maxAttrValueWidth, widths.maxAttrValueWidth);
    }

    // Node data providers.
    const nodeDataProviderKeyValuePairs =
      getOpNodeDataProviderKeyValuePairsForAttrsTable(
        node,
        modelGraph.id,
        showOnNodeItemTypes,
        nodeDataProviderRuns,
      );
    const nodeDataProviderWidths = getMaxAttrLabelAndValueWidth(
      nodeDataProviderKeyValuePairs,
    );
    maxAttrLabelWidth = Math.max(
      maxAttrLabelWidth,
      nodeDataProviderWidths.maxAttrLabelWidth,
    );
    maxAttrValueWidth = Math.max(
      maxAttrValueWidth,
      nodeDataProviderWidths.maxAttrValueWidth,
    );
  } else if (isGroupNode(node)) {
    // Basic info
    //
    // Gather basic info field ids for the selected show-on-node items.
    const basicInfoFieldIds: string[] =
      getGroupNodeFieldLabelsFromShowOnNodeItemTypes(showOnNodeItemTypes);
    // Calculate width.
    for (const fieldId of basicInfoFieldIds) {
      const attrLabelWidth = getLabelWidth(
        `${fieldId}:`,
        NODE_ATTRS_TABLE_FONT_SIZE,
        true,
      );
      const value = getNodeInfoFieldValue(node, fieldId);
      const attrValueWidth = getLabelWidth(
        value,
        NODE_ATTRS_TABLE_FONT_SIZE,
        false,
      );
      maxAttrLabelWidth = Math.max(maxAttrLabelWidth, attrLabelWidth);
      maxAttrValueWidth = Math.max(maxAttrValueWidth, attrValueWidth);
    }
  }
  maxAttrValueWidth = Math.min(
    maxAttrValueWidth,
    NODE_ATTRS_TABLE_VALUE_MAX_WIDTH,
  );
  let attrsTableWidth =
    maxAttrLabelWidth +
    maxAttrValueWidth +
    NODE_ATTRS_TABLE_LEFT_RIGHT_PADDING * 2 +
    NODE_ATTRS_TABLE_LABEL_VALUE_PADDING;
  if (attrsTableWidth !== NODE_ATTRS_TABLE_LABEL_VALUE_PADDING) {
    attrsTableWidth += ATTRS_TABLE_MARGIN_X * 2;
  }
  return Math.max(MIN_NODE_WIDTH, Math.max(labelWidth, attrsTableWidth));
}

/** An utility function to get the node height. */
export function getNodeHeight(
  node: ModelNode,
  modelGraph: ModelGraph,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  testMode = false,
  forceRecalculate = false,
) {
  if (testMode) {
    return DEFAULT_NODE_HEIGHT;
  }

  if (node.height != null && !forceRecalculate) {
    return node.height;
  }

  // Count how many rows will be in the attrs table.
  let attrsTableRowCount = 0;
  if (isOpNode(node)) {
    attrsTableRowCount = getOpNodeAttrsTableRowCount(
      showOnNodeItemTypes,
      node,
      nodeDataProviderRuns,
    );
  } else if (isGroupNode(node)) {
    attrsTableRowCount = getGroupNodeAttrsTableRowCount(showOnNodeItemTypes);
  }

  return (
    DEFAULT_NODE_HEIGHT +
    attrsTableRowCount * NODE_ATTRS_TABLE_ROW_HEIGHT +
    (attrsTableRowCount > 0 ? NODE_ATTRS_TABLE_MARGIN_TOP - 4 : 0)
  );
}

/** Gets a layout graph for the given nodes. */
export function getLayoutGraph(
  rootGroupNodeId: string,
  nodes: ModelNode[],
  modelGraph: ModelGraph,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  testMode = false,
  useFakeNodeSize = false,
): LayoutGraph {
  const layoutGraph: LayoutGraph = {
    nodes: {},
    incomingEdges: {},
    outgoingEdges: {},
  };

  // Create layout graph nodes.
  for (const node of nodes) {
    if (isOpNode(node) && node.hideInLayout) {
      continue;
    }
    const dagreNode: DagreNode = {
      id: node.id,
      width:
        node.width ||
        (useFakeNodeSize
          ? 10
          : getNodeWidth(
              node,
              modelGraph,
              showOnNodeItemTypes,
              nodeDataProviderRuns,
              testMode,
            )),
      height: useFakeNodeSize
        ? 10
        : getNodeHeight(
            node,
            modelGraph,
            showOnNodeItemTypes,
            nodeDataProviderRuns,
            testMode,
          ),
      config: isOpNode(node) ? node.config : undefined,
    };
    layoutGraph.nodes[node.id] = dagreNode;
  }

  // Set layout graph edges.
  const curLayoutGraphEdges =
    modelGraph.layoutGraphEdges[rootGroupNodeId] || {};
  for (const [fromNodeId, toNodeIds] of Object.entries(curLayoutGraphEdges)) {
    for (const toNodeId of Object.keys(toNodeIds)) {
      // Ignore edges from/to nodes pinned to group top.
      const fromNode = modelGraph.nodesById[fromNodeId];
      const toNode = modelGraph.nodesById[toNodeId];
      if (fromNode && isOpNode(fromNode) && fromNode.config?.pinToGroupTop) {
        continue;
      }
      if (toNode && isOpNode(toNode) && toNode.config?.pinToGroupTop) {
        continue;
      }
      addLayoutGraphEdge(layoutGraph, fromNodeId, toNodeId);
    }
  }

  return layoutGraph;
}

function getOpNodeAttrsTableRowCount(
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  node: OpNode,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
): number {
  // Basic info fields.
  const baiscFieldIds =
    getOpNodeFieldLabelsFromShowOnNodeItemTypes(showOnNodeItemTypes);

  // Node attributes.
  const attrsCount = showOnNodeItemTypes[ShowOnNodeItemType.OP_ATTRS]?.selected
    ? getOpNodeAttrsKeyValuePairsForAttrsTable(
        node,
        showOnNodeItemTypes[ShowOnNodeItemType.OP_ATTRS]?.filterRegex || '',
      ).length
    : 0;

  // Inputs.
  let inputsCount = showOnNodeItemTypes[ShowOnNodeItemType.OP_INPUTS]?.selected
    ? Object.keys(node.incomingEdges || []).length
    : 0;
  if (inputsCount > MAX_IO_ROWS_IN_ATTRS_TABLE) {
    inputsCount = MAX_IO_ROWS_IN_ATTRS_TABLE + 1;
  }

  // Outputs.
  let outputsCount = showOnNodeItemTypes[ShowOnNodeItemType.OP_OUTPUTS]
    ?.selected
    ? Object.keys(node.outputsMetadata || {}).length
    : 0;
  if (outputsCount > MAX_IO_ROWS_IN_ATTRS_TABLE) {
    outputsCount = MAX_IO_ROWS_IN_ATTRS_TABLE + 1;
  }

  // Node data providers.
  const nodeDataProviderCount = Object.keys(showOnNodeItemTypes)
    .filter((type) => showOnNodeItemTypes[type].selected)
    .filter(
      (showOnNodeItemType: string) =>
        showOnNodeItemType.startsWith(
          NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
        ) &&
        Object.values(nodeDataProviderRuns).some(
          (run) =>
            run.runName ===
            showOnNodeItemType.replace(
              NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
              '',
            ),
        ),
    ).length;

  return (
    baiscFieldIds.length +
    attrsCount +
    inputsCount +
    outputsCount +
    nodeDataProviderCount
  );
}

function getGroupNodeAttrsTableRowCount(
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
): number {
  const baiscFieldIds =
    getGroupNodeFieldLabelsFromShowOnNodeItemTypes(showOnNodeItemTypes);
  return baiscFieldIds.length;
}

function addLayoutGraphEdge(
  layoutGraph: LayoutGraph,
  fromNodeId: string,
  toNodeId: string,
) {
  if (layoutGraph.outgoingEdges[fromNodeId] == null) {
    layoutGraph.outgoingEdges[fromNodeId] = [];
  }
  layoutGraph.outgoingEdges[fromNodeId].push(toNodeId);

  if (layoutGraph.incomingEdges[toNodeId] == null) {
    layoutGraph.incomingEdges[toNodeId] = [];
  }
  layoutGraph.incomingEdges[toNodeId].push(fromNodeId);
}

function getMaxAttrLabelAndValueWidth(keyValuePairs: KeyValueList): {
  maxAttrLabelWidth: number;
  maxAttrValueWidth: number;
} {
  let maxAttrLabelWidth = 0;
  let maxAttrValueWidth = 0;
  for (const {key, value} of keyValuePairs) {
    const attrLabelWidth = getLabelWidth(key, NODE_ATTRS_TABLE_FONT_SIZE, true);
    maxAttrLabelWidth = Math.max(maxAttrLabelWidth, attrLabelWidth);
    const attrValueWidth = getLabelWidth(
      value,
      NODE_ATTRS_TABLE_FONT_SIZE,
      false,
    );
    maxAttrValueWidth = Math.max(maxAttrValueWidth, attrValueWidth);
  }
  return {maxAttrLabelWidth, maxAttrValueWidth};
}
