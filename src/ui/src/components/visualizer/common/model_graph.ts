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

import {LayoutConfigs} from './input_graph';
import {
  GraphNodeConfig,
  GraphNodeStyle,
  GroupNodeAttributes,
  GroupNodeConfig,
  IncomingEdge,
  KeyValuePairs,
  NodeAttributePairs,
  OutgoingEdge,
  Point,
} from './types';

/**
 * A model graph to be visualized.
 *
 * This is the internal format used by the visualizer. It is processed from an
 * input `Graph` (see `input_graph.ts`).
 */
export declare interface ModelGraph {
  /**
   * The id of the graph.
   *
   * It is the same as the corresponding input `Graph`.
   */
  id: string;

  /** The label of the collection this graph belongs to. */
  collectionLabel: string;

  /** All nodes in the model graph. */
  nodes: ModelNode[];

  /** Attributes for group nodes. */
  groupNodeAttributes?: GroupNodeAttributes;

  /**
   * Custom configs for group nodes.
   *
   * A group node will be matched to the first config whose namespace regex
   * matches its namespace.
   */
  groupNodeConfigs?: GroupNodeConfig[];

  /** Ids of all group nodes that are artificially created. */
  artificialGroupNodeIds?: string[];

  /** All nodes in the model graph indexed by node id. */
  nodesById: Record<string, ModelNode>;

  /** The root nodes. */
  rootNodes: Array<GroupNode | OpNode>;

  /** From the ids of group nodes to the edges of their subgraphs. */
  edgesByGroupNodeIds: {[id: string]: ModelEdge[]};

  /** Max count of descendant op nodes for across group nodes. */
  maxDescendantOpNodeCount: number;

  /** Min count of descendant op nodes for across group nodes. */
  minDescendantOpNodeCount: number;

  /** A map from output tensor id to their owner node id. */
  outputTensorIdToNodeId?: {[id: string]: string};

  /** Number of edge curve segments, used for webgl rendering. */
  numEdgeSegments?: number;

  /**
   * Number of end points for all edge curve segments, used for webgl
   * rendering.
   */
  numEdgeSegmentEndPoints?: number;

  /**
   * A map from the id of a group to the edges of its
   * NS children nodes (fromNodeId -> targetNodeIds).
   */
  layoutGraphEdges: Record<string, Record<string, Record<string, boolean>>>;

  /** Layout-related configs. */
  layoutConfigs?: LayoutConfigs;
}

/** Node tyoes in a model graph. */
export enum NodeType {
  OP_NODE,
  GROUP_NODE,
}

/** The base interface of a node in model graph. */
export declare interface ModelNodeBase {
  /** The type of the node. */
  nodeType: NodeType;

  /** The unique if of the node. */
  id: string;

  /** The label of the node. */
  label: string;

  /**
   * The namespace/hierarchy data of the node. See input_graph.ts for more
   * details.
   */
  namespace: string;

  /**
   * The namespace field above will be cleared when the graph's layers are
   * flattened. In these cases, we store the namespace data here for display
   * purpose.
   */
  savedNamespace?: string;

  /**
   * Model explorer removes layers if their only child node is an op node.
   * In these cases, the op node's namespace and savedNamespace fields are
   * updated to reflect the optimized namespace. The savedNamespace field
   * stores the original (unoptimized) namespace of the node, which is still
   * useful to display in the UI.
   */
  fullNamespace?: string;

  /**
   * The level of the node in the hierarchy. It is the number of components in
   * its namespace.
   *
   * For example, the level of a node with namespace 'a/b/c' is 3.
   */
  level: number;

  /**
   * The id of its parent node in namespace.
   *
   * For example, a node with namespace "a/b" is the `nsParent` of a node with
   * namespace "a/b/c".
   */
  nsParentId?: string;

  // Layout data.

  /** The width of the node. */
  width?: number;

  /** The height of the node. */
  height?: number;

  /**
   * The local position (x) of the node. This is relative to its namespace
   * parent.
   */
  x?: number;

  /**
   * The local position (y) of the node. This is relative to its namespace
   * parent.
   */
  y?: number;

  /**
   * Local offset (x), in order to accommodate the situations where:
   * - Subgraphs that are smaller than its parent.
   * - Edges going out of the bonding box of all the nodes.
   */
  localOffsetX?: number;

  /**
   * Local offset (y), in order to accommodate the situations where:
   * - The ns parent node has attrs table shown.
   */
  localOffsetY?: number;

  /**
   * The global position (x) of the node, relative to the svg element.
   */
  globalX?: number;

  /** The global position (y) of the node, relative to the svg element. */
  globalY?: number;
}

/** An operation node in a model graph.  */
export declare interface OpNode extends ModelNodeBase {
  nodeType: NodeType.OP_NODE;

  /** Incoming edges. */
  incomingEdges?: IncomingEdge[];

  /**
   * Outgoing edges.
   *
   * We populate edges for both direction for convenience.
   */
  outgoingEdges?: OutgoingEdge[];

  /** The attributes of the node. */
  attrs?: NodeAttributePairs;

  /**
   * Metadata for inputs, indexed by input ids. Each input can have multiple
   * key-value pairs as its metadata.
   */
  inputsMetadata?: Record<string, KeyValuePairs>;

  /**
   * Metadata for outputs, indexed by output ids. Each output can have multiple
   * key-value paris as its metadata.
   */
  outputsMetadata?: Record<string, KeyValuePairs>;

  /** Whether this node should be hidden in layout. */
  hideInLayout?: boolean;

  /** Ids for subgraphs. */
  subgraphIds?: string[];

  /** The style of the node. */
  style?: GraphNodeStyle;

  /** Custom configs for the node. */
  config?: GraphNodeConfig;
}

/**
 * A group node that groups op nodes and other group nodes.
 *
 * Grouping happens on namespace level. A group node will be created for each
 * unique namespace.
 */
export declare interface GroupNode extends ModelNodeBase {
  nodeType: NodeType.GROUP_NODE;

  /** Its children nodes under its namespace. */
  nsChildrenIds?: string[];

  /** All descendant nodes under this group's namespace. */
  descendantsNodeIds?: string[];

  /** All descendant op nodes under this group's namespace. */
  descendantsOpNodeIds?: string[];

  /** Whether this node is expanded (true) or collapsed (false). */
  expanded: boolean;

  /** Index of identical group. */
  identicalGroupIndex?: number;

  /**
   * Whether this group node serves as a section container to reduce number of
   * nodes to layout.
   */
  sectionContainer?: boolean;

  /** The op node that should be pinned to the top of the group. */
  pinToTopOpNode?: OpNode;
}

/** A node in a model graph. */
export type ModelNode = OpNode | GroupNode;

/** An edge in a model graph, */
export declare interface ModelEdge {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  points: Point[];

  // The following are for webgl rendering.
  curvePoints?: Point[];

  // The label of the edge.
  //
  // If set, it will be rendered on edge instead of tensor shape.
  label?: string;
}
