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
  GraphNodeStyle,
  IncomingEdge,
  KeyValueList,
  MetadataItem,
} from './types';

/** A collection of graphs. This is the input to the visualizer. */
export declare interface GraphCollection {
  /** The label of the collection. */
  label: string;

  /** The graphs inside the collection. */
  graphs: Graph[];
}

/** The collection sent from the built-in adapters. */
export declare interface GraphCollectionFromBuiltinAdapters {
  /** The partial label of the collection. */
  label: string;

  /** The graphs inside the collection. */
  subgraphs: Graph[];
}

/**
 * A graph to be visualized. Clients need to convert their own graphs into
 * this format and pass it into the visualizer through `GraphCollection` above.
 * The visualizer will then process the graph and convert it into its internal
 * format (see model_graph.ts) before visualizing it.
 *
 * The visualizer accepts a list of graphs. The first graph in the list is the
 * default one to be visualized. Users can pick a different graph from a
 * drop-down list.
 *
 * Graphs can also be `linked` together through the `subgraphIds` field of a
 * node (see comments below for more details).
 */
export declare interface Graph {
  /** The id of the graph. */
  id: string;

  /**
   * The label of the collection this graph belongs to. This field will be set
   * internally (i.e. users don't need to set it explicitly).
   */
  collectionLabel?: string;

  /** A list of nodes in the graph. */
  nodes: GraphNode[];

  //////////////////////////////////////////////////////////////////////////////
  // The following fields are set by model explorer. Users don't need to set
  // them.

  // The ids of all its subgraphs.
  subGraphIds?: string[];

  // The id of its parent graph. We assume each subgraph only belongs to one
  // parent.
  parentGraphId?: string;

  // The level in the graph tree.
  level?: number;
}

/** A single node in the graph. */
export declare interface GraphNode {
  /** The unique id of the node.  */
  id: string;

  /** The label of the node, displayed on the node in the model graph. */
  label: string;

  /**
   * The namespace/hierarchy data of the node in the form of a "path" (e.g.
   * a/b/c). Don't include the node label as the last component of the
   * namespace. The visualizer will use this data to visualize nodes in a nested
   * way.
   *
   * For example, for three nodes with the following label and namespace data:
   * - N1: a/b
   * - N2: a/b
   * - N3: a
   *
   * The visualizer will first show a collapsed box labeled 'a'. After the box
   * is expanded (by user clicking on it), it will show node N3, and another
   * collapsed box labeled 'b'. After the box 'b' is expanded, it will show two
   * nodes N1 and N2 inside the box 'b'.
   */
  namespace: string;

  /**
   * Ids of subgraphs that this node goes into.
   *
   * The graphs referenced here should be the ones from the `graphs` field in
   * `GraphList` above. Once set, users will be able to click this node, pick a
   * subgraph from a drop-down list, and see the visualization for the selected
   * subgraph.
   */
  subgraphIds?: string[];

  /** The attributes of the node.  */
  attrs?: KeyValueList;

  /** A list of incoming edges. */
  incomingEdges?: IncomingEdge[];

  /**
   * Metadata for inputs.
   */
  inputsMetadata?: MetadataItem[];

  /**
   * Metadata for outputs.
   */
  outputsMetadata?: MetadataItem[];

  /** The default style of the node. */
  style?: GraphNodeStyle;
}
