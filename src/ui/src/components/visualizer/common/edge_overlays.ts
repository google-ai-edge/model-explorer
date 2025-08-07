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

import {TaskData, TaskType} from './task';

/** The data for edge overlays. */
export declare interface EdgeOverlaysData extends TaskData {
  type: TaskType.EDGE_OVERLAYS;

  /** The name of this set of overlays, for UI display purposes. */
  name: string;

  /**
   * The name of the graph that this set of overlays is associated with.
   *
   * The name should be the label shown in the graph selector dropdown.
   *
   * If not set, this set of overlays will be shown in all graphs.
   */
  graphName?: string;

  /** A list of edge overlays. */
  overlays: EdgeOverlay[];
}

/** An edge overlay. */
export declare interface EdgeOverlay {
  /** The name displayed in the UI to identify this overlay. */
  name: string;

  /** The edges that define the overlay. */
  edges: Edge[];

  /**
   * The color of the overlay edges.
   *
   * They are rendered in this color when any of the nodes in this overlay is
   * selected.
   */
  edgeColor: string;

  /** The width of the overlay edges. Default to 2. */
  edgeWidth?: number;

  /** The font size of the edge labels. Default to 7.5. */
  edgeLabelFontSize?: number;

  /**
   * If set to `true`, only edges that are directly connected to the currently
   * selected node (i.e., edges that either start from or end at the selected
   * node) will be displayed for this overlay. All other edges within this
   * overlay will be hidden.
   */
  showEdgesConnectedToSelectedNodeOnly?: boolean;

  /**
   * The "range" of edges to show when showEdgesConnectedToSelectedNodeOnly is
   * set to `true. Default to 1.
   *
   * This value determines how many "layers" of connections will be displayed
   * around the selected node.
   *
   * For example:
   *
   * - A range of 1 shows only the edges directly connected to the selected
   *   node.
   * - A range of 2 shows the edges connected to the selected node, as well as
   *   the edges connected to all of its immediate neighbors.
   *
   * The higher the number, the more of the network will be visible around the
   * selected node.
   */
  visibleEdgeHops?: number;
}

/** An edge in the overlay. */
export declare interface Edge {
  /** The id of the source node. Op node only. */
  sourceNodeId: string;

  /** The id of the target node. Op node only. */
  targetNodeId: string;

  /** Label shown on the edge. */
  label?: string;
}

/** The processed edge overlays data. */
export declare interface ProcessedEdgeOverlaysData extends EdgeOverlaysData {
  /** A random id. */
  id: string;

  processedOverlays: ProcessedEdgeOverlay[];
}

/** The processed edge overlay. */
export declare interface ProcessedEdgeOverlay extends EdgeOverlay {
  /** A random id. */
  id: string;

  /** The set of node ids that are in this overlay. */
  nodeIds: Set<string>;

  /** A map from node id to the edges that start from or end at the node. */
  adjacencyMap: Map<string, Edge[]>;
}
