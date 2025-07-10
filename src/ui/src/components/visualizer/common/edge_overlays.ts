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
}
