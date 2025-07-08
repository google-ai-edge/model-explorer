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

import {Graph} from './input_graph';
import {ModelGraph} from './model_graph';
import {
  NodeAttributePairs,
  NodeDataProviderRunData,
  NodeStylerRule,
  Point,
  Rect,
  ShowOnNodeItemData,
} from './types';
import {VisualizerConfig} from './visualizer_config';

/** Various worker event types. */
export enum WorkerEventType {
  PROCESS_GRAPH_REQ,
  PROCESS_GRAPH_RESP,
  EXPAND_OR_COLLAPSE_GROUP_NODE_REQ,
  EXPAND_OR_COLLAPSE_GROUP_NODE_RESP,
  RELAYOUT_GRAPH_REQ,
  RELAYOUT_GRAPH_RESP,
  LOCATE_NODE_REQ,
  LOCATE_NODE_RESP,
  UPDATE_PROCESSING_PROGRESS,
  PREPARE_POPUP_REQ,
  PREPARE_POPUP_RESP,
  CLEANUP,
  UPDATE_MODEL_GRAPH_CACHE_WITH_NODE_ATTRIBUTES,
}

/** The base of all worker events. */
export declare interface WorkerEventBase {
  eventType: WorkerEventType;
}

/** The request for processing an input graph. */
export declare interface ProcessGraphRequest extends WorkerEventBase {
  eventType: WorkerEventType.PROCESS_GRAPH_REQ;
  graph: Graph;
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>;
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>;
  config?: VisualizerConfig;
  paneId: string;
  groupNodeChildrenCountThreshold?: number;
  flattenLayers?: boolean;
  keepLayersWithASingleChild?: boolean;
  initialLayout?: boolean;
}

/** The response for processing an input graph. */
export declare interface ProcessGraphResponse extends WorkerEventBase {
  eventType: WorkerEventType.PROCESS_GRAPH_RESP;
  modelGraph: ModelGraph;
  paneId: string;
}

/** The request for expanding/collapsing a group node. */
export declare interface ExpandOrCollapseGroupNodeRequest
  extends WorkerEventBase {
  eventType: WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_REQ;
  modelGraphId: string;
  // undefined when expanding/collapsing from root.
  groupNodeId?: string;
  expand: boolean;
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>;
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>;
  selectedNodeDataProviderRunId?: string;
  rendererId: string;
  paneId: string;
  // Expand or collapse all groups under the selected group.
  all?: boolean;
  // Timestamp of when the request is sent.
  ts?: number;
  config?: VisualizerConfig;
}

/** The response for expanding/collapsing a group node. */
export declare interface ExpandOrCollapseGroupNodeResponse
  extends WorkerEventBase {
  eventType: WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_RESP;
  modelGraph: ModelGraph;
  expanded: boolean;
  // undefined when expanding/collapsing from root.
  groupNodeId?: string;
  rendererId: string;
  // These are the deepest group nodes (in terms of level) that none of its
  // child group nodes is expanded.
  deepestExpandedGroupNodeIds: string[];
}

/**
 * The request for re-laying out the whole graph, keeping the current
 * collapse/expand states for all group nodes.
 */
export declare interface RelayoutGraphRequest extends WorkerEventBase {
  eventType: WorkerEventType.RELAYOUT_GRAPH_REQ;
  modelGraphId: string;
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>;
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>;
  selectedNodeDataProviderRunId?: string;
  targetDeepestGroupNodeIdsToExpand?: string[];
  selectedNodeId: string;
  rendererId: string;
  forRestoringUiState?: boolean;
  rectToZoomFit?: Rect;
  clearAllExpandStates?: boolean;
  forRestoringSnapshotAfterTogglingFlattenLayers?: boolean;
  nodeStylerQueries?: NodeStylerRule[];
  triggerNavigationSync?: boolean;
  config?: VisualizerConfig;
}

/** The response for re-laying out the whole graph. */
export declare interface RelayoutGraphResponse extends WorkerEventBase {
  eventType: WorkerEventType.RELAYOUT_GRAPH_RESP;
  modelGraph: ModelGraph;
  selectedNodeId: string;
  rendererId: string;
  forRestoringUiState?: boolean;
  rectToZoomFit?: Rect;
  forRestoringSnapshotAfterTogglingFlattenLayers?: boolean;
  targetDeepestGroupNodeIdsToExpand?: string[];
  triggerNavigationSync?: boolean;
}

/** The request for locating a node. */
export declare interface LocateNodeRequest extends WorkerEventBase {
  eventType: WorkerEventType.LOCATE_NODE_REQ;
  modelGraphId: string;
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>;
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>;
  selectedNodeDataProviderRunId?: string;
  nodeId: string;
  rendererId: string;
  noNodeShake?: boolean;
  select?: boolean;
  config?: VisualizerConfig;
}

/** The response for locating a node. */
export declare interface LocateNodeResponse extends WorkerEventBase {
  eventType: WorkerEventType.LOCATE_NODE_RESP;
  modelGraph: ModelGraph;
  nodeId: string;
  rendererId: string;
  // See comments above.
  deepestExpandedGroupNodeIds: string[];
  noNodeShake?: boolean;
  select?: boolean;
}

/** Labels for processing progress. */
export enum ProcessingLabel {
  PROCESSING_NODES_AND_EDGES = 'Processing nodes and edges',
  PROCESSING_LAYER_NAMESPACES = 'Processing layer namespaces',
  PROCESSING_LAYOUT_DATA = 'Processing layout data',
  SPLITTING_LARGE_LAYERS = 'Splitting large layers (if any)',
  LAYING_OUT_ROOT_LAYER = 'Laying out root layer',
  FINDING_IDENTICAL_LAYERS = 'Finding identical layers',
}

/** All processing labels. */
export const ALL_PROCESSING_LABELS = [
  ProcessingLabel.PROCESSING_NODES_AND_EDGES,
  ProcessingLabel.PROCESSING_LAYER_NAMESPACES,
  ProcessingLabel.PROCESSING_LAYOUT_DATA,
  ProcessingLabel.SPLITTING_LARGE_LAYERS,
  ProcessingLabel.LAYING_OUT_ROOT_LAYER,
  ProcessingLabel.FINDING_IDENTICAL_LAYERS,
];

/** The request for updating the processing progress (sent from worker). */
export declare interface UpdateProcessingProgressRequest
  extends WorkerEventBase {
  eventType: WorkerEventType.UPDATE_PROCESSING_PROGRESS;
  paneId: string;
  label: ProcessingLabel;
  error?: string;
}

/** The request for preparing a popup. */
export declare interface PreparePopupRequest extends WorkerEventBase {
  eventType: WorkerEventType.PREPARE_POPUP_REQ;
  modelGraphId: string;
  // The model graph of this pane id will be duplicated.
  paneId: string;
  // The duplicated model graph will be cached in this rendererId.
  rendererId: string;
  groupNodeId: string;
  initialPosition: Point;
}

/** The response for preparing a popup. */
export declare interface PreparePopupResponse extends WorkerEventBase {
  eventType: WorkerEventType.PREPARE_POPUP_RESP;
  paneId: string;
  rendererId: string;
  modelGraph: ModelGraph;
  groupNodeId: string;
  initialPosition: Point;
}

/** The request for cleaning up the worker. */
export declare interface CleanupRequest extends WorkerEventBase {
  eventType: WorkerEventType.CLEANUP;
}

/**
 * The request for updating the model graph cache with the given
 * node attributes.
 */
export declare interface UpdateModelGraphCacheWithNodeAttributesRequest
  extends WorkerEventBase {
  eventType: WorkerEventType.UPDATE_MODEL_GRAPH_CACHE_WITH_NODE_ATTRIBUTES;
  modelGraphId: string;
  nodeId: string;
  attrs: NodeAttributePairs;
  paneId: string;
}

/** Union of all worker events. */
export type WorkerEvent =
  | ProcessGraphRequest
  | ProcessGraphResponse
  | ExpandOrCollapseGroupNodeRequest
  | ExpandOrCollapseGroupNodeResponse
  | RelayoutGraphRequest
  | RelayoutGraphResponse
  | LocateNodeRequest
  | LocateNodeResponse
  | UpdateProcessingProgressRequest
  | PreparePopupRequest
  | PreparePopupResponse
  | CleanupRequest
  | UpdateModelGraphCacheWithNodeAttributesRequest;
