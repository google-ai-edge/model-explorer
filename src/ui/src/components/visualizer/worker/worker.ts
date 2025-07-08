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

/// <reference lib="webworker" />

import {Graph} from '../common/input_graph';
import {GroupNode, ModelGraph} from '../common/model_graph';
import {NodeDataProviderRunData, ShowOnNodeItemData} from '../common/types';
import {
  getDeepestExpandedGroupNodeIds,
  isGroupNode,
  isOpNode,
} from '../common/utils';
import {VisualizerConfig} from '../common/visualizer_config';
import {
  ExpandOrCollapseGroupNodeResponse,
  LocateNodeResponse,
  PreparePopupResponse,
  ProcessGraphResponse,
  ProcessingLabel,
  RelayoutGraphResponse,
  WorkerEvent,
  WorkerEventType,
} from '../common/worker_events';

import {Dagre} from './dagre_types';
import {GraphExpander} from './graph_expander';
import {GraphLayout} from './graph_layout';
import {GraphProcessor} from './graph_processor';
import {IdenticalGroupsFinder} from './identical_groups_finder';
import {updateProcessingProgress} from './utils';

import '../../../../public/static_files/worker_deps.js';

declare var dagre: Dagre;

// <rendererId + ModelGraphId> -> ModelGraph
let MODEL_GRAPHS_CACHE: Record<string, ModelGraph> = {};

self.addEventListener('message', (event: Event) => {
  const workerEvent = (event as ExtendableMessageEvent).data as WorkerEvent;
  switch (workerEvent.eventType) {
    // Handle processing input graph.
    case WorkerEventType.PROCESS_GRAPH_REQ: {
      const modelGraph = handleProcessGraph(
        workerEvent.paneId,
        workerEvent.graph,
        workerEvent.showOnNodeItemTypes,
        workerEvent.nodeDataProviderRuns,
        workerEvent.config,
        workerEvent.groupNodeChildrenCountThreshold,
        workerEvent.flattenLayers,
        workerEvent.keepLayersWithASingleChild,
        workerEvent.initialLayout,
      );
      cacheModelGraph(modelGraph, workerEvent.paneId);
      const resp: ProcessGraphResponse = {
        eventType: WorkerEventType.PROCESS_GRAPH_RESP,
        modelGraph,
        paneId: workerEvent.paneId,
      };
      postMessage(resp);
      break;
    }
    case WorkerEventType.PREPARE_POPUP_REQ: {
      // Clone the model graph from the pane to a graph for the renderer id
      // (the renderer id for the model graph in the popup).
      const modelGraph = getCachedModelGraph(
        workerEvent.modelGraphId,
        workerEvent.paneId,
      );
      const clonedModelGraph = JSON.parse(
        JSON.stringify(modelGraph),
      ) as ModelGraph;
      cacheModelGraph(clonedModelGraph, workerEvent.rendererId);

      const resp: PreparePopupResponse = {
        eventType: WorkerEventType.PREPARE_POPUP_RESP,
        modelGraph,
        paneId: workerEvent.paneId,
        rendererId: workerEvent.rendererId,
        groupNodeId: workerEvent.groupNodeId,
        initialPosition: workerEvent.initialPosition,
      };
      postMessage(resp);
      break;
    }
    case WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_REQ: {
      const modelGraph = getCachedModelGraph(
        workerEvent.modelGraphId,
        workerEvent.rendererId,
      );
      let deepestExpandedGroupNodeIds: string[] = [];
      if (workerEvent.expand) {
        deepestExpandedGroupNodeIds = handleExpandGroupNode(
          modelGraph,
          workerEvent.groupNodeId,
          workerEvent.showOnNodeItemTypes,
          workerEvent.nodeDataProviderRuns,
          workerEvent.selectedNodeDataProviderRunId,
          workerEvent.all === true,
          workerEvent.config,
        );
      } else {
        deepestExpandedGroupNodeIds = handleCollapseGroupNode(
          modelGraph,
          workerEvent.groupNodeId,
          workerEvent.showOnNodeItemTypes,
          workerEvent.nodeDataProviderRuns,
          workerEvent.selectedNodeDataProviderRunId,
          workerEvent.all === true,
          workerEvent.config,
        );
      }
      cacheModelGraph(modelGraph, workerEvent.rendererId);
      const resp: ExpandOrCollapseGroupNodeResponse = {
        eventType: WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_RESP,
        modelGraph,
        expanded: workerEvent.expand,
        groupNodeId: workerEvent.groupNodeId,
        rendererId: workerEvent.rendererId,
        deepestExpandedGroupNodeIds,
      };
      postMessage(resp);
      break;
    }
    case WorkerEventType.RELAYOUT_GRAPH_REQ: {
      const modelGraph = getCachedModelGraph(
        workerEvent.modelGraphId,
        workerEvent.rendererId,
      );
      handleReLayoutGraph(
        modelGraph,
        workerEvent.showOnNodeItemTypes,
        workerEvent.nodeDataProviderRuns,
        workerEvent.selectedNodeDataProviderRunId,
        workerEvent.targetDeepestGroupNodeIdsToExpand,
        workerEvent.clearAllExpandStates,
        workerEvent.config,
      );
      cacheModelGraph(modelGraph, workerEvent.rendererId);
      const resp: RelayoutGraphResponse = {
        eventType: WorkerEventType.RELAYOUT_GRAPH_RESP,
        modelGraph,
        selectedNodeId: workerEvent.selectedNodeId,
        rendererId: workerEvent.rendererId,
        forRestoringUiState: workerEvent.forRestoringUiState,
        rectToZoomFit: workerEvent.rectToZoomFit,
        forRestoringSnapshotAfterTogglingFlattenLayers:
          workerEvent.forRestoringSnapshotAfterTogglingFlattenLayers,
        targetDeepestGroupNodeIdsToExpand:
          workerEvent.targetDeepestGroupNodeIdsToExpand,
        triggerNavigationSync: workerEvent.triggerNavigationSync,
      };
      postMessage(resp);
      break;
    }
    case WorkerEventType.LOCATE_NODE_REQ: {
      const modelGraph = getCachedModelGraph(
        workerEvent.modelGraphId,
        workerEvent.rendererId,
      );
      const deepestExpandedGroupNodeIds = handleLocateNode(
        modelGraph,
        workerEvent.showOnNodeItemTypes,
        workerEvent.nodeDataProviderRuns,
        workerEvent.selectedNodeDataProviderRunId,
        workerEvent.nodeId,
        workerEvent.config,
      );
      cacheModelGraph(modelGraph, workerEvent.rendererId);
      const resp: LocateNodeResponse = {
        eventType: WorkerEventType.LOCATE_NODE_RESP,
        modelGraph,
        nodeId: workerEvent.nodeId,
        rendererId: workerEvent.rendererId,
        deepestExpandedGroupNodeIds,
        noNodeShake: workerEvent.noNodeShake,
        select: workerEvent.select,
      };
      postMessage(resp);
      break;
    }
    case WorkerEventType.CLEANUP: {
      MODEL_GRAPHS_CACHE = {};
      break;
    }
    case WorkerEventType.UPDATE_MODEL_GRAPH_CACHE_WITH_NODE_ATTRIBUTES: {
      const cachedModelGraph = getCachedModelGraph(
        workerEvent.modelGraphId,
        workerEvent.paneId,
      );
      if (cachedModelGraph) {
        const node = cachedModelGraph.nodesById[workerEvent.nodeId];
        if (node && isOpNode(node)) {
          node.attrs = {...node.attrs, ...workerEvent.attrs};
        }
      }
      break;
    }
    default:
      break;
  }
});

function handleProcessGraph(
  paneId: string,
  graph: Graph,
  showItemOnNodeTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  config?: VisualizerConfig,
  groupNodeChildrenCountThreshold?: number,
  flattenLayers?: boolean,
  keepLayersWithASingleChild?: boolean,
  initialLayout?: boolean,
): ModelGraph {
  let error: string | undefined = undefined;

  // Processes the given input graph `Graph` into a `ModelGraph`.
  const processor = new GraphProcessor(
    paneId,
    graph,
    config,
    showItemOnNodeTypes,
    {},
    groupNodeChildrenCountThreshold,
    false,
    flattenLayers,
    keepLayersWithASingleChild,
  );
  const modelGraph = processor.process();

  // Check nodes with empty ids.
  if (modelGraph.nodesById[''] != null) {
    error =
      'Some nodes have empty strings as ids which will cause layout failures. See console for details.';
    console.warn('Nodes with empty ids', modelGraph.nodesById['']);
  }

  // Do the initial layout.
  if (!error && initialLayout) {
    const layout = new GraphLayout(
      modelGraph,
      dagre,
      showItemOnNodeTypes,
      nodeDataProviderRuns,
      undefined,
    );
    try {
      layout.layout();
    } catch (e) {
      error = `Failed to layout graph: ${e}`;
    }
  }
  updateProcessingProgress(
    paneId,
    ProcessingLabel.LAYING_OUT_ROOT_LAYER,
    error,
  );

  // Find identical groups.
  const identicalGroupsFinder = new IdenticalGroupsFinder(modelGraph);
  identicalGroupsFinder.markIdenticalGroups();
  updateProcessingProgress(paneId, ProcessingLabel.FINDING_IDENTICAL_LAYERS);
  return modelGraph;
}

function handleExpandGroupNode(
  modelGraph: ModelGraph,
  groupNodeId: string | undefined,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  selectedNodeDataProviderRunId: string | undefined,
  all: boolean,
  config?: VisualizerConfig,
): string[] {
  const expander = new GraphExpander(
    modelGraph,
    dagre,
    showOnNodeItemTypes,
    nodeDataProviderRuns,
    selectedNodeDataProviderRunId,
    false,
    config,
  );

  // Expane group node.
  if (groupNodeId != null) {
    let deepestExpandedGroupNodeId: string[] | undefined = undefined;
    const groupNode = modelGraph.nodesById[groupNodeId];
    if (groupNode && isGroupNode(groupNode)) {
      groupNode.expanded = true;
      // Recursively expand child group node if there is only one child.
      let curGroupNode = groupNode;
      while (true) {
        const childrenIds = curGroupNode.nsChildrenIds || [];
        if (childrenIds.length === 1) {
          const child = modelGraph.nodesById[childrenIds[0]];
          if (child && isGroupNode(child)) {
            child.expanded = true;
            curGroupNode = child;
          } else {
            break;
          }
        } else {
          break;
        }
      }
      // Get the deepest expanded group nodes from the curGroupNode and we will
      // be doing relayout from there.
      const ids: string[] = [];
      getDeepestExpandedGroupNodeIds(curGroupNode, modelGraph, ids);
      deepestExpandedGroupNodeId = ids.length === 0 ? [curGroupNode.id] : ids;
      // Clear layout data for all nodes under curGroupNode.
      //
      // This is necessary because the node overlay might have been changed so
      // we need to re-calculate the node sizes.
      for (const nodeId of curGroupNode.descendantsNodeIds || []) {
        const node = modelGraph.nodesById[nodeId];
        node.width = undefined;
        node.height = undefined;
      }
    }
    if (all) {
      for (const childNodeId of (groupNode as GroupNode).descendantsNodeIds ||
        []) {
        const node = modelGraph.nodesById[childNodeId];
        if (isGroupNode(node)) {
          node.expanded = true;
        }
      }
      deepestExpandedGroupNodeId = undefined;
    }
    expander.reLayoutGraph(deepestExpandedGroupNodeId);

    const ids: string[] = [];
    getDeepestExpandedGroupNodeIds(undefined, modelGraph, ids);
    return ids;
  }
  // Expand all group nodes in the graph.
  else {
    return expander.expandAllGroups();
  }
}

function handleCollapseGroupNode(
  modelGraph: ModelGraph,
  groupNodeId: string | undefined,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  selectedNodeDataProviderRunId: string | undefined,
  all: boolean,
  config?: VisualizerConfig,
): string[] {
  const expander = new GraphExpander(
    modelGraph,
    dagre,
    showOnNodeItemTypes,
    nodeDataProviderRuns,
    selectedNodeDataProviderRunId,
    false,
    config,
  );

  if (groupNodeId != null) {
    if (all) {
      const groupNode = modelGraph.nodesById[groupNodeId] as GroupNode;
      for (const childNodeId of groupNode.descendantsNodeIds || []) {
        const node = modelGraph.nodesById[childNodeId];
        if (isGroupNode(node)) {
          node.expanded = false;
          node.width = undefined;
          node.height = undefined;
          delete modelGraph.edgesByGroupNodeIds[node.id];
        }
      }
    }
    return expander.collapseGroupNode(groupNodeId);
  } else {
    return expander.collapseAllGroup();
  }
}

function handleReLayoutGraph(
  modelGraph: ModelGraph,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  selectedNodeDataProviderRunId: string | undefined,
  targetDeepestGroupNodeIdsToExpand?: string[],
  clearAllExpandStates?: boolean,
  config?: VisualizerConfig,
) {
  const expander = new GraphExpander(
    modelGraph,
    dagre,
    showOnNodeItemTypes,
    nodeDataProviderRuns,
    selectedNodeDataProviderRunId,
    false,
    config,
  );
  expander.reLayoutGraph(
    targetDeepestGroupNodeIdsToExpand,
    clearAllExpandStates,
  );
}

function handleLocateNode(
  modelGraph: ModelGraph,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  nodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  selectedNodeDataProviderRunId: string | undefined,
  nodeId: string,
  config?: VisualizerConfig,
): string[] {
  const expander = new GraphExpander(
    modelGraph,
    dagre,
    showOnNodeItemTypes,
    nodeDataProviderRuns,
    selectedNodeDataProviderRunId,
    false,
    config,
  );
  return expander.expandToRevealNode(nodeId);
}

function cacheModelGraph(modelGraph: ModelGraph, rendererId: string) {
  MODEL_GRAPHS_CACHE[getModelGraphKey(modelGraph.id, rendererId)] = modelGraph;
}

function getCachedModelGraph(
  modelGraphId: string,
  rendererId: string,
): ModelGraph {
  const cachedModelGraph =
    MODEL_GRAPHS_CACHE[getModelGraphKey(modelGraphId, rendererId)];
  if (cachedModelGraph == null) {
    throw new Error(
      `ModelGraph with id "${modelGraphId}" not found for rendererId "${rendererId}"`,
    );
  }
  return cachedModelGraph;
}

function getModelGraphKey(modelGraphId: string, rendererId: string): string {
  return `${modelGraphId}___${rendererId}`;
}
