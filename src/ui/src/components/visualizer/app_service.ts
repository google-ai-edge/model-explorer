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

import {Injectable, signal} from '@angular/core';
import {Subject} from 'rxjs';

import {
  DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
  LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM,
  LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM_TYPES_V2,
  LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES,
} from './common/consts';
import {Graph, GraphCollection, GraphWithLevel} from './common/input_graph';
import {ModelGraph, ModelNode} from './common/model_graph';
import {
  AddSnapshotInfo,
  Command,
  DownloadAsPngInfo,
  ExpandOrCollapseAllGraphLayersInfo,
  LocateNodeInfo,
  ModelGraphProcessedEvent,
  NodeInfo,
  Pane,
  RendererInfo,
  RendererOwner,
  RestoreSnapshotInfo,
  SearchResults,
  SelectedNodeInfo,
  ShowOnEdgeItemData,
  ShowOnEdgeItemOldData,
  ShowOnEdgeItemType,
  ShowOnNodeItemData,
  SnapshotData,
} from './common/types';
import {genUid, isOpNode} from './common/utils';
import {VisualizerConfig} from './common/visualizer_config';
import {VisualizerUiState} from './common/visualizer_ui_state';
import {
  ProcessGraphRequest,
  WorkerEvent,
  WorkerEventType,
} from './common/worker_events';
import {LocalStorageService} from './local_storage_service';
import {UiStateService} from './ui_state_service';
import {WorkerService} from './worker_service';

/**
 * A service to manage shared data and their updates.
 *
 * It uses signals to store shared data. Various components can react to changes
 * to these signals.
 */
@Injectable()
export class AppService {
  readonly curGraphCollections = signal<GraphCollection[]>([]);

  readonly curToLocateNodeInfo = signal<LocateNodeInfo | undefined>(undefined);

  readonly curSelectedRenderer = signal<RendererInfo | undefined>(undefined);

  readonly spaceKeyToZoomFitClicked = new Subject<{}>();

  readonly searchKeyClicked = new Subject<{}>();

  readonly addSnapshotClicked = new Subject<AddSnapshotInfo>();

  readonly curSnapshotToRestore = new Subject<RestoreSnapshotInfo>();

  readonly expandOrCollapseAllGraphLayersClicked =
    new Subject<ExpandOrCollapseAllGraphLayersInfo>();

  readonly downloadAsPngClicked = new Subject<DownloadAsPngInfo>();

  readonly config = signal<VisualizerConfig | undefined>(undefined);

  readonly curInitialUiState = signal<VisualizerUiState | undefined>(undefined);

  // Panes in the app. Create a single pane by default.
  readonly panes = signal<Pane[]>([
    {
      id: genUid(),
      widthFraction: 1,
    },
  ]);

  readonly selectedPaneId = signal<string>(this.panes()[0].id);

  readonly modelGraphProcessed$ = new Subject<ModelGraphProcessedEvent>();

  readonly remoteNodeDataPaths = signal<string[]>([]);

  readonly selectedNode = signal<NodeInfo | undefined>(undefined);

  readonly hoveredNode = signal<NodeInfo | undefined>(undefined);

  readonly doubleClickedNode = signal<NodeInfo | undefined>(undefined);

  readonly command = new Subject<Command>();

  testMode = false;

  private groupNodeChildrenCountThresholdFromUrl: string | null = null;

  private paneIdToGraph: Record<string, Graph> = {};

  // An index to all "current" model graphs.
  //
  // After each operation such as expanding/collapsing a layer, the model graph
  // will be stored here.
  private paneIdToCurModelGraphs: Record<string, ModelGraph> = {};

  constructor(
    private readonly localStorageService: LocalStorageService,
    private readonly uiStateService: UiStateService,
    private readonly workerService: WorkerService,
  ) {
    this.listenToWorker();
    this.init();
  }

  addGraphCollections(graphCollections: GraphCollection[]) {
    this.curGraphCollections.update((prevCollections) => {
      const newCollections = [...prevCollections];
      // For graphs in a collection, sort them by number of nodes in descending
      // order.
      //
      // Original graph id to count.
      const graphIdToCount: Record<string, number> = {};
      for (const collection of graphCollections) {
        // Id to graph, and eliminate duplicated ids.
        const graphById: Record<string, Graph> = {};
        const graphIdToRenamedId: Record<string, string> = {};
        for (const graph of collection.graphs) {
          // Ensure there is no empty graph id.
          if (graph.id == null || graph.id === '') {
            graph.id = 'unnamed_graph';
          }
          const originalGraphId = graph.id;
          let count: number | undefined = graphIdToCount[originalGraphId];
          if (count == null) {
            graphIdToCount[originalGraphId] = 0;
            count = 0;
          } else {
            // Duplicated id detected. Update the graph id.
            graph.id = `${graph.id} (${count + 1})`;
            graphIdToRenamedId[originalGraphId] = graph.id;
          }
          graphIdToCount[originalGraphId] = count + 1;

          graphById[graph.id] = graph;
          graph.collectionLabel = collection.label;
        }

        // Find subgraphs for each graph.
        for (const graph of collection.graphs) {
          for (const node of graph.nodes) {
            if (node.subgraphIds != null && node.subgraphIds.length > 0) {
              node.subgraphIds = node.subgraphIds.map(
                (id) => graphIdToRenamedId[id] || id,
              );
              if (graph.subGraphIds == null) {
                graph.subGraphIds = [];
              }
              graph.subGraphIds.push(...node.subgraphIds);
              for (const subgraphId of node.subgraphIds) {
                const subgraph = graphById[subgraphId];
                if (subgraph) {
                  if (subgraph.parentGraphIds == null) {
                    subgraph.parentGraphIds = [];
                  }
                  if (subgraph.parentGraphIds.includes(graph.id)) {
                    subgraph.parentGraphIds.push(graph.id);
                  }
                }
              }
            }
          }
        }

        // Find the 'root' graphs.
        const rootGraphs: Graph[] = collection.graphs.filter(
          (graph) => graph.parentGraphIds == null,
        );

        // DFS from root graphs.
        const dfsOrderedGraphs: GraphWithLevel[] = [];
        const visitGraph = (root?: Graph, level = 0) => {
          let graphs: Graph[] = [];
          if (root == null) {
            graphs = rootGraphs;
          } else {
            graphs = (root.subGraphIds || [])
              .map((id) => graphById[id])
              .filter((graphs) => graphs != null);
          }

          // Dedup graphs by their ids.
          const uniqueGraphs: Graph[] = [];
          const seenIds: Record<string, boolean> = {};
          for (const graph of graphs) {
            if (!seenIds[graph.id]) {
              uniqueGraphs.push(graph);
              seenIds[graph.id] = true;
            }
          }
          graphs = uniqueGraphs;

          // Sort by node count.
          graphs.sort((g1, g2) => g2.nodes.length - g1.nodes.length);
          for (const graph of graphs) {
            dfsOrderedGraphs.push({graph, level});
            visitGraph(graph, level + 1);
          }
        };
        visitGraph();
        collection.graphsWithLevel = dfsOrderedGraphs;
      }
      newCollections.push(...graphCollections);
      return newCollections;
    });
  }

  selectGraphInPane(
    graph: Graph,
    paneIndex: number,
    flattenLayers = false,
    snapshot?: SnapshotData,
    initialLayout = true,
  ) {
    if (paneIndex === 1 && this.panes().length === 1) {
      this.openGraphInSplitPane(graph);
      return;
    }

    const curSelectedGraphId = this.panes()[paneIndex].modelGraph?.id || '';
    if (curSelectedGraphId === graph.id) {
      return;
    }

    const pane = this.panes()[paneIndex];
    pane.searchResults = undefined;
    pane.selectedNodeDataProviderRunId = undefined;
    const paneId = pane.id;
    this.paneIdToGraph[paneId] = graph;
    this.uiStateService.setSelectedGraphId(
      graph.id,
      graph.collectionLabel || '',
      this.getPaneIndexById(paneId),
    );

    // Update the last subgraph breadcrumb graph id to match the currently
    // selected graph.
    if (
      pane.subgraphBreadcrumbs != null &&
      pane.subgraphBreadcrumbs.length > 0
    ) {
      const subgraphBreadcrumbs = [...pane.subgraphBreadcrumbs];
      subgraphBreadcrumbs[subgraphBreadcrumbs.length - 1].graphId = graph.id;
      pane.subgraphBreadcrumbs = subgraphBreadcrumbs;
    }

    // Process the graph.
    this.processGraph(paneId, flattenLayers, snapshot, initialLayout);
  }

  selectGraphInCurrentPane(
    graph: Graph,
    flattenLayers = false,
    snapshot?: SnapshotData,
    initialLayout = true,
  ) {
    this.selectGraphInPane(
      graph,
      this.getPaneIndexById(this.selectedPaneId()),
      flattenLayers,
      snapshot,
      initialLayout,
    );
  }

  openGraphInSplitPane(
    graph: Graph,
    flattenLayers = false,
    initialLayout = true,
    openToLeft = false,
  ) {
    // Keep the current pane and remove the other pane when there are two panes.
    if (this.panes().length === 2) {
      this.panes.update((panes) => {
        if (openToLeft) {
          return [panes[1]];
        } else {
          return [panes[0]];
        }
      });
    }

    // Add a new pane.
    const paneId = genUid();
    this.paneIdToGraph[paneId] = graph;
    this.panes.update((panes) => {
      const firstPane = panes[0];
      firstPane.widthFraction = 0.5;
      const newPane: Pane = {
        id: paneId,
        widthFraction: 0.5,
        flattenLayers,
        showOnNodeItemTypes: {[paneId]: this.getSavedShowOnNodeItemTypes()},
      };
      const savedShowOnEdgeItem = this.getSavedShowOnEdgeItem();
      if (savedShowOnEdgeItem) {
        newPane.showOnEdgeItems = {[paneId]: savedShowOnEdgeItem};
      }
      if (openToLeft) {
        panes.unshift(newPane);
      } else {
        panes.push(newPane);
      }
      return [...panes];
    });

    this.curSelectedRenderer.set({
      // Pane id is the same as the renderer id.
      id: paneId,
      ownerType: RendererOwner.GRAPH_PANEL,
    });

    const paneIndex = this.getPaneIndexById(paneId);
    this.uiStateService.addPane();
    // Select it by default.
    //
    // Need to put it after adding pane in uiStateService so that the second
    // pane is available.
    this.selectPane(paneId);
    this.uiStateService.setSelectedGraphId(
      graph.id,
      graph.collectionLabel || '',
      paneIndex,
    );
    this.uiStateService.setFlattenLayers(flattenLayers, paneIndex);

    // Kick off graph processing.
    const processGraphReq: ProcessGraphRequest = {
      eventType: WorkerEventType.PROCESS_GRAPH_REQ,
      graph,
      showOnNodeItemTypes: this.getShowOnNodeItemTypes(paneId, paneId),
      nodeDataProviderRuns: {},
      config: this.config ? this.config() : undefined,
      paneId,
      groupNodeChildrenCountThreshold:
        this.getGroupNodeChildrenCountThreshold(),
      flattenLayers,
      keepLayersWithASingleChild: this.config()?.keepLayersWithASingleChild,
      initialLayout,
    };
    this.workerService.worker.postMessage(processGraphReq);
  }

  getIsGraphInRightPane(graphId: string): boolean {
    const panes = this.panes();
    return panes.length === 2 && panes[1].modelGraph?.id === graphId;
  }

  processGraph(
    paneId: string,
    flattenLayers = false,
    snapshotToRestore?: SnapshotData,
    initialLayout = true,
  ) {
    // Store snapshotToResotre into pane if set.
    if (snapshotToRestore != null) {
      const pane = this.getPaneById(paneId);
      if (pane) {
        pane.snapshotToRestore = snapshotToRestore;
      }
    }

    // Process the graph.
    //
    // TODO: properly cache the processed graph.
    this.setPaneLoading(paneId);
    const processGraphReq: ProcessGraphRequest = {
      eventType: WorkerEventType.PROCESS_GRAPH_REQ,
      graph: this.paneIdToGraph[paneId],
      showOnNodeItemTypes: this.getShowOnNodeItemTypes(paneId, paneId),
      nodeDataProviderRuns: {},
      config: this.config ? this.config() : undefined,
      paneId,
      groupNodeChildrenCountThreshold:
        this.getGroupNodeChildrenCountThreshold(),
      flattenLayers,
      keepLayersWithASingleChild: this.config()?.keepLayersWithASingleChild,
      initialLayout,
    };
    this.workerService.worker.postMessage(processGraphReq);
  }

  setFlattenLayersInCurrentPane(flatten: boolean) {
    const pane = this.getSelectedPane();
    if (!pane) {
      return;
    }
    this.panes.update((panes) => {
      pane.flattenLayers = flatten;
      pane.searchResults = undefined;
      return [...panes];
    });

    const paneIndex = this.getPaneIndexById(pane.id);
    this.uiStateService.setFlattenLayers(flatten, paneIndex);
    this.uiStateService.setDeepestExpandedGroupNodeIds([], paneIndex);
  }

  toggleFlattenLayers(paneId: string) {
    const pane = this.getPaneById(paneId);
    if (!pane) {
      return;
    }
    const curFlatten = pane.flattenLayers === true;
    this.panes.update((panes) => {
      pane.flattenLayers = !curFlatten;
      pane.searchResults = undefined;
      return [...panes];
    });

    const paneIndex = this.getPaneIndexById(paneId);
    this.uiStateService.setFlattenLayers(!curFlatten, paneIndex);
    this.uiStateService.setDeepestExpandedGroupNodeIds([], paneIndex);
  }

  getFlattenLayers(paneId: string): boolean {
    return this.getPaneById(paneId)?.flattenLayers === true;
  }

  selectPane(paneId: string) {
    this.selectedPaneId.set(paneId);
    this.uiStateService.selectPane(this.getPaneIndexById(paneId));
  }

  selectPaneByIndex(paneIndex: number) {
    const pane = this.panes()[paneIndex];
    if (pane) {
      this.selectPane(pane.id);
    }
  }

  selectNode(paneId: string, info?: SelectedNodeInfo) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.selectedNodeInfo = info;
      return [...panes];
    });

    this.uiStateService.setSelectedNodeId(
      info?.nodeId || '',
      this.getPaneIndexById(paneId),
    );

    // Post message to parent.
    const modelGraph = this.getPaneById(paneId)?.modelGraph;
    if (modelGraph) {
      const nodeId = info?.nodeId || '';
      const node = modelGraph.nodesById[nodeId];
      if (node && isOpNode(node) && window.parent) {
        const outputMetadata = node.outputsMetadata || {};
        const tensorNames: string[] = [];
        for (const outputId of Object.keys(outputMetadata)) {
          const curMetadata = outputMetadata[outputId];
          const tensorName = curMetadata['tensor_name'];
          if (tensorName) {
            tensorNames.push(tensorName);
          }
        }
        window.parent.postMessage(
          {
            'cmd': 'model-explorer-node-selected',
            'nodeId': nodeId,
            'outputTensorNames': tensorNames,
          },
          '*',
        );
      }
    }

    // Trigger event on visualizer component.
    if (modelGraph) {
      const nodeId = info?.nodeId || '';
      this.updateSelectedNode(
        nodeId,
        modelGraph.id,
        modelGraph.collectionLabel,
        modelGraph.nodesById[nodeId],
        paneId,
      );
    }
  }

  getModelGraphFromSelectedPane(): ModelGraph | undefined {
    return this.getPaneById(this.selectedPaneId())?.modelGraph;
  }

  getModelGraphFromPane(paneId: string): ModelGraph | undefined {
    return this.getPaneById(paneId)?.modelGraph;
  }

  getModelGraphFromPaneIndex(paneIndex: number): ModelGraph | undefined {
    const pane = this.panes()[paneIndex];
    return pane?.modelGraph;
  }

  getSelectedNodeInfoFromSelectedPane(): SelectedNodeInfo | undefined {
    return this.getPaneById(this.selectedPaneId())?.selectedNodeInfo;
  }

  getSelectedPane(): Pane | undefined {
    return this.getPaneById(this.selectedPaneId());
  }

  setPaneWidthFraction(leftFraction: number) {
    this.panes.update((panes) => {
      if (panes.length !== 2) {
        return panes;
      }
      panes[0].widthFraction = leftFraction;
      panes[1].widthFraction = 1 - leftFraction;
      return [...panes];
    });
    this.uiStateService.resizePane(leftFraction);
  }

  setSelectedNodeDataProviderRunId(paneId: string, runId: string | undefined) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.selectedNodeDataProviderRunId = runId;
      return [...panes];
    });
  }

  getSelectedNodeDataProviderRunId(paneId: string): string | undefined {
    return this.getPaneById(paneId)?.selectedNodeDataProviderRunId;
  }

  setPaneHasArtificialLayers(paneId: string, hasArtificialLayers: boolean) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.hasArtificialLayers = hasArtificialLayers;
      return [...panes];
    });
  }

  setNodeToReveal(paneId: string, nodeId: string | undefined) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.nodeIdToReveal = nodeId;
      return [...panes];
    });
  }

  closePane(paneId: string) {
    delete this.paneIdToGraph[paneId];
    delete this.paneIdToCurModelGraphs[paneId];
    this.panes.update((panes) => {
      // Remove pane.
      const index = panes.findIndex((pane) => pane.id === paneId);
      if (index >= 0) {
        panes.splice(index, 1);
        this.uiStateService.removePane(index);
      }

      // Update width.
      panes[0].widthFraction = 1;
      return [...panes];
    });

    // Set selected pane.
    this.selectPane(this.panes()[0].id);
  }

  swapPane() {
    this.panes.update((panes) => {
      if (panes.length !== 2) {
        return panes;
      }
      return [panes[1], panes[0]];
    });

    this.uiStateService.swapPane();
  }

  getPaneById(id: string): Pane | undefined {
    return this.panes().find((pane) => pane.id === id);
  }

  getPaneIndexById(id: string): number {
    return this.panes().findIndex((pane) => pane.id === id);
  }

  getPaneIdByIndex(index: number): string {
    return this.panes()[index]?.id ?? '';
  }

  addSnapshot(snapshotData: SnapshotData, graphId: string, paneId: string) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (pane) {
        if (pane.snapshots == null) {
          pane.snapshots = {};
        }
        if (pane.snapshots[graphId] == null) {
          pane.snapshots[graphId] = [];
        }
        pane.snapshots[graphId].push(snapshotData);
      }
      return [...panes];
    });
  }

  deleteSnapshot(index: number, graphId: string, paneId: string) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (pane && pane.snapshots && pane.snapshots[graphId]) {
        pane.snapshots[graphId].splice(index, 1);
      }
      return [...panes];
    });
  }

  getGraphById(id: string): Graph | undefined {
    for (const collection of this.curGraphCollections()) {
      for (const graph of collection.graphs) {
        if (graph.id === id) {
          return graph;
        }
      }
    }
    return undefined;
  }

  addSubgraphBreadcrumbItem(
    paneId: string,
    prevGraphId: string,
    curGraphId: string,
    prevGraphSnapshot: SnapshotData,
  ) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      const curSubgraphBreadcrumbs = [...(pane.subgraphBreadcrumbs || [])];
      if (curSubgraphBreadcrumbs.length === 0) {
        curSubgraphBreadcrumbs.push({
          graphId: prevGraphId,
          snapshot: prevGraphSnapshot,
        });
      } else {
        curSubgraphBreadcrumbs[curSubgraphBreadcrumbs.length - 1] = {
          graphId: prevGraphId,
          snapshot: prevGraphSnapshot,
        };
      }
      curSubgraphBreadcrumbs.push({graphId: curGraphId});
      pane.subgraphBreadcrumbs = curSubgraphBreadcrumbs;
      return [...panes];
    });
  }

  setCurrentSubgraphBreadcrumb(paneId: string, index: number) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      let curSubgraphBreadcrumbs = [...(pane.subgraphBreadcrumbs || [])];
      curSubgraphBreadcrumbs.splice(index + 1);
      if (curSubgraphBreadcrumbs.length === 1) {
        curSubgraphBreadcrumbs = [];
      }
      pane.subgraphBreadcrumbs = curSubgraphBreadcrumbs;
      return [...panes];
    });
  }

  setSearchResults(paneId: string, searchResults: SearchResults) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.searchResults = searchResults;
      return [...panes];
    });
  }

  clearSearchResults(paneId: string) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      pane.searchResults = {results: {}};
      return [...panes];
    });
  }

  toggleShowOnNode(
    paneId: string,
    rendererId: string,
    type: string,
    valueToSet?: boolean,
  ) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      if (!pane.showOnNodeItemTypes) {
        pane.showOnNodeItemTypes = {};
      }
      if (pane.showOnNodeItemTypes[rendererId] == null) {
        pane.showOnNodeItemTypes[rendererId] = {};
      }
      if (pane.showOnNodeItemTypes[rendererId][type] == null) {
        pane.showOnNodeItemTypes[rendererId][type] = {selected: false};
      }
      const curRendererShowOnNodeItemTypes =
        pane.showOnNodeItemTypes[rendererId][type].selected;
      pane.showOnNodeItemTypes[rendererId] = {
        ...pane.showOnNodeItemTypes[rendererId],
      };
      pane.showOnNodeItemTypes[rendererId][type].selected =
        valueToSet == null ? !curRendererShowOnNodeItemTypes : valueToSet;
      pane.showOnNodeItemTypes = {
        ...pane.showOnNodeItemTypes,
      };
      return [...panes];
    });
  }

  setShowOnEdge(
    paneId: string,
    rendererId: string,
    type: string,
    filterText?: string,
    outputMetadataKey?: string,
    inputMetadataKey?: string,
    sourceNodeAttrKey?: string,
    targetNodeAttrKey?: string,
  ) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      if (!pane.showOnEdgeItems) {
        pane.showOnEdgeItems = {};
      }
      pane.showOnEdgeItems[rendererId] = {
        type,
        filterText,
        outputMetadataKey,
        inputMetadataKey,
        sourceNodeAttrKey,
        targetNodeAttrKey,
      };
      pane.showOnEdgeItems = {
        ...pane.showOnEdgeItems,
      };
      return [...panes];
    });
  }

  setShowOnNodeFilter(
    paneId: string,
    rendererId: string,
    type: string,
    filterRegex: string,
  ) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      if (!pane.showOnNodeItemTypes) {
        pane.showOnNodeItemTypes = {};
      }
      if (pane.showOnNodeItemTypes[rendererId] == null) {
        pane.showOnNodeItemTypes[rendererId] = {};
      }
      if (pane.showOnNodeItemTypes[rendererId][type] == null) {
        pane.showOnNodeItemTypes[rendererId][type] = {selected: false};
      }
      pane.showOnNodeItemTypes[rendererId][type].filterRegex = filterRegex;
      pane.showOnNodeItemTypes = {
        ...pane.showOnNodeItemTypes,
      };
      return [...panes];
    });
  }

  setShowOnNode(
    paneId: string,
    rendererId: string,
    types: Record<string, ShowOnNodeItemData>,
  ) {
    this.panes.update((panes) => {
      const pane = this.getPaneById(paneId);
      if (!pane) {
        return panes;
      }
      if (!pane.showOnNodeItemTypes) {
        pane.showOnNodeItemTypes = {};
      }
      pane.showOnNodeItemTypes = {
        ...pane.showOnNodeItemTypes,
      };
      pane.showOnNodeItemTypes[rendererId] = types;
      return [...panes];
    });
  }

  deleteShowOnNodeItemType(types: string[]) {
    this.panes.update((panes) => {
      for (const pane of panes) {
        pane.showOnNodeItemTypes = {...pane.showOnNodeItemTypes};
        for (const rendererId of Object.keys(pane.showOnNodeItemTypes)) {
          for (const type of types) {
            const showOnNodeItemData =
              pane.showOnNodeItemTypes[rendererId][type];
            if (showOnNodeItemData) {
              showOnNodeItemData.selected = false;
            }
          }
        }
      }
      return [...panes];
    });
  }

  getShowOnNodeItemTypes(
    paneId: string,
    rendererId: string,
  ): Record<string, ShowOnNodeItemData> {
    const pane = this.getPaneById(paneId);
    if (!pane) {
      return {};
    }
    // Make sure to return a copy of the data so that caller's won't accidentally
    // mutate it.
    return JSON.parse(
      JSON.stringify((pane.showOnNodeItemTypes || {})[rendererId] || {}),
    ) as Record<string, ShowOnNodeItemData>;
  }

  getSavedShowOnNodeItemTypes(): Record<string, ShowOnNodeItemData> {
    let curTypes: Record<string, ShowOnNodeItemData> = {};
    if (!this.testMode) {
      const data = this.localStorageService.getItem(
        LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES,
      );
      if (data) {
        curTypes = JSON.parse(data) as Record<string, ShowOnNodeItemData>;
      }
    }
    return curTypes;
  }

  getSavedShowOnEdgeItem(): ShowOnEdgeItemData | undefined {
    let curItem: ShowOnEdgeItemData | undefined = undefined;
    if (!this.testMode) {
      const data = this.localStorageService.getItem(
        LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM,
      );
      if (data) {
        curItem = JSON.parse(data) as ShowOnEdgeItemData;
      }
      // Try to load the old version of the data.
      else {
        const oldData = this.localStorageService.getItem(
          LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM_TYPES_V2,
        );
        if (oldData) {
          const oldItems = JSON.parse(oldData) as Record<
            string,
            ShowOnEdgeItemOldData
          >;
          if (oldItems[ShowOnEdgeItemType.TENSOR_SHAPE]?.selected) {
            curItem = {
              type: ShowOnEdgeItemType.TENSOR_SHAPE,
            };
          }
        }
      }
    }
    return curItem;
  }

  getShowOnEdgeItem(
    paneId: string,
    rendererId: string,
  ): ShowOnEdgeItemData | undefined {
    const pane = this.getPaneById(paneId);
    if (!pane) {
      return undefined;
    }
    // Make sure to return a copy of the data so that caller's won't accidentally
    // mutate it.
    const curShowOnEdgeItem = (pane.showOnEdgeItems || {})[rendererId];
    if (!curShowOnEdgeItem) {
      return undefined;
    }
    return JSON.parse(JSON.stringify(curShowOnEdgeItem)) as ShowOnEdgeItemData;
  }

  getGraphByPaneId(paneId: string): Graph {
    return this.paneIdToGraph[paneId];
  }

  updateCurrentModelGraph(paneId: string, modelGraph: ModelGraph) {
    this.paneIdToCurModelGraphs[paneId] = modelGraph;
  }

  getCurrentModelGraphFromPane(paneId: string): ModelGraph | undefined {
    return this.paneIdToCurModelGraphs[paneId];
  }

  updateSelectedNode(
    nodeId: string,
    graphId: string,
    collectionLabel: string,
    node?: ModelNode,
    paneId?: string,
  ) {
    const curSelectedNode = this.selectedNode();
    if (
      curSelectedNode?.nodeId !== nodeId ||
      curSelectedNode?.graphId !== graphId ||
      curSelectedNode?.collectionLabel !== collectionLabel
    ) {
      this.selectedNode.set({
        nodeId,
        graphId,
        collectionLabel,
        node,
        paneId,
      });
    }
  }

  updateHoveredNode(
    nodeId: string,
    graphId: string,
    collectionLabel: string,
    node?: ModelNode,
  ) {
    const curHoveredNode = this.hoveredNode();
    if (
      curHoveredNode?.nodeId !== nodeId ||
      curHoveredNode?.graphId !== graphId ||
      curHoveredNode?.collectionLabel !== collectionLabel
    ) {
      this.hoveredNode.set({
        nodeId,
        graphId,
        collectionLabel,
        node,
      });
    }
  }

  updateDoubleClickedNode(
    nodeId: string,
    graphId: string,
    collectionLabel: string,
    node?: ModelNode,
  ) {
    const curDoubleClickedNode = this.doubleClickedNode();
    if (
      curDoubleClickedNode?.nodeId !== nodeId ||
      curDoubleClickedNode?.graphId !== graphId ||
      curDoubleClickedNode?.collectionLabel !== collectionLabel
    ) {
      this.doubleClickedNode.set({
        nodeId,
        graphId,
        collectionLabel,
        node,
      });
    }
  }

  reset() {
    this.workerService.worker.postMessage({eventType: WorkerEventType.CLEANUP});

    this.curGraphCollections.set([]);
    this.curToLocateNodeInfo.set(undefined);
    this.curSelectedRenderer.set(undefined);
    this.config.set(undefined);
    this.curInitialUiState.set(undefined);
    this.panes.set([{id: genUid(), widthFraction: 1}]);
    this.selectedPaneId.set(this.panes()[0].id);
    this.remoteNodeDataPaths.set([]);
    this.groupNodeChildrenCountThresholdFromUrl = null;
    this.paneIdToGraph = {};
    this.paneIdToCurModelGraphs = {};

    this.init();
  }

  private listenToWorker() {
    this.workerService.worker.addEventListener('message', (event) => {
      const workerEvent = event.data as WorkerEvent;
      switch (workerEvent.eventType) {
        // A `Graph` is processed into a `ModelGraph`.
        case WorkerEventType.PROCESS_GRAPH_RESP:
          this.handleGraphProcessed(workerEvent.modelGraph, workerEvent.paneId);
          break;
        default:
          break;
      }
    });
  }

  private init() {
    // Set default renderer.
    this.curSelectedRenderer.set({
      id: this.panes()[0].id,
      ownerType: RendererOwner.GRAPH_PANEL,
    });

    const params = new URLSearchParams(document.location.search);
    this.testMode = params.get('test_mode') === '1';
    this.groupNodeChildrenCountThresholdFromUrl = params.get(
      'groupNodeChildrenCountThreshold',
    );

    // Load saved show on node item types.
    const pane = this.panes()[0];
    pane.showOnNodeItemTypes = {[pane.id]: this.getSavedShowOnNodeItemTypes()};
    const savedShowOnEdgeItem = this.getSavedShowOnEdgeItem();
    if (savedShowOnEdgeItem) {
      pane.showOnEdgeItems = {[pane.id]: savedShowOnEdgeItem};
    } else {
      pane.showOnEdgeItems = {};
    }
  }

  private handleGraphProcessed(modelGraph: ModelGraph, paneId: string) {
    this.panes.update((panes) => {
      for (const pane of panes) {
        if (pane.id === paneId) {
          pane.modelGraph = modelGraph;
          break;
        }
      }
      return [...panes];
    });
    this.modelGraphProcessed$.next({
      paneIndex: this.getPaneIndexById(paneId),
      modelGraph,
    });
  }

  private setPaneLoading(paneId: string) {
    this.panes.update((panes) => {
      for (const pane of panes) {
        if (pane.id === paneId) {
          pane.modelGraph = undefined;
          break;
        }
      }
      return [...panes];
    });
  }

  private getGroupNodeChildrenCountThreshold() {
    let groupNodeChildrenCountThreshold =
      DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD;
    if (this.config) {
      groupNodeChildrenCountThreshold =
        this.config()?.artificialLayerNodeCountThreshold ||
        DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD;
    }
    if (this.groupNodeChildrenCountThresholdFromUrl != null) {
      groupNodeChildrenCountThreshold = Number(
        this.groupNodeChildrenCountThresholdFromUrl,
      );
    }
    return groupNodeChildrenCountThreshold;
  }
}
