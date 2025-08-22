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

import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  effect,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatSnackBar} from '@angular/material/snack-bar';

import {AppService} from './app_service';
import {BenchmarkRunner} from './benchmark_runner';
import {Graph, GraphCollection} from './common/input_graph';
import {ModelGraph, OpNode} from './common/model_graph';
import {
  CommandType,
  ModelGraphProcessedEvent,
  NodeAttributePairs,
  NodeDataProviderData,
  NodeDataProviderGraphData,
  NodeInfo,
  SyncNavigationModeChangedEvent,
  ViewOnEdgeMode,
} from './common/types';
import {genUid, inInputElement, isOpNode} from './common/utils';
import {type VisualizerConfig} from './common/visualizer_config';
import {type VisualizerUiState} from './common/visualizer_ui_state';
import {WorkerEventType} from './common/worker_events';
import {ExtensionService} from './extension_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {NodeStylerService} from './node_styler_service';
import {SplitPanesContainer} from './split_panes_container';
import {SyncNavigationService} from './sync_navigation_service';
import {ThreejsService} from './threejs_service';
import {TitleBar} from './title_bar';
import {UiStateService} from './ui_state_service';
import {WorkerService} from './worker_service';

/** The main model graph visualizer component. */
@Component({
  standalone: true,
  selector: 'model-graph-visualizer',
  imports: [BenchmarkRunner, CommonModule, TitleBar, SplitPanesContainer],
  templateUrl: './model_graph_visualizer.ng.html',
  styleUrls: ['./model_graph_visualizer.scss'],
  providers: [
    AppService,
    ExtensionService,
    NodeDataProviderExtensionService,
    NodeStylerService,
    SyncNavigationService,
    UiStateService,
    WorkerService,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ModelGraphVisualizer implements OnInit, OnDestroy, OnChanges {
  @Input({required: true}) graphCollections!: GraphCollection[];

  /** Some visualization related configs. See `VisualizerConfig` for details. */
  @Input() config?: VisualizerConfig;

  /** The UI state to restore on init. */
  @Input() initialUiState?: VisualizerUiState;

  /** Benchmark mode. */
  @Input() benchmark = false;

  /** The sources (file paths) of node data. */
  @Input() nodeDataSources: string[] = [];

  /** Triggered when the title is clicked. */
  @Output() readonly titleClicked = new EventEmitter<void>();

  /**
   * Triggered when UI state changes.
   *
   * UI state includes the current selected node, expanded layers, split-pane
   * status, etc. Save the event data and feed it back to `initialUiState` above
   * to restore the state on init.
   */
  @Output() readonly uiStateChanged = new EventEmitter<VisualizerUiState>();

  /** Triggered when a model graph has been processed. */
  @Output()
  readonly modelGraphProcessed = new EventEmitter<ModelGraphProcessedEvent>();

  /** Triggered when a remote node data paths are updated. */
  @Output() readonly remoteNodeDataPathsChanged = new EventEmitter<string[]>();

  /** Triggered when the sync navigation mode is changed. */
  @Output() readonly syncNavigationModeChanged =
    new EventEmitter<SyncNavigationModeChangedEvent>();

  /** Triggered when the selected node is changed. */
  @Output() readonly selectedNodeChanged = new EventEmitter<NodeInfo>();

  /** Triggered when the hovered node is changed. */
  @Output() readonly hoveredNodeChanged = new EventEmitter<NodeInfo>();

  /** Triggered when the double clicked node is changed. */
  @Output() readonly doubleClickedNodeChanged = new EventEmitter<NodeInfo>();

  curProcessedModelGraph?: ModelGraph;
  ready = false;

  private readonly mouseDownHandler = (event: MouseEvent) => {
    window.parent.postMessage(
      {
        'cmd': 'model-explorer-mousedown',
      },
      '*',
    );
  };

  constructor(
    readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly destroyRef: DestroyRef,
    private readonly el: ElementRef<HTMLElement>,
    private readonly workerService: WorkerService,
    private readonly snackBar: MatSnackBar,
    private readonly threejsService: ThreejsService,
    private readonly uiStateService: UiStateService,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly nodeStylerService: NodeStylerService,
    readonly syncNavigationService: SyncNavigationService,
  ) {

    effect(() => {
      const curUiState = this.uiStateService.curUiState();
      if (!curUiState) {
        return;
      }
      this.uiStateChanged.emit(curUiState);
    });

    effect(() => {
      this.remoteNodeDataPathsChanged.emit(
        this.appService.remoteNodeDataPaths(),
      );
    });

    effect(() => {
      this.selectedNodeChanged.emit(this.appService.selectedNode());
    });

    effect(() => {
      this.hoveredNodeChanged.emit(this.appService.hoveredNode());
    });

    effect(() => {
      this.doubleClickedNodeChanged.emit(this.appService.doubleClickedNode());
    });

    // Listen to postMessage.
    window.addEventListener('message', (e) => {
      const data = e.data;
      switch (data['cmd']) {
        case 'model-explorer-load-node-data-file':
          const path = data['path'];
          if (path) {
            this.handleGetNodeDataPathFromPostMessage(path);
          }
          break;
        case 'model-explorer-select-node-by-output-tensor-name':
          const tensorName = data['tensorName'];
          if (tensorName) {
            this.handleSelectNodeByOutputTensorNameFromPostMessage(tensorName);
          }
          break;
        case 'model-explorer-select-node-by-node-id':
          const nodeId = data['nodeId'];
          if (nodeId) {
            this.handleSelectNodeByNodeIdFromPostMessage(nodeId);
          }
          break;
        default:
          break;
      }
    });

    this.appService.modelGraphProcessed$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((event) => {
        this.modelGraphProcessed.next(event);
      });

    this.syncNavigationService.syncNavigationModeChanged$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((event) => {
        this.syncNavigationModeChanged.next(event);
      });

    this.initThreejs();
  }

  ngOnInit() {
    this.appService.config.set(this.config || {});
    this.appService.addGraphCollections(this.graphCollections);
    this.appService.curInitialUiState.set(this.initialUiState);
    if (this.config?.nodeStylerRules) {
      this.nodeStylerService.rules.set(this.config.nodeStylerRules);
    }

    // No initial ui state. Use the graph with the most node counts as the
    // default selected graph.
    if (!this.initialUiState || this.initialUiState.paneStates.length === 0) {
      if (
        this.graphCollections.length > 0 &&
        this.graphCollections[0].graphs.length > 0
      ) {
        // Sort graphs in graphCollections[0] by their nodes count in descending
        // order.
        const sortedGraphs = [...this.graphCollections[0].graphs].sort(
          (a, b) => b.nodes.length - a.nodes.length,
        );
        // Select the graph with the most node counts.
        const selectedGraph = sortedGraphs[0];
        this.appService.selectGraphInCurrentPane(selectedGraph);
      }
    }
    // Initial ui state exists.
    else {
      // One pane.
      if (this.initialUiState.paneStates.length === 1) {
        const paneState = this.initialUiState.paneStates[0];
        const initialLayout =
          paneState.selectedNodeId === '' &&
          paneState.deepestExpandedGroupNodeIds.length === 0;
        const selectedGraph = this.findGraphFromCollections(
          paneState.selectedCollectionLabel,
          paneState.selectedGraphId,
        );
        const flattenLayers = paneState.flattenLayers === true;
        if (selectedGraph) {
          this.appService.selectGraphInCurrentPane(
            selectedGraph,
            flattenLayers,
            undefined,
            initialLayout,
          );
        } else {
          // Fall back to first graph.
          const firstGraph = this.graphCollections[0].graphs[0];
          this.appService.selectGraphInCurrentPane(
            firstGraph,
            flattenLayers,
            undefined,
            initialLayout,
          );
        }
        this.appService.setFlattenLayersInCurrentPane(flattenLayers);
      }
      // Two panes.
      else if (this.initialUiState.paneStates.length === 2) {
        // Load graph in pane0.
        const pane0 = this.initialUiState.paneStates[0];
        const selectedGraph0 = this.findGraphFromCollections(
          pane0.selectedCollectionLabel,
          pane0.selectedGraphId,
        );
        const flattenLayers0 = pane0.flattenLayers === true;
        if (selectedGraph0) {
          this.appService.selectGraphInCurrentPane(
            selectedGraph0,
            flattenLayers0,
          );
        } else {
          // Fall back to first graph.
          const firstGraph = this.graphCollections[0].graphs[0];
          this.appService.selectGraphInCurrentPane(firstGraph, flattenLayers0);
        }
        this.appService.setFlattenLayersInCurrentPane(flattenLayers0);

        // Add graph in pane1.
        const pane1 = this.initialUiState.paneStates[1];
        const flattenLayers1 = pane1.flattenLayers === true;
        const selectedGraph1 = this.findGraphFromCollections(
          pane1.selectedCollectionLabel,
          pane1.selectedGraphId,
        );
        if (selectedGraph1) {
          this.appService.openGraphInSplitPane(selectedGraph1, flattenLayers1);
        } else {
          // Fall back to first graph.
          const firstGraph = this.graphCollections[0].graphs[0];
          this.appService.openGraphInSplitPane(firstGraph, flattenLayers1);
        }

        // Select pane.
        if (pane0.selected) {
          this.appService.selectPaneByIndex(0);
        } else if (pane1.selected) {
          this.appService.selectPaneByIndex(1);
        }

        // Pane width.
        this.appService.setPaneWidthFraction(pane0.widthFraction);
      }
    }

    this.el.nativeElement.addEventListener(
      'mousedown',
      this.mouseDownHandler,
      true,
    );
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['graphCollections']) {
      if (!changes['graphCollections'].isFirstChange()) {
        this.appService.reset();
        this.uiStateService.reset();
        this.cleanUp();
        this.ngOnInit();
      }
    }
  }

  ngOnDestroy() {
    this.cleanUp();
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyboardEvent(event: KeyboardEvent) {
    // Press "SPACE" to zoom-to-fit the currently selected renderer.
    if (event.key === ' ') {
      if (!inInputElement()) {
        this.appService.spaceKeyToZoomFitClicked.next({});
      }
    }
    // Press ctrl/cmd+f for search.
    else if (event.key === 'f' && (event.ctrlKey || event.metaKey)) {
      if (!this.config?.hideTitleBar) {
        event.preventDefault();
      }
      this.appService.searchKeyClicked.next({});
    }
  }

  handleDragOver(event: DragEvent) {
    // This needs to be here to disable unnecessary drop animation.
    event.preventDefault();
  }

  async handleDrop(event: DragEvent) {
    if (!event.dataTransfer) {
      return;
    }
    event.stopPropagation();
    event.preventDefault();

    // Collect the dropped node data files.
    const files: File[] = [];
    if (event.dataTransfer?.items) {
      // Use DataTransferItemList interface to access the file(s)
      Array.from(event.dataTransfer.items).forEach((item, i) => {
        // If dropped items aren't files, reject them
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) {
            files.push(file);
          }
        }
      });
    } else {
      // Use DataTransfer interface to access the file(s)
      files.push(...Array.from(event.dataTransfer?.files || []));
    }

    // Handle the dropped node data json files.
    let hasValidFiles = false;
    if (files.length !== 0) {
      for (const file of files) {
        if (!file.name.endsWith('.json')) {
          continue;
        }

        // Read json file.
        const fileReader = new FileReader();
        // tslint:disable-next-line:no-any Allow arbitrary types.
        const jsonObj = await new Promise<any>((resolve) => {
          fileReader.onload = (event) => {
            const jsonObj = JSON.parse(
              event.target?.result as string,
              // tslint:disable-next-line:no-any Allow arbitrary types.
            ) as any;
            resolve(jsonObj);
          };
          fileReader.readAsText(file);
        });

        // Add to graph
        //
        // Node data for a single graph.
        if (
          jsonObj['results'] != null &&
          jsonObj['results']['results'] == null
        ) {
          this.addNodeDataProviderData(file.name, jsonObj);
          hasValidFiles = true;
        }
        // Node data for a model.
        else if (
          Object.values(jsonObj).some(
            // tslint:disable-next-line:no-any Allow arbitrary types.
            (value: any) => value['results'] != null,
          )
        ) {
          this.addNodeDataProviderDataWithGraphIndex(file.name, jsonObj);
          hasValidFiles = true;
        }
      }
    }

    if (!hasValidFiles) {
      this.snackBar.open('File(s) not supported', 'Dismiss');
    }
  }

  /**
   * Select the given node with all its parent layers expanded.
   *
   * @param nodeId the id of the node to search for.
   * @param graphId the id of the graph to search for the given node.
   * @param collectionLabel (optional) the label of the collection to search for
   *     the given node. If unset, we will go through all collections in
   *     `graphCollections`.
   * @param paneIndex the index of the pane (0 or 1) to select the node in.
   */
  selectNode(
    nodeId: string,
    graphId: string,
    collectionLabel?: string,
    paneIndex = 0,
  ) {
    // Find the collection.
    let collectionsToSearch: GraphCollection[] = this.graphCollections;
    if (collectionLabel) {
      const collection = this.appService
        .curGraphCollections()
        .find(
          (collection) =>
            collection.label.toLowerCase() === collectionLabel.toLowerCase(),
        );
      if (!collection) {
        console.warn(
          `Failed to locate collection with label "${collectionLabel}"`,
        );
        return;
      }
      collectionsToSearch = [collection];
    }

    // Find the graph.
    let targetGraph: Graph | undefined = undefined;
    for (const collection of collectionsToSearch) {
      const graph = collection.graphs.find((graph) => graph.id === graphId);
      if (graph) {
        targetGraph = graph;
        break;
      }
    }
    if (!targetGraph) {
      console.warn(`Failed to locate graph with id "${graphId}"`);
      return;
    }

    // Reveal node.
    this.appService.selectGraphInPane(targetGraph, paneIndex);
    const paneId = this.appService.panes()[paneIndex].id;
    this.appService.curInitialUiState.set(undefined);
    this.appService.selectNode(paneId, undefined);
    this.appService.curToLocateNodeInfo.set(undefined);
    this.appService.setNodeToReveal(paneId, nodeId);
  }

  /**
   * Adds data for node data provider.
   *
   * @param name the name of the data to add.
   * @param data the data to add.
   * @param paneIndex the index of the pane to add data for.
   * @param clearExisting whether to clear existing data before adding new one.
   */
  addNodeDataProviderData(
    name: string,
    data: NodeDataProviderGraphData,
    paneIndex = 0,
    clearExisting = false,
  ) {
    const modelGraph = this.appService.getModelGraphFromPaneIndex(paneIndex);
    if (!modelGraph) {
      console.warn(`Model graph in pane with index ${paneIndex} doesn't exist`);
      return;
    }
    this.nodeDataProviderExtensionService.addRun(
      genUid(),
      name,
      '',
      modelGraph,
      {[modelGraph.id]: data},
      clearExisting,
    );
  }

  /**
   * Adds data with graph index for node data provider.
   *
   * @param name the name of the data to add.
   * @param data the data to add.
   * @param paneIndex the index of the pane to add data for.
   * @param clearExisting whether to clear existing data before adding new one.
   */
  addNodeDataProviderDataWithGraphIndex(
    name: string,
    data: NodeDataProviderData,
    paneIndex = 0,
    clearExisting = false,
  ) {
    const modelGraph = this.appService.getModelGraphFromPaneIndex(paneIndex);
    if (!modelGraph) {
      console.warn(`Model graph in pane with index ${paneIndex} doesn't exist`);
      return;
    }
    this.nodeDataProviderExtensionService.addRun(
      genUid(),
      name,
      '',
      modelGraph,
      data,
      clearExisting,
    );
  }

  /**
   * Adds attributes for the given node in the graph from the given pane.
   *
   * Note that this method only works when called after the graph has been
   * processed. To ensure this, listen to the `modelGraphProcessed` event and
   * call this method after the event is emitted. The event payload includes
   * the processed graph (from which you can get the graph ID) and the index of
   * the pane where the graph is located. You can use the graph ID to identify
   * the appropriate node attributes to add within your application, and use
   * the pane index to call this method.
   *
   * @param nodeId the id of the node to add attributes for.
   * @param attrs the attributes to add.
   * @param paneIndex the index of the pane to add attributes for. The system
   *   will try to find the node in the processed graph of this pane.
   */
  addNodeAttributes(nodeId: string, attrs: NodeAttributePairs, paneIndex = 0) {
    const modelGraph = this.appService.getModelGraphFromPaneIndex(paneIndex);
    if (!modelGraph) {
      console.warn(`Model graph in pane with index ${paneIndex} doesn't exist`);
      return;
    }
    const node = modelGraph.nodesById[nodeId];
    if (!node) {
      console.warn(`Node with id "${nodeId}" not found`);
      return;
    }
    if (isOpNode(node)) {
      node.attrs = {...node.attrs, ...attrs};
    }

    // Update the model graph cache in the worker so that it includes the
    // updated node attributes.
    this.workerService.worker.postMessage({
      eventType: WorkerEventType.UPDATE_MODEL_GRAPH_CACHE_WITH_NODE_ATTRIBUTES,
      modelGraphId: modelGraph.id,
      nodeId,
      attrs,
      paneId: this.appService.panes()[paneIndex].id,
    });
  }

  /**
   * Expands all layers for the graph in the given pane.
   *
   * @param paneIndex the index of the pane to expand all layers in.
   */
  expandAllLayers(paneIndex = 0) {
    this.appService.expandOrCollapseAllGraphLayersClicked.next({
      expandOrCollapse: true,
      rendererId: this.appService.getPaneIdByIndex(paneIndex),
    });
  }

  /**
   * Collapses all layers for the graph in the given pane.
   *
   * @param paneIndex the index of the pane to collapse all layers in.
   */
  collapseAllLayers(paneIndex = 0) {
    this.appService.expandOrCollapseAllGraphLayersClicked.next({
      expandOrCollapse: false,
      rendererId: this.appService.getPaneIdByIndex(paneIndex),
    });
  }

  /**
   * Collapses the info panel in the given pane.
   *
   * @param paneIndex the index of the pane to collapse the info panel in.
   */
  collapseInfoPanel(paneIndex = 0) {
    this.appService.command.next({
      type: CommandType.COLLAPSE_INFO_PANEL,
      paneIndex,
    });
  }

  /**
   * Shows the info panel in the given pane.
   *
   * @param paneIndex the index of the pane to show the info panel in.
   */
  showInfoPanel(paneIndex = 0) {
    this.appService.command.next({
      type: CommandType.SHOW_INFO_PANEL,
      paneIndex,
    });
  }

  /**
   * Sets the view on edge mode in the given pane.
   *
   * @param mode the view on edge mode to set.
   * @param paneIndex the index of the pane to set the view on edge mode in.
   * @param value the value to set if the mode needs a value to work (e.g. the
   *   metadata key in ViewOnEdgeMode.OUTPUT_METADATA).
   */
  setViewOnEdge(mode: ViewOnEdgeMode, paneIndex = 0, value = '') {
    this.appService.command.next({
      type: CommandType.SET_VIEW_ON_EDGE,
      mode,
      value,
      paneIndex,
    });
  }

  async loadRemoteNodeDataPaths(paths: string[], modelGraph: ModelGraph) {
    await Promise.all(
      paths.map((path) =>
        this.nodeDataProviderExtensionService.addRunFromRemoteSource(
          path,
          modelGraph,
        ),
      ),
    );
  }

  get hasNoGraphs(): boolean {
    // Calculate number of graphs in graphCollections and return
    // true if the count is 0.
    return (
      this.graphCollections.reduce(
        (acc, collection) => acc + collection.graphs.length,
        0,
      ) === 0
    );
  }

  get showTitleBar(): boolean {
    return !this.config?.hideTitleBar;
  }

  private findGraphFromCollections(
    collectionLabel: string,
    graphId: string,
  ): Graph | undefined {
    for (const collection of this.graphCollections) {
      for (const graph of collection.graphs) {
        if (
          graph.id === graphId &&
          // The old url has the collectionLabel set to ''.
          (collectionLabel === '' || graph.collectionLabel === collectionLabel)
        ) {
          return graph;
        }
      }
    }
    return undefined;
  }

  private handleGetNodeDataPathFromPostMessage(path: string) {
    const modelGraph = this.appService.getModelGraphFromPaneIndex(0);
    if (!modelGraph) {
      console.warn(`Model graph in pane with index 0 doesn't exist`);
      return;
    }
    this.loadRemoteNodeDataPaths([path], modelGraph);
  }

  private handleSelectNodeByOutputTensorNameFromPostMessage(
    tensorName: string,
  ) {
    const modelGraph = this.appService.getModelGraphFromSelectedPane();
    if (!modelGraph) {
      return;
    }

    let foundNode: OpNode | undefined = undefined;
    for (const node of modelGraph.nodes) {
      if (isOpNode(node)) {
        const outputMetadata = node.outputsMetadata || {};
        for (const outputId of Object.keys(outputMetadata)) {
          const curMetadata = outputMetadata[outputId];
          if (tensorName === curMetadata['tensor_name']) {
            foundNode = node;
            break;
          }
        }
        if (foundNode) {
          break;
        }
      }
    }
    if (foundNode) {
      this.selectNode(foundNode.id, modelGraph.id);
    }
  }

  private handleSelectNodeByNodeIdFromPostMessage(nodeId: string) {
    const modelGraph = this.appService.getModelGraphFromSelectedPane();
    if (!modelGraph) {
      return;
    }

    const node = modelGraph.nodesById[nodeId];
    if (!node) {
      return;
    }

    this.selectNode(node.id, modelGraph.id);
  }

  private async initThreejs() {
    await this.threejsService.depsLoadedPromise;

    this.ready = true;
    this.changeDetectorRef.markForCheck();
  }

  private cleanUp() {
    this.el.nativeElement.removeEventListener(
      'mousedown',
      this.mouseDownHandler,
      true,
    );
  }
}
