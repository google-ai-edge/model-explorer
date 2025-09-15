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

import {Overlay, OverlayConfig, OverlayRef} from '@angular/cdk/overlay';
import {ComponentPortal} from '@angular/cdk/portal';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  DestroyRef,
  effect,
  ElementRef,
  EventEmitter,
  inject,
  Input,
  NgZone,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  signal,
  Signal,
  SimpleChanges,
  untracked,
  ViewChild,
  ViewContainerRef,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule, MatMenuTrigger} from '@angular/material/menu';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltip, MatTooltipModule} from '@angular/material/tooltip';
import {setAnchorHref} from 'safevalues/dom';
import * as three from 'three';

import {AppService} from './app_service';
import {
  GLOBAL_KEY,
  LAYOUT_MARGIN_X,
  NODE_LABEL_HEIGHT,
  NODE_LABEL_LINE_HEIGHT,
  WEBGL_ELEMENT_Y_FACTOR,
} from './common/consts';
import {Graph, GraphNode} from './common/input_graph';
import {
  GroupNode,
  ModelEdge,
  type ModelGraph,
  ModelNode,
  NodeType,
  OpNode,
} from './common/model_graph';
import {SyncNavigationData, SyncNavigationMode} from './common/sync_navigation';
import {
  FontWeight,
  NodeDataProviderResultProcessedData,
  NodeDataProviderRunData,
  NodeStyleId,
  NodeStylerRule,
  Point,
  PopupPanelData,
  ProcessedNodeStylerRule,
  Rect,
  RendererInfo,
  SelectedNodeInfo,
  ShowOnEdgeItemData,
  ShowOnNodeItemData,
  WebglColor,
} from './common/types';
import {
  genUid,
  getDeepestExpandedGroupNodeIds,
  getHighQualityPixelRatio,
  getNodeStyleValue,
  getShowOnEdgeInputOutputMetadataKeys,
  hasNonEmptyQueries,
  IS_MAC,
  isGroupNode,
  isOpNode,
  matchNodeForQueries,
  processNodeStylerRules,
  splitLabel,
} from './common/utils';
import {
  ExpandOrCollapseGroupNodeRequest,
  LocateNodeRequest,
  PreparePopupRequest,
  RelayoutGraphRequest,
  WorkerEvent,
  WorkerEventType,
} from './common/worker_events';
import {DragArea} from './drag_area';
import {genIoTreeData, IoTree} from './io_tree';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {NodeStylerService} from './node_styler_service';
import {SplitPaneService} from './split_pane_service';
import {SubgraphSelectionService} from './subgraph_selection_service';
import {SyncNavigationService} from './sync_navigation_service';
import {ThreejsService} from './threejs_service';
import {UiStateService} from './ui_state_service';
import {WebglEdges} from './webgl_edges';
import {WebglRendererAttrsTableService} from './webgl_renderer_attrs_table_service';
import {WebglRendererEdgeOverlaysService} from './webgl_renderer_edge_overlays_service';
import {WebglRendererEdgeTextsService} from './webgl_renderer_edge_texts_service';
import {
  DEFAULT_DELETE_NODES_BORDER_COLOR,
  DEFAULT_HIGHLIGHT_NODES_BORDER_COLOR,
  DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH,
  DEFAULT_NEW_NODES_BORDER_COLOR,
  HighlightInfo,
  WebglRendererHighlightNodesService,
} from './webgl_renderer_highlight_node_service';
import {WebglRendererIdenticalLayerService} from './webgl_renderer_identical_layer_service';
import {
  IO_PICKER_ID_SEP,
  WebglRendererIoHighlightService,
} from './webgl_renderer_io_highlight_service';
import {
  IoTracingData,
  WebglRendererIoTracingService,
} from './webgl_renderer_io_tracing_service';
import {WebglRendererNdpService} from './webgl_renderer_ndp_service';
import {WebglRendererSearchResultsService} from './webgl_renderer_search_results_service';
import {WebglRendererSnapshotService} from './webgl_renderer_snapshot_service';
import {WebglRendererSubgraphSelectionService} from './webgl_renderer_subgraph_selection_service';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';
import {WorkerService} from './worker_service';

const NODE_BORDER_WIDTH = 1.2;
const SELECTED_NODE_BORDER_WIDTH = 2;
const IO_HIGHLIGHT_BORDER_WIDTH = 1.5;
const NODE_ANIMATION_DURATION = 200;
const ZOOM_FIT_ON_NODE_DURATION = 400;
const EDGE_WIDTH = 1.0;
const SUBGRAPH_INDICATOR_SIZE = 14;
const MAX_PNG_SIZE = 5000;

// The following offsets define the rendering order of the elements on top of
// the node body. They should be between 0 and WEBGL_ELEMENT_Y_FACTOR.
const ARTIFICIAL_GROUP_NODE_BORDER_Y_OFFSET = -WEBGL_ELEMENT_Y_FACTOR * 0.5;
const NODE_LABEL_Y_OFFSET = WEBGL_ELEMENT_Y_FACTOR * 0.4;
const GROUP_NODE_ICON_BG_OFFSET = WEBGL_ELEMENT_Y_FACTOR * 0.3;
const SUBGRAPH_INDICATOR_LABEL_Y_OFFSET = WEBGL_ELEMENT_Y_FACTOR * 0.4;

const NODE_ID_WITHOUT_ZOOMFIT = '______';

const THREE = three;

interface TriggerData {
  top: number;
  left: number;
  width: number;
  height: number;
  tooltip?: string;
}

/** The type of the element to render. */
enum RenderElementType {
  NODE,
  EDGE,
}

/** A node element to render. */
interface RenderElementNode {
  type: RenderElementType.NODE;
  id: string;
  node: ModelNode;
}

/** An edge element to render. */
interface RenderElementEdge {
  type: RenderElementType.EDGE;
  id: string;
  edge: ModelEdge;
}

/** Options for rendering the graph. */
interface RenderGraphOptions {
  skipReRenderEdges?: boolean;
  skipReRenderEdgeTexts?: boolean;
}

/** Union type of node and edge element to render. */
type RenderElement = RenderElementNode | RenderElementEdge;

/** A graph renderer that uses threejs/webgl for high-performance rendering */
@Component({
  standalone: true,
  selector: 'webgl-renderer',
  imports: [
    CommonModule,
    DragArea,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
  ],
  providers: [
    WebglRendererAttrsTableService,
    WebglRendererEdgeTextsService,
    WebglRendererEdgeOverlaysService,
    WebglRendererIdenticalLayerService,
    WebglRendererIoHighlightService,
    WebglRendererIoTracingService,
    WebglRendererNdpService,
    WebglRendererSearchResultsService,
    WebglRendererSnapshotService,
    WebglRendererSubgraphSelectionService,
    WebglRendererThreejsService,
  ],
  templateUrl: './webgl_renderer.ng.html',
  styleUrls: ['./webgl_renderer.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WebglRenderer implements OnInit, OnChanges, OnDestroy {
  /**
   * This is the model graph that has been processed from an input graph. The
   * renderer will update this graph in various situations (e.g. when user
   * expands a group node). We use the `curModelGraph` field below to store
   * the latest model graph after each update.
   */
  @Input({required: true}) modelGraph!: ModelGraph;
  @Input({required: true}) rendererId!: string;
  @Input({required: true}) paneId!: string;
  /** The id of the root node to render from. Undefined means all nodes. */
  @Input() rootNodeId?: string;
  /** Whether the renderer is in a popup. */
  @Input() inPopup = false;
  /** Whether to use the model graph to run benchmark or not. */
  @Input() benchmark = false;

  /** Triggered when the "open in popup" button is clickded. */
  @Output() readonly openInPopupClicked = new EventEmitter<PopupPanelData>();

  @ViewChild('container', {static: true}) container!: ElementRef<HTMLElement>;
  @ViewChild('canvas', {static: true}) canvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('snapshotCanvas', {static: true})
  snapshotCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('pngDownloaderCanvas', {static: true})
  pngDownloaderCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('hoverToolbar', {static: true})
  hoverToolbar!: ElementRef<HTMLCanvasElement>;
  @ViewChild('ioPicker', {static: true})
  ioPicker!: ElementRef<HTMLElement>;
  @ViewChild('subgraphMenuTrigger', {static: true})
  subgraphMenuTrigger!: MatMenuTrigger;
  @ViewChild('groupNodeIconMatTooltip', {static: false})
  groupNodeIconMatTooltip!: MatTooltip;
  @ViewChild('ioPickerMatTooltip', {static: false})
  ioPickerMatTooltip!: MatTooltip;
  @ViewChild('moreActionsMenuTrigger', {static: true})
  moreActionsMenuTrigger!: MatMenuTrigger;
  @ViewChild('rangeZoomDragArea', {static: true})
  rangeZoomDragArea!: DragArea;
  @ViewChild('dragToSelectDragArea', {static: true})
  dragToSelectDragArea!: DragArea;

  readonly appService: AppService = inject(AppService);
  private readonly threejsService: ThreejsService = inject(ThreejsService);

  readonly SELECTED_NODE_BORDER_COLOR = new THREE.Color('#1A73E8');
  readonly SELECTED_NODE_BG_COLOR = new THREE.Color('#C2E7FF');
  readonly HOVERED_NODE_BORDER_COLOR = new THREE.Color('#000');
  readonly HOVERED_GROUP_NODE_BORDER_COLOR = new THREE.Color('#666');
  readonly IDENTICAL_GROUPS_BG_COLOR = new THREE.Color('#e2edff');
  readonly NODE_LABEL_COLOR = new THREE.Color('#041E49');
  readonly OP_NODE_BORDER_COLOR = new THREE.Color('#777');
  readonly GROUP_NODE_BORDER_COLOR = new THREE.Color('#aaa');
  readonly GROUP_NODE_LABEL_SEPARATOR_COLOR = new THREE.Color('#DADCE0');
  readonly GROUP_NODE_ICON_COLOR = new THREE.Color('#444746');
  readonly GROUP_NODE_PIN_TO_TOP_SEPARATOR_COLOR = new THREE.Color('#bbb');
  readonly EDGE_COLOR = new THREE.Color(
    this.appService.config()?.edgeColor || '#aaa',
  );
  readonly EDGE_COLOR_INCOMING = new THREE.Color('#009e73');
  readonly EDGE_TEXT_COLOR_INCOMING = new THREE.Color('#125341');
  readonly EDGE_COLOR_OUTGOING = new THREE.Color('#d55e00');
  readonly EDGE_TEXT_COLOR_OUTGOING = new THREE.Color('#994d11');
  readonly ARTIFCIAL_GROUPS_BORDER_COLOR = new THREE.Color('#800080');
  readonly SUBGRAPH_INDICATOR_BORDER_COLOR = new THREE.Color('#135cbb');
  readonly SUBGRAPH_INDICATOR_BG_COLOR = new THREE.Color('#d5e7ff');
  readonly GROUP_NODE_BG_COLORS: three.Color[] = (() => {
    const startLightness = 96;
    const endLightness = 84;
    const count = 6;
    const factor = (endLightness - startLightness) / (count - 1);
    const colors: three.Color[] = [];
    for (let i = 0; i < count; i++) {
      const curLightness = startLightness + i * factor;
      colors.push(
        new THREE.Color(`hsl(212, 40%, ${Math.round(curLightness)}%)`),
      );
    }
    return colors;
  })();

  graphId = '';
  curModelGraph!: ModelGraph;
  tracing = false;
  showBusySpinner = false;
  selectedNodeId = '';
  flashing = false;
  hoveredNodeIdWhenClickingMoreActions = '';

  // Ranges along x and z axis for the current model graph.
  currentMinX = 0;
  currentMaxX = 0;
  currentMinZ = 0;
  currentMaxZ = 0;

  groupNodeIcon: TriggerData = {
    top: -1000,
    left: -1000,
    width: 0,
    height: 0,
  };

  ioPickerTop = -1000;
  ioPickerLeft = -1000;
  ioPickerWidth = 0;
  ioPickerHeight = 0;
  ioPickerTooltip = '';

  subgraphIndicatorTop = -1000;
  subgraphIndicatorLeft = -1000;
  subgraphIndicatorWidth = 0;
  subgraphIndicatorHeight = 0;
  curSubgraphIdsForMenu: string[] = [];

  curShowOnNodeItemTypes: Record<string, ShowOnNodeItemData> = {};
  curShowOnEdgeItem?: ShowOnEdgeItemData;
  nodesToRender: Array<{node: ModelNode; index: number}> = [];
  nodesToRenderMap: Record<string, {node: ModelNode; index: number}> = {};
  edgesToRender: Array<{edge: ModelEdge; index: number}> = [];
  curNodeDataProviderRuns: Record<string, NodeDataProviderRunData> = {};
  curHiddenInputOpNodeIds: Record<string, boolean> = {};
  curHiddenOutputIds: Record<string, boolean> = {};

  private elementsToRender: RenderElement[] = [];
  private updateNodesStylesSavedSelectedNodeId = '';
  private updateNodesStylesSavedIoTracingData?: IoTracingData;
  private curSelectedRenderer?: RendererInfo;
  private portal: ComponentPortal<IoTree> | null = null;
  private showBusySpinnerTimeoutRef = -1;
  private prevNodeDataProviderData:
    | Record<string, NodeDataProviderResultProcessedData>
    | undefined = undefined;
  private prevNodeDataProviderRun: NodeDataProviderRunData | undefined =
    undefined;
  private readonly nodeBodies = new WebglRoundedRectangles(6);
  private readonly groupNodeIcons = new WebglTexts(this.threejsService);
  private readonly groupNodeIconBgs = new WebglRoundedRectangles(99);
  private readonly artificialGroupBorders = new WebglRoundedRectangles(6);
  private readonly subgraphIndicatorBgs = new WebglRoundedRectangles(3);
  private readonly subgraphIndicatorIcons = new WebglTexts(this.threejsService);
  private readonly edges = new WebglEdges(this.EDGE_COLOR, EDGE_WIDTH);
  readonly texts = new WebglTexts(this.threejsService);
  private readonly mousePos = new THREE.Vector2();
  private readonly syncNavigationRelatedNodesHighlights!: WebglRendererHighlightNodesService;
  private readonly syncNavigationDiffHighlights!: WebglRendererHighlightNodesService;
  private draggingArea = false;
  private hoveredNodeId = '';
  private hoveredGroupNodeIconId = '';
  private nodeIdForHoveredGroupNodeIcon = '';
  private hoveredIoPickerId = '';
  private hoveredSubgraphIndicatorId = '';
  private savedUpdateNodeBgWhenFarProgress = -1;
  private curNodeStylerRules: NodeStylerRule[] = [];
  private curProcessedNodeStylerRules: ProcessedNodeStylerRule[] = [];
  private renderedEdgeIdsToHide: string[] = [];
  private relayoutDoneFn?: () => void;
  private readonly paneIdInternal = signal<string>('');
  private readonly paneIndex = computed(() =>
    this.appService.getPaneIndexById(this.paneIdInternal()),
  );
  private readonly paneGraphTitlesKey: Signal<string> = computed(() => {
    const panes = this.appService.panes();
    return panes
      .map((pane, index) => `${index}:${pane.modelGraph?.id ?? ''}`)
      .join(',');
  });
  private readonly paneCount = computed(() => this.appService.panes().length);
  private savedSyncNavigationMode: SyncNavigationMode | undefined = undefined;
  private savedSyncNavigationData: SyncNavigationData | undefined = undefined;
  private savedShowDiffHighlightsInMatchNodeIdMode: boolean | undefined =
    undefined;

  private readonly selectedNodeInfo = computed(() => {
    const pane = this.appService.getPaneById(this.paneId);
    if (!pane) {
      return;
    }
    return pane.selectedNodeInfo;
  });

  private readonly messageEventListener = (
    event: MessageEvent<WorkerEvent>,
  ) => {
    this.hideBusySpinner();
    const workerEvent = event.data;
    switch (workerEvent.eventType) {
      case WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_RESP:
        if (this.rendererId === workerEvent.rendererId) {
          this.handleExpandOrCollapseGroupNodeDone(
            workerEvent.modelGraph,
            workerEvent.rendererId,
            workerEvent.groupNodeId,
            workerEvent.expanded,
            workerEvent.deepestExpandedGroupNodeIds,
          );
        }
        break;
      case WorkerEventType.RELAYOUT_GRAPH_RESP:
        if (this.rendererId === workerEvent.rendererId) {
          this.handleReLayoutGraphDone(
            workerEvent.rendererId,
            workerEvent.modelGraph,
            workerEvent.selectedNodeId,
            workerEvent.forRestoringUiState,
            workerEvent.rectToZoomFit,
            workerEvent.forRestoringSnapshotAfterTogglingFlattenLayers,
            workerEvent.targetDeepestGroupNodeIdsToExpand,
            workerEvent.triggerNavigationSync,
          );
        }
        break;
      case WorkerEventType.LOCATE_NODE_RESP:
        if (this.rendererId === workerEvent.rendererId) {
          this.handleLocateNodeDone(
            workerEvent.rendererId,
            workerEvent.modelGraph,
            workerEvent.nodeId,
            workerEvent.deepestExpandedGroupNodeIds,
            workerEvent.noNodeShake === true,
            workerEvent.select === true,
          );
        }
        break;
      case WorkerEventType.PREPARE_POPUP_RESP:
        if (this.paneId === workerEvent.paneId) {
          this.openInPopupClicked.emit({
            id: workerEvent.rendererId,
            groupNode: workerEvent.modelGraph.nodesById[
              workerEvent.groupNodeId
            ] as GroupNode,
            initialPosition: workerEvent.initialPosition,
            curModelGraph: workerEvent.modelGraph,
          });
        }
        break;
      default:
        break;
    }
  };

  constructor(
    readonly changeDetectorRef: ChangeDetectorRef,
    private readonly destroyRef: DestroyRef,
    private readonly ngZone: NgZone,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly nodeStylerService: NodeStylerService,
    private readonly overlay: Overlay,
    private readonly snackBar: MatSnackBar,
    private readonly splitPaneService: SplitPaneService,
    private readonly subgraphSelectionService: SubgraphSelectionService,
    readonly syncNavigationService: SyncNavigationService,
    private readonly uiStateService: UiStateService,
    private readonly viewContainerRef: ViewContainerRef,
    private readonly webglRendererAttrsTableService: WebglRendererAttrsTableService,
    readonly webglRendererEdgeTextsService: WebglRendererEdgeTextsService,
    private readonly webglRendererEdgeOverlaysService: WebglRendererEdgeOverlaysService,
    private readonly webglRendererIdenticalLayerService: WebglRendererIdenticalLayerService,
    readonly webglRendererIoHighlightService: WebglRendererIoHighlightService,
    private readonly webglRendererIoTracingService: WebglRendererIoTracingService,
    private readonly webglRendererNdpService: WebglRendererNdpService,
    private readonly webglRendererSearchResultsService: WebglRendererSearchResultsService,
    private readonly webglRendererSnapshotService: WebglRendererSnapshotService,
    private readonly webglRendererSubgraphSelectionService: WebglRendererSubgraphSelectionService,
    readonly webglRendererThreejsService: WebglRendererThreejsService,
    private readonly workerService: WorkerService,
  ) {
    this.webglRendererAttrsTableService.init(this);
    this.webglRendererEdgeTextsService.init(this);
    this.webglRendererEdgeOverlaysService.init(this);
    this.webglRendererIdenticalLayerService.init(this);
    this.webglRendererIoHighlightService.init(this);
    this.webglRendererIoTracingService.init(this);
    this.webglRendererNdpService.init(this);
    this.webglRendererSearchResultsService.init(this);
    this.webglRendererSnapshotService.init(this);
    this.webglRendererSubgraphSelectionService.init(this);
    this.webglRendererThreejsService.init(this);
    this.syncNavigationRelatedNodesHighlights =
      new WebglRendererHighlightNodesService(
        this,
        -WEBGL_ELEMENT_Y_FACTOR * 0.3,
      );
    this.syncNavigationDiffHighlights = new WebglRendererHighlightNodesService(
      this,
      -WEBGL_ELEMENT_Y_FACTOR * 0.35,
    );

    this.workerService.worker.addEventListener(
      'message',
      this.messageEventListener,
    );

    effect(() => {
      this.curSelectedRenderer = this.appService.curSelectedRenderer();
    });

    // Handle zoom to fit shortcut (space key)
    this.appService.spaceKeyToZoomFitClicked
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((unused) => {
        if (this.rendererId === this.curSelectedRenderer?.id) {
          this.webglRendererThreejsService.zoomFitGraph();
        }
      });

    // Handle changes for node to locate.
    effect(() => {
      const nodeInfoToLocate = this.appService.curToLocateNodeInfo();
      if (nodeInfoToLocate?.rendererId !== this.rendererId) {
        return;
      }

      if (nodeInfoToLocate) {
        this.sendLocateNodeRequest(
          nodeInfoToLocate.nodeId,
          nodeInfoToLocate.rendererId,
          nodeInfoToLocate.noNodeShake,
          nodeInfoToLocate.select,
        );
      }
      this.appService.curToLocateNodeInfo.set(undefined);
    });

    // Handle changes for node to reveal
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane || !pane.modelGraph) {
        return;
      }

      const nodeIdToReveal = pane.nodeIdToReveal;
      if (!nodeIdToReveal) {
        return;
      }
      const success = this.revealNode(nodeIdToReveal);
      if (success) {
        this.appService.setNodeToReveal(this.paneId, undefined);
      }
    });

    effect(() => {
      const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
        this.curModelGraph,
      );
      this.curNodeDataProviderRuns = {};
      for (const run of runs) {
        this.curNodeDataProviderRuns[run.runId] = run;
      }
    });

    effect(() => {
      const results = this.webglRendererNdpService.curNodeDataProviderResults();
      const run = this.webglRendererNdpService.curNodeDataProviderRun();
      if (results !== this.prevNodeDataProviderData) {
        this.handleCurNodeDataProviderResultsChanged(
          this.prevNodeDataProviderRun,
          run,
        );
        this.prevNodeDataProviderData = results;
        this.prevNodeDataProviderRun = run;
      }
    });

    // Handle changes on show on node items.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane) {
        return;
      }
      const showOnNodeItemTypes = this.appService.getShowOnNodeItemTypes(
        this.paneId,
        this.rendererId,
      );
      if (
        JSON.stringify(showOnNodeItemTypes) ===
        JSON.stringify(this.curShowOnNodeItemTypes)
      ) {
        return;
      }
      this.curShowOnNodeItemTypes = showOnNodeItemTypes;

      // Relayout.
      this.sendRelayoutGraphRequest(this.selectedNodeId);
    });

    // Handle clicking on the expand/collpase all graph layers button.
    this.appService.expandOrCollapseAllGraphLayersClicked
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((info) => {
        if (info.rendererId !== this.rendererId) {
          return;
        }
        this.sendExpandOrCollapseGroupNodeRequest(
          undefined,
          true,
          info.expandOrCollapse,
        );
      });

    // Handle selected node changes.
    effect(() => {
      const info = this.selectedNodeInfo();
      if (info?.rendererId !== this.rendererId) {
        return;
      }

      const selectedNodeId = info?.nodeId || '';
      const selectedNodeChanged = this.selectedNodeId !== selectedNodeId;
      this.selectedNodeId = selectedNodeId;

      if (this.tracing) {
        if (
          this.selectedNodeId &&
          isOpNode(this.curModelGraph.nodesById[this.selectedNodeId])
        ) {
          this.webglRendererIoTracingService.genTracingData();
        } else {
          this.webglRendererIoTracingService.clearTracingData();
        }
      }

      // This has to be placed before updateNodesStyles because it calculates
      // data needed to update nodes styles correctly.
      this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
      this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
      this.webglRendererEdgeOverlaysService.updateOverlaysData();
      this.updateNodesStyles();
      this.webglRendererThreejsService.render();

      // Trigger a navigation sync request (if enabled).
      if (selectedNodeChanged && info.triggerNavigationSync) {
        this.syncNavigationService.updateNavigationSource({
          paneIndex: this.appService.getPaneIndexById(this.paneId) || 0,
          nodeId: this.selectedNodeId,
        });
      }

      // Automatically reveal all nodes in the edge overlays (if existed).
      if (this.webglRendererEdgeOverlaysService.curOverlays.length > 0) {
        const deepestExpandedGroupNodeIds =
          this.webglRendererEdgeOverlaysService.getDeepestExpandedGroupNodeIds();
        if (deepestExpandedGroupNodeIds.length > 0) {
          this.sendRelayoutGraphRequest(
            this.selectedNodeId,
            deepestExpandedGroupNodeIds,
          );
        } else {
          this.webglRendererEdgeOverlaysService.updateOverlaysEdges();
          this.webglRendererThreejsService.render();
        }
      } else {
        this.webglRendererEdgeOverlaysService.clearOverlaysEdges();
        this.webglRendererThreejsService.render();
      }
    });

    // Handle selected edge overlays changes.
    effect(() => {
      this.webglRendererEdgeOverlaysService.edgeOverlaysService.selectedOverlayIds();
      this.webglRendererEdgeOverlaysService.updateOverlaysData();

      // Automatically reveal all nodes in the edge overlays (if existed).
      if (this.selectedNodeId !== '') {
        if (this.webglRendererEdgeOverlaysService.curOverlays.length > 0) {
          const deepestExpandedGroupNodeIds =
            this.webglRendererEdgeOverlaysService.getDeepestExpandedGroupNodeIds();
          if (deepestExpandedGroupNodeIds.length > 0) {
            this.sendRelayoutGraphRequest(
              this.selectedNodeId,
              deepestExpandedGroupNodeIds,
            );
          } else {
            this.webglRendererEdgeOverlaysService.updateOverlaysEdges();
            this.webglRendererThreejsService.render();
          }
        } else {
          this.webglRendererEdgeOverlaysService.clearOverlaysEdges();
          this.webglRendererThreejsService.render();
        }
      }
    });

    // Handle "download as png".
    this.appService.downloadAsPngClicked
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        if (data.rendererId !== this.rendererId) {
          return;
        }
        this.handleDownloadAsPng(data.fullGraph, data.transparentBackground);
      });

    // Handle node styler changes.
    effect(() => {
      const curNodeStylerRules = this.nodeStylerService
        .rules()
        .filter(
          (rule) =>
            hasNonEmptyQueries(rule.queries) &&
            Object.keys(rule.styles).length > 0,
        );
      const strCurNodeStylerRules = JSON.stringify(curNodeStylerRules);
      if (JSON.stringify(this.curNodeStylerRules) !== strCurNodeStylerRules) {
        this.curNodeStylerRules = JSON.parse(
          strCurNodeStylerRules,
        ) as NodeStylerRule[];
        this.curProcessedNodeStylerRules = processNodeStylerRules(
          this.curNodeStylerRules,
        );
        this.renderGraph({
          skipReRenderEdges: true,
          skipReRenderEdgeTexts: true,
        });
        this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
        this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
        this.updateNodesStyles();
        this.renderDiffHighlights();
        this.webglRendererThreejsService.render();
      }
    });

    // Handle changes on show on edge items.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane) {
        return;
      }
      const showOnEdgeItem = this.appService.getShowOnEdgeItem(
        this.paneId,
        this.rendererId,
      );
      if (
        JSON.stringify(showOnEdgeItem) ===
        JSON.stringify(this.curShowOnEdgeItem)
      ) {
        return;
      }
      this.curShowOnEdgeItem = showOnEdgeItem;
      this.renderGraph();
      this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
      this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
      this.updateNodesStyles();
      this.renderDiffHighlights();
      this.webglRendererThreejsService.render();
    });

    // Handle input/output highlight visibility changes.
    effect(() => {
      this.curHiddenInputOpNodeIds =
        this.splitPaneService.hiddenInputOpNodeIds();
      this.curHiddenOutputIds = this.splitPaneService.hiddenOutputIds();
      this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
      this.updateNodesStyles();
      this.webglRendererThreejsService.render();
    });

    // Handle navigation sync source changes.
    this.syncNavigationService.navigationSourceChanged$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        this.syncNavigationRelatedNodesHighlights.clearNodeHighlights();

        if (!data) {
          return;
        }

        // Handle the case when the current pane is not the source pane, i.e.
        // when a node is selected in the other pane.
        if (data.paneIndex !== this.appService.getPaneIndexById(this.paneId)) {
          // Clicking on the empty space. Hide the no mapped node message.
          if (data.nodeId === '') {
            this.syncNavigationService.setShowNoMappedNodeMessage(false);
          }
          // Clicking on a node.
          else {
            const mappedNodeIds = this.syncNavigationService
              .getMappedNodeIds(data.paneIndex, data.nodeId)
              .filter((nodeId) => this.curModelGraph.nodesById[nodeId] != null);
            // Mapped to a single node.
            if (mappedNodeIds.length < 2) {
              const mappedNodeId = mappedNodeIds[0] ?? '';
              const mappedNode = this.curModelGraph.nodesById[mappedNodeId];
              const hideInLayout =
                isOpNode(mappedNode) && mappedNode.hideInLayout;
              if (
                mappedNode &&
                mappedNode.id !== this.selectedNodeId &&
                !hideInLayout
              ) {
                this.revealNode(mappedNodeId, false);
                this.syncNavigationService.setShowNoMappedNodeMessage(false);
              } else if (!mappedNode || hideInLayout) {
                this.syncNavigationService.setShowNoMappedNodeMessage(true);
              } else {
                this.syncNavigationService.setShowNoMappedNodeMessage(false);
              }
            }
            // Mapped to a list of nodes.
            else {
              this.revealAndHighlightNodes(
                mappedNodeIds,
                mappedNodeIds.length > 0 ? mappedNodeIds[0] : '',
                true,
              );
            }
          }
        }
        // This is the case when the current pane is the source pane, i.e. a
        // node is selected in the current pane that triggers a sync navigation
        // event.
        else {
          const nodeId = data.nodeId;
          const relatedNodeIds = this.syncNavigationService
            .getRelatedNodeIdsFromTheSameSide(data.paneIndex, nodeId)
            .filter((nodeId) => this.curModelGraph.nodesById[nodeId] != null);
          if (relatedNodeIds.length > 1) {
            this.revealAndHighlightNodes(relatedNodeIds, nodeId, false);
          }
        }
      });

    effect(() => {
      // Track the changes for the current sync navigation mode and data, and
      // render the diff highlights if there is any change.
      const curMode = this.syncNavigationService.mode();
      const curData =
        this.syncNavigationService.savedProcessedSyncNavigationData()[curMode];
      const curShowDiffHighlightsInMatchNodeIdMode =
        this.syncNavigationService.getShowDiffHighlightsInMatchNodeIdMode();
      if (
        curMode === this.savedSyncNavigationMode &&
        curData === this.savedSyncNavigationData &&
        curShowDiffHighlightsInMatchNodeIdMode ===
          this.savedShowDiffHighlightsInMatchNodeIdMode
      ) {
        return;
      }

      this.savedSyncNavigationMode = curMode;
      this.savedSyncNavigationData = curData;
      this.savedShowDiffHighlightsInMatchNodeIdMode =
        curShowDiffHighlightsInMatchNodeIdMode;

      this.renderDiffHighlights();
    });

    // Re-render diff highlights when graph changes in panes, e.g. when a new
    // graph is loaded in a pane, or when a pane is closed.
    effect(() => {
      const key = this.paneGraphTitlesKey();
      untracked(() => {
        this.renderDiffHighlights();
      });
    });
  }

  ngOnInit() {
    this.graphId = this.modelGraph.id;
    this.curModelGraph = this.modelGraph;
    this.appService.updateCurrentModelGraph(this.paneId, this.curModelGraph);

    // Load show on node item types from local storage.
    if (!this.inPopup) {
      this.curShowOnNodeItemTypes =
        this.appService.getSavedShowOnNodeItemTypes();
      this.curShowOnEdgeItem = this.appService.getSavedShowOnEdgeItem();
    }

    this.webglRendererThreejsService.setupZoomAndPan(
      this.container.nativeElement,
      0.0001,
      20,
    );
    this.webglRendererThreejsService.setupThreeJs();

    // Run outside Angular to not trigger change detection.
    this.ngZone.runOutsideAngular(() => {
      this.canvas.nativeElement.addEventListener('mousemove', (e) => {
        this.handleMouseMove(e);
      });
    });

    const initialUiState = this.appService.curInitialUiState();

    const initGraphFn = (nodeIdToZoomInto?: string) => {
      this.updateNodesAndEdgesToRender();
      this.renderGraph();
      this.webglRendererThreejsService.zoomFitGraph(0.9, 0);

      const pane = this.appService.getPaneById(this.paneId);
      // Restore snapshot if set in pane.
      if (pane?.snapshotToRestore != null) {
        const snapshot = pane.snapshotToRestore;
        // See comments in restoreSnapshot for more details why this is needed.
        this.curShowOnNodeItemTypes =
          pane.snapshotToRestore.showOnNodeItemTypes || {};
        this.appService.setShowOnNode(
          this.paneId,
          this.rendererId,
          this.curShowOnNodeItemTypes,
        );
        this.sendRelayoutGraphRequest(
          snapshot.selectedNodeId || '',
          snapshot.deepestExpandedGroupNodeIds || [],
          false,
          snapshot.rect,
          true,
          snapshot.showOnNodeItemTypes,
          true,
          false,
        );
        pane.snapshotToRestore = undefined;
      } else {
        if (nodeIdToZoomInto != null && nodeIdToZoomInto !== '') {
          setTimeout(() => {
            this.appService.curToLocateNodeInfo.set({
              nodeId: nodeIdToZoomInto,
              rendererId: this.rendererId,
              isGroupNode: false,
              noNodeShake: true,
            });
          });
        }
      }

      // Automatically expand the given root node if it is not expanded.
      this.sendExpandGroupNodeRequest(this.rootNodeId || '');
    };

    // No initial UI state to restore.
    if (
      !initialUiState ||
      initialUiState.paneStates.length === 0 ||
      this.inPopup
    ) {
      const selectedNodeId = this.inPopup
        ? undefined
        : this.appService.getPaneById(this.paneId)?.selectedNodeInfo?.nodeId;
      initGraphFn(selectedNodeId);
    }
    // Restore initial UI state.
    else {
      const paneIndex = this.appService.getPaneIndexById(this.paneId);
      const paneState = initialUiState.paneStates[paneIndex];
      if (!paneState) {
        initGraphFn();
      } else {
        // Expand all layers if paneState.deepestExpandedGroupNodeIds has only
        // one elemenet '___all___'.
        let deepestExpandedGroupNodeIds = paneState.deepestExpandedGroupNodeIds;
        if (
          deepestExpandedGroupNodeIds.length === 1 &&
          deepestExpandedGroupNodeIds[0] === '___all___'
        ) {
          const groupNodeIds: string[] = [];
          getDeepestExpandedGroupNodeIds(
            undefined,
            this.curModelGraph,
            groupNodeIds,
            true,
          );
          deepestExpandedGroupNodeIds = groupNodeIds;
        }
        // Add the parent node of the selected node if it is not set in
        // deepestExpandedGroupNodeIds.
        else {
          const selectedNode =
            this.curModelGraph.nodesById[paneState.selectedNodeId];
          const nsParentId = selectedNode?.nsParentId || '';
          if (
            selectedNode &&
            nsParentId &&
            !deepestExpandedGroupNodeIds.includes(nsParentId)
          ) {
            deepestExpandedGroupNodeIds.push(nsParentId);
          }
        }
        if (
          paneState.selectedNodeId !== '' ||
          deepestExpandedGroupNodeIds.length > 0
        ) {
          this.sendRelayoutGraphRequest(
            paneState.selectedNodeId,
            deepestExpandedGroupNodeIds,
            true,
            undefined,
            false,
            undefined,
            false,
            false,
          );
        } else {
          initGraphFn();
        }
        // This is needed for loading old perma-link.
        this.uiStateService.setDeepestExpandedGroupNodeIds(
          paneState.deepestExpandedGroupNodeIds,
          paneIndex,
        );
      }
    }

    // Store the renderer to global for testing purpose.
    // tslint:disable-next-line:no-any Allow arbitrary types.
    const windowAny = window as any;
    if (windowAny[GLOBAL_KEY] == null) {
      windowAny[GLOBAL_KEY] = {
        renderers: {},
      };
    }
    const paneIndex = this.inPopup
      ? -1
      : this.appService.getPaneIndexById(this.paneId);
    windowAny[GLOBAL_KEY].renderers[paneIndex] = this;

    if (this.benchmark) {
      this.startBenchmark();
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['paneId']) {
      this.paneIdInternal.set(this.paneId);
    }
  }

  ngOnDestroy() {
    this.workerService.worker.removeEventListener(
      'message',
      this.messageEventListener,
    );

    this.webglRendererThreejsService.dispose();

    document.body.style.cursor = 'default';
  }

  getActiveSelectedNodeInfo(): SelectedNodeInfo | undefined {
    if (!this.selectedNodeId) {
      return undefined;
    }

    return {
      nodeId: this.selectedNodeId,
      rendererId: this.rendererId,
      isGroupNode: isGroupNode(
        this.curModelGraph.nodesById[this.selectedNodeId],
      ),
    };
  }

  toggleIoTrace() {
    this.tracing = !this.tracing;

    if (this.tracing) {
      this.webglRendererIoTracingService.genTracingData();
    } else {
      this.webglRendererIoTracingService.clearTracingData();
    }

    this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
    this.updateNodesStyles();
    this.webglRendererThreejsService.render();
  }

  setZoomFactor(factor: number) {
    const container = this.container.nativeElement;
    const start = this.webglRendererThreejsService.convertScreenPosToScene(
      0,
      0,
    );
    const end = this.webglRendererThreejsService.convertScreenPosToScene(
      container.offsetWidth,
      container.offsetHeight,
    );
    const minX = Math.min(start.x, end.x);
    const maxX = Math.max(start.x, end.x);
    const minY = Math.min(start.y, end.y);
    const maxY = Math.max(start.y, end.y);
    const width = maxX - minX;
    const height = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const zoomedMinX = centerX - width / 2 / factor;
    const zoomedMaxX = centerX + width / 2 / factor;
    const zoomedMinY = centerY - height / 2 / factor;
    const zoomedMaxY = centerY + height / 2 / factor;
    this.webglRendererThreejsService.zoomFit(
      {
        x: zoomedMinX,
        y: zoomedMinY,
        width: zoomedMaxX - zoomedMinX,
        height: zoomedMaxY - zoomedMinY,
      },
      0.9,
      0,
      false,
      false,
    );
  }

  handleMouseDownCanvas(event: MouseEvent) {
    // Uncomment to show scene position on click for debugging purpose.
    //
    // const scenePos = this.convertScreenPosToScene(event.offsetX, event.offsetY);
    // console.log('screen pos', scenePos);

    // Range zoom.
    if (
      (IS_MAC && (event.metaKey || event.ctrlKey)) ||
      (!IS_MAC && event.ctrlKey)
    ) {
      this.draggingArea = true;
      this.rangeZoomDragArea.start(
        event,
        (isClick, startX, startY, endX, endY) => {
          const start =
            this.webglRendererThreejsService.convertScreenPosToScene(
              startX,
              startY,
            );
          const end = this.webglRendererThreejsService.convertScreenPosToScene(
            endX,
            endY,
          );
          const minX = Math.min(start.x, end.x);
          const maxX = Math.max(start.x, end.x);
          const minY = Math.min(start.y, end.y);
          const maxY = Math.max(start.y, end.y);
          this.webglRendererThreejsService.zoomFit(
            {x: minX, y: minY, width: maxX - minX, height: maxY - minY},
            0.9,
            200,
            false,
            false,
          );
          this.draggingArea = false;
        },
      );
    }
    // Drag to select subgraph.
    else if (
      event.shiftKey &&
      this.webglRendererSubgraphSelectionService.enableSubgraphSelection
    ) {
      this.draggingArea = true;
      this.dragToSelectDragArea.start(
        event,
        (isClick, startX, startY, endX, endY) => {
          this.draggingArea = false;

          // Click.
          if (isClick) {
            if (this.hoveredNodeId) {
              const node = this.curModelGraph.nodesById[this.hoveredNodeId];
              if (node) {
                this.handleShiftSelectNode(this.hoveredNodeId);
              }
            } else {
              this.handleClearSubgraphSelectedNodes();
            }
          }
          // Drag.
          else {
            const start =
              this.webglRendererThreejsService.convertScreenPosToScene(
                startX,
                startY,
              );
            const end =
              this.webglRendererThreejsService.convertScreenPosToScene(
                endX,
                endY,
              );
            const minAx = Math.min(start.x, end.x);
            const maxAx = Math.max(start.x, end.x);
            const minAy = Math.min(start.y, end.y);
            const maxAy = Math.max(start.y, end.y);
            const coveredNodeIds: string[] = [];
            for (const {node} of this.nodesToRender) {
              const x = this.getNodeX(node);
              const y = this.getNodeY(node);
              const w = this.getNodeWidth(node);
              const h = this.getNodeHeight(node);
              const minBx = x;
              const minBy = y;
              const maxBx = x + w;
              const maxBy = y + h;

              // Check if they intersect.
              const aLeftOfB = maxAx < minBx;
              const aRightOfB = minAx > maxBx;
              const aAboveB = minAy > maxBy;
              const aBelowB = maxAy < minBy;
              const intersect = !(aLeftOfB || aRightOfB || aAboveB || aBelowB);

              if (intersect) {
                coveredNodeIds.push(node.id);
              }
            }
            this.subgraphSelectionService.toggleNodes(coveredNodeIds);
          }
        },
      );
    }
  }

  handleMouseLeaveRenderer(event: MouseEvent) {
    // Ignore when a menu is opened.
    const relatedTarget = event.relatedTarget as HTMLElement;
    if (
      relatedTarget != null &&
      relatedTarget.classList.contains('cdk-overlay-backdrop')
    ) {
      return;
    }
    this.setHoveredNodeId('');
    this.updateNodesStyles();
    this.handleHoveredGroupNodeIconChanged();
    this.webglRendererThreejsService.render();
  }

  handleClickToggleExpandCollapse(all = false) {
    if (!this.hoveredNodeId) {
      return;
    }
    this.handleSelectNode(this.hoveredNodeId);
    const node = this.curModelGraph.nodesById[this.hoveredNodeId] as GroupNode;
    this.handleToggleExpandCollapse(node, all);
  }

  handleClickExpandAll(nodeId?: string) {
    const targetNodeId = nodeId ?? this.hoveredNodeId;
    if (!targetNodeId) {
      return;
    }
    this.handleSelectNode(targetNodeId);
    this.handleToggleExpandCollapse(
      this.curModelGraph.nodesById[targetNodeId],
      true,
      true,
    );
  }

  handleClickCollapseAll(nodeId?: string) {
    const targetNodeId = nodeId ?? this.hoveredNodeId;
    if (!targetNodeId) {
      return;
    }
    this.handleSelectNode(targetNodeId);
    this.handleToggleExpandCollapse(
      this.curModelGraph.nodesById[targetNodeId],
      true,
      false,
    );
  }

  handleClickOpenGroupNodeInPopup(mouseEvent: MouseEvent, nodeId?: string) {
    const targetNodeId = nodeId ?? this.hoveredNodeId;
    const groupNode = this.curModelGraph.nodesById[targetNodeId] as GroupNode;

    // Place the popup next to the target node (if it is collapsed) or its
    // overflow icon (if it is expanded).
    let popupX = 0;
    const x = this.getNodeX(groupNode);
    const width = this.getNodeWidth(groupNode);
    if (groupNode.expanded) {
      const labelSize = this.texts.getLabelSizes(
        this.getNodeLabel(groupNode),
        FontWeight.BOLD,
        NODE_LABEL_HEIGHT,
      ).sizes;
      const scale = NODE_LABEL_HEIGHT / this.texts.getFontSize();
      const labelWidth = (labelSize.maxX - labelSize.minX) * scale;
      const labelRight = x + width / 2 + labelWidth / 2;
      popupX = this.webglRendererThreejsService.convertScenePosToScreen(
        labelRight + 22,
        0,
      ).x;
    } else {
      popupX = this.webglRendererThreejsService.convertScenePosToScreen(
        x + width + 1,
        0,
      ).x;
    }

    const req: PreparePopupRequest = {
      eventType: WorkerEventType.PREPARE_POPUP_REQ,
      modelGraphId: this.curModelGraph.id,
      paneId: this.paneId,
      rendererId: genUid(),
      groupNodeId: groupNode.id,
      initialPosition: {
        x: popupX,
        y: this.webglRendererThreejsService.convertScenePosToScreen(
          0,
          this.getNodeY(groupNode),
        ).y,
      },
    };
    this.workerService.worker.postMessage(req);
  }

  handleClickDownloadGroupNode(nodeId?: string) {
    const targetNodeId = nodeId ?? this.hoveredNodeId;
    if (!targetNodeId) {
      return;
    }

    let graph = this.appService.getGraphById(this.curModelGraph.id);
    if (!graph) {
      return;
    }

    // Extract the subgraph containing only the descendant nodes of the target
    // group node.
    const groupNode: GroupNode = this.curModelGraph.nodesById[
      targetNodeId
    ] as GroupNode;
    const normalizedGroupNodeLabel = groupNode.label.replace(
      /[^a-zA-Z0-9]/g,
      '_',
    );
    const groupNodeDescendantNodeIds = new Set<String>(
      groupNode.descendantsOpNodeIds ?? [],
    );
    graph = JSON.parse(JSON.stringify(graph)) as Graph;
    const nodes: GraphNode[] = graph.nodes.filter((node) =>
      groupNodeDescendantNodeIds.has(node.id),
    );

    // Filter incoming edges to only keep those whose source nodes are in the
    // subgraph.
    for (const node of nodes) {
      if (node.incomingEdges) {
        node.incomingEdges = node.incomingEdges.filter((edge) =>
          groupNodeDescendantNodeIds.has(edge.sourceNodeId),
        );
      }
    }

    // Create a new graph with the subgraph.
    const subgraphId = `${graph.id}_${normalizedGroupNodeLabel}`;
    const subgraph: Graph = {
      id: subgraphId,
      collectionLabel: graph.collectionLabel,
      nodes,
    };

    // Download it.
    const link = document.createElement('a');
    link.download = `${subgraphId}.json`;
    const dataUrl = `data:text/json;charset=utf-8, ${encodeURIComponent(
      JSON.stringify([subgraph], null, 2),
    )}`;
    setAnchorHref(link, dataUrl);
    link.click();
  }

  handleClickGroupNodeIcon(event: MouseEvent) {
    event.stopPropagation();

    if (this.hoveredGroupNodeIconId.includes('_left')) {
      this.handleSelectNode(this.nodeIdForHoveredGroupNodeIcon);
      this.handleToggleExpandCollapse(
        this.curModelGraph.nodesById[this.nodeIdForHoveredGroupNodeIcon],
      );
    } else if (this.hoveredGroupNodeIconId.includes('_right')) {
      this.hoveredNodeIdWhenClickingMoreActions =
        this.nodeIdForHoveredGroupNodeIcon;
      this.moreActionsMenuTrigger.openMenu();
    }
  }

  handleClickIoPicker(event: MouseEvent) {
    event.stopPropagation();

    const isInput = this.hoveredIoPickerId.endsWith('input');
    const nodeId = this.hoveredIoPickerId.split(IO_PICKER_ID_SEP)[0];

    this.webglRendererIoHighlightService.handleClickIoPicker(isInput, nodeId);
  }

  handleClickSubgraphIndicator(event: MouseEvent) {
    if (!this.hoveredSubgraphIndicatorId) {
      return;
    }

    // Get the node.
    //
    // hoveredSubgraphIndicatorId is node id.
    const node = this.curModelGraph.nodesById[
      this.hoveredSubgraphIndicatorId
    ] as OpNode;
    if (!isOpNode(node)) {
      return;
    }

    // If there is only a single subgraph linked to the node, jump to it
    // directly.
    const subgraphIds = node.subgraphIds!;
    if (subgraphIds.length === 1) {
      this.clickSubgraph(subgraphIds[0], event);
    }
    // If there are multiple subgraphs linked to the node, open a menu to let
    // users select a subgraph to jump to.
    else if (subgraphIds.length > 1) {
      this.curSubgraphIdsForMenu = subgraphIds;
      this.subgraphMenuTrigger.openMenu();
    }
  }

  handleClickSubgraphId(subgraphId: string, event: MouseEvent) {
    this.clickSubgraph(subgraphId, event);
  }

  handleDoubleClickOnGraph(altDown: boolean, shiftDown: boolean) {
    // Expand/collapse node on double click. Alt key controls whether to do it
    // for all sub layers.
    if (this.selectedNodeId !== '' && !shiftDown) {
      const node = this.curModelGraph.nodesById[
        this.selectedNodeId
      ] as GroupNode;
      this.appService.updateDoubleClickedNode(
        this.selectedNodeId,
        this.curModelGraph.id,
        this.curModelGraph.collectionLabel || '',
        node,
      );
      this.handleToggleExpandCollapse(node, altDown);
    }
  }

  handleClickOnGraph(shiftDown: boolean): void {
    // Click on a node.
    if (this.hoveredNodeId) {
      const node = this.curModelGraph.nodesById[this.hoveredNodeId];
      if (node) {
        if (!shiftDown) {
          this.handleSelectNode(this.hoveredNodeId);
        }
      }
    }
    // Click on empty space.
    else {
      this.handleSelectNode('');
    }
  }

  handleMouseEnterGroupNodeIcon() {
    this.groupNodeIconMatTooltip.show();
  }

  handleMouseLeaveGroupNodeIcon() {
    this.groupNodeIconMatTooltip.hide();
  }

  handleMouseEnterIoPicker() {
    this.ioPickerMatTooltip.show();
  }

  handleMouseLeaveIoPicker() {
    this.ioPickerMatTooltip.hide();
  }

  handleHoveredGroupNodeIconChanged(rectangle?: RoundedRectangleData) {
    this.groupNodeIcon.top = -1000;
    this.groupNodeIcon.left = -1000;
    this.groupNodeIcon.width = 0;
    this.groupNodeIcon.height = 0;
    this.groupNodeIcon.tooltip = undefined;

    if (this.hoveredGroupNodeIconId !== '' && rectangle != null) {
      const {x, y} = this.webglRendererThreejsService.convertScenePosToScreen(
        rectangle.bound.x - rectangle.bound.width / 2,
        rectangle.bound.y - rectangle.bound.height / 2,
      );
      const {x: right, y: bottom} =
        this.webglRendererThreejsService.convertScenePosToScreen(
          rectangle.bound.x + rectangle.bound.width / 2,
          rectangle.bound.y + rectangle.bound.height / 2,
        );
      this.groupNodeIcon.top = y;
      this.groupNodeIcon.left = x;
      this.groupNodeIcon.width = right - x;
      this.groupNodeIcon.height = bottom - y;
      if (rectangle.id.includes('_left')) {
        const node = this.curModelGraph.nodesById[
          rectangle.nodeId!
        ] as GroupNode;
        this.groupNodeIcon.tooltip = node.expanded
          ? 'Collapse layer'
          : 'Expand layer';
      } else if (rectangle.id.includes('_right')) {
        this.groupNodeIcon.tooltip = 'More actions';
      }
    }
    this.changeDetectorRef.detectChanges();
  }

  isNodeRendered(nodeId: string): boolean {
    return this.nodesToRenderMap[nodeId] != null;
  }

  sendLocateNodeRequest(
    nodeId: string,
    rendererId: string,
    noNodeShake = false,
    select = false,
  ) {
    this.showBusySpinnerWithDelay();

    if (this.isNodeRendered(nodeId)) {
      this.hideBusySpinner();
      this.handleLocateNodeDone(
        rendererId,
        this.curModelGraph,
        nodeId,
        [],
        noNodeShake,
        select,
        true,
      );
    } else {
      const req: LocateNodeRequest = {
        eventType: WorkerEventType.LOCATE_NODE_REQ,
        modelGraphId: this.curModelGraph.id,
        showOnNodeItemTypes: this.curShowOnNodeItemTypes,
        nodeDataProviderRuns: this.curNodeDataProviderRuns,
        selectedNodeDataProviderRunId:
          this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
            this.paneId,
            this.curModelGraph,
          )?.runId,
        nodeId,
        rendererId,
        noNodeShake,
        select,
        config: this.appService.config(),
      };
      this.workerService.worker.postMessage(req);
    }
  }

  sendRelayoutGraphRequest(
    nodeId: string,
    targetDeepestGroupNodeIdsToExpand?: string[],
    forRestoringUiState = false,
    rectToZoomFit?: Rect,
    clearAllExpandStates = false,
    showOnNodeItemTypes?: Record<string, ShowOnNodeItemData>,
    forRestoringSnapshotAfterTogglingFlattenLayers?: boolean,
    triggerNavigationSync = true,
  ) {
    this.showBusySpinnerWithDelay();

    const req: RelayoutGraphRequest = {
      eventType: WorkerEventType.RELAYOUT_GRAPH_REQ,
      modelGraphId: this.curModelGraph.id,
      showOnNodeItemTypes: showOnNodeItemTypes || this.curShowOnNodeItemTypes,
      nodeDataProviderRuns: this.curNodeDataProviderRuns,
      selectedNodeDataProviderRunId:
        this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
          this.paneId,
          this.curModelGraph,
        )?.runId,
      selectedNodeId: nodeId,
      targetDeepestGroupNodeIdsToExpand,
      rendererId: this.rendererId,
      forRestoringUiState,
      rectToZoomFit,
      clearAllExpandStates,
      forRestoringSnapshotAfterTogglingFlattenLayers,
      triggerNavigationSync,
      config: this.appService.config(),
    };
    this.workerService.worker.postMessage(req);
  }

  animateIntoPositions(
    updateFn: (t: number) => void = (t) => {
      this.updateAnimatinProgress(t);
    },
  ) {
    const startTs = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTs;
      let t = this.appService.testMode
        ? 1
        : Math.min(1, elapsed / NODE_ANIMATION_DURATION);
      // ease out sine.
      t = Math.sin((t * Math.PI) / 2);

      updateFn(t);
      this.webglRendererThreejsService.render();

      if (t >= 1) {
        updateFn(t);
        this.webglRendererThreejsService.render();
        return;
      }

      requestAnimationFrame(animate);
    };
    animate();
  }

  flash() {
    this.flashing = true;
    this.changeDetectorRef.detectChanges();

    setTimeout(() => {
      this.flashing = false;
      this.changeDetectorRef.detectChanges();
    }, 300);
  }

  updateNodeBgColorWhenFar() {
    const t =
      this.webglRendererThreejsService.convertZFromSceneToScreen(30) *
      this.webglRendererThreejsService.curScale;
    const farStartT = 7.5;
    const farEndT = 7;
    const progress = Math.max(
      0,
      Math.min(1, (farStartT - t) / (farStartT - farEndT)),
    );
    if (Math.abs(progress - this.savedUpdateNodeBgWhenFarProgress) < 1e-5) {
      return;
    }
    this.savedUpdateNodeBgWhenFarProgress = progress;
    this.nodeBodies.setBgColorWhenFar(this.NODE_LABEL_COLOR, progress / 3);
  }

  showIoTree(
    root: HTMLElement,
    nodes: ModelNode[],
    ioType: 'incoming' | 'outgoing',
  ) {
    const overlayRef = this.createOverlay(root);
    const ref = overlayRef.attach(this.portal!);
    const data = genIoTreeData(nodes, [], ioType);
    ref.instance.solidBackground = true;
    ref.instance.rendererId = this.rendererId;
    ref.instance.updateData(data);
    ref.instance.onClose.subscribe(() => {
      overlayRef.dispose();
    });
  }

  getNodeX(node: ModelNode): number {
    return (node.x || 0) + (node.globalX || 0);
  }

  getNodeY(node: ModelNode): number {
    return (node.y || 0) + (node.globalY || 0);
  }

  getNodeWidth(node: ModelNode): number {
    return node.width || 0;
  }

  getNodeHeight(node: ModelNode): number {
    return node.height || 0;
  }

  getNodeRect(node: ModelNode): Rect {
    return {
      x: this.getNodeX(node),
      y: this.getNodeY(node),
      width: this.getNodeWidth(node),
      height: this.getNodeHeight(node),
    };
  }

  getNodeLabelRelativeY(node: ModelNode): number {
    return 14;
  }

  getNodeLabelSizes(node: ModelNode) {
    const scale = NODE_LABEL_HEIGHT / this.texts.getFontSize();
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let firstLineLabelHeight = 0;
    const lines = splitLabel(this.getNodeLabel(node));
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const labelSize = this.texts.getLabelSizes(
        line,
        FontWeight.BOLD,
        NODE_LABEL_HEIGHT,
      ).sizes;
      minX = Math.min(minX, labelSize.minX);
      maxX = Math.max(maxX, labelSize.maxX);
      if (i === 0) {
        firstLineLabelHeight = (labelSize.maxZ - labelSize.minZ) * scale;
      }
    }
    return {minX, maxX, firstLineLabelHeight};
  }

  // Used by tests only.
  getNodeTitleScreenPositionRelativeToCenter(nodeId: string): Point {
    // This is to workaround the issue where nodeId cannot contain '\n' when
    // called from protractor.
    nodeId = nodeId.replaceAll('%%%', '\n');
    const node = this.curModelGraph.nodesById[nodeId];
    const x = this.getNodeX(node) + this.getNodeWidth(node) / 2;
    const y = this.getNodeY(node) + 5;
    const pos = this.webglRendererThreejsService.convertScenePosToScreen(x, y);
    const container = this.container.nativeElement;
    return {
      x: Math.floor(pos.x - container.clientWidth / 2),
      y: Math.floor(pos.y - container.clientHeight / 2),
    };
  }

  // Used by tests only.
  getNodeIoChipScreenPositionRelativeToCenter(nodeId: string): Point {
    const node = this.curModelGraph.nodesById[nodeId];
    const x = this.getNodeX(node) + 5;
    const y = this.getNodeY(node) - 3;
    const pos = this.webglRendererThreejsService.convertScenePosToScreen(x, y);
    const container = this.container.nativeElement;
    return {
      x: Math.floor(pos.x - container.clientWidth / 2),
      y: Math.floor(pos.y - container.clientHeight / 2),
    };
  }

  // Used by tests only.
  getNodeExpandIconPositionRelativeToCenter(nodeId: string): Point {
    const node = this.curModelGraph.nodesById[nodeId] as GroupNode;
    const x = this.getNodeX(node);
    const y = this.getNodeY(node);
    const width = this.getNodeWidth(node);
    const {minX, maxX} = this.getNodeLabelSizes(node);
    const scale = NODE_LABEL_HEIGHT / this.texts.getFontSize();
    const labelWidth = (maxX - minX) * scale;
    const labelLeft = x + width / 2 - labelWidth / 2;
    const iconX = node.expanded ? labelLeft - 13 : (x + labelLeft + 1) / 2 + 1;
    const iconY = y + this.getNodeLabelRelativeY(node);
    const pos = this.webglRendererThreejsService.convertScenePosToScreen(
      iconX,
      iconY,
    );
    const container = this.container.nativeElement;
    return {
      x: Math.floor(pos.x - container.clientWidth / 2),
      y: Math.floor(pos.y - container.clientHeight / 2),
    };
  }

  // Used by tests only.
  getNodeOverflowIconPositionRelativeToCenter(nodeId: string): Point {
    const node = this.curModelGraph.nodesById[nodeId] as GroupNode;
    const x = this.getNodeX(node);
    const y = this.getNodeY(node);
    const width = this.getNodeWidth(node);
    const {minX, maxX} = this.getNodeLabelSizes(node);
    const scale = NODE_LABEL_HEIGHT / this.texts.getFontSize();
    const labelWidth = (maxX - minX) * scale;
    const labelRight = x + width / 2 + labelWidth / 2;
    const iconX = node.expanded
      ? labelRight + 12
      : (x + width + labelRight - 1) / 2 - 1;
    const iconY = y + this.getNodeLabelRelativeY(node);
    const pos = this.webglRendererThreejsService.convertScenePosToScreen(
      iconX,
      iconY,
    );
    const container = this.container.nativeElement;
    return {
      x: Math.floor(pos.x - container.clientWidth / 2),
      y: Math.floor(pos.y - container.clientHeight / 2),
    };
  }

  // Used by tests only.
  getSubgraphIndicatorScreenPositionRelativeToCenter(nodeId: string): Point {
    const node = this.curModelGraph.nodesById[nodeId];
    const x = this.getNodeX(node) + this.getNodeWidth(node) + 10;
    const y = this.getNodeY(node) + 5;
    const pos = this.webglRendererThreejsService.convertScenePosToScreen(x, y);
    const container = this.container.nativeElement;
    return {
      x: Math.floor(pos.x - container.clientWidth / 2),
      y: Math.floor(pos.y - container.clientHeight / 2),
    };
  }

  // Used by tests only
  scrollGraphArea(deltaX: number, deltaY: number) {
    this.webglRendererThreejsService.scrollGraphArea(deltaX, deltaY);
  }

  getShowCollapseAllButton(nodeId?: string): boolean {
    const targetNodeId = nodeId ?? this.hoveredNodeId;
    const node = this.curModelGraph.nodesById[targetNodeId];
    if (node != null && isGroupNode(node)) {
      return node.expanded;
    }
    return false;
  }

  get expandCollapseIcon(): string {
    const node = this.curModelGraph.nodesById[this.hoveredNodeId];
    if (node != null && isGroupNode(node)) {
      return node.expanded ? 'unfold_less' : 'unfold_more';
    }
    return 'unfold_more';
  }

  get expandCollapseIconTooltip(): string {
    const node = this.curModelGraph.nodesById[this.hoveredNodeId];
    if (node != null && isGroupNode(node)) {
      return `${
        node.expanded ? 'Collapse layer' : 'Expand layer'
      }\n(shortcut: double click on layer)`;
    }
    return '';
  }

  get showOpenInPopupButton(): boolean {
    return !this.inPopup;
  }

  get fps(): string {
    return this.webglRendererThreejsService.fps;
  }

  get subgraphIndicatorTooltip(): string {
    if (!this.hoveredSubgraphIndicatorId) {
      return '';
    }
    const node = this.curModelGraph.nodesById[
      this.hoveredSubgraphIndicatorId
    ] as OpNode;
    if (!isOpNode(node)) {
      return '';
    }

    const subgraphIds = node.subgraphIds!;
    if (subgraphIds.length === 1) {
      return `Jump to subgraph "${subgraphIds[0]}"\n(alt-click to open in split pane)`;
    } else {
      return 'Jump to subgraph';
    }
  }

  private handleSelectNode(nodeId: string, triggerNavigationSync = true) {
    this.appService.selectNode(this.paneId, {
      nodeId,
      rendererId: this.rendererId,
      isGroupNode:
        nodeId === ''
          ? false
          : isGroupNode(this.curModelGraph.nodesById[nodeId]),
      triggerNavigationSync,
    });
  }

  private handleExpandOrCollapseGroupNodeDone(
    modelGraph: ModelGraph,
    rendererId: string,
    groupNodeId: string | undefined,
    expanded: boolean,
    deepestExpandedGroupNodeIds: string[],
  ) {
    this.updateCurModelGraph(modelGraph);
    this.updateNodesAndEdgesToRender();

    // Deselect node if it is not rendered.
    //
    // This is possible when a op node inside a layer is selected, then user
    // clicks "collapse all" in the toolbar.
    if (this.selectedNodeId && !this.isNodeRendered(this.selectedNodeId)) {
      this.appService.selectNode(this.paneId, {
        nodeId: '',
        rendererId: this.rendererId,
        isGroupNode: false,
      });
    }

    this.renderGraph();
    if (groupNodeId != null) {
      this.webglRendererThreejsService.zoomFitOnNode(
        groupNodeId,
        modelGraph,
        ZOOM_FIT_ON_NODE_DURATION,
      );
    } else {
      this.webglRendererThreejsService.zoomFitGraph();
    }
    // This has to be placed before updateNodesStyles because it calculates data
    // needed to update nodes styles correctly.
    this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
    this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
    this.updateNodesStyles();
    this.renderDiffHighlights();
    this.webglRendererThreejsService.render();

    if (!this.inPopup) {
      this.uiStateService.setDeepestExpandedGroupNodeIds(
        deepestExpandedGroupNodeIds,
        this.appService.getPaneIndexById(this.paneId),
      );
    }
  }

  private handleToggleExpandCollapse(
    node: ModelNode,
    all = false,
    expandOverride?: boolean,
  ) {
    if (
      node.nodeType !== NodeType.GROUP_NODE ||
      (node.nsChildrenIds || []).length === 0
    ) {
      return;
    }

    // Expand or collapse.
    this.sendExpandOrCollapseGroupNodeRequest(node, all, expandOverride);
  }

  private handleReLayoutGraphDone(
    rendererId: string,
    modelGraph: ModelGraph,
    selectedNodeId: string | undefined,
    forRestoringUiState?: boolean,
    rectToZoomFit?: Rect,
    forRestoringSnapshotAfterTogglingFlattenLayers?: boolean,
    targetDeepestGroupNodeIdsToExpand?: string[],
    triggerNavigationSync?: boolean,
  ) {
    this.updateCurModelGraph(modelGraph);
    this.updateNodesAndEdgesToRender();

    this.renderGraph();
    this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
    this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
    this.webglRendererEdgeOverlaysService.updateOverlaysEdges();
    this.updateNodesStyles();
    this.renderDiffHighlights();
    if (rectToZoomFit) {
      const zoomFitFn = () => {
        this.webglRendererThreejsService.zoomFit(
          rectToZoomFit,
          1,
          200,
          false,
          // Don't cap the scale so we can fully zoom to the given rect.
          false,
        );
      };
      if (forRestoringSnapshotAfterTogglingFlattenLayers) {
        setTimeout(() => {
          zoomFitFn();
        });
      } else {
        zoomFitFn();
      }
    } else if (selectedNodeId !== NODE_ID_WITHOUT_ZOOMFIT) {
      this.webglRendererThreejsService.zoomFitOnNode(
        selectedNodeId,
        modelGraph,
        forRestoringUiState ? 0 : ZOOM_FIT_ON_NODE_DURATION,
      );
    }

    // Select node.
    if (this.selectedNodeId !== selectedNodeId) {
      this.handleSelectNode(selectedNodeId || '', triggerNavigationSync);
    }

    if (!this.inPopup) {
      this.uiStateService.setDeepestExpandedGroupNodeIds(
        targetDeepestGroupNodeIdsToExpand || [],
        this.appService.getPaneIndexById(this.paneId),
      );
    }

    if (this.relayoutDoneFn) {
      this.relayoutDoneFn();
      this.relayoutDoneFn = undefined;
    }
  }

  private handleCurNodeDataProviderResultsChanged(
    prevRun: NodeDataProviderRunData | undefined,
    curRun: NodeDataProviderRunData | undefined,
  ) {
    const prevShowExpandedSummaryOnGroupNode =
      prevRun?.nodeDataProviderData?.[this.curModelGraph.id]
        ?.showExpandedSummaryOnGroupNode;
    const curShowExpandedSummaryOnGroupNode =
      curRun?.nodeDataProviderData?.[this.curModelGraph.id]
        ?.showExpandedSummaryOnGroupNode;

    // Relayout the graph if `showExpandedSummaryOnGroupNode` is changed
    // between previous run and current run.
    if (
      prevShowExpandedSummaryOnGroupNode !== curShowExpandedSummaryOnGroupNode
    ) {
      this.sendRelayoutGraphRequest(this.selectedNodeId);
    }
    // Re-render the graph without re-laying out if
    // `showExpandedSummaryOnGroupNode` is not changed.do {
    else {
      this.renderGraph();
      this.updateNodesStyles();
      this.webglRendererThreejsService.render();
    }
  }

  private handleLocateNodeDone(
    rendererId: string,
    modelGraph: ModelGraph,
    nodeId: string,
    deepestExpandedGroupNodeIds: string[],
    noNodeShake: boolean,
    select: boolean,
    skipRendering = false,
  ) {
    this.updateCurModelGraph(modelGraph);
    this.updateNodesAndEdgesToRender();
    if (select) {
      this.appService.selectNode(this.paneId, {
        nodeId,
        rendererId,
        isGroupNode: isGroupNode(this.curModelGraph.nodesById[nodeId]),
      });
    }

    this.webglRendererThreejsService.zoomFitOnNode(
      nodeId,
      modelGraph,
      ZOOM_FIT_ON_NODE_DURATION,
    );
    if (!skipRendering) {
      this.renderGraph();
      // This has to be placed before updateNodesStyles because it calculates data
      // needed to update nodes styles correctly.
      this.webglRendererIoHighlightService.updateIncomingAndOutgoingHighlights();
      this.webglRendererIdenticalLayerService.updateIdenticalLayerIndicators();
      this.updateNodesStyles();
      this.renderDiffHighlights();
      this.webglRendererThreejsService.render();

      if (!this.inPopup) {
        this.uiStateService.setDeepestExpandedGroupNodeIds(
          deepestExpandedGroupNodeIds,
          this.appService.getPaneIndexById(this.paneId),
        );
      }
    }
    if (!noNodeShake) {
      setTimeout(() => {
        this.shakeNode(nodeId);
      }, 250);
    }
  }

  private sendExpandGroupNodeRequest(groupNodeId: string) {
    const groupNode = this.curModelGraph.nodesById[groupNodeId] as GroupNode;
    if (groupNode != null && !groupNode.expanded) {
      this.sendExpandOrCollapseGroupNodeRequest(groupNode, false, true);
    }
  }

  private sendExpandOrCollapseGroupNodeRequest(
    node: GroupNode | undefined,
    all = false,
    expandOverride?: boolean,
  ) {
    this.showBusySpinnerWithDelay();

    const req: ExpandOrCollapseGroupNodeRequest = {
      eventType: WorkerEventType.EXPAND_OR_COLLAPSE_GROUP_NODE_REQ,
      modelGraphId: this.curModelGraph.id,
      groupNodeId: node?.id,
      // node.expand will be updated in worker. Here we pass the current
      // state.
      expand: expandOverride == null ? !node?.expanded : expandOverride,
      showOnNodeItemTypes: this.curShowOnNodeItemTypes,
      nodeDataProviderRuns: this.curNodeDataProviderRuns,
      selectedNodeDataProviderRunId:
        this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
          this.paneId,
          this.curModelGraph,
        )?.runId,
      rendererId: this.rendererId,
      paneId: this.paneId,
      all,
      ts: Date.now(),
      config: this.appService.config(),
    };
    this.workerService.worker.postMessage(req);
  }

  private renderGraph(options?: RenderGraphOptions) {
    const extraMeshesToSkip: three.Object3D[] = [];
    if (options?.skipReRenderEdgeTexts) {
      const edgeTextsMesh = this.webglRendererEdgeTextsService.edgeTexts.mesh;
      if (edgeTextsMesh) {
        extraMeshesToSkip.push(edgeTextsMesh);
      }
    }
    if (options?.skipReRenderEdges) {
      const edgesMesh = this.edges.edgesMesh;
      if (edgesMesh) {
        extraMeshesToSkip.push(edgesMesh);
      }
      const arrowHeadsMesh = this.edges.arrowHeadsMesh;
      if (arrowHeadsMesh) {
        extraMeshesToSkip.push(arrowHeadsMesh);
      }
    }
    this.clearScene(extraMeshesToSkip);

    if (!options?.skipReRenderEdges) {
      this.renderEdges();
    }
    this.renderTexts();

    const keys = getShowOnEdgeInputOutputMetadataKeys(this.curShowOnEdgeItem);
    if (!options?.skipReRenderEdgeTexts) {
      if (
        keys.outputMetadataKey != null ||
        keys.inputMetadataKey != null ||
        keys.sourceNodeAttrKey != null ||
        keys.targetNodeAttrKey != null
      ) {
        this.webglRendererEdgeTextsService.renderEdgeTexts({
          outputMetadataKey: keys.outputMetadataKey,
          inputMetadataKey: keys.inputMetadataKey,
          sourceNodeAttrKey: keys.sourceNodeAttrKey,
          targetNodeAttrKey: keys.targetNodeAttrKey,
        });
      }
    }

    this.webglRendererAttrsTableService.renderAttrsTable();
    this.renderNodes();
    this.webglRendererNdpService.renderNodeDataProviderDistributionBars();
    this.renderArtificialGroupBorders();
    this.webglRendererSearchResultsService.renderSearchResults();
    this.webglRendererSubgraphSelectionService.renderSubgraphSelectedNodeMarkers();
    this.updateNodeBgColorWhenFar();

    this.animateIntoPositions((t) => {
      this.updateAnimatinProgress(t, options);
    });
  }

  private renderNodes() {
    this.currentMinX = Number.POSITIVE_INFINITY;
    this.currentMinZ = Number.POSITIVE_INFINITY;
    this.currentMaxX = Number.NEGATIVE_INFINITY;
    this.currentMaxZ = Number.NEGATIVE_INFINITY;

    const numNodes = this.nodesToRender.length;

    const nodeBodyRectangles: RoundedRectangleData[] = [];
    const groupNodeIcons: LabelData[] = [];
    const groupNodeIconBgs: RoundedRectangleData[] = [];
    const subgraphIndicatorRectangles: RoundedRectangleData[] = [];
    const subgraphIndicatorIcons: LabelData[] = [];
    const scale = NODE_LABEL_HEIGHT / this.texts.getFontSize();
    for (let i = 0; i < numNodes; i++) {
      const node = this.nodesToRender[i].node;
      const nodeIndex = this.nodesToRender[i].index;
      const x = this.getNodeX(node);
      const y = this.getNodeY(node);
      const width = this.getNodeWidth(node);
      const height = this.getNodeHeight(node);
      const isGroup = isGroupNode(node);
      let borderWidth = NODE_BORDER_WIDTH;
      let bgColor = isGroup
        ? this.getGroupNodeBgColor(node)
        : {r: 1, g: 1, b: 1};
      let borderColor = this.threeColorToRgb(
        isGroup ? this.GROUP_NODE_BORDER_COLOR : this.OP_NODE_BORDER_COLOR,
      );
      if (isOpNode(node) && node.style) {
        if (node.style.backgroundColor) {
          bgColor = new THREE.Color(node.style.backgroundColor);
        }
        if (node.style.borderColor) {
          borderColor = new THREE.Color(node.style.borderColor);
        }
        if (node.style.borderWidth) {
          borderWidth = node.style.borderWidth;
        }
      }
      let groupNodeIconColor = this.GROUP_NODE_ICON_COLOR;

      // Node styler.
      for (const rule of this.curProcessedNodeStylerRules) {
        if (
          matchNodeForQueries(
            node,
            rule.queries,
            this.curModelGraph,
            this.appService.config(),
          )
        ) {
          const nodeStylerBgColor = getNodeStyleValue(
            rule,
            NodeStyleId.NODE_BG_COLOR,
          );
          if (nodeStylerBgColor !== '') {
            bgColor = new THREE.Color(nodeStylerBgColor);
          }
          const nodeBorderColor = getNodeStyleValue(
            rule,
            NodeStyleId.NODE_BORDER_COLOR,
          );
          if (nodeBorderColor !== '') {
            borderColor = new THREE.Color(nodeBorderColor);
          }
          const textColor = getNodeStyleValue(
            rule,
            NodeStyleId.NODE_TEXT_COLOR,
          );
          if (textColor !== '') {
            groupNodeIconColor = new THREE.Color(textColor);
          }
          break;
        }
      }

      let isRounded = true;
      if (isGroupNode(node) && node.sectionContainer) {
        isRounded = false;
      }
      nodeBodyRectangles.push({
        id: node.id,
        index: nodeBodyRectangles.length,
        bound: {
          x: x + width / 2,
          y: y + height / 2,
          width,
          height,
        },
        yOffset: WEBGL_ELEMENT_Y_FACTOR * nodeIndex,
        isRounded,
        borderColor,
        bgColor,
        borderWidth,
        opacity: 1,
        changeColorWhenFar:
          (isOpNode(node) || !node.expanded) &&
          // Don't change color when the node has non-white bg color.
          bgColor.r === 1 &&
          bgColor.g === 1 &&
          bgColor.b === 1,
      });

      // Render separator between the pinned node and the rest of the nodes.
      if (isGroupNode(node) && node.expanded && node.pinToTopOpNode) {
        nodeBodyRectangles.push({
          id: `${node.id}_pin_to_top_separator`,
          index: nodeBodyRectangles.length,
          bound: {
            x: x + width / 2,
            y:
              (node.pinToTopOpNode.globalY || 0) +
              (node.pinToTopOpNode.height || 0) / 2 +
              12.5,
            width: width - LAYOUT_MARGIN_X * 2,
            height: 1,
          },
          yOffset: WEBGL_ELEMENT_Y_FACTOR * nodeIndex + 0.1,
          isRounded: true,
          borderColor: this.GROUP_NODE_PIN_TO_TOP_SEPARATOR_COLOR,
          bgColor: this.GROUP_NODE_PIN_TO_TOP_SEPARATOR_COLOR,
          borderWidth: 1,
          opacity: 1,
        });
      }

      // Subgraph indicators.
      if (isOpNode(node) && node.subgraphIds) {
        const indicatorWidth = SUBGRAPH_INDICATOR_SIZE;
        const indicatorHeight = SUBGRAPH_INDICATOR_SIZE;
        subgraphIndicatorRectangles.push({
          id: `${node.id}`,
          index: subgraphIndicatorRectangles.length,
          bound: {
            x:
              this.getNodeX(node) +
              this.getNodeWidth(node) +
              2 +
              indicatorWidth / 2,
            y: this.getNodeY(node) + indicatorHeight / 2,
            width: indicatorWidth,
            height: indicatorHeight,
          },
          yOffset:
            WEBGL_ELEMENT_Y_FACTOR * this.nodesToRenderMap[node.id].index,
          isRounded: true,
          borderColor: this.SUBGRAPH_INDICATOR_BORDER_COLOR,
          bgColor: this.SUBGRAPH_INDICATOR_BG_COLOR,
          borderWidth: 1,
          opacity: 1,
        });
        subgraphIndicatorIcons.push({
          id: node.id,
          nodeId: node.id,
          // subdirectory_arrow_right
          label: '0xe5da',
          height: 28,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.ICONS,
          color: this.SUBGRAPH_INDICATOR_BORDER_COLOR,
          x:
            this.getNodeX(node) +
            this.getNodeWidth(node) +
            2 +
            indicatorWidth / 2 +
            1,
          y:
            this.nodesToRenderMap[node.id].index * WEBGL_ELEMENT_Y_FACTOR +
            SUBGRAPH_INDICATOR_LABEL_Y_OFFSET,
          z: this.getNodeY(node) + indicatorHeight / 2 + 16,
          treatLabelAsAWhole: true,
        });
      }

      // Group node label icons.
      if (isGroupNode(node)) {
        // Get current node label width.
        const {minX, maxX, firstLineLabelHeight} = this.getNodeLabelSizes(node);
        const labelWidth = (maxX - minX) * scale;
        const labelLeft = x + width / 2 - labelWidth / 2;
        const labelRight = x + width / 2 + labelWidth / 2;

        // Expand icon.
        const iconZ =
          y + this.getNodeLabelRelativeY(node) + firstLineLabelHeight + 7.5;
        const leftIconX = node.expanded
          ? labelLeft - 13
          : (x + labelLeft + 1) / 2 + 1;
        const rightIconX = node.expanded
          ? labelRight + 12
          : (x + width + labelRight - 1) / 2 - 1;
        groupNodeIcons.push({
          id: node.id,
          nodeId: node.id,
          // unfold_more / unfold_less
          label: node.expanded ? '0xe5d6' : '0xe5d7',
          height: 32,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.ICONS,
          color: groupNodeIconColor,
          x: leftIconX,
          y: WEBGL_ELEMENT_Y_FACTOR * nodeIndex + NODE_LABEL_Y_OFFSET,
          z: iconZ,
          treatLabelAsAWhole: true,
        });
        // Overflow icon.
        groupNodeIcons.push({
          id: node.id,
          nodeId: node.id,
          // more_vert
          label: '0xe5d4',
          height: 32,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.ICONS,
          color: groupNodeIconColor,
          x: rightIconX,
          y: WEBGL_ELEMENT_Y_FACTOR * nodeIndex + NODE_LABEL_Y_OFFSET,
          z: iconZ,
          treatLabelAsAWhole: true,
        });

        const iconBgY = y + this.getNodeLabelRelativeY(node) - 1;
        groupNodeIconBgs.push({
          id: this.getGroupNodeLabelSeparatorId(node.id, 'left'),
          nodeId: node.id,
          index: groupNodeIconBgs.length,
          bound: {
            x: leftIconX,
            y: iconBgY,
            width: 16,
            height: 16,
          },
          yOffset:
            WEBGL_ELEMENT_Y_FACTOR * nodeIndex + GROUP_NODE_ICON_BG_OFFSET,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor: {r: 0, g: 0, b: 0},
          borderWidth: 0,
          opacity: 0,
        });
        groupNodeIconBgs.push({
          id: this.getGroupNodeLabelSeparatorId(node.id, 'right'),
          nodeId: node.id,
          index: groupNodeIconBgs.length,
          bound: {
            x: rightIconX,
            y: iconBgY,
            width: 16,
            height: 16,
          },
          yOffset:
            WEBGL_ELEMENT_Y_FACTOR * nodeIndex + GROUP_NODE_ICON_BG_OFFSET,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor: {r: 0, g: 0, b: 0},
          borderWidth: 0,
          opacity: 0,
        });
      }

      // Update graph's range.
      this.currentMinX = Math.min(this.currentMinX, x);
      this.currentMaxX = Math.max(this.currentMaxX, x + width);
      this.currentMinZ = Math.min(this.currentMinZ, y);
      this.currentMaxZ = Math.max(this.currentMaxZ, y + height);
    }
    this.nodeBodies.generateMesh(nodeBodyRectangles, true);
    this.webglRendererThreejsService.addToScene(this.nodeBodies.mesh);
    this.webglRendererThreejsService.addToScene(
      this.nodeBodies.meshForRayCasting,
    );
    this.groupNodeIcons.generateMesh(groupNodeIcons);
    this.webglRendererThreejsService.addToScene(this.groupNodeIcons.mesh);
    this.groupNodeIconBgs.generateMesh(groupNodeIconBgs, true);
    this.webglRendererThreejsService.addToScene(this.groupNodeIconBgs.mesh);
    this.webglRendererThreejsService.addToScene(
      this.groupNodeIconBgs.meshForRayCasting,
    );
    this.subgraphIndicatorBgs.generateMesh(subgraphIndicatorRectangles, true);
    this.webglRendererThreejsService.addToScene(this.subgraphIndicatorBgs.mesh);
    this.subgraphIndicatorIcons.generateMesh(subgraphIndicatorIcons);
    this.webglRendererThreejsService.addToScene(
      this.subgraphIndicatorIcons.mesh,
    );
  }

  private renderEdges() {
    this.renderedEdgeIdsToHide = [];

    if (this.edgesToRender.length > 0) {
      // Add the edges that go out of the layer to the edges to render list
      // if the option is on.
      if (this.appService.config()?.showOpNodeOutOfLayerEdgesWithoutSelecting) {
        for (const {node} of this.nodesToRender) {
          if (isOpNode(node) && node.nsParentId) {
            const {
              overlayEdges: incomingOverlayEdges,
              renderedEdges: incomingRenderedEdges,
            } =
              this.webglRendererIoHighlightService.getHighlightedIncomingNodesAndEdges(
                this.curHiddenInputOpNodeIds,
                node,
                {
                  ignoreEdgesWithinSameNamespace: true,
                  reuseRenderedEdgeCurvePoints: true,
                },
              );
            if (incomingOverlayEdges.length > 0) {
              this.renderedEdgeIdsToHide.push(
                ...incomingRenderedEdges.map((edge) => edge.id),
              );
              for (const edge of incomingOverlayEdges) {
                this.edgesToRender.push({
                  edge,
                  // make sure to pick a number less than 95 which is used for
                  // rendering io highlight edges.
                  index: 92 / WEBGL_ELEMENT_Y_FACTOR,
                });
              }
            }

            const {
              overlayEdges: outgoingOverlayEdges,
              renderedEdges: outgoingRenderedEdges,
            } =
              this.webglRendererIoHighlightService.getHighlightedOutgoingNodesAndEdges(
                this.curHiddenOutputIds,
                node,
                {
                  ignoreEdgesWithinSameNamespace: true,
                  reuseRenderedEdgeCurvePoints: true,
                },
              );
            if (outgoingOverlayEdges.length > 0) {
              this.renderedEdgeIdsToHide.push(
                ...outgoingRenderedEdges.map((edge) => edge.id),
              );
              for (const edge of outgoingOverlayEdges) {
                this.edgesToRender.push({
                  edge,
                  // make sure to pick a number less than 95 which is used for
                  // rendering io highlight edges.
                  index: 92 / WEBGL_ELEMENT_Y_FACTOR,
                });
              }
            }
          }
        }
      }
      this.edges.generateMesh(this.edgesToRender, this.curModelGraph);
      this.webglRendererThreejsService.addToScene(this.edges.edgesMesh);
      this.webglRendererThreejsService.addToScene(this.edges.arrowHeadsMesh);
    }
  }

  private renderTexts() {
    const labels: LabelData[] = [];
    // Node labels.
    for (const {node, index} of this.nodesToRender) {
      let color = this.NODE_LABEL_COLOR;
      if (isOpNode(node) && node.style?.textColor) {
        color = new THREE.Color(node.style.textColor);
      }

      // Node styler.
      for (const rule of this.curProcessedNodeStylerRules) {
        if (
          matchNodeForQueries(
            node,
            rule.queries,
            this.curModelGraph,
            this.appService.config(),
          )
        ) {
          const nodeStylerTextColor = getNodeStyleValue(
            rule,
            NodeStyleId.NODE_TEXT_COLOR,
          );
          if (nodeStylerTextColor !== '') {
            color = new THREE.Color(nodeStylerTextColor);
          }
          break;
        }
      }

      const lines = splitLabel(this.getNodeLabel(node));
      for (let i = 0; i < lines.length; i++) {
        const curLineLabel = lines[i];
        labels.push({
          id: `${node.id}_label_line${i}`,
          nodeId: node.id,
          label: curLineLabel,
          height: NODE_LABEL_HEIGHT,
          hAlign: 'center',
          vAlign: 'center',
          weight: isOpNode(node) ? FontWeight.MEDIUM : FontWeight.BOLD,
          x: this.getNodeX(node) + this.getNodeWidth(node) / 2,
          y: index * WEBGL_ELEMENT_Y_FACTOR + NODE_LABEL_Y_OFFSET,
          z:
            this.getNodeY(node) +
            this.getNodeLabelRelativeY(node) +
            NODE_LABEL_LINE_HEIGHT * i,
          color,
        });
      }
    }
    this.texts.generateMesh(labels);
    this.webglRendererThreejsService.addToScene(this.texts.mesh);
  }

  private renderArtificialGroupBorders() {
    const rectangles: RoundedRectangleData[] = [];
    for (const nodeId of this.curModelGraph.artificialGroupNodeIds || []) {
      if (!this.isNodeRendered(nodeId)) {
        continue;
      }

      const node = this.curModelGraph.nodesById[nodeId];
      const nodeIndex = this.nodesToRenderMap[nodeId].index;
      const x = this.getNodeX(node) - 1;
      const y = this.getNodeY(node) - 1;
      const width = this.getNodeWidth(node) + 2;
      const height = this.getNodeHeight(node) + 2;
      rectangles.push({
        id: nodeId,
        index: rectangles.length,
        bound: {
          x: x + width / 2,
          y: y + height / 2,
          width,
          height,
        },
        yOffset:
          WEBGL_ELEMENT_Y_FACTOR * nodeIndex +
          ARTIFICIAL_GROUP_NODE_BORDER_Y_OFFSET,
        isRounded: false,
        borderColor: {r: 1, g: 1, b: 1},
        bgColor: this.ARTIFCIAL_GROUPS_BORDER_COLOR,
        borderWidth: 0,
        opacity: 1,
      });
    }
    this.artificialGroupBorders.generateMesh(rectangles, false, false, true);
    this.webglRendererThreejsService.addToScene(
      this.artificialGroupBorders.mesh,
    );
  }

  private updateAnimatinProgress(t: number, options?: RenderGraphOptions) {
    this.nodeBodies.updateAnimationProgress(t);
    this.groupNodeIcons.updateAnimationProgress(t);
    this.groupNodeIconBgs.updateAnimationProgress(t);
    this.subgraphIndicatorBgs.updateAnimationProgress(t);
    this.subgraphIndicatorIcons.updateAnimationProgress(t);
    this.texts.updateAnimationProgress(t);
    if (!options?.skipReRenderEdgeTexts) {
      this.webglRendererEdgeTextsService.updateAnimationProgress(t);
    }
    this.webglRendererAttrsTableService.updateAnimationProgress(t);
    this.webglRendererNdpService.updateAnimationProgress(t);
    this.artificialGroupBorders.updateAnimationProgress(t);
    if (!options?.skipReRenderEdges) {
      this.edges.updateAnimationProgress(t);
    }
  }

  private handleMouseMove(event: MouseEvent) {
    // Ignore when dragging out an area.
    if (this.draggingArea) {
      return;
    }

    const canvas = this.canvas.nativeElement;
    this.mousePos.x = (event.offsetX / canvas.offsetWidth) * 2 - 1;
    this.mousePos.y = -(event.offsetY / canvas.offsetHeight) * 2 + 1;
    this.webglRendererThreejsService.raycaster.setFromCamera(
      this.mousePos,
      this.webglRendererThreejsService.camera,
    );

    // Intersect with node.
    this.nodeBodies.raycast(
      this.webglRendererThreejsService.raycaster,
      (recId) => {
        this.setHoveredNodeId(recId);
        this.updateNodesStyles();
        this.webglRendererThreejsService.render();
      },
    );

    // Intersect with group node icons.
    this.groupNodeIconBgs.raycast(
      this.webglRendererThreejsService.raycaster,
      (recId, rectangle) => {
        this.hoveredGroupNodeIconId = recId;
        this.nodeIdForHoveredGroupNodeIcon = rectangle?.nodeId || '';
        this.updateNodesStyles();
        this.handleHoveredGroupNodeIconChanged(rectangle);
        this.webglRendererThreejsService.render();
      },
      false,
    );

    // Intersect with io picker.
    this.webglRendererIoHighlightService.ioPickerBgs.raycast(
      this.webglRendererThreejsService.raycaster,
      (recId, rectangle) => {
        this.hoveredIoPickerId = recId;
        this.handleHoveredIoPickerChanged(rectangle);
      },
    );

    // Intersect with subgraph indicator.
    this.subgraphIndicatorBgs.raycast(
      this.webglRendererThreejsService.raycaster,
      (recId, rectangle) => {
        this.hoveredSubgraphIndicatorId = recId;
        this.handleHoveredSubgraphIndicatorChanged(rectangle);
      },
    );
  }

  private handleHoveredIoPickerChanged(rectangle: RoundedRectangleData) {
    this.ioPickerTop = -1000;
    this.ioPickerLeft = -1000;
    this.ioPickerTooltip = '';
    const isInput = this.hoveredIoPickerId.endsWith('input');

    if (this.hoveredIoPickerId !== '') {
      const {x, y} = this.webglRendererThreejsService.convertScenePosToScreen(
        rectangle.bound.x - rectangle.bound.width / 2,
        rectangle.bound.y - rectangle.bound.height / 2,
      );
      const {x: right, y: bottom} =
        this.webglRendererThreejsService.convertScenePosToScreen(
          rectangle.bound.x + rectangle.bound.width / 2,
          rectangle.bound.y + rectangle.bound.height / 2,
        );
      this.ioPickerTop = y;
      this.ioPickerLeft = x;
      this.ioPickerWidth = right - x;
      this.ioPickerHeight = bottom - y;
      this.ioPickerTooltip = `Click to reveal ${
        isInput ? 'input' : 'output'
      } node(s)`;
    }
    this.changeDetectorRef.detectChanges();
  }

  private handleHoveredSubgraphIndicatorChanged(
    rectangle: RoundedRectangleData,
  ) {
    this.subgraphIndicatorTop = -1000;
    this.subgraphIndicatorLeft = -1000;

    if (this.hoveredSubgraphIndicatorId !== '') {
      const {x, y} = this.webglRendererThreejsService.convertScenePosToScreen(
        rectangle.bound.x - rectangle.bound.width / 2,
        rectangle.bound.y - rectangle.bound.height / 2,
      );
      const {x: right, y: bottom} =
        this.webglRendererThreejsService.convertScenePosToScreen(
          rectangle.bound.x + rectangle.bound.width / 2,
          rectangle.bound.y + rectangle.bound.height / 2,
        );
      this.subgraphIndicatorTop = y;
      this.subgraphIndicatorLeft = x;
      this.subgraphIndicatorWidth = right - x;
      this.subgraphIndicatorHeight = bottom - y;
    }
    this.changeDetectorRef.detectChanges();
  }

  private updateNodesStyles() {
    let selectedNodeIdChanged = false;
    if (this.selectedNodeId !== this.updateNodesStylesSavedSelectedNodeId) {
      this.updateNodesStylesSavedSelectedNodeId = this.selectedNodeId;
      selectedNodeIdChanged = true;
    }

    let ioTracingDataChanged = false;
    if (
      this.webglRendererIoTracingService.curIoTracingData !==
      this.updateNodesStylesSavedIoTracingData
    ) {
      this.updateNodesStylesSavedIoTracingData =
        this.webglRendererIoTracingService.curIoTracingData;
      ioTracingDataChanged = true;
    }

    this.nodeBodies.restoreBorderColors();
    this.nodeBodies.restoreBgColors();
    this.nodeBodies.restoreBorderWidths();
    this.nodeBodies.restoreOpacities();
    this.groupNodeIconBgs.restoreOpacities();
    this.texts.restoreOpacities();
    this.texts.restoreColors();
    this.webglRendererEdgeTextsService.edgeTexts.restoreOpacities();
    this.groupNodeIcons.restoreOpacities();
    this.webglRendererAttrsTableService.attrsTableTexts.restoreOpacities();
    if (selectedNodeIdChanged || ioTracingDataChanged) {
      this.edges.restoreColors();
    }
    this.edges.restoreYOffsets();

    const node = this.curModelGraph.nodesById[this.selectedNodeId];

    // Identical groups.
    if (node != null && isGroupNode(node)) {
      const selectedIdenticalGroupIndex = node.identicalGroupIndex;
      if (selectedIdenticalGroupIndex != null) {
        const identicalGroupNodeIds: string[] = this.nodesToRender
          .filter(
            ({node: curNode}) =>
              isGroupNode(curNode) &&
              curNode.identicalGroupIndex === selectedIdenticalGroupIndex,
          )
          .map(({node}) => node.id);
        this.nodeBodies.updateBgColor(
          identicalGroupNodeIds,
          this.IDENTICAL_GROUPS_BG_COLOR,
        );
      }
    }

    // Border and bg color for hover/select.
    //
    // Hover.
    const hoveredNode = this.curModelGraph.nodesById[this.hoveredNodeId];
    let hoveredNodeBorderColor = isGroupNode(hoveredNode)
      ? this.HOVERED_GROUP_NODE_BORDER_COLOR
      : this.HOVERED_NODE_BORDER_COLOR;
    if (isOpNode(hoveredNode) && hoveredNode.style?.hoveredBorderColor) {
      hoveredNodeBorderColor = new THREE.Color(
        hoveredNode.style.hoveredBorderColor,
      );
    }
    this.nodeBodies.updateBorderColor(
      [this.hoveredNodeId],
      hoveredNodeBorderColor,
    );
    // Selected.
    if (this.selectedNodeId && node != null) {
      this.nodeBodies.updateBorderColor(
        [this.selectedNodeId],
        this.SELECTED_NODE_BORDER_COLOR,
      );
      this.nodeBodies.updateBorderWidth(
        [this.selectedNodeId],
        SELECTED_NODE_BORDER_WIDTH,
      );
      this.nodeBodies.updateBgColor(
        [this.selectedNodeId],
        this.SELECTED_NODE_BG_COLOR,
        isOpNode(node),
      );
    }

    // Group node icon.
    this.groupNodeIconBgs.updateOpacity([this.hoveredGroupNodeIconId], 0.07);

    // IO highlights.
    const highlightedIncomingNodeIds = Object.keys(
      this.webglRendererIoHighlightService.inputsByHighlightedNode,
    );
    if (highlightedIncomingNodeIds.length > 0) {
      this.nodeBodies.updateBorderColor(
        highlightedIncomingNodeIds,
        new THREE.Color(
          this.EDGE_COLOR_INCOMING.r,
          this.EDGE_COLOR_INCOMING.g,
          this.EDGE_COLOR_INCOMING.b,
        ),
      );
      for (const nodeId of highlightedIncomingNodeIds) {
        this.nodeBodies.updateBorderWidth([nodeId], IO_HIGHLIGHT_BORDER_WIDTH);
      }
    }
    const highlightedOutgoingNodeIds = Object.keys(
      this.webglRendererIoHighlightService.outputsByHighlightedNode,
    );
    if (highlightedOutgoingNodeIds.length > 0) {
      this.nodeBodies.updateBorderColor(
        highlightedOutgoingNodeIds,
        new THREE.Color(
          this.EDGE_COLOR_OUTGOING.r,
          this.EDGE_COLOR_OUTGOING.g,
          this.EDGE_COLOR_OUTGOING.b,
        ),
      );
      for (const nodeId of highlightedOutgoingNodeIds) {
        this.nodeBodies.updateBorderWidth([nodeId], IO_HIGHLIGHT_BORDER_WIDTH);
      }
    }
    // Hide all rendered edges to better shown highlighted edges.
    const ids = [
      ...this.webglRendererIoHighlightService.inputsRenderedEdges,
      ...this.webglRendererIoHighlightService.outputsRenderedEdges,
    ].map((edge) => edge.id);
    ids.push(...this.renderedEdgeIdsToHide);
    this.edges.updateYOffsets(ids, 1000);

    // Node data provider.
    //
    const nodeDataProviderResults =
      this.webglRendererNdpService.curNodeDataProviderResults() || {};
    // Update op node bg color.
    for (const nodeId of Object.keys(nodeDataProviderResults)) {
      if (!this.isNodeRendered(nodeId)) {
        continue;
      }
      if (!isOpNode(this.curModelGraph.nodesById[nodeId])) {
        continue;
      }
      const bgColor = nodeDataProviderResults[nodeId].bgColor;
      if (bgColor && bgColor !== 'transparent') {
        this.nodeBodies.updateBgColor([nodeId], new THREE.Color(bgColor));
      }
      const textColor = nodeDataProviderResults[nodeId].textColor;
      if (textColor) {
        this.texts.updateColorInNode([nodeId], new THREE.Color(textColor));
      }
    }

    // Tracing.
    if (this.webglRendererIoTracingService.curIoTracingData != null) {
      const nodeIds = Object.keys(this.curModelGraph.nodesById).filter(
        (id) =>
          !this.webglRendererIoTracingService.curIoTracingData!.visibleNodeIds.has(
            id,
          ) && this.isNodeRendered(id),
      );
      this.nodeBodies.updateOpacity(nodeIds, 0.2);
      this.texts.updateOpacityInNode(nodeIds, 0.3);
      this.groupNodeIcons.updateOpacityInNode(nodeIds, 0.3);
      this.webglRendererAttrsTableService.attrsTableTexts.updateOpacityInNode(
        nodeIds,
        0.3,
      );
      this.webglRendererEdgeTextsService.edgeTexts.updateOpacityInNode(
        nodeIds,
        0.3,
      );

      const edgeIdsToDim = this.edgesToRender
        .filter(
          ({edge}) =>
            !this.webglRendererIoTracingService.curIoTracingData!.visibleNodeIds.has(
              edge.fromNodeId,
            ) ||
            !this.webglRendererIoTracingService.curIoTracingData!.visibleNodeIds.has(
              edge.toNodeId,
            ),
        )
        .map(({edge}) => edge.id);
      this.edges.updateColors(edgeIdsToDim, {r: 0.92, g: 0.92, b: 0.92});
    }
  }

  private shakeNode(nodeId: string) {
    if (this.appService.testMode) {
      return;
    }

    // Animate
    const startTs = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTs;
      let t = Math.min(1, elapsed / 1100);

      // ease in out sine.
      t = -(Math.cos(Math.PI * t) - 1) / 2;

      const angle =
        Math.sin(t * Math.PI * 9 /* Number of shakes */) *
        8; /* Max shake angle in degree */
      this.nodeBodies.updateAngle(nodeId, angle);
      this.webglRendererSearchResultsService.searchResultsHighlightBorders.updateAngle(
        nodeId,
        angle,
      );
      this.webglRendererThreejsService.render();

      if (t >= 1) {
        this.nodeBodies.updateAngle(nodeId, 0);
        this.webglRendererSearchResultsService.searchResultsHighlightBorders.updateAngle(
          nodeId,
          0,
        );
        this.webglRendererThreejsService.render();
        return;
      }

      requestAnimationFrame(animate);
    };
    animate();
  }

  private clearScene(extraMeshesToSkip: three.Object3D[] = []) {
    // Remove all meshes from the scene and dispose their geometries and
    // materials.
    //
    // Self-managed meshes are the ones that are managed in their own functions.
    // For example, searchResultsHighlightBorders.mesh is removed from the scene
    // in clearSearchResults instead of here.
    const selfManagedMeshes = [
      this.webglRendererSearchResultsService.searchResultsHighlightBorders.mesh,
      this.webglRendererSearchResultsService.searchResultsNodeLabelHighlightBg
        .mesh,
      ...extraMeshesToSkip,
    ];
    this.webglRendererThreejsService.clearScene(selfManagedMeshes);

    this.updateNodesStylesSavedSelectedNodeId = '';
    this.updateNodesStylesSavedIoTracingData = undefined;
  }

  private async handleDownloadAsPng(
    fullGraph: boolean,
    transparentBackground: boolean,
  ) {
    let modelGraphWidth =
      this.container.nativeElement.offsetWidth / getHighQualityPixelRatio();
    let modelGraphHeight =
      this.container.nativeElement.offsetHeight / getHighQualityPixelRatio();
    let curCamera = this.webglRendererThreejsService.camera;

    if (fullGraph) {
      // Get model graph size with some padding.
      const padding = 20;
      const maxX = this.currentMaxX + padding;
      const minX = this.currentMinX - padding;
      const maxZ = this.currentMaxZ + padding;
      const minZ = this.currentMinZ - padding;
      modelGraphWidth = maxX - minX;
      modelGraphHeight = maxZ - minZ;

      const maxSize = MAX_PNG_SIZE / getHighQualityPixelRatio();
      if (modelGraphWidth > maxSize) {
        modelGraphHeight = (modelGraphHeight * maxSize) / modelGraphWidth;
        modelGraphWidth = maxSize;
      }
      if (modelGraphHeight > maxSize) {
        modelGraphWidth = (modelGraphWidth * maxSize) / modelGraphHeight;
        modelGraphHeight = maxSize;
      }

      // Create a camera used for rendering full graph for downloading.
      const camera = this.webglRendererThreejsService.createOrthographicCamera(
        minX,
        maxX,
        -minZ,
        -maxZ,
      );

      curCamera = camera;
    }

    // Render.
    const canvas = this.pngDownloaderCanvas.nativeElement;
    this.webglRendererThreejsService.setupPngDownloaderRenderer(
      canvas,
      transparentBackground,
      modelGraphWidth,
      modelGraphHeight,
    );
    // Don't render the "color blocks" on nodes when zooming out far.
    this.nodeBodies.setBgColorWhenFar(this.NODE_LABEL_COLOR, 0);
    this.webglRendererThreejsService.renderPngDownloader(curCamera);
    this.nodeBodies.setBgColorWhenFar(
      this.NODE_LABEL_COLOR,
      this.savedUpdateNodeBgWhenFarProgress / 3,
    );

    // Download canvas data as png.
    const link = document.createElement('a');
    link.download = 'model_explorer_graph.png';
    setAnchorHref(link, canvas.toDataURL());
    link.click();
    this.webglRendererThreejsService.setSceneBackground(
      new THREE.Color(0xffffff),
    );
  }

  private async openSubgraph(subgraphId: string) {
    const graph = this.appService.getGraphById(subgraphId);
    if (!graph) {
      const msg = `No graph found for subgraph id: "${subgraphId}"`;
      console.warn(msg);
      this.snackBar.open(msg, 'Dismiss');
      return;
    }

    // Add breadcrumb.
    this.appService.addSubgraphBreadcrumbItem(
      this.paneId,
      this.curModelGraph.id,
      subgraphId,
      await this.webglRendererSnapshotService.takeSnapshot(),
    );

    // Open the subgraph in current pane.
    this.appService.selectNode(this.paneId, undefined);
    this.appService.setFlattenLayersInCurrentPane(false);
    this.appService.curInitialUiState.set(undefined);
    this.appService.curToLocateNodeInfo.set(undefined);
    this.appService.selectGraphInCurrentPane(graph);
  }

  private getGroupNodeLabelSeparatorId(
    nodeId: string,
    side: 'left' | 'right',
  ): string {
    return `${nodeId}_${side}`;
  }

  private getGroupNodeBgColor(groupNode: GroupNode): WebglColor {
    const ns = groupNode.namespace || '';
    const level = ns.split('/').filter((part) => part !== '').length;
    const color =
      this.GROUP_NODE_BG_COLORS[
        Math.min(this.GROUP_NODE_BG_COLORS.length - 1, level)
      ];
    return this.threeColorToRgb(color);
  }

  private threeColorToRgb(color: three.Color): WebglColor {
    return {r: color.r, g: color.g, b: color.b};
  }

  private startBenchmark() {
    const step = () => {
      this.webglRendererThreejsService.render(true);
      requestAnimationFrame(step);
    };
    step();
  }

  private handleShiftSelectNode(nodeId: string) {
    if (!this.webglRendererSubgraphSelectionService.enableSubgraphSelection) {
      return;
    }

    this.subgraphSelectionService.toggleNode(nodeId);
  }

  private handleClearSubgraphSelectedNodes() {
    if (!this.webglRendererSubgraphSelectionService.enableSubgraphSelection) {
      return;
    }

    this.subgraphSelectionService.clearSelection();
  }

  private createOverlay(ele: HTMLElement): OverlayRef {
    const config = new OverlayConfig({
      positionStrategy: this.overlay
        .position()
        .flexibleConnectedTo(ele)
        .withPositions([
          {
            originX: 'start',
            originY: 'bottom',
            overlayX: 'start',
            overlayY: 'top',
          },
          {
            originX: 'start',
            originY: 'top',
            overlayX: 'start',
            overlayY: 'bottom',
          },
        ])
        .withDefaultOffsetX(ele.clientWidth)
        .withViewportMargin(20),
      hasBackdrop: true,
      backdropClass: 'cdk-overlay-transparent-backdrop',
      maxHeight: '400px',
      panelClass: 'io-tree-popup-container',
    });
    const overlayRef = this.overlay.create(config);
    this.portal = new ComponentPortal(IoTree, this.viewContainerRef);
    overlayRef.backdropClick().subscribe(() => {
      overlayRef.dispose();
    });
    return overlayRef;
  }

  private showBusySpinnerWithDelay() {
    this.hideBusySpinner();

    this.showBusySpinnerTimeoutRef = window.setTimeout(() => {
      if (this.showBusySpinnerTimeoutRef < 0) {
        return;
      }
      this.snackBar.open('Processing. Please wait...');
      this.showBusySpinner = true;
      this.changeDetectorRef.detectChanges();
    }, 1000);
  }

  private hideBusySpinner() {
    if (this.showBusySpinnerTimeoutRef >= 0) {
      clearTimeout(this.showBusySpinnerTimeoutRef);
      this.showBusySpinnerTimeoutRef = -1;
    }
    this.snackBar.dismiss();
    this.showBusySpinner = false;
    this.changeDetectorRef.detectChanges();
  }

  private revealNode(nodeId: string, triggerNavigationSync = true): boolean {
    const node = this.curModelGraph.nodesById[nodeId];
    if (!node) {
      return false;
    }
    if (!this.isNodeRendered(nodeId)) {
      this.sendRelayoutGraphRequest(
        nodeId,
        node.nsParentId ? [node.nsParentId] : [],
        false,
        undefined,
        false,
        undefined,
        false,
        triggerNavigationSync,
      );
    } else {
      this.webglRendererThreejsService.zoomFitOnNode(
        nodeId,
        this.curModelGraph,
        ZOOM_FIT_ON_NODE_DURATION,
      );
      this.handleSelectNode(nodeId, triggerNavigationSync);
    }
    return true;
  }

  private updateCurModelGraph(modelGraph: ModelGraph) {
    const edgesByGroupNodeIds = this.curModelGraph.edgesByGroupNodeIds;
    this.curModelGraph = {
      ...modelGraph,
      edgesByGroupNodeIds: {
        ...edgesByGroupNodeIds,
        ...modelGraph.edgesByGroupNodeIds,
      },
    };
    this.appService.updateCurrentModelGraph(this.paneId, this.curModelGraph);
  }

  private updateNodesAndEdgesToRender() {
    if (!this.curModelGraph) {
      return;
    }

    // Collect node ids.
    this.elementsToRender = [];
    this.nodesToRender = [];
    this.nodesToRenderMap = {};
    this.edgesToRender = [];

    // Add elements to render
    let firstIteration = true;
    let hasArtificialLayers = false;
    const visitNode = (parentNodeId: string | undefined) => {
      const parentNode = this.curModelGraph.nodesById[
        parentNodeId || ''
      ] as GroupNode;

      // Add the root node (the node passed-in in the first iteration) if it
      // exists.
      if (firstIteration && parentNode) {
        this.elementsToRender.push({
          type: RenderElementType.NODE,
          id: parentNode.id,
          node: parentNode,
        });
        const nodeToRender = {
          node: parentNode,
          index: this.elementsToRender.length - 1,
        };
        this.nodesToRender.push(nodeToRender);
        this.nodesToRenderMap[nodeToRender.node.id] = nodeToRender;
        if (isGroupNode(parentNode) && parentNode.sectionContainer) {
          hasArtificialLayers = true;
        }
      }
      firstIteration = false;

      // Add edges in the current layer.
      if (
        (parentNodeId && parentNode && parentNode.expanded) ||
        !parentNodeId
      ) {
        for (const edge of this.curModelGraph.edgesByGroupNodeIds[
          parentNodeId || ''
        ] || []) {
          this.elementsToRender.push({
            type: RenderElementType.EDGE,
            id: edge.id,
            edge,
          });
          this.edgesToRender.push({
            edge,
            index: this.elementsToRender.length - 1,
          });
        }
      }

      // Get its ns children nodes in the current layer.
      let nodes: ModelNode[] = [];
      if (!parentNodeId) {
        nodes = this.curModelGraph.rootNodes;
      } else {
        if (parentNode.expanded) {
          nodes = (parentNode.nsChildrenIds || []).map(
            (nodeId) => this.curModelGraph.nodesById[nodeId],
          );
        }
      }

      // For each ns child, add itself, and recursively add the edges and nodes
      // inside.
      for (const childNode of nodes) {
        const renderElementNode: RenderElementNode = {
          type: RenderElementType.NODE,
          id: childNode.id,
          node: childNode,
        };
        if (
          !hasArtificialLayers &&
          isGroupNode(childNode) &&
          childNode.sectionContainer
        ) {
          hasArtificialLayers = true;
        }
        this.elementsToRender.push(renderElementNode);
        const nodeToRender = {
          node: childNode,
          index: this.elementsToRender.length - 1,
        };
        this.nodesToRender.push(nodeToRender);
        this.nodesToRenderMap[nodeToRender.node.id] = nodeToRender;

        if (isGroupNode(childNode) && childNode.expanded) {
          visitNode(childNode.id);
        }
      }
    };
    visitNode(this.rootNodeId);
    this.appService.setPaneHasArtificialLayers(
      this.paneId,
      hasArtificialLayers,
    );
  }

  private getNodeLabel(node: ModelNode): string {
    if (isOpNode(node)) {
      // Special handling for placeholders.
      if (node.label === 'Placeholder') {
        return node.id;
      }
      return node.label;
    } else if (isGroupNode(node)) {
      return node.label;
    }
    return '-';
  }

  private setHoveredNodeId(id: string) {
    this.hoveredNodeId = id;
    this.appService.updateHoveredNode(
      id,
      this.curModelGraph.id,
      this.curModelGraph.collectionLabel || '',
      this.curModelGraph.nodesById[id],
    );
  }

  private revealAndHighlightNodes(
    nodeIds: string[],
    nodeIdToSelect: string,
    processShowNoMappedNodeMessage: boolean,
  ) {
    // Things to do for the nodes.
    const processNodesFn = () => {
      // Zoom the group to fit the nodes.
      this.webglRendererThreejsService.zoomFitOnNodes(
        nodeIds,
        this.curModelGraph,
        ZOOM_FIT_ON_NODE_DURATION,
      );
      // Select the first node in the list.
      if (nodeIdToSelect) {
        this.appService.selectNode(this.paneId, {
          nodeId: nodeIdToSelect,
          rendererId: this.rendererId,
          isGroupNode: isGroupNode(
            this.curModelGraph.nodesById[nodeIdToSelect],
          ),
          triggerNavigationSync: false,
        });
      }
      // Highlight nodes.
      this.syncNavigationRelatedNodesHighlights.setNodeHighlights(
        nodeIds.reduce(
          (acc, nodeId) => {
            acc[nodeId] = {
              nodeId,
              borderColor:
                this.syncNavigationService.getSyncNavigationData()
                  ?.relatedNodesBorderColor ??
                DEFAULT_HIGHLIGHT_NODES_BORDER_COLOR,
              borderWidth:
                this.syncNavigationService.getSyncNavigationData()
                  ?.relatedNodesBorderWidth ??
                DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH,
            };
            return acc;
          },
          {} as {[nodeId: string]: HighlightInfo},
        ),
      );
    };
    // Calculate the deepest expanded group node ids.
    const deepestExpandedGroupNodeIds: string[] =
      this.getDeepestExpandedGroupNodeIdsForNodes(nodeIds);
    // Reveal them if the ids are non-empty.
    if (deepestExpandedGroupNodeIds.length > 0) {
      // Set up the callback function after relayout is done.
      this.relayoutDoneFn = processNodesFn;
      // Reveal them and set the first node in the list as selected.
      this.sendRelayoutGraphRequest('', deepestExpandedGroupNodeIds);
      if (processShowNoMappedNodeMessage) {
        this.syncNavigationService.setShowNoMappedNodeMessage(false);
      }
    }
    // No need to reveal them if the ids are empty. Just process them
    // directly.
    else if (nodeIds.length > 0 && deepestExpandedGroupNodeIds.length === 0) {
      processNodesFn();
      if (processShowNoMappedNodeMessage) {
        this.syncNavigationService.setShowNoMappedNodeMessage(false);
      }
    } else {
      if (processShowNoMappedNodeMessage) {
        this.syncNavigationService.setShowNoMappedNodeMessage(true);
      }
    }
  }

  private getDeepestExpandedGroupNodeIdsForNodes(nodeIds: string[]): string[] {
    const deepestExpandedGroupNodeIdsSet = new Set<string>();
    for (const nodeId of nodeIds) {
      const node = this.curModelGraph.nodesById[nodeId];
      if (isOpNode(node) && node.hideInLayout) {
        continue;
      }
      if (node?.nsParentId) {
        const parentNode = this.curModelGraph.nodesById[
          node.nsParentId
        ] as GroupNode;
        if (!parentNode.expanded || !this.isNodeRendered(parentNode.id)) {
          deepestExpandedGroupNodeIdsSet.add(node.nsParentId);
        }
      }
    }
    return [...deepestExpandedGroupNodeIdsSet];
  }

  private clickSubgraph(subgraphId: string, event: MouseEvent) {
    if (!event.altKey) {
      this.openSubgraph(subgraphId);
    }
    // Alt-clicking opens the subgraph in a split pane.
    else {
      const subgraph = this.appService.getGraphById(subgraphId);
      if (subgraph) {
        const openToLeft = this.appService.getIsGraphInRightPane(
          this.curModelGraph.id,
        );
        this.appService.openGraphInSplitPane(subgraph, false, true, openToLeft);
      }
    }
  }

  /**
   * Renders diff highlights on nodes based on the current sync navigation mode.
   *
   * This function checks if the current sync navigation mode is enabled. If it
   * is, it iterates through all nodes in the current model graph. For each
   * node, it checks if it's rendered and if it has mapped nodes in the other
   * pane. If all mapped nodes are missing in the other pane, it highlights the
   * current node with a specific border color and width to indicate a diff.
   */
  private renderDiffHighlights() {
    const paneIndex = this.paneIndex();
    const curHighlights: Record<string, HighlightInfo> = {};
    const curSyncNavigationData =
      this.syncNavigationService.getSyncNavigationData();
    const showDiffHighlights =
      curSyncNavigationData?.showDiffHighlights ||
      this.syncNavigationService.getShowDiffHighlightsInMatchNodeIdMode();
    if (
      this.paneCount() === 2 &&
      showDiffHighlights &&
      this.syncNavigationService.mode() !== SyncNavigationMode.DISABLED
    ) {
      for (const node of this.curModelGraph.nodes) {
        if (!this.isNodeRendered(node.id)) {
          continue;
        }
        const mappedNodeIds = this.syncNavigationService.getMappedNodeIds(
          paneIndex,
          node.id,
        );
        const otherGraph =
          this.appService.panes()[paneIndex === 0 ? 1 : 0].modelGraph;
        if (otherGraph) {
          let allMappedNodesMissing = true;
          // Check if any of the mapped nodes exists in the other graph.
          for (const mappedNodeId of mappedNodeIds) {
            if (otherGraph.nodesById[mappedNodeId]) {
              allMappedNodesMissing = false;
              break;
            }
          }
          // If all mapped nodes are missing, highlight the current node.
          if (allMappedNodesMissing) {
            curHighlights[node.id] = {
              nodeId: node.id,
              borderWidth:
                paneIndex === 0
                  ? curSyncNavigationData?.deletedNodesBorderWidth ??
                    DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH
                  : curSyncNavigationData?.newNodesBorderWidth ??
                    DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH,
              borderColor:
                paneIndex === 0
                  ? curSyncNavigationData?.deletedNodesBorderColor ??
                    DEFAULT_DELETE_NODES_BORDER_COLOR
                  : curSyncNavigationData?.newNodesBorderColor ??
                    DEFAULT_NEW_NODES_BORDER_COLOR,
            };
          }
        }
      }
    }
    this.syncNavigationDiffHighlights.setNodeHighlights(curHighlights, true);
  }
}
