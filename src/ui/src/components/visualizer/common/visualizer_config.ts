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

import {EdgeOverlaysData} from './edge_overlays';
import {SyncNavigationData} from './sync_navigation';
import {
  LegendConfig,
  NodeStylerRule,
  RendererType,
  ToolbarConfig,
  ViewOnNodeConfig,
} from './types';

/** Configs for the visualizer. */
export declare interface VisualizerConfig {
  /**
   * A list of labels. Nodes whose label matches any label in the list
   * (case-insensitive) will be hidden from the visualzation.
   */
  nodeLabelsToHide?: string[];

  /**
   * Key-value pairs of node attribute key to value regex. Nodes whose
   * attribute value matches any regex in the dict will be hidden from the
   * visualzation.
   */
  nodeAttrsToHide?: Record<string, string>;

  /** The maximum number of child nodes under a layer node. */
  artificialLayerNodeCountThreshold?: number;

  /** The font size of the edge label. */
  edgeLabelFontSize?: number;

  /** The color of the edges. */
  edgeColor?: string;

  /** The maximum number of constant values to display. */
  maxConstValueCount?: number;

  /** Whether to disallow laying out edge labels vertically. */
  disallowVerticalEdgeLabels?: boolean;

  /** Whether to enable subgraph selection. */
  enableSubgraphSelection?: boolean;

  /**
   * Whether to enable the "export to resource" button in the
   * selection panel.
   */
  enableExportToResource?: boolean;

  /**
   * Whether to enable the "export selected nodes" button in the
   * selection panel.
   */
  enableExportSelectedNodes?: boolean;

  /**
   * The label to override the "export selected nodes" button in the
   * selection panel.
   */
  exportSelectedNodesButtonLabel?: string;

  /**
   * The icon to override the "export selected nodes" button in the
   * selection panel.
   */
  exportSelectedNodesButtonIcon?: string;

  /** Whether to keep layers with a single child. */
  keepLayersWithASingleChild?: boolean;

  /**
   * Whether to show op node edges to other nodes out of the layer without
   * needing to select the node first.
   */
  showOpNodeOutOfLayerEdgesWithoutSelecting?: boolean;

  /** Whether to highlight layer node inputs and outputs. */
  highlightLayerNodeInputsOutputs?: boolean;

  /** Whether to hide empty node data entries. */
  hideEmptyNodeDataEntries?: boolean;

  /** The default node styler rules. */
  nodeStylerRules?: NodeStylerRule[];

  /** The data for navigation syncing. */
  syncNavigationData?: SyncNavigationData;

  /** List of data for edge overlays that will be applied to the left pane. */
  edgeOverlaysDataListLeftPane?: EdgeOverlaysData[];

  /** List of data for edge overlays that will be applied to the right pane. */
  edgeOverlaysDataListRightPane?: EdgeOverlaysData[];

  /**
   * Default graph renderer.
   *
   * @deprecated This field is no longer used.
   */
  defaultRenderer?: RendererType;

  /**
   * Whether to hide the title bar.
   */
  hideTitleBar?: boolean;

  /**
   * Whether to hide the tool bar.
   */
  hideToolBar?: boolean;

  /**
   * Whether to hide the info panel.
   */
  hideInfoPanel?: boolean;

  /**
   * Whether to hide the node data in the info panel.
   * Node data can still be seen on the node overlays.
   * This only shows/hides the node data in the info panel.
   */
  hideNodeDataInInfoPanel?: boolean;

  /**
   * Whether to hide the legends.
   */
  hideLegends?: boolean;

  /**
   * A list of node info keys (regex) to hide from the "node info" section in
   * the side panel.
   */
  nodeInfoKeysToHide?: string[];

  /**
   * A list of metadata keys (regex) to hide from the "inputs" section in the
   * side panel.
   */
  inputMetadataKeysToHide?: string[];

  /**
   * A list of metadata keys (regex) to hide from the "outputs" section in the
   * side panel.
   */
  outputMetadataKeysToHide?: string[];

  /**
   * If set, rename the "op name" item in the node info to this string.
   */
  renameNodeInfoOpNameTo?: string;

  /**
   * If set, show the side panel only when a node is selected.
   */
  showSidePanelOnNodeSelection?: boolean;

  /**
   * If set, rename the node data provider panel title to this string.
   */
  renameNodeDataProviderPanelTitleTo?: string;

  /**
   * Config for the legends panel.
   */
  legendConfig?: LegendConfig;

  /**
   * Config for the toolbar.
   */
  viewOnNodeConfig?: ViewOnNodeConfig;

  /**
   * Config for the toolbar.
   */
  toolbarConfig?: ToolbarConfig;

  /**
   * If set, this factor will be applied to the zoom-fit scale on node.
   */
  extraZoomFactorOnNode?: number;
}
