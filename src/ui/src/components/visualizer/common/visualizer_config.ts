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

import {NodeStylerRule, RendererType} from './types';

/** Configs for the visualizer. */
export declare interface VisualizerConfig {
  /**
   * A list of labels. Nodes whose label matches any label in the list
   * (case-insensitive) will be hidden from the visualzation.
   */
  nodeLabelsToHide?: string[];

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

  /** Whether to enable export to resource. */
  enableExportToResource?: boolean;

  /** Whether to keep layers with a single child. */
  keepLayersWithASingleChild?: boolean;

  /**
   * Whether to show op node edges to other nodes out of the layer without
   * needing to select the node first.
   */
  showOpNodeOutOfLayerEdgesWithoutSelecting?: boolean;

  /** The default node styler rules. */
  nodeStylerRules?: NodeStylerRule[];

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
   * Whether to hide the legends.
   */
  hideLegends?: boolean;
}
