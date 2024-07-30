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

/**
 * The UI state of the visualizer.
 */
export declare interface VisualizerUiState {
  paneStates: PaneState[];
}

/** The UI state for a pane. */
export declare interface PaneState {
  /**
   * The ids of expanded group nodes that none of their child group nodes is
   * expanded.
   */
  deepestExpandedGroupNodeIds: string[];

  /** Id of the node currently selected. */
  selectedNodeId: string;

  /** Id of the selected graph. */
  selectedGraphId: string;

  /** Label of the collection that the graph belongs to. */
  selectedCollectionLabel: string;

  /** Width fraction. */
  widthFraction: number;

  /** Whether the pane is selected or not. */
  selected?: boolean;

  /** Whether to flatten all layers in the graph. */
  flattenLayers?: boolean;
}
