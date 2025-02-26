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

import {GraphCollection} from '../components/visualizer/common/input_graph';
import {
  NodeDataProviderData,
  NodeDataProviderGraphData,
} from '../components/visualizer/common/types';
import {VisualizerConfig} from '../components/visualizer/common/visualizer_config';
import {VisualizerUiState} from '../components/visualizer/common/visualizer_ui_state';

/** Type for model explorer options exposed to global scope. */
export declare interface ModelExplorerGlobal {
  /**
   * The base url for all the asset files (e.g. styles, font texture)
   * required by the Model Explorer visualizer UI component.
   *
   * Don't include the trailing "/".
   *
   * By default, the asset files are served at 'static_files/*'.
   */
  assetFilesBaseUrl?: string;

  /**
   * The base url for the webworker script.
   *
   * Don't include the trailing "/".
   *
   * This should typically be a relative/absolute path on the host serving the
   * site, e.g. "/path/to/worker.js". By default, the worker script is served
   * at 'worker.js'.
   */
  workerScriptPath?: string;
}

// See comments in wrapper.ts for more details.
export declare interface ModelExplorerVisualizer {
  graphCollections?: GraphCollection[];
  config?: VisualizerConfig;
  initialUiState?: VisualizerUiState;
  benchmark?: boolean;

  selectNode: (
    nodeId: string,
    graphId: string,
    collectionLabel?: string,
    paneIndex?: number,
  ) => void;

  addNodeDataProviderData: (
    name: string,
    data: NodeDataProviderGraphData,
    paneIndex?: number,
    clearExisting?: boolean,
  ) => void;

  addNodeDataProviderDataWithGraphIndex: (
    name: string,
    data: NodeDataProviderData,
    paneIndex?: number,
    clearExisting?: boolean,
  ) => void;
}

type WithProperties<P> = {
  [property in keyof P]: P[property];
};

declare global {
  var modelExplorer: ModelExplorerGlobal;

  interface HTMLElementTagNameMap {
    'model-explorer-visualizer': HTMLElement &
      WithProperties<ModelExplorerVisualizer>;
  }
}

export * from '../components/visualizer/common/edge_overlays';
export * from '../components/visualizer/common/input_graph';
export * from '../components/visualizer/common/sync_navigation';
export * from '../components/visualizer/common/types';
export * from '../components/visualizer/common/visualizer_config';
export * from '../components/visualizer/common/visualizer_ui_state';
