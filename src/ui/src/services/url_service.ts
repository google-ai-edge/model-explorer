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

import {Injectable} from '@angular/core';
import {Params, Router} from '@angular/router';

import {SyncNavigationModeChangedEvent} from '../components/visualizer/common/types';
import {VisualizerUiState} from '../components/visualizer/common/visualizer_ui_state';

/** All URL query parameter keys. */
enum QueryParamKey {
  DATA = 'data',
  RENDERER = 'renderer',
  SHOW_OPEN_IN_NEW_TAB = 'show_open_in_new_tab',
  BENCHMARK = 'benchmark',
  ENABLE_SUBGRAPH_SELECTION = 'ess',
  ENABLE_EXPORT_TO_RESOURCE = 'eetr',
  ENABLE_EXPORT_SELECTED_NODES = 'eesn',
  EXPORT_SELECTED_NODES_BUTTON_LABEL = 'esnbl',
  EXPORT_SELECTED_NODES_BUTTON_ICON = 'esnbi',
  INTERNAL_COLAB = 'internal_colab',
  NODE_ATTRIBUTES_TO_HIDE = 'nath',
}

declare interface OldEncodedUrlData {
  modelUrl: string;
  tfVersion: string;
  deepestExpandedGroupNodeIds?: string[];
  selectedNodeId?: string;
  selectedGraphId?: string;
}

declare interface EncodedUrlData {
  models: ModelSource[];
  nodeData?: string[];
  sync?: SyncNavigationModeChangedEvent;
  // Target model names (e.g. model.tflite) that each of the `nodeData` above
  // is applied to.
  nodeDataTargets?: string[];
  uiState?: VisualizerUiState;
}

/** Model source encoded in url. */
export declare interface ModelSource {
  url: string;
  converterId?: string;
  adapterId?: string;
}

/**
 * A service to manage url (permalink).
 */
@Injectable({
  providedIn: 'root',
})
export class UrlService {
  private models: ModelSource[] = [];
  private nodeData?: string[] = [];
  private syncNavigation?: SyncNavigationModeChangedEvent;
  private nodeDataTargets?: string[] = [];
  private uiState?: VisualizerUiState;
  private prevQueryParamStr = '';

  renderer = 'webgl';
  showOpenInNewTab = false;
  internalColab = false;
  benchmark = false;
  enableSubgraphSelection = false;
  enableExportToResource = false;
  enableExportSelectedNodes = false;
  exportSelectedNodesButtonLabel = '';
  exportSelectedNodesButtonIcon = '';
  nodeAttributesToHide: Record<string, string> = {};

  constructor(private readonly router: Router) {
    this.decodeUrl();
  }

  setModels(models: ModelSource[]) {
    this.models = models;
    this.updateUrl();
  }

  getModels(): ModelSource[] {
    return this.models;
  }

  setUiState(uiState?: VisualizerUiState) {
    this.uiState = uiState;
    this.updateUrl();
  }

  getUiState(): VisualizerUiState | undefined {
    return this.uiState;
  }

  getNodeDataSources(): string[] {
    return this.nodeData || [];
  }

  setNodeDataSources(nodeDataSources: string[]) {
    this.nodeData = nodeDataSources;
    this.updateUrl();
  }

  getSyncNavigation(): SyncNavigationModeChangedEvent | undefined {
    return this.syncNavigation;
  }

  setSyncNavigation(syncNavigation: SyncNavigationModeChangedEvent) {
    this.syncNavigation = syncNavigation;
    this.updateUrl();
  }

  getNodeDataTargets(): string[] {
    return this.nodeDataTargets || [];
  }

  setNodeDataTargets(targets: string[]) {
    this.nodeDataTargets = targets;
    this.updateUrl();
  }

  private updateUrl() {
    const queryParams: Params = {};

    if (!this.benchmark) {
      const data: EncodedUrlData = {
        models: this.models,
        nodeData: this.nodeData,
        nodeDataTargets: this.nodeDataTargets,
        uiState: this.uiState,
        sync: this.syncNavigation,
      };
      queryParams[QueryParamKey.DATA] = JSON.stringify(data);
      queryParams[QueryParamKey.RENDERER] = this.renderer;
      queryParams[QueryParamKey.SHOW_OPEN_IN_NEW_TAB] = this.showOpenInNewTab
        ? '1'
        : '0';
      queryParams[QueryParamKey.INTERNAL_COLAB] = this.internalColab
        ? '1'
        : '0';
      queryParams[QueryParamKey.ENABLE_SUBGRAPH_SELECTION] = this
        .enableSubgraphSelection
        ? '1'
        : '0';
      queryParams[QueryParamKey.ENABLE_EXPORT_TO_RESOURCE] = this
        .enableExportToResource
        ? '1'
        : '0';
      queryParams[QueryParamKey.ENABLE_EXPORT_SELECTED_NODES] = this
        .enableExportSelectedNodes
        ? '1'
        : '0';
      if (this.exportSelectedNodesButtonLabel) {
        queryParams[QueryParamKey.EXPORT_SELECTED_NODES_BUTTON_LABEL] =
          this.exportSelectedNodesButtonLabel;
      }
      if (this.exportSelectedNodesButtonIcon) {
        queryParams[QueryParamKey.EXPORT_SELECTED_NODES_BUTTON_ICON] =
          this.exportSelectedNodesButtonIcon;
      }
      if (Object.keys(this.nodeAttributesToHide).length > 0) {
        queryParams[QueryParamKey.NODE_ATTRIBUTES_TO_HIDE] = JSON.stringify(
          this.nodeAttributesToHide,
        );
      }
    } else {
      queryParams[QueryParamKey.BENCHMARK] = '1';
    }

    // Dedup url update.
    const curQueryParamsStr = JSON.stringify(queryParams);
    if (curQueryParamsStr === this.prevQueryParamStr) {
      return;
    }
    this.prevQueryParamStr = curQueryParamsStr;

    // Update url.
    this.router.navigate([], {
      queryParams,
      // Use '' as the params handling method so that the whole query params
      // string will be replaced by the current content of 'queryParams'.
      queryParamsHandling: '',
      replaceUrl: false,
    });
  }

  private decodeUrl() {
    const params = new URLSearchParams(document.location.search);
    const data = params.get(QueryParamKey.DATA);
    if (data) {
      const json = JSON.parse(data);
      let decodedData = json as EncodedUrlData;

      // Check if the url is from the older version. If so, convert it to the
      // current version.
      const oldDecodedData = json as OldEncodedUrlData;
      if (oldDecodedData.modelUrl != null) {
        decodedData = {
          models: [{url: oldDecodedData.modelUrl}],
          uiState: {
            paneStates: [
              {
                deepestExpandedGroupNodeIds:
                  oldDecodedData.deepestExpandedGroupNodeIds || [],
                selectedNodeId: oldDecodedData.selectedNodeId || '',
                selectedGraphId: oldDecodedData.selectedGraphId || '',
                selectedCollectionLabel: '',
                widthFraction: 1,
              },
            ],
          },
        };
      }

      this.models = decodedData.models;
      this.uiState = decodedData.uiState;
      this.nodeData = decodedData.nodeData;
      this.syncNavigation = decodedData.sync;
      this.nodeDataTargets = decodedData.nodeDataTargets;
    }

    const renderer = params.get(QueryParamKey.RENDERER);
    this.renderer = renderer || 'webgl';
    this.showOpenInNewTab =
      params.get(QueryParamKey.SHOW_OPEN_IN_NEW_TAB) === '1';
    this.internalColab = params.get(QueryParamKey.INTERNAL_COLAB) === '1';
    this.enableSubgraphSelection =
      params.get(QueryParamKey.ENABLE_SUBGRAPH_SELECTION) === '1';
    this.enableExportToResource =
      params.get(QueryParamKey.ENABLE_EXPORT_TO_RESOURCE) === '1';
    this.enableExportSelectedNodes =
      params.get(QueryParamKey.ENABLE_EXPORT_SELECTED_NODES) === '1';
    this.exportSelectedNodesButtonLabel =
      params.get(QueryParamKey.EXPORT_SELECTED_NODES_BUTTON_LABEL) ?? '';
    this.exportSelectedNodesButtonIcon =
      params.get(QueryParamKey.EXPORT_SELECTED_NODES_BUTTON_ICON) ?? '';
    this.nodeAttributesToHide = JSON.parse(
      params.get(QueryParamKey.NODE_ATTRIBUTES_TO_HIDE) ?? '{}',
    ) as Record<string, string>;
    this.benchmark = params.get(QueryParamKey.BENCHMARK) === '1';
  }
}
