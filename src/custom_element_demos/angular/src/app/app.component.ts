/**
 * @license
 * Copyright 2025 The Model Explorer Authors. All Rights Reserved.
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

import {
  AfterViewInit,
  Component,
  ElementRef,
  signal,
  viewChild,
} from '@angular/core';
import { graphCollections } from './graph_collections';
import 'ai-edge-model-explorer-visualizer';
import {
  ModelExplorerVisualizer,
  NodeInfo,
  NodeQueryType,
  NodeStyleId,
  SearchMatchType,
  VisualizerConfig,
  VisualizerUiState,
} from 'ai-edge-model-explorer-visualizer';
import { CommonModule } from '@angular/common';

enum ActionOnInit {
  HideNodes = 'hide_nodes',
  HidePanels = 'hide_panels',
  SetNodeStyler = 'set_node_styler',
  RestoreUiState = 'restore_ui_state',
  Reset = 'reset',
}

interface ActionOnInitItem {
  action: ActionOnInit;
  label: string;
}

const ACTION_ON_INIT_PARAM_KEY = 'action_on_init';

/**
 * This app shows a panel at left and the Model Explorer Visualizer at center.
 * The left-side panel has three sections to demonstrate how some of the
 * visualizer API works.
 *
 * - Properties section:
 *
 *   Clicking a link in this section will refresh the page with
 *   "?action_on_init=some_action" url parameter added. When the page is loaded,
 *   the contructor below will check the value of the parameter, and set up
 *   certain properties to be set to visualizer later.
 *
 * - Events section:
 *
 *   This section tracks the ids of the selected node, the hovered node, and the
 *   double-clicked node. The event listeners are hooked up in the
 *   `ngAfterViewInit` method below.
 *
 * - Methods section:
 *
 *   Clicking a link in this section will call certain API method on the
 *   visualizer. See the `handleSelectNode` and `handleAddNodeData` below.
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements AfterViewInit {
  main = viewChild<ElementRef<HTMLElement>>('main');

  readonly ActionOnInit = ActionOnInit;
  readonly actions: ActionOnInitItem[] = [
    {
      action: ActionOnInit.HideNodes,
      label: 'Hide "InlandDrive" nodes on init',
    },
    {
      action: ActionOnInit.HidePanels,
      label: 'Hide surrounding panels on init',
    },
    {
      action: ActionOnInit.SetNodeStyler,
      label: 'Set node styler rules on init',
    },
    {
      action: ActionOnInit.RestoreUiState,
      label: 'Restore UI state on init',
    },
    {
      action: ActionOnInit.Reset,
      label: 'Reset',
    },
  ];

  readonly selectedNodeId = signal<string>('-');
  readonly hoveredNodeId = signal<string>('-');
  readonly doubleClickNodeId = signal<string>('-');

  curActionOnInit?: string;

  private visualizer?: ModelExplorerVisualizer;
  private visualizerConfig?: VisualizerConfig;
  private uiState?: VisualizerUiState;
  private nodeDataIndex = 0;

  constructor() {
    // Process the 'action_on_init' (ACTION_ON_INIT_PARAM_KEY) query parameter
    // to configure visualizer properties.
    //
    // This parameter is set when clicking links in the "Properties" section.
    const urlParams = new URLSearchParams(window.location.search);
    const action = urlParams.get(ACTION_ON_INIT_PARAM_KEY) ?? '';
    this.curActionOnInit = action;

    switch (action) {
      case ActionOnInit.HideNodes:
        this.visualizerConfig = {
          nodeLabelsToHide: ['Salt lake city', 'Las Vegas'],

          // This is only here to clear node styler rules (which will be
          // persisted in local storage) that was (possibly) set in the
          // SetNodeStyler action below.
          nodeStylerRules: [],
        };
        break;
      case ActionOnInit.HidePanels:
        this.visualizerConfig = {
          hideInfoPanel: true,
          hideLegends: true,
          hideToolBar: true,
          hideTitleBar: true,

          // This is only here to clear node styler rules (which will be
          // persisted in local storage) that was (possibly) set in the
          // SetNodeStyler action below.
          nodeStylerRules: [],
        };
        break;
      case ActionOnInit.SetNodeStyler:
        this.visualizerConfig = {
          nodeStylerRules: [
            // Use "Vancouver" regex to match node label and color its
            // backgroud green and foreground white.
            {
              queries: [
                {
                  type: NodeQueryType.REGEX,
                  queryRegex: 'Vancouver',
                  matchTypes: [SearchMatchType.NODE_LABEL],
                },
              ],
              styles: {
                [NodeStyleId.NODE_BG_COLOR]: 'green',
                [NodeStyleId.NODE_TEXT_COLOR]: 'white',
              },
            },
            // Use "Angeles" regex to match node label and color its
            // backgroud red and foreground white.
            {
              queries: [
                {
                  type: NodeQueryType.REGEX,
                  queryRegex: 'Angeles',
                  matchTypes: [SearchMatchType.NODE_LABEL],
                },
              ],
              styles: {
                [NodeStyleId.NODE_BG_COLOR]: 'red',
                [NodeStyleId.NODE_TEXT_COLOR]: 'white',
              },
            },
          ],
        };
        break;
      case ActionOnInit.RestoreUiState:
        this.uiState = {
          paneStates: [
            {
              deepestExpandedGroupNodeIds: [],
              selectedNodeId: 'las_vegas',
              selectedGraphId: 'road_trip',
              selectedCollectionLabel: 'my collection',
              widthFraction: 0.5,
            },
            {
              deepestExpandedGroupNodeIds: [],
              selectedNodeId: 'sf_golden_gate_bridge',
              selectedGraphId: 'road_trip',
              selectedCollectionLabel: 'my collection',
              widthFraction: 0.5,
            },
          ],
        };
        this.visualizerConfig = {
          // This is only here to clear node styler rules (which will be
          // persisted in local storage) that was (possibly) set in the
          // SetNodeStyler action above.
          nodeStylerRules: [],
        };
        break;
      default:
        this.visualizerConfig = {
          // This is only here to clear node styler rules (which will be
          // persisted in local storage) that was (possibly) set in the
          // SetNodeStyler action below.
          nodeStylerRules: [],
        };
        break;
    }
  }

  ngAfterViewInit() {
    const mainEle = this.main()?.nativeElement;
    if (!mainEle) {
      return;
    }

    // Create the custom element.
    const visualizer = document.createElement('model-explorer-visualizer');
    this.visualizer = visualizer;

    // Set graph collections to be visualized.
    visualizer.graphCollections = graphCollections;

    // Set visualizer config if available.
    if (this.visualizerConfig) {
      visualizer.config = this.visualizerConfig;
    }

    // Set initial ui state if available.
    if (this.uiState) {
      visualizer.initialUiState = this.uiState;
    }

    // Add to DOM.
    mainEle.appendChild(visualizer);

    // Listen to selected node changes.
    visualizer.addEventListener('selectedNodeChanged', (e) => {
      const customEvent = e as CustomEvent<NodeInfo>;
      let nodeId = customEvent.detail?.nodeId ?? '';
      if (nodeId === '') {
        nodeId = '-';
      }
      this.selectedNodeId.set(nodeId);
    });

    // Listen to hovered node changes.
    visualizer.addEventListener('hoveredNodeChanged', (e) => {
      const customEvent = e as CustomEvent<NodeInfo>;
      let nodeId = customEvent.detail?.nodeId ?? '';
      if (nodeId === '') {
        nodeId = '-';
      }
      this.hoveredNodeId.set(nodeId);
    });

    // Listen to double-clicked node changes.
    visualizer.addEventListener('doubleClickedNodeChanged', (e) => {
      const customEvent = e as CustomEvent<NodeInfo>;
      let nodeId = customEvent.detail?.nodeId ?? '';
      if (nodeId === '') {
        nodeId = '-';
      }
      this.doubleClickNodeId.set(nodeId);
    });
  }

  /**
   * Called when any link under the "Properties" section is clicked.
   */
  refreshPageWithInitAction(action: ActionOnInit) {
    const url = new URL(window.location.href);
    const baseUrl = url.origin + url.pathname;
    if (action === ActionOnInit.Reset) {
      window.location.assign(baseUrl);
      return;
    }
    const newUrl = `${baseUrl}?${ACTION_ON_INIT_PARAM_KEY}=${action}`;
    window.location.assign(newUrl);
  }

  /**
   * Called when `Select "Golden gate bridge"` is clicked.
   */
  handleSelectNode() {
    this.visualizer?.selectNode(
      'sf_golden_gate_bridge',
      graphCollections[0].graphs[0].id
    );
  }

  /**
   * Called when `Add "temperature" node data` is clicked.
   */
  handleAddNodeData() {
    const genRandomTemperature = () => Math.floor(40 + Math.random() * 40);
    this.visualizer?.addNodeDataProviderData(
      `temperature-${this.nodeDataIndex++}`,
      {
        results: {
          vancouver: {
            value: genRandomTemperature(),
          },
          salt_lake_city: {
            value: genRandomTemperature(),
          },
          las_vegas: {
            value: genRandomTemperature(),
          },
          seattle: {
            value: genRandomTemperature(),
          },
          sf_golden_gate_bridge: {
            value: genRandomTemperature(),
          },
          sf_pier_39: {
            value: genRandomTemperature(),
          },
          la: {
            value: genRandomTemperature(),
          },
        },
        gradient: [
          {
            stop: 0,
            bgColor: 'blue',
          },
          {
            stop: 1,
            bgColor: 'red',
          },
        ],
      }
    );
  }
}
