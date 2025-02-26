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

import {CommonModule} from '@angular/common';
import {
  Component,
  ElementRef,
  Input,
  NgZone,
  ViewEncapsulation,
  input,
  output,
  viewChild,
} from '@angular/core';
import {ModelGraphVisualizer} from '../components/visualizer/model_graph_visualizer';
import {
  GraphCollection,
  ModelExplorerGlobal,
  ModelGraphProcessedEvent,
  NodeDataProviderData,
  NodeDataProviderGraphData,
  NodeInfo,
  VisualizerConfig,
  VisualizerUiState,
} from './index';

const MATERIAL_ICONS_CSS =
  'https://fonts.googleapis.com/icon?family=Material+Icons';

declare global {
  interface Window {
    modelExplorer: ModelExplorerGlobal;
  }
}

/**
 * A wrapper component for the model explorer visualizer.
 */
@Component({
  selector: 'custom-element-wrapper',
  standalone: true,
  imports: [CommonModule, ModelGraphVisualizer],
  templateUrl: './wrapper.ng.html',
  styleUrls: ['./wrapper.scss'],
  encapsulation: ViewEncapsulation.ShadowDom,
})
export class Wrapper {
  //////////////////////////////////////////////////////////////////////////////
  // Inputs
  //
  // Set them before adding the element to the DOM. Example:
  //
  // const visualizer = document.createElement('model-explorer-visualizer');
  // visualizer.graphCollections = [...];
  // document.body.appendChild(visualizer);

  /**
   * The graph collections to visualize.
   *
   * Each graph collection is a collection of graphs.
   */
  graphCollections = input<GraphCollection[]>([]);

  /** Configs for the visualizer. */
  config = input<VisualizerConfig>();

  /**
   * The UI state (e.g. selected node, expanded layers, etc.) to restore on
   * init.
   */
  initialUiState = input<VisualizerUiState>();

  /**
   * Whether to run in a special benchmark mode.
   *
   * When set to true, the visualizer will run in a special benchmark mode
   * where you can specify number of nodes and edges to render with a FPS
   * counter. All other fields above will be ignored.
   */
  benchmark = input<boolean>(false);

  //////////////////////////////////////////////////////////////////////////////
  // Outputs
  //
  // They are emitted as custom events on the element. Example:
  //
  // visualizer.addEventListener('selectedNodeChanged', (e: Event) => {
  //   const customEvent = e as CustomEvent<NodeInfo>;
  //   console.log(customEvent.detail?.nodeId);
  // });

  /** Triggered when the title is clicked. */
  titleClicked = output<void>();

  /**
   * Triggered when UI state (e.g. selected node, expanded layers, etc.)
   * changes.
   *
   * You can use the state emitted here to restore the UI state on init using
   * the `initialUiState` input above.
   */
  uiStateChanged = output<VisualizerUiState>();

  /** Triggered when the default graph has been processed.  */
  modelGraphProcessed = output<ModelGraphProcessedEvent>();

  /** Triggered when the selected node is changed. */
  selectedNodeChanged = output<NodeInfo>();

  /** Triggered when the hovered node is changed. */
  hoveredNodeChanged = output<NodeInfo>();

  /** Triggered when the double clicked node is changed. */
  doubleClickedNodeChanged = output<NodeInfo>();

  visualizer = viewChild(ModelGraphVisualizer);

  constructor(
    private readonly el: ElementRef<HTMLElement>,
    private readonly ngZone: NgZone,
  ) {
    const assetFilesBaseUrl: string =
      window.modelExplorer.assetFilesBaseUrl ?? 'static_files';

    // Inject material icons font.
    this.el.nativeElement.shadowRoot!.appendChild(
      createStyleElement(MATERIAL_ICONS_CSS),
    );
    document.head.appendChild(createStyleElement(MATERIAL_ICONS_CSS));

    // Inject material styles.
    const stylesCssUrl = `${assetFilesBaseUrl}/styles.css`;
    this.el.nativeElement.shadowRoot!.appendChild(
      createStyleElement(stylesCssUrl),
    );
    document.head.appendChild(createStyleElement(stylesCssUrl));
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
  @Input()
  selectNode = (
    nodeId: string,
    graphId: string,
    collectionLabel?: string,
    paneIndex = 0,
  ) => {
    this.ngZone.run(() => {
      this.visualizer()?.selectNode(
        nodeId,
        graphId,
        collectionLabel,
        paneIndex,
      );
    });
  };

  /**
   * Adds data for node data provider.
   *
   * This only works after the model graph is processed. Call it when
   * `modelGraphProcessed` event above is emitted.
   *
   * @param name the name of the data to add.
   * @param data the data to add.
   * @param paneIndex the index of the pane to add data for.
   * @param clearExisting whether to clear existing data before adding new one.
   */
  @Input()
  addNodeDataProviderData = (
    name: string,
    data: NodeDataProviderGraphData,
    paneIndex = 0,
    clearExisting = false,
  ) => {
    this.ngZone.run(() => {
      this.visualizer()?.addNodeDataProviderData(
        name,
        data,
        paneIndex,
        clearExisting,
      );
    });
  };

  /**
   * Adds data with graph index for node data provider.
   *
   * This only works after the model graph is processed. Call it when
   * `modelGraphProcessed` event aboce is emitted.
   *
   * @param name the name of the data to add.
   * @param data the data to add.
   * @param paneIndex the index of the pane to add data for.
   * @param clearExisting whether to clear existing data before adding new one.
   */
  @Input()
  addNodeDataProviderDataWithGraphIndex = (
    name: string,
    data: NodeDataProviderData,
    paneIndex = 0,
    clearExisting = false,
  ) => {
    this.ngZone.run(() => {
      this.visualizer()?.addNodeDataProviderDataWithGraphIndex(
        name,
        data,
        paneIndex,
        clearExisting,
      );
    });
  };
}

function createStyleElement(url: string): HTMLLinkElement {
  const link = document.createElement('link');
  link.href = url;
  link.rel = 'stylesheet';
  link.type = 'text/css';
  return link;
}
