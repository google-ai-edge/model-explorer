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

import {effect, Injectable} from '@angular/core';
import * as three from 'three';

import {AppService} from './app_service';
import {
  NODE_LABEL_HEIGHT,
  NODE_LABEL_LINE_HEIGHT,
  WEBGL_ELEMENT_Y_FACTOR,
} from './common/consts';
import {ModelNode} from './common/model_graph';
import {FontWeight, SearchMatchType, SearchResults} from './common/types';
import {isGroupNode, splitLabel} from './common/utils';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';

const SEARCH_RESULTS_HIGHLIGHT_BORDER_Y_OFFSET = -WEBGL_ELEMENT_Y_FACTOR * 0.3;
const SEARCH_RESULTS_NODE_LABEL_HIGHLIGHT_Y_OFFSET =
  WEBGL_ELEMENT_Y_FACTOR * 0.3;

const THREE = three;

/**
 * Service for rendering search results.
 */
@Injectable()
export class WebglRendererSearchResultsService {
  readonly SEARCH_RESULTS_HIGHLIGHT_COLOR = new THREE.Color('#f5d55a');

  readonly searchResultsHighlightBorders = new WebglRoundedRectangles(8);
  readonly searchResultsNodeLabelHighlightBg = new WebglRoundedRectangles(4);

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private curSearchResults: SearchResults | undefined = undefined;

  constructor(private readonly appService: AppService) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;

    // Handle search results changes.
    effect(() => {
      const pane = this.appService.getPaneById(this.webglRenderer.paneId);
      if (!pane || !pane.modelGraph) {
        return;
      }
      if (this.curSearchResults === pane.searchResults) {
        return;
      }
      this.curSearchResults = pane.searchResults;
      this.renderSearchResults();
      this.webglRendererThreejsService.render();
    });
  }

  renderSearchResults() {
    if (!this.curSearchResults) {
      return;
    }
    this.clearSearchResults();
    // Find ids of nodes that are currently rendered/visible. For nodes that are
    // not rendered, find their closest rendered group node.
    const visibleNodeIds = new Set<string>();
    for (const nodeId of Object.keys(this.curSearchResults.results)) {
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
      if (!this.webglRenderer.isNodeRendered(nodeId)) {
        let curNode: ModelNode = node;
        while (curNode) {
          curNode =
            this.webglRenderer.curModelGraph.nodesById[
              curNode.nsParentId || ''
            ];
          if (!curNode || this.webglRenderer.isNodeRendered(curNode.id)) {
            break;
          }
        }
        visibleNodeIds.add(curNode.id);
      } else {
        visibleNodeIds.add(nodeId);
      }
    }
    // Render highlight borders for the result nodes.
    const rectangles: RoundedRectangleData[] = [];
    for (const nodeId of visibleNodeIds) {
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
      const nodeIndex = this.webglRenderer.nodesToRenderMap[nodeId].index;
      const x = this.webglRenderer.getNodeX(node) - 2;
      const y = this.webglRenderer.getNodeY(node) - 2;
      const width = this.webglRenderer.getNodeWidth(node) + 4;
      const height = this.webglRenderer.getNodeHeight(node) + 4;
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
          SEARCH_RESULTS_HIGHLIGHT_BORDER_Y_OFFSET,
        isRounded: true,
        borderColor: {r: 1, g: 1, b: 1},
        bgColor: this.SEARCH_RESULTS_HIGHLIGHT_COLOR,
        borderWidth: 0,
        opacity: 1,
      });
    }
    this.searchResultsHighlightBorders.generateMesh(rectangles);
    this.webglRendererThreejsService.addToScene(
      this.searchResultsHighlightBorders.mesh,
    );
    // Render bgs for matched node labels.
    const bgRectangles: RoundedRectangleData[] = [];
    const scale = NODE_LABEL_HEIGHT / this.webglRenderer.texts.getFontSize();
    for (const nodeId of Object.keys(this.curSearchResults.results)) {
      if (!this.webglRenderer.isNodeRendered(nodeId)) {
        continue;
      }
      const matches = this.curSearchResults.results[nodeId];
      for (const match of matches) {
        if (match.type === SearchMatchType.NODE_LABEL) {
          const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
          const nodeIndex = this.webglRenderer.nodesToRenderMap[nodeId].index;
          // Center.
          const x =
            this.webglRenderer.getNodeX(node) +
            this.webglRenderer.getNodeWidth(node) / 2;
          let y = 0;
          let height = 0;
          let width = 0;
          const lines = splitLabel(node.label);
          if (lines.length === 1) {
            const labelSizes = this.webglRenderer.texts.getLabelSizes(
              node.label,
              isGroupNode(node) ? FontWeight.BOLD : FontWeight.MEDIUM,
              NODE_LABEL_HEIGHT,
            ).sizes;
            width = (labelSizes.maxX - labelSizes.minX) * scale + 4;
            height = (labelSizes.maxZ - labelSizes.minZ) * scale + 4;
            y =
              this.webglRenderer.getNodeY(node) +
              this.webglRenderer.getNodeLabelRelativeY(node) -
              2 * scale;
          } else {
            const {minX, maxX} = this.webglRenderer.getNodeLabelSizes(node);
            width = (maxX - minX) * scale + 4;
            height = lines.length * NODE_LABEL_LINE_HEIGHT + 4;
            y =
              this.webglRenderer.getNodeY(node) + height / 2 + 4.5 - 2 * scale;
          }
          bgRectangles.push({
            id: nodeId,
            index: rectangles.length,
            bound: {
              x,
              y,
              width,
              height,
            },
            yOffset:
              WEBGL_ELEMENT_Y_FACTOR * nodeIndex +
              SEARCH_RESULTS_NODE_LABEL_HIGHLIGHT_Y_OFFSET,
            isRounded: true,
            borderColor: {r: 1, g: 1, b: 1},
            bgColor: this.SEARCH_RESULTS_HIGHLIGHT_COLOR,
            borderWidth: 0,
            opacity: 1,
          });
        }
      }
    }
    this.searchResultsNodeLabelHighlightBg.generateMesh(bgRectangles);
    this.webglRendererThreejsService.addToScene(
      this.searchResultsNodeLabelHighlightBg.mesh,
    );
    this.webglRenderer.animateIntoPositions((t) => {
      this.searchResultsHighlightBorders.updateAnimationProgress(t);
      this.searchResultsNodeLabelHighlightBg.updateAnimationProgress(t);
    });
  }

  private clearSearchResults() {
    for (const mesh of [
      this.searchResultsHighlightBorders.mesh,
      this.searchResultsNodeLabelHighlightBg.mesh,
    ]) {
      if (!mesh) {
        continue;
      }
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      this.webglRendererThreejsService.removeFromScene(mesh);
    }
  }
}
