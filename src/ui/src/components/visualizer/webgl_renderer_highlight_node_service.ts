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

import {effect, Injectable, signal} from '@angular/core';
import * as three from 'three';

import {WEBGL_ELEMENT_Y_FACTOR} from './common/consts';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';

const HIGHLIGHT_NODES_HIGHLIGHT_BORDER_Y_OFFSET = -WEBGL_ELEMENT_Y_FACTOR * 0.3;

const THREE = three;

const DEFAULT_HIGHLIGHT_NODES_BORDER_COLOR = '#ff00be';
const DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH = 2;

/**
 * Service for rendering node highlight (a border around the node).
 */
@Injectable()
export class WebglRendererHighlightNodesService {
  readonly highlightNodesBorders = new WebglRoundedRectangles(8);

  private readonly highlightNodeIds = signal<string[]>([]);
  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;

    // Handle highlight nodes changes.
    effect(() => {
      this.clearAndRenderHighlightNodes();
    });
  }

  setHighlightNodeIds(nodeIds: string[]) {
    this.highlightNodeIds.set(nodeIds);
  }

  private clearAndRenderHighlightNodes() {
    this.clearHighlightNodes();

    const nodeIdsToHighlight = this.highlightNodeIds();
    if (nodeIdsToHighlight.length > 0) {
      // Render highlight borders for the result nodes.
      const rectangles: RoundedRectangleData[] = [];
      const bgColor = new THREE.Color(this.borderColor);
      const borderWidth =
        this.webglRenderer.syncNavigationService.getSyncNavigationData()
          ?.relatedNodesBorderWidth ?? DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH;
      for (const nodeId of nodeIdsToHighlight) {
        const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
        const nodeIndex = this.webglRenderer.nodesToRenderMap[nodeId].index;
        const x = this.webglRenderer.getNodeX(node) - borderWidth;
        const y = this.webglRenderer.getNodeY(node) - borderWidth;
        const width = this.webglRenderer.getNodeWidth(node) + borderWidth * 2;
        const height = this.webglRenderer.getNodeHeight(node) + borderWidth * 2;
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
            HIGHLIGHT_NODES_HIGHLIGHT_BORDER_Y_OFFSET,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor,
          borderWidth: 0,
          opacity: 1,
        });
      }
      this.highlightNodesBorders.generateMesh(rectangles);
      this.webglRendererThreejsService.addToScene(
        this.highlightNodesBorders.mesh,
      );
    }

    this.webglRenderer.animateIntoPositions((t) => {
      this.highlightNodesBorders.updateAnimationProgress(t);
    });
  }

  private clearHighlightNodes() {
    for (const mesh of [this.highlightNodesBorders.mesh]) {
      if (!mesh) {
        continue;
      }
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      this.webglRendererThreejsService.removeFromScene(mesh);
    }
  }

  private get borderColor(): string {
    return (
      this.webglRenderer.syncNavigationService.getSyncNavigationData()
        ?.relatedNodesBorderColor ?? DEFAULT_HIGHLIGHT_NODES_BORDER_COLOR
    );
  }
}
