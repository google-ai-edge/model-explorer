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

import {effect, signal} from '@angular/core';
import * as three from 'three';

import {WEBGL_ELEMENT_Y_FACTOR} from './common/consts';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';

const THREE = three;

/**
 * The default border color used to highlight nodes.
 */
export const DEFAULT_HIGHLIGHT_NODES_BORDER_COLOR = '#ff00be';

/**
 * The default border width used to highlight nodes.
 */
export const DEFAULT_HIGHLIGHT_NODES_BORDER_WIDTH = 2;

/**
 * The default border color used to highlight deleted nodes.
 */
export const DEFAULT_DELETE_NODES_BORDER_COLOR = '#f26868';

/**
 * The default border color used to highlight new nodes.
 */
export const DEFAULT_NEW_NODES_BORDER_COLOR = '#aedcae';

/**
 * The highlight info for a node.
 */
export interface HighlightInfo {
  nodeId: string;
  borderColor: string;
  borderWidth: number;
}

/**
 * Service for rendering node highlight (an extra border around the node,
 * outside the node border).
 */
export class WebglRendererHighlightNodesService {
  readonly highlightNodesBorders = new WebglRoundedRectangles(8);

  private readonly highlights = signal<{[nodeId: string]: HighlightInfo}>({});
  private webglRendererThreejsService!: WebglRendererThreejsService;

  constructor(
    private readonly webglRenderer: WebglRenderer,
    private readonly yRelativeOffset: number,
  ) {
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;

    // Handle highlight nodes changes.
    effect(() => {
      this.clearAndRenderHighlightNodes();
    });
  }

  setNodeHighlights(
    highlights: {[nodeId: string]: HighlightInfo},
    clear = false,
  ) {
    if (Object.keys(highlights).length === 0) {
      this.clearNodeHighlights();
    } else {
      this.highlights.update((prevHighlights) => {
        if (clear) {
          return highlights;
        }
        return {...prevHighlights, ...highlights};
      });
    }
  }

  clearNodeHighlights() {
    this.highlights.set({});
  }

  private clearAndRenderHighlightNodes() {
    this.clearHighlightNodes();

    const highlights = this.highlights();
    if (Object.keys(highlights).length > 0) {
      // Render highlight borders for the result nodes.
      const rectangles: RoundedRectangleData[] = [];
      for (const nodeId of Object.keys(highlights)) {
        const curHighlight = highlights[nodeId];
        const borderWidth = curHighlight.borderWidth;
        const bgColor = new THREE.Color(curHighlight.borderColor);
        const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
        if (!node) {
          continue;
        }
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
          yOffset: WEBGL_ELEMENT_Y_FACTOR * nodeIndex + this.yRelativeOffset,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor,
          borderWidth: 0,
          opacity: 1,
        });
      }
      this.highlightNodesBorders.generateMesh(
        rectangles,
        false,
        false,
        false,
        true, // disable initial animation
      );
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
}
