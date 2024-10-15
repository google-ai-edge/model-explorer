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

import {Injectable, inject} from '@angular/core';
import * as three from 'three';
import {WEBGL_ELEMENT_Y_FACTOR} from './common/consts';
import {EdgeOverlay} from './common/edge_overlays';
import {GroupNode, ModelEdge, OpNode} from './common/model_graph';
import {getIntersectionPoints} from './common/utils';
import {EdgeOverlaysService} from './edge_overlays_service';
import {ThreejsService} from './threejs_service';
import {WebglEdges} from './webgl_edges';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {WebglTexts} from './webgl_texts';

const THREE = three;

const DEFAULT_EDGE_WIDTH = 1.5;

/**
 * Service for managing edge overlays related tasks in webgl renderer.
 */
@Injectable()
export class WebglRendererEdgeOverlaysService {
  private readonly threejsService: ThreejsService = inject(ThreejsService);

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private overlaysEdgesList: WebglEdges[] = [];
  private overlaysEdgeTextsList: WebglTexts[] = [];

  readonly edgeOverlaysService = inject(EdgeOverlaysService);
  curOverlays: EdgeOverlay[] = [];

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  updateOverlaysData() {
    this.clearOverlaysData();

    const selectedNodeId = this.webglRenderer.selectedNodeId;
    if (!selectedNodeId) {
      return;
    }

    // Find overlays that contain the node from the selected overlays.
    const selectedOverlays = this.edgeOverlaysService.selectedOverlays();
    for (const selectedOverlay of selectedOverlays) {
      if (selectedOverlay.nodeIds.has(selectedNodeId)) {
        this.curOverlays.push(selectedOverlay);
      }
    }
  }

  clearOverlaysData() {
    this.curOverlays = [];
  }

  updateOverlaysEdges() {
    this.clearOverlaysEdges();

    if (this.curOverlays.length === 0) {
      return;
    }

    for (let i = 0; i < this.curOverlays.length; i++) {
      const subgraph = this.curOverlays[i];
      const edgeWidth = subgraph.edgeWidth ?? DEFAULT_EDGE_WIDTH;
      const edges: Array<{edge: ModelEdge; index: number}> = [];
      const curWebglEdges = new WebglEdges(
        new THREE.Color(subgraph.edgeColor),
        edgeWidth,
        edgeWidth / DEFAULT_EDGE_WIDTH,
      );
      for (const {sourceNodeId, targetNodeId, label} of subgraph.edges) {
        const sourceNode = this.webglRenderer.curModelGraph.nodesById[
          sourceNodeId
        ] as OpNode;
        const targetNode = this.webglRenderer.curModelGraph.nodesById[
          targetNodeId
        ] as OpNode;
        const {intersection1, intersection2} = getIntersectionPoints(
          this.webglRenderer.getNodeRect(sourceNode),
          this.webglRenderer.getNodeRect(targetNode),
        );
        // Edge.
        edges.push({
          edge: {
            id: `overlay_edge_${i}_${sourceNodeId}_${targetNodeId}`,
            fromNodeId: sourceNodeId,
            toNodeId: targetNodeId,
            label: label ?? '',
            points: [],
            curvePoints: [
              {
                x: intersection1.x - (sourceNode?.globalX || 0),
                y: intersection1.y - (sourceNode?.globalY || 0),
              },
              {
                x: intersection2.x - (sourceNode.globalX || 0),
                y: intersection2.y - (sourceNode.globalY || 0),
              },
            ],
          },
          // Use anything > 95 which is used for rendering io highlight edges.
          index: 96 / WEBGL_ELEMENT_Y_FACTOR,
        });
      }
      curWebglEdges.generateMesh(edges, this.webglRenderer.curModelGraph);
      this.webglRendererThreejsService.addToScene(curWebglEdges.edgesMesh);
      this.webglRendererThreejsService.addToScene(curWebglEdges.arrowHeadsMesh);
      this.overlaysEdgesList.push(curWebglEdges);

      // Edge labels.
      const labels =
        this.webglRenderer.webglRendererEdgeTextsService.genLabelsOnEdges(
          edges,
          new THREE.Color(subgraph.edgeColor),
          edgeWidth / 2,
          96.5,
          subgraph.edgeLabelFontSize ?? 7.5,
        );
      const curWebglTexts = new WebglTexts(this.threejsService);
      curWebglTexts.generateMesh(labels, true, false, true);
      this.webglRendererThreejsService.addToScene(curWebglTexts.mesh);
      this.overlaysEdgeTextsList.push(curWebglTexts);
    }
  }

  clearOverlaysEdges() {
    for (const webglEdges of this.overlaysEdgesList) {
      webglEdges.clear();
    }
    for (const webglTexts of this.overlaysEdgeTextsList) {
      if (webglTexts.mesh && webglTexts.mesh.geometry) {
        webglTexts.mesh.geometry.dispose();
        this.webglRendererThreejsService.removeFromScene(webglTexts.mesh);
      }
    }

    this.overlaysEdgesList = [];
    this.overlaysEdgeTextsList = [];
  }

  getDeepestExpandedGroupNodeIds(): string[] {
    if (this.curOverlays.length === 0) {
      return [];
    }

    const ids = new Set<string>();

    const addNsParentId = (nodeId: string) => {
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
      if (node.nsParentId) {
        const parentNode = this.webglRenderer.curModelGraph.nodesById[
          node.nsParentId
        ] as GroupNode;
        if (
          !parentNode.expanded ||
          !this.webglRenderer.isNodeRendered(parentNode.id)
        ) {
          ids.add(node.nsParentId);
        }
      }
    };
    for (const subgraph of this.curOverlays) {
      for (const {sourceNodeId, targetNodeId} of subgraph.edges) {
        addNsParentId(sourceNodeId);
        addNsParentId(targetNodeId);
      }
    }
    return [...ids];
  }
}
