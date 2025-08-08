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
import {Edge, EdgeOverlay, ProcessedEdgeOverlay} from './common/edge_overlays';
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

interface QueueItem {
  nodeId: string;
  hops: number;
}

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
  private bfsEdgeCache: Map<string, Set<Edge>> = new Map();

  readonly edgeOverlaysService = inject(EdgeOverlaysService);
  curOverlays: ProcessedEdgeOverlay[] = [];

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

    // Keep track of number of edges for a given pair of nodes. If there are
    // more than 1 edges, we will shift the edges to avoid overlapping.
    //
    // From sorted edge key (nodeId1->nodeId2) to the number of edges for that
    // pair.
    const seenEdgePairs: Record<string, number> = {};
    const totalEdgePairs: Record<string, number> = {};

    // Populate totalEdgePairs.
    for (let i = 0; i < this.curOverlays.length; i++) {
      const subgraph = this.curOverlays[i];
      for (const edge of subgraph.edges) {
        const {sourceNodeId, targetNodeId, label} = edge;
        if (!this.shouldShowEdge(subgraph, edge)) {
          continue;
        }
        this.addToEdgePairs(sourceNodeId, targetNodeId, totalEdgePairs);
      }
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
      for (const edge of subgraph.edges) {
        const {sourceNodeId, targetNodeId, label} = edge;
        if (!this.shouldShowEdge(subgraph, edge)) {
          continue;
        }

        const sourceNode = this.webglRenderer.curModelGraph.nodesById[
          sourceNodeId
        ] as OpNode;
        const targetNode = this.webglRenderer.curModelGraph.nodesById[
          targetNodeId
        ] as OpNode;
        if (!sourceNode || !targetNode) {
          continue;
        }
        const curEdgesCount = this.addToEdgePairs(
          sourceNodeId,
          targetNodeId,
          seenEdgePairs,
        );
        const totalEdgesCount =
          totalEdgePairs[this.getEdgeKey(sourceNodeId, targetNodeId)];
        const xOffsetFactor = (1 / (totalEdgesCount + 1)) * curEdgesCount - 0.5;
        const {intersection1, intersection2} = getIntersectionPoints(
          this.webglRenderer.getNodeRect(sourceNode),
          this.webglRenderer.getNodeRect(targetNode),
          xOffsetFactor,
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
      if (node?.nsParentId) {
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
      for (const edge of subgraph.edges) {
        const {sourceNodeId, targetNodeId} = edge;
        if (!this.shouldShowEdge(subgraph, edge)) {
          continue;
        }

        addNsParentId(sourceNodeId);
        addNsParentId(targetNodeId);
      }
    }
    return [...ids];
  }

  private addToEdgePairs(
    nodeId1: string,
    nodeId2: string,
    pairs: Record<string, number>,
  ): number {
    const key = this.getEdgeKey(nodeId1, nodeId2);
    if (pairs[key] === undefined) {
      pairs[key] = 0;
    }
    pairs[key]++;
    return pairs[key];
  }

  private getEdgeKey(nodeId1: string, nodeId2: string): string {
    return nodeId1.localeCompare(nodeId2) < 0
      ? `${nodeId1}___${nodeId2}`
      : `${nodeId2}___${nodeId1}`;
  }

  /**
   * Determines whether a given edge should be visible.
   *
   * This function first checks if the `edgeOverlay` is configured to show
   * only edges connected to the selected node. If not, all edges are visible,
   * and the function returns `true`.
   *
   * If the overlay is restricted, a Breadth-First Search (BFS) is performed
   * starting from the `selectedNodeId`. The search explores the graph up to
   * `visibleEdgeHops`. If the provided `edge` is encountered during
   * this search, it is considered visible and the function returns `true`.
   *
   * If the BFS completes without finding the edge, it means the edge is
   * outside the specified range from the selected node, and the function
   * returns `false`.
   */
  private shouldShowEdge(
    edgeOverlay: ProcessedEdgeOverlay,
    edge: Edge,
  ): boolean {
    if (!edgeOverlay.showEdgesConnectedToSelectedNodeOnly) {
      return true;
    }

    const selectedNodeId = this.webglRenderer.selectedNodeId;
    const maxHops = edgeOverlay.visibleEdgeHops ?? 1;

    // Perform BFS to find all the edges connected to the selected node within
    // the given number of hops.
    //
    // Try to find the result in the cache.
    const cacheKey = `${maxHops}-${edgeOverlay.id}-${selectedNodeId}`;
    if (this.bfsEdgeCache.has(cacheKey)) {
      const foundEdges = this.bfsEdgeCache.get(cacheKey)!;
      return foundEdges.has(edge);
    }

    // Not found in the cache, so we perform BFS.
    const queue: QueueItem[] = [{nodeId: selectedNodeId, hops: 0}];
    const visitedNodes = new Set<string>();
    const foundEdges = new Set<Edge>();

    visitedNodes.add(selectedNodeId);

    let head = 0;
    while (head < queue.length) {
      const {nodeId: currentNodeId, hops: currentHops} = queue[head++];

      // If we have reached the maximum number of hops, we stop exploring
      // further.
      if (currentHops >= maxHops) {
        continue;
      }

      // Get the neighbors of the current node from the adjacency map.
      const neighboringEdges =
        edgeOverlay.adjacencyMap.get(currentNodeId) || [];
      for (const curEdge of neighboringEdges) {
        foundEdges.add(curEdge);

        // Determine the next node to visit.
        const nextNodeId =
          curEdge.sourceNodeId === currentNodeId
            ? curEdge.targetNodeId
            : curEdge.sourceNodeId;

        // If we haven't visited this node yet, add it to the queue for the next
        // step.
        if (!visitedNodes.has(nextNodeId)) {
          visitedNodes.add(nextNodeId);
          queue.push({nodeId: nextNodeId, hops: currentHops + 1});
        }
      }
    }

    // Add the result to the cache.
    this.bfsEdgeCache.set(cacheKey, foundEdges);

    return foundEdges.has(edge);
  }
}
