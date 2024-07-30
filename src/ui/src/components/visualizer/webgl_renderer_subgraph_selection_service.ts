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

import {effect, inject, Injectable} from '@angular/core';
import * as three from 'three';

import {AppService} from './app_service';
import {GroupNode, ModelNode, OpNode} from './common/model_graph';
import {FontWeight, WebglColor} from './common/types';
import {SubgraphSelectionService} from './subgraph_selection_service';
import {ThreejsService} from './threejs_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';

const COLOR_WHITE: WebglColor = {r: 1, g: 1, b: 1};
const SUBGRAPH_SELECTION_MARKER_SIZE = 14;

const THREE = three;

/**
 * Service for rendering subgraph selection related UI elements.
 */
@Injectable()
export class WebglRendererSubgraphSelectionService {
  readonly SUBGRAPH_SELECTED_NODE_MARKER_BG_COLOR = new THREE.Color('#09B83E');

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly threejsService: ThreejsService = inject(ThreejsService);

  private curSubgraphSelectedNodeIds: Record<string, boolean> = {};
  private readonly subgraphsSelectedNodeMarkerBgs = new WebglRoundedRectangles(
    99,
  );
  private readonly subgraphSelectedNodeMarkerTexts = new WebglTexts(
    this.threejsService,
  );

  constructor(
    private readonly appService: AppService,
    private readonly subgraphSelectionService: SubgraphSelectionService,
  ) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;

    // Handle subgraph selected nodes changes.
    effect(() => {
      this.curSubgraphSelectedNodeIds =
        this.subgraphSelectionService.selectedNodeIds();
      this.renderSubgraphSelectedNodeMarkers();
    });
  }

  renderSubgraphSelectedNodeMarkers() {
    if (!this.enableSubgraphSelection) {
      return;
    }

    const nodeIds = Object.keys(this.curSubgraphSelectedNodeIds);
    const nodeIdsSet = new Set<string>(nodeIds);

    this.clearSubgraphSelectedNodeMarkers();

    if (nodeIds.length === 0) {
      this.webglRendererThreejsService.render();
      return;
    }

    // Render marker bgs.
    const rectangles: RoundedRectangleData[] = [];
    // Get all the ancestor group nodes from the current selection.
    // Render a marker on each group node to indicate how many descendant
    // op nodes are selected.
    const ancestorGroupNodeIds = new Set<string>();
    for (const nodeId of nodeIds) {
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId] as OpNode;
      let curNode: ModelNode = node;
      while (true) {
        const parentNode: ModelNode =
          this.webglRenderer.curModelGraph.nodesById[curNode.nsParentId || ''];
        if (parentNode) {
          ancestorGroupNodeIds.add(parentNode.id);
          curNode = parentNode;
        } else {
          break;
        }
      }
    }
    const labels: LabelData[] = [];
    for (const nodeId of ancestorGroupNodeIds) {
      if (!this.webglRenderer.isNodeRendered(nodeId)) {
        continue;
      }
      const node = this.webglRenderer.curModelGraph.nodesById[
        nodeId
      ] as GroupNode;
      const x =
        this.webglRenderer.getNodeX(node) +
        this.webglRenderer.getNodeWidth(node);
      const y = this.webglRenderer.getNodeY(node);

      // Count label.
      const selectedNodeCounts = (node.descendantsOpNodeIds || []).filter(
        (id) => {
          const node = this.webglRenderer.curModelGraph.nodesById[id];
          return nodeIdsSet.has(node.id);
        },
      ).length;
      labels.push({
        id: `${nodeId}_subgraph_count_label`,
        nodeId,
        label: `${selectedNodeCounts}`,
        height: 8,
        hAlign: 'center',
        vAlign: 'center',
        weight: FontWeight.MEDIUM,
        color: this.webglRenderer.NODE_LABEL_COLOR,
        x,
        y: 96,
        z: y + 1,
      });

      // Bg.
      const width =
        SUBGRAPH_SELECTION_MARKER_SIZE * (selectedNodeCounts >= 1000 ? 2 : 1.5);
      const height = SUBGRAPH_SELECTION_MARKER_SIZE;
      rectangles.push({
        id: nodeId,
        index: rectangles.length,
        bound: {
          x,
          y,
          width,
          height,
        },
        // This should be over io picker which has 95 yOffset.
        yOffset: 95.5,
        isRounded: true,
        borderColor: this.SUBGRAPH_SELECTED_NODE_MARKER_BG_COLOR,
        bgColor: COLOR_WHITE,
        borderWidth: 1.5,
        opacity: 1,
      });
    }

    for (const nodeId of nodeIds) {
      if (!this.webglRenderer.isNodeRendered(nodeId)) {
        continue;
      }
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
      const x =
        this.webglRenderer.getNodeX(node) +
        this.webglRenderer.getNodeWidth(node);
      const y = this.webglRenderer.getNodeY(node);
      const width = SUBGRAPH_SELECTION_MARKER_SIZE;
      const height = SUBGRAPH_SELECTION_MARKER_SIZE;
      rectangles.push({
        id: nodeId,
        index: rectangles.length,
        bound: {
          x,
          y,
          width,
          height,
        },
        // This should be over io picker which has 95 yOffset.
        yOffset: 95.5,
        isRounded: true,
        borderColor: this.SUBGRAPH_SELECTED_NODE_MARKER_BG_COLOR,
        bgColor: COLOR_WHITE,
        borderWidth: 1.5,
        opacity: 1,
      });
      // Check icon.
      labels.push({
        id: `${node.id}_checkmark`,
        nodeId: node.id,
        // icon name: done
        label: '0xe876',
        height: 24,
        hAlign: 'center',
        vAlign: 'center',
        weight: FontWeight.ICONS,
        color: this.SUBGRAPH_SELECTED_NODE_MARKER_BG_COLOR,
        x,
        y: 96,
        z: y + 14,
        treatLabelAsAWhole: true,
        weightLevel: 0.9,
      });
    }
    this.subgraphsSelectedNodeMarkerBgs.generateMesh(
      rectangles,
      false,
      false,
      false,
      true,
    );
    this.webglRendererThreejsService.addToScene(
      this.subgraphsSelectedNodeMarkerBgs.mesh,
    );
    this.subgraphSelectedNodeMarkerTexts.generateMesh(
      labels,
      false,
      true,
      true,
    );
    this.webglRendererThreejsService.addToScene(
      this.subgraphSelectedNodeMarkerTexts.mesh,
    );

    this.webglRenderer.animateIntoPositions((t) => {
      this.subgraphsSelectedNodeMarkerBgs.updateAnimationProgress(t);
      this.subgraphSelectedNodeMarkerTexts.updateAnimationProgress(t);
    });
  }

  get enableSubgraphSelection(): boolean {
    return this.appService.config()?.enableSubgraphSelection === true;
  }

  private clearSubgraphSelectedNodeMarkers() {
    for (const mesh of [
      this.subgraphsSelectedNodeMarkerBgs.mesh,
      this.subgraphSelectedNodeMarkerTexts.mesh,
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
