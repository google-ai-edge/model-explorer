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

import {inject, Injectable} from '@angular/core';
import * as three from 'three';

import {GroupNode} from './common/model_graph';
import {FontWeight} from './common/types';
import {isGroupNode} from './common/utils';
import {ThreejsService} from './threejs_service';
import {WebglRenderer} from './webgl_renderer';
import {IO_PICKER_HEIGHT} from './webgl_renderer_io_highlight_service';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';

const IDENTICAL_LAYER_INDICATOR_HEIGHT = IO_PICKER_HEIGHT;
const IDENTICAL_LAYER_INDICATOR_WIDTH = 68;

const THREE = three;

/** Service for rendering identical layers indicator. */
@Injectable()
export class WebglRendererIdenticalLayerService {
  readonly IDENTICAL_GROUPS_BG_COLOR = new THREE.Color('#e2edff');
  readonly IDENTICAL_GROUPS_INDICATOR_BG_COLOR = new THREE.Color('#e3e3e3');
  readonly IDENTICAL_GROUPS_INDICATOR_BORDER_COLOR = new THREE.Color('#ccc');

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly threejsService: ThreejsService = inject(ThreejsService);

  private readonly identicalLayerIndicatorBgs = new WebglRoundedRectangles(99);
  private readonly identicalLayerIndicatorTexts = new WebglTexts(
    this.threejsService,
  );

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  updateIdenticalLayerIndicators() {
    if (!this.webglRenderer.curModelGraph) {
      return;
    }
    this.clearIdenticalLayerIndicators();

    const ioHighlightService =
      this.webglRenderer.webglRendererIoHighlightService;
    const selectedNode =
      this.webglRenderer.curModelGraph.nodesById[
        this.webglRenderer.selectedNodeId
      ];
    const identicalGroupBgRectangles: RoundedRectangleData[] = [];
    const identicalGroupLabels: LabelData[] = [];
    if (
      isGroupNode(selectedNode) &&
      selectedNode?.identicalGroupIndex != null
    ) {
      const selectedIdenticalGroupIndex = selectedNode.identicalGroupIndex;
      const identicalGroupNodes: GroupNode[] = this.webglRenderer.nodesToRender
        .filter(
          ({node: curNode}) =>
            isGroupNode(curNode) &&
            curNode.identicalGroupIndex === selectedIdenticalGroupIndex,
        )
        .map(
          ({node}) => this.webglRenderer.curModelGraph.nodesById[node.id],
        ) as GroupNode[];
      for (const node of identicalGroupNodes) {
        if (node.id === selectedNode.id) {
          continue;
        }
        const badgeX =
          this.webglRenderer.getNodeX(node) +
          IDENTICAL_LAYER_INDICATOR_WIDTH / 2;
        const badgeY =
          this.webglRenderer.getNodeY(node) -
          IDENTICAL_LAYER_INDICATOR_HEIGHT / 2 +
          IDENTICAL_LAYER_INDICATOR_HEIGHT / 4;

        // Move the badge up if there is IO chip on the node.
        let badgeYOffset = 0;
        if (
          isGroupNode(node) &&
          (ioHighlightService.inputsByHighlightedNode[node.id] != null ||
            ioHighlightService.outputsByHighlightedNode[node.id] != null)
        ) {
          badgeYOffset = -15;
        }

        identicalGroupBgRectangles.push({
          id: node.id,
          index: identicalGroupBgRectangles.length,
          bound: {
            x: badgeX,
            y: badgeY + badgeYOffset,
            width: IDENTICAL_LAYER_INDICATOR_WIDTH,
            height: IDENTICAL_LAYER_INDICATOR_HEIGHT,
          },
          yOffset: 95.2,
          isRounded: true,
          borderColor: this.IDENTICAL_GROUPS_INDICATOR_BORDER_COLOR,
          bgColor: this.IDENTICAL_GROUPS_INDICATOR_BG_COLOR,
          borderWidth: 1,
          opacity: 1,
        });
        identicalGroupLabels.push({
          id: node.id,
          label: 'Identical layer',
          height: 8,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.MEDIUM,
          color: {r: 0, g: 0, b: 0},
          x: badgeX,
          y: 96,
          z: badgeY + badgeYOffset,
        });
      }
    }

    this.identicalLayerIndicatorBgs.generateMesh(
      identicalGroupBgRectangles,
      false,
      false,
      true,
      true,
    );
    this.webglRendererThreejsService.addToScene(
      this.identicalLayerIndicatorBgs.mesh,
    );
    this.identicalLayerIndicatorTexts.generateMesh(
      identicalGroupLabels,
      false,
      true,
      true,
    );
    this.webglRendererThreejsService.addToScene(
      this.identicalLayerIndicatorTexts.mesh,
    );

    this.webglRenderer.animateIntoPositions((t) => {
      this.identicalLayerIndicatorBgs.updateAnimationProgress(t);
      this.identicalLayerIndicatorTexts.updateAnimationProgress(t);
    });
  }

  private clearIdenticalLayerIndicators() {
    for (const mesh of [
      this.identicalLayerIndicatorBgs.mesh,
      this.identicalLayerIndicatorTexts.mesh,
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
