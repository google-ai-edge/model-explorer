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

import {computed, Injectable} from '@angular/core';
import * as three from 'three';

import {
  NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT,
  WEBGL_ELEMENT_Y_FACTOR,
} from './common/consts';
import {GroupNode} from './common/model_graph';
import {isGroupNode} from './common/utils';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';

const NODE_DATA_PROVIDER_DISTRIBUTION_BAR_Y_OFFSET =
  WEBGL_ELEMENT_Y_FACTOR * 0.5;

const THREE = three;

/**
 * Service for rendering node data provider related UI elements.
 */
@Injectable()
export class WebglRendererNdpService {
  readonly curNodeDataProviderResults = computed(() => {
    if (!this.webglRenderer) {
      return undefined;
    }

    const selectedRun =
      this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
        this.webglRenderer.paneId,
        this.webglRenderer.curModelGraph,
      );
    return (selectedRun?.results || {})[this.webglRenderer.curModelGraph.id];
  });

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly nodeDataProviderDistributionBars =
    new WebglRoundedRectangles(0);

  constructor(
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
  ) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  renderNodeDataProviderDistributionBars() {
    const results = this.curNodeDataProviderResults() || {};
    if (Object.keys(results).length === 0) {
      return;
    }

    const {groupIdToDescendantsBgColorCounts, sortedBgColors} =
      this.genGroupIdToDescendantsBgColorCounts();

    const rectangles: RoundedRectangleData[] = [];
    for (const {node, index} of this.webglRenderer.nodesToRender) {
      if (!groupIdToDescendantsBgColorCounts[node.id]) {
        continue;
      }
      const groupNode = node as GroupNode;
      const groupNodeWidth = groupNode.width || 0;

      const curBgColors = groupIdToDescendantsBgColorCounts[node.id];
      let countSum = 0;
      for (const count of Object.values(curBgColors)) {
        countSum += count;
      }
      let widthSum = 0;
      let colorIndex = 0;
      for (const bgColor of sortedBgColors) {
        if (curBgColors[bgColor] == null) {
          continue;
        }
        if (bgColor === 'transparent') {
          continue;
        }
        const count = curBgColors[bgColor];
        const width = (count / countSum) * groupNodeWidth;
        const height = NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT;
        const x = widthSum;

        rectangles.push({
          id: `${node.id}_${colorIndex}`,
          index: rectangles.length,
          bound: {
            x: this.webglRenderer.getNodeX(groupNode) + x + width / 2,
            y:
              this.webglRenderer.getNodeY(groupNode) +
              this.webglRenderer.getNodeHeight(groupNode) -
              NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT +
              height / 2,
            width,
            height,
          },
          yOffset:
            WEBGL_ELEMENT_Y_FACTOR * index +
            NODE_DATA_PROVIDER_DISTRIBUTION_BAR_Y_OFFSET,
          isRounded: false,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor: new THREE.Color(bgColor),
          borderWidth: 0,
          opacity: 1,
        });

        widthSum += width;
        colorIndex++;
      }
    }

    this.nodeDataProviderDistributionBars.generateMesh(rectangles);
    this.webglRendererThreejsService.addToScene(
      this.nodeDataProviderDistributionBars.mesh,
    );
  }

  updateAnimationProgress(t: number) {
    this.nodeDataProviderDistributionBars.updateAnimationProgress(t);
  }

  private genGroupIdToDescendantsBgColorCounts(): {
    groupIdToDescendantsBgColorCounts: Record<string, Record<string, number>>;
    sortedBgColors: string[];
  } {
    const results = this.curNodeDataProviderResults() || {};
    const groupIdToDescendantsBgColorCounts: Record<
      string,
      Record<string, number>
    > = {};
    const allBgColors = new Set<string>();
    for (const {node: groupNode} of this.webglRenderer.nodesToRender) {
      if (isGroupNode(groupNode) && !groupNode.expanded) {
        const bgColorCounts: Record<string, number> = {};
        for (const nodeId of groupNode.descendantsOpNodeIds || []) {
          const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
          const bgColor = results[node.id]?.bgColor || '';
          if (bgColor) {
            if (bgColorCounts[bgColor] == null) {
              bgColorCounts[bgColor] = 0;
            }
            bgColorCounts[bgColor]++;
            allBgColors.add(bgColor);
          }
        }
        groupIdToDescendantsBgColorCounts[groupNode.id] = bgColorCounts;
      }
    }
    return {
      groupIdToDescendantsBgColorCounts,
      sortedBgColors: [...allBgColors].sort((a, b) => a.localeCompare(b)),
    };
  }
}
