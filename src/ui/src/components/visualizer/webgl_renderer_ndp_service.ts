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

import {computed, inject, Injectable} from '@angular/core';
import * as three from 'three';

import {
  EXPANDED_NODE_DATA_PROVIDER_SUMMARY_BOTTOM_PADDING,
  EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT,
  EXPANDED_NODE_DATA_PROVIDER_SUMMARY_TOP_PADDING,
  EXPANDED_NODE_DATA_PROVIDER_SYUMMARY_FONT_SIZE,
  NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT,
  WEBGL_ELEMENT_Y_FACTOR,
} from './common/consts';
import {GroupNode} from './common/model_graph';
import {FontWeight, NodeDataProviderValueInfo} from './common/types';
import {genSortedValueInfos, isGroupNode} from './common/utils';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {ThreejsService} from './threejs_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';

const NODE_DATA_PROVIDER_DISTRIBUTION_BAR_Y_OFFSET =
  WEBGL_ELEMENT_Y_FACTOR * 0.5;

const THREE = three;

/**
 * Service for rendering node data provider related UI elements.
 */
@Injectable()
export class WebglRendererNdpService {
  readonly curNodeDataProviderRun = computed(() => {
    if (!this.webglRenderer) {
      return undefined;
    }

    const selectedRun =
      this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
        this.webglRenderer.paneId,
        this.webglRenderer.curModelGraph,
      );
    return selectedRun;
  });

  readonly curNodeDataProviderResults = computed(() => {
    const selectedRun = this.curNodeDataProviderRun();
    return (selectedRun?.results || {})[this.webglRenderer.curModelGraph.id];
  });

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly threejsService: ThreejsService = inject(ThreejsService);
  private readonly nodeDataProviderDistributionBars =
    new WebglRoundedRectangles(0);
  private readonly nodeDataProviderSummaryTexts = new WebglTexts(
    this.threejsService,
  );

  constructor(
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
  ) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  renderNodeDataProviderDistributionBars() {
    const results = this.curNodeDataProviderRun() || {};
    if (Object.keys(results).length === 0) {
      return;
    }

    const curRun = this.curNodeDataProviderRun();
    if (!curRun) {
      return;
    }

    const groupIdToSortedValueInfos = this.genGroupIdToSortedValueInfos();
    const showExpandedSummaryOnGroupNode = (curRun.nodeDataProviderData ?? {})[
      this.webglRenderer.curModelGraph.id
    ]?.showExpandedSummaryOnGroupNode;

    const rectangles: RoundedRectangleData[] = [];
    const texts: LabelData[] = [];
    for (const {node, index} of this.webglRenderer.nodesToRender) {
      if (!groupIdToSortedValueInfos[node.id]) {
        continue;
      }
      const groupNode = node as GroupNode;
      const groupNodeWidth = groupNode.width || 0;

      const curSortedValueInfos = groupIdToSortedValueInfos[node.id];
      const countSum = curSortedValueInfos.reduce(
        (sum, cur) => sum + cur.count,
        0,
      );
      let widthSum = 0;
      let colorIndex = 0;
      let expandedSummaryHeight = 0;
      if (showExpandedSummaryOnGroupNode && !groupNode.expanded) {
        expandedSummaryHeight =
          EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT *
            curSortedValueInfos.length +
          EXPANDED_NODE_DATA_PROVIDER_SUMMARY_TOP_PADDING +
          EXPANDED_NODE_DATA_PROVIDER_SUMMARY_BOTTOM_PADDING;
      }

      for (let i = 0; i < curSortedValueInfos.length; i++) {
        const valueInfo = curSortedValueInfos[i];
        const bgColor = valueInfo.bgColor;
        if (bgColor === 'transparent') {
          continue;
        }
        const count = valueInfo.count;
        const width = (count / countSum) * groupNodeWidth;
        const height = NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT;
        const x = widthSum;

        const barY =
          this.webglRenderer.getNodeY(groupNode) +
          this.webglRenderer.getNodeHeight(groupNode) -
          expandedSummaryHeight -
          NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT +
          height / 2;
        rectangles.push({
          id: `${node.id}_${colorIndex}`,
          index: rectangles.length,
          bound: {
            x: this.webglRenderer.getNodeX(groupNode) + x + width / 2,
            y: barY,
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

        if (showExpandedSummaryOnGroupNode && !groupNode.expanded) {
          // The "index" color block in the summary section.
          const indexColorBlockY =
            barY +
            EXPANDED_NODE_DATA_PROVIDER_SUMMARY_TOP_PADDING +
            height / 2 +
            i * EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT +
            EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT / 2;
          rectangles.push({
            id: `${node.id}_${colorIndex}_summary`,
            index: rectangles.length,
            bound: {
              x: this.webglRenderer.getNodeX(groupNode) + 8,
              y: indexColorBlockY,
              width: 3,
              height: EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT - 2,
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

          // Color label.
          texts.push({
            id: `${node.id}_${colorIndex}_summary`,
            label: valueInfo.label,
            height: EXPANDED_NODE_DATA_PROVIDER_SYUMMARY_FONT_SIZE,
            hAlign: 'left',
            vAlign: 'center',
            weight: FontWeight.MEDIUM,
            color: {r: 0, g: 0, b: 0},
            x: this.webglRenderer.getNodeX(groupNode) + 12,
            y: 96,
            z: indexColorBlockY,
          });

          // Pct + count.
          texts.push({
            id: `${node.id}_${colorIndex}_summary_pct_count`,
            label: `${Math.floor((count / countSum) * 100)}% (${count})`,
            height: EXPANDED_NODE_DATA_PROVIDER_SYUMMARY_FONT_SIZE,
            hAlign: 'right',
            vAlign: 'center',
            weight: FontWeight.MEDIUM,
            color: {r: 0, g: 0, b: 0},
            x:
              this.webglRenderer.getNodeX(groupNode) +
              this.webglRenderer.getNodeWidth(groupNode) -
              6,
            y: 96,
            z: indexColorBlockY,
          });
        }

        widthSum += width;
        colorIndex++;
      }
    }

    this.nodeDataProviderDistributionBars.generateMesh(rectangles);
    this.webglRendererThreejsService.addToScene(
      this.nodeDataProviderDistributionBars.mesh,
    );
    this.nodeDataProviderSummaryTexts.generateMesh(texts, false, true, true);
    this.webglRendererThreejsService.addToScene(
      this.nodeDataProviderSummaryTexts.mesh,
    );
  }

  updateAnimationProgress(t: number) {
    this.nodeDataProviderDistributionBars.updateAnimationProgress(t);
    this.nodeDataProviderSummaryTexts.updateAnimationProgress(t);
  }

  private genGroupIdToSortedValueInfos(): Record<
    string,
    NodeDataProviderValueInfo[]
  > {
    const results = this.curNodeDataProviderResults() || {};
    const groupIdToDescendantsBgColorCounts: Record<
      string,
      NodeDataProviderValueInfo[]
    > = {};
    for (const {node: groupNode} of this.webglRenderer.nodesToRender) {
      if (isGroupNode(groupNode) && !groupNode.expanded) {
        const sortedValueInfos = genSortedValueInfos(
          groupNode,
          this.webglRenderer.curModelGraph,
          results,
        );
        if (sortedValueInfos.length > 0) {
          groupIdToDescendantsBgColorCounts[groupNode.id] = sortedValueInfos;
        }
      }
    }
    return groupIdToDescendantsBgColorCounts;
  }
}
