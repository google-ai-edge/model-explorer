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

import {
  NODE_ATTRS_TABLE_LABEL_VALUE_PADDING,
  NODE_ATTRS_TABLE_MARGIN_TOP,
  NODE_ATTRS_TABLE_ROW_HEIGHT,
  NODE_ATTRS_TABLE_VALUE_MAX_WIDTH,
  WEBGL_ELEMENT_Y_FACTOR,
} from './common/consts';
import {ModelNode} from './common/model_graph';
import {FontWeight, ShowOnNodeItemType} from './common/types';
import {
  getGroupNodeAttrsKeyValuePairsForAttrsTable,
  getGroupNodeFieldLabelsFromShowOnNodeItemTypes,
  getMultiLineLabelExtraHeight,
  getNodeInfoFieldValue,
  getOpNodeAttrsKeyValuePairsForAttrsTable,
  getOpNodeDataProviderKeyValuePairsForAttrsTable,
  getOpNodeFieldLabelsFromShowOnNodeItemTypes,
  getOpNodeInputsKeyValuePairsForAttrsTable,
  getOpNodeOutputsKeyValuePairsForAttrsTable,
  isGroupNode,
  isOpNode,
} from './common/utils';
import {ThreejsService} from './threejs_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';
import {DEFAULT_NODE_HEIGHT} from './worker/graph_layout';

const ATTRS_TABLE_BG_Y_OFFSET = WEBGL_ELEMENT_Y_FACTOR * 0.2;
const ATTRS_TABLE_TEXT_Y_OFFSET = WEBGL_ELEMENT_Y_FACTOR * 0.4;

const THREE = three;

/**
 * Service for managing attributes table related tasks.
 *
 * Attributes table shows the key-value paris checked in the "view on node"
 * menu.
 */
@Injectable()
export class WebglRendererAttrsTableService {
  readonly ATTRS_TABLE_KEY_COLOR = new THREE.Color('#808080');
  readonly ATTRS_TABLE_VALUE_COLOR = new THREE.Color('#0d0d0d');

  private readonly threejsService: ThreejsService = inject(ThreejsService);
  readonly attrsTableTexts = new WebglTexts(this.threejsService);

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly attrsTableBgs = new WebglRoundedRectangles(4);

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  renderAttrsTable() {
    if (
      Object.keys(this.webglRenderer.curShowOnNodeItemTypes).filter(
        (type) => this.webglRenderer.curShowOnNodeItemTypes[type].selected,
      ).length === 0
    ) {
      return;
    }

    const labels: LabelData[] = [];
    const fontSize = this.attrsTableTexts.getFontSize();
    const scale = 9 / fontSize;
    const tableBgRectangles: RoundedRectangleData[] = [];
    for (const {node, index} of this.webglRenderer.nodesToRender) {
      const rows: Array<{
        keyLabelData: LabelData;
        valueLabelData: LabelData;
      }> = [];
      let curZ =
        DEFAULT_NODE_HEIGHT +
        NODE_ATTRS_TABLE_MARGIN_TOP -
        4 +
        getMultiLineLabelExtraHeight(node.label);
      let maxKeyWidth = 0;
      let maxValueWidth = 0;
      const keyValuePairs: Array<{key: string; value: string}> = [];

      if (isOpNode(node)) {
        const fieldIds = getOpNodeFieldLabelsFromShowOnNodeItemTypes(
          this.webglRenderer.curShowOnNodeItemTypes,
        );
        // Node info.
        for (const fieldId of fieldIds) {
          const value = getNodeInfoFieldValue(node, fieldId);
          keyValuePairs.push({key: fieldId, value});
        }

        // Attrs.
        if (
          this.webglRenderer.curShowOnNodeItemTypes[ShowOnNodeItemType.OP_ATTRS]
            ?.selected
        ) {
          keyValuePairs.push(
            ...getOpNodeAttrsKeyValuePairsForAttrsTable(
              node,
              this.webglRenderer.curShowOnNodeItemTypes[
                ShowOnNodeItemType.OP_ATTRS
              ]?.filterRegex || '',
            ),
          );
        }

        // Inputs.
        if (
          this.webglRenderer.curShowOnNodeItemTypes[
            ShowOnNodeItemType.OP_INPUTS
          ]?.selected
        ) {
          keyValuePairs.push(
            ...getOpNodeInputsKeyValuePairsForAttrsTable(
              node,
              this.webglRenderer.curModelGraph,
            ),
          );
        }

        // Outputs.
        if (
          this.webglRenderer.curShowOnNodeItemTypes[
            ShowOnNodeItemType.OP_OUTPUTS
          ]?.selected
        ) {
          keyValuePairs.push(
            ...getOpNodeOutputsKeyValuePairsForAttrsTable(node),
          );
        }

        // Node data provider.
        keyValuePairs.push(
          ...getOpNodeDataProviderKeyValuePairsForAttrsTable(
            node,
            this.webglRenderer.curModelGraph.id,
            this.webglRenderer.curShowOnNodeItemTypes,
            this.webglRenderer.curNodeDataProviderRuns,
            this.webglRenderer.appService.config(),
          ),
        );
      } else if (isGroupNode(node)) {
        const fieldIds = getGroupNodeFieldLabelsFromShowOnNodeItemTypes(
          this.webglRenderer.curShowOnNodeItemTypes,
        );
        // Node info.
        for (const fieldId of fieldIds) {
          const value = getNodeInfoFieldValue(node, fieldId);
          keyValuePairs.push({key: fieldId, value});
        }

        // Attrs.
        if (
          this.webglRenderer.curShowOnNodeItemTypes[
            ShowOnNodeItemType.LAYER_NODE_ATTRS
          ]?.selected
        ) {
          keyValuePairs.push(
            ...getGroupNodeAttrsKeyValuePairsForAttrsTable(
              node,
              this.webglRenderer.curModelGraph,
              this.webglRenderer.curShowOnNodeItemTypes[
                ShowOnNodeItemType.LAYER_NODE_ATTRS
              ]?.filterRegex || '',
            ),
          );
        }
      }

      // Generate rows.
      for (const {key, value} of keyValuePairs) {
        const {keyLabelData, keyLabelWidth, valueLabelData, valueLabelWidth} =
          this.createAttrsTableKeyValueLabels(
            node,
            index,
            key,
            value,
            curZ,
            scale,
          );
        labels.push(keyLabelData, valueLabelData);
        maxKeyWidth = Math.max(keyLabelWidth, maxKeyWidth);
        maxValueWidth = Math.max(valueLabelWidth, maxValueWidth);

        rows.push({
          keyLabelData,
          valueLabelData,
        });
        curZ += NODE_ATTRS_TABLE_ROW_HEIGHT;
      }

      // Adjust positions.
      const maxRowWidth =
        maxKeyWidth + maxValueWidth + NODE_ATTRS_TABLE_LABEL_VALUE_PADDING;
      const tableOffsetX =
        (this.webglRenderer.getNodeWidth(node) - maxRowWidth) / 2;
      for (const row of rows) {
        row.keyLabelData.x = this.webglRenderer.getNodeX(node) + maxKeyWidth;
        row.valueLabelData.x =
          this.webglRenderer.getNodeX(node) +
          maxKeyWidth +
          NODE_ATTRS_TABLE_LABEL_VALUE_PADDING;

        row.keyLabelData.x += tableOffsetX;
        row.valueLabelData.x += tableOffsetX;
      }

      // Create table bg rectangle (only for op nodes).
      if (rows.length > 0 && isOpNode(node)) {
        const padding = 16;
        const width = this.webglRenderer.getNodeWidth(node) - padding;
        const height = rows.length * NODE_ATTRS_TABLE_ROW_HEIGHT;
        tableBgRectangles.push({
          id: node.id,
          index: tableBgRectangles.length,
          bound: {
            x: this.webglRenderer.getNodeX(node) + padding / 2 + width / 2,
            y:
              rows[0].keyLabelData.z +
              height / 2 -
              NODE_ATTRS_TABLE_ROW_HEIGHT / 2,
            width,
            height,
          },
          yOffset: WEBGL_ELEMENT_Y_FACTOR * index + ATTRS_TABLE_BG_Y_OFFSET,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor: {r: 1, g: 1, b: 1},
          borderWidth: 1,
          opacity: 1,
        });
      }
    }

    if (labels.length > 0) {
      this.attrsTableTexts.generateMesh(labels);
      this.webglRendererThreejsService.addToScene(this.attrsTableTexts.mesh);

      this.attrsTableBgs.generateMesh(tableBgRectangles);
      this.webglRendererThreejsService.addToScene(this.attrsTableBgs.mesh);
    }
  }

  updateAnimationProgress(t: number) {
    this.attrsTableTexts.updateAnimationProgress(t);
    this.attrsTableBgs.updateAnimationProgress(t);
  }

  private createAttrsTableKeyValueLabels(
    node: ModelNode,
    index: number,
    key: string,
    value: string,
    zOffset: number,
    scale: number,
  ) {
    const keyLabelData: LabelData = {
      id: `${node.id}_attrs_table_${key}_key`,
      nodeId: node.id,
      label: `${key}:`,
      height: 9,
      hAlign: 'right',
      vAlign: 'center',
      weight: FontWeight.MEDIUM,
      x: this.webglRenderer.getNodeX(node),
      y: index * WEBGL_ELEMENT_Y_FACTOR + ATTRS_TABLE_TEXT_Y_OFFSET,
      z: this.webglRenderer.getNodeY(node) + zOffset,
      color: this.ATTRS_TABLE_KEY_COLOR,
    };
    const keyLabelSizes = this.attrsTableTexts.getLabelSizes(
      keyLabelData.label,
      keyLabelData.weight,
      keyLabelData.height,
    ).sizes;
    const keyLabelWidth = (keyLabelSizes.maxX - keyLabelSizes.minX) * scale;

    const valueLabelData: LabelData = {
      id: `${node.id}_attrs_table_${key}_value`,
      nodeId: node.id,
      label: value,
      height: 9,
      hAlign: 'left',
      vAlign: 'center',
      weight: FontWeight.REGULAR,
      x: this.webglRenderer.getNodeX(node),
      y: index * WEBGL_ELEMENT_Y_FACTOR + ATTRS_TABLE_TEXT_Y_OFFSET,
      z: this.webglRenderer.getNodeY(node) + zOffset,
      color: this.ATTRS_TABLE_VALUE_COLOR,
      maxWidth: NODE_ATTRS_TABLE_VALUE_MAX_WIDTH,
    };
    const {sizes: valueLabelSizes, updatedLabel} =
      this.attrsTableTexts.getLabelSizes(
        valueLabelData.label,
        valueLabelData.weight,
        valueLabelData.height,
        valueLabelData.maxWidth,
      );
    if (updatedLabel != null) {
      valueLabelData.label = updatedLabel;
    }
    const valueLabelWidth =
      (valueLabelSizes.maxX - valueLabelSizes.minX) * scale;

    return {keyLabelData, keyLabelWidth, valueLabelData, valueLabelWidth};
  }
}
