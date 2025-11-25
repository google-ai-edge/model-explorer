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
  DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE,
  NODE_ATTRS_TABLE_FONT_SIZE_TO_HEIGHT_RATIO,
  NODE_ATTRS_TABLE_LABEL_VALUE_PADDING,
  NODE_ATTRS_TABLE_MARGIN_TOP,
  NODE_ATTRS_TABLE_VALUE_MAX_CHAR_COUNT,
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
import {
  ColorVariable,
  VisualizerThemeService,
} from './visualizer_theme_service';
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
  private readonly threejsService: ThreejsService = inject(ThreejsService);
  readonly attrsTableTexts = new WebglTexts(this.threejsService);

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly visualizerThemeService: VisualizerThemeService = inject(
    VisualizerThemeService,
  );
  private readonly attrsTableBgs = new WebglRoundedRectangles(
    4,
    this.visualizerThemeService,
  );

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

    const fontSize =
      this.webglRenderer.appService.config()?.nodeAttrsTableFontSize ??
      DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE;
    const rowHeight = fontSize * NODE_ATTRS_TABLE_FONT_SIZE_TO_HEIGHT_RATIO;
    const labels: LabelData[] = [];
    const webglTextsFontSize = this.attrsTableTexts.getFontSize();
    const scale = fontSize / webglTextsFontSize;
    const tableBgRectangles: RoundedRectangleData[] = [];
    const attrTableBgColor = new THREE.Color(
      this.webglRenderer.visualizerThemeService.getColor(
        ColorVariable.SURFACE_CONTAINER_LOWEST_COLOR,
      ),
    );
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
        curZ += rowHeight;
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
        const height = rows.length * rowHeight;
        tableBgRectangles.push({
          id: node.id,
          index: tableBgRectangles.length,
          bound: {
            x: this.webglRenderer.getNodeX(node) + padding / 2 + width / 2,
            y: rows[0].keyLabelData.z + height / 2 - rowHeight / 2,
            width,
            height,
          },
          yOffset: WEBGL_ELEMENT_Y_FACTOR * index + ATTRS_TABLE_BG_Y_OFFSET,
          isRounded: true,
          borderColor: attrTableBgColor,
          bgColor: attrTableBgColor,
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
    const fontSize =
      this.webglRenderer.appService.config()?.nodeAttrsTableFontSize ??
      DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE;
    const nodeAttrsTableValueMaxWidth =
      fontSize * NODE_ATTRS_TABLE_VALUE_MAX_CHAR_COUNT;
    const attrTableKeyColor = new THREE.Color(
      this.webglRenderer.visualizerThemeService.getColor(
        ColorVariable.ON_SURFACE_VARIANT_COLOR,
      ),
    );
    const attrTableTextColor = new THREE.Color(
      this.webglRenderer.visualizerThemeService.getColor(
        ColorVariable.ON_SURFACE_COLOR,
      ),
    );
    const keyLabelData: LabelData = {
      id: `${node.id}_attrs_table_${key}_key`,
      nodeId: node.id,
      label: `${key}:`,
      height: fontSize,
      hAlign: 'right',
      vAlign: 'center',
      weight: FontWeight.MEDIUM,
      x: this.webglRenderer.getNodeX(node),
      y: index * WEBGL_ELEMENT_Y_FACTOR + ATTRS_TABLE_TEXT_Y_OFFSET,
      z: this.webglRenderer.getNodeY(node) + zOffset,
      color: attrTableKeyColor,
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
      height: fontSize,
      hAlign: 'left',
      vAlign: 'center',
      weight: FontWeight.REGULAR,
      x: this.webglRenderer.getNodeX(node),
      y: index * WEBGL_ELEMENT_Y_FACTOR + ATTRS_TABLE_TEXT_Y_OFFSET,
      z: this.webglRenderer.getNodeY(node) + zOffset,
      color: attrTableTextColor,
      maxWidth: nodeAttrsTableValueMaxWidth,
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
