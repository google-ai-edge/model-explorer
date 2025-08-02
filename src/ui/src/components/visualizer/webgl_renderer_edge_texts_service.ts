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

import {AppService} from './app_service';
import {DEFAULT_EDGE_LABEL_FONT_SIZE} from './common/consts';
import {ModelEdge} from './common/model_graph';
import {FontWeight} from './common/types';
import {getNodeAttrStringValue, isOpNode} from './common/utils';
import {ThreejsService} from './threejs_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {LabelData, WebglTexts} from './webgl_texts';

const THREE = three;

/** Service for rendering edge texts. */
@Injectable()
export class WebglRendererEdgeTextsService {
  readonly EDGE_TEXT_COLOR = new THREE.Color('#041E49');

  private readonly threejsService: ThreejsService = inject(ThreejsService);
  readonly edgeTexts = new WebglTexts(this.threejsService);

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;

  constructor(private readonly appService: AppService) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  renderEdgeTexts(data?: {
    outputMetadataKey?: string;
    inputMetadataKey?: string;
    sourceNodeAttrKey?: string;
    targetNodeAttrKey?: string;
  }) {
    const labels = this.genLabelsOnEdges(
      this.webglRenderer.edgesToRender,
      this.EDGE_TEXT_COLOR,
      0,
      95,
      undefined,
      data?.outputMetadataKey,
      data?.inputMetadataKey,
      data?.sourceNodeAttrKey,
      data?.targetNodeAttrKey,
    );
    this.edgeTexts.generateMesh(labels);
    this.webglRendererThreejsService.addToScene(this.edgeTexts.mesh);
  }

  genLabelsOnEdges(
    edges: Array<{index: number; edge: ModelEdge}>,
    color: three.Color,
    extraOffsetToEdge = 0,
    y = 95,
    fontSize?: number,
    outputMetadataKey?: string,
    inputMetadataKey?: string,
    sourceNodeAttrKey?: string,
    targetNodeAttrKey?: string,
  ): LabelData[] {
    const edgeLabelFontSize =
      fontSize ??
      this.appService.config()?.edgeLabelFontSize ??
      DEFAULT_EDGE_LABEL_FONT_SIZE;
    const disallowVerticalEdgeLabels =
      this.appService.config()?.disallowVerticalEdgeLabels || false;
    const labels: LabelData[] = [];
    const charsInfo = this.threejsService.getCharsInfo(FontWeight.MEDIUM);
    for (const {edge} of edges) {
      const fromNode =
        this.webglRenderer.curModelGraph.nodesById[edge.fromNodeId];
      const toNode = this.webglRenderer.curModelGraph.nodesById[edge.toNodeId];

      if (!isOpNode(fromNode) || !isOpNode(toNode)) {
        continue;
      }

      // Find the edge label.
      let edgeLabel = '?';
      if (edge.label != null) {
        edgeLabel = edge.label;
        if (edgeLabel === '') {
          continue;
        }
      } else if (outputMetadataKey != null) {
        const outputsMetadata = fromNode.outputsMetadata || {};
        for (const outputId of Object.keys(outputsMetadata)) {
          const outgoingEdge = (fromNode.outgoingEdges || []).find(
            (curEdge) =>
              curEdge.sourceNodeOutputId === outputId &&
              curEdge.targetNodeId === edge.toNodeId,
          );
          if (outgoingEdge != null) {
            edgeLabel = outputsMetadata[outputId][outputMetadataKey] || '?';
            edgeLabel = edgeLabel
              .split('')
              .map((char) => {
                if (char === 'x') {
                  char = 'x';
                }
                if (char === '∗') {
                  char = '*';
                }
                if (char === '') {
                  char = '';
                }
                return charsInfo[char] == null ? '?' : char;
              })
              .join('');
            break;
          }
        }
      } else if (inputMetadataKey != null) {
        const inputsMetadata = toNode.inputsMetadata || {};
        for (const inputId of Object.keys(inputsMetadata)) {
          const incomingEdge = (toNode.incomingEdges || []).find(
            (curEdge) =>
              curEdge.sourceNodeId === edge.fromNodeId &&
              curEdge.targetNodeInputId === inputId,
          );
          if (incomingEdge != null) {
            edgeLabel = inputsMetadata[inputId][inputMetadataKey] || '?';
            break;
          }
        }
      } else if (sourceNodeAttrKey != null) {
        edgeLabel = getNodeAttrStringValue(fromNode, sourceNodeAttrKey) || '?';
      } else if (targetNodeAttrKey != null) {
        edgeLabel = getNodeAttrStringValue(toNode, targetNodeAttrKey) || '?';
      }

      const curvePoints = edge.curvePoints || [];
      const fromNodeGlobalX = fromNode.globalX || 0;
      const fromNodeGlobalY = fromNode.globalY || 0;

      // Construct a curve path representing the whole edge.
      const curvePath = new THREE.CurvePath();
      for (let i = 0; i < curvePoints.length - 1; i++) {
        const curPoint = curvePoints[i];
        const nextPoint = curvePoints[i + 1];
        const lineCurve = new THREE.LineCurve(
          new THREE.Vector2(
            curPoint.x + fromNodeGlobalX,
            curPoint.y + fromNodeGlobalY,
          ),
          new THREE.Vector2(
            nextPoint.x + fromNodeGlobalX,
            nextPoint.y + fromNodeGlobalY,
          ),
        );
        curvePath.add(lineCurve);
      }

      // Check whether the text is longer than the curve. If so, render the
      // text as a whole string at the middle of the curve.
      //
      // Use '3' to take some padding into account when calculating text length.
      const curveLength = curvePath.getLength();
      const space = edgeLabelFontSize / 2 / curveLength;
      const textLongerThanCurve = space * (edgeLabel.length + 3) > 1;
      const renderWholeTextFn = () => {
        const pos = curvePath.getPointAt(0.5) as three.Vector2;
        const posX = pos.x;
        const posY =
          curvePoints[0].y === curvePoints[curvePoints.length - 1].y
            ? pos.y - 10 - extraOffsetToEdge
            : pos.y;
        labels.push({
          id: `${edge.id}_${edgeLabel}`,
          nodeId: edge.toNodeId,
          label: edgeLabel,
          height: edgeLabelFontSize,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.MEDIUM,
          x: posX,
          y,
          z: posY,
          color,
          borderColor: {r: 1, g: 1, b: 1},
        });
      };
      if (textLongerThanCurve) {
        renderWholeTextFn();
      }
      // Text is shorter than curve. Render text character by character along
      // the curve.
      else {
        // Get character positions and angles.
        //
        // We are trying to find a segment on the curve that is mostly "smooth".
        // A segment is considered smooth when angles between characters don't
        // change too much.
        let charInfoList: Array<{
          pos: three.Vector2;
          position: number;
          angle: number;
          tan: three.Vector2;
          char: string;
        }> = [];
        const startPosition = Math.max(
          0,
          // 5 is the estimated height of the arrow head.
          Math.min(0.25, 1 - edgeLabel.length * space - 5 / curveLength),
        );
        const maxOffset = Math.max(
          0.05,
          // 5 is the estimated height of the arrow head.
          1 - 5 / curveLength - startPosition - space * edgeLabel.length,
        );
        // const step = 10 / curveLength;
        const step = 0.05;
        let smooth = true;
        const scale = edgeLabelFontSize / this.edgeTexts.getFontSize();
        for (let offset = 0; offset < maxOffset; offset += step) {
          const curStartPosition = startPosition + offset;
          smooth = true;
          let prevAngle: number | undefined = undefined;
          charInfoList = [];
          let curPosition = curStartPosition;
          for (let i = 0; i < edgeLabel.length; i++) {
            const char = edgeLabel[i];
            const pos = curvePath.getPointAt(
              Math.min(curPosition, 1),
            ) as three.Vector2;
            const tan = curvePath.getTangentAt(
              Math.min(curPosition, 1),
            ) as three.Vector2;
            let angle =
              (Math.PI * 2 - Math.atan(tan.y / tan.x)) % (Math.PI * 2);
            if (angle < 0) {
              angle += Math.PI * 2;
            }
            // Skip if the label is too vertical.
            if (
              disallowVerticalEdgeLabels &&
              angle >= Math.PI / 4 &&
              angle <= Math.PI * 1.75
            ) {
              smooth = false;
              break;
            }
            charInfoList.push({
              pos,
              position: Math.min(curPosition, 1),
              angle,
              tan,
              char,
            });

            if (prevAngle != null) {
              const delta = Math.abs(angle - prevAngle);
              const minDelta = Math.min(delta, Math.abs(delta - Math.PI));
              if (minDelta > 0.15) {
                smooth = false;
                if (offset + 0.05 < maxOffset) {
                  break;
                }
              }
            }
            prevAngle = angle;

            const charInfo = charsInfo[char];
            let nextCharXadvance = 0;
            if (i !== edgeLabel.length - 1) {
              const nextChar = edgeLabel[i + 1];
              nextCharXadvance = charsInfo[nextChar].xadvance;
            }
            const delta =
              ((charInfo.xadvance / 2 + nextCharXadvance / 2) * scale) /
              curveLength;
            curPosition += delta;
          }
          if (smooth) {
            break;
          }
        }

        // If we still cannot find a "smooth" segment, render the whole string
        // without following along the curve.
        if (!smooth) {
          renderWholeTextFn();
        } else {
          // Reverse the string based on whether the first character is at left
          // of the last character or not.
          const reverse =
            charInfoList[0].pos.x > charInfoList[charInfoList.length - 1].pos.x;
          const isVertical =
            Math.abs(
              charInfoList[0].pos.x -
                charInfoList[charInfoList.length - 1].pos.x,
            ) < 1e-7;

          if (reverse) {
            const newCharInfoList: Array<{
              pos: three.Vector2;
              angle: number;
              tan: three.Vector2;
              position: number;
              char: string;
            }> = [];
            let curPosition = charInfoList[0].position;
            for (let i = edgeLabel.length - 1; i >= 0; i--) {
              const char = edgeLabel[i];
              const pos = curvePath.getPointAt(
                Math.min(1, curPosition),
              ) as three.Vector2;
              const tan = curvePath.getTangentAt(
                Math.min(1, curPosition),
              ) as three.Vector2;
              let angle =
                (Math.PI * 2 - Math.atan(tan.y / tan.x)) % (Math.PI * 2);
              if (angle < 0) {
                angle += Math.PI * 2;
              }
              newCharInfoList.push({
                pos,
                angle,
                tan,
                position: curPosition,
                char,
              });
              const charInfo = charsInfo[char];
              let nextCharXadvance = 0;
              if (i >= 1) {
                const nextCharInfo = charsInfo[edgeLabel[i - 1]];
                nextCharXadvance = nextCharInfo.xadvance;
              }
              const delta =
                ((charInfo.xadvance / 2 + nextCharXadvance / 2) * scale) /
                curveLength;
              curPosition += delta;
            }
            charInfoList = newCharInfoList;
          }

          // Generate data for mesh.
          for (let i = 0; i < charInfoList.length; i++) {
            const charInfo = charInfoList[i];
            const char = charInfo.char;
            const pos = charInfo.pos;
            const tan = charInfo.tan;
            let angle = charInfo.angle;
            if (Math.abs(tan.x) < 1e-7) {
              angle =
                ((reverse || (isVertical && tan.y === -1) ? 1 : -1) * Math.PI) /
                2;
            }
            labels.push({
              id: `${edge.id}_${char}_${i}`,
              nodeId: edge.toNodeId,
              label: char,
              height: edgeLabelFontSize,
              hAlign: '',
              vAlign: '',
              weight: FontWeight.MEDIUM,
              x:
                pos.x +
                Math.sin(angle) *
                  (-edgeLabelFontSize * 1.5 - extraOffsetToEdge),
              y,
              z:
                pos.y +
                Math.cos(angle) *
                  (-edgeLabelFontSize * 1.5 - extraOffsetToEdge),
              color,
              angle,
              edgeTextMode: true,
              borderColor: {r: 1, g: 1, b: 1},
            });
          }
        }
      }
    }

    return labels;
  }

  updateAnimationProgress(t: number) {
    this.edgeTexts.updateAnimationProgress(t);
  }
}
