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
import * as d3 from 'd3';
import * as three from 'three';

import {WEBGL_ELEMENT_Y_FACTOR} from './common/consts';
import {GroupNode, ModelEdge, ModelNode, OpNode} from './common/model_graph';
import {FontWeight, LayoutDirection, Point} from './common/types';
import {
  findCommonNamespace,
  generateCurvePoints,
  getLayoutDirection,
  getShowOnEdgeInputOutputMetadataKeys,
  isGroupNode,
  isOpNode,
} from './common/utils';
import {ThreejsService} from './threejs_service';
import {WebglEdges} from './webgl_edges';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';
import {
  RoundedRectangleData,
  WebglRoundedRectangles,
} from './webgl_rounded_rectangles';
import {LabelData, WebglTexts} from './webgl_texts';

const EDGE_WIDTH_IO_HIGHLIGHT = 1.5;
/** The separator between the io picker id and the type of the io. */
export const IO_PICKER_ID_SEP = '||||';
/** The height of the io picker bg. */
export const IO_PICKER_HEIGHT = 14;
const IO_PICKER_WIDTH = 40;

const THREE = three;

/** An overlay model edge. */
interface OverlayModelEdge extends ModelEdge {
  type: 'incoming' | 'outgoing';
}

/** Service for managing input/output highlighting related tasks. */
@Injectable()
export class WebglRendererIoHighlightService {
  readonly EDGE_COLOR_INCOMING = new THREE.Color('#009e73');
  readonly EDGE_TEXT_COLOR_INCOMING = new THREE.Color('#125341');
  readonly EDGE_COLOR_OUTGOING = new THREE.Color('#d55e00');
  readonly EDGE_TEXT_COLOR_OUTGOING = new THREE.Color('#994d11');

  inputsRenderedEdges: ModelEdge[] = [];
  outputsRenderedEdges: ModelEdge[] = [];
  inputsByHighlightedNode: Record<string, OpNode[]> = {};
  outputsByHighlightedNode: Record<string, OpNode[]> = {};

  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private readonly threejsService: ThreejsService = inject(ThreejsService);
  readonly ioPickerBgs = new WebglRoundedRectangles(99);
  private readonly ioPickerTexts = new WebglTexts(this.threejsService);
  private readonly incomingHighlightedEdges = new WebglEdges(
    this.EDGE_COLOR_INCOMING,
    EDGE_WIDTH_IO_HIGHLIGHT,
  );
  private readonly outgoingHighlightedEdges = new WebglEdges(
    this.EDGE_COLOR_OUTGOING,
    EDGE_WIDTH_IO_HIGHLIGHT,
  );
  private readonly incomingHighlightedEdgeTexts = new WebglTexts(
    this.threejsService,
  );
  private readonly outgoingHighlightedEdgeTexts = new WebglTexts(
    this.threejsService,
  );

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  updateIncomingAndOutgoingHighlights() {
    if (!this.webglRenderer.curModelGraph) {
      return;
    }

    this.clearIncomingAndOutgoingHighlights();

    if (!this.shouldUpdateIncomingAndOutgoingEdgesHighlights()) {
      this.incomingHighlightedEdges.clearSavedDataForAnimation();
      this.outgoingHighlightedEdges.clearSavedDataForAnimation();
      this.incomingHighlightedEdgeTexts.clearSavedDataForAnimation();
      this.outgoingHighlightedEdgeTexts.clearSavedDataForAnimation();
      this.ioPickerBgs.clearSavedDataForAnimation();
      this.ioPickerTexts.clearSavedDataForAnimation();
      return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Incoming edges and nodes.
    //
    // tslint:disable-next-line:no-any
    const showOpNodeOutOfLayerEdgesWithoutSelecting =
      this.webglRenderer.appService.config()
        ?.showOpNodeOutOfLayerEdgesWithoutSelecting;
    const incoming = this.getHighlightedIncomingNodesAndEdges(
      this.webglRenderer.curHiddenInputOpNodeIds,
      undefined,
      {
        reuseRenderedEdgeCurvePoints: showOpNodeOutOfLayerEdgesWithoutSelecting,
      },
    );
    if (incoming.overlayEdges.length > 0) {
      const edges: Array<{edge: ModelEdge; index: number}> =
        incoming.overlayEdges.map((edge) => {
          const fromNodeId = edge.fromNodeId;
          const fromNode = this.webglRenderer.curModelGraph.nodesById[
            fromNodeId
          ] as OpNode;
          const layoutDirection = getLayoutDirection(
            this.webglRenderer.curModelGraph,
            fromNode.nsParentId ?? '',
          );
          return showOpNodeOutOfLayerEdgesWithoutSelecting
            ? {
                edge,
                index: 95 / WEBGL_ELEMENT_Y_FACTOR,
              }
            : {
                edge: {
                  ...edge,
                  curvePoints: generateCurvePoints(
                    edge.points,
                    d3.line,
                    layoutDirection === LayoutDirection.TOP_BOTTOM
                      ? d3.curveMonotoneY
                      : d3.curveMonotoneX,
                    THREE,
                    layoutDirection === LayoutDirection.TOP_BOTTOM,
                  ),
                },
                index: 95 / WEBGL_ELEMENT_Y_FACTOR,
              };
        });
      this.incomingHighlightedEdges.generateMesh(
        edges,
        this.webglRenderer.curModelGraph,
      );
      this.webglRendererThreejsService.addToScene(
        this.incomingHighlightedEdges.edgesMesh,
      );
      this.webglRendererThreejsService.addToScene(
        this.incomingHighlightedEdges.arrowHeadsMesh,
      );

      // Edge texts.
      const {
        outputMetadataKey,
        inputMetadataKey,
        sourceNodeAttrKey,
        targetNodeAttrKey,
      } = getShowOnEdgeInputOutputMetadataKeys(
        this.webglRenderer.curShowOnEdgeItem,
      );
      if (
        outputMetadataKey != null ||
        inputMetadataKey != null ||
        sourceNodeAttrKey != null ||
        targetNodeAttrKey != null
      ) {
        const labels =
          this.webglRenderer.webglRendererEdgeTextsService.genLabelsOnEdges(
            edges,
            this.EDGE_TEXT_COLOR_INCOMING,
            0,
            95,
            undefined,
            outputMetadataKey,
            inputMetadataKey,
            sourceNodeAttrKey,
            targetNodeAttrKey,
          );
        this.incomingHighlightedEdgeTexts.generateMesh(
          labels,
          false,
          true,
          true,
        );
        this.webglRendererThreejsService.addToScene(
          this.incomingHighlightedEdgeTexts.mesh,
        );
      }
    }
    this.inputsByHighlightedNode = incoming.inputsByHighlightedNode;
    this.inputsRenderedEdges = incoming.renderedEdges;

    ////////////////////////////////////////////////////////////////////////////
    // Outgoing edges and nodes.
    //
    const outgoing = this.getHighlightedOutgoingNodesAndEdges(
      this.webglRenderer.curHiddenOutputIds,
      undefined,
      {
        reuseRenderedEdgeCurvePoints: showOpNodeOutOfLayerEdgesWithoutSelecting,
      },
    );
    if (outgoing.overlayEdges.length > 0) {
      const edges = outgoing.overlayEdges.map((edge) => {
        const fromNodeId = edge.fromNodeId;
        const fromNode = this.webglRenderer.curModelGraph.nodesById[
          fromNodeId
        ] as OpNode;
        const layoutDirection = getLayoutDirection(
          this.webglRenderer.curModelGraph,
          fromNode.nsParentId ?? '',
        );
        return showOpNodeOutOfLayerEdgesWithoutSelecting
          ? {
              edge,
              index: 95 / WEBGL_ELEMENT_Y_FACTOR,
            }
          : {
              edge: {
                ...edge,
                curvePoints: generateCurvePoints(
                  edge.points,
                  d3.line,
                  layoutDirection === LayoutDirection.TOP_BOTTOM
                    ? d3.curveMonotoneY
                    : d3.curveMonotoneX,
                  THREE,
                  layoutDirection === LayoutDirection.TOP_BOTTOM,
                ),
              },
              index: 95 / WEBGL_ELEMENT_Y_FACTOR,
            };
      });
      this.outgoingHighlightedEdges.generateMesh(
        edges,
        this.webglRenderer.curModelGraph,
      );
      this.webglRendererThreejsService.addToScene(
        this.outgoingHighlightedEdges.edgesMesh,
      );
      this.webglRendererThreejsService.addToScene(
        this.outgoingHighlightedEdges.arrowHeadsMesh,
      );

      // Edge texts.
      const {
        outputMetadataKey,
        inputMetadataKey,
        sourceNodeAttrKey,
        targetNodeAttrKey,
      } = getShowOnEdgeInputOutputMetadataKeys(
        this.webglRenderer.curShowOnEdgeItem,
      );
      if (
        outputMetadataKey != null ||
        inputMetadataKey != null ||
        sourceNodeAttrKey != null ||
        targetNodeAttrKey != null
      ) {
        const labels =
          this.webglRenderer.webglRendererEdgeTextsService.genLabelsOnEdges(
            edges,
            this.EDGE_TEXT_COLOR_OUTGOING,
            undefined,
            95,
            undefined,
            outputMetadataKey,
            inputMetadataKey,
            sourceNodeAttrKey,
            targetNodeAttrKey,
          );
        this.outgoingHighlightedEdgeTexts.generateMesh(
          labels,
          false,
          true,
          true,
        );
        this.webglRendererThreejsService.addToScene(
          this.outgoingHighlightedEdgeTexts.mesh,
        );
      }
    }
    this.outputsByHighlightedNode = outgoing.outputsByHighlightedNode;
    this.outputsRenderedEdges = outgoing.renderedEdges;

    // Io picker bgs and texts.
    const ioPickerBgRectangles: RoundedRectangleData[] = [];
    const ioPickerLabels: LabelData[] = [];
    for (const nodeId of Object.keys({
      ...this.inputsByHighlightedNode,
      ...this.outputsByHighlightedNode,
    })) {
      const node = this.webglRenderer.curModelGraph.nodesById[nodeId];
      if (isGroupNode(node)) {
        const width = IO_PICKER_WIDTH;
        const height = IO_PICKER_HEIGHT;
        const isInput = this.inputsByHighlightedNode[nodeId] != null;
        const numIOs = isInput
          ? this.inputsByHighlightedNode[nodeId].length
          : this.outputsByHighlightedNode[nodeId].length;
        ioPickerBgRectangles.push({
          id: `${nodeId}${IO_PICKER_ID_SEP}${isInput ? 'input' : 'output'}`,
          index: ioPickerBgRectangles.length,
          bound: {
            x: this.webglRenderer.getNodeX(node) + width / 2,
            y: this.webglRenderer.getNodeY(node) - height / 4,
            width,
            height,
          },
          yOffset: 95,
          isRounded: true,
          borderColor: {r: 1, g: 1, b: 1},
          bgColor: isInput
            ? this.EDGE_COLOR_INCOMING
            : this.EDGE_COLOR_OUTGOING,
          borderWidth: 0,
          opacity: 1,
        });
        ioPickerLabels.push({
          id: `${nodeId}${IO_PICKER_ID_SEP}${isInput ? 'input' : 'output'}`,
          nodeId,
          label: `${numIOs} ${isInput ? 'input' : 'output'}${
            numIOs !== 1 ? 's' : ''
          }`,
          height: 8,
          hAlign: 'center',
          vAlign: 'center',
          weight: FontWeight.MEDIUM,
          color: {r: 1, g: 1, b: 1},
          x: this.webglRenderer.getNodeX(node) + width / 2,
          y: 96,
          z: this.webglRenderer.getNodeY(node) - height / 4 + 1,
        });
      }
    }
    this.ioPickerTexts.generateMesh(ioPickerLabels, false, true, true);
    this.webglRendererThreejsService.addToScene(this.ioPickerTexts.mesh);
    this.ioPickerBgs.generateMesh(
      ioPickerBgRectangles,
      true,
      false,
      false,
      true,
    );
    this.webglRendererThreejsService.addToScene(this.ioPickerBgs.mesh);
    this.webglRendererThreejsService.addToScene(
      this.ioPickerBgs.meshForRayCasting,
    );

    this.webglRenderer.animateIntoPositions((t) => {
      this.incomingHighlightedEdges.updateAnimationProgress(t);
      this.outgoingHighlightedEdges.updateAnimationProgress(t);
      this.incomingHighlightedEdgeTexts.updateAnimationProgress(t);
      this.outgoingHighlightedEdgeTexts.updateAnimationProgress(t);
      this.ioPickerBgs.updateAnimationProgress(t);
      this.ioPickerTexts.updateAnimationProgress(t);
    });
  }

  handleClickIoPicker(isInput: boolean, nodeId: string) {
    if (isInput) {
      if (this.inputsByHighlightedNode[nodeId].length === 1) {
        this.webglRenderer.sendLocateNodeRequest(
          this.inputsByHighlightedNode[nodeId][0].id,
          this.webglRenderer.rendererId,
        );
      } else {
        this.webglRenderer.showIoTree(
          this.webglRenderer.ioPicker.nativeElement,
          this.inputsByHighlightedNode[nodeId],
          'incoming',
        );
      }
    } else {
      if (this.outputsByHighlightedNode[nodeId].length === 1) {
        this.webglRenderer.sendLocateNodeRequest(
          this.outputsByHighlightedNode[nodeId][0].id,
          this.webglRenderer.rendererId,
        );
      } else {
        this.webglRenderer.showIoTree(
          this.webglRenderer.ioPicker.nativeElement,
          this.outputsByHighlightedNode[nodeId],
          'outgoing',
        );
      }
    }
  }

  getHighlightedIncomingNodesAndEdges(
    hiddenInputNodeIds: Record<string, boolean>,
    selectedNode?: ModelNode,
    options?: {
      ignoreEdgesWithinSameNamespace?: boolean;
      reuseRenderedEdgeCurvePoints?: boolean;
    },
  ) {
    const ignoreEdgesWithinSameNamespace =
      options?.ignoreEdgesWithinSameNamespace ?? false;
    const reuseRenderedEdgeCurvePoints =
      options?.reuseRenderedEdgeCurvePoints ?? false;

    if (!selectedNode) {
      selectedNode =
        this.webglRenderer.curModelGraph.nodesById[
          this.webglRenderer.selectedNodeId
        ];
    }
    const renderedEdges: ModelEdge[] = [];
    const highlightedNodes: ModelNode[] = [];
    const inputsByHighlightedNode: Record<string, OpNode[]> = {};
    const overlayEdges: OverlayModelEdge[] = [];

    const opNodes: OpNode[] = [];
    const ignoredIncomingNodesIds = new Set<string>();
    const seenIncomingNodesIds = new Set<string>();
    if (isOpNode(selectedNode)) {
      opNodes.push(selectedNode);
    } else if (isGroupNode(selectedNode)) {
      for (const id of selectedNode.descendantsOpNodeIds || []) {
        const node = this.webglRenderer.curModelGraph.nodesById[id] as OpNode;
        opNodes.push(node);
        ignoredIncomingNodesIds.add(id);
      }
    }

    for (const opNode of opNodes) {
      for (const incomingEdge of opNode.incomingEdges || []) {
        if (hiddenInputNodeIds[incomingEdge.sourceNodeId]) {
          continue;
        }
        const sourceNode = this.webglRenderer.curModelGraph.nodesById[
          incomingEdge.sourceNodeId
        ] as OpNode;
        if (!sourceNode) {
          continue;
        }

        if (ignoredIncomingNodesIds.has(sourceNode.id)) {
          continue;
        }

        if (seenIncomingNodesIds.has(sourceNode.id)) {
          continue;
        }
        seenIncomingNodesIds.add(sourceNode.id);

        if (
          ignoreEdgesWithinSameNamespace &&
          sourceNode.namespace === opNode.namespace
        ) {
          continue;
        }

        // Find the common namespace prefix.
        const commonNamespace = findCommonNamespace(
          sourceNode.namespace,
          opNode.namespace,
        );

        // Go from the given node to all its ns ancestors, find the last collapsed
        // node before reaching the given namespace. If all ancestor nodes are
        // expanded, return the given node.
        const highlightedNode = this.getLastCollapsedAncestorNode(
          sourceNode,
          commonNamespace,
        );
        highlightedNodes.push(highlightedNode);

        // Update inputsByHighlighedNode.
        if (inputsByHighlightedNode[highlightedNode.id] == null) {
          inputsByHighlightedNode[highlightedNode.id] = [];
        }
        inputsByHighlightedNode[highlightedNode.id].push(sourceNode);

        // Find the existing edge in the common namespace that connects two
        // nodes n1 and n2 where n1 contains `sourceNode` and n2 contains
        // `node`.
        const renderedEdge = this.findEdgeConnectingTwoNodesInNamespace(
          commonNamespace,
          sourceNode.id,
          opNode.id,
        );

        // Start to construct an edge from the source node to the selected node.
        //
        const points: Point[] = [];
        const curvePoints: Point[] = [];

        if (renderedEdge) {
          renderedEdges.push(renderedEdge);
          const renderedEdgeCurvePoints = renderedEdge.curvePoints || [];

          // Add a point from the highlighted node that connects to the first
          // point of the rendered edge above.
          const renderedEdgeFromNode =
            this.webglRenderer.curModelGraph.nodesById[renderedEdge.fromNodeId];
          if (renderedEdge.fromNodeId !== highlightedNode.id) {
            const renderedEdgeStartX =
              renderedEdge.points[0].x + (renderedEdgeFromNode.globalX || 0);
            const renderedEdgeStartY =
              renderedEdge.points[0].y + (renderedEdgeFromNode.globalY || 0);
            const startPt = this.getBestAnchorPointOnNode(
              renderedEdgeStartX,
              renderedEdgeStartY,
              highlightedNode,
            );
            points.push({
              x: startPt.x - (highlightedNode.globalX || 0),
              y: startPt.y - (highlightedNode.globalY || 0),
            });
            if (reuseRenderedEdgeCurvePoints) {
              curvePoints.push(
                {
                  x: startPt.x - (highlightedNode.globalX || 0),
                  y: startPt.y - (highlightedNode.globalY || 0),
                },
                {
                  x:
                    renderedEdgeCurvePoints[0].x -
                    (highlightedNode.globalX || 0) +
                    (renderedEdgeFromNode.globalX || 0),
                  y:
                    renderedEdgeCurvePoints[0].y -
                    (highlightedNode.globalY || 0) +
                    (renderedEdgeFromNode.globalY || 0),
                },
              );
            }
          }

          // Add the points in rendered edge.
          let targetPoints: Point[] = points;
          let sourcePoints: Point[] = renderedEdge.points;
          if (reuseRenderedEdgeCurvePoints) {
            targetPoints = curvePoints;
            sourcePoints = renderedEdgeCurvePoints;
          }
          targetPoints.push(
            ...sourcePoints.map((pt) => {
              return {
                x:
                  pt.x -
                  (highlightedNode.globalX || 0) +
                  (renderedEdgeFromNode.globalX || 0),
                y:
                  pt.y -
                  (highlightedNode.globalY || 0) +
                  (renderedEdgeFromNode.globalY || 0),
              };
            }),
          );

          // Add a point from the selected node that connects to the last point of
          // the rendered edge.
          if (renderedEdge.toNodeId !== opNode?.id && isOpNode(selectedNode)) {
            const renderedEdgeLastX =
              renderedEdge.points[renderedEdge.points.length - 1].x +
              (renderedEdgeFromNode.globalX || 0);
            const renderedEdgeLastY =
              renderedEdge.points[renderedEdge.points.length - 1].y +
              (renderedEdgeFromNode.globalY || 0);
            const endPt = this.getBestAnchorPointOnNode(
              renderedEdgeLastX,
              renderedEdgeLastY,
              opNode,
            );
            if (!reuseRenderedEdgeCurvePoints) {
              points.push({
                x: endPt.x - (highlightedNode.globalX || 0),
                y: endPt.y - (highlightedNode.globalY || 0),
              });
            } else {
              curvePoints.push(
                {
                  x:
                    renderedEdgeCurvePoints[renderedEdgeCurvePoints.length - 1]
                      .x -
                    (highlightedNode.globalX || 0) +
                    (renderedEdgeFromNode.globalX || 0),
                  y:
                    renderedEdgeCurvePoints[renderedEdgeCurvePoints.length - 1]
                      .y -
                    (highlightedNode.globalY || 0) +
                    (renderedEdgeFromNode.globalY || 0),
                },
                {
                  x: endPt.x - (highlightedNode.globalX || 0),
                  y: endPt.y - (highlightedNode.globalY || 0),
                },
              );
            }
          }
        } else if (
          isGroupNode(highlightedNode) ||
          (isOpNode(highlightedNode) && !highlightedNode.hideInLayout)
        ) {
          (reuseRenderedEdgeCurvePoints ? curvePoints : points).push(
            ...this.getDirectEdgeBetweenNodes(highlightedNode, opNode),
          );
        }

        // Use these points to form an edge and add it as an overlay edge.
        if (!reuseRenderedEdgeCurvePoints) {
          if (points.length > 0) {
            overlayEdges.push({
              id: `overlay_${highlightedNode.id}___${opNode.id}`,
              fromNodeId: highlightedNode.id,
              toNodeId: opNode.id,
              points,
              type: 'incoming',
            });
          }
        } else {
          if (curvePoints.length > 0) {
            overlayEdges.push({
              id: `overlay_${highlightedNode.id}___${opNode.id}`,
              fromNodeId: highlightedNode.id,
              toNodeId: opNode.id,
              points: [],
              curvePoints,
              type: 'incoming',
            });
          }
        }
      }
    }

    return {
      renderedEdges,
      highlightedNodes,
      inputsByHighlightedNode,
      overlayEdges,
    };
  }

  getHighlightedOutgoingNodesAndEdges(
    hiddenOutputIds: Record<string, boolean>,
    selectedNode?: ModelNode,
    options?: {
      ignoreEdgesWithinSameNamespace?: boolean;
      reuseRenderedEdgeCurvePoints?: boolean;
    },
  ) {
    const ignoreEdgesWithinSameNamespace =
      options?.ignoreEdgesWithinSameNamespace ?? false;
    const reuseRenderedEdgeCurvePoints =
      options?.reuseRenderedEdgeCurvePoints ?? false;

    if (!selectedNode) {
      selectedNode =
        this.webglRenderer.curModelGraph.nodesById[
          this.webglRenderer.selectedNodeId
        ];
    }
    const renderedEdges: ModelEdge[] = [];
    const highlightedNodes: ModelNode[] = [];
    const outputsByHighlightedNode: Record<string, OpNode[]> = {};
    const overlayEdges: OverlayModelEdge[] = [];

    const opNodes: OpNode[] = [];
    const ignoredOutgoingNodesIds = new Set<string>();
    const seenOutgoingNodesIds = new Set<string>();
    if (isOpNode(selectedNode)) {
      opNodes.push(selectedNode);
    } else if (isGroupNode(selectedNode)) {
      for (const id of selectedNode.descendantsOpNodeIds || []) {
        const node = this.webglRenderer.curModelGraph.nodesById[id] as OpNode;
        opNodes.push(node);
        ignoredOutgoingNodesIds.add(id);
      }
    }

    for (const opNode of opNodes) {
      for (const outgoingEdges of opNode.outgoingEdges || []) {
        if (
          hiddenOutputIds[`${opNode.id}___${outgoingEdges.sourceNodeOutputId}`]
        ) {
          continue;
        }

        const targetNode = this.webglRenderer.curModelGraph.nodesById[
          outgoingEdges.targetNodeId
        ] as OpNode;
        if (!targetNode) {
          continue;
        }

        if (ignoredOutgoingNodesIds.has(targetNode.id)) {
          continue;
        }

        if (seenOutgoingNodesIds.has(targetNode.id)) {
          continue;
        }
        seenOutgoingNodesIds.add(targetNode.id);

        if (
          ignoreEdgesWithinSameNamespace &&
          targetNode.namespace === opNode.namespace
        ) {
          continue;
        }

        // Find the common namespace prefix.
        const commonNamespace = findCommonNamespace(
          targetNode.namespace,
          opNode.namespace,
        );

        // Go from the given node to all its ns ancestors, find the last
        // collapsed node before reaching the given namespace, and style it with
        // the given class. If all ancestor nodes are expanded, style the given
        // node.
        const highlightedNode = this.getLastCollapsedAncestorNode(
          targetNode,
          commonNamespace,
        );
        highlightedNodes.push(highlightedNode);

        // Update outputsByHighlighedNode.
        if (outputsByHighlightedNode[highlightedNode.id] == null) {
          outputsByHighlightedNode[highlightedNode.id] = [];
        }
        outputsByHighlightedNode[highlightedNode.id].push(targetNode);

        // Find the existing edge in the common namespace that connects two
        // nodes n1 and n2 where n1 contains `sourceNode` and n2 contains
        // `node`.
        const renderedEdge = this.findEdgeConnectingTwoNodesInNamespace(
          commonNamespace,
          opNode.id,
          targetNode.id,
        );

        // Start to construct an edge from the selected node to target node.
        //
        const points: Point[] = [];
        const curvePoints: Point[] = [];

        if (renderedEdge) {
          renderedEdges.push(renderedEdge);
          const renderedEdgeCurvePoints = renderedEdge.curvePoints || [];

          const renderedEdgeFromNode =
            this.webglRenderer.curModelGraph.nodesById[renderedEdge.fromNodeId];

          // Add a point from the selected node that connects to the first point
          // of the rendered edge.
          if (isOpNode(selectedNode)) {
            if (renderedEdge.fromNodeId !== opNode?.id) {
              const renderedEdgeStartX =
                renderedEdge.points[0].x + (renderedEdgeFromNode.globalX || 0);
              const renderedEdgeStartY =
                renderedEdge.points[0].y + (renderedEdgeFromNode.globalY || 0);
              const endPt = this.getBestAnchorPointOnNode(
                renderedEdgeStartX,
                renderedEdgeStartY,
                opNode,
              );
              points.push({
                x: endPt.x - (opNode.globalX || 0),
                y: endPt.y - (opNode.globalY || 0),
              });
              if (reuseRenderedEdgeCurvePoints) {
                curvePoints.push(
                  {
                    x: endPt.x - (opNode.globalX || 0),
                    y: endPt.y - (opNode.globalY || 0),
                  },
                  {
                    x:
                      renderedEdgeCurvePoints[0].x -
                      (opNode.globalX || 0) +
                      (renderedEdgeFromNode.globalX || 0),
                    y:
                      renderedEdgeCurvePoints[0].y -
                      (opNode.globalY || 0) +
                      (renderedEdgeFromNode.globalY || 0),
                  },
                );
              }
            }
          }

          // Add the points in rendered edge.
          let targetPoints: Point[] = points;
          let sourcePoints: Point[] = renderedEdge.points;
          if (reuseRenderedEdgeCurvePoints) {
            targetPoints = curvePoints;
            sourcePoints = renderedEdgeCurvePoints;
          }
          targetPoints.push(
            ...sourcePoints.map((pt) => {
              return {
                x:
                  pt.x -
                  (opNode.globalX || 0) +
                  (renderedEdgeFromNode.globalX || 0),
                y:
                  pt.y -
                  (opNode.globalY || 0) +
                  (renderedEdgeFromNode.globalY || 0),
              };
            }),
          );

          // Add a point from the highlighted node that connects to the first
          // point of the rendered edge above.
          if (renderedEdge.toNodeId !== highlightedNode.id) {
            const renderedEdgeLastX =
              renderedEdge.points[renderedEdge.points.length - 1].x +
              (renderedEdgeFromNode.globalX || 0);
            const renderedEdgeLastY =
              renderedEdge.points[renderedEdge.points.length - 1].y +
              (renderedEdgeFromNode.globalY || 0);
            const startPt = this.getBestAnchorPointOnNode(
              renderedEdgeLastX,
              renderedEdgeLastY,
              highlightedNode,
            );
            if (!reuseRenderedEdgeCurvePoints) {
              points.push({
                x: startPt.x - (opNode.globalX || 0),
                y: startPt.y - (opNode.globalY || 0),
              });
            } else {
              curvePoints.push(
                {
                  x:
                    renderedEdgeCurvePoints[renderedEdgeCurvePoints.length - 1]
                      .x -
                    (opNode.globalX || 0) +
                    (renderedEdgeFromNode.globalX || 0),
                  y:
                    renderedEdgeCurvePoints[renderedEdgeCurvePoints.length - 1]
                      .y -
                    (opNode.globalY || 0) +
                    (renderedEdgeFromNode.globalY || 0),
                },
                {
                  x: startPt.x - (opNode.globalX || 0),
                  y: startPt.y - (opNode.globalY || 0),
                },
              );
            }
          }
        } else if (
          isGroupNode(highlightedNode) ||
          (isOpNode(highlightedNode) && !highlightedNode.hideInLayout)
        ) {
          (reuseRenderedEdgeCurvePoints ? curvePoints : points).push(
            ...this.getDirectEdgeBetweenNodes(opNode, highlightedNode),
          );
        }

        // Use these points to form an edge and add it as an overlay edge.
        if (!reuseRenderedEdgeCurvePoints) {
          if (points.length > 0) {
            overlayEdges.push({
              id: `overlay_${opNode.id}___${highlightedNode.id}`,
              fromNodeId: opNode.id,
              toNodeId: highlightedNode.id,
              points,
              type: 'outgoing',
            });
          }
        } else {
          if (curvePoints.length > 0) {
            overlayEdges.push({
              id: `overlay_${opNode.id}___${highlightedNode.id}`,
              fromNodeId: opNode.id,
              toNodeId: highlightedNode.id,
              points: [],
              curvePoints,
              type: 'outgoing',
            });
          }
        }
      }
    }

    return {
      renderedEdges,
      highlightedNodes,
      outputsByHighlightedNode,
      overlayEdges,
    };
  }

  /**
   * Go from the given node to all its ns ancestors, find the last collapsed
   * node before reaching the given namespace. If all ancestor nodes are
   * expanded, return the given node.
   */
  private getLastCollapsedAncestorNode(
    node: ModelNode,
    namespace: string,
  ): ModelNode {
    let curNode: ModelNode = node;
    const collapsedNodes: GroupNode[] = [];
    while (curNode) {
      if (isGroupNode(curNode) && !curNode.expanded) {
        collapsedNodes.push(curNode);
      }
      if (curNode.namespace === namespace) {
        break;
      }
      curNode =
        this.webglRenderer.curModelGraph.nodesById[curNode.nsParentId || ''];
    }
    const targetNode =
      collapsedNodes.length > 0
        ? collapsedNodes[collapsedNodes.length - 1]
        : node;
    return targetNode;
  }

  private shouldUpdateIncomingAndOutgoingEdgesHighlights() {
    // Ignore when clicking on empty space.
    if (!this.webglRenderer.selectedNodeId) {
      return false;
    }

    // Ignore when clicking on a group node and the corresponding config option
    // is not enabled.
    const selectedNode =
      this.webglRenderer.curModelGraph.nodesById[
        this.webglRenderer.selectedNodeId
      ];
    if (
      isGroupNode(selectedNode) &&
      !this.webglRenderer.appService.config()?.highlightLayerNodeInputsOutputs
    ) {
      return false;
    }

    // Ignore if the selected node is not in the given root node (if set).
    const rootNode =
      this.webglRenderer.curModelGraph.nodesById[
        this.webglRenderer.rootNodeId || ''
      ];
    if (
      rootNode &&
      isGroupNode(rootNode) &&
      !(rootNode.descendantsOpNodeIds || []).includes(
        this.webglRenderer.selectedNodeId,
      )
    ) {
      return false;
    }

    // Ignore if the selected node is not rendered.
    if (!this.webglRenderer.isNodeRendered(this.webglRenderer.selectedNodeId)) {
      return false;
    }

    return true;
  }

  private clearIncomingAndOutgoingHighlights() {
    this.incomingHighlightedEdges.clear();
    this.outgoingHighlightedEdges.clear();
    this.inputsByHighlightedNode = {};
    this.outputsByHighlightedNode = {};
    this.inputsRenderedEdges = [];
    this.outputsRenderedEdges = [];

    for (const mesh of [
      this.ioPickerBgs.mesh,
      this.ioPickerBgs.meshForRayCasting,
      this.ioPickerTexts.mesh,
      this.incomingHighlightedEdgeTexts.mesh,
      this.outgoingHighlightedEdgeTexts.mesh,
    ]) {
      if (!mesh) {
        continue;
      }
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      this.webglRendererThreejsService.removeFromScene(mesh);
    }
    // This stops the raycasting logic from reacting to non-existent
    // ioPickerBgs.mesh.
    this.ioPickerBgs.meshForRayCasting = undefined;
  }

  /**
   * Given source and target node id, find the edge in the given namespace that
   * connects to n1 and n2 where source node is within n1 and target node is
   * within n2.
   */
  private findEdgeConnectingTwoNodesInNamespace(
    namespace: string,
    sourceNodeId: string,
    targetNodeId: string,
  ): ModelEdge | undefined {
    const groupNodeId = namespace === '' ? '' : `${namespace}___group___`;
    return (
      this.webglRenderer.curModelGraph.edgesByGroupNodeIds[groupNodeId] ?? []
    ).find((edge) => {
      const fromNode =
        this.webglRenderer.curModelGraph.nodesById[edge.fromNodeId];
      const toNode = this.webglRenderer.curModelGraph.nodesById[edge.toNodeId];
      const fromNodeContainsSourceNode = this.containNode(
        fromNode,
        sourceNodeId,
      );
      const toNodeContainsTargetNode = this.containNode(toNode, targetNodeId);
      return fromNodeContainsSourceNode && toNodeContainsTargetNode;
    });
  }

  private containNode(parentNode: ModelNode, targetNodeId: string): boolean {
    return (
      (isOpNode(parentNode) && parentNode.id === targetNodeId) ||
      (isGroupNode(parentNode) &&
        (parentNode.descendantsOpNodeIds || []).includes(targetNodeId))
    );
  }

  private getDirectEdgeBetweenNodes(
    startNode: ModelNode,
    endNode: ModelNode,
  ): Point[] {
    const points: Point[] = [];

    const startX = startNode.globalX || 0;
    const startY = startNode.globalY || 0;
    const startWidth = startNode.width || 0;
    const startHeight = startNode.height || 0;
    const endX = endNode.globalX || 0;
    const endY = endNode.globalY || 0;
    const endWidth = endNode.width || 0;
    const endHeight = endNode.height || 0;

    const startAnchorX = startX + startWidth / 2;
    const startAnchorY = endY > startY ? startY + startHeight : startY;
    const endAnchorX = endX + endWidth / 2;
    const endAnchorY = endY > startY ? endY : endY + endHeight;

    points.push(
      {
        x: startAnchorX + (startNode.x || 0) - startX,
        y: startAnchorY + (startNode.y || 0) - startY,
      },
      {
        x: endAnchorX + (endNode.x || 0) - startX,
        y: endAnchorY + (endNode.y || 0) - startY,
      },
    );

    return points;
  }

  private getBestAnchorPointOnNode(
    startX: number,
    startY: number,
    node: ModelNode,
  ): Point {
    const nodeX = this.webglRenderer.getNodeX(node);
    const nodeY = this.webglRenderer.getNodeY(node);
    const nodeWidth = this.webglRenderer.getNodeWidth(node);
    const nodeHeight = this.webglRenderer.getNodeHeight(node);
    const items = [
      // top-middle
      {
        point: {x: nodeX + nodeWidth / 2, y: nodeY},
        distance: 0,
        direction: 'horizontal',
      },
      // right-middle
      {
        point: {x: nodeX + nodeWidth, y: nodeY + nodeHeight / 2},
        distance: 0,
        direction: 'vertical',
      },
      // bottom-middle
      {
        point: {x: nodeX + nodeWidth / 2, y: nodeY + nodeHeight},
        distance: 0,
        direction: 'horizontal',
      },
      // left-middle
      {
        point: {x: nodeX, y: nodeY + nodeHeight / 2},
        distance: 0,
        direction: 'vertical',
      },
    ];
    for (const item of items) {
      item.distance = this.getDistanceSquared(
        startX,
        startY,
        item.point.x,
        item.point.y,
      );
    }
    items.sort((a, b) => a.distance - b.distance);
    if (items[0].direction !== items[1].direction) {
      const angle0 = this.getAngle(
        startX,
        startY,
        items[0].point.x,
        items[0].point.y,
        items[0].direction,
      );
      const angle1 = this.getAngle(
        startX,
        startY,
        items[1].point.x,
        items[1].point.y,
        items[1].direction,
      );
      return angle0 >= angle1 ? items[0].point : items[1].point;
    }
    return items[0].point;
  }

  private getAngle(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    direction: string,
  ): number {
    if (direction === 'horizontal') {
      return Math.atan(Math.abs(y2 - y1) / Math.abs(x2 - x1));
    } else {
      return Math.atan(Math.abs(x2 - x1) / Math.abs(y2 - y1));
    }
  }

  private getDistanceSquared(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): number {
    return Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
  }
}
