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

import {TrustedResourceUrl} from 'safevalues';

import {
  CATMULLROM_CURVE_TENSION,
  EXPORT_TO_RESOURCE_CMD,
  MAX_IO_ROWS_IN_ATTRS_TABLE,
  NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
  NODE_LABEL_LINE_HEIGHT,
  TENSOR_TAG_METADATA_KEY,
  TENSOR_VALUES_KEY,
  WEBGL_CURVE_SEGMENTS,
} from './consts';
import {
  GroupNode,
  ModelGraph,
  ModelNode,
  NodeType,
  OpNode,
} from './model_graph';
import {
  FieldLabel,
  KeyValueList,
  KeyValuePairs,
  LayoutDirection,
  NodeAttributeValueType,
  NodeDataProviderResultProcessedData,
  NodeDataProviderRunData,
  NodeDataProviderValueInfo,
  NodeQuery,
  NodeQueryType,
  NodeStyleId,
  NodeStylerRule,
  Point,
  ProcessedNodeQuery,
  ProcessedNodeRegexQuery,
  ProcessedNodeStylerRule,
  Rect,
  SearchMatch,
  SearchMatchType,
  SearchNodeType,
  ShowOnEdgeItemData,
  ShowOnEdgeItemType,
  ShowOnNodeItemData,
  ShowOnNodeItemType,
} from './types';
import {VisualizerConfig} from './visualizer_config';

const CANVAS = new OffscreenCanvas(300, 300);

/** Cache for label width indexed by label. */
const LABEL_WIDTHS: {[label: string]: number} = {};

/** Whether the current browser is Mac. */
export const IS_MAC =
  typeof navigator !== 'undefined' && /Macintosh/.test(navigator.userAgent);

/** Checks whether the given node is an op node. */
export function isOpNode(node: ModelNode | undefined): node is OpNode {
  return node?.nodeType === NodeType.OP_NODE;
}

/** Checks whether the given node is a group node. */
export function isGroupNode(node: ModelNode | undefined): node is GroupNode {
  return node?.nodeType === NodeType.GROUP_NODE;
}

/**
 * Checks whether the given node is a group node and it doesn't have any
 * children nodes.
 */
export function isGroupNodeWithoutChildren(node: ModelNode): boolean {
  return isGroupNode(node) && (node.nsChildrenIds || []).length === 0;
}

/** Gets the op node field labels from the given showOnNodeItemTypes. */
export function getOpNodeFieldLabelsFromShowOnNodeItemTypes(
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
): FieldLabel[] {
  const fieldIds: FieldLabel[] = [];
  for (const [itemType, itemData] of Object.entries(showOnNodeItemTypes)) {
    switch (itemType) {
      case ShowOnNodeItemType.OP_NODE_ID:
        if (itemData.selected) {
          fieldIds.push(FieldLabel.OP_NODE_ID);
        }
        break;
      default:
        break;
    }
  }
  return fieldIds;
}

/** Gets the group node field labels from the given showOnNodeItemTypes. */
export function getGroupNodeFieldLabelsFromShowOnNodeItemTypes(
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
): FieldLabel[] {
  const fieldIds: FieldLabel[] = [];
  for (const [itemType, itemData] of Object.entries(showOnNodeItemTypes)) {
    switch (itemType) {
      case ShowOnNodeItemType.LAYER_NODE_CHILDREN_COUNT:
        if (itemData.selected) {
          fieldIds.push(FieldLabel.NUMBER_OF_CHILDREN);
        }
        break;
      case ShowOnNodeItemType.LAYER_NODE_DESCENDANTS_COUNT:
        if (itemData.selected) {
          fieldIds.push(FieldLabel.NUMBER_OF_DESCENDANTS);
        }
        break;
      default:
        break;
    }
  }
  return fieldIds;
}

/**
 * Gets the value for the given field id of a node's info.
 */
export function getNodeInfoFieldValue(
  node: ModelNode,
  fieldId: string,
): string {
  if (isOpNode(node)) {
    switch (fieldId.toLowerCase()) {
      case FieldLabel.OP_NODE_ID:
        return node.id;
      case 'namespace':
        return getNamespaceLabel(node);
      default:
        break;
    }
  } else if (isGroupNode(node)) {
    switch (fieldId.toLowerCase()) {
      case 'namespace':
        return getNamespaceLabel(node);
      case FieldLabel.NUMBER_OF_CHILDREN:
        return String((node.nsChildrenIds || []).length);
      case FieldLabel.NUMBER_OF_DESCENDANTS:
        return String((node.descendantsNodeIds || []).length);
      default:
        break;
    }
  }
  return '';
}

/** Gets namespace display label. */
export function getNamespaceLabel(node: ModelNode): string {
  return node.fullNamespace || node.namespace || '<root>';
}

/** Generates unique id. */
export function genUid(): string {
  return Math.random().toString(36).slice(-6);
}

/** Gets the deepest expanded group node ids. */
export function getDeepestExpandedGroupNodeIds(
  root: GroupNode | undefined,
  modelGraph: ModelGraph,
  deepestExpandedGroupNodeIds: string[],
  ignoreExpandedState = false,
) {
  let nsChildrenIds: string[] = [];
  if (root == null) {
    nsChildrenIds = modelGraph.rootNodes.map((node) => node.id);
  } else {
    nsChildrenIds = root.nsChildrenIds || [];
  }
  for (const nsChildNodeId of nsChildrenIds) {
    const childNode = modelGraph.nodesById[nsChildNodeId];
    if (!childNode) {
      continue;
    }
    if (
      isGroupNode(childNode) &&
      (ignoreExpandedState || (!ignoreExpandedState && childNode.expanded))
    ) {
      const nsChildrenIds = childNode.nsChildrenIds || [];
      const isDeepest = ignoreExpandedState
        ? nsChildrenIds.filter((id) => isGroupNode(modelGraph.nodesById[id]))
            .length === 0
        : nsChildrenIds
            .filter((id) => isGroupNode(modelGraph.nodesById[id]))
            .every((id) => !(modelGraph.nodesById[id] as GroupNode).expanded);
      if (isDeepest) {
        deepestExpandedGroupNodeIds.push(childNode.id);
      }
      getDeepestExpandedGroupNodeIds(
        childNode,
        modelGraph,
        deepestExpandedGroupNodeIds,
        ignoreExpandedState,
      );
    }
  }
}

/** Gets the points from a smooth curve that go through the given points. */
export function generateCurvePoints(
  points: Point[],
  // tslint:disable-next-line:no-any Allow arbitrary types.
  d3Line: any,
  // tslint:disable-next-line:no-any Allow arbitrary types.
  d3CurveMonotone: any,
  // tslint:disable-next-line:no-any Allow arbitrary types.
  three: any,
  // true: vertical curve, false: horizontal curve.
  verticalOrHorizontal: boolean,
): Point[] {
  let curvePoints: Point[] = [];
  if (points.length === 2) {
    curvePoints = points;
  } else if (
    (points.length === 3 &&
      verticalOrHorizontal &&
      points[0].x === points[1].x &&
      points[1].x === points[2].x) ||
    (!verticalOrHorizontal &&
      points[0].y === points[1].y &&
      points[1].y === points[2].y)
  ) {
    curvePoints = points;
  } else {
    // Check if points are sorted by their X or Y coordinate.
    let isSorted = true;
    let curOrder = 0;
    for (let i = 0; i < points.length - 1; i++) {
      const curPt = points[i];
      const nextPt = points[i + 1];
      const order =
        (verticalOrHorizontal ? nextPt.y : nextPt.x) >
        (verticalOrHorizontal ? curPt.y : curPt.x)
          ? 1
          : -1;
      if (curOrder !== 0 && curOrder !== order) {
        isSorted = false;
        break;
      }
      curOrder = order;
    }

    // If points are sorted, use d3's curveMonotoneX/Y to generate curves and
    // convert them to a CurvePath in threejs. curveMonotoneX/Y looks better
    // then catmullrom curves.
    const vec3 = three['Vector3'];
    if (isSorted) {
      const d3Curve = d3Line()
        .x((d: Point) => d.x)
        .y((d: Point) => d.y)
        .curve(d3CurveMonotone)(points) as string;
      const parts = d3Curve
        .split(/M|C/)
        .filter((s) => s !== '')
        .map((s) => s.split(',').map((s) => Number(s)));
      let curStartPoint = new vec3(parts[0][0], parts[0][1], 0);
      const curvePath = new three['CurvePath']();
      for (let i = 1; i < parts.length; i++) {
        const curPart = parts[i];
        if (curPart.length === 6) {
          const ptStart = curStartPoint;
          const c1 = new vec3(curPart[0], curPart[1]);
          const c2 = new vec3(curPart[2], curPart[3]);
          const ptEnd = new vec3(curPart[4], curPart[5]);
          curStartPoint = ptEnd;
          const curve = new three['CubicBezierCurve3'](ptStart, c1, c2, ptEnd);
          curvePath.add(curve);
        }
      }
      curvePoints = curvePath['getPoints'](WEBGL_CURVE_SEGMENTS);
    }
    // Otherwise, use the catmullrom curve.
    else {
      const v3Points = points.map((point) => new vec3(point.x, point.y, 0));
      const curve = new three['CatmullRomCurve3'](
        v3Points,
        false,
        'catmullrom',
        CATMULLROM_CURVE_TENSION,
      );
      curvePoints = curve['getPoints'](WEBGL_CURVE_SEGMENTS);
    }
  }
  return curvePoints;
}

/** Checks whether the active element is an input element. */
export function inInputElement() {
  const activeEle = getActiveElement();
  if (!activeEle) {
    return false;
  }
  const isInputElement =
    activeEle.tagName === 'INPUT' ||
    activeEle.tagName === 'SELECT' ||
    activeEle.tagName === 'TEXTAREA' ||
    activeEle.contentEditable === 'true';
  return isInputElement;
}

function getActiveElement(
  root: Document | ShadowRoot = document,
): HTMLElement | null {
  const activeEl: HTMLElement | null = root.activeElement as HTMLElement;
  if (!activeEl) {
    return null;
  }
  if (activeEl.shadowRoot) {
    return getActiveElement(activeEl.shadowRoot);
  } else {
    return activeEl;
  }
}

/** Gets the label width by measureing its size in canvas. */
export function getLabelWidth(
  label: string,
  fontSize: number,
  bold: boolean,
  saveToCache = true,
): number {
  // Check cache first.
  const key = `${label}___${fontSize}___${bold}`;
  let labelWidth = LABEL_WIDTHS[key];
  if (labelWidth == null) {
    // On cache miss, render the text to a offscreen canvas to get its width.
    const context = CANVAS.getContext('2d')! as {} as CanvasRenderingContext2D;
    context.font = `${fontSize}px "Google Sans Text", Arial, Helvetica, sans-serif`;
    if (bold) {
      context.font = `bold ${context.font}`;
    }
    const metrics = context.measureText(label);
    const width = metrics.width;
    if (saveToCache) {
      LABEL_WIDTHS[key] = width;
    }
    labelWidth = width;
  }
  return labelWidth;
}

/** Gets the input label for the attrs table from the given node. */
export function getInputLabelForAttrsTable(
  index: number,
  node: OpNode,
  metadata: KeyValuePairs,
): string {
  const tensorTag = metadata[TENSOR_TAG_METADATA_KEY];
  return tensorTag
    ? `Input${index}:${tensorTag} (${node.label})`
    : `Input${index} (${node.label})`;
}

/** Gets the output label for the attrs table from the given node. */
export function getOutputLabelForAttrsTable(
  index: number,
  outputMetadata: KeyValuePairs,
  node: OpNode,
): string {
  let label = `Output${index}`;
  // Special handling for "GraphInputs".
  if (node.label === 'GraphInputs') {
    const tensorName = outputMetadata['tensor_name'];
    if (tensorName != null) {
      label = `${label} (${tensorName})`;
    }
  } else {
    const tensorTag = outputMetadata[TENSOR_TAG_METADATA_KEY];
    if (tensorTag) {
      label = `Output${index}:${tensorTag}`;
    }
  }
  return label;
}

/** Gets the shape for the attrs table from the given node. */
export function getShapeForAttrsTable(items?: KeyValuePairs): string {
  let shape = ((items || {})['shape'] || '')
    .replace(/ /g, '')
    .replace(/×/g, 'x');
  if (shape === '') {
    shape = '?';
  }
  return shape;
}

/** Gets the key value pairs for the given node's attrs for attrs table. */
export function getOpNodeAttrsKeyValuePairsForAttrsTable(
  node: OpNode,
  filterRegex = '',
) {
  const attrs = node.attrs || {};
  const keyValuePairs: KeyValueList = [];
  const regex = new RegExp(filterRegex, 'i');
  for (const attrId of Object.keys(attrs)) {
    const key = attrId;
    const value = attrs[attrId];
    if (typeof value === 'string') {
      const matchTargets = [`${key}:${value}`, `${key}=${value}`];
      if (
        filterRegex.trim() === '' ||
        matchTargets.some((matchTarget) => regex.test(matchTarget))
      ) {
        // Remove new line chars and spaces.
        let processedValue = value;
        if (key === TENSOR_VALUES_KEY) {
          // For __value attribute, remove all white space chars.
          processedValue = value.replace(/\s/gm, '');
        } else {
          // For other attributes, only remove newline chars.
          processedValue = value.replace(/(\r\n|\n|\r)/gm, ' ');
        }
        keyValuePairs.push({
          key,
          value: processedValue,
        });
      }
    }
  }
  return keyValuePairs;
}

/**
 * Gets the key value pairs for the given group node's attrs for attrs table.
 */
export function getGroupNodeAttrsKeyValuePairsForAttrsTable(
  node: GroupNode,
  modelGraph: ModelGraph,
  filterRegex = '',
) {
  const attrs =
    modelGraph.groupNodeAttributes?.[node.id.replace('___group___', '')] || {};
  const keyValuePairs: KeyValueList = [];
  const regex = new RegExp(filterRegex, 'i');
  for (const attrId of Object.keys(attrs)) {
    const key = attrId;
    const value = attrs[attrId];
    const matchTargets = [`${key}:${value}`, `${key}=${value}`];
    if (
      filterRegex.trim() === '' ||
      matchTargets.some((matchTarget) => regex.test(matchTarget))
    ) {
      // Remove new line chars and spaces.
      const processedValue = value.replace(/(\r\n|\n|\r)/gm, ' ');
      keyValuePairs.push({
        key,
        value: processedValue,
      });
    }
  }
  return keyValuePairs;
}

/** Gets the key value pairs for the givn node's input for attrs table. */
export function getOpNodeInputsKeyValuePairsForAttrsTable(
  node: OpNode,
  modelGraph: ModelGraph,
): KeyValueList {
  const incomingEdges = node.incomingEdges || [];
  const keyValuePairs: KeyValueList = [];
  for (
    let i = 0;
    i < Math.min(MAX_IO_ROWS_IN_ATTRS_TABLE, incomingEdges.length);
    i++
  ) {
    const incomingEdge = incomingEdges[i];
    const sourceNodeId = incomingEdge.sourceNodeId;
    const sourceNode = modelGraph.nodesById[sourceNodeId] as OpNode;
    const sourceNodeShape = getShapeForAttrsTable(
      (sourceNode.outputsMetadata || {})[incomingEdge.sourceNodeOutputId],
    );
    const inputMetadata =
      (node.inputsMetadata || {})[incomingEdge.targetNodeInputId] || {};
    keyValuePairs.push({
      key: getInputLabelForAttrsTable(i, sourceNode, inputMetadata),
      value: sourceNodeShape,
    });
  }

  if (incomingEdges.length > MAX_IO_ROWS_IN_ATTRS_TABLE) {
    const overMaxCount = incomingEdges.length - MAX_IO_ROWS_IN_ATTRS_TABLE;
    keyValuePairs.push({
      key: `(${overMaxCount} more input${
        overMaxCount === 1 ? '' : 's'
      } omitted)`,
      value: '...',
    });
  }

  return keyValuePairs;
}

/** Gets the key value pairs for the given node's outputs for attrs table. */
export function getOpNodeOutputsKeyValuePairsForAttrsTable(
  node: OpNode,
): KeyValueList {
  const keyValuePairs: KeyValueList = [];
  const outputsMetadata = node.outputsMetadata || {};
  const outputDataList = Object.values(outputsMetadata);
  for (
    let i = 0;
    i < Math.min(MAX_IO_ROWS_IN_ATTRS_TABLE, outputDataList.length);
    i++
  ) {
    const outputData = outputDataList[i];
    const shape = getShapeForAttrsTable(outputData);
    keyValuePairs.push({
      key: getOutputLabelForAttrsTable(i, outputData, node),
      value: shape,
    });
  }

  if (outputDataList.length > MAX_IO_ROWS_IN_ATTRS_TABLE) {
    const overMaxCount = outputDataList.length - MAX_IO_ROWS_IN_ATTRS_TABLE;
    keyValuePairs.push({
      key: `(${overMaxCount} more output${
        overMaxCount === 1 ? '' : 's'
      } omitted)`,
      value: '...',
    });
  }

  return keyValuePairs;
}

/** Gets the key value pairs for the given node's data provider runs. */
export function getOpNodeDataProviderKeyValuePairsForAttrsTable(
  node: OpNode,
  modelGraphId: string,
  showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
  curNodeDataProviderRuns: Record<string, NodeDataProviderRunData>,
  config?: VisualizerConfig,
): KeyValueList {
  const keyValuePairs: KeyValueList = [];
  const runNames = Object.keys(showOnNodeItemTypes)
    .filter((type) => showOnNodeItemTypes[type].selected)
    .filter((itemType: string) =>
      itemType.startsWith(NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX),
    )
    .map((type) =>
      type.replace(NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX, ''),
    );
  const runs = Object.values(curNodeDataProviderRuns).filter((run) =>
    runNames.includes(getRunName(run, {id: modelGraphId})),
  );
  for (const run of runs) {
    const result = ((run.results || {})?.[modelGraphId] || {})[node.id];
    if (config?.hideEmptyNodeDataEntries && !result) {
      continue;
    }
    const value = result?.strValue || '-';
    keyValuePairs.push({
      key: getRunName(run, {id: modelGraphId}),
      value,
    });
  }
  return keyValuePairs;
}

/**
 * Given two namespace strings, e.g. a/b/c/d and a/b/x, returns the common
 * prefix, e.g. a/b.
 */
export function findCommonNamespace(ns1: string, ns2: string): string {
  const ns1Parts = ns1.split('/');
  const ns2Parts = ns2.split('/');
  let commonPrefix = '';
  for (let i = Math.min(ns1Parts.length, ns2Parts.length); i > 0; i--) {
    const ns1Prefix = ns1Parts.slice(0, i).join('/');
    const ns2Prefix = ns2Parts.slice(0, i).join('/');
    if (ns1Prefix === ns2Prefix) {
      commonPrefix = ns2Prefix;
      break;
    }
  }
  return commonPrefix;
}

/** Gets the next level namespace part right after baseNs up to fullNs. */
export function getNextLevelNsPart(baseNs: string, fullNs: string): string {
  if (baseNs === fullNs) {
    return '';
  }
  const baseNsParts = baseNs.split('/').filter((part) => part !== '');
  const fullNsParts = fullNs.split('/').filter((part) => part !== '');
  if (fullNsParts.length === 0) {
    return '';
  }
  return fullNsParts[baseNsParts.length];
}

/** Loads the given trusted script. */
export async function loadTrustedScript(
  trustedScript: TrustedResourceUrl,
): Promise<void> {
  return new Promise<void>((resolve) => {
    const script = document.createElement('script');
    script.src = trustedScript.toString();
    script.onload = () => {
      script.remove();
      resolve();
    };
    document.body.appendChild(script);
  });
}

/** Processes the error message to make it more clear. */
export function processErrorMessage(msg: string): string {
  if (
    new RegExp(
      /Only `SavedModel`s with \d+ MetaGraph are supported. Instead, it has \d+/,
    ).test(msg)
  ) {
    return `${msg}. Try using the "TF adapter (direct)" adapter.`;
  }
  return msg;
}

/** Gets the search matches for the given node using regex. */
export function getRegexMatchesForNode(
  shouldMatchTypes: Set<string>,
  regex: RegExp,
  node: ModelNode,
  modelGraph: ModelGraph,
  config?: VisualizerConfig,
): {
  matches: SearchMatch[];
  matchTypes: Set<string>;
} {
  const matches: SearchMatch[] = [];
  const matchTypes = new Set<string>();

  // Node label.
  if (
    shouldMatchTypes.has(SearchMatchType.NODE_LABEL) &&
    regex.test(node.label.replace(/\n/gm, ''))
  ) {
    matches.push({
      type: SearchMatchType.NODE_LABEL,
    });
    matchTypes.add(SearchMatchType.NODE_LABEL);
  }
  // Attribute.
  if (shouldMatchTypes.has(SearchMatchType.ATTRIBUTE)) {
    const attrs = getAttributesFromNode(node, modelGraph, config);
    for (const attrId of Object.keys(attrs)) {
      const value = attrs[attrId];
      const text1 = `${attrId}:${value}`;
      const text2 = `${attrId}=${value}`;
      if (regex.test(text1) || regex.test(text2)) {
        matches.push({
          type: SearchMatchType.ATTRIBUTE,
          matchedAttrId: attrId,
        });
        matchTypes.add(SearchMatchType.ATTRIBUTE);
      }
    }
  }
  // Inputs
  if (shouldMatchTypes.has(SearchMatchType.INPUT_METADATA) && isOpNode(node)) {
    const inputMetadataKeysToHide = config?.inputMetadataKeysToHide ?? [];
    for (const incomingEdge of node.incomingEdges || []) {
      // Match source node's label.
      const sourceNode = modelGraph.nodesById[
        incomingEdge.sourceNodeId
      ] as OpNode;
      if (regex.test(sourceNode.label)) {
        matches.push({
          type: SearchMatchType.INPUT_METADATA,
          matchedText: sourceNode.label,
        });
        matchTypes.add(SearchMatchType.INPUT_METADATA);
      }

      // Match tensor tag in current node's input metadata.
      const inputsMetadata = node.inputsMetadata || {};
      const tensorTag = (inputsMetadata[incomingEdge.targetNodeInputId] || {})[
        TENSOR_TAG_METADATA_KEY
      ];
      if (tensorTag && regex.test(tensorTag)) {
        matches.push({
          type: SearchMatchType.INPUT_METADATA,
          matchedText: tensorTag,
        });
        matchTypes.add(SearchMatchType.INPUT_METADATA);
      }

      // Match source node's output metadata.
      const metadata =
        (sourceNode.outputsMetadata || {})[incomingEdge.sourceNodeOutputId] ||
        {};
      for (const metadataKey of Object.keys(metadata)) {
        if (metadataKey.startsWith('__')) {
          continue;
        }
        if (inputMetadataKeysToHide.some((regex) => metadataKey.match(regex))) {
          continue;
        }
        const value = metadata[metadataKey];
        const text1 = `${metadataKey}:${value}`;
        const text2 = `${metadataKey}=${value}`;
        if (regex.test(value) || regex.test(text1) || regex.test(text2)) {
          matches.push({
            type: SearchMatchType.INPUT_METADATA,
            matchedText: value,
          });
          matchTypes.add(SearchMatchType.INPUT_METADATA);
        }
      }

      // Match target node's input metadata.
      const curInputMetadata =
        inputsMetadata[incomingEdge.targetNodeInputId] || {};
      for (const metadataKey of Object.keys(curInputMetadata)) {
        if (metadataKey.startsWith('__')) {
          continue;
        }
        if (inputMetadataKeysToHide.some((regex) => metadataKey.match(regex))) {
          continue;
        }
        const value = curInputMetadata[metadataKey];
        const text1 = `${metadataKey}:${value}`;
        const text2 = `${metadataKey}=${value}`;
        if (regex.test(value) || regex.test(text1) || regex.test(text2)) {
          matches.push({
            type: SearchMatchType.INPUT_METADATA,
            matchedText: value,
          });
          matchTypes.add(SearchMatchType.INPUT_METADATA);
        }
      }
    }
  }
  // Outputs
  if (shouldMatchTypes.has(SearchMatchType.OUTPUT_METADATA) && isOpNode(node)) {
    const outputsMetadata = node.outputsMetadata || {};
    const outputMetadataKeysToHide = config?.outputMetadataKeysToHide ?? [];

    for (const outgoingEdge of node.outgoingEdges || []) {
      const targetNode = modelGraph.nodesById[
        outgoingEdge.targetNodeId
      ] as OpNode;
      if (regex.test(targetNode.label)) {
        matches.push({
          type: SearchMatchType.OUTPUT_METADATA,
          matchedText: targetNode.label,
        });
        matchTypes.add(SearchMatchType.OUTPUT_METADATA);
      }

      // Match tensor tag in current node's output metadata.
      const tensorTag = (outputsMetadata[outgoingEdge.sourceNodeOutputId] ||
        {})[TENSOR_TAG_METADATA_KEY];
      if (tensorTag && regex.test(tensorTag)) {
        matches.push({
          type: SearchMatchType.OUTPUT_METADATA,
          matchedText: tensorTag,
        });
        matchTypes.add(SearchMatchType.OUTPUT_METADATA);
      }
    }

    for (const metadata of Object.values(outputsMetadata)) {
      for (const metadataKey of Object.keys(metadata)) {
        if (metadataKey.startsWith('__')) {
          continue;
        }
        if (
          outputMetadataKeysToHide.some((regex) => metadataKey.match(regex))
        ) {
          continue;
        }
        const value = metadata[metadataKey];
        const text1 = `${metadataKey}:${value}`;
        const text2 = `${metadataKey}=${value}`;
        if (regex.test(value) || regex.test(text1) || regex.test(text2)) {
          matches.push({
            type: SearchMatchType.OUTPUT_METADATA,
            matchedText: value,
          });
          matchTypes.add(SearchMatchType.OUTPUT_METADATA);
        }
      }
    }
  }

  return {matches, matchTypes};
}

/** Gets the attributes from the given node. */
export function getAttributesFromNode(
  node: ModelNode,
  modelGraph: ModelGraph,
  config?: VisualizerConfig,
): KeyValuePairs {
  let attrs: KeyValuePairs = {};
  const nodeInfoKeysToHide = config?.nodeInfoKeysToHide ?? [];
  if (isOpNode(node)) {
    for (const [key, value] of Object.entries(node.attrs || {})) {
      if (typeof value === 'string') {
        attrs[key] = value;
      } else {
        switch (value.type) {
          case NodeAttributeValueType.NODE_IDS:
            attrs[key] = value.nodeIds.join(',');
            break;
          default:
            break;
        }
      }
    }
    // Add id to attribute.
    attrs['id'] = node.id;
  } else if (isGroupNode(node)) {
    attrs = {
      '#descendants': `${(node.descendantsNodeIds || []).length}`,
      '#children': `${(node.nsChildrenIds || []).length}`,
      'namespace': node.namespace || node.savedNamespace || '<root>',
    };
    const customAttrs =
      modelGraph.groupNodeAttributes?.[node.id.replace('___group___', '')] ||
      {};
    attrs = {...attrs, ...customAttrs};
  }

  // Filter out node info keys specified in the config.
  attrs = Object.fromEntries(
    Object.entries(attrs).filter(
      ([key, value]) => !nodeInfoKeysToHide.some((regex) => key.match(regex)),
    ),
  );

  return attrs;
}

/** Gets the search matches for the given node using attr value range. */
export function getAttrValueRangeMatchesForNode(
  attrName: string,
  min: number,
  max: number,
  node: ModelNode,
  modelGraph: ModelGraph,
  config?: VisualizerConfig,
): SearchMatch[] {
  const matches: SearchMatch[] = [];

  const attrs = getAttributesFromNode(node, modelGraph, config);
  const value = attrs[attrName];
  if (value != null) {
    const numValue = Number(value);
    if (!isNaN(numValue) && numValue >= min && numValue <= max) {
      matches.push({
        type: SearchMatchType.ATTRIBUTE,
        matchedAttrId: attrName,
      });
    }
  }

  return matches;
}

/** Checks if the given queries have non-empty queries. */
export function hasNonEmptyQueries(queries: NodeQuery[]): boolean {
  for (const query of queries.filter(
    (query) => query.type !== NodeQueryType.NODE_TYPE,
  )) {
    switch (query.type) {
      case NodeQueryType.REGEX:
        if (query.queryRegex !== '') {
          return true;
        }
        break;
      case NodeQueryType.ATTR_VALUE_RANGE:
        if (query.attrName !== '') {
          return true;
        }
        break;
      default:
        break;
    }
  }
  return false;
}

/** Processes the given node styler rules. */
export function processNodeStylerRules(
  rules: NodeStylerRule[],
): ProcessedNodeStylerRule[] {
  return rules.map((rule) => {
    const processedQueries = rule.queries.map((query) => {
      switch (query.type) {
        case NodeQueryType.REGEX: {
          let regex = new RegExp('', 'i');
          try {
            regex = new RegExp(query.queryRegex, 'i');
          } catch (e) {
            console.warn('Failed to create regex', e);
          }
          const processedQuery: ProcessedNodeRegexQuery = {
            type: NodeQueryType.REGEX,
            queryRegex: regex,
            matchTypes: new Set<SearchMatchType>(query.matchTypes),
          };
          return processedQuery;
        }
        case NodeQueryType.NODE_TYPE:
        case NodeQueryType.ATTR_VALUE_RANGE: {
          return query;
        }
        default:
          return undefined;
      }
    });
    return {
      queries: processedQueries as ProcessedNodeQuery[],
      nodeType: rule.nodeType,
      styles: rule.styles,
    };
  });
}

/** Checks if the given node matches the given queries. */
export function matchNodeForQueries(
  node: ModelNode,
  queries: ProcessedNodeQuery[],
  modelGraph: ModelGraph,
  config?: VisualizerConfig,
): boolean {
  let matchedAll = true;
  for (const query of queries) {
    if (query.type === NodeQueryType.NODE_TYPE) {
      let matched = true;
      if (
        (isOpNode(node) && query.nodeType === SearchNodeType.LAYER_NODES) ||
        (isGroupNode(node) && query.nodeType === SearchNodeType.OP_NODES)
      ) {
        matched = false;
      }
      if (!matched) {
        matchedAll = false;
        break;
      }
    } else if (query.type === NodeQueryType.REGEX) {
      const matches = getRegexMatchesForNode(
        query.matchTypes,
        query.queryRegex,
        node,
        modelGraph,
        config,
      ).matches;
      if (matches.length === 0) {
        matchedAll = false;
        break;
      }
    } else if (query.type === NodeQueryType.ATTR_VALUE_RANGE) {
      if (query.attrName !== '') {
        const matches = getAttrValueRangeMatchesForNode(
          query.attrName,
          query.min ?? Number.NEGATIVE_INFINITY,
          query.max ?? Number.POSITIVE_INFINITY,
          node,
          modelGraph,
          config,
        );
        if (matches.length === 0) {
          matchedAll = false;
          break;
        }
      }
    } else {
      matchedAll = false;
    }
  }
  return matchedAll;
}

/** Exports to IDE resource. */
// tslint:disable-next-line:no-any
export function exportToResource(name: string, resource: any) {
  window.parent.postMessage(
    {
      'cmd': EXPORT_TO_RESOURCE_CMD,
      'name': name,
      'resource': resource,
    },
    '*',
  );
}

/** Gets the high quality pixel ratio. */
export function getHighQualityPixelRatio(): number {
  return window.devicePixelRatio === 1
    ? 1.5 /* This makes rendering result sharper on non-retina displays */
    : window.devicePixelRatio;
}

/** Get the value for the given style. */
export function getNodeStyleValue(
  rule: ProcessedNodeStylerRule | NodeStylerRule,
  styleId: NodeStyleId,
): string {
  const curStyle = rule.styles[styleId];
  if (curStyle) {
    if (typeof curStyle === 'string') {
      return curStyle;
    } else {
      return curStyle.value;
    }
  }
  return '';
}

/** Splits the given label. */
export function splitLabel(label: string): string[] {
  return label
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '');
}

/** Get the extra height for multi-line label. */
export function getMultiLineLabelExtraHeight(label: string): number {
  return (splitLabel(label).length - 1) * NODE_LABEL_LINE_HEIGHT;
}

/**
 * Calculates the closest intersection points of a line (L) connecting
 * the centers of two rectangles (rect1 and rect2) with the sides of these
 * rectangles.
 *
 * xOffsetFactor is used to shift the center of the rectangle to the left or
 * right by a certain factor of the width of the rectangle.
 */
export function getIntersectionPoints(
  rect1: Rect,
  rect2: Rect,
  xOffsetFactor = 0,
) {
  // Function to calculate the center of a rectangle
  function getCenter(rect: Rect) {
    return {
      x: rect.x + rect.width / 2 + xOffsetFactor * rect.width,
      y: rect.y + rect.height / 2,
    };
  }

  // Function to calculate intersection between a line and a rectangle
  function getIntersection(rect: Rect, center1: Point, center2: Point) {
    // Line parameters
    const dx = center2.x - center1.x;
    const dy = center2.y - center1.y;

    // Check for intersection with each of the four sides of the rectangle
    let tMin = Number.MAX_VALUE;
    let intersection: Point = {x: 0, y: 0};

    // Left side (x = rect.x)
    if (dx !== 0) {
      const t = (rect.x - center1.x) / dx;
      const y = center1.y + t * dy;
      if (t >= 0 && y >= rect.y && y <= rect.y + rect.height && t < tMin) {
        tMin = t;
        intersection = {x: rect.x, y};
      }
    }

    // Right side (x = rect.x + rect.width)
    if (dx !== 0) {
      const t = (rect.x + rect.width - center1.x) / dx;
      const y = center1.y + t * dy;
      if (t >= 0 && y >= rect.y && y <= rect.y + rect.height && t < tMin) {
        tMin = t;
        intersection = {x: rect.x + rect.width, y};
      }
    }

    // Top side (y = rect.y)
    if (dy !== 0) {
      const t = (rect.y - center1.y) / dy;
      const x = center1.x + t * dx;
      if (t >= 0 && x >= rect.x && x <= rect.x + rect.width && t < tMin) {
        tMin = t;
        intersection = {x, y: rect.y};
      }
    }

    // Bottom side (y = rect.y + rect.height)
    if (dy !== 0) {
      const t = (rect.y + rect.height - center1.y) / dy;
      const x = center1.x + t * dx;
      if (t >= 0 && x >= rect.x && x <= rect.x + rect.width && t < tMin) {
        tMin = t;
        intersection = {x, y: rect.y + rect.height};
      }
    }

    return intersection;
  }

  // Get the centers of the rectangles
  const center1 = getCenter(rect1);
  const center2 = getCenter(rect2);

  // Find the closest intersection point of the line with rect1 and rect2
  const intersection1 = getIntersection(rect1, center1, center2);
  const intersection2 = getIntersection(rect2, center2, center1);

  return {intersection1, intersection2};
}

/** Gets the run name for the given run. */
export function getRunName(
  run: NodeDataProviderRunData,
  modelGraphIdLike?: {id: string},
): string {
  return (
    run.nodeDataProviderData?.[modelGraphIdLike?.id || '']?.name ?? run.runName
  );
}

/** Generates the sorted value infos for the given group node. */
export function genSortedValueInfos(
  groupNode: GroupNode | undefined,
  modelGraph: ModelGraph,
  results: Record<string, NodeDataProviderResultProcessedData>,
): NodeDataProviderValueInfo[] {
  const bgColorToValueInfo: Record<string, NodeDataProviderValueInfo> = {};
  const descendantsOpNodeIds =
    groupNode?.descendantsOpNodeIds || modelGraph.nodes.map((node) => node.id);
  for (const nodeId of descendantsOpNodeIds) {
    const node = modelGraph.nodesById[nodeId];
    const bgColor = results[node.id]?.bgColor || '';
    if (bgColor) {
      if (!bgColorToValueInfo[bgColor]) {
        bgColorToValueInfo[bgColor] = {
          label: `${results[nodeId]?.value || ''}`,
          bgColor,
          count: 1,
        };
      } else {
        bgColorToValueInfo[bgColor].count++;
      }
    }
  }
  return Object.values(bgColorToValueInfo).sort((a, b) =>
    a.bgColor.localeCompare(b.bgColor),
  );
}

/** Gets the input/output metadata keys for the given show on edge item. */
export function getShowOnEdgeInputOutputMetadataKeys(
  curShowOnEdgeItem?: ShowOnEdgeItemData,
): {
  inputMetadataKey?: string;
  outputMetadataKey?: string;
  sourceNodeAttrKey?: string;
  targetNodeAttrKey?: string;
} {
  let outputMetadataKey: string | undefined = undefined;
  let inputMetadataKey: string | undefined = undefined;
  let sourceNodeAttrKey: string | undefined = undefined;
  let targetNodeAttrKey: string | undefined = undefined;
  switch (curShowOnEdgeItem?.type) {
    case ShowOnEdgeItemType.TENSOR_SHAPE:
      outputMetadataKey = 'shape';
      break;
    case ShowOnEdgeItemType.OUTPUT_METADATA:
      outputMetadataKey = curShowOnEdgeItem.filterText ?? '';
      break;
    case ShowOnEdgeItemType.INPUT_METADATA:
      inputMetadataKey = curShowOnEdgeItem.filterText ?? '';
      break;
    case ShowOnEdgeItemType.SOURCE_NODE_ATTR:
      sourceNodeAttrKey = curShowOnEdgeItem.filterText ?? '';
      break;
    case ShowOnEdgeItemType.TARGET_NODE_ATTR:
      targetNodeAttrKey = curShowOnEdgeItem.filterText ?? '';
      break;
    default:
      break;
  }
  return {
    outputMetadataKey,
    inputMetadataKey,
    sourceNodeAttrKey,
    targetNodeAttrKey,
  };
}

/** Gets the string value for the given node attribute. */
export function getNodeAttrStringValue(node: OpNode, key: string): string {
  const attrValue = (node.attrs ?? {})[key];
  if (attrValue == null) {
    return '';
  } else {
    if (typeof attrValue === 'string') {
      return attrValue;
    } else {
      switch (attrValue.type) {
        case NodeAttributeValueType.NODE_IDS:
          return attrValue.nodeIds.join(', ');
        default:
          break;
      }
    }
  }
  return '';
}

export function getLayoutDirection(
  modelGraph: ModelGraph,
  groupNodeId: string,
): LayoutDirection {
  const namespaceName = groupNodeId.replace('___group___', '');
  for (const config of modelGraph.groupNodeConfigs || []) {
    try {
      const regex = new RegExp(config.namespaceRegex);
      if (regex.test(namespaceName)) {
        return config.layoutDirection ?? LayoutDirection.TOP_BOTTOM;
      }
    } catch (e) {
      console.warn(
        'Invalid regex in groupNodeConfigs',
        config.namespaceRegex,
        e,
      );
    }
  }
  return LayoutDirection.TOP_BOTTOM;
}
