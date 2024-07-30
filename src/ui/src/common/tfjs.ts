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

import {Graph, GraphCollection, GraphNode,} from '../components/visualizer/common/input_graph';
import {IncomingEdge, KeyValueList,} from '../components/visualizer/common/types';

declare interface TfjsGraph {
  modelTopology: {node: TfjsNode[]};
  weightsManifest?: Array<{weights: WeightsManifestEntry[]}>;
}

declare interface TfjsNode {
  name: string;
  op: string;
  input?: string[];
  attr?: {[key: string]: TfjsAttrValue};
}

declare interface TfjsAttrValue {
  list?: TfjsListValue;
  s?: number[];
  i?: number;
  f?: number;
  b?: boolean;
  type?: TfjsDataType;
}

declare interface TfjsListValue {
  s?: number[][];
  i?: number[];
  f?: number[];
  b?: boolean[];
  type?: TfjsDataType[];
}

/** DataType enum. */
enum TfjsDataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,  // Arbitrary C++ data types
  DT_UINT32 = 22,
  DT_UINT64 = 23,
}

const TFJS_DATA_TYPE_MAP = new Map<TfjsDataType, string>([
  [TfjsDataType.DT_FLOAT, 'float'],
  [TfjsDataType.DT_DOUBLE, 'double'],
  [TfjsDataType.DT_INT32, 'int32'],
  [TfjsDataType.DT_UINT8, 'uint8'],
  [TfjsDataType.DT_INT16, 'int16'],
  [TfjsDataType.DT_INT8, 'int8'],
  [TfjsDataType.DT_STRING, 'string'],
  [TfjsDataType.DT_COMPLEX64, 'complex64'],
  [TfjsDataType.DT_INT64, 'int64'],
  [TfjsDataType.DT_BOOL, 'bool'],
  [TfjsDataType.DT_QINT8, 'qint8'],
  [TfjsDataType.DT_QUINT8, 'qint8'],
  [TfjsDataType.DT_QINT32, 'qint32'],
  [TfjsDataType.DT_BFLOAT16, 'bfloat16'],
  [TfjsDataType.DT_QINT16, 'qint16'],
  [TfjsDataType.DT_QUINT16, 'qint16'],
  [TfjsDataType.DT_UINT16, 'uint16'],
  [TfjsDataType.DT_COMPLEX128, 'complex128'],
  [TfjsDataType.DT_HALF, 'half'],
  [TfjsDataType.DT_RESOURCE, 'resource'],
  [TfjsDataType.DT_VARIANT, 'variant'],
  [TfjsDataType.DT_UINT32, 'uint32'],
  [TfjsDataType.DT_UINT64, 'uint64'],
]);

declare interface WeightsManifestEntry {
  name: string;
  shape: number[];
  dtype: 'float32'|'int32'|'bool'|'string'|'complex64';
}

/** Loads Tfjs model. */
export function loadTfjsModel(
    fileName: string,
    json: TfjsGraph,
    ): GraphCollection {
  // Create a map from node name to weight manifest entry.
  const nameToWeightManifestEntry: {[name: string]: WeightsManifestEntry} = {};
  if (json.weightsManifest != null) {
    for (const manifest of json.weightsManifest) {
      for (const entry of manifest.weights) {
        nameToWeightManifestEntry[entry.name] = entry;
      }
    }
  }

  // Popuplate nodes.
  const nodeIndex: Record<string, GraphNode> = {};
  const nodes: GraphNode[] = json.modelTopology.node.map((tfjsNode) => {
    const graphNode: GraphNode = {
      id: tfjsNode.name,
      label: tfjsNode.op,
      namespace: tfjsNode.name,
      incomingEdges: genIncomingEdges(tfjsNode),
      attrs: genAttrs(tfjsNode),
    };
    nodeIndex[graphNode.id] = graphNode;
    return graphNode;
  });

  // Populate outputs and inputs metadata.
  for (const tfjsNode of json.modelTopology.node) {
    const curNode = nodeIndex[tfjsNode.name];
    if (!curNode) {
      continue;
    }

    // For each of the input nodes, populate its outputs metadata.
    //
    // Use the "x" in ":x" in the name as the outputId.
    const inputs = tfjsNode.input || [];
    for (let i = 0; i < inputs.length; i++) {
      const curInputNodeName = inputs[i];
      const parts = curInputNodeName.split(':');
      let inputNodeName = curInputNodeName;
      let outputId = '0';
      if (parts.length === 2) {
        inputNodeName = parts[0];
        outputId = parts[1];
      }
      const inputNode = nodeIndex[inputNodeName];
      if (inputNode) {
        if (inputNode.outputsMetadata == null) {
          inputNode.outputsMetadata = [];
        }
        inputNode.outputsMetadata.push({
          id: outputId,
          attrs: [],
        });
      }
    }
  }

  // Fill in the outputs metadata from weights manifest.
  for (const node of nodes) {
    const entry = nameToWeightManifestEntry[node.id];
    if (entry) {
      if (node.outputsMetadata == null) {
        node.outputsMetadata = [];
      }
      node.outputsMetadata.push({
        id: '0',
        attrs: [
          {key: 'tensor_name', value: entry.name},
          {key: 'shape', value: entry.shape.join('x')},
          {key: 'dtype', value: entry.dtype},
        ],
      });
    }
  }

  const graph: Graph = {
    id: 'default',
    nodes,
  };
  return {label: fileName, graphs: [graph]};
}

function genIncomingEdges(tfjsNode: TfjsNode): IncomingEdge[] {
  return (tfjsNode.input || []).map((inputTfjsNodeId, index) => {
    const parts = inputTfjsNodeId.split(':');
    const incomingEdge: IncomingEdge = {
      sourceNodeId: parts.length === 2 ? parts[0] : inputTfjsNodeId,
      sourceNodeOutputId: parts.length === 2 ? parts[1] : '0',
      targetNodeInputId: `${index}`,
    };
    return incomingEdge;
  });
}

function genAttrs(tfjsNode: TfjsNode): KeyValueList {
  const tfjsAttrs = tfjsNode.attr || {};
  const keyValueList: KeyValueList = [];
  for (const key of Object.keys(tfjsAttrs)) {
    const tfjsAttrValue = tfjsAttrs[key];
    let value = '';
    if (tfjsAttrValue.list != null) {
      if ((tfjsAttrValue.list.s || []).length > 0) {
        value = (tfjsAttrValue.list.s || [])
                    .map((v) => decodeAttrValue({s: v}))
                    .join(', ');
      } else if ((tfjsAttrValue.list.b || []).length > 0) {
        value = (tfjsAttrValue.list.b || [])
                    .map((v) => decodeAttrValue({b: v}))
                    .join(', ');
      } else if ((tfjsAttrValue.list.f || []).length > 0) {
        value = (tfjsAttrValue.list.f || [])
                    .map((v) => decodeAttrValue({f: v}))
                    .join(', ');
      } else if ((tfjsAttrValue.list.i || []).length > 0) {
        value = (tfjsAttrValue.list.i || [])
                    .map((v) => decodeAttrValue({i: v}))
                    .join(', ');
      } else {
        value = '[]';
      }
    } else {
      value = decodeAttrValue(tfjsAttrValue);
    }
    keyValueList.push({key, value});
  }
  return keyValueList;
}

function decodeAttrValue(tfjsAttrValue: TfjsAttrValue): string {
  let value = '';
  if (tfjsAttrValue.s != null) {
    if (Array.isArray(tfjsAttrValue.s)) {
      value = `${
          tfjsAttrValue.s
              .map((v) => {
                if (typeof v === 'number') {
                  return String.fromCharCode(v);
                }
                return `${v}`;
              })
              .join('')}`;
    } else {
      value = atob(tfjsAttrValue.s);
    }
  } else if (tfjsAttrValue.i != null) {
    value = `${tfjsAttrValue.i}`;
  } else if (tfjsAttrValue.f != null) {
    value = `${tfjsAttrValue.f}`;
  } else if (tfjsAttrValue.b != null) {
    value = tfjsAttrValue ? 'true' : 'false';
  } else if (tfjsAttrValue.type != null) {
    value = TFJS_DATA_TYPE_MAP.get(tfjsAttrValue.type) || 'unknown';
  }
  return value;
}
