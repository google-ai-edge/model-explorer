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

import {GroupNode, ModelGraph, OpNode} from '../common/model_graph';
import {isGroupNode} from '../common/utils';

const BIG_PRIME = 10000019;

/**
 * Finds identical subgraphs using hash.
 *
 * TODO: add tests.
 */
export class IdenticalGroupsFinder {
  constructor(private readonly modelGraph: ModelGraph) {}

  markIdenticalGroups() {
    // From group hash to a list of group nodes with that hash.
    const identicalGroups: {[hash: number]: GroupNode[]} = {};
    for (const node of this.modelGraph.nodes) {
      if (!isGroupNode(node)) {
        continue;
      }

      let hash = 0;

      // Add hashes for nodes.
      //
      // Only consider nodes that are not hidden in layout.
      const opNodes = (node.descendantsOpNodeIds || [])
        .map((id) => this.modelGraph.nodesById[id])
        .filter((node) => !(node as OpNode).hideInLayout) as OpNode[];
      const opNodeIdsSet = new Set<string>(opNodes.map((node) => node.id));
      for (const opNode of opNodes) {
        hash = (hash + this.getNodeHash(opNode, opNodeIdsSet)) % BIG_PRIME;
      }

      // Add hashes for edges.
      //
      // Only consider edges within the subgraph.
      for (const opNode of opNodes) {
        for (const edge of opNode.outgoingEdges || []) {
          const targetNodeId = edge.targetNodeId;
          if (!opNodeIdsSet.has(targetNodeId)) {
            continue;
          }
          const targetNode = this.modelGraph.nodesById[targetNodeId] as OpNode;
          hash = (hash + this.getEdgeHash(opNode, targetNode)) % BIG_PRIME;
        }
      }

      if (!identicalGroups[hash]) {
        identicalGroups[hash] = [];
      }
      identicalGroups[hash].push(node);
    }

    let identicalGroupIndex = 0;
    for (const groups of Object.values(identicalGroups)) {
      // Ignore groups with a single group node.
      if (groups.length <= 1) {
        continue;
      }

      // Ignore groups where group nodes only have one single op node.
      //
      // Re-enable this if needed.
      // if ((groups[0].descendantsOpNodeIds || []).length <= 1) {
      //   continue;
      // }

      // Ignore groups where there are only group nodes and one group node is
      // the NS parent of the other.
      if (groups.length === 2) {
        if (
          groups[0].nsParentId === groups[1].id ||
          groups[1].nsParentId === groups[0].id
        ) {
          continue;
        }
      }

      for (const groupNode of groups) {
        groupNode.identicalGroupIndex = identicalGroupIndex;
      }
      identicalGroupIndex++;
    }
  }

  private getNodeHash(opNode: OpNode, allowedOpNodeIds: Set<string>): number {
    let hash = 0;

    // Op.
    hash = this.addToHash(hash, opNode.label);

    // Incoming nodes.
    //
    // Limit the sources of its incoming edges among the nodes within the group.
    let incomingEdgeCount = 0;
    for (const edge of opNode.incomingEdges || []) {
      const sourceNodeId = edge.sourceNodeId;
      if (allowedOpNodeIds.has(sourceNodeId)) {
        const sourceNode = this.modelGraph.nodesById[sourceNodeId] as OpNode;
        hash = this.addToHash(hash, `in ${sourceNode.label}`);
        incomingEdgeCount++;
      }
    }

    // Outputing nodes.
    //
    // Limit the targets of its outgoing edges among the nodes within the group.
    let outgoingEdgeCount = 0;
    for (const edge of opNode.outgoingEdges || []) {
      const targetNodeId = edge.targetNodeInputId;
      if (allowedOpNodeIds.has(targetNodeId)) {
        const targetNode = this.modelGraph.nodesById[targetNodeId] as OpNode;
        hash = this.addToHash(hash, `out ${targetNode.label}`);
        outgoingEdgeCount++;
      }
    }

    // Incoming and outgoing edge count.
    hash = this.addToHash(hash, `${incomingEdgeCount}`);
    hash = this.addToHash(hash, `${outgoingEdgeCount}`);
    return hash;
  }

  private getEdgeHash(fromNode: OpNode, toNode: OpNode): number {
    return this.genHash(fromNode.label + toNode.label) % BIG_PRIME;
  }

  private genHash(str: string | undefined): number {
    let hash = 5381;
    str = str || '';

    for (let i = 0, len = str.length; i < len; i++) {
      hash += (hash << 5) + str.charCodeAt(i);
    }
    return hash & 0x7fffffff;
  }

  private addToHash(hash: number, str: string | undefined): number {
    return (hash + this.genHash(str)) % BIG_PRIME;
  }
}
