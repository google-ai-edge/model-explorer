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

import {ModelGraph, OpNode} from '../common/model_graph';
import {OutgoingEdge} from '../common/types';

/**
 * A class that finds paths between points in a model graph.
 */
export class GraphPathfinder {
  constructor(private readonly modelGraph: ModelGraph) {}

  /**
   * Returns an array of valid paths given a starting node and an end node using
   * a variation of Depth-first Search.
   */
  findPathsBetweenTwoNodes(
    startNodeId: string,
    endNodeId: string,
  ): OutgoingEdge[][] {
    if (
      !(
        startNodeId in this.modelGraph.nodesById &&
        endNodeId in this.modelGraph.nodesById
      )
    ) {
      return [];
    }

    const n1 = this.modelGraph.nodesById[startNodeId] as OpNode;
    const n2 = this.modelGraph.nodesById[endNodeId] as OpNode;
    const allPaths: OutgoingEdge[][] = [];

    const dfs = (
      startNode: OpNode,
      endNode: OpNode,
      currentPath: OutgoingEdge[],
    ) => {
      if (startNode === endNode) {
        // End node reached, record the complete path from start to end
        allPaths.push(currentPath);
      } else {
        if (startNode.outgoingEdges) {
          // Proceed to the next child node
          for (const edge of startNode.outgoingEdges) {
            const childNode = this.modelGraph.nodesById[
              edge.targetNodeId
            ] as OpNode;
            // Continue DFS with the updated path chain
            dfs(childNode, endNode, [...currentPath, edge]);
          }
        }
      }
    };

    dfs(n1, n2, []);
    return allPaths;
  }

  findPathsBetweenTwoNodeSets(
    startNodeIds: string[],
    endNodeIds: string[],
  ): OutgoingEdge[][] {
    const allPaths: OutgoingEdge[][] = [];
    for (const startNodeId of startNodeIds) {
      for (const endNodeId of endNodeIds) {
        allPaths.push(...this.findPathsBetweenTwoNodes(startNodeId, endNodeId));
      }
    }
    return allPaths;
  }
}
