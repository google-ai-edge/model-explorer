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

import {Injectable, Signal, computed, signal} from '@angular/core';
import {AppService} from './app_service';
import {Graph, GraphNode} from './common/input_graph';
import {ModelGraph, OpNode} from './common/model_graph';
import {IncomingEdge} from './common/types';
import {isGroupNode, isOpNode} from './common/utils';

/** A service for managing subgraph selection. */
@Injectable()
export class SubgraphSelectionService {
  readonly selectedNodeIds = signal<Record<string, boolean>>({});

  readonly hasSelectedNodes = computed(
    () => Object.keys(this.selectedNodeIds()).length > 0,
  );

  readonly selectedNodeCount = computed(
    () => Object.keys(this.selectedNodeIds()).length,
  );

  readonly selectedNodes: Signal<OpNode[]> = computed(() => {
    if (!this.modelGraph) {
      return [];
    }
    const selectedNodeIds = Object.keys(this.selectedNodeIds()).filter(
      (nodeId) => this.selectedNodeIds()[nodeId],
    );
    return selectedNodeIds.map(
      (nodeId) => this.modelGraph!.nodesById[nodeId] as OpNode,
    );
  });

  paneId = '';

  constructor(private readonly appService: AppService) {}

  toggleNode(nodeId: string) {
    this.selectedNodeIds.update((ids) => {
      if (!this.modelGraph) {
        return ids;
      }

      const node = this.modelGraph.nodesById[nodeId];
      // Group node.
      if (isGroupNode(node)) {
        // Check if any of the descendant nodes are selected.
        const descendantOpNodeIds = node.descendantsOpNodeIds || [];
        const hasSelectedDescendantNodes = descendantOpNodeIds.some(
          (id) => ids[id],
        );
        // If so, deselect all descandant nodes. Otherwise,
        // select all descandant nodes.
        for (const id of descendantOpNodeIds) {
          if (isOpNode(this.modelGraph.nodesById[id])) {
            if (hasSelectedDescendantNodes) {
              delete ids[id];
            } else {
              ids[id] = true;
            }
          }
        }
      }
      // Op node.
      else {
        if (ids[nodeId]) {
          delete ids[nodeId];
        } else {
          ids[nodeId] = true;
        }
      }

      return {...ids};
    });
  }

  toggleNodes(nodeIds: string[]) {
    if (nodeIds.length === 0) {
      return;
    }
    if (!this.modelGraph) {
      return;
    }

    const ids = {...this.selectedNodeIds()};
    for (const id of nodeIds) {
      const node = this.modelGraph.nodesById[id];
      // Toggle op node.
      if (isOpNode(node)) {
        if (ids[id]) {
          delete ids[id];
        } else {
          ids[id] = true;
        }
      }
      // Also toggle all op nodes inside a collapsed group node.
      else if (isGroupNode(node) && !node.expanded) {
        for (const descendantOpNodeId of node.descendantsOpNodeIds || []) {
          if (ids[descendantOpNodeId]) {
            delete ids[descendantOpNodeId];
          } else {
            ids[descendantOpNodeId] = true;
          }
        }
      }
    }

    this.selectedNodeIds.set(ids);
  }

  clearSelection() {
    this.selectedNodeIds.set({});
  }

  isHiddenFromSelection(node: GraphNode): boolean {
    return node.label === 'pseudo_const';
  }

  getSelectedSubgraph(): Graph | undefined {
    if (!this.modelGraph) {
      return undefined;
    }

    let graph = this.appService.getGraphById(this.modelGraph.id);
    if (!graph) {
      return undefined;
    }
    graph = JSON.parse(JSON.stringify(graph)) as Graph;
    const graphNodes: Record<string, GraphNode> = {};
    for (const node of graph.nodes) {
      graphNodes[node.id] = node;
    }
    const selectedNodeIds = this.selectedNodeIds();

    // Only leave the selected nodes in the graph.
    const nodes: GraphNode[] = graph.nodes.filter(
      (node) => selectedNodeIds[node.id] === true,
    );

    // Find the input edges of the selected subgraph and add them to the
    // "GraphInputs" node. These edges are the connections from outside nodes to
    // the nodes within the selected subgraph.
    let graphInputsNode = nodes.find((node) => node.label === 'GraphInputs');
    if (!graphInputsNode) {
      const originalGraphInputsNode = graph.nodes.find(
        (node) => node.label === 'GraphInputs',
      );
      if (!originalGraphInputsNode) {
        throw new Error('GraphInputs node not found in the original graph.');
      }
      graphInputsNode = structuredClone(originalGraphInputsNode);
      graphInputsNode.outputsMetadata = [];
      nodes.push(graphInputsNode);
    }
    graphInputsNode.outputsMetadata = graphInputsNode.outputsMetadata || [];
    // Map from tensor index to the output metadata id of the GraphInputs node.
    const inputTensorIndexMap = new Map<string, string>();
    let intputsMetadataId = graphInputsNode.outputsMetadata.length;
    for (const node of nodes) {
      const incomingEdgesOutsideSubgraph = (node.incomingEdges || []).filter(
        (edge) => !selectedNodeIds[edge.sourceNodeId],
      );
      for (const edge of incomingEdgesOutsideSubgraph) {
        const sourceNode = graphNodes[edge.sourceNodeId];
        // If the source node is hidden from selection, add it to the subgraph
        // and skip adding the edge to the GraphInputs node.
        if (this.isHiddenFromSelection(sourceNode)) {
          nodes.push(sourceNode);
          continue;
        }
        const sourceNodeOutputMetadata = (
          sourceNode.outputsMetadata || []
        ).find((metadata) => metadata.id === edge.sourceNodeOutputId);
        if (!sourceNodeOutputMetadata) {
          continue;
        }
        const tensorIndex = (sourceNodeOutputMetadata.attrs || []).find(
          (attr) => attr.key === 'tensor_index',
        )?.value;
        if (!tensorIndex) {
          continue;
        }
        // Avoid adding duplicate edges to the GraphInputs node.
        if (!inputTensorIndexMap.has(tensorIndex)) {
          inputTensorIndexMap.set(tensorIndex, intputsMetadataId.toString());
          graphInputsNode.outputsMetadata.push({
            id: intputsMetadataId.toString(),
            attrs: sourceNodeOutputMetadata.attrs,
          });
          intputsMetadataId++;
        }
        // Update the edge to point to the GraphInputs node.
        edge.sourceNodeId = graphInputsNode.id;
        edge.sourceNodeOutputId = inputTensorIndexMap.get(tensorIndex)!;
      }
    }

    // Find the output edges of the selected subgraph and add them to the
    // "GraphOutputs" node. These edges are the connections from the nodes
    // within the selected subgraph to outside nodes.
    let graphOutputsNode = nodes.find((node) => node.label === 'GraphOutputs');
    if (!graphOutputsNode) {
      const originalGraphOutputsNode = graph.nodes.find(
        (node) => node.label === 'GraphOutputs',
      );
      if (!originalGraphOutputsNode) {
        throw new Error('GraphOutputs node not found in the original graph.');
      }
      graphOutputsNode = structuredClone(originalGraphOutputsNode);
      graphOutputsNode.incomingEdges = [];
      nodes.push(graphOutputsNode);
    }
    graphOutputsNode.incomingEdges = graphOutputsNode.incomingEdges || [];
    // Set of tensor indices that will be connected to GraphOutputs node.
    const outputTensorIndexSet = new Set<string>();
    let targetNodeInputId = graphOutputsNode.incomingEdges.length;
    for (const node of graph.nodes) {
      if (selectedNodeIds[node.id]) {
        continue;
      }
      const incomingEdgesFromSubgraph: IncomingEdge[] = (
        node.incomingEdges || []
      ).filter((edge) => selectedNodeIds[edge.sourceNodeId]);
      for (const edge of incomingEdgesFromSubgraph) {
        const sourceNode = graphNodes[edge.sourceNodeId];
        const sourceNodeOutputMetadata = (
          sourceNode.outputsMetadata || []
        ).find((metadata) => metadata.id === edge.sourceNodeOutputId);
        if (!sourceNodeOutputMetadata) {
          continue;
        }
        const tensorIndex = (sourceNodeOutputMetadata.attrs || []).find(
          (attr) => attr.key === 'tensor_index',
        )?.value;
        if (!tensorIndex) {
          continue;
        }
        // Avoid adding duplicate edges to the GraphOutputs node.
        if (!outputTensorIndexSet.has(tensorIndex)) {
          outputTensorIndexSet.add(tensorIndex);
          const edgeClone = structuredClone(edge);
          edgeClone.targetNodeInputId = targetNodeInputId.toString();
          graphOutputsNode.incomingEdges.push(edgeClone);
          targetNodeInputId++;
        }
      }
    }

    return {
      id: `${graph.id}_subgraph`,
      collectionLabel: graph.collectionLabel,
      nodes,
    };
  }

  get modelGraph(): ModelGraph | undefined {
    return this.appService.getCurrentModelGraphFromPane(this.paneId);
  }
}
