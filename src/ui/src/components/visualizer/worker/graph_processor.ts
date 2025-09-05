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

import {
  DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
  TENSOR_VALUES_KEY,
} from '../common/consts';
import {Graph} from '../common/input_graph';
import {
  GroupNode,
  ModelGraph,
  ModelNode,
  NodeType,
  OpNode,
} from '../common/model_graph';
import {
  KeyValuePairs,
  MetadataItem,
  NodeAttributeList,
  NodeAttributePairs,
  NodeAttributeValue,
  NodeDataProviderRunData,
  ShowOnNodeItemData,
} from '../common/types';
import {
  findCommonNamespace,
  getNextLevelNsPart,
  isGroupNode,
  isOpNode,
} from '../common/utils';
import {VisualizerConfig} from '../common/visualizer_config';
import {ProcessingLabel} from '../common/worker_events';

import {getLayoutGraph} from './graph_layout';
import {updateProcessingProgress} from './utils';

const CONST_VALUE_REGEX = /dense<([^>]*)>/;

/**
 * A class that processes a given `Graph` into a `ModelGraph`.
 */
export class GraphProcessor {
  private readonly nodeLabelsToHide: Set<string>;

  constructor(
    private readonly paneId: string,
    private readonly graph: Graph,
    private readonly config?: VisualizerConfig,
    private readonly showOnNodeItemTypes: Record<
      string,
      ShowOnNodeItemData
    > = {},
    private readonly nodeDataProviderRuns: Record<
      string,
      NodeDataProviderRunData
    > = {},
    private readonly groupNodeChildrenCountThreshold = DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
    private readonly testMode = false,
    private readonly flattenLayers = false,
    private readonly keepLayersWithASingleChild = false,
  ) {
    this.nodeLabelsToHide = new Set<string>(
      (this.config?.nodeLabelsToHide || []).map((label) => label.toLowerCase()),
    );
  }

  process(): ModelGraph {
    const modelGraph = this.createEmptyModelGraph();

    this.processNodes(modelGraph);
    this.processEdgeRelationships(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_NODES_AND_EDGES,
    );

    this.processNamespaceRelationships(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_LAYER_NAMESPACES,
    );

    this.generateLayoutGraphConnections(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_LAYOUT_DATA,
    );

    this.splitLargeGroupNodes(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.SPLITTING_LARGE_LAYERS,
    );

    this.populateDescendantsAndCounts(modelGraph);

    return modelGraph;
  }

  /**
   * Scans nodes in `Graph` and creates the corresponding `OpNode` and
   * `GroupNode` in the `ModelGraph` (see model_graph.ts for more details).
   */
  processNodes(modelGraph: ModelGraph) {
    const seenNamespaces = new Set<string>();
    for (const graphNode of this.graph.nodes) {
      // Add an `OpNode` to the model graph for each node in the input graph.
      //
      // If namespace is a ";" separated string, use the last component as the
      // actual namespace.
      const namespace = graphNode.namespace;
      const parts = namespace.split(';').filter((part) => part !== '');
      if (parts.length > 1) {
        graphNode.namespace = parts[parts.length - 1];
      }
      const opNode: OpNode = {
        nodeType: NodeType.OP_NODE,
        id: graphNode.id,
        namespace: this.flattenLayers ? '' : graphNode.namespace,
        savedNamespace: graphNode.namespace,
        fullNamespace: graphNode.namespace,
        label: graphNode.label,
        level: this.getNonEmptyNamespaceComponents(graphNode.namespace).length,
      };
      if (graphNode.subgraphIds && graphNode.subgraphIds.length > 0) {
        opNode.subgraphIds = graphNode.subgraphIds;
      }
      if (this.nodeLabelsToHide.has(graphNode.label.toLowerCase())) {
        opNode.hideInLayout = true;
      }
      if (this.config?.nodeAttrsToHide) {
        const nodeAttrsWithBasicInfo: NodeAttributeList = [];
        if (graphNode.attrs != null) {
          nodeAttrsWithBasicInfo.push(...graphNode.attrs);
        }
        nodeAttrsWithBasicInfo.push({
          key: 'id',
          value: graphNode.id,
        });
        nodeAttrsWithBasicInfo.push({
          key: 'name',
          value: graphNode.label,
        });
        nodeAttrsWithBasicInfo.push({
          key: 'namespace',
          value: graphNode.namespace,
        });
        for (const [attrKey, attrValueRegex] of Object.entries(
          this.config.nodeAttrsToHide,
        )) {
          const attrValue = nodeAttrsWithBasicInfo.find(
            (attr) => attr.key === attrKey,
          )?.value;
          if (
            attrValue &&
            typeof attrValue === 'string' &&
            attrValue.match(attrValueRegex)
          ) {
            opNode.hideInLayout = true;
            break;
          }
        }
      }
      if (graphNode.attrs) {
        const attrs: NodeAttributePairs = {};
        for (const attr of graphNode.attrs) {
          attrs[attr.key] = this.processAttrValue(attr.key, attr.value);
        }
        opNode.attrs = attrs;
      }
      if (graphNode.inputsMetadata) {
        opNode.inputsMetadata = this.processMetadataList(
          graphNode.inputsMetadata,
        );
      }
      if (graphNode.outputsMetadata) {
        opNode.outputsMetadata = this.processMetadataList(
          graphNode.outputsMetadata,
        );
      }
      if (graphNode.style) {
        opNode.style = graphNode.style;
      }
      if (graphNode.config) {
        opNode.config = graphNode.config;
      }
      modelGraph.nodes.push(opNode);
      modelGraph.nodesById[opNode.id] = opNode;

      // Add group nodes for all ancestor namespaces from this op node.
      //
      // For example, if an op node's namespace is a/b/c, then add the following
      // group nodes.
      //
      // - namespace: a/b, label: c.
      // - namespace: a, label: b.
      // - namespace: <empty>, label a.
      if (!opNode.hideInLayout && !this.flattenLayers) {
        const ancestorNamespaces = this.getAncestorNamespaces(opNode.namespace);
        for (const ns of ancestorNamespaces) {
          if (seenNamespaces.has(ns)) {
            continue;
          }
          seenNamespaces.add(ns);

          const components = ns.split('/');
          // Use the last component of the namespace as its display label.
          const label = components.splice(-1)[0];
          // Group node's namespace doesn't contain the last component.
          const namespace = components.join('/');
          const groupNode: GroupNode = {
            nodeType: NodeType.GROUP_NODE,
            id: this.getGroupNodeIdFromNamespace(ns),
            namespace,
            label,
            level: components.length,
            expanded: false,
          };
          modelGraph.nodes.push(groupNode);
          modelGraph.nodesById[groupNode.id] = groupNode;
        }
      }
    }
  }

  /**
   * Sets edges in the given model graph based on the edges in the input graph.
   */
  processEdgeRelationships(modelGraph: ModelGraph) {
    for (const graphNode of this.graph.nodes) {
      const node = modelGraph.nodesById[graphNode.id] as OpNode;
      if (!node) {
        continue;
      }

      // From the graph node's incoming edges, populate the incoming and
      // outgoing edges for the corresponding node in the model graph.
      for (const incomingEdge of graphNode.incomingEdges || []) {
        const sourceNodeId = incomingEdge.sourceNodeId;
        const sourceNode = modelGraph.nodesById[sourceNodeId] as OpNode;
        if (!sourceNode) {
          continue;
        }

        // Incoming edges.
        if (node.incomingEdges == null) {
          node.incomingEdges = [];
        }
        if (
          node.incomingEdges.find(
            (edge) =>
              edge.sourceNodeId === sourceNodeId &&
              edge.sourceNodeOutputId === incomingEdge.sourceNodeOutputId &&
              edge.targetNodeInputId === incomingEdge.targetNodeInputId,
          ) == null
        ) {
          node.incomingEdges.push({...incomingEdge});
        }

        // Outgoing edges.
        if (sourceNode.outgoingEdges == null) {
          sourceNode.outgoingEdges = [];
        }
        if (
          sourceNode.outgoingEdges.find(
            (edge) =>
              edge.targetNodeId === node.id &&
              edge.sourceNodeOutputId === incomingEdge.sourceNodeOutputId &&
              edge.targetNodeInputId === incomingEdge.targetNodeInputId,
          ) == null
        ) {
          sourceNode.outgoingEdges.push({
            targetNodeId: node.id,
            sourceNodeOutputId: incomingEdge.sourceNodeOutputId,
            targetNodeInputId: incomingEdge.targetNodeInputId,
          });
        }
      }
    }
  }

  /**
   * Sets namespace relationships in model graph based on the hierarchy data
   * stored in input node's `namespace`.
   */
  processNamespaceRelationships(modelGraph: ModelGraph) {
    for (const node of modelGraph.nodes) {
      if (isOpNode(node) && node.hideInLayout) {
        continue;
      }

      const ns = node.namespace;

      // Root node.
      if (ns === '') {
        modelGraph.rootNodes.push(node);
        continue;
      }

      // Set namespace parent.
      const parentNodeId = this.getGroupNodeIdFromNamespace(ns);
      const parentGroupNode = modelGraph.nodesById[parentNodeId] as GroupNode;
      if (parentGroupNode) {
        node.nsParentId = parentGroupNode.id;
      } else {
        console.warn(
          `Failed to find the NS parent of node "${node.id}": "${parentNodeId}"`,
        );
      }

      // Set namespace children.
      if (parentGroupNode) {
        if (parentGroupNode.nsChildrenIds == null) {
          parentGroupNode.nsChildrenIds = [];
        }
        if (!parentGroupNode.nsChildrenIds.includes(node.id)) {
          parentGroupNode.nsChildrenIds.push(node.id);
          if (isOpNode(node) && node.config?.pinToGroupTop) {
            parentGroupNode.pinToTopOpNode = node;
          }
        }
      }
    }

    // Find group nodes that only have one single op node as its child. For
    // these nodes, remove the group node and move the child op node up a level
    // from its namespace.
    //
    // Repeatedly do this until no such nodes are found.
    if (!this.keepLayersWithASingleChild) {
      while (true) {
        let numNodeProcessed = 0;
        for (const node of modelGraph.nodes) {
          if (!isGroupNode(node)) {
            continue;
          }
          if (node.nsChildrenIds != null && node.nsChildrenIds.length === 1) {
            const opNode = modelGraph.nodesById[node.nsChildrenIds[0]];
            if (isOpNode(opNode)) {
              numNodeProcessed++;
              // Delete group node.
              const index = modelGraph.nodes.indexOf(node);
              if (index >= 0) {
                modelGraph.nodes.splice(index, 1);
              }
              delete modelGraph.nodesById[node.id];

              // Move op node up one level in namespace.
              const ns = opNode.namespace;
              const parts = this.getNonEmptyNamespaceComponents(ns);
              parts.pop();
              opNode.namespace = parts.join('/');
              opNode.savedNamespace = opNode.namespace;
              opNode.level = parts.length;
              opNode.nsParentId = node.nsParentId;

              // Update root node if necessary.
              const indexInRootNodes = modelGraph.rootNodes.indexOf(node);
              if (indexInRootNodes >= 0) {
                modelGraph.rootNodes.splice(indexInRootNodes, 1);
                modelGraph.rootNodes.push(opNode);
              }

              // Remove this node from its NS parent node's nsChildrenIds, and add
              // the op node to it.
              if (node.nsParentId) {
                const nsParent = modelGraph.nodesById[
                  node.nsParentId
                ] as GroupNode;
                const index = nsParent.nsChildrenIds!.indexOf(node.id);
                nsParent.nsChildrenIds!.splice(index, 1);
                nsParent.nsChildrenIds!.push(opNode.id);
              }
            }
          }
        }
        if (numNodeProcessed === 0) {
          break;
        }
      }
    }
  }

  /**
   * Generates layout graph connections for the given model graph.
   */
  generateLayoutGraphConnections(modelGraph: ModelGraph) {
    modelGraph.layoutGraphEdges = {};

    // Find all op nodes that don't have incoming edges.
    let seedOpNodes: OpNode[] = [];
    const allNonHiddenOpNodes: OpNode[] = [];
    for (const node of modelGraph.nodes) {
      if (!isOpNode(node) || node.hideInLayout) {
        continue;
      }
      allNonHiddenOpNodes.push(node);
      const filteredIncomingEdges = (node.incomingEdges || []).filter(
        (edge) =>
          !(modelGraph.nodesById[edge.sourceNodeId] as OpNode).hideInLayout,
      );
      if (filteredIncomingEdges.length === 0) {
        seedOpNodes.push(node);
      }
    }

    // If seedOpNodes is empty, it means all the nodes in the graph have
    // incoming edges. This indicates that the graph must contain at least one
    // full cycle without any "root" nodes. For example, the graph might have
    // one circle, or two disjoint circles, etc.
    //
    // Instead of picking one node from each of these disjointed cycles (which
    // might be expensive to calculate), we will just use all the nodes in the
    // graph as the seed nodes. The DFS procedure below will handle the dedup
    // logic correctly.
    if (seedOpNodes.length === 0 && allNonHiddenOpNodes.length > 0) {
      seedOpNodes = allNonHiddenOpNodes;
    }

    // Do a BFS from seedOpNodes.
    const queue: OpNode[] = [...seedOpNodes];
    const seenNodeIds = new Set<string>();
    while (queue.length > 0) {
      const curNode = queue.shift();
      if (curNode == null || curNode.hideInLayout) {
        continue;
      }
      if (seenNodeIds.has(curNode.id)) {
        continue;
      }
      seenNodeIds.add(curNode.id);

      // For each edge going from curNode (A), find the common namespace of
      // curNode and edge's target node (B), and mark the connection between the
      // top-level node that contains A and B within the common namespace.
      //
      // For example, op node X's namespae is a/b/c, op node Y's namespace
      // is a/b/d, and X has an edge to Y. X and Y's common namespace is a/b.
      // So we mark a/b/c and a/b/d to be connected.
      const outgoingEdges = curNode.outgoingEdges || [];
      for (const edge of outgoingEdges) {
        const targetNode = modelGraph.nodesById[edge.targetNodeId] as OpNode;
        if (targetNode.hideInLayout) {
          continue;
        }
        const commonNs = findCommonNamespace(
          curNode.namespace,
          targetNode.namespace,
        );
        const sourceNodeNextLevelNsPart = getNextLevelNsPart(
          commonNs,
          curNode.namespace,
        );
        const connectionFromNodeId =
          sourceNodeNextLevelNsPart === ''
            ? curNode.id
            : `${commonNs}${
                commonNs === '' ? '' : '/'
              }${sourceNodeNextLevelNsPart}___group___`;
        const targetNodeNextLevelNsPart = getNextLevelNsPart(
          commonNs,
          targetNode.namespace,
        );
        const connectionToNodeId =
          targetNodeNextLevelNsPart === ''
            ? targetNode.id
            : `${commonNs}${
                commonNs === '' ? '' : '/'
              }${targetNodeNextLevelNsPart}___group___`;

        const commonNsGroupId = commonNs === '' ? '' : `${commonNs}___group___`;
        if (modelGraph.layoutGraphEdges[commonNsGroupId] == null) {
          modelGraph.layoutGraphEdges[commonNsGroupId] = {};
        }
        if (
          modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId] ==
          null
        ) {
          modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId] =
            {};
        }
        modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId][
          connectionToNodeId
        ] = true;
      }

      for (const edge of outgoingEdges) {
        const targetNode = modelGraph.nodesById[edge.targetNodeId] as OpNode;
        queue.push(targetNode);
      }
    }
  }

  /**
   * Finds group nodes with a large number of children, and splits them into
   * different groups
   */
  splitLargeGroupNodes(modelGraph: ModelGraph) {
    // From root, do a BFS search on all group nodes.
    const queue: Array<GroupNode | undefined> = [undefined];
    let hasLargeGroupNodes = false;
    while (queue.length > 0) {
      const curGroupNode = queue.shift();
      let children: ModelNode[] =
        curGroupNode == null
          ? modelGraph.rootNodes
          : (curGroupNode.nsChildrenIds || []).map(
              (id) => modelGraph.nodesById[id],
            );

      // Split the group node if its child count is over the threshold.
      if (children.length > this.groupNodeChildrenCountThreshold) {
        hasLargeGroupNodes = true;
        const layoutGraph = getLayoutGraph(
          curGroupNode?.id || '',
          children,
          modelGraph,
          this.showOnNodeItemTypes,
          this.nodeDataProviderRuns,
          undefined,
          this.testMode,
          // Use fake node size.
          true,
          this.config,
        );

        // Find root nodes of the layout graph.
        const rootNodes: ModelNode[] = [];
        for (const nodeId of Object.keys(layoutGraph.nodes)) {
          if (layoutGraph.incomingEdges[nodeId] == null) {
            rootNodes.push(modelGraph.nodesById[nodeId]);
          }
        }

        // Do a DFS from the layout graph root nodes. Create a new group
        // whenever the node counts reaches the threshold.
        const groups: ModelNode[][] = [];
        let curGroup: ModelNode[] = [];
        const visitedNodeIds = new Set<string>();
        const visit = (curNodeId: string) => {
          if (visitedNodeIds.has(curNodeId)) {
            return;
          }
          visitedNodeIds.add(curNodeId);
          const node = modelGraph.nodesById[curNodeId];
          curGroup.push(node);
          if (curGroup.length === this.groupNodeChildrenCountThreshold) {
            groups.push(curGroup);
            curGroup = [];
          }
          for (const childId of layoutGraph.outgoingEdges[node.id] || []) {
            visit(childId);
          }
        };
        for (const rootNode of rootNodes) {
          visit(rootNode.id);
        }
        if (
          curGroup.length < this.groupNodeChildrenCountThreshold &&
          curGroup.length > 0
        ) {
          groups.push(curGroup);
        }

        // Create a new group node for each group.
        const newGroupNodes: GroupNode[] = [];
        for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
          const nodes = groups[groupIndex];
          const newGroupNodeNamespace =
            curGroupNode == null
              ? ''
              : `${curGroupNode.namespace}/${curGroupNode.label}`;
          const newGroupNodeLabel = `section_${groupIndex + 1}_of_${
            groups.length
          }`;
          const newGroupNodeId =
            curGroupNode == null
              ? `${newGroupNodeLabel}___group___`
              : `${newGroupNodeNamespace}/${newGroupNodeLabel}___group___`;
          const newGroupNode: GroupNode = {
            nodeType: NodeType.GROUP_NODE,
            id: newGroupNodeId,
            label: newGroupNodeLabel,
            namespace: newGroupNodeNamespace,
            level: newGroupNodeNamespace.split('/').filter((c) => c !== '')
              .length,
            nsParentId: curGroupNode?.id,
            nsChildrenIds: nodes.map((node) => node.id),
            expanded: false,
            sectionContainer: true,
          };
          newGroupNodes.push(newGroupNode);

          // Add the new group node to the model graph.
          modelGraph.nodes.push(newGroupNode);
          modelGraph.nodesById[newGroupNode.id] = newGroupNode;
          if (modelGraph.artificialGroupNodeIds == null) {
            modelGraph.artificialGroupNodeIds = [];
          }
          modelGraph.artificialGroupNodeIds.push(newGroupNode.id);

          // Update the ns parent for all nodes in the new group.
          for (const node of nodes) {
            node.nsParentId = newGroupNode.id;
          }

          // Update the namespace of all nodes and their desendents in the new
          // group.
          const newNamespacePart = newGroupNodeId.replace('___group___', '');
          const updateNamespace = (node: ModelNode) => {
            const oldNamespace = node.namespace;
            if (oldNamespace === '') {
              node.namespace = newNamespacePart;
            } else {
              if (curGroupNode == null) {
                node.namespace = `${newNamespacePart}/${node.namespace}`;
              } else {
                node.namespace = (node.nsParentId || '').replace(
                  '___group___',
                  '',
                );
              }
            }
            node.level = node.namespace
              .split('/')
              .filter((c) => c !== '').length;
            if (isGroupNode(node)) {
              // Update group node id since its namespace has been changed.
              const oldNodeId = node.id;
              delete modelGraph.nodesById[node.id];
              node.id = `${node.namespace}/${node.label}___group___`;
              modelGraph.nodesById[node.id] = node;

              // Update its parent's nsChildren to use the new id.
              if (node.nsParentId) {
                const nsParent = modelGraph.nodesById[
                  node.nsParentId
                ] as GroupNode;
                const index = (nsParent.nsChildrenIds || []).indexOf(oldNodeId);
                if (index >= 0) {
                  (nsParent.nsChildrenIds || [])[index] = node.id;
                }
              }

              for (const nsChildId of node.nsChildrenIds || []) {
                const childNode = modelGraph.nodesById[nsChildId];
                if (childNode != null) {
                  // Update its children's nsParent id.
                  childNode.nsParentId = node.id;
                  // BFS.
                  updateNamespace(childNode);
                }
              }
            }
          };
          for (const node of nodes) {
            updateNamespace(node);
          }

          if (curGroupNode == null) {
            // Remove the nodes in the current new group if they are in the root
            // node list.
            for (const node of nodes) {
              const index = modelGraph.rootNodes.indexOf(node);
              if (index >= 0) {
                modelGraph.rootNodes.splice(index, 1);
              }
            }

            // Add the new group node to root node list if its namespace is
            // empty.
            if (newGroupNode.namespace === '') {
              modelGraph.rootNodes.push(newGroupNode);
            }
          }

          children = newGroupNodes;
        }

        // Update curGropNode's nsChildrenIds.
        if (curGroupNode != null) {
          curGroupNode.nsChildrenIds = newGroupNodes.map((node) => node.id);
        }
      }

      for (const child of children) {
        if (isGroupNode(child)) {
          queue.push(child);
        }
      }
    }

    if (hasLargeGroupNodes) {
      this.generateLayoutGraphConnections(modelGraph);
    }
  }

  populateDescendantsAndCounts(modelGraph: ModelGraph) {
    // For each group node, gather all its descendant nodes.
    let minOpNodeCount = Number.MAX_VALUE;
    let maxOpNodeCount = Number.NEGATIVE_INFINITY;
    for (const node of modelGraph.nodes) {
      if (isGroupNode(node)) {
        const descendants: ModelNode[] = [];
        this.gatherDescendants(modelGraph, node, descendants);
        node.descendantsNodeIds = descendants.map((node) => node.id);
        node.descendantsOpNodeIds = descendants
          .filter((node) => node.nodeType === NodeType.OP_NODE)
          .map((node) => node.id);
        const opNodeCount = (node.descendantsOpNodeIds || []).length;
        minOpNodeCount = Math.min(opNodeCount, minOpNodeCount);
        maxOpNodeCount = Math.max(opNodeCount, maxOpNodeCount);
      }
    }
    modelGraph.minDescendantOpNodeCount = minOpNodeCount;
    modelGraph.maxDescendantOpNodeCount = maxOpNodeCount;
  }

  createEmptyModelGraph(): ModelGraph {
    const modelGraph: ModelGraph = {
      id: this.graph.id,
      collectionLabel: this.graph.collectionLabel || '',
      nodes: [],
      nodesById: {},
      rootNodes: [],
      edgesByGroupNodeIds: {},
      layoutGraphEdges: {},
      minDescendantOpNodeCount: -1,
      maxDescendantOpNodeCount: -1,
    };
    if (this.graph.groupNodeAttributes) {
      modelGraph.groupNodeAttributes = this.graph.groupNodeAttributes;
    }
    if (this.graph.groupNodeConfigs) {
      modelGraph.groupNodeConfigs = this.graph.groupNodeConfigs;
    }
    if (this.graph.layoutConfigs) {
      modelGraph.layoutConfigs = this.graph.layoutConfigs;
    }

    return modelGraph;
  }

  private getAncestorNamespaces(ns: string): string[] {
    // The returned namespaces include `ns` as well.
    const components = this.getNonEmptyNamespaceComponents(ns);
    const namespaces: string[] = [];
    while (components.length > 0) {
      namespaces.push(components.join('/'));
      components.pop();
    }
    return namespaces;
  }

  private getNonEmptyNamespaceComponents(ns: string): string[] {
    return ns.split('/').filter((component) => component !== '');
  }

  private getGroupNodeIdFromNamespace(ns: string): string {
    return `${ns}___group___`;
  }

  private gatherDescendants(
    modelGraph: ModelGraph,
    curRoot: GroupNode,
    descendants: ModelNode[],
  ) {
    for (const childId of curRoot.nsChildrenIds || []) {
      const child = modelGraph.nodesById[childId];
      if (isGroupNode(child) || (isOpNode(child) && !child.hideInLayout)) {
        descendants.push(child);
      }
      if (isGroupNode(child)) {
        this.gatherDescendants(modelGraph, child, descendants);
      }
    }
  }

  private processAttrValue(
    key: string,
    value: NodeAttributeValue,
  ): NodeAttributeValue {
    if (typeof value === 'string') {
      // Process const value that in `dense<...>` format. This is for backward
      // compatibility.
      if (value.startsWith('dense<')) {
        const matches = value.match(CONST_VALUE_REGEX);
        if (matches != null && matches.length > 1) {
          const strTensorValue = matches[1];
          return formatTensorValues(strTensorValue);
        }
      }
      // Process tensor values.
      else if (key === TENSOR_VALUES_KEY) {
        return formatTensorValues(value);
      }
      return value.replaceAll('"', '') || '<empty>';
    } else {
      return value;
    }
  }

  private processMetadataList(metadataItems: MetadataItem[]) {
    const metadata: Record<string, KeyValuePairs> = {};
    for (const metadataItem of metadataItems) {
      const attrs: KeyValuePairs = {};
      for (const attr of metadataItem.attrs) {
        let key = attr.key;
        let value = attr.value;
        // Special handlings.
        if (key === 'tensor_shape') {
          key = 'shape';
          value = value
            .replace('tensor<', '')
            .replace('>', '')
            .replace('*', '∗')
            .split('x')
            .join(' x ');
        }
        attrs[key] = value;
      }
      metadata[metadataItem.id] = attrs;
    }
    return metadata;
  }
}

/**
 * Formats the given tensor values string.
 *
 * The given string is in the form of:
 * [[[1, 2], [3, 4]]]
 *
 * And we want to format it to:
 * [
 *   [
 *     [
 *       1,
 *       2
 *     ],
 *     [
 *       3,
 *       4
 *     ]
 *   ]
 * ]
 */
export function formatTensorValues(strValues: string): string {
  try {
    return JSON.stringify(JSON.parse(strValues), null, 2)
      .replaceAll('\\n', '\n')
      .trim();
  } catch (e) {
    return strValues;
  }
}
