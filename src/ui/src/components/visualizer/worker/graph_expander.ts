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

import {LAYOUT_MARGIN_X, NODE_LABEL_LINE_HEIGHT} from '../common/consts';
import {GroupNode, ModelGraph, OpNode} from '../common/model_graph';
import {
  NodeDataProviderRunData,
  Rect,
  ShowOnNodeItemData,
} from '../common/types';
import {
  getDeepestExpandedGroupNodeIds,
  getMultiLineLabelExtraHeight,
  isGroupNode,
  splitLabel,
} from '../common/utils';
import {VisualizerConfig} from '../common/visualizer_config';

import {Dagre, DagreGraphInstance} from './dagre_types';
import {
  GraphLayout,
  LAYOUT_MARGIN_BOTTOM,
  LAYOUT_MARGIN_TOP,
  getNodeHeight,
  getNodeWidth,
} from './graph_layout';

/**
 * A class that handles expanding and collapsing group nodes in a model graph.
 */
export class GraphExpander {
  /** This is for testing purpose. */
  readonly dagreGraphs: DagreGraphInstance[] = [];

  constructor(
    private readonly modelGraph: ModelGraph,
    private readonly dagre: Dagre,
    private readonly showOnNodeItemTypes: Record<string, ShowOnNodeItemData>,
    private readonly nodeDataProviderRuns: Record<
      string,
      NodeDataProviderRunData
    >,
    private readonly selectedNodeDataProviderRunId: string | undefined,
    private readonly testMode = false,
    private readonly config?: VisualizerConfig,
  ) {}

  /** Expands the given group node to show its child nodes. */
  expandGroupNode(groupNodeId: string) {
    const groupNode = this.modelGraph.nodesById[groupNodeId];
    if (groupNode && isGroupNode(groupNode)) {
      if (groupNode.expanded) {
        return;
      }
      groupNode.expanded = true;
    }

    // From the given group node, layout its children, grow its size, and
    // continue to do the same for all its ancestors until reaching the root.
    let curGroupNodeId: string | undefined = groupNodeId;
    while (curGroupNodeId != null) {
      const curGroupNode = this.modelGraph.nodesById[
        curGroupNodeId
      ] as GroupNode;
      if (!curGroupNode) {
        break;
      }
      curGroupNode.expanded = true;

      // Layout children.
      const layout = new GraphLayout(
        this.modelGraph,
        this.dagre,
        this.showOnNodeItemTypes,
        this.nodeDataProviderRuns,
        this.selectedNodeDataProviderRunId,
        this.testMode,
        this.config,
      );
      const rect = layout.layout(curGroupNodeId);
      if (this.testMode) {
        this.dagreGraphs.push(layout.dagreGraph);
      }

      // Grow size.
      const curTargetWidth = rect.width + LAYOUT_MARGIN_X * 2;
      const curTargetHeight = this.getTargetGroupNodeHeight(rect, curGroupNode);
      curGroupNode.width = curTargetWidth;
      curGroupNode.height = curTargetHeight;

      // Continue with parent.
      curGroupNodeId = curGroupNode.nsParentId;
    }

    // Layout the root level nodes.
    const layout = new GraphLayout(
      this.modelGraph,
      this.dagre,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
      this.testMode,
      this.config,
    );
    layout.layout();
    if (this.testMode) {
      this.dagreGraphs.push(layout.dagreGraph);
    }

    // From root, update offsets of all nodes that have x, y set (meaning they
    // have the layout data).
    for (const node of this.modelGraph.rootNodes) {
      if (isGroupNode(node)) {
        this.updateNodeOffset(node);
      }
    }
  }

  /** Expands from the given deepest group nodes back to root. */
  expandFromDeepestGroupNodes(groupNodeIds: string[]) {
    // Get all ancestors from the given group node ids.
    const seenGroupNodeIds = new Set<string>();
    const queue: string[] = [...groupNodeIds];
    while (queue.length > 0) {
      const curGroupNodeId = queue.shift()!;
      if (
        seenGroupNodeIds.has(curGroupNodeId) ||
        !this.modelGraph.nodesById[curGroupNodeId]
      ) {
        continue;
      }
      seenGroupNodeIds.add(curGroupNodeId);
      const groupNode = this.modelGraph.nodesById[curGroupNodeId] as GroupNode;
      const parentGroupNodeId = groupNode?.nsParentId;
      if (parentGroupNodeId) {
        queue.push(parentGroupNodeId);
      }
    }

    // Sort them by level in descending order.
    const sortedGroupNodeIds = Array.from(seenGroupNodeIds).sort((a, b) => {
      const nodeA = this.modelGraph.nodesById[a];
      const nodeB = this.modelGraph.nodesById[b];
      return nodeB.level - nodeA.level;
    });

    // Layout group nodes in this sorted list.
    for (const groupNodeId of sortedGroupNodeIds) {
      const groupNode = this.modelGraph.nodesById[groupNodeId] as GroupNode;
      groupNode.expanded = true;

      // Layout children.
      const layout = new GraphLayout(
        this.modelGraph,
        this.dagre,
        this.showOnNodeItemTypes,
        this.nodeDataProviderRuns,
        this.selectedNodeDataProviderRunId,
        this.testMode,
        this.config,
      );
      const rect = layout.layout(groupNodeId);
      if (this.testMode) {
        this.dagreGraphs.push(layout.dagreGraph);
      }

      // Grow size.
      const curTargetWidth = rect.width + LAYOUT_MARGIN_X * 2;
      const curTargetHeight = this.getTargetGroupNodeHeight(rect, groupNode);
      groupNode.width = curTargetWidth;
      groupNode.height = curTargetHeight;
    }

    // Layout the root level nodes.
    const layout = new GraphLayout(
      this.modelGraph,
      this.dagre,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
      this.testMode,
      this.config,
    );
    layout.layout();
    if (this.testMode) {
      this.dagreGraphs.push(layout.dagreGraph);
    }

    // From root, update offsets of all nodes that have x, y set (meaning they
    // have the layout data).
    for (const node of this.modelGraph.rootNodes) {
      if (isGroupNode(node)) {
        this.updateNodeOffset(node);
      }
    }
  }

  /** Expands the graph to reveal the given node. */
  expandToRevealNode(nodeId: string): string[] {
    const node = this.modelGraph.nodesById[nodeId];
    const groupNodes: GroupNode[] = [];
    let curNode = node;
    while (true) {
      const nsParent = this.modelGraph.nodesById[
        curNode.nsParentId || ''
      ] as GroupNode;
      if (!nsParent) {
        break;
      }
      groupNodes.unshift(nsParent);
      curNode = nsParent;
    }
    for (const groupNode of groupNodes) {
      this.expandGroupNode(groupNode.id);
    }

    const deepestExpandedGroupNodeIds: string[] = [];
    getDeepestExpandedGroupNodeIds(
      undefined,
      this.modelGraph,
      deepestExpandedGroupNodeIds,
    );
    return deepestExpandedGroupNodeIds;
  }

  /** Collapses the given group node to hide all its child nodes. */
  collapseGroupNode(groupNodeId: string): string[] {
    const groupNode = this.modelGraph.nodesById[groupNodeId] as GroupNode;
    if (!groupNode) {
      return [];
    }
    groupNode.expanded = false;
    delete this.modelGraph.edgesByGroupNodeIds[groupNodeId];

    // Shrink size for the current group node.
    groupNode.width = getNodeWidth(
      groupNode,
      this.modelGraph,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
    );
    groupNode.height = getNodeHeight(
      groupNode,
      this.modelGraph,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
      this.testMode,
      true,
      this.config,
    );

    // From the given group node's parent, layout, update size, and continue to
    // do the same for all its ancestors until reaching the root.
    let curGroupNodeId: string | undefined = groupNode.nsParentId;
    while (curGroupNodeId != null) {
      const curGroupNode = this.modelGraph.nodesById[
        curGroupNodeId
      ] as GroupNode;
      if (!curGroupNode) {
        break;
      }

      // Layout.
      const layout = new GraphLayout(
        this.modelGraph,
        this.dagre,
        this.showOnNodeItemTypes,
        this.nodeDataProviderRuns,
        this.selectedNodeDataProviderRunId,
        this.testMode,
        this.config,
      );
      const rect = layout.layout(curGroupNodeId);
      if (this.testMode) {
        this.dagreGraphs.push(layout.dagreGraph);
      }

      // Shrink size.
      const curTargetWidth = rect.width + LAYOUT_MARGIN_X * 2;
      const curTargetHeight = this.getTargetGroupNodeHeight(rect, curGroupNode);
      curGroupNode.width = curTargetWidth;
      curGroupNode.height = curTargetHeight;

      // Continue with parent.
      curGroupNodeId = curGroupNode.nsParentId;
    }

    // Layout the root level nodes.
    const layout = new GraphLayout(
      this.modelGraph,
      this.dagre,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
      this.testMode,
      this.config,
    );
    layout.layout();
    if (this.testMode) {
      this.dagreGraphs.push(layout.dagreGraph);
    }

    // From root, update offsets of all nodes that have x, y set (meaning they
    // have the layout data).
    for (const node of this.modelGraph.rootNodes) {
      if (isGroupNode(node)) {
        this.updateNodeOffset(node);
      }
    }

    const deepestExpandedGroupNodeIds: string[] = [];
    getDeepestExpandedGroupNodeIds(
      undefined,
      this.modelGraph,
      deepestExpandedGroupNodeIds,
    );
    return deepestExpandedGroupNodeIds;
  }

  /**
   * Uses the current collapse/expand states of the group nodes and re-lays out
   * the entire graph.
   */
  reLayoutGraph(
    targetDeepestGroupNodeIdsToExpand?: string[],
    clearAllExpandStates?: boolean,
  ): string[] {
    let curTargetDeepestGroupNodeIdsToExpand: string[] | undefined =
      targetDeepestGroupNodeIdsToExpand;
    if (!curTargetDeepestGroupNodeIdsToExpand) {
      // Find the deepest group nodes that non of its child group nodes is
      // expanded.
      const deepestExpandedGroupNodeIds: string[] = [];
      this.clearLayoutData(undefined);
      getDeepestExpandedGroupNodeIds(
        undefined,
        this.modelGraph,
        deepestExpandedGroupNodeIds,
      );
      curTargetDeepestGroupNodeIdsToExpand = deepestExpandedGroupNodeIds;
    } else {
      if (clearAllExpandStates) {
        this.clearLayoutData(undefined, true);
      }
    }

    // Expand those nodes one by one.
    if (curTargetDeepestGroupNodeIdsToExpand.length > 0) {
      this.expandFromDeepestGroupNodes(curTargetDeepestGroupNodeIdsToExpand);
    } else {
      const layout = new GraphLayout(
        this.modelGraph,
        this.dagre,
        this.showOnNodeItemTypes,
        this.nodeDataProviderRuns,
        this.selectedNodeDataProviderRunId,
        this.testMode,
        this.config,
      );
      layout.layout();
    }

    return curTargetDeepestGroupNodeIdsToExpand;
  }

  expandAllGroups(): string[] {
    this.clearLayoutData(undefined, true);

    // Find all deepest group nodes.
    const deepestGroupNodeIds = this.modelGraph.nodes
      .filter(
        (node) =>
          isGroupNode(node) &&
          (node.nsChildrenIds || []).filter((id) =>
            isGroupNode(this.modelGraph.nodesById[id]),
          ).length === 0,
      )
      .map((node) => node.id);

    // Expand from them.
    if (deepestGroupNodeIds.length > 0) {
      this.expandFromDeepestGroupNodes(deepestGroupNodeIds);
    }

    return deepestGroupNodeIds;
  }

  collapseAllGroup(): string[] {
    this.clearLayoutData(undefined, true);

    // Layout the root level nodes.
    const layout = new GraphLayout(
      this.modelGraph,
      this.dagre,
      this.showOnNodeItemTypes,
      this.nodeDataProviderRuns,
      this.selectedNodeDataProviderRunId,
      this.testMode,
      this.config,
    );
    layout.layout();

    // From root, update offsets of all nodes that have x, y set (meaning they
    // have the layout data).
    for (const node of this.modelGraph.rootNodes) {
      if (isGroupNode(node)) {
        this.updateNodeOffset(node);
      }
    }

    return [];
  }

  private updateNodeOffset(groupNode: GroupNode) {
    for (const nodeId of groupNode.nsChildrenIds || []) {
      const node = this.modelGraph.nodesById[nodeId];
      if (node.x != null && node.y != null) {
        node.globalX =
          (groupNode.x || 0) +
          (groupNode.globalX || 0) +
          (node.localOffsetX || 0);
        node.globalY =
          (groupNode.y || 0) +
          (groupNode.globalY || 0) +
          (node.localOffsetY || 0);

        // Move the node down to make space for multi-line node label.
        const extraLabelHeight =
          (splitLabel(groupNode.label).length - 1) * NODE_LABEL_LINE_HEIGHT;
        if (extraLabelHeight > 0) {
          node.globalY += extraLabelHeight;
        }

        // Move the node down if the current group node has a node pinned to
        // top.
        if (
          groupNode.pinToTopOpNode &&
          node.id !== groupNode.pinToTopOpNode.id
        ) {
          node.globalY += this.getPinToTopNodeVerticalSpace(
            groupNode.pinToTopOpNode,
          );
        }

        // For the pinned-to-top node, move it to the top-middle of the group
        // node.
        if (groupNode.pinToTopOpNode?.id === node.id) {
          node.globalX =
            (groupNode.x || 0) +
            (groupNode.globalX || 0) +
            (groupNode.width || 0) / 2;
          node.globalY =
            (groupNode.y || 0) +
            (groupNode.globalY || 0) +
            (node.localOffsetY || 0) +
            this.getPinToTopNodeVerticalSpace(node as OpNode) -
            (node.height || 0) / 2 +
            10;
        }
      }
      if (isGroupNode(node)) {
        this.updateNodeOffset(node);
      }
    }
  }

  private clearLayoutData(
    root: GroupNode | undefined,
    clearAllExpandStates?: boolean,
  ) {
    let nsChildrenIds: string[] = [];
    if (root == null) {
      nsChildrenIds = this.modelGraph.rootNodes.map((node) => node.id);
    } else {
      nsChildrenIds = root.nsChildrenIds || [];
    }
    if (clearAllExpandStates && root != null) {
      root.expanded = false;
      delete this.modelGraph.edgesByGroupNodeIds[root.id];
    }
    for (const nsChildNodeId of nsChildrenIds) {
      const childNode = this.modelGraph.nodesById[nsChildNodeId];
      if (!childNode) {
        continue;
      }
      childNode.width = undefined;
      childNode.height = undefined;
      if (isGroupNode(childNode) && childNode.expanded) {
        this.clearLayoutData(childNode, clearAllExpandStates);
      }
    }
  }

  private getPinToTopNodeVerticalSpace(node: OpNode): number {
    return (node.height || 0) + 20;
  }

  private getTargetGroupNodeHeight(rect: Rect, groupNode: GroupNode): number {
    const extraMultiLineLabelHeight = getMultiLineLabelExtraHeight(
      groupNode.label,
    );
    let targetHeight =
      rect.height +
      LAYOUT_MARGIN_TOP +
      LAYOUT_MARGIN_BOTTOM +
      extraMultiLineLabelHeight;
    if (groupNode.pinToTopOpNode) {
      targetHeight += this.getPinToTopNodeVerticalSpace(
        groupNode.pinToTopOpNode,
      );
    }
    return targetHeight;
  }
}
