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

import {Injectable} from '@angular/core';
import {OpNode} from './common/model_graph';
import {isOpNode} from './common/utils';
import {WebglRenderer} from './webgl_renderer';

/** IO tracing related data. */
export interface IoTracingData {
  tracedNodeId: string;
  visibleNodeIds: Set<string>;
}

/** Service for managing input/output tracing related tasks. */
@Injectable()
export class WebglRendererIoTracingService {
  curIoTracingData?: IoTracingData;

  private webglRenderer!: WebglRenderer;

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
  }

  genTracingData() {
    if (
      !this.webglRenderer.selectedNodeId ||
      !isOpNode(
        this.webglRenderer.curModelGraph.nodesById[
          this.webglRenderer.selectedNodeId
        ],
      )
    ) {
      return;
    }

    const visibleNodeIds = new Set<string>();

    // Find all ancestor op nodes.
    const seenAncestorNodeIds = new Set<string>();
    let queue: string[] = [this.webglRenderer.selectedNodeId];
    while (queue.length > 0) {
      const curNodeId = queue.shift()!;
      if (seenAncestorNodeIds.has(curNodeId)) {
        continue;
      }
      seenAncestorNodeIds.add(curNodeId);
      const curNode = this.webglRenderer.curModelGraph.nodesById[
        curNodeId
      ] as OpNode;
      if (!curNode.hideInLayout) {
        visibleNodeIds.add(curNodeId);
      }
      for (const incomingEdge of curNode.incomingEdges || []) {
        queue.push(incomingEdge.sourceNodeId);
      }
    }

    // Find all descendant op nodes.
    const seenDescendantNodeIds = new Set<string>();
    queue = [this.webglRenderer.selectedNodeId];
    while (queue.length > 0) {
      const curNodeId = queue.shift()!;
      if (seenDescendantNodeIds.has(curNodeId)) {
        continue;
      }
      seenDescendantNodeIds.add(curNodeId);
      const curNode = this.webglRenderer.curModelGraph.nodesById[
        curNodeId
      ] as OpNode;
      if (!curNode.hideInLayout) {
        visibleNodeIds.add(curNodeId);
      }
      for (const outgoingEdge of curNode.outgoingEdges || []) {
        queue.push(outgoingEdge.targetNodeId);
      }
    }

    // Add all their parent group nodes to `visibleNodeIds`.
    for (const nodeId of [...visibleNodeIds]) {
      let curNodeId = nodeId;
      while (true) {
        const node = this.webglRenderer.curModelGraph.nodesById[curNodeId];
        if (!node.nsParentId || visibleNodeIds.has(node.nsParentId)) {
          break;
        }
        curNodeId = node.nsParentId;
        visibleNodeIds.add(curNodeId);
      }
    }

    this.curIoTracingData = {
      tracedNodeId: this.webglRenderer.selectedNodeId,
      visibleNodeIds,
    };
  }

  clearTracingData() {
    this.curIoTracingData = undefined;
  }
}
