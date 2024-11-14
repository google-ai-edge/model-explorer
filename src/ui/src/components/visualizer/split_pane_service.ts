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

import {Injectable, signal} from '@angular/core';

/** A service for managing data scoped within a single split pane. */
@Injectable()
export class SplitPaneService {
  /**
   * The input op node ids that are hidden (i.e. not highlighted) in the
   * model graph.
   */
  readonly hiddenInputOpNodeIds = signal<Record<string, boolean>>({});

  /**
   * The {nodeId}___{outputId} that are hidden (i.e. not highlighted) in the
   * model graph.
   */
  readonly hiddenOutputIds = signal<Record<string, boolean>>({});

  toggleInputOpNodeVisibility(nodeId: string) {
    this.hiddenInputOpNodeIds.update((ids) => {
      const visible = ids[nodeId] === true;
      if (!visible) {
        ids[nodeId] = true;
      } else {
        delete ids[nodeId];
      }
      return {...ids};
    });
  }

  setInputOpNodeVisible(nodeId: string, allNodeIds: string[]) {
    // Check if the node is the only visible one.
    let isNodeTheOnlyVisibleOne = this.hiddenInputOpNodeIds()[nodeId] !== true;
    for (const id of allNodeIds) {
      if (id !== nodeId) {
        if (!this.hiddenInputOpNodeIds()[id]) {
          isNodeTheOnlyVisibleOne = false;
        }
      }
    }

    // If so, show all the nodes.
    if (isNodeTheOnlyVisibleOne) {
      this.hiddenInputOpNodeIds.set({});
    }
    // If not, hide the other nodes.
    else {
      const ids: Record<string, boolean> = {};
      for (const id of allNodeIds) {
        if (id !== nodeId) {
          ids[id] = true;
        }
      }
      this.hiddenInputOpNodeIds.set(ids);
    }
  }

  toggleOutputVisibility(nodeId: string, outputId: string) {
    this.hiddenOutputIds.update((ids) => {
      const key = `${nodeId}___${outputId}`;
      const visible = ids[key] === true;
      if (!visible) {
        ids[key] = true;
      } else {
        delete ids[key];
      }
      return {...ids};
    });
  }

  setOutputVisible(
    nodeId: string,
    outputId: string,
    allNodeAndOutputIds: Array<{nodeId: string; outputId: string}>,
  ) {
    // Check if the output is the only visible one.
    const key = `${nodeId}___${outputId}`;
    let isNodeTheOnlyVisibleOne = this.hiddenOutputIds()[key] !== true;
    for (const {nodeId, outputId} of allNodeAndOutputIds) {
      const curKey = `${nodeId}___${outputId}`;
      if (curKey !== key) {
        if (!this.hiddenOutputIds()[curKey]) {
          isNodeTheOnlyVisibleOne = false;
        }
      }
    }

    // If so, show all the outputs.
    if (isNodeTheOnlyVisibleOne) {
      this.hiddenOutputIds.set({});
    }
    // If not, hide the other outputs.
    else {
      const ids: Record<string, boolean> = {};
      for (const {nodeId, outputId} of allNodeAndOutputIds) {
        const curKey = `${nodeId}___${outputId}`;
        if (curKey !== key) {
          ids[curKey] = true;
        }
      }
      this.hiddenOutputIds.set(ids);
    }
  }

  getInputOpNodeVisible(nodeId: string): boolean {
    return !this.hiddenInputOpNodeIds()[nodeId];
  }

  getOutputVisible(nodeId: string, outputId: string): boolean {
    const key = `${nodeId}___${outputId}`;
    return !this.hiddenOutputIds()[key];
  }

  resetInputOutputHiddenIds() {
    this.hiddenInputOpNodeIds.set({});
    this.hiddenOutputIds.set({});
  }
}
