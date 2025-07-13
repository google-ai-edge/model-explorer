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

import {TaskData, TaskType} from './task';

/** The data for navigation syncing. */
export declare interface SyncNavigationData extends TaskData {
  type: TaskType.SYNC_NAVIGATION;

  /**
   * Specifies the mapping for navigation syncing.
   *
   * When user selects a node on one side, Model Explorer will try to find the
   * mapped node on the other side and select it automacitally. If the mapped
   * node is not found, Model Explorer will try to find the node with the same
   * node id on the other side. This fallback behavior can be disabled by
   * setting `disableMappingFallback` below to true.
   */
  mapping?: SyncNavigationMapping;

  /**
   * The more flexible mapping specification for navigation syncing that
   * supports one to many, many to many, and many to one mapping.
   *
   * Model Explorer assumes that a node id from either side will only appear in
   * one SyncNavigationMappingEntry. For example, the following mapping is not
   * recommended because 'a' appears in two SyncNavigationMappingEntries:
   *
   * ```
   * {
   *   mappingEntries: [
   *     {
   *       leftNodeIds: ['a'],
   *       rightNodeIds: ['b', 'c'],
   *     },
   *     {
   *       leftNodeIds: ['a', 'd'],
   *       rightNodeIds: ['x'],
   *     },
   *   ],
   * }
   * ```
   *
   * This field supersedes the `mapping` field above.
   */
  mappingEntries?: SyncNavigationMappingEntry[];

  /**
   * Whether to disable the fallback behavior (find the node with the same id)
   * when the mapped node is not found from the `mapping` field above.
   */
  disableMappingFallback?: boolean;

  /**
   * The border color used to highlight "related nodes". Default to #ff00be
   * (pink). The highlight is rendered as a colored border around the node.
   *
   * == What are "related nodes"?
   *
   * Assume we have a mapping from 'a' to ['b', 'c', 'd']. In this case, 'b',
   * 'c' and 'd' are "related nodes".
   *
   * == When are related nodes highlighted?
   *
   * - When 'a' is selected in the left pane, 'b', 'c', and 'd' in the right
   *   pane will be highlighted.
   * - When 'b', 'c', OR 'd' is selected in the right pane, all of 'b', 'c', and
   *   'd' in the right pane will be highlighted.
   */
  relatedNodesBorderColor?: string;

  /**
   * The border width used to highlight "related nodes". Default to 2.
   *
   * See comments above for "related nodes".
   */
  relatedNodesBorderWidth?: number;

  /**
   * Whether to show diff highlights.
   *
   * When enabled, Model Explorer will render diff highlights on nodes based on
   * the current sync navigation mode. If a node has mapped nodes in the other
   * pane, but all mapped nodes are missing in the other pane, Model Explorer
   * will highlight the current node with a specific border color and width (can
   * be configured below) to indicate a diff.
   */
  showDiffHighlights?: boolean;

  /**
   * The border color and width used to highlight deleted nodes. Default to red
   * color and 2 px width.
   *
   * The deleted nodes are nodes that exist in the left pane but not in the right
   * pane based on the mapping, and will be highlighted in the left pane.
   */
  deletedNodesBorderColor?: string;
  deletedNodesBorderWidth?: number;

  /**
   * The border color and width used to highlight new nodes. Default to green
   * color and 2 px width.
   *
   * The new nodes are nodes that exist in the right pane but not in the left
   * pane based on the mapping, and will be highlighted in the right pane.
   */
  newNodesBorderColor?: string;
  newNodesBorderWidth?: number;
}

/**
 * The mapping for navigation syncing, from node id from left side to node id
 * from right side.
 */
export type SyncNavigationMapping = Record<string, string>;

/**
 * The mapping entry for navigation syncing, used for more flexible mapping
 * specification, such as one to many, many to many, and many to one mapping.
 */
export declare interface SyncNavigationMappingEntry {
  leftNodeIds: string[];
  rightNodeIds: string[];
}

/** The mode of navigation syncing. */
export enum SyncNavigationMode {
  DISABLED = 'disabled',
  MATCH_NODE_ID = 'match_node_id',
  VISUALIZER_CONFIG = 'visualizer_config',
  UPLOAD_MAPPING_FROM_COMPUTER = 'from_computer',
  LOAD_MAPPING_FROM_CNS = 'from_cns',
}

/** The labels for sync navigation modes. */
export const SYNC_NAVIGATION_MODE_LABELS = {
  [SyncNavigationMode.DISABLED]: 'Disabled',
  [SyncNavigationMode.MATCH_NODE_ID]: 'Match node id',
  [SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER]:
    'Upload mapping from computer',
  [SyncNavigationMode.LOAD_MAPPING_FROM_CNS]: 'Load mapping from CNS',
  [SyncNavigationMode.VISUALIZER_CONFIG]: 'From Visualizer Config',
};

/** Information about the source of navigation. */
export interface NavigationSourceInfo {
  paneIndex: number;
  nodeId: string;
}
