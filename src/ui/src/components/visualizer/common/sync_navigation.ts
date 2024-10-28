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
  mapping: SyncNavigationMapping;

  /**
   * Whether to disable the fallback behavior (find the node with the same id)
   * when the mapped node is not found from the `mapping` field above.
   */
  disableMappingFallback?: boolean;
}

/**
 * The mapping for navigation syncing, from node id from left side to node id
 * from right side.
 */
export type SyncNavigationMapping = Record<string, string>;

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
