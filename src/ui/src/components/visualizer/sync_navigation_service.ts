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
  NavigationSourceInfo,
  SyncNavigationData,
  SyncNavigationMode,
} from './common/sync_navigation';
import {ReadFileResp, SyncNavigationModeChangedEvent} from './common/types';

import {Injectable, signal} from '@angular/core';
import {Subject} from 'rxjs';

declare interface ProcessedSyncNavigationData extends SyncNavigationData {
  inversedMapping: Record<string, string>;
}

/** A service for split pane sync navigation related tasks. */
@Injectable()
export class SyncNavigationService {
  readonly mode = signal<SyncNavigationMode>(SyncNavigationMode.DISABLED);
  readonly navigationSourceChanged$ = new Subject<NavigationSourceInfo>();
  readonly loadingFromCns = signal<boolean>(false);

  // Used for notifying mode change to other components.
  readonly syncNavigationModeChanged$ =
    new Subject<SyncNavigationModeChangedEvent>();

  // {} means showing the message, and undefined means hiding the message.
  readonly showNoMappedNodeMessageTrigger$ = new Subject<{} | undefined>();

  private savedProcessedSyncNavigationData: Record<
    string,
    ProcessedSyncNavigationData
  > = {};

  updateNavigationSource(info: NavigationSourceInfo) {
    if (this.mode() === SyncNavigationMode.DISABLED) {
      return;
    }
    this.navigationSourceChanged$.next(info);
  }

  updateSyncNavigationData(mode: SyncNavigationMode, data: SyncNavigationData) {
    // Generate inversed mapping.
    const processedData: ProcessedSyncNavigationData = {
      ...data,
      inversedMapping: {},
    };
    for (const key of Object.keys(data.mapping)) {
      processedData.inversedMapping[data.mapping[key]] = key;
    }

    // Save it.
    this.savedProcessedSyncNavigationData[mode] = processedData;
  }

  getMappedNodeId(paneIndex: number, nodeId: string): string {
    const mode = this.mode();
    const curSyncNavigationData: ProcessedSyncNavigationData | undefined =
      this.savedProcessedSyncNavigationData[mode];

    switch (mode) {
      case SyncNavigationMode.MATCH_NODE_ID: {
        return nodeId;
      }
      case SyncNavigationMode.VISUALIZER_CONFIG:
      case SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER:
      case SyncNavigationMode.LOAD_MAPPING_FROM_CNS: {
        // Get mapped node id from mapping.
        // Fallback to the original node id if not found.
        const mapping = curSyncNavigationData?.mapping ?? {};
        const inversedMapping = curSyncNavigationData?.inversedMapping ?? {};
        return mapping[nodeId] ?? inversedMapping[nodeId] ?? nodeId;
      }
      default:
        return nodeId;
    }
  }

  async loadFromCns(path: string): Promise<string> {
    // Call API to read file content.
    this.loadingFromCns.set(true);
    const url = `/read_file?path=${path}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      this.loadingFromCns.set(false);
      return `Failed to load JSON file "${path}"`;
    }

    // Parse response.
    const json = JSON.parse(
      (await resp.text()).replace(")]}'\n", ''),
    ) as ReadFileResp;

    const error = this.processJsonData(
      json.content,
      SyncNavigationMode.LOAD_MAPPING_FROM_CNS,
    );

    this.loadingFromCns.set(false);

    return error;
  }

  async loadSyncNavigationDataFromEvent(event: SyncNavigationModeChangedEvent) {

    // Set mode.
    this.mode.set(event.mode);
  }

  processJsonData(str: string, mode: SyncNavigationMode): string {
    try {
      const data = JSON.parse(str) as SyncNavigationData;
      this.updateSyncNavigationData(mode, data);
    } catch (e) {
      return `Failed to parse JSON file. ${e}`;
    }
    return '';
  }
}
