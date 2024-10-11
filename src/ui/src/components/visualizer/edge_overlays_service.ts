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

import {Injectable, computed, signal} from '@angular/core';
import {
  EdgeOverlaysData,
  ProcessedEdgeOverlay,
  ProcessedEdgeOverlaysData,
} from './common/edge_overlays';
import {ReadFileResp} from './common/types';
import {genUid} from './common/utils';

/** A service for managing edge overlays. */
@Injectable()
export class EdgeOverlaysService {
  readonly remoteSourceLoading = signal<boolean>(false);

  readonly loadedEdgeOverlays = signal<ProcessedEdgeOverlaysData[]>([]);

  readonly selectedOverlayIds = signal<string[]>([]);

  readonly selectedOverlays = computed(() => {
    const overlays: ProcessedEdgeOverlay[] = [];
    for (const overlayData of this.loadedEdgeOverlays()) {
      for (const overlay of overlayData.processedOverlays) {
        if (this.selectedOverlayIds().includes(overlay.id)) {
          overlays.push(overlay);
        }
      }
    }
    return overlays;
  });

  addOverlay(overlay: EdgeOverlaysData) {
    this.loadedEdgeOverlays.update((loadedOverlays) => {
      return [...loadedOverlays, processOverlay(overlay)];
    });
  }

  deleteOverlayData(id: string) {
    const overlaysDataToDelete = this.loadedEdgeOverlays().find(
      (overlaysData) => overlaysData.id === id,
    );
    this.loadedEdgeOverlays.update((overlayDataList) => {
      return overlayDataList.filter((overlayData) => overlayData.id !== id);
    });

    // Update selected overlays.
    if (overlaysDataToDelete) {
      const overlayIdsToDelete = new Set<string>(
        overlaysDataToDelete.processedOverlays.map((overlay) => overlay.id),
      );
      this.selectedOverlayIds.update((selectedOverlayIds) => {
        return selectedOverlayIds.filter((id) => !overlayIdsToDelete.has(id));
      });
    }
  }

  toggleOverlaySelection(idToToggle: string) {
    this.selectedOverlayIds.update((selectedOverlayIds) => {
      let ids = [...selectedOverlayIds];
      if (selectedOverlayIds.includes(idToToggle)) {
        ids = ids.filter((id) => id !== idToToggle);
      } else {
        ids.push(idToToggle);
      }
      return ids;
    });
  }

  addEdgeOverlayData(data: EdgeOverlaysData) {
    this.addOverlay(data);

    // Select all newly-added overlays.
    this.selectedOverlayIds.update((selectedOverlayIds) => {
      const loadedOverlaysDataList = this.loadedEdgeOverlays();
      const newOverlayData =
        loadedOverlaysDataList[loadedOverlaysDataList.length - 1];
      const newIds = newOverlayData.processedOverlays.map(
        (overlay) => overlay.id,
      );
      return [...selectedOverlayIds, ...newIds];
    });
  }

  addEdgeOverlayDataFromJsonData(str: string): string {
    try {
      const data = JSON.parse(str) as EdgeOverlaysData;
      this.addEdgeOverlayData(data);
    } catch (e) {
      return `Failed to parse JSON file. ${e}`;
    }
    return '';
  }

  async loadFromCns(path: string): Promise<string> {
    // Call API to read file content.
    this.remoteSourceLoading.set(true);
    const url = `/read_file?path=${path}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      this.remoteSourceLoading.set(false);
      return `Failed to load JSON file "${path}"`;
    }

    // Parse response.
    const json = JSON.parse(
      (await resp.text()).replace(")]}'\n", ''),
    ) as ReadFileResp;

    const error = this.addEdgeOverlayDataFromJsonData(json.content);

    this.remoteSourceLoading.set(false);

    return error;
  }
}

function processOverlay(
  overlayData: EdgeOverlaysData,
): ProcessedEdgeOverlaysData {
  const processedOverlayData: ProcessedEdgeOverlaysData = {
    id: genUid(),
    processedOverlays: [],
    ...overlayData,
  };
  for (const overlay of overlayData.overlays) {
    const processedOverlay: ProcessedEdgeOverlay = {
      id: genUid(),
      nodeIds: new Set<string>(),
      ...overlay,
    };
    processedOverlayData.processedOverlays.push(processedOverlay);
    for (const edge of overlay.edges) {
      processedOverlay.nodeIds.add(edge.sourceNodeId);
      processedOverlay.nodeIds.add(edge.targetNodeId);
    }
  }
  return processedOverlayData;
}
