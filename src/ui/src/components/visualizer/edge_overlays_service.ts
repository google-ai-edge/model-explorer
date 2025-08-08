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
import {AppService} from './app_service';
import {
  Edge,
  EdgeOverlaysData,
  ProcessedEdgeOverlay,
  ProcessedEdgeOverlaysData,
} from './common/edge_overlays';
import {Pane, ReadFileResp} from './common/types';
import {genUid} from './common/utils';

/** A service for managing edge overlays. */
@Injectable()
export class EdgeOverlaysService {
  readonly pane = signal<Pane | undefined>(undefined);

  readonly remoteSourceLoading = signal<boolean>(false);

  readonly allLoadedEdgeOverlays = signal<ProcessedEdgeOverlaysData[]>([]);

  readonly filteredLoadedEdgeOverlays = computed(() => {
    const curPane = this.appService.getPaneById(this.pane()?.id ?? '');
    if (!curPane) {
      return [];
    }
    const allLoadedEdgeOverlays = this.allLoadedEdgeOverlays();
    return allLoadedEdgeOverlays.filter(
      (overlayData) =>
        !overlayData.graphName ||
        overlayData.graphName === curPane.modelGraph?.id,
    );
  });

  readonly selectedOverlayIds = signal<string[]>([]);

  readonly selectedOverlays = computed(() => {
    const overlays: ProcessedEdgeOverlay[] = [];
    for (const overlayData of this.filteredLoadedEdgeOverlays()) {
      for (const overlay of overlayData.processedOverlays) {
        if (this.selectedOverlayIds().includes(overlay.id)) {
          overlays.push(overlay);
        }
      }
    }
    return overlays;
  });

  constructor(private readonly appService: AppService) {}

  setPane(pane: Pane) {
    this.pane.set(pane);
  }

  addOverlay(overlay: EdgeOverlaysData) {
    this.allLoadedEdgeOverlays.update((allLoadedOverlays) => {
      return [...allLoadedOverlays, processOverlay(overlay)];
    });
  }

  deleteOverlayData(id: string) {
    const overlaysDataToDelete = this.filteredLoadedEdgeOverlays().find(
      (overlaysData) => overlaysData.id === id,
    );
    this.allLoadedEdgeOverlays.update((overlayDataList) => {
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
      const allLoadedOverlaysDataList = this.allLoadedEdgeOverlays();
      const newOverlayData =
        allLoadedOverlaysDataList[allLoadedOverlaysDataList.length - 1];
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

  toggleShowEdgesConnectedToSelectedNodeOnly(overlayId: string) {
    this.allLoadedEdgeOverlays.update((loadedOverlays) => {
      const overlay = this.getProcessedEdgeOverlayById(overlayId);
      if (!overlay) {
        return loadedOverlays;
      }
      overlay.showEdgesConnectedToSelectedNodeOnly =
        !overlay.showEdgesConnectedToSelectedNodeOnly;
      return [...loadedOverlays];
    });
  }

  setVisibleEdgeHops(overlayId: string, visibleEdgeHops: number) {
    this.allLoadedEdgeOverlays.update((loadedOverlays) => {
      const overlay = this.getProcessedEdgeOverlayById(overlayId);
      if (!overlay) {
        return loadedOverlays;
      }
      overlay.visibleEdgeHops = visibleEdgeHops;
      return [...loadedOverlays];
    });
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

  private getProcessedEdgeOverlayById(
    overlayId: string,
  ): ProcessedEdgeOverlay | undefined {
    for (const overlayData of this.filteredLoadedEdgeOverlays()) {
      for (const overlay of overlayData.processedOverlays) {
        if (overlay.id === overlayId) {
          return overlay;
        }
      }
    }
    return undefined;
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
    const adjacencyMap = new Map<string, Edge[]>();
    const processedOverlay: ProcessedEdgeOverlay = {
      id: genUid(),
      nodeIds: new Set<string>(),
      adjacencyMap,
      ...overlay,
    };
    processedOverlayData.processedOverlays.push(processedOverlay);
    for (const edge of overlay.edges) {
      processedOverlay.nodeIds.add(edge.sourceNodeId);
      processedOverlay.nodeIds.add(edge.targetNodeId);

      // Build the adjacency map.
      //
      // Add the edge to the source node's list
      if (!adjacencyMap.has(edge.sourceNodeId)) {
        adjacencyMap.set(edge.sourceNodeId, []);
      }
      adjacencyMap.get(edge.sourceNodeId)?.push(edge);

      // Add the same edge to the target node's list
      if (!adjacencyMap.has(edge.targetNodeId)) {
        adjacencyMap.set(edge.targetNodeId, []);
      }
      adjacencyMap.get(edge.targetNodeId)?.push(edge);
    }
  }
  return processedOverlayData;
}
