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

import {DestroyRef, Injectable} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {AppService} from './app_service';
import {Rect, SnapshotData} from './common/types';
import {genUid, getDeepestExpandedGroupNodeIds} from './common/utils';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';

const MAX_SCREENSHOT_WIDTH = 320;

/** Service for managing snapshot related tasks. */
@Injectable()
export class WebglRendererSnapshotService {
  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;

  constructor(
    private readonly appService: AppService,
    private readonly destroyRef: DestroyRef,
  ) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;

    // Handle "add snapshot".
    this.appService.addSnapshotClicked
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        if (data.rendererId !== this.webglRenderer.rendererId) {
          return;
        }
        this.addSnapshot();
      });

    // Handle "snapshot to restore".
    this.appService.curSnapshotToRestore
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        if (data.rendererId !== this.webglRenderer.rendererId) {
          return;
        }
        this.restoreSnapshot(data.snapshot);
      });
  }

  async addSnapshot() {
    this.webglRenderer.flash();

    const snapshot = await this.takeSnapshot();
    this.webglRenderer.appService.addSnapshot(
      snapshot,
      this.webglRenderer.curModelGraph.id,
      this.webglRenderer.paneId,
    );
  }

  async takeSnapshot(): Promise<SnapshotData> {
    // Gather snapshot data.
    //
    // Rect for the current area.
    const container = this.webglRenderer.container.nativeElement;
    const start = this.webglRendererThreejsService.convertScreenPosToScene(
      0,
      0,
    );
    const end = this.webglRendererThreejsService.convertScreenPosToScene(
      container.clientWidth,
      container.clientHeight,
    );
    const rect: Rect = {
      x: start.x,
      y: start.y,
      width: end.x - start.x,
      height: end.y - start.y,
    };

    // Expanded groups.
    const deepestExpandedGroupNodeIds: string[] = [];
    getDeepestExpandedGroupNodeIds(
      undefined,
      this.webglRenderer.curModelGraph,
      deepestExpandedGroupNodeIds,
    );

    // Screenshot using the snapshot renderer.
    const canvas = this.webglRenderer.canvas.nativeElement;
    const snapshotCanvas = this.webglRenderer.snapshotCanvas.nativeElement;
    const canvasWidth = MAX_SCREENSHOT_WIDTH;
    const canvasHeight = (MAX_SCREENSHOT_WIDTH / canvas.width) * canvas.height;
    this.webglRendererThreejsService.renderSnapshot(canvasWidth, canvasHeight);
    const pixelRatio = window.devicePixelRatio;
    const offscreenCanvas = new OffscreenCanvas(
      canvasWidth * pixelRatio,
      canvasHeight * pixelRatio,
    );
    const ctx = offscreenCanvas.getContext('2d')!;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(
      snapshotCanvas,
      0,
      0,
      snapshotCanvas.width,
      snapshotCanvas.height,
      0,
      0,
      offscreenCanvas.width,
      offscreenCanvas.height,
    );
    const imageBitmap = await createImageBitmap(offscreenCanvas);

    // Flatten layers toggle.
    const flattenLayers = this.webglRenderer.appService.getFlattenLayers(
      this.webglRenderer.paneId,
    );

    // Create snapshot.
    const snapshot: SnapshotData = {
      id: genUid(),
      rect,
      imageBitmap,
      selectedNodeId: this.webglRenderer.selectedNodeId,
      deepestExpandedGroupNodeIds,
      showOnNodeItemTypes: {...this.webglRenderer.curShowOnNodeItemTypes},
      showOnEdgeItem: this.webglRenderer.curShowOnEdgeItem
        ? {...this.webglRenderer.curShowOnEdgeItem}
        : undefined,
      flattenLayers,
    };

    return snapshot;
  }

  restoreSnapshot(snapshot: SnapshotData) {
    if (snapshot.showOnNodeItemTypes) {
      // The setShowOnNode call below will actually update
      // curShowOnNodeItemTypes in one of the effects in renderer.ts which
      // will trigger a relayout. We don't really want to do relayout there
      // because we want to do the relayout below with the correct data from
      // the snapshot.
      //
      // To accomplish this, we manually set the value of
      // curShowOnNodeItemTypes here to match the updated value so that the
      // effect will skip when checking the equality.
      this.webglRenderer.curShowOnNodeItemTypes = {
        ...snapshot.showOnNodeItemTypes,
      };
      this.webglRenderer.appService.setShowOnNode(
        this.webglRenderer.paneId,
        this.webglRenderer.rendererId,
        this.webglRenderer.curShowOnNodeItemTypes,
      );
    }
    if (snapshot.showOnEdgeItem) {
      this.webglRenderer.curShowOnEdgeItem = {
        ...snapshot.showOnEdgeItem,
      };
      this.webglRenderer.appService.setShowOnEdge(
        this.webglRenderer.paneId,
        this.webglRenderer.rendererId,
        this.webglRenderer.curShowOnEdgeItem.type,
        this.webglRenderer.curShowOnEdgeItem.filterText,
        this.webglRenderer.curShowOnEdgeItem.outputMetadataKey,
        this.webglRenderer.curShowOnEdgeItem.inputMetadataKey,
        this.webglRenderer.curShowOnEdgeItem.sourceNodeAttrKey,
        this.webglRenderer.curShowOnEdgeItem.targetNodeAttrKey,
      );
    }

    // Switch to the stored flatten layers state in the snapshot.
    const curFlattenLayers = this.webglRenderer.appService.getFlattenLayers(
      this.webglRenderer.paneId,
    );
    const snapshotFlattenLayers = snapshot.flattenLayers === true;
    if (curFlattenLayers !== snapshotFlattenLayers) {
      this.webglRenderer.appService.processGraph(
        this.webglRenderer.paneId,
        snapshotFlattenLayers,
        snapshot,
      );
      this.webglRenderer.appService.setFlattenLayersInCurrentPane(
        snapshotFlattenLayers,
      );
    } else {
      this.webglRenderer.sendRelayoutGraphRequest(
        snapshot.selectedNodeId || '',
        snapshot.deepestExpandedGroupNodeIds || [],
        false,
        snapshot.rect,
        true,
        snapshot.showOnNodeItemTypes,
      );
    }
  }
}
