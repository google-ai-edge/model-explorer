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

import {OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  inject,
  Input,
  Signal,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {Bubble} from '../bubble/bubble';
import {BubbleClick} from '../bubble/bubble_click';
import {AppService} from './app_service';
import {ProcessedEdgeOverlay} from './common/edge_overlays';
import {EdgeOverlaysService} from './edge_overlays_service';
import {LocalStorageService} from './local_storage_service';

interface OverlaysSet {
  id: string;
  name: string;
  overlays: OverlayItem[];
}

interface OverlayItem {
  id: string;
  name: string;
  selected: boolean;
  processedOverlay: ProcessedEdgeOverlay;
}

/** The edge overlays dropdown panel with the trigger button. */
@Component({
  standalone: true,
  selector: 'edge-overlays-dropdown',
  imports: [
    Bubble,
    BubbleClick,
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  templateUrl: './edge_overlays_dropdown.ng.html',
  styleUrls: ['./edge_overlays_dropdown.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class EdgeOverlaysDropdown {
  @Input({required: true}) paneId!: string;
  @Input({required: true}) rendererId!: string;
  @ViewChild(BubbleClick) popup!: BubbleClick;

  private readonly appService = inject(AppService);
  private readonly localStorageService = inject(LocalStorageService);
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly edgeOverlaysService = inject(EdgeOverlaysService);
  private readonly snackBar = inject(MatSnackBar);

  readonly overlaysSets: Signal<OverlaysSet[]> = computed(() => {
    const overlays = this.edgeOverlaysService.filteredLoadedEdgeOverlays();
    return overlays.map((overlay) => ({
      id: overlay.id,
      name: overlay.name,
      overlays: overlay.processedOverlays.map((overlay) => ({
        id: overlay.id,
        name: overlay.name,
        selected: this.edgeOverlaysService
          .selectedOverlayIds()
          .includes(overlay.id),
        processedOverlay: overlay,
      })),
    }));
  });

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly edgeOverlaysPopupSize: OverlaySizeConfig = {
    minWidth: 280,
    minHeight: 0,
  };

  readonly remoteSourceLoading = this.edgeOverlaysService.remoteSourceLoading;
  opened = false;

  constructor() {
  }

  handleClickOnEdgeOverlaysButton() {
    if (this.opened) {
      this.popup.closeDialog();
    }
  }

  handleClickUpload(input: HTMLInputElement) {
    const files = input.files;
    if (!files || files.length === 0) {
      return;
    }
    const file = files[0];
    const fileReader = new FileReader();
    fileReader.onload = (event) => {
      const error = this.edgeOverlaysService.addEdgeOverlayDataFromJsonData(
        event.target?.result as string,
      );
      if (error) {
        this.showError(error);
      }
    };
    fileReader.readAsText(file);
    input.value = '';
  }

  handleDeleteOverlaySet(overlaySet: OverlaysSet) {
    this.edgeOverlaysService.deleteOverlayData(overlaySet.id);
  }

  toggleOverlaySelection(overlay: OverlayItem) {
    this.edgeOverlaysService.toggleOverlaySelection(overlay.id);
  }

  handleClickViewOverlay(overlay: OverlayItem) {
    // Get the first node of the overlay.
    const edges = overlay.processedOverlay.edges;
    if (edges.length === 0) {
      return;
    }
    const firstNodeId = edges[0].sourceNodeId;

    // Reveal it.
    this.appService.setNodeToReveal(this.paneId, firstNodeId);
  }

  toggleShowEdgesConnectedToSelectedNode(overlay: OverlayItem) {
    this.edgeOverlaysService.toggleShowEdgesConnectedToSelectedNodeOnly(
      overlay.id,
    );
  }

  handleMaxHopsChanged(overlay: OverlayItem, value: number) {
    this.edgeOverlaysService.setVisibleEdgeHops(overlay.id, value);
  }

  private showError(message: string) {
    console.error(message);
    this.snackBar.open(message, 'Dismiss', {
      duration: 5000,
    });
  }
}
