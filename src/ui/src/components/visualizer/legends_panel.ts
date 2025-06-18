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

import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  effect,
  Input,
  signal,
  viewChildren,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {AppService} from './app_service';

/**
 * The panel that shows the lengends.
 */
@Component({
  standalone: true,
  selector: 'legends-panel',
  imports: [CommonModule, MatIconModule],
  templateUrl: './legends_panel.ng.html',
  styleUrls: ['./legends_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LegendsPanel {
  @Input({required: true}) paneId!: string;
  readonly legendItems = viewChildren('item');

  showSelectedNodeKey = false;
  isSelectedNodeGroup = false;
  hasArtificialLayers = false;

  protected showPanel = signal<boolean>(true);
  protected showDivider = signal<boolean>(true);

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane) {
        return;
      }

      this.hasArtificialLayers = pane.hasArtificialLayers === true;
      const selectedNodeInfo = pane.selectedNodeInfo;
      if (!selectedNodeInfo) {
        this.showSelectedNodeKey = false;
        this.changeDetectorRef.markForCheck();
        return;
      }

      this.showSelectedNodeKey = selectedNodeInfo.nodeId !== '';
      this.isSelectedNodeGroup = selectedNodeInfo.isGroupNode;
      this.changeDetectorRef.markForCheck();
    });

    effect(() => {
      const legendItems = this.legendItems();
      this.showDivider.set(legendItems.length > 0);
      this.showPanel.set(legendItems.length > 0 || !this.hideShortcuts);
    });
  }

  get opLabel(): string {
    return this.appService.config()?.legendConfig?.renameOpTo ?? 'Op';
  }

  get layerLabel(): string {
    return this.appService.config()?.legendConfig?.renameLayerTo ?? 'Layer';
  }

  get inputsLabel(): string {
    return this.appService.config()?.legendConfig?.renameInputsTo ?? 'Inputs';
  }

  get outputsLabel(): string {
    return this.appService.config()?.legendConfig?.renameOutputsTo ?? 'Outputs';
  }

  get selectedItemLabel(): string {
    const config = this.appService.config();
    if (this.isSelectedNodeGroup) {
      return `Selected ${config?.legendConfig?.renameLayerTo ?? 'layer'}`;
    } else {
      return `Selected ${config?.legendConfig?.renameOpTo ?? 'op'}`;
    }
  }

  get identicalLayerLabel(): string {
    return `Identical ${this.appService.config()?.legendConfig?.renameLayerTo ?? 'layer'} (if any)`;
  }

  get hideOp(): boolean {
    return this.appService.config()?.legendConfig?.hideOp ?? false;
  }

  get hideLayer(): boolean {
    return this.appService.config()?.legendConfig?.hideLayer ?? false;
  }

  get hideArtificialLayers(): boolean {
    return (
      this.appService.config()?.legendConfig?.hideArtificialLayers ?? false
    );
  }

  get hideSelectedOp(): boolean {
    return this.appService.config()?.legendConfig?.hideSelectedOp ?? false;
  }

  get hideSelectedLayer(): boolean {
    return this.appService.config()?.legendConfig?.hideSelectedLayer ?? false;
  }

  get hideIdenticalLayers(): boolean {
    return this.appService.config()?.legendConfig?.hideIdenticalLayers ?? false;
  }

  get hideInputs(): boolean {
    return this.appService.config()?.legendConfig?.hideInputs ?? false;
  }

  get hideOutputs(): boolean {
    return this.appService.config()?.legendConfig?.hideOutputs ?? false;
  }

  get hideShortcuts(): boolean {
    return this.appService.config()?.legendConfig?.hideShortcuts ?? false;
  }
}
