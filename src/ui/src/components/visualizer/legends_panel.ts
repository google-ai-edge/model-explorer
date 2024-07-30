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

  showSelectedNodeKey = false;
  isSelectedNodeGroup = false;
  hasArtificialLayers = false;

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
  }
}
