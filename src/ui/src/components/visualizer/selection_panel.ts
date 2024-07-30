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
  Component,
  computed,
  Input,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {safeAnchorEl} from 'safevalues/dom';

import {AppService} from './app_service';
import {exportToResource} from './common/utils';
import {SubgraphSelectionService} from './subgraph_selection_service';

/** A component to handle drag events. */
@Component({
  standalone: true,
  selector: 'selection-panel',
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './selection_panel.ng.html',
  styleUrls: ['./selection_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SelectionPanel {
  @Input({required: true}) paneId!: string;

  readonly hasSelectedNodes;

  readonly selectedNodeCountLabel = computed(() => {
    const count = this.subgraphSelectionService.selectedNodeCount();
    return `${count} node${count === 1 ? '' : 's'}`;
  });

  constructor(
    private readonly appService: AppService,
    private readonly subgraphSelectionService: SubgraphSelectionService,
  ) {
    this.hasSelectedNodes = this.subgraphSelectionService.hasSelectedNodes;
  }

  handleClickClear() {
    this.subgraphSelectionService.clearSelection();
  }

  handleClickDownloadSubgraph() {
    const subgraph = this.subgraphSelectionService.getSelectedSubgraph();
    if (!subgraph) {
      return;
    }

    // Download it.
    const link = document.createElement('a');
    link.download = `${subgraph.collectionLabel}_subgraph.json`;
    const dataUrl = `data:text/json;charset=utf-8, ${encodeURIComponent(
      JSON.stringify([subgraph], null, 2),
    )}`;
    safeAnchorEl.setHref(link, dataUrl);
    link.click();
  }

  handleClickExportToResource() {
    const subgraph = this.subgraphSelectionService.getSelectedSubgraph();
    if (!subgraph) {
      return;
    }

    exportToResource(`${subgraph.collectionLabel ?? ''}_subgraph.json`, [
      subgraph,
    ]);
  }

  get enableExportToResource(): boolean {
    return this.appService.config()?.enableExportToResource === true;
  }
}
