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
  effect,
  Input,
  Signal,
  ViewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {Bubble} from '../bubble/bubble';
import {BubbleClick} from '../bubble/bubble_click';

import {AppService} from './app_service';
import {
  LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM_TYPES,
  LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES,
  NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
} from './common/consts';
import {
  ShowOnEdgeItemData,
  ShowOnEdgeItemType,
  ShowOnNodeItemData,
  ShowOnNodeItemType,
} from './common/types';
import {LocalStorageService} from './local_storage_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';

interface ShowOnNodeItem {
  type: string;
  selected: boolean;
  filterRegex?: string;
}

interface ShowOnEdgeItem {
  type: string;
  selected: boolean;
}

const ALL_SHOW_ON_NODE_ITEM_TYPES: ShowOnNodeItemType[] = [
  ShowOnNodeItemType.OP_NODE_ID,
  ShowOnNodeItemType.OP_ATTRS,
  ShowOnNodeItemType.OP_INPUTS,
  ShowOnNodeItemType.OP_OUTPUTS,
  ShowOnNodeItemType.LAYER_NODE_CHILDREN_COUNT,
  ShowOnNodeItemType.LAYER_NODE_DESCENDANTS_COUNT,
];

const ALL_SHOW_ON_EDGE_ITEM_TYPES: ShowOnEdgeItemType[] = [
  ShowOnEdgeItemType.TENSOR_SHAPE,
];

/** The view-on-node and its popup in the toolbar. */
@Component({
  standalone: true,
  selector: 'view-on-node',
  imports: [Bubble, BubbleClick, CommonModule, MatIconModule, MatTooltipModule],
  templateUrl: './view_on_node.ng.html',
  styleUrls: ['./view_on_node.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ViewOnNode {
  @Input({required: true}) rendererId!: string;
  @Input({required: true}) paneId!: string;
  @Input() inPopup = false;
  @ViewChild(BubbleClick) popup!: BubbleClick;

  private savedShowOnNodeItemTypes?: Record<
    string,
    Record<string, ShowOnNodeItemData>
  >;
  private readonly nodeDataProviderRunNames: Signal<string[]> = computed(() => {
    const modelGraph = this.appService.getModelGraphFromPane(this.paneId);
    const runs = modelGraph
      ? Object.values(
          this.nodeDataProviderExtensionService.getRunsForModelGraph(
            modelGraph,
          ),
        )
      : [];
    return runs.map((run) => run.runName);
  });
  private savedNodeDataProviderRunNames: string[] = [];

  private savedShowOnEdgeItemTypes?: Record<
    string,
    Record<string, ShowOnEdgeItemData>
  >;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly viewPopupSize: OverlaySizeConfig = {
    minWidth: 280,
    minHeight: 0,
  };

  showOnNodeItems: ShowOnNodeItem[] = [];
  showOnEdgeItems: ShowOnEdgeItem[] = [];
  curAttrsFilterText = '';
  opened = false;

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly localStorageService: LocalStorageService,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
  ) {
    // Handle changes on show on node items.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      const curShowOnNodeItemTypes = pane?.showOnNodeItemTypes || {};
      const curNodeDataProviderRunNames = this.nodeDataProviderRunNames();

      if (
        curShowOnNodeItemTypes === this.savedShowOnNodeItemTypes &&
        JSON.stringify(curNodeDataProviderRunNames) ===
          JSON.stringify(this.savedNodeDataProviderRunNames)
      ) {
        return;
      }
      this.savedNodeDataProviderRunNames = curNodeDataProviderRunNames;
      this.savedShowOnNodeItemTypes = curShowOnNodeItemTypes;

      // Node info fields and attrs.
      const items: ShowOnNodeItem[] = [];
      for (const type of ALL_SHOW_ON_NODE_ITEM_TYPES) {
        const item: ShowOnNodeItem = {
          type,
          selected: (curShowOnNodeItemTypes[this.rendererId] || {})[type]
            ?.selected,
        };
        items.push(item);
        if (type === ShowOnNodeItemType.OP_ATTRS) {
          item.filterRegex =
            (curShowOnNodeItemTypes[this.rendererId] || {})[type]
              ?.filterRegex || '';
          this.curAttrsFilterText = item.filterRegex;
        }
      }

      // Node data provider runs.
      for (const runName of this.savedNodeDataProviderRunNames) {
        const type = `${NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX}${runName}`;
        items.push({
          type,
          selected: (curShowOnNodeItemTypes[this.rendererId] || {})[type]
            ?.selected,
        });
      }
      this.showOnNodeItems = items;
      this.changeDetectorRef.markForCheck();
    });

    // Handle changes on show on edge items.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      const curShowOnEdgeItemTypes = pane?.showOnEdgeItemTypes || {};

      if (curShowOnEdgeItemTypes === this.savedShowOnEdgeItemTypes) {
        return;
      }
      this.savedShowOnEdgeItemTypes = curShowOnEdgeItemTypes;

      const items: ShowOnEdgeItem[] = [];
      for (const type of ALL_SHOW_ON_EDGE_ITEM_TYPES) {
        const item: ShowOnEdgeItem = {
          type,
          selected: (curShowOnEdgeItemTypes[this.rendererId] || {})[type]
            ?.selected,
        };
        items.push(item);
      }

      this.showOnEdgeItems = items;
      this.changeDetectorRef.markForCheck();
    });
  }

  handleClickOnViewOnNode(event: MouseEvent) {
    if (this.opened) {
      this.popup.closeDialog();
    }
  }

  handleToggleShowOnNode(item: ShowOnNodeItem) {
    this.appService.toggleShowOnNode(this.paneId, this.rendererId, item.type);

    // Save to local storage.
    this.saveShowOnNodeItemsToLocalStorage();
  }

  handleToggleShowOnEdge(item: ShowOnEdgeItem) {
    this.appService.toggleShowOnEdge(this.paneId, this.rendererId, item.type);

    // Save to local storage.
    this.saveShowOnEdgeItemsToLocalStorage();
  }

  handleAttrsFilterChanged(item: ShowOnNodeItem) {
    this.appService.setShowOnNodeFilter(
      this.paneId,
      this.rendererId,
      item.type,
      this.curAttrsFilterText,
    );

    // Save to local storage.
    this.saveShowOnNodeItemsToLocalStorage();
  }

  getIsAttrs(item: ShowOnNodeItem): boolean {
    return item.type === ShowOnNodeItemType.OP_ATTRS;
  }

  private saveShowOnNodeItemsToLocalStorage() {
    if (!this.inPopup && !this.appService.testMode) {
      const showOnNodeItems = this.appService.getShowOnNodeItemTypes(
        this.paneId,
        this.rendererId,
      );
      // Don't save selections for node data providers.
      const keysToDelete = Object.keys(showOnNodeItems).filter((type: string) =>
        type.startsWith(NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX),
      );
      for (const key of keysToDelete) {
        delete showOnNodeItems[key];
      }
      this.localStorageService.setItem(
        LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES,
        JSON.stringify(showOnNodeItems),
      );
    }
  }

  private saveShowOnEdgeItemsToLocalStorage() {
    if (!this.inPopup && !this.appService.testMode) {
      const showOnEdgeItems = this.appService.getShowOnEdgeItemTypes(
        this.paneId,
        this.rendererId,
      );
      this.localStorageService.setItem(
        LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM_TYPES,
        JSON.stringify(showOnEdgeItems),
      );
    }
  }
}
