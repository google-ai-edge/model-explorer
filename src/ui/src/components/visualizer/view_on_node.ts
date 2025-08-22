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
  DestroyRef,
  effect,
  Input,
  Signal,
  ViewChild,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {Bubble} from '../bubble/bubble';
import {BubbleClick} from '../bubble/bubble_click';
import {AppService} from './app_service';
import {
  LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM,
  LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES,
  NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX,
} from './common/consts';
import {
  CommandType,
  SetViewOnEdgeCommand,
  ShowOnEdgeItemData,
  ShowOnEdgeItemType,
  ShowOnNodeItemData,
  ShowOnNodeItemType,
  ViewOnEdgeMode,
} from './common/types';
import {getRunName} from './common/utils';
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
  ShowOnNodeItemType.LAYER_NODE_ATTRS,
];

const ALL_SHOW_ON_EDGE_ITEM_TYPES: ShowOnEdgeItemType[] = [
  ShowOnEdgeItemType.OFF,
  ShowOnEdgeItemType.TENSOR_SHAPE,
  ShowOnEdgeItemType.SOURCE_NODE_ATTR,
  ShowOnEdgeItemType.TARGET_NODE_ATTR,
  ShowOnEdgeItemType.OUTPUT_METADATA,
  ShowOnEdgeItemType.INPUT_METADATA,
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
    return runs.map((run) => getRunName(run, modelGraph));
  });
  private savedNodeDataProviderRunNames: string[] = [];

  private savedShowOnEdgeItems?: Record<string, ShowOnEdgeItemData>;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly viewPopupSize: OverlaySizeConfig = {
    minWidth: 280,
    minHeight: 0,
    maxHeight: 800,
  };

  showOnNodeItems: ShowOnNodeItem[] = [];
  showOnEdgeItems: ShowOnEdgeItem[] = [];
  curOpAttrsFilterText = '';
  curGroupAttrsFilterText = '';
  curSourceNodeAttrKeyText = '';
  curTargetNodeAttrKeyText = '';
  curOutputMetadataKeyText = '';
  curInputMetadataKeyText = '';
  opened = false;

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly destroyRef: DestroyRef,
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
      const config = this.appService.config();
      const items: ShowOnNodeItem[] = [];
      for (const type of ALL_SHOW_ON_NODE_ITEM_TYPES) {
        // Hide items if specified in the config.
        if (
          (config?.viewOnNodeConfig?.hideOpNodeId &&
            type === ShowOnNodeItemType.OP_NODE_ID) ||
          (config?.viewOnNodeConfig?.hideOpNodeAttributes &&
            type === ShowOnNodeItemType.OP_ATTRS) ||
          (config?.viewOnNodeConfig?.hideOpNodeInputs &&
            type === ShowOnNodeItemType.OP_INPUTS) ||
          (config?.viewOnNodeConfig?.hideOpNodeOutputs &&
            type === ShowOnNodeItemType.OP_OUTPUTS) ||
          (config?.viewOnNodeConfig?.hideLayerNodeChildrenCount &&
            type === ShowOnNodeItemType.LAYER_NODE_CHILDREN_COUNT) ||
          (config?.viewOnNodeConfig?.hideLayerNodeDescendantsCount &&
            type === ShowOnNodeItemType.LAYER_NODE_DESCENDANTS_COUNT) ||
          (config?.viewOnNodeConfig?.hideLayerNodeAttributes &&
            type === ShowOnNodeItemType.LAYER_NODE_ATTRS)
        ) {
          continue;
        }
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
          this.curOpAttrsFilterText = item.filterRegex;
        } else if (type === ShowOnNodeItemType.LAYER_NODE_ATTRS) {
          item.filterRegex =
            (curShowOnNodeItemTypes[this.rendererId] || {})[type]
              ?.filterRegex || '';
          this.curGroupAttrsFilterText = item.filterRegex;
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
      const curShowOnEdgeItems = pane?.showOnEdgeItems || {};

      if (curShowOnEdgeItems === this.savedShowOnEdgeItems) {
        return;
      }
      this.savedShowOnEdgeItems = curShowOnEdgeItems;

      const curShowOnEdgeItem = curShowOnEdgeItems[this.rendererId];
      this.curInputMetadataKeyText = curShowOnEdgeItem?.inputMetadataKey ?? '';
      this.curOutputMetadataKeyText =
        curShowOnEdgeItem?.outputMetadataKey ?? '';
      this.curSourceNodeAttrKeyText =
        curShowOnEdgeItem?.sourceNodeAttrKey ?? '';
      this.curTargetNodeAttrKeyText =
        curShowOnEdgeItem?.targetNodeAttrKey ?? '';

      const items: ShowOnEdgeItem[] = [];
      if (!this.appService.config()?.viewOnNodeConfig?.hideViewOnEdgesSection) {
        for (const type of ALL_SHOW_ON_EDGE_ITEM_TYPES) {
          const item: ShowOnEdgeItem = {
            type,
            selected: type === curShowOnEdgeItems[this.rendererId]?.type,
          };
          // Select "off" by default if the saved types are empty.
          if (
            type === ShowOnEdgeItemType.OFF &&
            curShowOnEdgeItems[this.rendererId] == null
          ) {
            item.selected = true;
          }
          items.push(item);
        }
      }

      this.showOnEdgeItems = items;
      this.changeDetectorRef.markForCheck();
    });

    // React to commands.
    this.appService.command
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((command) => {
        // Ignore commands not for this pane.
        if (
          command.paneIndex !== this.appService.getPaneIndexById(this.paneId)
        ) {
          return;
        }

        // Handle commands.
        switch (command.type) {
          case CommandType.SET_VIEW_ON_EDGE:
            this.handleSetViewOnEdgeCommand(command);
            break;
          default:
            break;
        }
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

  handleSetShowOnEdge(checked: boolean, item: ShowOnEdgeItem) {
    this.appService.setShowOnEdge(
      this.paneId,
      this.rendererId,
      item.type,
      this.getEdgeItemMetadataKeyText(item),
      this.curOutputMetadataKeyText,
      this.curInputMetadataKeyText,
      this.curSourceNodeAttrKeyText,
      this.curTargetNodeAttrKeyText,
    );

    // Save to local storage.
    this.saveShowOnEdgeItemsToLocalStorage();
  }

  handleAttrsFilterChanged(item: ShowOnNodeItem) {
    this.appService.setShowOnNodeFilter(
      this.paneId,
      this.rendererId,
      item.type,
      this.getAttrsFilterText(item),
    );

    // Save to local storage.
    this.saveShowOnNodeItemsToLocalStorage();
  }

  handleEdgeItemFilterChanged(item: ShowOnEdgeItem) {
    this.appService.setShowOnEdge(
      this.paneId,
      this.rendererId,
      item.type,
      this.getEdgeItemMetadataKeyText(item),
      this.curOutputMetadataKeyText,
      this.curInputMetadataKeyText,
      this.curSourceNodeAttrKeyText,
      this.curTargetNodeAttrKeyText,
    );

    // Save to local storage.
    this.saveShowOnEdgeItemsToLocalStorage();
  }

  getShowOnNodeItemLabel(item: ShowOnNodeItem): string {
    switch (item.type) {
      case ShowOnNodeItemType.OP_NODE_ID:
        return (
          this.appService.config()?.viewOnNodeConfig?.renameOpNodeIdTo ??
          item.type
        );
      case ShowOnNodeItemType.OP_ATTRS:
        return (
          this.appService.config()?.viewOnNodeConfig
            ?.renameOpNodeAttributesTo ?? item.type
        );
      case ShowOnNodeItemType.OP_INPUTS:
        return (
          this.appService.config()?.viewOnNodeConfig?.renameOpNodeInputsTo ??
          item.type
        );
      case ShowOnNodeItemType.OP_OUTPUTS:
        return (
          this.appService.config()?.viewOnNodeConfig?.renameOpNodeOutputsTo ??
          item.type
        );
      default:
        return item.type;
    }
  }

  getAttrsFilterText(item: ShowOnNodeItem): string {
    switch (item.type) {
      case ShowOnNodeItemType.OP_ATTRS:
        return this.curOpAttrsFilterText;
      case ShowOnNodeItemType.LAYER_NODE_ATTRS:
        return this.curGroupAttrsFilterText;
      default:
        return '';
    }
  }

  setAttrsFilterText(item: ShowOnNodeItem, text: string) {
    switch (item.type) {
      case ShowOnNodeItemType.OP_ATTRS:
        this.curOpAttrsFilterText = text;
        break;
      case ShowOnNodeItemType.LAYER_NODE_ATTRS:
        this.curGroupAttrsFilterText = text;
        break;
      default:
        break;
    }
  }

  getEdgeItemMetadataKeyText(item: ShowOnEdgeItem): string {
    switch (item.type) {
      case ShowOnEdgeItemType.OUTPUT_METADATA:
        return this.curOutputMetadataKeyText;
      case ShowOnEdgeItemType.INPUT_METADATA:
        return this.curInputMetadataKeyText;
      case ShowOnEdgeItemType.SOURCE_NODE_ATTR:
        return this.curSourceNodeAttrKeyText;
      case ShowOnEdgeItemType.TARGET_NODE_ATTR:
        return this.curTargetNodeAttrKeyText;
      default:
        return '';
    }
  }

  setEdgeItemMetadataKeyText(item: ShowOnEdgeItem, text: string) {
    switch (item.type) {
      case ShowOnEdgeItemType.OUTPUT_METADATA:
        this.curOutputMetadataKeyText = text;
        break;
      case ShowOnEdgeItemType.INPUT_METADATA:
        this.curInputMetadataKeyText = text;
        break;
      case ShowOnEdgeItemType.SOURCE_NODE_ATTR:
        this.curSourceNodeAttrKeyText = text;
        break;
      case ShowOnEdgeItemType.TARGET_NODE_ATTR:
        this.curTargetNodeAttrKeyText = text;
        break;
      default:
        break;
    }
  }

  getIsAttrs(item: ShowOnNodeItem): boolean {
    return (
      item.type === ShowOnNodeItemType.OP_ATTRS ||
      item.type === ShowOnNodeItemType.LAYER_NODE_ATTRS
    );
  }

  getEdgeItemHaveFilter(item: ShowOnEdgeItem): boolean {
    return (
      item.type === ShowOnEdgeItemType.OUTPUT_METADATA ||
      item.type === ShowOnEdgeItemType.INPUT_METADATA ||
      item.type === ShowOnEdgeItemType.SOURCE_NODE_ATTR ||
      item.type === ShowOnEdgeItemType.TARGET_NODE_ATTR
    );
  }

  getEdgeItemPlaceholder(item: ShowOnEdgeItem): string {
    switch (item.type) {
      case ShowOnEdgeItemType.OUTPUT_METADATA:
      case ShowOnEdgeItemType.INPUT_METADATA:
        return 'Metadata key';
      case ShowOnEdgeItemType.SOURCE_NODE_ATTR:
      case ShowOnEdgeItemType.TARGET_NODE_ATTR:
        return 'Attribute key';
      default:
        return '';
    }
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
      const showOnEdgeItem = this.appService.getShowOnEdgeItem(
        this.paneId,
        this.rendererId,
      );
      this.localStorageService.setItem(
        LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM,
        JSON.stringify(showOnEdgeItem),
      );
    }
  }

  private handleSetViewOnEdgeCommand(command: SetViewOnEdgeCommand) {
    // Unselect all items first.
    for (const item of this.showOnEdgeItems) {
      item.selected = false;
    }

    // Select the item corresponding to the given mode.
    let item: ShowOnEdgeItem | undefined = undefined;
    switch (command.mode) {
      case ViewOnEdgeMode.OFF:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.OFF,
        );
        break;
      case ViewOnEdgeMode.TENSOR_SHAPE:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.TENSOR_SHAPE,
        );
        break;
      case ViewOnEdgeMode.SOURCE_NODE_ATTR:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.SOURCE_NODE_ATTR,
        );
        this.curSourceNodeAttrKeyText = command.value ?? '';
        break;
      case ViewOnEdgeMode.TARGET_NODE_ATTR:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.TARGET_NODE_ATTR,
        );
        this.curTargetNodeAttrKeyText = command.value ?? '';
        break;
      case ViewOnEdgeMode.OUTPUT_METADATA:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.OUTPUT_METADATA,
        );
        this.curOutputMetadataKeyText = command.value ?? '';
        break;
      case ViewOnEdgeMode.INPUT_METADATA:
        item = this.showOnEdgeItems.find(
          (item) => item.type === ShowOnEdgeItemType.INPUT_METADATA,
        );
        this.curInputMetadataKeyText = command.value ?? '';
        break;
      default:
        break;
    }
    if (item) {
      item.selected = true;
      this.handleSetShowOnEdge(true, item);
    }
  }
}
