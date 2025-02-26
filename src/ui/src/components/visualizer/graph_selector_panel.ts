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
  EventEmitter,
  Input,
  Output,
  computed,
} from '@angular/core';
import {ReactiveFormsModule} from '@angular/forms';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {AppService} from './app_service';
import {GraphCollectionItem, GraphItem} from './graph_selector';
import {MENU_ANIMATIONS} from './ui_utils';

const DEFAULT_PADDING_LEFT = 24;

/**
 * The panel to show for selecting a graph when clicking the graph selector.
 */
@Component({
  standalone: true,
  selector: 'graph-selector-panel',
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatIconModule,
    MatSelectModule,
    MatTooltipModule,
    ReactiveFormsModule,
  ],
  templateUrl: './graph_selector_panel.ng.html',
  styleUrls: ['./graph_selector_panel.scss'],
  animations: MENU_ANIMATIONS,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphSelectorPanel {
  @Input({required: true}) graphCollectionItems: GraphCollectionItem[] = [];
  @Output() readonly onClose = new EventEmitter<{}>();

  hasFilteredOutGraphs = false;

  readonly selectedGraphId = computed(() => {
    const pane = this.appService.getSelectedPane();
    if (!pane || !pane.modelGraph) {
      return '';
    }
    return pane.modelGraph.id;
  });

  private curFilterText = '';

  constructor(private readonly appService: AppService) {}

  getGraphNonHiddenNodeCountLabel(count: number): string {
    return `${count} node${count === 1 ? '' : 's'}`;
  }

  handleSelectGraph(graphItem: GraphItem) {
    this.onClose.next({});
    this.resetFilter();

    this.appService.selectGraphInCurrentPane(graphItem.graph);
    this.appService.curInitialUiState.set(undefined);
    this.appService.selectNode(this.appService.selectedPaneId(), undefined);
    this.appService.curToLocateNodeInfo.set(undefined);
    this.appService.setFlattenLayersInCurrentPane(false);
  }

  handleFilterTextChanged(value: string) {
    this.curFilterText = value.toLowerCase();
  }

  handleClickOpenInSplitPane(event: MouseEvent, graphItem: GraphItem) {
    event.stopPropagation();

    this.onClose.next({});
    this.resetFilter();
    this.appService.openGraphInSplitPane(graphItem.graph);
  }

  showIndentSymbol(graphItem: GraphItem): boolean {
    return !this.hasFilteredOutGraphs && (graphItem.level ?? 0) > 0;
  }

  getGraphItemPaddingLeft(graphItem: GraphItem): number {
    // Don't show tree indentation in filter mode.
    if (this.hasFilteredOutGraphs) {
      return DEFAULT_PADDING_LEFT;
    }
    return DEFAULT_PADDING_LEFT + (graphItem.level ?? 0) * 12;
  }

  trackByCollection(
    index: number,
    graphCollectionItem: GraphCollectionItem,
  ): string {
    return `${index}`;
  }

  trackByGraph(index: number, graphItem: GraphItem): string {
    return `${graphItem.graph.collectionLabel}___${graphItem.graph.id}`;
  }

  get curGraphCollectionItems(): GraphCollectionItem[] {
    const graphCollectionItems: GraphCollectionItem[] = [];
    this.hasFilteredOutGraphs = false;
    for (const {label, collection, graphs} of this.graphCollectionItems) {
      const collectionItem: GraphCollectionItem = {
        label,
        collection,
        graphs: [],
      };
      for (const graph of graphs) {
        if (
          this.curFilterText !== '' &&
          !graph.id.toLowerCase().includes(this.curFilterText)
        ) {
          this.hasFilteredOutGraphs = true;
          continue;
        }
        collectionItem.graphs.push(graph);
      }
      if (collectionItem.graphs.length > 0) {
        graphCollectionItems.push(collectionItem);
      }
    }
    return graphCollectionItems;
  }

  get showOpenInSplitPane(): boolean {
    return this.appService.panes().length === 1;
  }

  private resetFilter() {
    this.curFilterText = '';
    this.hasFilteredOutGraphs = false;
  }
}
