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

import {animate, state, style, transition, trigger} from '@angular/animations';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  Input,
  OnInit,
  Signal,
  computed,
} from '@angular/core';
import {AppService} from './app_service';
import type {Pane} from './common/types';
import {EdgeOverlaysService} from './edge_overlays_service';
import {GraphPanel} from './graph_panel';
import {InfoPanel} from './info_panel';
import {SplitPaneService} from './split_pane_service';
import {SubgraphSelectionService} from './subgraph_selection_service';

/** A wrapper panel around the graph renderer. */
@Component({
  standalone: true,
  selector: 'split-pane',
  imports: [CommonModule, GraphPanel, InfoPanel],
  providers: [EdgeOverlaysService, SubgraphSelectionService, SplitPaneService],
  templateUrl: './split_pane.ng.html',
  styleUrls: ['./split_pane.scss'],
  animations: [
    trigger('showModelGraph', [
      state(
        'void',
        style({
          opacity: 0,
          transform: 'scale(1.03, 1.03)',
        }),
      ),
      transition(
        'void => *',
        animate(
          '150ms 100ms ease-out',
          style({opacity: 1, transform: 'scale(1, 1)'}),
        ),
      ),
    ]),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SplitPane implements OnInit {
  @Input({required: true}) pane!: Pane;

  readonly hasSelectedNode: Signal<boolean> = computed(() => {
    const selectedNode = this.appService.selectedNode();
    if (!selectedNode || selectedNode.nodeId === '') {
      return false;
    }
    if (selectedNode.paneId !== this.pane.id) {
      return false;
    }
    return true;
  });

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly edgeOverlaysService: EdgeOverlaysService,
  ) {}

  ngOnInit() {
    this.edgeOverlaysService.setPane(this.pane);

    // Load edge overlays stored in config.
    const config = this.appService.config();
    const panes = this.appService.panes();
    const isLeftPane = panes.length > 0 && panes[0].id === this.pane.id;
    const isRightPane = panes.length > 1 && panes[1].id === this.pane.id;
    if (isLeftPane && config?.edgeOverlaysDataListLeftPane) {
      for (const data of config.edgeOverlaysDataListLeftPane) {
        this.edgeOverlaysService.addEdgeOverlayData(data);
      }
    } else if (isRightPane && config?.edgeOverlaysDataListRightPane) {
      for (const data of config.edgeOverlaysDataListRightPane) {
        this.edgeOverlaysService.addEdgeOverlayData(data);
      }
    }

    // Load edge overlays stored in graph.
    const graphCollections = this.appService.curGraphCollections();
    for (const graphCollection of graphCollections) {
      for (const graph of graphCollection.graphs) {
        if (isLeftPane && graph.tasksData?.edgeOverlaysDataListLeftPane) {
          for (const data of graph.tasksData.edgeOverlaysDataListLeftPane) {
            this.edgeOverlaysService.addEdgeOverlayData(data);
          }
        } else if (
          isRightPane &&
          graph.tasksData?.edgeOverlaysDataListRightPane
        ) {
          for (const data of graph.tasksData.edgeOverlaysDataListRightPane) {
            this.edgeOverlaysService.addEdgeOverlayData(data);
          }
        }
      }
    }
  }

  refresh() {
    this.changeDetectorRef.markForCheck();
  }

  get disableAnimation(): boolean {
    return this.appService.testMode;
  }

  get hideInfoPanel(): boolean {
    return this.appService.config()?.hideInfoPanel === true;
  }

  get showSidePanelOnNodeSelection(): boolean {
    return this.appService.config()?.showSidePanelOnNodeSelection === true;
  }
}
