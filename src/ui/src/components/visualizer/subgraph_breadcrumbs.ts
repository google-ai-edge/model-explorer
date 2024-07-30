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
  Input,
  effect,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {AppService} from './app_service';
import {SubgraphBreadcrumbItem} from './common/types';

/** Breadcrumbs for navigating through subgraphs. */
@Component({
  standalone: true,
  selector: 'subgraph-breadcrumbs',
  imports: [CommonModule, MatIconModule, MatTooltipModule],
  templateUrl: './subgraph_breadcrumbs.ng.html',
  styleUrls: ['./subgraph_breadcrumbs.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SubgraphBreadcrumbs {
  @Input({required: true}) paneId!: string;

  curSubgraphBreadcrumbs: SubgraphBreadcrumbItem[] = [];

  private savedSubgraphBreadcrumbs?: SubgraphBreadcrumbItem[];

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      const curSubgraphBreadcrumbs = pane?.subgraphBreadcrumbs;
      if (curSubgraphBreadcrumbs === this.savedSubgraphBreadcrumbs) {
        return;
      }
      this.savedSubgraphBreadcrumbs = curSubgraphBreadcrumbs;
      this.curSubgraphBreadcrumbs = curSubgraphBreadcrumbs || [];
      this.changeDetectorRef.markForCheck();
    });
  }

  handleClickItem(i: number) {
    if (i === this.curSubgraphBreadcrumbs.length - 1) {
      return;
    }

    // Update current subgraph breadcrumb.
    this.appService.setCurrentSubgraphBreadcrumb(this.paneId, i);

    // Restore snapshot for the clicked item.
    const item = this.curSubgraphBreadcrumbs[i];
    const snapshot = item.snapshot;
    if (snapshot) {
      if (
        item.graphId ===
        this.appService.getPaneById(this.paneId)?.modelGraph?.id
      ) {
        this.appService.curSnapshotToRestore.next({
          rendererId: this.paneId,
          snapshot,
        });
      } else {
        const graph = this.appService.getGraphById(item.graphId);
        if (graph) {
          this.appService.selectGraphInCurrentPane(
            graph,
            snapshot.flattenLayers,
            snapshot,
          );
        }
      }
    }
  }
}
