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
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {Bubble} from '../bubble/bubble';
import {AppService} from './app_service';
import {type ModelGraph} from './common/model_graph';
import {
  PopupPanelData,
  SelectedNodeInfo,
  SubgraphBreadcrumbItem,
} from './common/types';
import {isGroupNode} from './common/utils';
import {EdgeOverlaysDropdown} from './edge_overlays_dropdown';
import {SearchBar} from './search_bar';
import {SnapshotManager} from './snapshot_manager';
import {SubgraphBreadcrumbs} from './subgraph_breadcrumbs';
import {ViewOnNode} from './view_on_node';
import {WebglRenderer} from './webgl_renderer';

/** A wrapper panel around various renderers. */
@Component({
  standalone: true,
  selector: 'renderer-wrapper',
  imports: [
    Bubble,
    CommonModule,
    EdgeOverlaysDropdown,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
    ReactiveFormsModule,
    SearchBar,
    SnapshotManager,
    SubgraphBreadcrumbs,
    ViewOnNode,
    WebglRenderer,
  ],
  templateUrl: './renderer_wrapper.ng.html',
  styleUrls: ['./renderer_wrapper.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RendererWrapper {
  @Input({required: true}) modelGraph!: ModelGraph;
  @Input({required: true}) rendererId!: string;
  @Input({required: true}) paneId!: string;
  /** The id of the root node to render from. Undefined means all nodes. */
  @Input() rootNodeId?: string;
  @Input() inPopup = false;
  @Output() readonly openInPopupClicked = new EventEmitter<PopupPanelData>();

  @ViewChild('webglRenderer') webglRenderer?: WebglRenderer;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxWidth: 340,
  };

  flattenAllLayers = computed(() =>
    this.appService.getFlattenLayers(this.paneId),
  );
  disableDownloadPngHelpPopup = false;
  transparentPngBackground = new FormControl<boolean>(false);

  private curSubgraphBreadcrumbs: SubgraphBreadcrumbItem[] = [];

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      this.curSubgraphBreadcrumbs = pane?.subgraphBreadcrumbs || [];
      this.changeDetectorRef.markForCheck();
    });
  }

  handleOpenOnPopupClicked(data: PopupPanelData) {
    this.openInPopupClicked.emit(data);
  }

  handleClickZoomFitIcon() {
    this.appService.spaceKeyToZoomFitClicked.next({});
  }

  handleClickExpandAllLayers() {
    this.appService.expandOrCollapseAllGraphLayersClicked.next({
      expandOrCollapse: true,
      rendererId: this.rendererId,
    });
  }

  handleClickCollapseAllLayers() {
    this.appService.expandOrCollapseAllGraphLayersClicked.next({
      expandOrCollapse: false,
      rendererId: this.rendererId,
    });
  }

  handleClickFlattenAllLayers() {
    // Deselect group node if selected.
    const selectedNodeId = this.appService.getPaneById(this.paneId)
      ?.selectedNodeInfo?.nodeId;
    if (
      selectedNodeId != null &&
      isGroupNode(this.modelGraph.nodesById[selectedNodeId])
    ) {
      this.appService.selectNode(this.paneId, undefined);
    }

    // Toggle and re-process the graph.
    this.appService.toggleFlattenLayers(this.paneId);
    this.appService.processGraph(
      this.paneId,
      this.appService.getFlattenLayers(this.paneId),
    );

    // Clear init graph state.
    this.appService.curInitialUiState.set(undefined);
  }

  handleClickDownloadAsPng(fullGraph: boolean) {
    this.appService.downloadAsPngClicked.next({
      rendererId: this.rendererId,
      fullGraph,
      transparentBackground: this.transparentPngBackground.value === true,
    });
  }

  handleClickTrace() {
    this.webglRenderer?.toggleIoTrace();
  }

  handleClickToggleTransparentPngBackground(event: MouseEvent) {
    event.stopPropagation();

    this.transparentPngBackground.setValue(
      !this.transparentPngBackground.value,
    );
  }

  getActiveSelectedNodeInfo(): SelectedNodeInfo | undefined {
    return this.webglRenderer?.getActiveSelectedNodeInfo();
  }

  /** Whether to show the search bar. */
  get showSearchBar(): boolean {
    return !this.inPopup;
  }

  get showExpandCollapseAllLayers(): boolean {
    return (
      !this.inPopup &&
      this.appService.config()?.toolbarConfig?.hideExpandCollapseAllLayers !==
        true
    );
  }

  get showFlattenLayers(): boolean {
    return (
      !this.inPopup &&
      this.appService.config()?.toolbarConfig?.hideFlattenAllLayers !== true
    );
  }

  get showDownloadPng(): boolean {
    return !this.inPopup;
  }

  get showSnapshotManager(): boolean {
    return !this.inPopup;
  }

  get showSubgraphBreadcrumbs(): boolean {
    return !this.inPopup && this.curSubgraphBreadcrumbs.length > 1;
  }

  get showEdgeOverlaysDropdown(): boolean {
    return (
      !this.inPopup &&
      this.appService.config()?.toolbarConfig?.hideCustomEdgeOverlays !== true
    );
  }

  get disableExpandCollapseAllButton(): boolean {
    return this.appService.getFlattenLayers(this.paneId);
  }

  get tracing(): boolean {
    return this.webglRenderer?.tracing === true;
  }

  get showToolBar(): boolean {
    return !this.appService.config()?.hideToolBar;
  }

  get isTestMode(): boolean {
    return this.appService.testMode;
  }
}
