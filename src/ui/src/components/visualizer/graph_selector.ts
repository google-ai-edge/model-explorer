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

import {Overlay, OverlayConfig, OverlayRef} from '@angular/cdk/overlay';
import {ComponentPortal} from '@angular/cdk/portal';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  Signal,
  ViewChild,
  ViewContainerRef,
  computed,
  effect,
  signal,
} from '@angular/core';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatSelect, MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {setAnchorHref} from 'safevalues/dom';
import {AppService} from './app_service';
import {Graph, GraphCollection} from './common/input_graph';
import {exportToResource} from './common/utils';
import {GraphSelectorPanel} from './graph_selector_panel';

/** A graph collection in the dropdown menu. */
export interface GraphCollectionItem {
  label: string;
  graphs: GraphItem[];
  collection: GraphCollection;
}

/** A graph in a collection in the dropdown menu. */
export interface GraphItem {
  id: string;
  graph: Graph;
  level: number;
  nonHiddenNodeCount: number;
  width: number;
}

const CANVAS = new OffscreenCanvas(500, 300);
const LABEL_WIDTHS: {[label: string]: number} = {};

/**
 * The graph selector component.
 *
 * It allows users to select one of the input graphs from a dropdown list.
 */
@Component({
  standalone: true,
  selector: 'graph-selector',
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatIconModule,
    MatSelectModule,
    MatTooltipModule,
    ReactiveFormsModule,
  ],
  templateUrl: './graph_selector.ng.html',
  styleUrls: ['./graph_selector.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphSelector {
  @ViewChild(MatSelect) mySelector!: MatSelect;
  @ViewChild('input') filterInput!: ElementRef<HTMLInputElement>;

  selectedGraph = new FormControl<Graph | undefined>(undefined);
  selectedGraphNodeCount = 0;
  selectedGraphCollectionLabel = '';
  selectedCollection?: GraphCollection;
  maxGraphItemIdWidth = 0;

  graphCollectionItems: Signal<GraphCollectionItem[]> = computed(() => {
    const config = this.appService.config();
    if (!config) {
      return [];
    }

    const collections = this.appService.curGraphCollections();
    this.nodeLabelsToHide = new Set<string>(
      (config.nodeLabelsToHide || []).map((label) => label.toLowerCase()),
    );

    // Calculate count for non-hidden nodes in each graph.
    const graphCollectionItems: GraphCollectionItem[] = [];
    const filterText = this.curFilterText().toLowerCase();
    for (const collection of collections) {
      const collectionItem: GraphCollectionItem = {
        label: collection.label,
        collection,
        graphs: [],
      };
      for (const {graph, level} of collection.graphsWithLevel ?? []) {
        if (filterText !== '' && !graph.id.toLowerCase().includes(filterText)) {
          continue;
        }
        const nonHiddenNodeCount = graph.nodes.filter(
          (node) => !this.nodeLabelsToHide.has(node.label.toLowerCase()),
        ).length;
        const width =
          this.getLabelWidth(` ${graph.id}    ${nonHiddenNodeCount} nodes`) +
          30;
        collectionItem.graphs.push({
          id: graph.id,
          graph,
          level,
          nonHiddenNodeCount,
          width,
        });
        this.maxGraphItemIdWidth = Math.max(
          width + 30,
          this.maxGraphItemIdWidth,
        );
      }
      if (collectionItem.graphs.length > 0) {
        graphCollectionItems.push(collectionItem);
        const collectionLabelWidth =
          this.getLabelWidth(collection.label, 12, true) + 30;
        this.maxGraphItemIdWidth = Math.max(
          collectionLabelWidth,
          this.maxGraphItemIdWidth,
        );
      }
    }
    return graphCollectionItems;
  });

  graphsCount: Signal<number> = computed(() => {
    let count = 0;
    const collections = this.appService.curGraphCollections();
    for (const collection of collections) {
      count += collection.graphs.length;
    }
    return count;
  });

  private nodeLabelsToHide = new Set<string>();
  private readonly curFilterText = signal<string>('');
  private portal: ComponentPortal<GraphSelectorPanel> | null = null;

  private readonly selectedGraphId = computed(() => {
    const pane = this.appService.getSelectedPane();
    if (!pane || !pane.modelGraph) {
      return '';
    }
    return pane.modelGraph.id;
  });

  constructor(
    private readonly appService: AppService,
    private readonly overlay: Overlay,
    private readonly viewContainerRef: ViewContainerRef,
  ) {
    // Update selected graph when the data source in app service is updated.
    effect(() => {
      const selectedGraphId = this.selectedGraphId();
      if (!selectedGraphId) {
        return;
      }
      this.updateSelectedGraphInfo(selectedGraphId);
    });
  }

  handleFilterTextChanged(value: string) {
    this.curFilterText.set(value);
  }

  handleClickOpenGraphDropdown(selector: HTMLElement) {
    // this.mySelector.open();

    const overlayRef = this.createOverlay(selector);
    const ref = overlayRef.attach(this.portal!);
    ref.instance.graphCollectionItems = this.graphCollectionItems();
    ref.instance.onClose.subscribe(() => {
      overlayRef.dispose();
    });
  }

  handleGraphSelectorOpenedChanged(opened: boolean) {
    // Clear filter text when the selector is closed.
    if (!opened) {
      this.filterInput.nativeElement.value = '';
      this.curFilterText.set('');
    }
  }

  handleGraphSelected() {
    if (this.selectedGraph.value) {
      this.updateSelectedGraphInfo(this.selectedGraph.value.id);
      this.appService.selectGraphInCurrentPane(this.selectedGraph.value);
      this.appService.curInitialUiState.set(undefined);
      this.appService.selectNode(this.appService.selectedPaneId(), undefined);
      this.appService.curToLocateNodeInfo.set(undefined);
      this.appService.setFlattenLayersInCurrentPane(false);
    }
  }

  handleClickOpenInSplitPane(event: MouseEvent, graphItem: GraphItem) {
    event.stopPropagation();

    this.mySelector.close();
    this.appService.openGraphInSplitPane(graphItem.graph);
  }

  handleClickDownloadGraphJson() {
    if (this.selectedCollection == null) {
      return;
    }

    // Download the currently selected graph collection.
    const link = document.createElement('a');
    link.download = `${this.selectedGraphCollectionLabel}.json`;
    const dataUrl = `data:text/json;charset=utf-8, ${encodeURIComponent(
      JSON.stringify(this.selectedCollection, null, 2),
    )}`;
    setAnchorHref(link, dataUrl);
    link.click();
  }

  handleClickExportGraphJsonToResource() {
    if (!this.selectedCollection == null) {
      return;
    }

    exportToResource(
      `${this.selectedGraphCollectionLabel}.json`,
      this.selectedCollection,
    );
  }

  getGraphLabel(graph: Graph): string {
    return `${graph.id} (${graph.nodes.length} nodes)`;
  }

  get graphSelectorDropdownWidth(): number {
    return this.maxGraphItemIdWidth;
  }

  get showOpenInSplitPane(): boolean {
    return this.appService.panes().length === 1;
  }

  get enableExportToResource(): boolean {
    return this.appService.config()?.enableExportToResource === true;
  }

  private getLabelWidth(label: string, fontSize = 12, bold = false): number {
    // Check cache first.
    const key = label;
    let labelWidth = LABEL_WIDTHS[key];
    if (labelWidth == null) {
      // On cache miss, render the text to a offscreen canvas to get its width.
      const context = CANVAS.getContext(
        '2d',
      )! as {} as CanvasRenderingContext2D;
      context.font = `${fontSize}px "Google Sans Text", Arial, Helvetica, sans-serif`;
      if (bold) {
        context.font = `bold ${context.font}`;
      }
      const metrics = context.measureText(label);
      const width = metrics.width;
      LABEL_WIDTHS[key] = width;
      labelWidth = width;
    }
    return labelWidth;
  }

  private updateSelectedGraphInfo(selectedGraphId: string) {
    let foundGraph = false;
    for (const collectionItem of this.graphCollectionItems()) {
      for (const graphItem of collectionItem.graphs) {
        if (graphItem.graph.id === selectedGraphId) {
          this.selectedGraphNodeCount = graphItem.nonHiddenNodeCount;
          this.selectedGraphCollectionLabel = collectionItem.label;
          this.selectedCollection = collectionItem.collection;
          this.selectedGraph.setValue(graphItem.graph);
          foundGraph = true;
          break;
        }
      }
      if (foundGraph) {
        break;
      }
    }
  }

  private createOverlay(ele: HTMLElement): OverlayRef {
    const config = new OverlayConfig({
      positionStrategy: this.overlay
        .position()
        .flexibleConnectedTo(ele)
        .withPositions([
          {
            originX: 'end',
            originY: 'bottom',
            overlayX: 'end',
            overlayY: 'top',
          },
        ]),
      maxHeight: 'calc(100% - 70px)',
      hasBackdrop: true,
      backdropClass: 'cdk-overlay-transparent-backdrop',
      scrollStrategy: this.overlay.scrollStrategies.reposition(),
      panelClass: 'graph-selector-panel',
    });
    const overlayRef = this.overlay.create(config);
    this.portal = new ComponentPortal(
      GraphSelectorPanel,
      this.viewContainerRef,
    );
    overlayRef.backdropClick().subscribe(() => {
      overlayRef.dispose();
    });
    return overlayRef;
  }
}
