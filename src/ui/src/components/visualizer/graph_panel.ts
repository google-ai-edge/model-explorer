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
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  QueryList,
  SimpleChanges,
  ViewChildren,
  effect,
} from '@angular/core';

import {AppService} from './app_service';
import {type ModelGraph} from './common/model_graph';
import {PopupPanelData, RendererOwner} from './common/types';
import {LegendsPanel} from './legends_panel';
import {PopupPanel} from './popup_panel';
import {RendererWrapper} from './renderer_wrapper';
import {SelectionPanel} from './selection_panel';
import {SubgraphSelectionService} from './subgraph_selection_service';

/** A wrapper panel around the graph renderer. */
@Component({
  standalone: true,
  selector: 'graph-panel',
  imports: [
    CommonModule,
    LegendsPanel,
    PopupPanel,
    RendererWrapper,
    SelectionPanel,
  ],
  templateUrl: './graph_panel.ng.html',
  styleUrls: ['./graph_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphPanel implements OnInit, OnChanges, AfterViewInit, OnDestroy {
  @Input({required: true}) modelGraph!: ModelGraph;
  @Input({required: true}) rendererId!: string;
  @Input({required: true}) paneId!: string;
  @ViewChildren('popupPanel')
  popupPanelComponents = new QueryList<PopupPanel>();

  readonly popupPanels: PopupPanelData[] = [];

  showRenderer = true;

  private readonly mouseDownListener = (e: MouseEvent) => {
    if ((e.target as HTMLElement).closest('popup-panel') != null) {
      return;
    }
    this.appService.curSelectedRenderer.set({
      id: this.rendererId,
      ownerType: RendererOwner.GRAPH_PANEL,
    });
  };

  private resizeObserver?: ResizeObserver;

  constructor(
    private readonly root: ElementRef<HTMLElement>,
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly subgraphSelectionService: SubgraphSelectionService,
  ) {
    // Move selected panel to the top.
    effect(() => {
      const selectedRenderer = this.appService.curSelectedRenderer();
      const prevTopPopupPanel =
        this.root.nativeElement.querySelector('popup-panel.top');
      if (prevTopPopupPanel) {
        prevTopPopupPanel.classList.remove('top');
      }
      const popupPanelComponent = this.popupPanelComponents.find(
        (comp) => comp.rendererWrapper?.rendererId === selectedRenderer?.id,
      );
      if (popupPanelComponent) {
        popupPanelComponent.root.nativeElement.classList.add('top');
      }
    });
  }

  ngOnInit() {
    this.root.nativeElement.addEventListener(
      'mousedown',
      this.mouseDownListener,
      true /* capture phase */,
    );
    this.subgraphSelectionService.paneId = this.paneId;
  }

  ngOnChanges(changes: SimpleChanges) {
    // Make sure to re-initialize the renderer-wrapper from the scratch when
    // modelGraph changes.
    if (changes['modelGraph'] && this.modelGraph) {
      this.subgraphSelectionService.clearSelection();
      this.showRenderer = false;
      this.changeDetectorRef.detectChanges();
      this.showRenderer = true;
      this.changeDetectorRef.detectChanges();
    }
  }

  ngAfterViewInit() {
    const root = this.root.nativeElement;
    this.resizeObserver = new ResizeObserver((entries) => {
      this.handleResize();
    });
    this.resizeObserver.observe(root);
  }

  ngOnDestroy() {
    this.root.nativeElement.removeEventListener(
      'mousedown',
      this.mouseDownListener,
      true /* capture phase */,
    );
    if (this.resizeObserver) {
      this.resizeObserver.unobserve(this.root.nativeElement);
    }
  }

  trackByPopupPanelId(index: number, panelData: PopupPanelData): string {
    return panelData.id;
  }

  handleOpenOnPopupClicked(data: PopupPanelData) {
    this.popupPanels.push(data);
  }

  handleClickClosePanel(id: string) {
    const index = this.popupPanels.findIndex((data) => data.id === id);
    if (index >= 0) {
      this.popupPanels.splice(index, 1);
    }
  }

  get showLegends(): boolean {
    return !this.appService.config()?.hideLegends;
  }

  private handleResize() {
    const root = this.root.nativeElement;

    // Move popups that go out of bound.
    for (const popup of this.popupPanelComponents) {
      const popupEl = popup.root.nativeElement;
      if (!popupEl) {
        continue;
      }
      const popupWidth = Number(popupEl.style.width.replace('px', ''));
      const popupHeight = Number(popupEl.style.height.replace('px', ''));
      const right = Number(popupEl.style.left.replace('px', '')) + popupWidth;
      const bottom = Number(popupEl.style.top.replace('px', '')) + popupHeight;
      if (right > root.offsetWidth) {
        popupEl.style.left = `${root.offsetWidth - popupWidth}px`;
      }
      if (bottom > root.offsetHeight) {
        popupEl.style.top = `${root.offsetHeight - popupHeight}px`;
      }
    }
  }
}
