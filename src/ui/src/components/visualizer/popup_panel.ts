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

import {
  ChangeDetectionStrategy,
  Component,
  computed,
  ElementRef,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
  ViewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {AppService} from './app_service';
import {type GroupNode, type ModelGraph} from './common/model_graph';
import {RendererOwner, type Point} from './common/types';
import {PopupPanelTransformer} from './popup_panel_transformer';
import {RendererWrapper} from './renderer_wrapper';

const DEFAULT_WIDTH = 400;
const DEFAULT_HEIGHT = 400;
const MINIMIZED_HEIGHT = 26;

/** A popup panel that shows a user-selected group node. */
@Component({
  standalone: true,
  selector: 'popup-panel',
  imports: [MatIconModule, RendererWrapper],
  templateUrl: './popup_panel.ng.html',
  styleUrls: ['./popup_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PopupPanel implements OnInit, OnDestroy {
  @Input({required: true}) id!: string;
  @Input({required: true}) paneId!: string;
  @Input({required: true}) groupNode!: GroupNode;
  @Input({required: true}) initialPosition!: Point;
  @Input({required: true}) curModelGraph!: ModelGraph;
  @Output() readonly closeClicked = new EventEmitter<string>();

  @ViewChild('rendererWrapper') rendererWrapper?: RendererWrapper;

  selected = computed(
    () =>
      this.appService.curSelectedRenderer()?.id ===
        this.rendererWrapper?.rendererId &&
      this.appService.selectedPaneId() === this.paneId,
  );
  minimized = false;

  private savedHeight = '';

  private readonly mouseDownListener = (e: MouseEvent) => {
    if (this.rendererWrapper) {
      this.appService.curSelectedRenderer.set({
        id: this.rendererWrapper.rendererId,
        ownerType: RendererOwner.POPUP,
      });

      // Set the current active selected node in the popup as the selected node
      // for the pane.
      this.appService.selectNode(
        this.paneId,
        this.rendererWrapper.getActiveSelectedNodeInfo(),
      );

      // Set the selected pane.
      this.appService.selectPane(this.paneId);
    }
  };

  constructor(
    private readonly appService: AppService,
    readonly root: ElementRef<HTMLElement>,
  ) {}

  ngOnInit() {
    this.root.nativeElement.addEventListener(
      'mousedown',
      this.mouseDownListener,
      true /* capture phase */,
    );

    const root = this.root.nativeElement;
    root.style.left = `${this.initialPosition.x}px`;
    root.style.top = `${this.initialPosition.y}px`;
    root.style.width = `${DEFAULT_WIDTH}px`;
    root.style.height = `${DEFAULT_HEIGHT}px`;
  }

  ngOnDestroy() {
    this.root.nativeElement.removeEventListener(
      'mousedown',
      this.mouseDownListener,
      true /* capture phase */,
    );
  }

  toggleMinimize() {
    this.minimized = !this.minimized;

    if (this.minimized) {
      this.savedHeight = this.root.nativeElement.style.height;
      this.root.nativeElement.style.height = `${MINIMIZED_HEIGHT}px`;
    } else {
      this.root.nativeElement.style.height = this.savedHeight;
    }
  }

  handleMouseDown(event: MouseEvent) {
    event.stopPropagation();
    if (event.button === 2) {
      return;
    }

    new PopupPanelTransformer(this.root.nativeElement).drag(event);
  }

  handleMouseUp(event: MouseEvent) {}

  get title(): string {
    return this.groupNode.label;
  }
}
