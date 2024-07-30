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

import {ConnectedPosition, OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  effect,
  ElementRef,
  HostListener,
  Input,
  QueryList,
  ViewChildren,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {Bubble} from '../bubble/bubble';

import {AppService} from './app_service';
import {ModelGraph} from './common/model_graph';
import {SnapshotData} from './common/types';
import {inInputElement} from './common/utils';

const NUMBERS = new Set<string>(['1', '2', '3', '4', '5', '6', '7', '8', '9']);

/**
 * A component that manages the snapshots of the graph. A snapshot records the
 * current state (e.g. expanded layers, selected node, etc).
 */
@Component({
  standalone: true,
  selector: 'snapshot-manager',
  imports: [Bubble, CommonModule, MatIconModule, MatTooltipModule],
  templateUrl: './snapshot_manager.ng.html',
  styleUrls: ['./snapshot_manager.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SnapshotManager {
  @Input({required: true}) paneId!: string;
  @Input({required: true}) rendererId!: string;
  @ViewChildren('dialog') snapshotPopups = new QueryList<Bubble>();
  @ViewChildren('trigger')
  snapshotTriggers = new QueryList<ElementRef<HTMLElement>>();

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxWidth: 340,
  };

  readonly snapshotPopupSize: OverlaySizeConfig = {
    maxWidth: 1000,
    maxHeight: 1000,
  };

  readonly snapshotPopupPosition: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'bottom',
      overlayX: 'start',
      overlayY: 'top',
      offsetY: 12,
    },
  ];

  curSnapshots: SnapshotData[] = [];

  private curModelGraph?: ModelGraph;

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      this.curModelGraph = pane?.modelGraph;
      if (pane?.modelGraph != null) {
        this.curSnapshots = (pane?.snapshots || {})[pane.modelGraph.id] || [];
        this.changeDetectorRef.markForCheck();
      }
    });
  }

  @HostListener('document:keypress', ['$event'])
  handleKeyboardEvent(event: KeyboardEvent) {
    if (
      NUMBERS.has(event.key) &&
      this.appService.curSelectedRenderer()?.id === this.rendererId &&
      !inInputElement()
    ) {
      const snapshotIndex = Number(event.key) - 1;
      if (snapshotIndex <= this.curSnapshots.length - 1) {
        this.handleClickSnapshot(snapshotIndex);
        const snapshotTrigger =
          this.snapshotTriggers.get(snapshotIndex)?.nativeElement;
        if (snapshotTrigger) {
          snapshotTrigger.classList.add('clicked');
          setTimeout(() => {
            snapshotTrigger.classList.remove('clicked');
          }, 50);
        }
      }
    }
  }

  handleClickAddSnapshot() {
    if (this.disableAddSnapshotButton) {
      return;
    }

    this.appService.addSnapshotClicked.next({rendererId: this.rendererId});
  }

  handleClickSnapshot(index: number) {
    const snapshot = this.curSnapshots[index];
    this.snapshotPopups.get(index)?.closeDialog();
    this.appService.curSnapshotToRestore.next({
      snapshot,
      rendererId: this.rendererId,
    });
  }

  handleClickDeleteSnapshot(index: number) {
    if (this.curModelGraph) {
      this.appService.deleteSnapshot(index, this.curModelGraph.id, this.paneId);
    }
  }

  handleSnapshotPopupOpened(snapshot: SnapshotData) {
    const canvas = document.querySelector(
      '.model-explorer-snapshot-popup canvas',
    ) as HTMLCanvasElement;
    const snapshotCanvasWidth = this.getSnapshotCanvasWidth(snapshot);
    const snapshotCanvasHeight = this.getSnapshotCanvasHeight(snapshot);
    canvas.width = snapshotCanvasWidth;
    canvas.height = snapshotCanvasHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(
      snapshot.imageBitmap,
      0,
      0,
      snapshotCanvasWidth,
      snapshotCanvasHeight,
    );
    canvas.classList.add('show');
  }

  getSnapshotCanvasWidth(snapshot: SnapshotData, forCss = false): number {
    return snapshot.imageBitmap.width / (forCss ? window.devicePixelRatio : 1);
  }

  getSnapshotCanvasHeight(snapshot: SnapshotData, forCss = false): number {
    return snapshot.imageBitmap.height / (forCss ? window.devicePixelRatio : 1);
  }

  trackBySnapshotId(index: number, value: SnapshotData): string {
    return value.id;
  }

  get addSnapshotTooltip(): string {
    return this.disableAddSnapshotButton
      ? 'Maximum bookmark count reached'
      : 'Bookmark the current graph states to restore later';
  }

  get disableAddSnapshotButton(): boolean {
    return this.curSnapshots.length >= 9;
  }

  get hasSnapshots(): boolean {
    return this.curSnapshots.length > 0;
  }
}
