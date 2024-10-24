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

import {animate, style, transition, trigger} from '@angular/animations';
import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  DestroyRef,
  effect,
  ElementRef,
  QueryList,
  Signal,
  ViewChild,
  ViewChildren,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';
import {combineLatest, fromEvent} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {Bubble} from '../bubble/bubble';
import {AppService} from './app_service';
import {Pane} from './common/types';
import {
  ALL_PROCESSING_LABELS,
  UpdateProcessingProgressRequest,
  WorkerEvent,
  WorkerEventType,
} from './common/worker_events';
import {SplitPane} from './split_pane';
import {SyncNavigationButton} from './sync_navigation_button';
import {SyncNavigationService} from './sync_navigation_service';
import {WorkerService} from './worker_service';

interface ProcessingTask {
  label: string;
  processing: boolean;
  error?: string;
}

/** A containe for split panes. */
@Component({
  standalone: true,
  selector: 'split-panes-container',
  imports: [
    Bubble,
    CommonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    SplitPane,
    SyncNavigationButton,
  ],
  templateUrl: './split_panes_container.ng.html',
  styleUrls: ['./split_panes_container.scss'],
  animations: [
    trigger('hideProcessingPanel', [
      transition(
        '* => void',
        animate(
          '150ms 100ms ease-out',
          style({opacity: 0, transform: 'scale(0.95, 0.95)'}),
        ),
      ),
    ]),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SplitPanesContainer implements AfterViewInit {
  @ViewChild('panesContainer') panesContainer!: ElementRef<HTMLElement>;
  @ViewChild('noMappedNodeMessage')
  noMappedNodeMessage?: ElementRef<HTMLElement>;
  @ViewChildren('splitPane') splitPanes = new QueryList<SplitPane>();

  readonly processingTasks: Record<string, ProcessingTask[]> = {};
  readonly allPanesLoaded: Signal<boolean>;

  resizingSplitPane = false;
  curLeftWidthFraction = 1;
  panes;

  curUpdateProcessingProgressReq?: UpdateProcessingProgressRequest;

  private hideNoMappedNodeMessageTimeoutId = -1;

  constructor(
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly appService: AppService,
    private readonly destroyRef: DestroyRef,
    private readonly syncNavigationService: SyncNavigationService,
    private readonly workerService: WorkerService,
  ) {
    this.panes = this.appService.panes;
    this.allPanesLoaded = computed(() =>
      this.panes().every((pane) => pane.modelGraph != null),
    );

    effect(() => {
      const panes = this.panes();
      if (panes.length >= 1) {
        this.curLeftWidthFraction = panes[0].widthFraction;
      }
      for (const pane of panes) {
        if (!pane.modelGraph) {
          this.processingTasks[pane.id] = ALL_PROCESSING_LABELS.map(
            (label) => ({label, processing: true}),
          );
        }
      }
      this.changeDetectorRef.detectChanges();

      for (let i = 0; i < this.splitPanes.length; i++) {
        const splitPane = this.splitPanes.get(i);
        splitPane?.refresh();
      }
    });

    this.workerService.worker.addEventListener('message', (event) => {
      const workerEvent = event.data as WorkerEvent;
      switch (workerEvent.eventType) {
        case WorkerEventType.UPDATE_PROCESSING_PROGRESS:
          this.handleUpdateProcessingProgressRequest(event.data);
          break;
        default:
          break;
      }
    });

    this.syncNavigationService.showNoMappedNodeMessageTrigger$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        if (data === undefined) {
          this.hideNoMappedNodeMessage();
        } else {
          this.showNoMappedNodeMessage();
        }
      });
  }

  ngAfterViewInit() {
    // Handle mouse click to select pane.
    this.panesContainer.nativeElement.addEventListener(
      'mousedown',
      (e: MouseEvent) => {
        // Find the cloeset pane-container and extract its data-pane-id
        // attribute value as the pane id to select.
        const paneEle = (e.target as HTMLElement).closest(
          '.pane-container',
        ) as HTMLElement;
        // This is the case when mousedown on the resizer.
        if (!paneEle) {
          return;
        }
        const paneId = paneEle.dataset['paneId'] || '';
        this.appService.selectPane(paneId);
      },
      true /* capture phase */,
    );
  }

  handleClickSwapPane() {
    this.appService.swapPane();
  }

  handleClickClosePane(pane: Pane) {
    this.appService.closePane(pane.id);
  }

  getPaneTitle(pane: Pane): string {
    const modelGraph = pane.modelGraph;
    if (!modelGraph) {
      return '-';
    }
    return `${modelGraph.collectionLabel} | ${modelGraph.id}`;
  }

  handleMouseDownResizer(event: MouseEvent, panesContainer: HTMLElement) {
    event.preventDefault();

    document.body.style.cursor = 'ew-resize';

    const move = fromEvent<MouseEvent>(document, 'mousemove');
    const up = fromEvent<MouseEvent>(window, 'mouseup');
    const panes = this.appService.panes();
    const savedLeftFraction = panes[0].widthFraction;
    const containerWidth = panesContainer.offsetWidth;
    const savedLeft = containerWidth * savedLeftFraction;

    // Hit position.
    const hitPtX = event.clientX;
    this.resizingSplitPane = true;
    this.changeDetectorRef.markForCheck();

    combineLatest([move])
      .pipe(takeUntil(up))
      .subscribe({
        next: ([moveEvent]) => {
          // Calculate delta.
          const delta = moveEvent.clientX - hitPtX;
          const curLeft = Math.min(
            containerWidth - 200,
            Math.max(200, savedLeft + delta),
          );
          this.curLeftWidthFraction = curLeft / containerWidth;
          this.changeDetectorRef.markForCheck();
        },
        complete: () => {
          document.body.style.cursor = 'default';
          this.resizingSplitPane = false;
          this.appService.setPaneWidthFraction(this.curLeftWidthFraction);
          this.changeDetectorRef.markForCheck();
        },
      });
  }

  isPaneSelected(pane: Pane): boolean {
    return pane.id === this.appService.selectedPaneId();
  }

  getPaneWidthPct(paneIndex: number): number {
    return (
      (paneIndex === 0
        ? this.curLeftWidthFraction
        : 1 - this.curLeftWidthFraction) * 100
    );
  }

  trackByPaneId(index: number, value: Pane): string {
    return value.id;
  }

  getProcessingTasksForPane(paneId: string): ProcessingTask[] {
    return this.processingTasks[paneId] || [];
  }

  getProgressPct(paneId: string): number {
    const curTasks = this.processingTasks[paneId];
    if (!curTasks) {
      return 0;
    }
    const numDoneProcessing = curTasks.filter(
      (task) => !task.processing,
    ).length;
    return (numDoneProcessing / curTasks.length) * 100;
  }

  getShowLoading(pane: Pane): boolean {
    return (
      pane.modelGraph == null ||
      this.getProcessingTasksForPane(pane.id).some(
        (task) => task.error != null && task.error !== '',
      )
    );
  }

  getProcessingError(pane: Pane): string {
    return (
      this.getProcessingTasksForPane(pane.id).find(
        (task) => task.error != null && task.error !== '',
      )?.error ?? ''
    );
  }

  get hasSplitPane(): boolean {
    return this.appService.panes().length > 1;
  }

  get resizerLeft(): string {
    return `calc(${this.curLeftWidthFraction * 100}% - 5px)`;
  }

  get disableAnimation(): boolean {
    return this.appService.testMode;
  }

  private handleUpdateProcessingProgressRequest(
    req: UpdateProcessingProgressRequest,
  ) {
    const paneId = req.paneId;
    const pane = this.appService.getPaneById(paneId);
    if (!pane) {
      return;
    }
    const curTasks = this.processingTasks[paneId];
    const task = curTasks.find((task) => task.label === req.label);
    if (task != null) {
      task.processing = false;
      task.error = req.error;
      this.changeDetectorRef.detectChanges();
    }
  }

  private hideNoMappedNodeMessage() {
    const ele = this.noMappedNodeMessage?.nativeElement;
    if (!ele) {
      return;
    }

    if (this.hideNoMappedNodeMessageTimeoutId >= 0) {
      clearTimeout(this.hideNoMappedNodeMessageTimeoutId);
      this.hideNoMappedNodeMessageTimeoutId = -1;
    }

    ele.classList.remove('show');
  }

  private showNoMappedNodeMessage() {
    const ele = this.noMappedNodeMessage?.nativeElement;
    if (!ele) {
      return;
    }

    if (this.hideNoMappedNodeMessageTimeoutId >= 0) {
      clearTimeout(this.hideNoMappedNodeMessageTimeoutId);
      this.hideNoMappedNodeMessageTimeoutId = -1;
    }

    // Hide after 3 seconds.
    ele.classList.add('show');
    this.hideNoMappedNodeMessageTimeoutId = setTimeout(() => {
      ele.classList.remove('show');
    }, 3000);
  }
}
