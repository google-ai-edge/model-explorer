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
} from '@angular/core';
import {combineLatest, fromEvent} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A component to handle drag events. */
@Component({
  standalone: true,
  selector: 'drag-area',
  imports: [CommonModule],
  templateUrl: './drag_area.ng.html',
  styleUrls: ['./drag_area.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DragArea {
  @Input({required: true}) borderColor!: string;
  @Input({required: true}) bgColor!: string;

  startX = -1;
  startY = -1;
  endX = -1;
  endY = -1;

  private readonly isMac =
    typeof navigator !== 'undefined' && /Macintosh/.test(navigator.userAgent);

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  start(
    event: MouseEvent,
    onCompleteFn: (
      isClick: boolean,
      startX: number,
      startY: number,
      endX: number,
      endY: number,
    ) => void,
  ) {
    // Uncomment to show scene position on click for debugging purpose.
    //
    // const scenePos = this.convertScreenPosToScene(event.offsetX, event.offsetY);
    // console.log('screen pos', scenePos);

    event.preventDefault();
    event.stopPropagation();

    this.startX = event.offsetX;
    this.startY = event.offsetY;
    this.endX = this.startX;
    this.endY = this.startY;

    const move = fromEvent<MouseEvent>(document, 'mousemove');
    const up = fromEvent<MouseEvent>(window, 'mouseup');

    // Hit position.
    let deltaX = 0;
    let deltaY = 0;

    combineLatest([move])
      .pipe(takeUntil(up))
      .subscribe({
        next: ([moveEvent]) => {
          // Calculate delta.
          deltaX = moveEvent.offsetX - this.startX;
          deltaY = moveEvent.offsetY - this.startY;
          this.endX = this.startX + deltaX;
          this.endY = this.startY + deltaY;
          this.changeDetectorRef.detectChanges();
        },
        complete: () => {
          const isClick = Math.abs(deltaX) < 5 && Math.abs(deltaY) < 5;
          onCompleteFn(isClick, this.startX, this.startY, this.endX, this.endY);
          this.startX = -1;
          this.startY = -1;
          this.endX = -1;
          this.endY = -1;
          this.changeDetectorRef.detectChanges();
        },
      });
  }

  get top(): number {
    return Math.min(this.startY, this.endY);
  }

  get left(): number {
    return Math.min(this.startX, this.endX);
  }

  get width(): number {
    return Math.abs(this.endX - this.startX);
  }

  get height(): number {
    return Math.abs(this.endY - this.startY);
  }
}
