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

import {combineLatest, fromEvent} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const MIN_WIDTH = 48;
const MIN_HEIGHT = 48;

/** Handles moving and resizing a node runner panel. */
export class PopupPanelTransformer {
  constructor(private readonly rootEle: HTMLElement) {}

  drag(event: MouseEvent) {
    const mousePositionX = event.clientX;
    const mousePositionY = event.clientY;
    const mouseOffsetX = event.offsetX;

    const savedTop = Number(this.rootEle.style.top.replace('px', ''));
    const savedLeft = Number(this.rootEle.style.left.replace('px', ''));
    const savedWidth = Number(this.rootEle.style.width.replace('px', ''));
    const savedHeight = Number(this.rootEle.style.height.replace('px', ''));

    const move = fromEvent<MouseEvent>(document, 'mousemove');
    const up = fromEvent<MouseEvent>(window, 'mouseup');

    const dataPosition = (event.target as HTMLElement).dataset['position'];
    const containerHeight = this.rootEle.parentElement!.offsetHeight;
    const containerWidth = this.rootEle.parentElement!.offsetWidth;

    combineLatest([move])
      .pipe(takeUntil(up))
      .subscribe({
        next: ([moveEvent]) => {
          // This is to prevent selecting text while dragging.
          moveEvent.preventDefault();

          // Calculate delta.
          const deltaX = moveEvent.clientX - mousePositionX;
          const deltaY = moveEvent.clientY - mousePositionY;

          if (deltaX === 0 && deltaY === 0) {
            return;
          }

          // Move.
          //
          // Don't allow move the panel completely out of the bound of its
          // container.
          if (dataPosition == null) {
            this.rootEle.style.top = `${Math.min(
              containerHeight - 28,
              Math.max(savedTop + deltaY, 0),
            )}px`;
            this.rootEle.style.left = `${Math.min(
              containerWidth - mouseOffsetX,
              Math.max(-mouseOffsetX, savedLeft + deltaX),
            )}px`;
          }
          // Resize.
          else {
            if (dataPosition.includes('right')) {
              this.rootEle.style.width = `${Math.max(
                MIN_WIDTH,
                savedWidth + deltaX,
              )}px`;
            }
            if (dataPosition.includes('bottom')) {
              this.rootEle.style.height = `${Math.max(
                MIN_HEIGHT,
                savedHeight + deltaY,
              )}px`;
            }
            if (dataPosition.includes('left')) {
              const newWidth = Math.max(MIN_WIDTH, savedWidth - deltaX);
              this.rootEle.style.width = `${newWidth}px`;
              this.rootEle.style.left = `${
                savedLeft + (savedWidth - newWidth)
              }px`;
            }
            if (dataPosition.includes('top')) {
              const newHeight = Math.max(MIN_HEIGHT, savedHeight - deltaY);
              this.rootEle.style.height = `${newHeight}px`;
              this.rootEle.style.top = `${
                savedTop + (savedHeight - newHeight)
              }px`;
            }
          }
        },
        complete: () => {},
      });
  }
}
