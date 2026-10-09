/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
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
  DestroyRef,
  Directive,
  ElementRef,
  inject,
  input,
  output,
} from '@angular/core';
import {
  clampSize,
  draggedSize,
  steppedSize,
  type ResizeOrientation,
} from './resize_math';

/**
 * Draggable panel edge: pointer capture, keyboard steps (arrows, Shift for fine steps,
 * Home/End) and the separator ARIA state. The host owns the size signal and clamps
 * nothing itself; bounds come in as inputs so viewport-dependent limits stay reactive.
 */
@Directive({
  selector: '[resizeHandle]',
  host: {
    role: 'separator',
    tabindex: '0',
    '[attr.aria-orientation]': 'orientation()',
    '[attr.aria-valuenow]': 'size()',
    '[attr.aria-valuemin]': 'min()',
    '[attr.aria-valuemax]': 'max()',
    '(pointerdown)': 'begin($event)',
    '(keydown)': 'key($event)',
    '(dblclick)': 'reset()',
  },
})
export class ResizeHandle {
  readonly size = input.required<number>();
  readonly min = input.required<number>();
  readonly max = input.required<number>();
  readonly orientation = input<ResizeOrientation>('vertical');
  /** Right- or bottom-anchored panels grow when the pointer moves toward the page start. */
  readonly reverse = input(false);
  readonly step = input(20);
  readonly fineStep = input(5);
  /** Double-click restores this size when set. */
  readonly resetSize = input<number | null>(null);
  readonly sizeChange = output<number>();
  readonly resizingChange = output<boolean>();
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  private cleanup: (() => void) | null = null;

  constructor() {
    inject(DestroyRef).onDestroy(() => this.cleanup?.());
  }

  private bounds() {
    return {min: this.min(), max: this.max()};
  }

  begin(event: PointerEvent) {
    if (event.button !== 0) return;
    event.preventDefault();
    this.cleanup?.();
    const target = this.element.nativeElement;
    const pointerId = event.pointerId;
    const horizontal = this.orientation() === 'horizontal';
    const startPosition = horizontal ? event.clientY : event.clientX;
    // Start from the clamped size so the first movement never jumps.
    const startSize = clampSize(this.size(), this.bounds());
    const move = (next: PointerEvent) => {
      if (next.pointerId !== pointerId) return;
      const position = horizontal ? next.clientY : next.clientX;
      this.sizeChange.emit(
        draggedSize(
          startSize,
          startPosition,
          position,
          this.bounds(),
          this.reverse(),
        ),
      );
    };
    const finish = (next: PointerEvent) => {
      if (next.pointerId === pointerId) this.cleanup?.();
    };
    this.cleanup = () => {
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', finish);
      target.removeEventListener('pointercancel', finish);
      target.removeEventListener('lostpointercapture', finish);
      if (target.hasPointerCapture(pointerId))
        target.releasePointerCapture(pointerId);
      this.cleanup = null;
      this.resizingChange.emit(false);
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', finish);
    target.addEventListener('pointercancel', finish);
    target.addEventListener('lostpointercapture', finish);
    target.setPointerCapture(pointerId);
    this.resizingChange.emit(true);
  }

  key(event: KeyboardEvent) {
    const next = steppedSize(this.size(), event.key, this.bounds(), {
      orientation: this.orientation(),
      reverse: this.reverse(),
      step: event.shiftKey ? this.fineStep() : this.step(),
    });
    if (next === null) return;
    event.preventDefault();
    this.sizeChange.emit(next);
  }

  reset() {
    const value = this.resetSize();
    if (value !== null) this.sizeChange.emit(clampSize(value, this.bounds()));
  }
}
