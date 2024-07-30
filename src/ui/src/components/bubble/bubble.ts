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

import {ConfigurableFocusTrapFactory} from '@angular/cdk/a11y';
import {ESCAPE} from '@angular/cdk/keycodes';
import {ConnectedPosition, Overlay, OverlayConfig} from '@angular/cdk/overlay';
import {DOCUMENT} from '@angular/common';
import {
  Directive,
  ElementRef,
  Inject,
  Injector,
  Input,
  NgZone,
  OnInit,
  ViewContainerRef,
} from '@angular/core';
import {Subject, fromEvent} from 'rxjs';
import {audit, debounceTime, takeUntil} from 'rxjs/operators';

import {BubbleBase} from './bubble_base';

/**
 * Default delay in milliseconds of showing the dialog with mouse interactions.
 */
export const DEFAULT_MOUSE_DELAY_MS = 500;

/**
 * An attribute directive that shows a popup dialog when hovered on the element
 * it is placed on.
 */
@Directive({
  selector: '[bubble]',
  exportAs: 'bubble',
  host: {'class': 'bubble'},
  inputs: ['dialog: bubble', 'disabled: bubbleDisabled'],
  standalone: true,
})
export class Bubble extends BubbleBase implements OnInit {
  /** Controls hover delay in milliseconds after mouseenter from mouse rest. */
  @Input() hoverDelayMs: number = DEFAULT_MOUSE_DELAY_MS;

  private readonly activity = new Subject<void>();

  constructor(
    ngZone: NgZone,
    overlay: Overlay,
    elementRef: ElementRef,
    viewContainerRef: ViewContainerRef,
    @Inject(DOCUMENT) document: Document,
    focusTrapFactory: ConfigurableFocusTrapFactory,
    injector: Injector,
  ) {
    super(
      ngZone,
      overlay,
      elementRef,
      viewContainerRef,
      document,
      focusTrapFactory,
      injector,
    );
    this.attachMouseEventListeners(elementRef.nativeElement);
  }

  override ngOnInit(): void {
    super.ngOnInit();
    this.listenForOpenEvents(this.hoverDelayMs);
  }

  private listenForOpenEvents(hoverDelayMs: number): void {
    // Delay opening/closing until mouse has stopped moving and a short amount
    // of time (can be configured via `hoverDelayMs` input field) has passed in
    // order to ignore accidental movements.
    const debouncedStatusChange = this.openStatusChange.pipe(
      audit(() => this.activity.pipe(debounceTime(hoverDelayMs))),
    );

    debouncedStatusChange
      .pipe(takeUntil(this.destroyed))
      .subscribe((opened) => {
        opened ? this.openDialog() : this.closeDialog();
      });
  }

  attachMouseEventListeners(element: HTMLElement): void {
    this.ngZone.runOutsideAngular(() => {
      fromEvent(element, 'mouseenter')
        .pipe(takeUntil(this.destroyed))
        .subscribe(() => {
          this.openingDialog();
        });

      fromEvent(element, 'click')
        .pipe(takeUntil(this.destroyed))
        .subscribe((event) => {
          const element = event.target as HTMLElement;
          if (element.closest('[bubbleClose]')) {
            this.closingDialog();
          } else {
            this.openingDialog();
            this.openDialog();
          }
        });

      fromEvent(element, 'mouseleave')
        .pipe(takeUntil(this.destroyed))
        .subscribe(() => {
          this.closingDialog();
        });

      fromEvent(element, 'mousemove')
        .pipe(takeUntil(this.destroyed))
        .subscribe(() => {
          this.activity.next();
        });
    });
  }

  attachKeyboardCloseEventListeners(element: HTMLElement): void {
    this.ngZone.runOutsideAngular(() => {
      fromEvent<KeyboardEvent>(element, 'keydown')
        .pipe(takeUntil(this.destroyed))
        .subscribe((event: KeyboardEvent) => {
          // tslint:disable-next-line:deprecation consistency with keyCode usage in other event listeners in bubble
          const keyCode = event.keyCode;
          switch (keyCode) {
            case ESCAPE:
              // Prevents escape keydown event propagation if dialog is open.
              if (this.overlayRef?.hasAttached()) {
                event.stopPropagation();
              }
              this.closeDialog();
              return;
            default:
              return;
          }
        });
    });
  }

  /**
   * Open the dialog after cursor stops moving for the period of time set in
   * `hoverDelayMs` and if it's in the triggering area.
   */
  private openingDialog(): void {
    if (this.disabled) return;
    this.openStatusChange.next(true);
    this.activity.next();
  }

  /**
   * Close the dialog after cursor stops moving for the period of time set in
   * `hoverDelayMs` and if it's not in the triggering area.
   */
  private closingDialog(): void {
    this.openStatusChange.next(false);
    this.activity.next();
  }

  createOverlayConfig(positions: ConnectedPosition[]): OverlayConfig {
    return new OverlayConfig({
      ...this.overlayDimensions,
      positionStrategy: super.createPositionStrategy(positions),
      scrollStrategy: this.createScrollStrategy(),
      panelClass: this.panelClassInternal,
    });
  }
}
