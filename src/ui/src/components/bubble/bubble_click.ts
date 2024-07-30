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
import {ENTER, ESCAPE, SPACE} from '@angular/cdk/keycodes';
import {ConnectedPosition, Overlay, OverlayConfig} from '@angular/cdk/overlay';
import {DOCUMENT} from '@angular/common';
import {
  Directive,
  ElementRef,
  Inject,
  Injector,
  NgZone,
  OnInit,
  ViewContainerRef,
} from '@angular/core';
import {fromEvent} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {BubbleBase} from './bubble_base';

/**
 * An attribute directive that shows a popup dialog when the element it is
 * placed on is clicked.
 */
@Directive({
  selector: '[bubbleClick]',
  exportAs: 'bubbleClick',
  host: {'class': 'bubble-click'},
  inputs: ['dialog: bubbleClick', 'disabled: bubbleDisabled'],
  standalone: true,
})
export class BubbleClick extends BubbleBase implements OnInit {
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
    this.listenForOpenEvents();
  }

  listenForOpenEvents(): void {
    this.openStatusChange
      .pipe(takeUntil(this.destroyed))
      .subscribe((opened) => {
        opened ? this.openDialog() : this.closeDialog();
      });
  }

  attachMouseEventListeners(element: HTMLElement): void {
    this.ngZone.runOutsideAngular(() => {
      fromEvent(element, 'click')
        .pipe(takeUntil(this.destroyed))
        .subscribe((event) => {
          const element = event.target as HTMLElement;
          if (element.closest('[bubbleClose]')) {
            this.closingDialog();
          } else {
            this.openingDialog();
          }
        });

      this.overlayRef
        ?.backdropClick()
        .pipe(takeUntil(this.destroyed))
        .subscribe(() => {
          this.closingDialog();
        });
    });
  }

  attachKeyboardCloseEventListeners(element: HTMLElement): void {
    this.ngZone.runOutsideAngular(() => {
      fromEvent<KeyboardEvent>(element, 'keyup')
        .pipe(takeUntil(this.destroyed))
        .subscribe((event: KeyboardEvent) => {
          // tslint:disable-next-line:deprecation consistency with keyCode usage in other event listeners in bubble
          const keyCode = event.keyCode;
          const element = event.target as HTMLElement;
          switch (keyCode) {
            case ESCAPE:
              this.closingDialog();
              return;
            case SPACE:
              if (element.closest('[bubbleClose]')) {
                this.closingDialog();
              }
              return;
            default:
              return;
          }
        });

      fromEvent<KeyboardEvent>(element, 'keydown')
        .pipe(takeUntil(this.destroyed))
        .subscribe((event: KeyboardEvent) => {
          // tslint:disable-next-line:deprecation consistency with keyCode usage in other event listeners in bubble
          const keyCode = event.keyCode;
          const element = event.target as HTMLElement;
          switch (keyCode) {
            case ENTER:
              if (element.closest('[bubbleClose]')) {
                this.closingDialog();
              }
              return;
            default:
              return;
          }
        });
    });
  }

  private openingDialog(): void {
    if (this.disabled) return;
    this.openStatusChange.next(true);
  }

  private closingDialog(): void {
    this.openStatusChange.next(false);
  }

  createOverlayConfig(positions: ConnectedPosition[]): OverlayConfig {
    return new OverlayConfig({
      ...this.overlayDimensions,
      positionStrategy: super.createPositionStrategy(positions),
      scrollStrategy: this.createScrollStrategy(),
      hasBackdrop: true,
      backdropClass: 'cdk-overlay-transparent-backdrop',
      panelClass: this.panelClassInternal,
    });
  }
}
