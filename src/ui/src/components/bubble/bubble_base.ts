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
  ConfigurableFocusTrap,
  ConfigurableFocusTrapFactory,
} from '@angular/cdk/a11y';
import {
  ConnectedPosition,
  Overlay,
  OverlayConfig,
  OverlayRef,
  PositionStrategy,
  ScrollStrategy,
  type OverlaySizeConfig,
} from '@angular/cdk/overlay';
import {
  ComponentPortal,
  ComponentType,
  Portal,
  TemplatePortal,
} from '@angular/cdk/portal';
import {
  ComponentRef,
  Directive,
  ElementRef,
  EventEmitter,
  Injector,
  Input,
  NgZone,
  OnDestroy,
  OnInit,
  Output,
  TemplateRef,
  ViewContainerRef,
} from '@angular/core';
import {BehaviorSubject, ReplaySubject} from 'rxjs';
import {filter} from 'rxjs/operators';

import {ANIMATION_TRANSITION_TIME_MS} from './bubble_animation';
import {BubbleContainer} from './bubble_container';

/**
 * Default set of positions for the dialog. Follows the behavior of a dropdown.
 */
export const DEFAULT_POSITION_LIST: ConnectedPosition[] = [
  // bottom right
  {
    originX: 'start',
    originY: 'bottom',
    overlayX: 'start',
    overlayY: 'top',
    offsetY: 8,
  },
  // top right
  {
    originX: 'start',
    originY: 'top',
    overlayX: 'start',
    overlayY: 'bottom',
    offsetY: -8,
  },
  // top left
  {
    originX: 'end',
    originY: 'top',
    overlayX: 'end',
    overlayY: 'bottom',
    offsetY: -8,
  },
  // bottom left
  {
    originX: 'end',
    originY: 'bottom',
    overlayX: 'end',
    overlayY: 'top',
    offsetY: 8,
  },
];

/**
 * Dimensions (in px) of overlay.
 */
export const DEFAULT_OVERLAY_DIMENSIONS: BubbleOverlayDimensions = {
  minWidth: 220,
  maxWidth: 420,
  minHeight: 64,
  maxHeight: 420,
};

/**
 * Interface for customize dimensions (in px) of overlay.
 */
export interface BubbleOverlayDimensions {
  minWidth: number;
  maxWidth?: number;
  minHeight: number;
  maxHeight?: number;
}

/**
 * A base attribute directive that shows a popup panel when the element it is
 * placed on is triggered.
 */
@Directive({
  standalone: true,
})
export abstract class BubbleBase implements OnInit, OnDestroy {
  protected overlayRef?: OverlayRef;

  private portal?: Portal<unknown>;
  private disabledInternal = false;
  panelClassInternal?: string | string[];

  set dialog(
    dialog: TemplateRef<unknown> | ComponentType<unknown> | undefined,
  ) {
    // Don't set the portal if dialog is null or an empty string in the case
    // where this directive is applied without square brackets.
    if (!dialog) return;

    if (dialog instanceof TemplateRef) {
      this.setPortal(new TemplatePortal(dialog, this.viewContainerRef));
    } else {
      this.setPortal(new ComponentPortal(dialog, this.viewContainerRef));
    }
  }

  /**
   * When set to true, remove the dialog effects and style the trigger like a
   * normal element.
   */
  set disabled(input: boolean) {
    this.disabledInternal = input;
    if (this.disabledInternal) {
      this.closeDialog();
    }
  }

  get disabled() {
    return this.disabledInternal;
  }

  /** Overlay dimensions for the dialog. */
  @Input() overlaySize?: OverlaySizeConfig;

  /**
   * Overlay positions for the dialog. If the list is empty, no change is made
   * to the overlay position.
   */
  @Input() overlayPositions?: ConnectedPosition[];

  /** The dimension of an overlay. */
  @Input()
  overlayDimensions: BubbleOverlayDimensions = DEFAULT_OVERLAY_DIMENSIONS;

  /**
   * Sets a panel class on the overlay for the dialog.
   */
  @Input()
  set panelClass(panelClass: string | string[]) {
    if (this.panelClassInternal === panelClass) {
      return;
    }
    if (this.panelClassInternal) {
      this.overlayRef?.removePanelClass(this.panelClassInternal);
    }
    if (panelClass) {
      this.overlayRef?.addPanelClass(panelClass);
    }
    this.panelClassInternal = panelClass;
  }

  /** Emits when the dialog is opened. */
  @Output() readonly opened = new EventEmitter<void>();

  /** Emits when the dialog is closed. */
  @Output() readonly closed = new EventEmitter<void>();

  protected readonly destroyed = new ReplaySubject<void>();

  protected readonly openStatusChange = new BehaviorSubject<boolean>(false);
  protected readonly openings = this.openStatusChange.pipe(
    filter((isOpening) => isOpening && !this.disabled),
  );
  private bubbleContainerRef: ComponentRef<BubbleContainer> | undefined;

  private focusTrap?: ConfigurableFocusTrap;

  /**
   * Element that was focused before the dialog was opened. Save this to restore
   * upon close.
   */
  private elementFocusedBeforeDialogWasOpened: HTMLElement | undefined =
    undefined;

  abstract attachMouseEventListeners(element: HTMLElement): void;

  abstract createOverlayConfig(positions: ConnectedPosition[]): OverlayConfig;

  constructor(
    protected readonly ngZone: NgZone,
    protected readonly overlay: Overlay,
    protected readonly elementRef: ElementRef,
    protected readonly viewContainerRef: ViewContainerRef,
    protected readonly document: Document,
    protected readonly focusTrapFactory: ConfigurableFocusTrapFactory,
    protected readonly injector: Injector,
  ) {
    this.attachKeyboardCloseEventListeners(elementRef.nativeElement);
  }

  ngOnInit(): void {}

  ngOnDestroy(): void {
    this.closeDialog();
    this.openStatusChange.complete();
    this.destroyed.next();
    this.destroyed.complete();
    if (this.overlayRef) {
      this.overlayRef.dispose();
    }
  }

  setPortal(portal: Portal<unknown>) {
    this.portal = portal;
  }

  abstract attachKeyboardCloseEventListeners(element: HTMLElement): void;

  /** Open the dialog with no preconditions. */
  openDialog(): void {
    if (this.disabled) return;
    // Don't do anything if the dialog is already open.
    if (this.overlayRef?.hasAttached()) {
      return;
    }

    // Store a local copy of the portal, so that the reference can't be
    // reassigned from underneath while running in the Zone.
    const portal = this.portal;

    // If there's no dialog, don't open a Portal.
    if (portal == null) return;

    this.ngZone.run(() => {
      this.bubbleContainerRef = this.createAndAttachBubbleContainer();
      this.bubbleContainerRef.instance.attach(portal);

      const dialogContent = this.bubbleContainerRef.location.nativeElement;

      this.attachMouseEventListeners(dialogContent);
      this.attachKeyboardCloseEventListeners(dialogContent);

      this.trapFocus(dialogContent);

      this.bubbleContainerRef.instance.toggleAnimation(true);

      // Only emit the event when there are observers.
      if (this.opened.observers.length) {
        // Allow the animation to finish before emitting opened event.
        setTimeout(() => {
          this.opened.emit();
        }, ANIMATION_TRANSITION_TIME_MS);
      }
    });
  }

  /** Close the dialog with no preconditions. */
  closeDialog(): void {
    // Don't do anything if the dialog is already closed.
    if (!this.overlayRef?.hasAttached()) {
      return;
    }

    this.bubbleContainerRef!.instance.toggleAnimation(false);

    // Allow the animation to finish before destroying the container.
    setTimeout(() => {
      this.ngZone.run(() => {
        if (this.overlayRef) {
          this.overlayRef.detach();
        }

        this.cleanupFocusTrap();
        this.restoreFocus();

        this.cleanupBubbleContainer();
        this.closed.emit();
      });
    }, ANIMATION_TRANSITION_TIME_MS);
  }

  /** Moves the focus inside the focus trap. */
  private trapFocus(element: HTMLElement) {
    const activeElement = this.document.activeElement;
    const triggerElement = this.elementRef.nativeElement;

    // Do not trap focus if we don't have a focused element, or the focused
    // element is not the trigger element. That is, trap focus only if the user
    // is using keyboard to interact with the dialog.
    const shouldTrapFocus = activeElement && activeElement === triggerElement;
    if (!shouldTrapFocus) {
      return;
    }

    // Saves a reference to the element that was focused before the dialog was
    // opened, so we can restore focus to this element after the dialog is
    // closed.
    if (this.document) {
      this.elementFocusedBeforeDialogWasOpened = this.document
        .activeElement as HTMLElement;
    }

    this.focusTrap = this.focusTrapFactory.create(element);
    this.focusTrap.attachAnchors();
  }

  /**
   * Focus on the initial element. This is usually done automatically
   * while opening the dialog, but can be manually controlled to defer the focus
   * to deal with cases like that the dialog content is deferred loaded
   * asynchronously.
   */
  focusInitialElement(): void {
    if (!this.focusTrap) return;
    this.focusTrap.focusInitialElementWhenReady();
  }

  private cleanupFocusTrap(): void {
    if (this.focusTrap) {
      this.focusTrap.destroy();
      this.focusTrap = undefined;
    }
  }

  /**
   * Restores focus to the element that was focused before the dialog opened.
   */
  private restoreFocus() {
    const toFocus = this.elementFocusedBeforeDialogWasOpened;

    // Don't do anything if we don't have an element to restore focus, or it
    // cannot be focused.
    if (!toFocus || typeof toFocus.focus !== 'function') {
      return;
    }

    const dialogContent = this.bubbleContainerRef?.location.nativeElement;
    const activeElement = this.document.activeElement;

    // Don't do anything if we don't have a focused element, or the focused
    // element is not inside the dialog.
    if (!activeElement || !dialogContent?.contains(activeElement)) {
      return;
    }

    toFocus.focus();
    this.elementFocusedBeforeDialogWasOpened = undefined;
  }

  private cleanupBubbleContainer(): void {
    if (this.bubbleContainerRef) {
      this.bubbleContainerRef.destroy();
      this.bubbleContainerRef = undefined;
    }
  }

  protected createPositionStrategy(
    positions: ConnectedPosition[],
  ): PositionStrategy {
    return this.overlay
      .position()
      .flexibleConnectedTo(this.elementRef)
      .withPositions(positions)
      .setOrigin(this.elementRef);
  }

  protected createScrollStrategy(): ScrollStrategy {
    const strategies = this.overlay.scrollStrategies;
    return strategies.close();
  }

  private createAndAttachBubbleContainer(): ComponentRef<BubbleContainer> {
    const containerInjector = Injector.create({
      parent: this.injector,
      providers: [],
    });
    const containerPortal = new ComponentPortal(
      BubbleContainer,
      null,
      containerInjector,
    );
    if (this.overlayRef == null) {
      this.overlayRef = this.overlay.create(
        this.createOverlayConfig(DEFAULT_POSITION_LIST),
      );
    }
    if (this.overlaySize) {
      this.overlayRef.updateSize(this.overlaySize);
    }
    if (this.overlayPositions && this.overlayPositions.length > 0) {
      this.overlayRef.updatePositionStrategy(
        this.createPositionStrategy(this.overlayPositions),
      );
    }
    return this.overlayRef.attach<BubbleContainer>(containerPortal);
  }
}
