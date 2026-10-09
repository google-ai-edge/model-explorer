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

import {CdkTrapFocus} from '@angular/cdk/a11y';
import {hasModifierKey} from '@angular/cdk/keycodes';
import {CdkConnectedOverlay} from '@angular/cdk/overlay';
import {Directive, inject, input, OnInit, output} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {OverlayPlacement, overlayPositions} from './overlay_positions';

/**
 * An anchored pop-up dialog on the CDK connected overlay with the application's
 * conventions baked in: transparent backdrop, push-into-viewport, a placement with its
 * vertical flip as fallback, and `closed` on backdrop click or Escape. The host keeps
 * the `open` signal, so a programmatic close never echoes back as a user close.
 *
 *   <ng-template overlayDialog [origin]="origin" [open]="open()" (closed)="open.set(false)">
 *     <section overlayPanel aria-label="…">…</section>
 *   </ng-template>
 */
@Directive({
  selector: 'ng-template[overlayDialog]',
  hostDirectives: [
    {
      directive: CdkConnectedOverlay,
      inputs: [
        'cdkConnectedOverlayOrigin: origin',
        'cdkConnectedOverlayOpen: open',
        'cdkConnectedOverlayWidth: width',
      ],
      outputs: ['overlayKeydown'],
    },
  ],
})
export class OverlayDialog implements OnInit {
  private readonly overlay = inject(CdkConnectedOverlay);
  readonly placement = input<OverlayPlacement>('below-start');
  /** Distance between origin and panel, in px. */
  readonly gap = input(8);
  /** Horizontal nudge of the panel, in px. */
  readonly inset = input(0);
  readonly viewportMargin = input(8);
  /** The user dismissed the panel (backdrop click or Escape). */
  readonly closed = output<void>();
  constructor() {
    const overlay = this.overlay;
    overlay.hasBackdrop = true;
    overlay.backdropClass = 'cdk-overlay-transparent-backdrop';
    overlay.push = true;
    overlay.positions = overlayPositions('below-start', 8);
    overlay.backdropClick
      .pipe(takeUntilDestroyed())
      .subscribe(() => this.closed.emit());
    // The CDK detaches on Escape itself; the host's open state has to follow.
    overlay.overlayKeydown.pipe(takeUntilDestroyed()).subscribe((event) => {
      if (event.key === 'Escape' && !hasModifierKey(event)) this.closed.emit();
    });
  }
  ngOnInit() {
    this.overlay.positions = overlayPositions(
      this.placement(),
      this.gap(),
      this.inset(),
    );
    this.overlay.viewportMargin = this.viewportMargin();
  }
}

/** The dialog surface inside an OverlayDialog: a focus trap that captures focus on open. */
@Directive({
  selector: '[overlayPanel]',
  hostDirectives: [CdkTrapFocus],
  host: {role: 'dialog'},
})
export class OverlayPanel {
  constructor() {
    inject(CdkTrapFocus).autoCapture = true;
  }
}
