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
  BasePortalOutlet,
  CdkPortalOutlet,
  ComponentPortal,
  PortalModule,
  TemplatePortal,
} from '@angular/cdk/portal';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ComponentRef,
  EmbeddedViewRef,
  ViewChild,
  ViewEncapsulation,
  inject,
} from '@angular/core';

import {BUBBLE_ANIMATIONS} from './bubble_animation';

/**
 * A wrapper component of `bubble` that wraps user-provided content.
 */
@Component({
  selector: 'bubble-container',
  template: `
    <div>
      <ng-template cdkPortalOutlet>
        <a cdkFocusInitial tabindex="0"></a>
      </ng-template>
    </div>`,
  styleUrls: ['./bubble_container.scss'],
  animations: [BUBBLE_ANIMATIONS.bubbleContainer],
  encapsulation: ViewEncapsulation.None,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    'class': 'bubble-container',
    'role': 'dialog',
    '[@bubbleContainer]': 'animationState',
  },
  standalone: true,
  imports: [PortalModule],
})
export class BubbleContainer extends BasePortalOutlet {
  /**
   * The portal outlet inside of this container into which the content
   * will be loaded.
   */
  @ViewChild(CdkPortalOutlet, {static: true}) portalOutlet!: CdkPortalOutlet;

  /** State of the animation. */
  animationState: 'hidden' | 'visible' | 'void' = 'hidden';

  private readonly changeDetector = inject(ChangeDetectorRef);

  attachComponentPortal<T>(portal: ComponentPortal<T>): ComponentRef<T> {
    return this.portalOutlet.attachComponentPortal(portal);
  }

  attachTemplatePortal<C>(portal: TemplatePortal<C>): EmbeddedViewRef<C> {
    return this.portalOutlet.attachTemplatePortal(portal);
  }

  toggleAnimation(open: boolean) {
    this.animationState = open ? 'visible' : 'hidden';
    this.changeDetector.markForCheck();
  }
}
