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
  afterNextRender,
  Component,
  ElementRef,
  inject,
  NgZone,
  OnDestroy,
} from '@angular/core';
import {mountEmptyChatDemo} from '../empty_chat_demo';
@Component({
  selector: 'empty-chat-demo',
  templateUrl: './empty_chat_demo.ng.html',
  styleUrl: './empty_chat_demo.scss',
})
export class EmptyChatDemo implements OnDestroy {
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly zone = inject(NgZone);
  private dispose = () => {};
  constructor() {
    afterNextRender(() => {
      const host = this.element.nativeElement;
      this.dispose = this.zone.runOutsideAngular(() =>
        mountEmptyChatDemo(
          host.closest('conversation-panel')!,
          host.querySelector('.inPlaceDemo')!,
        ),
      );
    });
  }
  ngOnDestroy() {
    this.dispose();
  }
}
