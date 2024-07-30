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

import {ConnectedPosition, OverlaySizeConfig} from '@angular/cdk/overlay';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  Input,
  ViewChild,
} from '@angular/core';

import {Bubble} from '../bubble/bubble';

/** A wrapper panel around the graph renderer. */
@Component({
  standalone: true,
  selector: 'hoverable-label',
  imports: [Bubble],
  template: `
<div class="container" #container
    [bubble]="popup"
    [overlaySize]="popupSize"
    [overlayPositions]="popupPosition"
    [hoverDelayMs]="10"
    [bubbleDisabled]="!showPopup">
  {{label}}
</div>
<ng-template #popup>
  <div class="model-explorer-hoverable-label-popup">
    {{label}}
  </div>
</ng-template>
`,
  styles: `
:host {
  overflow: hidden;
}

.container {
  overflow: hidden;
  text-overflow: ellipsis;
}

::ng-deep bubble-container:has(.model-explorer-hoverable-label-popup) {
  width: 100%;
  box-shadow: none;
  border: 1px solid #ccc;
  border-radius: 4px;
}

::ng-deep .model-explorer-hoverable-label-popup {
  padding: 2px;
  font-size: 12px;
  line-height: 12px;
  background-color: white;
  color: #999;
  font-family: 'Google Sans Text', Arial, Helvetica, sans-serif;
}
`,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HoverableLabel implements AfterViewInit {
  @Input() label = '';
  @ViewChild('container') container!: ElementRef<HTMLElement>;

  showPopup = false;

  readonly popupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly popupPosition: ConnectedPosition[] = [
    {
      originX: 'end',
      originY: 'top',
      overlayX: 'end',
      overlayY: 'top',
      offsetY: -1,
    },
  ];

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  ngAfterViewInit() {
    setTimeout(() => {
      const ele = this.container.nativeElement;
      this.showPopup = ele.scrollWidth > ele.offsetWidth;
      this.changeDetectorRef.markForCheck();
    });
  }
}
