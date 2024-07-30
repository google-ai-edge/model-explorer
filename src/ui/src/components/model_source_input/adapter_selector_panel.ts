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

import {animate, state, style, transition, trigger} from '@angular/animations';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';
import {ReactiveFormsModule} from '@angular/forms';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {type AdapterExtension} from '../../common/types';

/**
 * The panel to show for selecting a adapter.
 */
@Component({
  standalone: true,
  selector: 'adapter-selector-panel',
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatIconModule,
    MatSelectModule,
    MatTooltipModule,
    ReactiveFormsModule,
  ],
  templateUrl: './adapter_selector_panel.ng.html',
  styleUrls: ['./adapter_selector_panel.scss'],
  animations: [
    trigger('transformPanel', [
      state(
        'void',
        style({
          opacity: 0,
          transform: 'scale(1, 0.8)',
        }),
      ),
      transition(
        'void => showing',
        animate(
          '120ms cubic-bezier(0, 0, 0.2, 1)',
          style({
            opacity: 1,
            transform: 'scale(1, 1)',
          }),
        ),
      ),
      transition('* => void', animate('100ms linear', style({opacity: 0}))),
    ]),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AdapterSelectorPanel {
  @Input() candidates: AdapterExtension[] = [];
  @Input() selectedAdapter?: AdapterExtension;
  @Output() readonly onClose = new EventEmitter<AdapterExtension | undefined>();

  handleSelectCandidate(candidate: AdapterExtension) {
    this.onClose.next(candidate);
  }
}
