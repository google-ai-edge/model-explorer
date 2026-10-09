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
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
  ViewEncapsulation,
} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';
import {findNavigatorState} from './find_navigator_state';

/**
 * First / previous / count / next controls for find results. Styles are global so the
 * hosting panels' context classes (.debug-primary, .toolbar, .match-nav) keep applying.
 */
@Component({
  selector: 'find-navigator',
  imports: [MatTooltipModule],
  templateUrl: './find_navigator.ng.html',
  styleUrl: './find_navigator.scss',
  encapsulation: ViewEncapsulation.None,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {class: 'find-tools', role: 'group', '[attr.aria-label]': 'label()'},
})
export class FindNavigator {
  readonly index = input.required<number>();
  readonly total = input.required<number>();
  readonly busy = input(false);
  readonly pending = input(false);
  readonly label = input('Find navigation');
  readonly firstLabel = input('First result');
  readonly previousLabel = input('Previous result');
  readonly nextLabel = input('Next result');
  /** Render the count as a button that emits `count` (open the query, toggle results). */
  readonly countAction = input(false);
  readonly countLabel = input('');
  readonly countTooltip = input('');
  readonly countExpanded = input<boolean | null>(null);
  readonly countControls = input<string | null>(null);
  readonly first = output<void>();
  readonly previous = output<void>();
  readonly next = output<void>();
  readonly count = output<void>();
  readonly state = computed(() =>
    findNavigatorState({
      index: this.index(),
      total: this.total(),
      busy: this.busy(),
      pending: this.pending(),
    }),
  );
}
