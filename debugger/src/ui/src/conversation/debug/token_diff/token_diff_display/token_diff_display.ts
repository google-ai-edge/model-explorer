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

import {ChangeDetectionStrategy, Component, inject} from '@angular/core';

import {TokenDiff} from '../token_diff';
import {TokenDisplay} from '../token_display/token_display';

/** The Token Diff's display in the composer's bottom row: what colors the tokens, its legend,
 *  and the Display settings. */
@Component({
  selector: 'token-diff-display',
  imports: [TokenDisplay],
  templateUrl: './token_diff_display.ng.html',
  styleUrl: './token_diff_display.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TokenDiffDisplay {
  protected readonly tokens = inject(TokenDiff);
}
