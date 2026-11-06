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

import {ConfigPicker} from '../../../../widgets/config_picker/config_picker';
import {ConversationState} from '../../../conversation_state';
import {StageAnalyses} from '../../stage_analysis_rail/stage_analyses';
import {TokenDiff} from '../token_diff';
import {TokenDiffDom} from '../token_diff_dom';
import {TokenQueryControl} from '../token_query_control/token_query_control';
import {FindNavigator} from './find_navigator/find_navigator';

/** The Token Diff's tools in the composer: the alignment, and the find formula with its
 *  results. */
@Component({
  selector: 'token-diff-find',
  imports: [ConfigPicker, FindNavigator, TokenQueryControl],
  templateUrl: './token_diff_find.ng.html',
  styleUrl: './token_diff_find.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TokenDiffFind {
  protected readonly conversation = inject(ConversationState);
  protected readonly tokens = inject(TokenDiff);
  protected readonly analyses = inject(StageAnalyses);
  protected readonly dom = inject(TokenDiffDom);
  readonly alignmentChoices = ['Context', 'Steps'];
  readonly alignmentDescriptions = {
    Context: 'Match output content',
    Steps: 'Match generation-step index',
  };
}
