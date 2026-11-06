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

import {ChangeDetectionStrategy, Component, computed, inject, input} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {WorkspaceStateService} from '../../../state/workspace_state_service';
import {RunnerStatus} from '../runner_status/runner_status';
import {ConversationState} from '../../conversation_state';
import {RejectedTurn} from '../../rejected_turn/rejected_turn';
import {comparisonPairs} from '../../token_alignment';
import {EmptyChatDemo} from '../empty_chat_demo/empty_chat_demo';

/**
 * The Chat reading: both runtimes' messages side by side, each side's Runner status over its
 * column, the reply in progress with its thinking as it streams and, before the first turn while
 * the chat can take one, the empty chat's animation.
 */
@Component({
  selector: 'chat-reading',
  imports: [
    EmptyChatDemo,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    RejectedTurn,
    RunnerStatus,
  ],
  templateUrl: './chat_reading.ng.html',
  styleUrl: './chat_reading.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatReading {
  /** The Token Diff's color and gap palette: long replies draw their alignment gaps by them. */
  readonly tokenColor = input.required<string>();
  readonly gapPalette = input.required<string>();
  protected readonly conversation = inject(ConversationState);
  protected readonly workspace = inject(WorkspaceStateService);
  /**
   * Whether the chat takes a turn now, as the server's `send` action says: the empty chat's
   * animation types into the message box, so a chat that cannot send shows none.
   */
  protected readonly canSend = computed(() => !!this.workspace.chat()?.actions.send.allowed);
  private readonly chatText = computed(() => {
    const result: Record<string, {text: string; mismatch: boolean}[]> = {};
    for (const turn of this.conversation.turns()) {
      if (this.conversation.virtualOutput(turn.n)) {
        continue;
      }
      const rows = comparisonPairs(
        turn.comparison?.alignments.content.rows,
        this.conversation.captured(turn.n, 'ref')?.tokens,
        this.conversation.captured(turn.n, 'target')?.tokens,
      );
      for (const run of ['ref', 'target']) {
        const parts = rows.flatMap((r) => {
          const t = run === 'ref' ? r.ref : r.target;
          return t?.text != null && (t.phase ?? 'response') === 'response'
            ? [{text: t.text, mismatch: r.match === false}]
            : [];
        });
        if (parts.map((t) => t.text).join('') === this.conversation.captured(turn.n, run)?.output) {
          result[`${turn.n}:${run}`] = parts;
        }
      }
    }
    return result;
  });
  chatTokens(turn: number, run: string): {text: string; mismatch: boolean}[] {
    return this.chatText()[`${turn}:${run}`] ?? [];
  }
}
