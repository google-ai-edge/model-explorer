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

import {ConversationState, liveTurn} from './conversation_state';
import type {CapturedConversation} from '../server/contracts/capture';

describe("a chat's ending, as the server says it", () => {
  function panel(conversation: CapturedConversation[], notice = '') {
    const view = Object.create(ConversationState.prototype) as ConversationState;
    Object.defineProperties(view, {
      workspace: {
        value: {
          jobs: {job: () => null},
          chat: () => ({n: 1, notice}),
        },
      },
      runs: {value: () => [{id: 'ref'}, {id: 'target'}]},
      captured: {
        value: (turn: number, run: string) =>
          conversation.find((item) => item.turn === turn && item.run === run),
      },
    });
    return view;
  }
  const full = 'Context full (1024 tokens)';

  it("shows each reply's end in the server's words", () => {
    const view = panel([
      {turn: 1, run: 'ref', stop_reason: 'context_full', stop_text: full},
      {
        turn: 1,
        run: 'target',
        stop_reason: 'max_output_tokens',
        stop_text: 'Output limit reached (8 tokens)',
      },
    ]);
    expect(view.stageEnding(1, 'ref')).toBe(full);
    expect(view.stageEnding(1, 'target')).toBe('Output limit reached (8 tokens)');
    expect(view.turnEnding(1)).toBe('Conversation complete');
  });

  it('says nothing of a reply the server gives no reason for', () => {
    const view = panel([
      {turn: 1, run: 'ref', stop_reason: 'eos', stop_text: 'Stop token'},
      {turn: 1, run: 'target', stop_reason: 'something new'},
    ]);
    expect(view.stageEnding(1, 'target')).toBe('');
    expect(view.stageEnding(2, 'ref')).toBe('');
    expect(view.turnEnding(1)).toBe('End of captured conversation');
  });

  it("shows the chat's notice once the chat is over", () => {
    const notice = 'A new chat was started.';
    expect(panel([], notice).chatNotice()).toBe(notice);
    expect(panel([]).chatNotice()).toBe('');
  });
});

describe('the turn in progress', () => {
  const job = (status: string, extra = {}) => ({status, turn: 2, ...extra});

  it('shows until the capture shows its turn, read or rejected, once', () => {
    expect(liveTurn(job('running'), [{n: 1}])).not.toBeNull();
    // Ended but not kept, as a stopped turn or one that failed before its capture: it stays.
    expect(liveTurn(job('failed'), [{n: 1}])).not.toBeNull();
    expect(liveTurn(job('completed'), [{n: 1}, {n: 2}])).toBeNull();
    // A rejected turn is among the capture's turns too: it shows there, not twice.
    expect(liveTurn(job('failed'), [{n: 1}, {n: 2}])).toBeNull();
    // Still running, a turn the capture has is not ended yet.
    expect(liveTurn(job('running'), [{n: 2}])).not.toBeNull();
  });

  it('shows nothing of a turn that did not fit', () => {
    expect(liveTurn(job('failed', {does_not_fit: {}}), [])).toBeNull();
  });
});
