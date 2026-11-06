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

import {signal} from '@angular/core';
import {ConversationComposerController} from './conversation_composer';

describe('ConversationComposerController', () => {
  interface State {
    draft: string;
    refreshError: string;
    send: {allowed: boolean; reason: string};
    busy: boolean;
    selected: boolean;
    job: {
      id?: string;
      operation: string;
      progress?: string;
      error?: string;
      does_not_fit?: unknown;
    } | null;
    started: [string, number, string][];
    stopped: number;
  }

  function workspace(overrides: Partial<State> = {}) {
    const state: State = {
      draft: 'hello',
      refreshError: '',
      send: {allowed: true, reason: ''},
      busy: false,
      selected: true,
      job: null,
      started: [],
      stopped: 0,
      ...overrides,
    };
    return {
      state,
      chatDraft: () => state.draft,
      setChatDraft: (value: string) => (state.draft = value),
      chat: () => (state.selected ? {n: 2, actions: {send: state.send}} : null),
      sessions: {refreshError: () => state.refreshError},
      runtime: {
        busy: () => state.busy,
        selected: () => (state.selected ? {id: 's'} : undefined),
        start: async (operation: string, chat: number, prompt: string) => {
          state.started.push([operation, chat, prompt]);
          return true;
        },
        stop: async () => {
          state.stopped++;
        },
      },
      jobs: {job: () => state.job},
    };
  }
  const over = {
    allowed: false,
    reason: 'This chat is over; its turns can still be inspected in Debug.',
  };

  it('sending needs the server to allow it, no request in flight, a fresh listing and a draft within 4 MiB', async () => {
    const w = workspace();
    const composer = new ConversationComposerController(w);
    expect(composer.canSend()).toBe(true);
    await composer.send();
    // A turn of the chat shown, which the server refuses if another page replaced it.
    expect(w.state.started).toStrictEqual([['generate', 2, 'hello']]);
    for (const blocker of [
      {draft: '  '},
      {send: over},
      {busy: true},
      {refreshError: 'stale'},
      {draft: 'x'.repeat(4 * 1024 * 1024 + 1)},
    ]) {
      const blocked = new ConversationComposerController(workspace(blocker));
      expect(blocked.canSend(), JSON.stringify(Object.keys(blocker))).toBe(false);
    }
    expect(
      new ConversationComposerController(
        workspace({draft: '😀'.repeat(1024 * 1024 + 1)}),
      ).promptError(),
    ).toMatch(/4 MiB/);
  });

  it('the notice names the most specific blocker first', () => {
    const cases: [Partial<State>, string][] = [
      [{selected: false}, 'This draft has no runtime context'],
      [{refreshError: 'x'}, 'Runner status is unknown'],
      [{busy: true, job: {operation: 'initialize'}}, 'Initializing runtime'],
      [{busy: true, job: {operation: 'chat'}}, 'Starting chat'],
      [{busy: true, job: {operation: 'generate'}}, 'Generation in progress'],
      // Once the reply ended, its capture arriving: the composer waits for it.
      [
        {
          busy: true,
          job: {operation: 'generate', progress: 'Retrieving the capture · 1.2 GB so far'},
        },
        'Retrieving the capture · 1.2 GB so far',
      ],
      [{send: over}, over.reason],
      [
        {send: {allowed: false, reason: 'Wait for the Model Server to come on.'}},
        'Wait for the Model Server to come on.',
      ],
      [{}, ''],
    ];
    for (const [overrides, expected] of cases)
      expect(new ConversationComposerController(workspace(overrides)).sendNotice()).toBe(expected);
  });

  it('an ended chat takes no turn: its box is disabled and says why', () => {
    const ended = {allowed: false, reason: 'This chat has ended.'};
    const composer = new ConversationComposerController(workspace({send: ended}));
    expect(composer.inputDisabled()).toBe(true);
    expect(composer.canSend()).toBe(false);
    expect(composer.sendNotice()).toBe('This chat has ended.');
  });

  it('the message box is disabled while the server refuses a turn or one is in flight', () => {
    expect(new ConversationComposerController(workspace()).inputDisabled()).toBe(false);
    expect(new ConversationComposerController(workspace({send: over})).inputDisabled()).toBe(true);
    expect(new ConversationComposerController(workspace({busy: true})).inputDisabled()).toBe(true);
    expect(new ConversationComposerController(workspace({send: over})).refusal()).toBe(over.reason);
    expect(new ConversationComposerController(workspace({send: over, busy: true})).refusal()).toBe(
      '',
    );
  });

  it('a turn that did not fit keeps its text and says so under it until the text changes', () => {
    const reason =
      "This input doesn't fit: this turn needs 2100 tokens, the chat has 1990 left. Shorten it and send again.";
    const refused = {
      id: 'job-1',
      operation: 'generate',
      status: 'failed',
      error: reason,
      does_not_fit: {ref: {tokens: 2100, left: 1990}, target: {tokens: 2100, left: 1990}},
    };
    const w = workspace({draft: 'a long message'});
    const job = signal<State['job']>(refused);
    w.jobs.job = job;
    const composer = new ConversationComposerController(w);
    expect(composer.doesNotFit()).toBe(reason);
    expect(w.state.draft).toBe('a long message');
    // Any other failure says nothing here.
    expect(
      new ConversationComposerController(
        workspace({job: {...refused, does_not_fit: undefined}}),
      ).doesNotFit(),
    ).toBe('');
    const input = (value: string) =>
      composer.draftInput({target: {value, style: {}, scrollHeight: 24}} as unknown as Event);
    input('a long message');
    expect(composer.doesNotFit(), 'the same text').toBe(reason);
    input('a shorter one');
    expect(composer.doesNotFit()).toBe('');
    expect(w.state.draft).toBe('a shorter one');
    // The next turn that does not fit says so again.
    job.set({...refused, id: 'job-2'});
    expect(composer.doesNotFit()).toBe(reason);
  });

  it('Enter sends, Shift+Enter and IME composition do not', async () => {
    const w = workspace();
    const composer = new ConversationComposerController(w);
    const key = (init: Partial<KeyboardEvent>) => {
      const event = {
        key: 'Enter',
        shiftKey: false,
        isComposing: false,
        prevented: 0,
        preventDefault() {
          this.prevented++;
        },
        ...init,
      };
      composer.draftKey(event as unknown as KeyboardEvent);
      return event;
    };
    expect(key({shiftKey: true}).prevented).toBe(0);
    expect(key({isComposing: true}).prevented).toBe(0);
    expect(key({key: 'a'}).prevented).toBe(0);
    expect(key({}).prevented).toBe(1);
    await Promise.resolve();
    expect(w.state.started.length).toBe(1);
    await composer.stop();
    expect(w.state.stopped).toBe(1);
    // The Stop control says that it ends the chat.
    expect(composer.stopLabel).toBe('Stop and end chat');
    expect(composer.stopHint).toMatch(/cannot continue/);
  });
});
