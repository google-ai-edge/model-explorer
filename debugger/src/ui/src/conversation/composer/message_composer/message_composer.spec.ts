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
import {TestBed} from '@angular/core/testing';
import type {Permission} from '../../../server/contracts/sessions';
import {allowed, refused} from '../../../server/testing/session_fixtures';
import {WorkspaceStateService} from '../../../state/workspace_state_service';
import {ConversationComposerController} from '../conversation_composer';
import {MessageComposer} from './message_composer';

const ENDED = 'This Session has ended; its turns are shown in Debug.';

describe('MessageComposer', () => {
  /** Draws the composer in a mode, with the modes the server allows. */
  function render(
    mode: 'Chat' | 'Debug',
    modes: {Chat: Permission; Debug: Permission},
    turn: 'none' | 'generating' | 'retrieving' = 'none',
    doesNotFit = '',
  ) {
    const asked: string[] = [];
    const workspace = {
      mode: signal(mode),
      modes: signal(modes),
      setMode: (to: string) => asked.push(to),
      chatDraft: () => '',
      jobs: {generating: () => turn !== 'none', retrieving: () => turn === 'retrieving'},
      runtime: {
        busy: () => turn !== 'none',
        execution: () => (turn === 'none' ? undefined : {phase: 'active'}),
      },
    };
    const composer = {
      draftInput() {},
      draftKey() {},
      inputDisabled: () => true,
      sendNotice: () => '',
      refusal: () => '',
      doesNotFit: () => doesNotFit,
      canSend: () => false,
      stopLabel: 'Stop and end chat',
      stopHint: '',
    };
    TestBed.configureTestingModule({
      providers: [
        {provide: WorkspaceStateService, useValue: workspace},
        {provide: ConversationComposerController, useValue: composer},
      ],
    });
    const fixture = TestBed.createComponent(MessageComposer);
    fixture.detectChanges();
    const host = fixture.nativeElement as HTMLElement;
    const toggle = host.querySelector<HTMLButtonElement>('.debug-toggle button')!;
    const stop = () =>
      Array.from(host.querySelectorAll('button')).some((b) =>
        b.textContent?.includes('Stop and end chat'),
      );
    return {asked, toggle, stop, host};
  }

  it('a turn that did not fit is said in red under the message box', () => {
    const modes = {Chat: allowed, Debug: refused('This chat has no turn to inspect.')};
    const reason =
      "This input doesn't fit: this turn needs 2100 tokens, the chat has 1990 left. Shorten it and send again.";
    const line = render('Chat', modes, 'none', reason).host.querySelector('.does-not-fit');
    expect(line?.textContent?.trim()).toBe(reason);
    expect(line?.getAttribute('role')).toBe('alert');
    expect(line?.previousElementSibling?.tagName).toBe('TEXTAREA');
    TestBed.resetTestingModule();
    expect(render('Chat', modes).host.querySelector('.does-not-fit')).toBeNull();
  });

  it('Stop shows while the reply is generated, not once its capture is arriving', () => {
    const modes = {Chat: allowed, Debug: refused('This chat has no turn to inspect.')};
    expect(render('Chat', modes, 'generating').stop()).toBe(true);
    TestBed.resetTestingModule();
    expect(render('Chat', modes, 'retrieving').stop()).toBe(false);
  });

  it('the Debug switch goes back to Chat only while the server allows Chat', () => {
    const open = render('Debug', {Chat: allowed, Debug: allowed});
    expect(open.toggle.disabled).toBe(false);
    open.toggle.click();
    expect(open.asked).toStrictEqual(['Chat']);
    TestBed.resetTestingModule();
    // An ended Session with a turn kept: Debug only, and the switch says why.
    const ended = render('Debug', {Chat: refused(ENDED), Debug: allowed});
    expect(ended.toggle.disabled).toBe(true);
    expect(ended.toggle.closest('[aria-description]')?.getAttribute('aria-description')).toBe(
      ENDED,
    );
  });

  it('the Debug switch goes to Debug only once the server allows it', () => {
    const none = render('Chat', {
      Chat: allowed,
      Debug: refused('This chat has no turn to inspect.'),
    });
    expect(none.toggle.disabled).toBe(true);
    TestBed.resetTestingModule();
    const some = render('Chat', {Chat: allowed, Debug: allowed});
    some.toggle.click();
    expect(some.asked).toStrictEqual(['Debug']);
  });
});
