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
import {WorkspaceStateService} from '../../../state/workspace_state_service';
import type {RuntimeJob} from '../../../server/contracts/jobs';
import {RuntimeService} from '../../../server/runtime_service';
import {ConversationState} from '../../conversation_state';
import {ChatReading} from './chat_reading';

/** A generate in flight whose Reference side is thinking. */
const generating: RuntimeJob = {
  id: 'job-1',
  session_id: 'chat',
  operation: 'generate',
  status: 'running',
  error: '',
  sequence: 3,
  output: {ref: '', target: ''},
  reasoning: {ref: 'Seventeen times twenty', target: ''},
  turn: 1,
  prompt: 'What is 17 times 23?',
};

describe('ChatReading', () => {
  const live = signal<RuntimeJob | null>(null);
  /** The turns captured, each with the Reference side's thinking and reply. */
  const captured = signal<number[]>([]);

  /** Draws the reading of a chat with a capture, or without one, that may or may not send. */
  function render({hasCapture = true, send = true} = {}) {
    const workspace = {
      chatHasCapture: () => hasCapture,
      chatDraft: () => '',
      chat: () => ({
        n: 1,
        actions: {send: {allowed: send, reason: send ? '' : 'This chat has ended.'}},
      }),
      runtime: {
        error: () => '',
        busy: () => true,
      },
    };
    // The thinking a reader opened is the conversation's own; the rest is stubbed.
    const conversation = Object.assign(Object.create(ConversationState.prototype), {
      thinkingExpanded: signal(new Set<string>()),
      liveThinkingOpened: new Set<string>(),
      workspace,
      runs: () => [
        {id: 'ref', runtime: 'LiteRT-LM', backend: 'CPU'},
        {id: 'target', runtime: 'LiteRT-LM', backend: 'CPU'},
      ],
      turns: () => captured().map((n) => ({n})),
      rejectedTurns: () => [],
      sideIds: () => ['ref', 'target'],
      liveGeneration: () => live(),
      chatNotice: () => '',
      usesVirtual: () => false,
      virtualInput: () => false,
      virtualOutput: () => false,
      captured: (turn: number, run: string) =>
        run === 'ref' && captured().includes(turn)
          ? {turn, run, input: 'What is 17 times 23?', output: '391', thinking: 'Seventeen times'}
          : undefined,
      stageEnding: () => '',
      trackReadingIntent: () => false,
      readingScrolled: () => undefined,
    }) as ConversationState;
    TestBed.configureTestingModule({
      providers: [
        {provide: ConversationState, useValue: conversation},
        {provide: WorkspaceStateService, useValue: workspace},
        {
          provide: RuntimeService,
          useValue: {execution: () => undefined, sessions: {refreshError: () => ''}},
        },
      ],
    });
    const fixture = TestBed.createComponent(ChatReading);
    // The empty chat's animation runs in its page, here one without a message box: it waits.
    const page = document.createElement('conversation-panel');
    document.body.append(page);
    page.append(fixture.nativeElement);
    fixture.componentRef.setInput('tokenColor', 'match');
    fixture.componentRef.setInput('gapPalette', 'default');
    /** Draws the reading after the job changed, as ConversationPanel's effect then opens a
     *  reply's thinking. */
    const update = () => {
      conversation.openLiveThinking();
      fixture.detectChanges();
    };
    update();
    return {fixture, host: fixture.nativeElement as HTMLElement, update};
  }

  beforeEach(() => captured.set([]));
  afterEach(() => document.querySelectorAll('conversation-panel').forEach((page) => page.remove()));

  it("shows the empty chat's animation only while the chat can take a turn", () => {
    const demo = render({hasCapture: false, send: true}).host.querySelector('empty-chat-demo');
    expect(demo).not.toBeNull();
    // Labelled so it is not taken for real output.
    expect(demo!.querySelector('.demoExampleLabel')?.textContent).toBe('Example');
    TestBed.resetTestingModule();
    // A chat that ended, or a Session that ended, before any turn: nothing to type into.
    expect(
      render({hasCapture: false, send: false}).host.querySelector('empty-chat-demo'),
    ).toBeNull();
  });

  const side = (host: HTMLElement, run: string) =>
    host.querySelector(`.live-turn [data-runtime="${run}"]`) as HTMLElement;
  /** The Thinking toggle of a turn's Reference side, captured or in progress. */
  const thinking = (host: HTMLElement, turn: number | 'live') =>
    host.querySelector(
      turn === 'live'
        ? '.live-turn [data-runtime="ref"] .thinking-toggle'
        : `[data-turn="${turn}"] [data-runtime="ref"] .thinking-toggle`,
    ) as HTMLButtonElement | null;

  it("shows a side's thinking as it streams, open until the reader closes it", async () => {
    live.set(generating);
    const {fixture, host, update} = render();
    const ref = side(host, 'ref');
    const toggle = ref.querySelector('.chat-thinking .thinking-toggle') as HTMLButtonElement;
    expect(toggle.textContent).toContain('Thinking');
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(ref.querySelector('.thinking-reveal.expanded p')?.textContent).toBe(
      'Seventeen times twenty',
    );
    // A side that has not thought has no thinking, and waits for its reply.
    expect(side(host, 'target').querySelector('.chat-thinking')).toBeNull();
    expect(side(host, 'target').querySelector('.response > p')?.textContent?.trim()).toBe(
      'Waiting for output…',
    );

    live.set({...generating, reasoning: {ref: 'Seventeen times twenty-three', target: ''}});
    update();
    await fixture.whenStable();
    expect(ref.querySelector('.thinking-reveal p')?.textContent).toBe(
      'Seventeen times twenty-three',
    );

    toggle.click();
    fixture.detectChanges();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(ref.querySelector('.thinking-reveal')?.hasAttribute('inert')).toBe(true);
    // More thinking streams: it stays as the reader left it.
    live.set({...generating, reasoning: {ref: 'Seventeen times twenty-three is', target: ''}});
    update();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
  });

  it('keeps the thinking as it was once the turn is captured: nothing jumps', () => {
    live.set(generating);
    const {host, update} = render();
    expect(thinking(host, 'live')?.getAttribute('aria-expanded')).toBe('true');
    // Turn 1 is captured: the reply in progress gives way to the captured turn, still open.
    live.set(null);
    captured.set([1]);
    update();
    expect(thinking(host, 'live')).toBeNull();
    expect(thinking(host, 1)?.getAttribute('aria-expanded')).toBe('true');

    // Turn 2's thinking, closed by the reader while it streams, is captured closed.
    live.set({...generating, id: 'job-2', turn: 2});
    update();
    thinking(host, 'live')!.click();
    update();
    expect(thinking(host, 'live')?.getAttribute('aria-expanded')).toBe('false');
    live.set(null);
    captured.set([1, 2]);
    update();
    expect(thinking(host, 2)?.getAttribute('aria-expanded')).toBe('false');
    expect(thinking(host, 1)?.getAttribute('aria-expanded')).toBe('true');
  });

  it('shows no thinking for a reply that streams none', () => {
    live.set({
      ...generating,
      reasoning: {ref: '', target: ''},
      output: {ref: 'It is 4.', target: 'It is'},
    });
    const {host} = render();
    expect(host.querySelector('.live-turn .chat-thinking')).toBeNull();
    expect(side(host, 'ref').querySelector('.response > p')?.textContent?.trim()).toBe('It is 4.');
  });

  it('marks failed only the side the server says failed', () => {
    const failed = {...generating, status: 'failed', status_text: 'Generation failed'};
    const status = (host: HTMLElement, run: string) =>
      side(host, run).querySelector('.generation-status')?.textContent?.trim();
    live.set({...failed, failed_sides: ['target']});
    const {host, update} = render();
    expect(status(host, 'ref')).toBe('');
    expect(status(host, 'target')).toBe('Generation failed');
    // A failure the server names no side of marks both.
    live.set(failed);
    update();
    expect(status(host, 'ref')).toBe('Generation failed');
    expect(status(host, 'target')).toBe('Generation failed');
  });
});
