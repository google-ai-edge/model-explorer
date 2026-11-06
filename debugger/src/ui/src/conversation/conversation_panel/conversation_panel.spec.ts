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

import {Component, input} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {provideRouter} from '@angular/router';

import type {DecodeComparison, Session} from '../../server/contracts/capture';
import {HOME} from '../../state/page_address/page_address';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {MAX_PROMPT_BYTES} from '../composer/conversation_composer';
import {MessageComposer} from '../composer/message_composer/message_composer';
import {ConversationState} from '../conversation_state';
import {readingPositionKey} from '../conversation_view_state';
import {DebugAddress} from '../debug/debug_address';
import {ConversationViews} from '../debug/saved_views/conversation_views';
import {AnalysisPanels} from '../debug/stage_analysis_rail/analysis_panels';
import {TokenDiff} from '../debug/token_diff/token_diff';
import type {TokenColor} from '../token_analysis';
import {ConversationPanel} from './conversation_panel';

@Component({
  selector: 'chat-reading',
  template: '<div class="reading chat-reading-stub"></div>',
})
class StubChatReading {
  readonly tokenColor = input<TokenColor>('token_match');
  readonly gapPalette = input<'logits' | 'activation'>('activation');
}

@Component({
  selector: 'token-diff-reading',
  template: '<div class="reading debug-reading-stub"></div>',
})
class StubTokenDiffReading {}

@Component({
  selector: 'details-panel',
  template: '<aside class="details-panel-stub"></aside>',
})
class StubDetailsPanel {}

@Component({
  selector: 'message-composer',
  providers: [{provide: MessageComposer, useExisting: StubMessageComposer}],
  template:
    '<footer class="composer"><ng-content select="[composer-tools]" /><ng-content select="[composer-status]" /></footer>',
})
class StubMessageComposer {
  readonly inspectorWidth = input(0);
}

@Component({
  selector: 'token-diff-find',
  template: '<div class="token-diff-find-stub"></div>',
})
class StubTokenDiffFind {}

@Component({
  selector: 'token-diff-display',
  template: '<div class="token-diff-display-stub"></div>',
})
class StubTokenDiffDisplay {}

function sampleSession(): Session {
  const decode: DecodeComparison = {
    rows: [{ref: 0, target: 0, same: false}],
    first_mismatch: 0,
    first_divergence: null,
    conclusion: {kind: 'diverged', step: 0, prefix: 0},
    stages: {
      prefill: {match_rate: 1, cumulative_match_rate: 1, first_mismatch: false},
      decode: {match_rate: 0, cumulative_match_rate: 0, first_mismatch: true},
    },
    positions: {start: 0, pairs: [[0, 0]]},
  };
  return {
    model: 'gemma',
    runs: [
      {id: 'ref', runtime: 'cpu', backend: 'cpu', artifact: 'ref.tflite'},
      {id: 'target', runtime: 'gpu', backend: 'gpu', artifact: 'target.tflite'},
    ],
    turns: [
      {
        n: 1,
        step_start: 0,
        comparison: {
          history: {prefill: null, decode: 'same'},
          prefill: {
            rows: [],
            messages: [],
            conclusion: {kind: 'same', row: null},
            positions: {start: 0, pairs: []},
          },
          alignments: {content: decode, steps: decode},
        },
      },
    ],
    batches: [],
    conversation: [
      {turn: 1, run: 'ref', output: 'hello', tokens: [{step: 0, text: 'hello'}]},
      {turn: 1, run: 'target', output: 'world', tokens: [{step: 0, text: 'world'}]},
    ],
  };
}

describe('ConversationPanel', () => {
  beforeEach(() => {
    // No server: nothing the page asks for arrives.
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => new Promise<Response>(() => {}));
    // jsdom lays nothing out.
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe(): void {}
        unobserve(): void {}
        disconnect(): void {}
      },
    );
    TestBed.configureTestingModule({providers: [provideRouter([])]});
    // Keep ConversationPanel's real template and host bindings while stubbing leaf subtrees.
    TestBed.overrideComponent(ConversationPanel, {
      set: {
        imports: [
          StubChatReading,
          StubDetailsPanel,
          StubMessageComposer,
          StubTokenDiffDisplay,
          StubTokenDiffFind,
          StubTokenDiffReading,
        ],
      },
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  /** Mounts ConversationPanel with its component-level providers. */
  function open() {
    const fixture = TestBed.createComponent(ConversationPanel);
    fixture.detectChanges();
    const {injector} = fixture.debugElement;
    return {
      fixture,
      workspace: injector.get(WorkspaceStateService),
      conversation: injector.get(ConversationState),
      tokens: injector.get(TokenDiff),
      panels: injector.get(AnalysisPanels),
      views: injector.get(ConversationViews),
      address: injector.get(DebugAddress),
    };
  }

  it("keeps the Chat's Token Diff view and where its reading was", () => {
    const first = open();
    first.tokens.formula.set('token_match');
    first.tokens.panelWidth.set(480);
    const reading = first.fixture.nativeElement.querySelector('.reading') as HTMLElement;
    reading.scrollTop = 240;
    const key = readingPositionKey(
      first.conversation.contextKey,
      first.conversation.workspace.mode(),
    );
    // Leaving the Chat ends its page.
    first.fixture.destroy();

    const second = open();
    expect(second.conversation.contextKey).toBe(first.conversation.contextKey);
    expect(second.tokens.formula()).toBe('token_match');
    expect(second.tokens.panelWidth()).toBe(480);
    expect(second.conversation.memory.readingPositions.get(key)?.scrollTop).toBe(240);
    second.fixture.destroy();
  });

  it('switches template branches and composer bindings between Chat and Debug modes', () => {
    const page = open();
    const host = page.fixture.nativeElement as HTMLElement;

    expect(host.querySelector('chat-reading')).not.toBeNull();
    expect(host.querySelector('token-diff-reading')).toBeNull();
    expect(host.querySelector('details-panel')).toBeNull();
    expect(host.querySelector('token-diff-find')).toBeNull();
    expect(host.querySelector('token-diff-display')).toBeNull();

    page.tokens.color.set('none');
    page.fixture.detectChanges();
    const chatReading = page.fixture.debugElement.query(By.directive(StubChatReading))
      .componentInstance as StubChatReading;
    expect(chatReading.tokenColor()).toBe('none');

    page.workspace.address.set({...HOME, mode: 'Debug'});
    page.tokens.panelWidth.set(420);
    page.tokens.details.set(true);
    page.fixture.detectChanges();

    expect(host.querySelector('chat-reading')).toBeNull();
    expect(host.querySelector('token-diff-reading')).not.toBeNull();
    expect(host.querySelector('details-panel')).not.toBeNull();
    expect(host.querySelector('token-diff-find')).not.toBeNull();
    expect(host.querySelector('token-diff-display')).not.toBeNull();

    const composer = page.fixture.debugElement.query(By.directive(StubMessageComposer))
      .componentInstance as StubMessageComposer;
    expect(composer.inspectorWidth()).toBe(420);

    page.tokens.details.set(false);
    page.fixture.detectChanges();
    expect(composer.inspectorWidth()).toBe(0);
    page.fixture.destroy();
  });

  it('renders simultaneous prompt, storage, and address alerts in a stacked container', () => {
    const page = open();
    const host = page.fixture.nativeElement as HTMLElement;
    expect(host.querySelector('.chat-warnings')).toBeNull();

    page.workspace.setChatDraft('x'.repeat(MAX_PROMPT_BYTES + 1));
    page.workspace.storageError.set('Local storage quota exceeded.');
    page.address.notice.set('Requested turn is unavailable.');
    page.views.viewNotice.set('Restored saved view');
    page.fixture.detectChanges();

    const warnings = Array.from(host.querySelectorAll('.chat-warnings .chat-warning')).map((el) =>
      el.textContent?.trim(),
    );
    expect(warnings).toEqual([
      'Message exceeds the 4 MiB upload limit. Your full draft is retained.',
      'Local storage quota exceeded.',
      'Requested turn is unavailable.',
    ]);
    expect(host.querySelector('.view-notice')?.textContent).toBe('Restored saved view');
    page.fixture.destroy();
  });

  it('closes expanded analysis before token selection when Escape is pressed', () => {
    const page = open();
    const host = page.fixture.nativeElement as HTMLElement;
    page.panels.expandedAnalysis.set({
      id: 'panel-1',
      turn: 1,
      stage: 'decode',
      kind: 'kv',
      open: true,
    });
    page.tokens.selection.set({kind: 'token', turn: 1, index: 0});
    page.fixture.detectChanges();

    host.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    expect(page.panels.expandedAnalysis()).toBeNull();
    expect(page.tokens.selection()).toEqual({kind: 'token', turn: 1, index: 0});

    host.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    expect(page.tokens.selection()).toBeNull();
    page.fixture.destroy();
  });

  it('delegates virtualizer token clicks to TokenDiff selection in Debug mode only', () => {
    const page = open();
    page.conversation.state.session.set(sampleSession());
    page.workspace.address.set({...HOME, mode: 'Debug'});
    page.fixture.detectChanges();

    const reading = page.fixture.nativeElement.querySelector('.reading') as HTMLElement;
    reading.innerHTML = `<div class="virtualTextHost" data-v-kind="output">
      <button data-turn="1" data-step="0" data-side="unknown">unknown</button>
      <button data-turn="1" data-step="0" data-side="target">world</button>
    </div>`;
    const unknownButton = reading.querySelector<HTMLButtonElement>('button[data-side="unknown"]');
    const targetButton = reading.querySelector<HTMLButtonElement>('button[data-side="target"]');

    expect(() => unknownButton?.click()).not.toThrow();
    expect(page.tokens.selection()).toBeNull();

    targetButton?.click();
    expect(page.tokens.selection()).toEqual({kind: 'token', turn: 1, index: 0});

    page.tokens.selection.set(null);
    page.workspace.address.set({...HOME, mode: 'Chat'});
    page.fixture.detectChanges();
    const chatReading = page.fixture.nativeElement.querySelector('.reading') as HTMLElement;
    chatReading.innerHTML = `<div class="virtualTextHost" data-v-kind="output">
      <button data-turn="1" data-step="0" data-side="target">world</button>
    </div>`;
    chatReading.querySelector<HTMLButtonElement>('button[data-side="target"]')?.click();
    expect(page.tokens.selection()).toBeNull();
    page.fixture.destroy();
  });
});
