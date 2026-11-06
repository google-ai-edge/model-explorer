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

import {ConversationReadingController} from './conversation_reading_controller';
import type {ConversationMode, ConversationReadingPosition} from './conversation_view_state';
import type {FlowAnchor, FlowVirtualizer} from './flow_virtualization';

describe('ConversationReadingController', () => {
  // Animation frames run when a test flushes them.
  const frames: FrameRequestCallback[] = [];
  beforeEach(() => {
    frames.length = 0;
    vi.stubGlobal('requestAnimationFrame', (work: FrameRequestCallback) => frames.push(work));
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });
  afterEach(() => vi.unstubAllGlobals());

  function fixture() {
    let mode: ConversationMode = 'Chat';
    let anchor: FlowAnchor | null = null;
    const tasks: (() => void)[] = [];
    const events: (string | [string, FlowAnchor | null])[] = [];
    const positions = new Map<string, ConversationReadingPosition>();
    const listeners = new Set<() => void>();
    const reading = {
      scrollTop: 0,
      isConnected: true,
      addEventListener: (type: string, listener: () => void) =>
        type === 'scroll' && listeners.add(listener),
      removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
      scroll: () => listeners.forEach((listener) => listener()),
      getBoundingClientRect: () => ({top: 0, bottom: 800, left: 0, right: 1000}),
      querySelector: (selector: string) =>
        selector.includes('data-node-turn="2"') || selector === '.turn[data-turn="2"]'
          ? {getBoundingClientRect: () => ({top: 1400 - reading.scrollTop})}
          : null,
    };
    const root = {querySelector: (selector: string) => (selector === '.reading' ? reading : null)};
    const virtualizer = {
      capture: () => {
        events.push('capture');
        return structuredClone(anchor);
      },
      mount: () => events.push('mount'),
      paint: () => events.push('paint'),
      restore: (next: FlowAnchor | null) => {
        events.push(['restore', structuredClone(next)]);
        anchor = structuredClone(next);
      },
      destroy: () => {
        events.push('destroy');
        anchor = null;
      },
    };
    const controller = new ConversationReadingController({
      context: 'parent:child',
      positions,
      root: () => root as unknown as HTMLElement,
      mode: () => mode,
      virtualizer: () => virtualizer as unknown as FlowVirtualizer,
      afterRender: (work: () => void) => tasks.push(work),
      afterRestore: () => events.push('restored'),
    });
    return {
      controller,
      reading,
      positions,
      events,
      tasks,
      setMode: (next: ConversationMode) => (mode = next),
      setAnchor: (next: FlowAnchor | null) => (anchor = structuredClone(next)),
      flush: () => {
        for (const task of tasks.splice(0)) task();
      },
      flushFrames: () => {
        for (const frame of frames.splice(0)) frame(0);
      },
      listeners,
    };
  }

  const chatAnchor: FlowAnchor = {kind: 'input', ti: 1, position: 40, stage: 'Prefill', offset: 25};
  const debugAnchor: FlowAnchor = {
    kind: 'token',
    ti: 1,
    side: 'target',
    step: 2049,
    stage: 'Output',
    offset: 70,
  };

  it('controller restores each rendered mode offset and semantic anchor independently', () => {
    const f = fixture();
    f.controller.syncMode();
    f.flush();
    f.reading.scrollTop = 300;
    f.setAnchor(chatAnchor);
    f.setMode('Debug');
    f.controller.syncMode();
    f.flush();
    expect(f.reading.scrollTop).toBe(0);
    expect(
      f.events.some((event) => Array.isArray(event)),
      'first visit cannot restore outgoing anchor',
    ).toBe(false);
    f.reading.scrollTop = 9000;
    f.setAnchor(debugAnchor);
    f.setMode('Chat');
    f.controller.syncMode();
    f.flush();
    expect(f.reading.scrollTop).toBe(300);
    expect(f.events.filter(Array.isArray).at(-1)![1]).toStrictEqual(chatAnchor);
    f.setMode('Debug');
    f.controller.syncMode();
    f.flush();
    expect(f.reading.scrollTop).toBe(9000);
    expect(f.events.filter(Array.isArray).at(-1)![1]).toStrictEqual(debugAnchor);
  });

  it('a rapid mode round-trip cancels obsolete callbacks without assigning old DOM geometry to a new mode', () => {
    const f = fixture();
    f.controller.syncMode();
    f.flush();
    f.reading.scrollTop = 350;
    f.setAnchor(chatAnchor);
    f.setMode('Debug');
    f.controller.syncMode();
    f.setMode('Chat');
    f.controller.syncMode();
    f.flush();
    expect(f.reading.scrollTop).toBe(350);
    expect(f.events.filter(Array.isArray).at(-1)![1]).toStrictEqual(chatAnchor);
    expect(
      f.events.filter((event) => event === 'restored').length,
      'initial layout and newest callback only',
    ).toBe(2);
  });

  it('shared Turn/phase return locates the current stage without replaying the old token anchor', () => {
    const f = fixture();
    f.setMode('Debug');
    f.controller.apply({mode: 'Debug', scrollTop: 9000, anchor: debugAnchor}, false, {
      turn: 2,
      phase: 'prefill',
    });
    f.controller.syncMode();
    f.flush();
    expect(f.reading.scrollTop).toBe(1392);
    expect(
      f.events.some(Array.isArray),
      'cross-Turn return must not restore the stale token anchor',
    ).toBe(false);
  });

  it('disposal saves final semantic location before virtualizer destruction and suppresses delayed restoration', () => {
    const f = fixture();
    f.controller.syncMode();
    f.flush();
    f.reading.scrollTop = 500;
    f.setAnchor(chatAnchor);
    f.controller.apply({mode: 'Chat', scrollTop: 9000, anchor: debugAnchor}, true, null);
    f.controller.destroy();
    const count = f.events.length;
    f.flush();
    expect(f.events.length, 'destroyed controller must reject pending restore').toBe(count);
    expect(f.events.at(-2)).toBe('capture');
    expect(f.events.at(-1)).toBe('destroy');
    expect(f.controller.capture()).toStrictEqual({
      mode: 'Chat',
      scrollTop: 500,
      anchor: chatAnchor,
    });
    const stored = [...f.positions.values()];
    expect(
      stored.some(
        (value) => value.scrollTop === 500 && (value.anchor as {position: number}).position === 40,
      ),
    ).toBeTruthy();
  });

  it('leaving the page keeps the position the reading was last laid out at, not the detached zero', () => {
    const f = fixture();
    f.setMode('Debug');
    f.controller.syncMode();
    f.flush();
    f.reading.scrollTop = 918;
    f.setAnchor(debugAnchor);
    f.reading.scroll();
    f.flushFrames();
    // Angular detaches the page's elements before it destroys the page: the reading then has no
    // layout, no scroll position and no anchor.
    f.reading.isConnected = false;
    f.reading.scrollTop = 0;
    f.setAnchor(null);
    f.controller.destroy();
    expect(f.listeners.size, 'destroyed controller stops following the reading').toBe(0);
    expect(f.controller.capture()).toStrictEqual({
      mode: 'Debug',
      scrollTop: 918,
      anchor: debugAnchor,
    });
    expect([...f.positions.values()].at(-1)).toStrictEqual({scrollTop: 918, anchor: debugAnchor});
  });
});
