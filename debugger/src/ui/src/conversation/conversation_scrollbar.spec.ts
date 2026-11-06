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

import {createConversationScrollbar} from './conversation_scrollbar';

describe('the conversation scrollbar', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('follows only the current reading as the page switches between Chat and Debug', () => {
    const observed = new Set<Element>();
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe(target: Element) {
          observed.add(target);
        }
        unobserve(target: Element) {
          observed.delete(target);
        }
        disconnect() {
          observed.clear();
        }
      },
    );
    vi.stubGlobal('requestAnimationFrame', () => 0);
    const root = document.createElement('div');
    root.innerHTML = '<div class="composer"></div>';
    document.body.append(root);
    /** The page shows a new reading element, as switching Chat and Debug does. */
    const showReading = () => {
      root.querySelector('.reading')?.remove();
      const reading = document.createElement('div');
      reading.className = 'reading';
      reading.innerHTML = '<div class="turns"></div>';
      root.prepend(reading);
      return reading;
    };
    const scrollbar = createConversationScrollbar(root, () => true);
    const bar = root.querySelector('.conversationScrollbar')!;
    const readings: HTMLElement[] = [];
    for (let i = 0; i < 6; i++) {
      readings.push(showReading());
      scrollbar.mount();
      scrollbar.mount();
      const current = readings[readings.length - 1];
      expect([...observed]).toEqual(
        expect.arrayContaining([root, current, current.querySelector('.turns')!]),
      );
      expect(observed.size).toBe(3);
    }
    // A reading the page no longer shows moves the scrollbar no more; the current one does.
    for (const old of readings.slice(0, -1)) old.dispatchEvent(new Event('scroll'));
    expect(bar.classList.contains('visible')).toBe(false);
    readings[readings.length - 1].dispatchEvent(new Event('scroll'));
    expect(bar.classList.contains('visible')).toBe(true);
    scrollbar.destroy();
    expect(observed.size).toBe(0);
    root.remove();
  });
});
