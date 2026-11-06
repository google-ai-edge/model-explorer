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

import {mountEmptyChatDemo} from './empty_chat_demo';

describe('empty chat demo', () => {
  let root: HTMLElement;
  let demo: HTMLElement;
  let toolbar: HTMLElement;

  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0);
      return 0;
    });
    if (!document.fonts) {
      Object.defineProperty(document, 'fonts', {
        value: {ready: new Promise(() => {})},
        configurable: true,
      });
    }
    // The workspace mode switch is in the app's toolbar, outside the panel.
    toolbar = document.createElement('div');
    root = document.createElement('div');
    root.innerHTML = `<div class="composer"><textarea></textarea></div>`;
    demo = document.createElement('section');
    demo.innerHTML = '<span class="demoDraftText"></span>';
    root.append(demo);
    document.body.append(toolbar, root);
  });

  afterEach(() => {
    toolbar.remove();
    root.remove();
    vi.unstubAllGlobals();
  });

  function withdrawn(): boolean {
    return demo.classList.contains('withdrawn');
  }

  it('mounts in a conversation whose mode switch is in the toolbar', () => {
    const dispose = mountEmptyChatDemo(root, demo);
    const input = root.querySelector('textarea')!;
    input.focus();
    input.dispatchEvent(new Event('focus'));
    expect(withdrawn()).toBe(true);
    input.blur();
    dispose();
    expect(root.classList.contains('demoRunning')).toBe(false);
  });

  it('hides while the toolbar mode switch has the focus or a click', () => {
    const dispose = mountEmptyChatDemo(root, demo);
    expect(withdrawn()).toBe(false);
    // Shown on the selected record after the panel mounted.
    toolbar.innerHTML = `<div class="mode-switch"><button data-mode="chat"></button></div>`;
    const button = toolbar.querySelector('button')!;
    button.focus();
    expect(withdrawn()).toBe(true);
    button.blur();
    expect(withdrawn()).toBe(false);
    button.focus();
    button.click();
    expect(withdrawn()).toBe(true);
    button.blur();
    // Focus and clicks elsewhere in the document leave the demo shown.
    toolbar.insertAdjacentHTML('beforeend', '<button class="other"></button>');
    const other = toolbar.querySelector<HTMLElement>('.other')!;
    other.focus();
    other.click();
    expect(withdrawn()).toBe(false);
    other.blur();
    dispose();
    button.focus();
    expect(withdrawn()).toBe(false);
    button.blur();
  });

  it('does not throw during resize if the composer element is removed', () => {
    const dispose = mountEmptyChatDemo(root, demo);
    root.querySelector('.composer')?.remove();
    expect(() => window.dispatchEvent(new Event('resize'))).not.toThrow();
    dispose();
  });
});
