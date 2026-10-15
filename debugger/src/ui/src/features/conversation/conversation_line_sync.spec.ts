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

import {syncPairWidths} from './conversation_line_sync';

describe('paired token geometry', () => {
  it('normalizes fractional widths on both runtimes and remeasures wrapped height', () => {
    const root = document.createElement('div');
    root.innerHTML = `<div class="tokens"><button data-row="0" data-turn="1" data-side="ref"></button>
      <button data-row="0" data-turn="1" data-side="target"></button></div>`;
    document.body.append(root);
    const [ref, target] = Array.from(root.querySelectorAll('button'));
    for (const token of [ref, target]) token.style.marginBottom = '1px';
    // Model the CSS margin (the synchronizer clears its previous inline compensation).
    const style = document.createElement('style');
    style.textContent = '.tokens button {margin-bottom: 1px}';
    root.append(style);
    ref.getBoundingClientRect = () =>
      ({width: ref.style.width ? 31 : 30.2, height: ref.style.width ? 26 : 52}) as DOMRect;
    target.getBoundingClientRect = () =>
      ({width: target.style.width ? 31 : 30.4, height: 52}) as DOMRect;
    try {
      syncPairWidths(root);
      expect(ref.style.width).toBe('31px');
      expect(target.style.width).toBe('31px');
      expect(ref.style.marginBottom).toBe('27px');
      syncPairWidths(root);
      expect(ref.style.width).toBe('31px');
      expect(ref.style.marginBottom).toBe('27px');
    } finally {
      root.remove();
    }
  });
  it('does not constrain an unpaired or collapsed token', () => {
    const root = document.createElement('div');
    root.innerHTML =
      '<div class="tokens"><button data-row="0" data-turn="1" data-side="ref"></button></div>';
    syncPairWidths(root);
    expect(root.querySelector('button')!.style.width).toBe('');
  });
});
