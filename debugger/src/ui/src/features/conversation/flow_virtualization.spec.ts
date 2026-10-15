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

import {createFlowVirtualizer, type FlowConfig} from './flow_virtualization';

describe('virtual token clicks', () => {
  it('selects both real tokens and alignment gaps without jumping the reading position', () => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    const reading = document.createElement('div');
    reading.innerHTML = `<div class="virtualTextHost" data-v-kind="output" data-v-side="target"
      data-v-turn="2" data-v-stage="Output">
      <button data-turn="2" data-step="0" data-side="target">token</button>
      <button data-v-gap="0" data-v-turn="2" data-v-existing-step="0" data-v-existing-side="target">∅</button>
    </div>`;
    document.body.append(reading);
    reading.scrollTop = 100;
    const rows = [
      {index: 0, ref: 0, target: 0},
      {index: 1, ref: 1, target: 1},
    ];
    const onSelect = vi.fn();
    const config: FlowConfig = {
      threshold: 1,
      getScroller: () => reading,
      getRows: () => rows,
      getTurn: () => ({
        steps: {ref: ['a', 'b'], target: ['a', 'b']},
        inputTokens: {ref: [], target: []},
        alignments: {content: rows},
      }),
      getAlignment: () => 'content',
      isDebug: () => true,
      phase: () => 'Output',
      findRow: () => rows[0],
      getSelectedRow: () => null,
      renderOutputToken: () => '',
      renderInputToken: () => '',
      displayKey: () => '',
      styleKey: () => '',
      onSelect,
    };
    const virtualizer = createFlowVirtualizer(config);
    virtualizer.mount();
    const host = reading.firstElementChild as HTMLElement & {_vLayout: unknown};
    host._vLayout = {offsetByRow: new Map([[0, 0]]), lineByOffset: [5]};
    host.getBoundingClientRect = () => ({top: 300}) as DOMRect;
    try {
      for (const button of Array.from(reading.querySelectorAll('button'))) {
        button.click();
        expect(reading.scrollTop).toBe(100);
      }
      expect(onSelect).toHaveBeenCalledTimes(2);
      expect(onSelect).toHaveBeenCalledWith(2, 0, 'target');
    } finally {
      virtualizer.destroy();
      reading.remove();
      vi.unstubAllGlobals();
    }
  });
});
