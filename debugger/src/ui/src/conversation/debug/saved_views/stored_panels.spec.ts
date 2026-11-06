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

import type {StageAnalysisPanel} from '../../stage_analysis';
import {
  STORED_PANELS_KEY,
  STORED_PANELS_SESSIONS,
  readStoredPanels,
  storePanels,
  type PanelStorage,
} from './stored_panels';

function storage(): PanelStorage & {kept: Map<string, string>} {
  const kept = new Map<string, string>();
  return {kept, read: (key) => kept.get(key) ?? null, save: (key, value) => kept.set(key, value)};
}
const KV: StageAnalysisPanel = {
  id: 'analysis-1',
  turn: 1,
  stage: 'prefill',
  kind: 'kv',
  open: true,
};

describe('stored analysis panels', () => {
  it('are each Session’s own', () => {
    const browser = storage();
    storePanels(browser, 'a', [KV]);
    storePanels(browser, 'b', []);
    expect(readStoredPanels(browser, 'a')).toEqual([KV]);
    expect(readStoredPanels(browser, 'b')).toEqual([]);
    expect(readStoredPanels(browser, 'c')).toBeNull();
    expect(readStoredPanels(browser, '')).toBeNull();
  });

  it('keep only the valid panels of what the browser holds', () => {
    const browser = storage();
    browser.save(
      STORED_PANELS_KEY,
      JSON.stringify({a: {at: 1, panels: [KV, {id: 'nope'}]}, b: {at: 2, panels: 'x'}}),
    );
    expect(readStoredPanels(browser, 'a')).toEqual([KV]);
    expect(readStoredPanels(browser, 'b')).toBeNull();
    browser.save(STORED_PANELS_KEY, '{not json');
    expect(readStoredPanels(browser, 'a')).toBeNull();
  });

  it('keep the most recently changed Sessions, up to their number', () => {
    const browser = storage();
    for (let k = 0; k <= STORED_PANELS_SESSIONS; k++) storePanels(browser, 's' + k, [KV], k);
    expect(readStoredPanels(browser, 's0')).toBeNull();
    expect(readStoredPanels(browser, 's' + STORED_PANELS_SESSIONS)).toEqual([KV]);
  });
});
