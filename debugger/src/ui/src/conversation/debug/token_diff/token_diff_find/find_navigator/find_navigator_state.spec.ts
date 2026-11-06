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

import {findNavigatorState} from './find_navigator_state';

describe('findNavigatorState', () => {
  it('no results disables everything and shows a dash', () => {
    expect(findNavigatorState({index: -1, total: 0})).toStrictEqual({
      firstDisabled: true,
      previousDisabled: true,
      nextDisabled: true,
      label: '— / 0',
    });
  });

  it('first, middle and last results enable the right directions', () => {
    expect(findNavigatorState({index: 0, total: 3})).toStrictEqual({
      firstDisabled: true,
      previousDisabled: true,
      nextDisabled: false,
      label: '1 / 3',
    });
    expect(findNavigatorState({index: 1, total: 3})).toStrictEqual({
      firstDisabled: false,
      previousDisabled: false,
      nextDisabled: false,
      label: '2 / 3',
    });
    expect(findNavigatorState({index: 2, total: 3})).toStrictEqual({
      firstDisabled: false,
      previousDisabled: false,
      nextDisabled: true,
      label: '3 / 3',
    });
  });

  it('busy and pending block navigation; pending hides the count', () => {
    expect(findNavigatorState({index: 1, total: 3, busy: true}).nextDisabled).toBe(true);
    expect(findNavigatorState({index: 1, total: 3, busy: true}).label).toBe('2 / 3');
    const pending = findNavigatorState({index: -1, total: 0, pending: true});
    expect(pending.label).toBe('… / …');
    expect(pending.firstDisabled).toBe(true);
  });
});
