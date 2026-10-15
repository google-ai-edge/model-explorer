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

import {positionPairs, tokenPosition} from './token_alignment';

describe('token positions', () => {
  it('reads each side at its own position when the histories differ', () => {
    // The Target has a system prompt the Reference lacks: its reply starts 8 later.
    const pairs = [
      {
        index: 0,
        ref: {text: 'A', step: 0, position: 17},
        target: {text: 'A', step: 0, position: 25},
        match: true,
      },
      {index: 1, target: {text: 'B', step: 1, position: 26}, match: false},
    ];
    expect(positionPairs(pairs, 17)).toEqual([
      [17, 25],
      [null, 26],
    ]);
  });

  it('counts from the turn origin when no position was captured', () => {
    expect(tokenPosition({text: 'A', step: 3}, 17)).toBe(20);
  });
});
