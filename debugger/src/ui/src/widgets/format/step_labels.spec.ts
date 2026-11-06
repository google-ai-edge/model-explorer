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

import {stepRows} from './step_labels';

describe('stepRows', () => {
  it('both sides that say the same step are one row', () => {
    const step = 'Decode step 20 · reads "," · produces "blue"';
    expect(stepRows({ref: step, target: step})).toStrictEqual([{label: 'Step', value: step}]);
  });

  it('sides that differ, or one side alone, are a row each', () => {
    expect(stepRows({ref: 'Decode step 20 · a', target: 'Decode step 22 · b'})).toStrictEqual([
      {label: 'Reference step', value: 'Decode step 20 · a'},
      {label: 'Target step', value: 'Decode step 22 · b'},
    ]);
    expect(stepRows({ref: null, target: 'Written at decode step 3, when x'})).toStrictEqual([
      {label: 'Target step', value: 'Written at decode step 3, when x'},
    ]);
    expect(stepRows(null)).toStrictEqual([]);
  });

  it("a column's context, where its sides read different tokens, follows its steps", () => {
    const context = 'After the divergence at position 17 the two sides read different tokens';
    expect(stepRows({ref: 'a', target: 'a', context})).toStrictEqual([
      {label: 'Step', value: 'a'},
      {label: 'Context', value: context},
    ]);
    expect(stepRows({ref: 'a', target: 'b', context})).toStrictEqual([
      {label: 'Reference step', value: 'a'},
      {label: 'Target step', value: 'b'},
      {label: 'Context', value: context},
    ]);
  });
});
