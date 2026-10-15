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

import {summarizeStageTokens, type StageTokenInput} from './stage_token_summary';

function stage(
  turn: number,
  phase: 'prefill' | 'decode',
  matches: (boolean | null)[],
  complete = true,
): StageTokenInput {
  return {turn, stage: phase, complete, pairs: matches.map((match, index) => ({index, match}))};
}

describe('conversation stage token summaries', () => {
  it('weights cumulative rates by positions and marks only the first divergent stage', () => {
    const result = summarizeStageTokens([
      stage(1, 'prefill', [true, true, true]),
      stage(1, 'decode', [false]),
      stage(2, 'prefill', [true, true]),
      stage(2, 'decode', [false, true]),
    ]);
    expect(result.map((r) => r.historyMatch)).toEqual([null, true, false, false]);
    expect(result.map((r) => r.matchRate)).toEqual([1, 0, 1, 0.5]);
    expect(result.map((r) => r.cumulativeMatchRate)).toEqual([1, 0.75, 5 / 6, 6 / 8]);
    expect(result.map((r) => r.firstDivergence)).toEqual([false, true, false, false]);
  });

  it('includes one-sided gaps in the denominator', () => {
    const input = stage(1, 'decode', [true, false]);
    input.pairs = [
      {index: 0, ref: {step: 0, text: '<bos>'}, target: {step: 0, text: '<bos>'}, match: true},
      {index: 1, target: {step: 1, text: 'extra'}, match: false},
    ];
    expect(summarizeStageTokens([input])[0].matchRate).toBe(0.5);
  });

  it('does not report full rates or matching history when earlier data is unknown', () => {
    const result = summarizeStageTokens([
      stage(1, 'prefill', [], false),
      stage(1, 'decode', [null, true]),
      stage(2, 'prefill', [true]),
      stage(2, 'decode', [false]),
    ]);
    expect(result.map((r) => r.historyMatch)).toEqual([null, null, null, null]);
    expect(result.map((r) => r.matchRate)).toEqual([null, null, 1, 0]);
    expect(result.every((r) => r.cumulativeMatchRate === null)).toBe(true);
    expect(result.map((r) => r.firstDivergence)).toEqual([null, null, false, true]);
  });

  it('reports empty stages without dividing by zero', () => {
    const result = summarizeStageTokens([stage(1, 'prefill', []), stage(1, 'decode', [true])]);
    expect(result[0].matchRate).toBeNull();
    expect(result[0].cumulativeMatchRate).toBeNull();
    expect(result[1].historyMatch).toBeNull();
    expect(result[1].cumulativeMatchRate).toBe(1);
  });
});
