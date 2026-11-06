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

import type {
  CapturedToken,
  ComparisonRow,
  FirstDivergence,
  InputMessage,
} from '../server/contracts/capture';
import {comparisonPairs, forkPair, messageRows} from './token_alignment';

describe('token alignment', () => {
  // The server aligns (analysis/alignment.py); the UI only reads its rows as pairs of tokens.
  const tokens = (words: string[]): CapturedToken[] =>
    words.map((text, step) => ({text, step, position: 40 + step}) as CapturedToken);
  const ref = tokens(['a', 'b', 'c']);
  const target = tokens(['a', 'x', 'b', 'c']);
  const rows: ComparisonRow[] = [
    {ref: 0, target: 0, same: true},
    {ref: null, target: 1, same: false},
    {ref: 1, target: 2, same: true},
    {ref: 2, target: 3, same: null},
  ];

  it('server rows become pairs of the captured tokens, gaps stay empty, rows are cached', () => {
    const pairs = comparisonPairs(rows, ref, target);
    expect(pairs.map((p) => [p.index, p.ref?.text, p.target?.text, p.match])).toStrictEqual([
      [0, 'a', 'a', true],
      [1, undefined, 'x', false],
      [2, 'b', 'b', true],
      [3, 'c', 'c', null],
    ]);
    expect(pairs[2].ref, 'pairs hold the captured tokens themselves').toBe(ref[1]);
    expect('ref' in pairs[1], 'a gap has no token on its side').toBe(false);
    expect(comparisonPairs(rows, ref, target), 'one rows array, one list of pairs').toBe(pairs);
    expect(comparisonPairs(undefined, ref, target)).toStrictEqual([]);
  });

  it('the first differing step pairs the tokens the server names', () => {
    // The server's first divergence: a content row that leaves one side empty is filled with that
    // side's token of the same step.
    const pairs = comparisonPairs(rows, ref, target);
    const divergence = (fields: object) => fields as FirstDivergence;
    const fork = forkPair(
      divergence({row: 1, ref: 1, target: 1, filled: 'ref', partner: 2}),
      pairs,
      ref,
      target,
    )!;
    expect([
      fork.index,
      fork.filled,
      fork.partner,
      fork.pair.ref!.text,
      fork.pair.target!.text,
    ]).toStrictEqual([1, 'ref', 2, 'b', 'x']);
    expect(fork.pair.match).toBe(false);
    const swap = forkPair(
      divergence({row: 1, ref: 1, target: 1, filled: null, partner: null}),
      pairs,
      ref,
      target,
    )!;
    expect(swap.pair, 'a row that holds both tokens is the pair as it stands').toBe(pairs[1]);
    expect(forkPair(null, pairs, ref, target)).toBe(null);
  });

  it("the server's messages group the items they index", () => {
    const input = tokens(['<bos>', '<|turn>', 'user', 'Hi']);
    const messages = messageRows(
      [{role: 'user', leading: [0, 1, 2], text: [3], trailing: []}],
      input,
    )!;
    expect(
      messages.map((m) => [m.role, m.leading.map((t) => t.text), m.text.map((t) => t.text)]),
    ).toStrictEqual([['user', ['<bos>', '<|turn>', 'user'], ['Hi']]]);
    const grouped: InputMessage[] = [{role: 'user', leading: [0, 1, 2], text: [3], trailing: []}];
    expect(
      messageRows(grouped, input),
      'one messages array over one items array is grouped once',
    ).toBe(messageRows(grouped, input));
    expect(messageRows(grouped, input)).not.toBe(messageRows(grouped, [...input]));
    expect(messageRows([], input)).toBe(null);
    expect(messageRows(undefined, input)).toBe(null);
  });
});
