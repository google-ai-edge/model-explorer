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

import type {CapturedToken, Session} from '../../server/contracts/capture';
import {columnSteps, generatedSteps} from './column_steps';

const token = (position: number, step: number, side: string): CapturedToken =>
  ({
    text: 'x',
    step,
    position,
    produced_text: `${side} produced ${position}`,
    written_text: `${side} wrote ${position}`,
  }) as CapturedToken;
const session = {
  model: 'm',
  runs: [],
  turns: [],
  batches: [],
  conversation: (['ref', 'target'] as const).map((run) => ({
    turn: 2,
    run,
    input_tokens: [token(10, 0, run), token(11, 1, run)],
    // The target's history is one token longer.
    tokens: [token(run === 'ref' ? 12 : 13, 0, run), token(run === 'ref' ? 13 : 14, 1, run)],
  })),
} as unknown as Session;

describe('column steps', () => {
  it('a column without an alignment is its position on both sides, input or output', () => {
    expect(columnSteps(session, 2, 11, null, 'produced')).toStrictEqual({
      ref: 'ref produced 11',
      target: 'target produced 11',
    });
    expect(columnSteps(session, 2, 12, null, 'written')).toStrictEqual({
      ref: 'ref wrote 12',
      target: null,
    });
  });

  it("an aligned column reads each side's token at its own position", () => {
    const alignment = {
      start: 12,
      pairs: [
        [12, 13],
        [13, null],
      ] as [number | null, number | null][],
    };
    expect(columnSteps(session, 2, 12, alignment, 'produced')).toStrictEqual({
      ref: 'ref produced 12',
      target: 'target produced 13',
    });
    expect(columnSteps(session, 2, 13, alignment, 'produced')).toStrictEqual({
      ref: 'ref produced 13',
      target: null,
    });
    expect(columnSteps(session, 2, 99, alignment, 'produced')).toStrictEqual({
      ref: null,
      target: null,
    });
    expect(columnSteps(session, null, 12, alignment, 'produced')).toBe(null);
  });

  it("after the divergence, a two-sided column says why the analysis's sides still compare", () => {
    const reason = {
      code: 'context_different',
      side: null,
      text: 'After the divergence at position 12 the two sides read different tokens; the K and V still compare.',
    };
    const alignment = {
      start: 12,
      pairs: [
        [12, 12],
        [13, 14],
        [null, 15],
      ] as [number | null, number | null][],
      columns: {kv: {start: 12, end: 15, omitted: [], notes: [], context: {start: 13, reason}}},
    };
    // Before the context's first column, and in an analysis without one: no context.
    expect(columnSteps(session, 2, 12, alignment, 'written', 'kv')?.context).toBeUndefined();
    expect(
      columnSteps(session, 2, 13, alignment, 'produced', 'intermediate_tensors')?.context,
    ).toBeUndefined();
    expect(columnSteps(session, 2, 13, alignment, 'written', 'kv')).toStrictEqual({
      ref: 'ref wrote 13',
      target: 'target wrote 14',
      context: reason.text,
    });
    // A column of one side alone compares nothing.
    expect(columnSteps(session, 2, 14, alignment, 'written', 'kv')?.context).toBeUndefined();
    expect(generatedSteps(session, 2, 1, 1, reason.text)?.context).toBe(reason.text);
  });

  it("a Final Logits row's steps are its generated tokens'", () => {
    expect(generatedSteps(session, 2, 1, 0)).toStrictEqual({
      ref: 'ref produced 13',
      target: 'target produced 13',
    });
    expect(generatedSteps(session, 2, null, 1)).toStrictEqual({
      ref: null,
      target: 'target produced 14',
    });
  });
});
