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

import {signal} from '@angular/core';
import {debugParts} from '../testing/debug_parts';

/** A read turn whose stages start where the server says, the same on both sides. */
function turnAt(n: number, prefill: number, decode: number) {
  const stage = (start: number) => ({
    positions: {ref: {start, end: start + 1}, target: {start, end: start + 1}},
  });
  return {n, stages: {prefill: stage(prefill), decode: stage(decode)}};
}

describe('Stage Info token scope', () => {
  function setup() {
    const reference = [
      {step: 0, text: '<bos>', kind: 'special'},
      {step: 1, text: '<turn>', kind: 'template'},
      {step: 2, text: 'hello'},
    ];
    const target = [...reference.slice(0, 2), {step: 2, text: 'extra'}, {step: 3, text: 'hello'}];
    const pairs = [
      {index: 0, ref: reference[0], target: target[0], match: true},
      {index: 1, ref: reference[1], target: target[1], match: true},
      {index: 2, target: target[2], match: false},
      {index: 3, ref: reference[2], target: target[3], match: true},
    ];
    const selection = signal<{turn: number; stage: string; range?: {start: number; end: number}}>({
      turn: 1,
      stage: 'prefill',
    });
    const captures = new Map([
      ['ref', {input_tokens: reference, tokens: reference}],
      ['target', {input_tokens: target, tokens: target}],
    ]);
    const {detailsInfo: info} = debugParts({
      conversation: {
        alignment: () => 'content',
        turns: () => [turnAt(1, 0, 4)],
        captured: (_turn: number, side: string) => captures.get(side),
      },
      tokens: {
        selectedStage: selection,
        selection: () => ({kind: 'stage', ...selection()}),
        inputPairRows: () => [{leading: [], text: pairs, trailing: []}],
        rows: () => pairs,
      },
    });
    return {info, selection, captures, pairs};
  }

  it('counts special and template tokens but excludes aligned gaps', () => {
    const {info, selection} = setup();
    expect(info.stagePositions(1, 'prefill')).toBe('0–3');
    expect(info.stageInfoCount(1, 'ref', 'prefill')).toBe(3);
    expect(info.stageInfoCount(1, 'target', 'prefill')).toBe(4);
    expect(info.matchRateRequest()).toEqual({turn: 1, stage: 'prefill', alignment: 'content'});
    selection.set({turn: 1, stage: 'prefill', range: {start: 0, end: 2}});
    expect(info.stageInfoCount(1, 'ref', 'prefill')).toBe(2);
    expect(info.matchRateRequest()?.rows).toEqual([0, 1]);
    selection.set({turn: 1, stage: 'prefill', range: {start: 2, end: 3}});
    expect(info.stagePositions(1, 'prefill')).toBe('2');
    expect(info.stageInfoCount(1, 'ref', 'prefill')).toBe(0);
    expect(info.stageInfoCount(1, 'target', 'prefill')).toBe(1);
    expect(info.matchRateRequest()?.rows).toEqual([2]);
  });

  it('uses aligned range positions instead of each runtime token step', () => {
    const {info, selection} = setup();
    selection.set({turn: 1, stage: 'prefill', range: {start: 3, end: 4}});
    expect(info.stageInfoCount(1, 'ref', 'prefill')).toBe(1);
    expect(info.stageInfoCount(1, 'target', 'prefill')).toBe(1);
    expect(info.matchRateRequest()?.rows).toEqual([3]);
    selection.set({turn: 1, stage: 'decode', range: {start: 7, end: 8}});
    expect(info.stageInfoCount(1, 'ref', 'decode')).toBe(1);
    expect(info.matchRateRequest()).toEqual({
      turn: 1,
      stage: 'decode',
      alignment: 'content',
      rows: [3],
    });
  });

  it("shows the server's match rate of the rows it asked for, and a dash otherwise", () => {
    const {info, selection, captures} = setup();
    selection.set({turn: 1, stage: 'prefill', range: {start: 10, end: 11}});
    expect(info.matchRateRequest()?.rows).toEqual([]);
    captures.delete('target');
    expect(info.stageInfoCount(1, 'target', 'prefill')).toBe('Not captured');
    const query = {turn: 1, stage: 'prefill', alignment: 'content', rows: [0, 1]} as const;
    const answer = signal<{match_rate: number | null} | null>(null);
    Object.assign(info, {
      matchRateQuery: () => query,
      matchRate: {data: () => answer() && {...query, ...answer()}},
    });
    expect(info.stageTokenMatchRate(1, 'prefill')).toBe('-');
    answer.set({match_rate: 0.75});
    expect(info.stageTokenMatchRate(1, 'prefill')).toBe('75%');
    expect(info.stageTokenMatchRate(1, 'decode')).toBe('-');
    answer.set({match_rate: null});
    expect(info.stageTokenMatchRate(1, 'prefill')).toBe('-');
  });
});
