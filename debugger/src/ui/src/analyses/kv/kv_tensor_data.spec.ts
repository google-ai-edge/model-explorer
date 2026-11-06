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

import {defaultKvView, parseKvView, restoreKvView} from './kv_view_state';
import type {KvAnalysis} from '../../server/contracts/kv';
import {readTensorHeads} from './kv_tensor_data';
import type {KvApi} from '../../server/kv_api';
import type {KvHeadEvidence, KvHeadsEvidence} from '../../server/contracts/kv';
const selection = {
  turn: 2,
  contextId: 'snapshot',
  kind: 'key',
  layer: 3,
  position: 42,
  alignment: {
    stage: 'prefill' as const,
    alignment: 'content' as const,
    start: 42,
    pairs: [[null, 40]] as [number | null, number | null][],
  },
};
const answer = (heads: number[], fields: Partial<KvHeadsEvidence['selection']> = {}) =>
  ({
    selection: {
      turn: 2,
      context_id: 'snapshot',
      kind: 'key',
      layer: 3,
      position: 42,
      batch: null,
      ...fields,
    },
    heads: heads.map((head) => ({head}) as KvHeadEvidence),
  }) as KvHeadsEvidence;
describe('tensor head source loading', () => {
  it('reads every head in one request and forwards the aligned source identity', async () => {
    const calls: unknown[][] = [];
    const api = {
      kvHead: async (...args: unknown[]) => {
        calls.push(args);
        return answer([0, 1, 2]);
      },
    } as Pick<KvApi, 'kvHead'>;
    const rows = await readTensorHeads(api, 'capture', selection, new AbortController().signal);
    expect(calls).toHaveLength(1);
    expect(calls[0].slice(0, 7)).toEqual(['capture', 2, 'snapshot', 3, 'key', 42, null]);
    expect(calls[0][8]).toBe(selection.alignment);
    expect(rows.map((row) => [row.label, row.head, row.evidence?.head, row.error])).toEqual([
      ['Head 0', 0, 0, ''],
      ['Head 1', 1, 1, ''],
      ['Head 2', 2, 2, ''],
    ]);
  });
  it('refuses an answer of another source or heads out of order', async () => {
    for (const value of [answer([0], {position: 41}), answer([1, 0])]) {
      const api = {kvHead: async () => value} as Pick<KvApi, 'kvHead'>;
      await expect(
        readTensorHeads(api, 'capture', selection, new AbortController().signal),
      ).rejects.toThrow('does not match');
    }
  });
});

describe('tensor view persistence', () => {
  const saved = {
    ...defaultKvView(),
    contextId: 'snapshot',
    tensor: {
      active: true,
      plot: 'scatter' as const,
      position: 42,
      layer: 3,
      head: 1,
    },
  };
  const analysis = {
    contexts: [
      {
        id: 'snapshot',
        turn: 2,
        layers: [{kind: 'key', layer: 3, head_count: 4, position_start: 0, position_count: 64}],
      },
    ],
  } as unknown as KvAnalysis;
  it('retains valid tensor choices and rejects malformed saved coordinates', () => {
    expect(parseKvView(saved)?.tensor).toEqual(saved.tensor);
    expect(parseKvView({...saved, tensor: {...saved.tensor, head: -1}})).toBeNull();
    expect(parseKvView({...saved, tensor: {...saved.tensor, plot: 'unknown'}})).toBeNull();
    expect(restoreKvView(saved, analysis, {captureId: 'capture', turn: 2}).state.tensor).toEqual(
      saved.tensor,
    );
  });
  it('clears an out-of-range tensor position instead of opening a different position', () => {
    const restored = restoreKvView({...saved, tensor: {...saved.tensor, position: 999}}, analysis, {
      captureId: 'capture',
      turn: 2,
    });
    expect(restored.state.tensor?.position).toBeNull();
    expect(restored.state.tensor?.head).toBe(1);
  });
});
