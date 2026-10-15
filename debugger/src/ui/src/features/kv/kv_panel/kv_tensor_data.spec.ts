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
import type {KvAnalysis} from '../../../data/contracts/kv';
import {readTensorHeads} from './kv_tensor_data';
import type {ReportApiService} from '../../../data/report_api_service';
import type {KvHeadEvidence} from '../../../data/contracts/kv';
const selection = {
  turn: 2,
  contextId: 'snapshot',
  kind: 'key',
  layer: 3,
  position: 42,
  head: 0,
  alignment: {start: 42, pairs: [[null, 40]] as [number | null, number | null][]},
};
const result = (head: number) =>
  ({
    selection: {
      turn: 2,
      context_id: 'snapshot',
      kind: 'key',
      layer: 3,
      position: 42,
      head,
      batch: null,
    },
  }) as KvHeadEvidence;
describe('tensor head source loading', () => {
  it('bounds concurrency, keeps ordering and forwards aligned source identity', async () => {
    let active = 0,
      peak = 0;
    const calls: unknown[][] = [];
    const api = {
      kvHead: async (...args: unknown[]) => {
        calls.push(args);
        peak = Math.max(peak, ++active);
        await new Promise((resolve) => setTimeout(resolve, Number(args[6]) % 2));
        active--;
        return result(Number(args[6]));
      },
    } as Pick<ReportApiService, 'kvHead'>;
    const controller = new AbortController();
    const rows = await readTensorHeads(api, 'capture', selection, 9, controller.signal);
    expect(peak).toBeLessThanOrEqual(4);
    expect(rows.map((row) => row.head)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(rows.every((row) => row.evidence && !row.error)).toBe(true);
    expect(calls.every((args) => args[9] === selection.alignment)).toBe(true);
  });
  it('retains available heads when a different head fails or returns the wrong identity', async () => {
    const api = {
      kvHead: async (...args: unknown[]) => {
        const head = Number(args[6]);
        if (head === 1) throw new Error('Read failed');
        return head === 2 ? result(99) : result(head);
      },
    } as Pick<ReportApiService, 'kvHead'>;
    const rows = await readTensorHeads(api, 'capture', selection, 3, new AbortController().signal);
    expect(rows[0].evidence?.selection.head).toBe(0);
    expect(rows[1].error).toBe('Read failed');
    expect(rows[2].evidence).toBeNull();
    expect(rows[2].error).toContain('does not match');
  });
  it('stops scheduling work and discards in-flight values when selection is cancelled', async () => {
    const controller = new AbortController();
    let calls = 0;
    const api = {
      kvHead: async (...args: unknown[]) => {
        calls++;
        controller.abort();
        return result(Number(args[6]));
      },
    } as Pick<ReportApiService, 'kvHead'>;
    const rows = await readTensorHeads(api, 'capture', selection, 20, controller.signal);
    expect(calls).toBe(1);
    expect(rows.every((row) => row.evidence === null)).toBe(true);
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
        layers: [
          {kind: 'key', layer: 3, head_count: 4, position_start: 0, position_count: 64, cells: []},
        ],
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
