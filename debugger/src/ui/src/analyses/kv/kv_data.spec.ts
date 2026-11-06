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

import type {KvAnalysis, KvAnalysisContext} from '../../server/contracts/kv';
import type {KvApi} from '../../server/kv_api';
import {KvDataController} from './kv_data';
import type {KvViewIdentity} from './kv_view_state';

describe('KvDataController', () => {
  const identity: KvViewIdentity = {captureId: 'capture-a', turn: 2};
  const context = (id: string, turn = 2) =>
    ({
      id,
      label: id,
      turn,
      source: 'snapshot',
      phase: 'decode',
      moment: 'terminal',
      step: 9,
      runtime: 'CPU',
      layers: ['key', 'value'].map((kind) => ({
        layer: 3,
        kind,
        status: 'ok',
        head_count: 2,
        position_start: 10,
        position_count: 90,
      })),
    }) as KvAnalysisContext;
  const analysis: KvAnalysis = {
    contexts: [
      context('first'),
      {...context('second'), moment: 'prefill_post', phase: 'prefill', step: null},
    ],
  };

  /** A request the fake server holds until the spec answers it. */
  interface Call {
    method: string;
    args: unknown[];
    promise: Promise<unknown>;
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }
  const deferred = () => {
    let resolve!: (value: unknown) => void;
    let reject!: (error: Error) => void;
    const promise = new Promise((a, b) => {
      resolve = a;
      reject = b;
    });
    return {promise, resolve, reject};
  };
  const flush = async () => {
    for (let i = 0; i < 10; i++) {
      await Promise.resolve();
    }
  };
  const calls: Call[] = [];
  const api = Object.fromEntries(
    ['kvMetadata', 'kvRange'].map((method) => [
      method,
      (...args: unknown[]) => {
        const result = deferred();
        calls.push({method, args, ...result});
        return result.promise;
      },
    ]),
  ) as unknown as Pick<KvApi, 'kvMetadata' | 'kvRange'>;
  /** The abort signal a call was given at `index` of its arguments. */
  const signalOf = (call: Call, index: number) => call.args.at(index) as AbortSignal;
  const data = new KvDataController(api);

  it('metadata captures identity at dispatch, ignores late responses, clears stale values on error and supports retry', async () => {
    const received: KvAnalysis[] = [];
    data.loadMetadata(identity, (value) => received.push(value));
    const old = calls.at(-1)!;
    data.loadMetadata({...identity, captureId: 'capture-b', turn: 3}, (value) =>
      received.push(value),
    );
    const current = calls.at(-1)!;
    expect(signalOf(old, -1).aborted).toBe(true);
    current.resolve({contexts: [context('new', 3)]});
    await flush();
    old.resolve(analysis);
    await flush();
    expect(data.metadata.data()!.contexts[0].id).toBe('new');
    expect(received.length).toBe(1);
    data.loadMetadata(identity, () => {});
    calls.at(-1)!.reject(new Error('unavailable'));
    await flush();
    expect(data.metadata.data()).toBe(null);
    expect(data.metadata.error()).toMatch(/unavailable/);
    data.loadMetadata(identity, () => {});
    expect(data.metadata.error()).toBe('');
    calls.at(-1)!.resolve(analysis);
    await flush();
    expect(data.metadata.loading()).toBe(false);
  });

  it('debounce cancelled, scale bounded, destroy aborts every request', async () => {
    const range = {
      ...identity,
      contextId: 'first',
      range: {start: 10, end: 30},
      bins: 10,
      kind: 'key',
      head: 'max',
    };
    const before = calls.length;
    const cancel = data.loadRange(range);
    cancel();
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(calls.length, 'cancelled resize debounce cannot issue a request').toBe(before);
    data.loadScale(range);
    const scale = calls.at(-1)!;
    expect(scale.args[5]).toBe(1);
    data.destroy();
    expect(signalOf(scale, 8).aborted).toBe(true);
    scale.resolve({context_id: 'first', start: 10, end: 30, bins: [], rows: []});
    await flush();
    expect(data.scale.data()).toBe(null);
  });

  it('a range reply for another observation or position range is refused; a matching one is published', async () => {
    const ranges = new KvDataController(api);
    const range = {
      ...identity,
      contextId: 'first',
      range: {start: 10, end: 30},
      bins: 10,
      kind: 'key',
      head: 'max',
    };
    const reply = {context_id: 'first', start: 10, end: 30, bins: [], rows: []};
    const answer = async (response: object) => {
      const before = calls.length;
      ranges.loadRange(range);
      await new Promise((resolve) => setTimeout(resolve, 120));
      expect(calls.length).toBe(before + 1);
      const call = calls.at(-1)!;
      expect(call.method).toBe('kvRange');
      expect(call.args.slice(2, 5)).toStrictEqual(['first', 10, 30]);
      call.resolve(response);
      await flush();
    };
    for (const mismatch of [{context_id: 'second'}, {start: 11}, {end: 29}]) {
      await answer({...reply, ...mismatch});
      expect(ranges.range.data(), JSON.stringify(mismatch)).toBe(null);
      expect(ranges.range.error()).toBe(
        'KV response did not match the requested observation and position range.',
      );
      expect(ranges.range.loading()).toBe(false);
    }
    await answer(reply);
    expect(ranges.range.data()).toStrictEqual(reply);
    expect(ranges.range.error()).toBe('');
    ranges.destroy();
  });
});
