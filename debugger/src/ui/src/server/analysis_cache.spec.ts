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

import {TestBed} from '@angular/core/testing';
import {AnalysisCache, binsAt} from './analysis_cache';
import {CaptureStore} from './capture_store';
import type {Session} from './contracts/capture';
import {ServerClient} from './server_client';
import {TokenAnalysisApi} from './token_analysis_api';

/** A Session of one turn whose Decode has 4 columns by content and 3 by steps. */
function session(): Session {
  const columns = (end: number) => ({
    final_logits: {start: 16, end, omitted: [], notes: [], context: null},
    sampling: {start: 16, end, omitted: [], notes: [], context: null},
    kv: {start: 16, end: end - 1, omitted: [], notes: [], context: null},
  });
  return {
    model: 'm',
    runs: [],
    batches: [],
    turns: [
      {
        n: 1,
        comparison: {
          alignments: {
            content: {positions: {start: 16, pairs: [], columns: columns(20)}},
            steps: {positions: {start: 16, pairs: [], columns: columns(19)}},
          },
        },
      },
    ],
  } as unknown as Session;
}

describe('AnalysisCache', () => {
  let posted: unknown[];
  let api: TokenAnalysisApi;
  let state: CaptureStore;
  beforeEach(() => {
    posted = [];
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ServerClient,
          useValue: {
            postCapture: (_capture: string, _path: string, request: unknown) => {
              posted.push(request);
              return Promise.resolve({request});
            },
          },
        },
      ],
    });
    state = TestBed.inject(CaptureStore);
    state.session.set(session());
    api = TestBed.inject(TokenAnalysisApi);
  });
  const settle = async () => {
    for (let i = 0; i < 6; i++) {
      await Promise.resolve();
    }
  };
  const request = (alignment: 'content' | 'steps', end: number, bins: number) =>
    ({
      turn: 1,
      stage: 'decode',
      alignment,
      analysis: 'sampling',
      metric: 'entropy',
      start: 16,
      end,
      bins,
    }) as const;

  it('asks the whole stage under the other alignment beside the first, answers once both are, then serves both from what it kept', async () => {
    const signal = new AbortController().signal;
    await api.tokenRange('c', request('content', 20, 4), signal, 4);
    await settle();
    expect(posted).toEqual([request('content', 20, 4), request('steps', 19, 3)]);
    // Switching to Steps, and back, asks nothing.
    await api.tokenRange('c', request('steps', 19, 3), signal, 4);
    await api.tokenRange('c', request('content', 20, 4), signal, 4);
    expect(posted.length).toBe(2);
  });

  it('answers a view once its other alignment is in too, and not after a failure of that one', async () => {
    const cache = TestBed.inject(AnalysisCache);
    let release!: () => void;
    const twin = new Promise<void>((resolve) => (release = resolve));
    let answered = false;
    const view = cache
      .prefetch(Promise.resolve('first'), () => twin)
      .then((value) => {
        answered = true;
        return value;
      });
    await settle();
    expect(answered).toBe(false);
    release();
    expect(await view).toBe('first');
    // A failure of the other alignment is the panels' to see when they ask it.
    expect(await cache.prefetch(Promise.resolve('x'), () => Promise.reject(new Error('no')))).toBe(
      'x',
    );
  });

  it('asks no other alignment for part of the stage, nor without the bins it would be drawn in', async () => {
    const signal = new AbortController().signal;
    await api.tokenRange('c', request('content', 18, 2), signal, 4);
    await api.tokenRange('c', request('content', 20, 4), signal);
    await settle();
    expect(posted).toEqual([request('content', 18, 2), request('content', 20, 4)]);
    // A statistic of the whole stage, one bin, needs no capacity.
    await api.tokenRange('c', request('content', 20, 1), signal);
    await settle();
    expect(posted.slice(2)).toEqual([request('content', 20, 1), request('steps', 19, 1)]);
  });

  it('forgets what it kept once the Session view changes, and after a failure', async () => {
    const signal = new AbortController().signal;
    await api.tokenRange('c', request('content', 18, 2), signal);
    state.session.set(session());
    await api.tokenRange('c', request('content', 18, 2), signal);
    expect(posted.length).toBe(2);
    const cache = TestBed.inject(AnalysisCache);
    let calls = 0;
    const failing = () => {
      calls++;
      return Promise.reject(new Error('no'));
    };
    await expect(cache.read('k', failing)).rejects.toThrow('no');
    await settle();
    await expect(cache.read('k', failing)).rejects.toThrow('no');
    expect(calls).toBe(2);
  });

  it('a caller that gives up stops waiting; the answer stays kept for the next', async () => {
    const cache = TestBed.inject(AnalysisCache);
    let resolve!: (value: string) => void;
    let calls = 0;
    const fetch = () => {
      calls++;
      return new Promise<string>((r) => (resolve = r));
    };
    const controller = new AbortController();
    const first = expect(cache.read('k', fetch, controller.signal)).rejects.toHaveProperty(
      'name',
      'AbortError',
    );
    controller.abort();
    await first;
    resolve('answer');
    expect(await cache.read('k', fetch)).toBe('answer');
    expect(calls).toBe(1);
  });

  it('keeps two large answers, the last asked for: one stage under both alignments', async () => {
    const cache = TestBed.inject(AnalysisCache);
    let calls = 0;
    const fetch = () => Promise.resolve(++calls);
    for (const key of ['a', 'b', 'c']) {
      await cache.read(key, fetch, undefined, true);
    }
    expect(await cache.read('b', fetch, undefined, true)).toBe(2);
    expect(await cache.read('c', fetch, undefined, true)).toBe(3);
    expect(await cache.read('a', fetch, undefined, true)).toBe(4);
    // Asking 'a' again dropped 'b', the oldest asked for.
    expect(await cache.read('b', fetch, undefined, true)).toBe(5);
  });

  it('counts bins as the panels do (range_geometry.rangeBinCount)', () => {
    // The same chart's capacity bins the two alignments' stages differently.
    expect(binsAt(2119, 140)).toBe(132);
    expect(binsAt(2039, 140)).toBe(135);
    expect(binsAt(3, 160)).toBe(3);
    expect(binsAt(0, 0)).toBe(1);
  });
});
