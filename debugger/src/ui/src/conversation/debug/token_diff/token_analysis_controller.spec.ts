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

import type {CapturedToken} from '../../../server/contracts/capture';
import type {
  TokenAnalysisPair,
  TokenAnalysisRequest,
  TokenAnalysisResponse,
} from '../../../server/contracts/token_analysis';
import type {TokenPair} from '../../token_alignment';
import {
  ANALYSIS_CONCURRENCY,
  AnalysisRequest,
  TokenAnalysisController,
} from './token_analysis_controller';

describe('TokenAnalysisController', () => {
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve();
    }
  };

  /** One request the fake server has taken, and how the spec answers it. */
  interface Call {
    captureId: string | null;
    request: TokenAnalysisRequest;
    signal: AbortSignal;
    done: boolean;
    resolve: () => void;
    reject: (error: Error) => void;
  }

  /** The server: row k pairs steps (k, k); the first divergence pairs steps (1000, 1001). */
  function fakeApi() {
    const calls: Call[] = [];
    return {
      calls,
      inFlight: () => calls.filter((c) => !c.done).length,
      peak: 0,
      tokenAnalysis(captureId: string | null, request: TokenAnalysisRequest, signal: AbortSignal) {
        const call = {captureId, request, signal, done: false} as Call;
        calls.push(call);
        this.peak = Math.max(this.peak, this.inFlight());
        return new Promise<TokenAnalysisResponse>((resolve, reject) => {
          call.resolve = () => {
            call.done = true;
            const pair = (row: number, ref: number, target: number, first_divergence: boolean) =>
              ({
                stage: 'decode',
                alignment: request.alignment,
                row,
                first_divergence,
                ref_step: ref,
                target_step: target,
                metrics: {},
              }) as TokenAnalysisPair;
            resolve({
              turn: request.turn,
              stage: 'decode',
              alignment: request.alignment,
              pairs: [
                ...request.rows.map((k) => pair(k, k, k, false)),
                ...(request.first_divergence ? [pair(3, 1000, 1001, true)] : []),
              ],
            } as TokenAnalysisResponse);
          };
          call.reject = (error) => {
            call.done = true;
            reject(error);
          };
        });
      },
    };
  }

  /** Rows of the server's comparison, with the tokens they name. */
  const pairs = (n: number, from = 0): TokenPair[] =>
    Array.from({length: n}, (_, i) => ({
      index: from + i,
      ref: {step: from + i} as CapturedToken,
      target: {step: from + i} as CapturedToken,
      match: true,
    }));
  const fork: TokenPair = {
    index: 3,
    ref: {step: 1000} as CapturedToken,
    target: {step: 1001} as CapturedToken,
    match: false,
  };

  const request = (
    turn: number,
    rows: TokenPair[],
    firstDivergence: TokenPair | null = null,
  ): AnalysisRequest => ({
    turn,
    alignment: 'content',
    rows,
    firstDivergence,
  });

  it('at most four batches are in flight and every pair lands in the results', async () => {
    const api = fakeApi();
    const controller = new TokenAnalysisController(api);
    controller.load('cap', [request(1, pairs(1280))], new AbortController().signal);
    await flush();
    expect(api.inFlight()).toBe(ANALYSIS_CONCURRENCY);
    expect(controller.loading()).toBe(true);
    while (api.calls.some((c) => !c.done)) {
      api.calls.find((c) => !c.done)!.resolve();
      await flush();
    }
    expect(api.calls.length).toBe(10);
    expect(api.peak).toBe(ANALYSIS_CONCURRENCY);
    expect(controller.results().size).toBe(1280);
    expect(controller.loading()).toBe(false);
    expect(controller.error()).toBe('');
  });

  it('a second alignment only fetches pairs never seen; a new capture starts over', async () => {
    const api = fakeApi();
    const controller = new TokenAnalysisController(api);
    controller.load('cap', [request(1, pairs(100))], new AbortController().signal);
    await flush();
    api.calls[0].resolve();
    await flush();
    controller.load(
      'cap',
      [{...request(1, [...pairs(100), ...pairs(10, 100)]), alignment: 'steps'}],
      new AbortController().signal,
    );
    await flush();
    expect(api.calls.length).toBe(2);
    expect(api.calls[1].request, 'only the ten unseen rows are requested, by index').toStrictEqual({
      turn: 1,
      stage: 'decode',
      alignment: 'steps',
      rows: pairs(10, 100).map((p) => p.index),
    });
    expect(controller.results().size, 'cached metrics show before the fetch lands').toBe(100);
    api.calls[1].resolve();
    await flush();
    expect(controller.results().size).toBe(110);
    controller.load('other', [request(1, pairs(5))], new AbortController().signal);
    await flush();
    expect(controller.results().size).toBe(0);
    expect(api.calls[2].captureId).toBe('other');
  });

  it('aborted loads change nothing; failures report once and never stick to a later load', async () => {
    const api = fakeApi();
    const controller = new TokenAnalysisController(api);
    const aborter = new AbortController();
    controller.load('cap', [request(1, pairs(3))], aborter.signal);
    await flush();
    aborter.abort();
    api.calls[0].resolve();
    await flush();
    expect(controller.results().size).toBe(0);
    expect(controller.loading(), 'the superseding load owns the loading flag').toBe(true);
    controller.load('cap', [request(2, pairs(3))], new AbortController().signal);
    await flush();
    api.calls[1].reject(new Error('offline'));
    await flush();
    expect(controller.error()).toBe('Token metrics could not be loaded.');
    expect(controller.loading()).toBe(false);
    controller.load('cap', [request(2, pairs(3))], new AbortController().signal);
    expect(controller.error()).toBe('');
  });

  it('no capture or nothing missing resolves immediately without a request', () => {
    const api = fakeApi();
    const controller = new TokenAnalysisController(api);
    controller.load(null, [request(1, pairs(3))], new AbortController().signal);
    controller.load('cap', [], new AbortController().signal);
    expect(api.calls.length).toBe(0);
    expect(controller.loading()).toBe(false);
  });

  it('the first divergence rides with the last batch, or alone, and is asked for once', async () => {
    const api = fakeApi();
    const controller = new TokenAnalysisController(api);
    controller.load('cap', [request(1, pairs(130), fork)], new AbortController().signal);
    await flush();
    expect(
      api.calls.map((c) => [c.request.rows.length, !!c.request.first_divergence]),
    ).toStrictEqual([
      [128, false],
      [2, true],
    ]);
    for (const call of api.calls) {
      call.resolve();
    }
    await flush();
    expect(controller.results().size).toBe(131);
    expect(controller.results().get('1:1000:1001')?.first_divergence).toBe(true);
    controller.load('cap', [request(1, pairs(130), fork)], new AbortController().signal);
    await flush();
    expect(api.calls.length, 'nothing is asked for again').toBe(2);
    controller.load('cap', [request(2, pairs(128), fork)], new AbortController().signal);
    await flush();
    expect(
      api.calls.slice(2).map((c) => [c.request.rows.length, !!c.request.first_divergence]),
    ).toStrictEqual([
      [128, false],
      [0, true],
    ]);
  });
});
