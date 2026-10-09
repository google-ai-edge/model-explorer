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
import type {
  TokenAnalysisPair,
  TokenAnalysisResponse,
} from '../../../data/contracts/token_analysis';
import {analysisKey} from '../token_analysis';

export interface TokenAnalysisApi {
  tokenAnalysis(
    captureId: string,
    turn: number,
    pairs: {ref: number | null; target: number | null}[],
    signal: AbortSignal,
  ): Promise<TokenAnalysisResponse>;
}
export interface AnalysisRequest {
  turn: number;
  pairs: {ref: number | null; target: number | null}[];
}
export const ANALYSIS_BATCH = 128;
export const ANALYSIS_CONCURRENCY = 4;

/**
 * Loads per-token metrics for the aligned pairs of one capture.
 *
 * Results are cached by (turn, ref step, target step), which does not depend on
 * the alignment mode, so switching Context/Steps only fetches pairs never seen;
 * at most four batches of 128 pairs are in flight at once.
 */
export class TokenAnalysisController {
  readonly results = signal(new Map<string, TokenAnalysisPair>());
  readonly loading = signal(false);
  readonly error = signal('');
  private captureId: string | null = null;
  private readonly cache = new Map<string, TokenAnalysisPair>();
  constructor(private readonly api: TokenAnalysisApi) {}

  load(
    captureId: string | null,
    requests: AnalysisRequest[],
    signal: AbortSignal,
  ) {
    if (captureId !== this.captureId) {
      this.captureId = captureId;
      this.cache.clear();
    }
    this.error.set('');
    this.results.set(new Map(this.cache));
    const batches: AnalysisRequest[] = [];
    for (const request of requests) {
      const missing = request.pairs.filter(
        (pair) =>
          !this.cache.has(analysisKey(request.turn, pair.ref, pair.target)),
      );
      for (let offset = 0; offset < missing.length; offset += ANALYSIS_BATCH)
        batches.push({
          turn: request.turn,
          pairs: missing.slice(offset, offset + ANALYSIS_BATCH),
        });
    }
    if (!captureId || !batches.length) {
      this.loading.set(false);
      return;
    }
    this.loading.set(true);
    const worker = async () => {
      while (batches.length && !signal.aborted) {
        const batch = batches.shift()!;
        const data = await this.api.tokenAnalysis(
          captureId,
          batch.turn,
          batch.pairs,
          signal,
        );
        if (signal.aborted) return;
        for (const pair of data.pairs)
          this.cache.set(
            analysisKey(data.turn, pair.ref_step, pair.target_step),
            pair,
          );
        this.results.set(new Map(this.cache));
      }
    };
    void Promise.all(
      Array.from(
        {length: Math.min(ANALYSIS_CONCURRENCY, batches.length)},
        worker,
      ),
    )
      .catch(() => {
        if (!signal.aborted)
          this.error.set('Token metrics could not be loaded.');
      })
      .finally(() => {
        if (!signal.aborted) this.loading.set(false);
      });
  }
}
