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

import type {TokenAnalysisPair} from '../../../server/contracts/token_analysis';
import type {TokenAnalysisApi} from '../../../server/token_analysis_api';
import type {AlignmentMode, TokenPair} from '../../token_alignment';
import {analysisKey, pairKey} from '../../token_analysis';

/** The Decode rows of one turn to analyze, of one alignment, and the first divergence's pair
 *  when the server fills it in (its row holds one side alone). */
export interface AnalysisRequest {
  turn: number;
  alignment: AlignmentMode;
  rows: TokenPair[];
  firstDivergence: TokenPair | null;
}
export const ANALYSIS_BATCH = 128;
export const ANALYSIS_CONCURRENCY = 4;

interface Batch {
  turn: number;
  alignment: AlignmentMode;
  rows: number[];
  firstDivergence: boolean;
}

/**
 * Loads per-token metrics for rows of the server's comparison of one capture.
 *
 * A request names rows by index; the server answers each with the steps it compared. Results are
 * cached by (turn, ref step, target step), which does not depend on the alignment mode, so
 * switching Context/Steps only fetches pairs never seen; at most four batches of 128 rows are in
 * flight at once.
 */
export class TokenAnalysisController {
  readonly results = signal(new Map<string, TokenAnalysisPair>());
  readonly loading = signal(false);
  readonly error = signal('');
  private captureId: string | null = null;
  private readonly cache = new Map<string, TokenAnalysisPair>();
  constructor(private readonly api: Pick<TokenAnalysisApi, 'tokenAnalysis'>) {}

  load(captureId: string | null, requests: AnalysisRequest[], signal: AbortSignal): void {
    if (captureId !== this.captureId) {
      this.captureId = captureId;
      this.cache.clear();
    }
    this.error.set('');
    this.results.set(new Map(this.cache));
    const batches: Batch[] = [];
    const asked = new Set<string>();
    const missing = (turn: number, pair: TokenPair): boolean => {
      const key = pairKey(turn, pair);
      if (this.cache.has(key) || asked.has(key)) {
        return false;
      }
      asked.add(key);
      return true;
    };
    for (const request of requests) {
      const {turn, alignment} = request;
      const rows = request.rows.filter((pair) => missing(turn, pair)).map((pair) => pair.index);
      for (let offset = 0; offset < rows.length; offset += ANALYSIS_BATCH) {
        batches.push({
          turn,
          alignment,
          rows: rows.slice(offset, offset + ANALYSIS_BATCH),
          firstDivergence: false,
        });
      }
      if (!request.firstDivergence || !missing(turn, request.firstDivergence)) {
        continue;
      }
      // The first divergence rides with this request's last batch while it has room.
      const last = rows.length ? batches[batches.length - 1] : null;
      if (last && last.rows.length < ANALYSIS_BATCH) {
        last.firstDivergence = true;
      } else {
        batches.push({turn, alignment, rows: [], firstDivergence: true});
      }
    }
    if (!captureId || !batches.length) {
      this.loading.set(false);
      return;
    }
    this.loading.set(true);
    const worker = async (): Promise<void> => {
      while (batches.length && !signal.aborted) {
        const batch = batches.shift();
        if (!batch) {
          break;
        }
        const data = await this.api.tokenAnalysis(
          captureId,
          {
            turn: batch.turn,
            stage: 'decode',
            alignment: batch.alignment,
            rows: batch.rows,
            ...(batch.firstDivergence ? {first_divergence: true} : {}),
          },
          signal,
        );
        if (signal.aborted) {
          return;
        }
        for (const pair of data.pairs) {
          this.cache.set(analysisKey(data.turn, pair.ref_step, pair.target_step), pair);
        }
        this.results.set(new Map(this.cache));
      }
    };
    void Promise.all(Array.from({length: Math.min(ANALYSIS_CONCURRENCY, batches.length)}, worker))
      .catch(() => {
        if (!signal.aborted) {
          this.error.set('Token metrics could not be loaded.');
        }
      })
      .finally(() => {
        if (!signal.aborted) {
          this.loading.set(false);
        }
      });
  }
}
