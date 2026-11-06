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

import {inject, Injectable} from '@angular/core';
import {AnalysisCache} from './analysis_cache';
import {ServerClient} from './server_client';
import type {
  TokenAnalysisRequest,
  TokenAnalysisResponse,
  TokenRangeRequest,
  TokenRangeResponse,
} from './contracts/token_analysis';

/** /api/token-analysis and /api/token-range: per-token comparison metrics of rows of the server's
 *  comparison, and one metric of a stage's rows binned by column with its statistics. */
@Injectable({providedIn: 'root'})
export class TokenAnalysisApi {
  private readonly client = inject(ServerClient);
  private readonly cache = inject(AnalysisCache);

  tokenAnalysis(
    captureId: string | null,
    request: TokenAnalysisRequest,
    signal: AbortSignal,
  ): Promise<TokenAnalysisResponse> {
    return this.client.postCapture<TokenAnalysisResponse>(
      captureId,
      'token-analysis',
      request,
      signal,
    );
  }
  /** The rows' metric binned; kept, and with the whole stage asked, the whole stage under the
   *  other alignment asked beside it (AnalysisCache), in its bins at the chart's `capacity` (one
   *  bin, a statistic of the stage, needs none). */
  tokenRange(
    captureId: string | null,
    request: TokenRangeRequest,
    signal: AbortSignal,
    capacity: number | null = null,
  ): Promise<TokenRangeResponse> {
    const answer = this.readRange(captureId, request, signal);
    return this.cache.prefetchRange(
      answer,
      {
        turn: request.turn,
        stage: request.stage,
        alignment: request.alignment,
        analysis: request.analysis,
        start: request.start,
        end: request.end,
        bins: request.bins,
        capacity,
      },
      (twin, bins) =>
        this.readRange(captureId, {
          ...request,
          alignment: twin.alignment,
          start: twin.start,
          end: twin.end,
          bins,
        }),
      signal,
    );
  }
  private readRange(
    captureId: string | null,
    request: TokenRangeRequest,
    signal?: AbortSignal,
  ): Promise<TokenRangeResponse> {
    return this.cache.read(
      JSON.stringify([captureId, 'token-range', request]),
      (own) => this.client.postCapture<TokenRangeResponse>(captureId, 'token-range', request, own),
      signal,
    );
  }
}
