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
import {ServerClient} from './server_client';
import type {
  FinalLogitsAddress,
  FinalLogitsBinsRequest,
  FinalLogitsBinsResponse,
  FinalLogitsPointsRequest,
  FinalLogitsPointsResponse,
  FinalLogitsRequest,
  FinalLogitsResponse,
  FinalLogitsSummaryResponse,
  FinalLogitsZoomedPointsResponse,
} from './contracts/final_logits';

/** /api/final-logits*: the final logits of one Decode row: pages of rows, what is known of the
 *  whole vectors, and what the Inspection's charts draw of them. */
@Injectable({providedIn: 'root'})
export class FinalLogitsApi {
  private readonly client = inject(ServerClient);
  finalLogits(
    captureId: string | null,
    payload: FinalLogitsRequest,
    signal: AbortSignal,
  ): Promise<FinalLogitsResponse> {
    return this.client.postCapture<FinalLogitsResponse>(captureId, 'final-logits', payload, signal);
  }
  finalLogitsSummary(
    captureId: string | null,
    payload: FinalLogitsAddress,
    signal: AbortSignal,
  ): Promise<FinalLogitsSummaryResponse> {
    return this.client.postCapture<FinalLogitsSummaryResponse>(
      captureId,
      'final-logits-summary',
      payload,
      signal,
    );
  }
  finalLogitsBins(
    captureId: string | null,
    payload: FinalLogitsBinsRequest,
    signal: AbortSignal,
  ): Promise<FinalLogitsBinsResponse> {
    return this.client.postCapture<FinalLogitsBinsResponse>(
      captureId,
      'final-logits-bins',
      payload,
      signal,
    );
  }
  /** The points of the paired values' whole extent, with the extent and the paired runs. */
  finalLogitsPoints(
    captureId: string | null,
    payload: FinalLogitsAddress,
    signal: AbortSignal,
  ): Promise<FinalLogitsPointsResponse> {
    return this.client.postCapture<FinalLogitsPointsResponse>(
      captureId,
      'final-logits-points',
      payload,
      signal,
    );
  }
  /** The points of a value range. */
  finalLogitsZoomedPoints(
    captureId: string | null,
    payload: FinalLogitsPointsRequest,
    signal: AbortSignal,
  ): Promise<FinalLogitsZoomedPointsResponse> {
    return this.client.postCapture<FinalLogitsZoomedPointsResponse>(
      captureId,
      'final-logits-points',
      payload,
      signal,
    );
  }
}
