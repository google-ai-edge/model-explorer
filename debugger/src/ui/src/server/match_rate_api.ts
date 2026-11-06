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
import type {MatchRateRequest, MatchRateResponse} from './contracts/match_rate';

/** /api/match-rate: how many of a stage's rows are the same, as the server counts them. */
@Injectable({providedIn: 'root'})
export class MatchRateApi {
  private readonly client = inject(ServerClient);
  matchRate(
    captureId: string | null,
    request: MatchRateRequest,
    signal: AbortSignal,
  ): Promise<MatchRateResponse> {
    return this.client.postCapture<MatchRateResponse>(captureId, 'match-rate', request, signal);
  }
}
